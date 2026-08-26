"""
NewsBreak Ads Launcher — Flask application.
"""
from __future__ import annotations

import logging
import os
import secrets
import sys
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stderr,
    force=True,
)

from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)

load_dotenv()

import storage
from bulk_launcher import bulk_launch, creative_from_upload
from newsbreak_api import NewsBreakAPIError, NewsBreakClient, unwrap_list_response
from platforms import (
    DEFAULT_PLATFORM,
    PLATFORM_CURRENCIES,
    PLATFORM_LABELS,
    PLATFORMS,
    get_adapter,
    normalize_platform,
)
from rules_engine import (
    RULE_TEMPLATES,
    build_report_payload,
    instantiate_template,
    normalize_report_rows,
    run_rules_for_account,
)
from scheduler import run_scheduled_rules, start_scheduler

try:
    from agent_api import bp as _agent_bp
except ImportError:  # pragma: no cover
    _agent_bp = None

try:
    import config as _cfg
except ImportError:
    _cfg = None


def _cfg_val(name: str, default: str = "") -> str:
    if _cfg is not None and hasattr(_cfg, name):
        return str(getattr(_cfg, name) or default)
    return os.environ.get(name, default)


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)
# Hard ceiling for the entire multipart /launch POST (all creatives + form
# fields combined). Keep in sync with templates/launch.html
# (NB_MAX_UPLOAD_BYTES / NB_MAX_BATCH_BYTES). Several videos at once can
# hit this even when each file is under the limit.
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

storage.ensure_dirs()


def _upload_too_large_response():
    """Shared 413 response: redirect launch POSTs back to the form with a flag."""
    cl = request.content_length
    app.logger.warning(
        "upload.too_large path=%s content_length=%s max=%s",
        request.path,
        cl,
        MAX_UPLOAD_BYTES,
    )
    mb = (cl / (1024 * 1024)) if cl else None
    if request.path.rstrip("/").endswith("launch"):
        return redirect(
            url_for(
                "launch",
                upload_error="too_large",
                upload_size_mb=f"{mb:.0f}" if mb else "",
            )
        )
    limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
    size_bit = f" ({mb:.0f} MB)" if mb else ""
    return (
        f"Upload too large{size_bit}. Max is {limit_mb} MB for all creatives "
        f"combined — remove some files or compress "
        f"(H.264 MP4 under ~100 MB each is typical) and try again.",
        413,
    )


@app.errorhandler(413)
def handle_request_entity_too_large(e):
    """Surface a readable message when a video/image POST exceeds MAX_CONTENT_LENGTH.

    Without this, browsers show a blank/generic 413 page and the launch form
    never gets a chance to explain that the creative must be compressed.
    """
    return _upload_too_large_response()


@app.before_request
def _reject_oversized_uploads_early():
    """Fail fast on Content-Length before the launch view runs.

    Flask only raises RequestEntityTooLarge when the body stream is read.
    /launch may redirect to /login (no token) before that, so oversized
    videos would never hit the 413 handler. Check the header up front.
    """
    if request.method not in ("POST", "PUT", "PATCH"):
        return None
    cl = request.content_length
    if cl is not None and cl > MAX_UPLOAD_BYTES:
        return _upload_too_large_response()
    return None

if _agent_bp is not None:
    app.register_blueprint(_agent_bp)


def _user_id() -> str:
    if "uid" not in session:
        session["uid"] = str(uuid.uuid4())
    return session["uid"]


def _active_platform() -> str:
    """Currently selected platform for this session (``newsbreak`` by default)."""
    return normalize_platform(session.get("platform") or DEFAULT_PLATFORM)


def _org_ids_from_form() -> list[str]:
    raw = (request.form.get("org_ids") or "").strip()
    if raw:
        return [x.strip() for x in raw.split(",") if x.strip()]
    default = _cfg_val("NEWSBREAK_DEFAULT_ORG_IDS", "")
    if default:
        return [x.strip() for x in default.split(",") if x.strip()]
    return []


def _env_access_token() -> str:
    return (_cfg_val("NEWSBREAK_ACCESS_TOKEN", "") or "").strip()


def _env_org_ids() -> list[str]:
    raw = _cfg_val("NEWSBREAK_DEFAULT_ORG_IDS", "")
    return [x.strip() for x in raw.split(",") if x.strip()]


def _env_smartnews_client_id() -> str:
    return (_cfg_val("SMARTNEWS_CLIENT_ID", "") or "").strip()


def _env_smartnews_client_secret() -> str:
    # Accept the legacy SMARTNEWS_API_KEY name as a fallback — the old v1 key
    # doubled as the new client_secret for many developer apps.
    return (
        _cfg_val("SMARTNEWS_CLIENT_SECRET", "")
        or _cfg_val("SMARTNEWS_API_KEY", "")
        or ""
    ).strip()


def _env_smartnews_account_ids() -> list[str]:
    raw = _cfg_val("SMARTNEWS_DEFAULT_ACCOUNT_IDS", "")
    return [x.strip() for x in raw.split(",") if x.strip()]


def _env_smartnews_configured() -> bool:
    return bool(_env_smartnews_client_id() and _env_smartnews_client_secret())


def _env_outbrain_token() -> str:
    return (_cfg_val("OUTBRAIN_ACCESS_TOKEN", "") or "").strip()


def _env_outbrain_username() -> str:
    return (_cfg_val("OUTBRAIN_USERNAME", "") or "").strip()


def _env_outbrain_password() -> str:
    return (_cfg_val("OUTBRAIN_PASSWORD", "") or "").strip()


def _env_outbrain_marketer_ids() -> list[str]:
    raw = _cfg_val("OUTBRAIN_DEFAULT_MARKETER_IDS", "")
    return [x.strip() for x in raw.split(",") if x.strip()]


def _env_outbrain_configured() -> bool:
    return bool(
        _env_outbrain_token()
        or (_env_outbrain_username() and _env_outbrain_password())
    )


def _env_mediago_api_token() -> str:
    return (_cfg_val("MEDIAGO_API_TOKEN", "") or "").strip()


def _env_mediago_auth_level() -> str:
    return (_cfg_val("MEDIAGO_AUTH_LEVEL", "auto") or "auto").strip().lower() or "auto"


def _env_mediago_account_ids() -> list[str]:
    raw = _cfg_val("MEDIAGO_DEFAULT_ACCOUNT_IDS", "")
    return [x.strip() for x in raw.split(",") if x.strip()]


def _env_mediago_configured() -> bool:
    return bool(_env_mediago_api_token())


def _effective_token(platform: str | None = None) -> dict | None:
    """Prefer env-configured credentials so Render redeploys don't force re-login.

    Returns a platform-shaped dict or None:
      NewsBreak: {"access_token", "org_ids", "platform"}
      SmartNews: {"client_id", "client_secret", "org_ids", "platform"}
    """
    p = normalize_platform(platform or _active_platform())
    if p == "newsbreak":
        env_tok = _env_access_token()
        if env_tok:
            return {"access_token": env_tok, "org_ids": _env_org_ids(), "platform": p}
        return storage.load_token(_user_id(), platform=p)
    if p == "smartnews":
        if _env_smartnews_configured():
            return {
                "client_id": _env_smartnews_client_id(),
                "client_secret": _env_smartnews_client_secret(),
                "org_ids": _env_smartnews_account_ids(),
                "platform": p,
            }
        saved = storage.load_token(_user_id(), platform=p)
        if saved and saved.get("client_id") and saved.get("client_secret"):
            return saved
        return None
    if p == "outbrain":
        if _env_outbrain_configured():
            return {
                "access_token": _env_outbrain_token(),
                "username": _env_outbrain_username(),
                "password": _env_outbrain_password(),
                "org_ids": _env_outbrain_marketer_ids(),
                "platform": p,
            }
        saved = storage.load_token(_user_id(), platform=p)
        if saved and (
            saved.get("access_token")
            or (saved.get("username") and saved.get("password"))
        ):
            return saved
        return None
    if p == "mediago":
        if _env_mediago_configured():
            return {
                "api_token": _env_mediago_api_token(),
                "auth_level": _env_mediago_auth_level(),
                "org_ids": _env_mediago_account_ids(),
                "platform": p,
            }
        saved = storage.load_token(_user_id(), platform=p)
        if saved and (saved.get("api_token") or saved.get("access_token")):
            return saved
        return None
    return None


def _client() -> NewsBreakClient | None:
    """Legacy accessor — returns a raw NewsBreakClient when NB platform is active.

    Prefer ``_adapter()`` for platform-agnostic logic.
    """
    if _active_platform() != "newsbreak":
        return None
    tok = _effective_token("newsbreak")
    if not tok:
        return None
    return NewsBreakClient(tok["access_token"])


def _adapter(platform: str | None = None):
    """Return the configured adapter for the named (or active) platform."""
    p = normalize_platform(platform or _active_platform())
    tok = _effective_token(p)
    if not tok:
        return None
    if p == "newsbreak":
        credentials: dict = {
            "access_token": tok["access_token"],
            "org_ids": tok.get("org_ids") or [],
        }
    elif p == "smartnews":
        credentials = {
            "client_id": tok.get("client_id"),
            "client_secret": tok.get("client_secret"),
            "account_ids": tok.get("org_ids") or tok.get("account_ids") or [],
        }
    elif p == "outbrain":
        credentials = {
            "access_token": tok.get("access_token"),
            "username": tok.get("username"),
            "password": tok.get("password"),
            "marketer_ids": tok.get("org_ids") or tok.get("marketer_ids") or [],
        }
    elif p == "mediago":
        credentials = {
            "api_token": tok.get("api_token") or tok.get("access_token"),
            "auth_level": tok.get("auth_level") or _env_mediago_auth_level(),
            "account_ids": tok.get("org_ids") or tok.get("account_ids") or [],
        }
    else:
        return None
    try:
        return get_adapter(p, **credentials)
    except Exception as e:
        app.logger.warning("adapter init failed platform=%s err=%s", p, e)
        return None


def _ad_accounts_for_current_token() -> dict:
    """Return {ad_account_id: name} for the current session's token. Empty dict on failure."""
    cl = _client()
    tok = _effective_token()
    if not cl or not tok:
        return {}
    try:
        raw = cl.get_ad_accounts(tok.get("org_ids") or [])
    except Exception:
        return {}
    return {
        str(a.get("id")): a.get("name", a.get("id"))
        for a in _flatten_ad_accounts(raw)
        if a.get("id")
    }


def _flatten_ad_accounts(api_response) -> list[dict]:
    rows: list[dict] = []
    if not isinstance(api_response, dict):
        return rows
    data = api_response.get("data") or api_response.get("result") or api_response
    groups = data if isinstance(data, list) else data.get("groups") or data.get("list") or []
    if isinstance(groups, dict):
        groups = [groups]
    if not isinstance(groups, list):
        return rows
    for g in groups:
        if not isinstance(g, dict):
            continue
        org_id = g.get("id") or g.get("organizationId")
        accounts = g.get("adAccounts") or g.get("ad_accounts") or []
        if isinstance(accounts, dict):
            accounts = [accounts]
        for a in accounts or []:
            if isinstance(a, dict):
                rows.append({**a, "_org_id": org_id})
    return rows


def _basic_auth_configured() -> bool:
    u = os.environ.get("BASIC_AUTH_USER", "").strip()
    p = os.environ.get("BASIC_AUTH_PASSWORD", "").strip()
    return bool(u and p)


def _basic_auth_ok() -> bool:
    if not _basic_auth_configured():
        return True
    auth = request.authorization
    if not auth:
        return False
    u = os.environ.get("BASIC_AUTH_USER", "")
    p = os.environ.get("BASIC_AUTH_PASSWORD", "")
    return secrets.compare_digest(auth.username or "", u) and secrets.compare_digest(
        auth.password or "", p
    )


@app.before_request
def _require_basic_auth():
    if request.path.startswith("/static"):
        return None
    # Managed-agent surface uses its own HMAC-SHA256 signed-request auth
    # (see agent_api.py). Wrapping it in Basic Auth too would force every
    # caller to share a browser password, defeating the point of HMAC.
    if request.path.startswith("/api/agent/"):
        return None
    # Public creative images must be fetchable by Outbrain's image crawler
    # (it pulls the imageUrl when a promoted link is created), so this route
    # is intentionally exempt from Basic Auth. Files are opaque uuids.
    if request.path.startswith("/public/creative/"):
        return None
    if _basic_auth_ok():
        return None
    return Response(
        "Authentication required",
        401,
        {"WWW-Authenticate": 'Basic realm="NewsBreak Launcher"', "Content-Type": "text/plain"},
    )


@app.before_request
def _honor_platform_query():
    """Keep ``?platform=mediago`` (etc.) deep-links working on HTML pages."""
    if request.path.startswith("/api/") or request.path.startswith("/static"):
        return None
    raw = (request.args.get("platform") or "").strip().lower()
    if raw in PLATFORMS:
        session["platform"] = raw
    return None


@app.before_request
def _start_jobs():
    if not app.config.get("_scheduler_started"):
        start_scheduler(15)
        app.config["_scheduler_started"] = True


@app.context_processor
def _inject_platform():
    p = _active_platform()
    return {
        "active_platform": p,
        "active_platform_label": PLATFORM_LABELS.get(p, p.title()),
        "active_platform_currency": PLATFORM_CURRENCIES.get(p, "USD"),
        "supports_ad_set_scope": p not in ("outbrain", "mediago"),
        "available_platforms": [
            {"id": pid, "label": PLATFORM_LABELS.get(pid, pid.title())}
            for pid in PLATFORMS
        ],
    }


def _studio_link_launch_if_any(platform: str, gen_id, result) -> None:
    """Record launched ad IDs on an AI Studio generation batch, if linked.

    Called by both the NewsBreak and SmartNews launch flows. Gracefully
    no-ops when the studio module isn't importable, when no ``gen_id`` was
    submitted, or when the result payload doesn't expose ad IDs.
    """
    if not gen_id:
        return
    try:
        from ai_studio.feedback import link_launch
    except Exception:  # noqa: BLE001
        return
    ad_ids: list = []
    if not isinstance(result, dict):
        return
    for ad_set in result.get("ad_sets") or []:
        for ad in (ad_set or {}).get("ads") or []:
            aid = ad.get("id") or ad.get("ad_id")
            if aid:
                ad_ids.append(aid)
    for ad in result.get("ads") or []:
        if isinstance(ad, dict):
            aid = ad.get("id") or ad.get("ad_id")
            if aid:
                ad_ids.append(aid)
        elif ad:
            ad_ids.append(ad)
    if not ad_ids:
        return
    try:
        link_launch(str(gen_id), ad_ids, platform=platform)
    except Exception as exc:  # noqa: BLE001
        app.logger.warning("studio link_launch failed: %s", exc)


def _mark_library_used_if_any(platform: str, form, result) -> None:
    """Auto mark-off prebuilt library items consumed by a launch.

    The launch form's "Use from library" picker submits the chosen ids in a
    hidden ``library_ids`` field (comma-separated). When the launch actually
    succeeded we stamp those rows ``consumed_at`` + ``used_in`` so the
    backlog view stays honest. Failed launches leave the items unused so
    the operator can retry without losing them.
    """
    raw = (form.get("library_ids") or "").strip()
    if not raw:
        return
    ids = [x.strip() for x in raw.split(",") if x.strip()]
    if not ids:
        return
    if not isinstance(result, dict):
        return
    ok = bool(result.get("success") or result.get("ok"))
    if not ok:
        return
    used_in = {
        "platform": platform,
        "campaign_id": result.get("campaign_id"),
        "launched_at": datetime.now(timezone.utc).isoformat(),
        "via": "launch_form",
    }
    try:
        updated = storage.set_library_consumed(
            ids, True, platform=platform, used_in=used_in
        )
        app.logger.info(
            "library mark-off: %d/%d items marked used (platform=%s campaign=%s)",
            len(updated), len(ids), platform, result.get("campaign_id"),
        )
    except Exception as exc:  # noqa: BLE001
        app.logger.warning("library mark-off failed: %s", exc)


@app.route("/platform/switch", methods=["POST"])
def platform_switch():
    target = normalize_platform(request.form.get("platform") or request.args.get("platform"))
    session["platform"] = target
    nxt = request.form.get("next") or request.referrer or url_for("dashboard")
    return redirect(nxt)


@app.route("/")
def index():
    if _effective_token():
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    platform = _active_platform()
    # If env-configured credentials exist for this platform, skip login.
    if (
        (platform == "newsbreak" and _env_access_token())
        or (platform == "smartnews" and _env_smartnews_configured())
        or (platform == "outbrain" and _env_outbrain_configured())
        or (platform == "mediago" and _env_mediago_configured())
    ):
        return redirect(url_for("dashboard"))
    err = None
    if request.method == "POST":
        org_ids = _org_ids_from_form()
        try:
            if platform == "newsbreak":
                token = (request.form.get("access_token") or "").strip()
                if not token:
                    err = "Access token is required."
                elif not org_ids:
                    err = (
                        "At least one Organization ID is required "
                        "(comma-separated). Find it in NewsBreak Ad Manager "
                        "or API docs."
                    )
                else:
                    credentials = {"access_token": token, "org_ids": org_ids}
                    adapter = get_adapter(platform, **credentials)
                    adapter.verify()
                    storage.save_token(_user_id(), token, org_ids, platform=platform)
                    return redirect(url_for("dashboard"))
            elif platform == "outbrain":
                token = (request.form.get("access_token") or "").strip()
                username = (request.form.get("username") or "").strip()
                password = (request.form.get("password") or "").strip()
                if not token and not (username and password):
                    err = (
                        "Outbrain requires an OB-TOKEN-V1 access token "
                        "or a username + password."
                    )
                else:
                    credentials = {
                        "access_token": token,
                        "username": username,
                        "password": password,
                        "marketer_ids": org_ids,
                    }
                    adapter = get_adapter(platform, **credentials)
                    adapter.verify()
                    saved = (
                        {"access_token": token}
                        if token
                        else {"username": username, "password": password}
                    )
                    storage.save_token(_user_id(), saved, org_ids, platform=platform)
                    return redirect(url_for("dashboard"))
            elif platform == "mediago":
                api_token = (
                    request.form.get("api_token") or request.form.get("access_token") or ""
                ).strip()
                auth_level = (request.form.get("auth_level") or "auto").strip().lower() or "auto"
                if not api_token:
                    err = (
                        "MediaGo requires the Base64 API token from your "
                        "account manager (exchanged for a Bearer access token)."
                    )
                else:
                    credentials = {
                        "api_token": api_token,
                        "auth_level": auth_level,
                        "account_ids": org_ids,
                    }
                    adapter = get_adapter(platform, **credentials)
                    adapter.verify()
                    storage.save_token(
                        _user_id(),
                        {"api_token": api_token, "auth_level": auth_level},
                        org_ids,
                        platform=platform,
                    )
                    return redirect(url_for("dashboard"))
            else:  # smartnews
                client_id = (
                    request.form.get("client_id") or request.form.get("access_token") or ""
                ).strip()
                client_secret = (request.form.get("client_secret") or "").strip()
                if not client_id or not client_secret:
                    err = (
                        "SmartNews v3 requires both Developer App Client ID and "
                        "Client Secret (OAuth client_credentials)."
                    )
                else:
                    credentials = {
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "account_ids": org_ids,
                    }
                    adapter = get_adapter(platform, **credentials)
                    adapter.verify()
                    storage.save_token(
                        _user_id(),
                        {"client_id": client_id, "client_secret": client_secret},
                        org_ids,
                        platform=platform,
                    )
                    return redirect(url_for("dashboard"))
        except NewsBreakAPIError as e:
            err = str(e)
        except Exception as e:
            err = f"Login failed: {e}"
    if platform == "newsbreak":
        default_org = _cfg_val("NEWSBREAK_DEFAULT_ORG_IDS", "")
    elif platform == "smartnews":
        default_org = _cfg_val("SMARTNEWS_DEFAULT_ACCOUNT_IDS", "")
    elif platform == "outbrain":
        default_org = _cfg_val("OUTBRAIN_DEFAULT_MARKETER_IDS", "")
    elif platform == "mediago":
        default_org = _cfg_val("MEDIAGO_DEFAULT_ACCOUNT_IDS", "")
    else:
        default_org = ""
    return render_template(
        "login.html",
        error=err,
        default_org_ids=default_org,
    )


@app.route("/logout")
def logout():
    storage.delete_token(_user_id(), platform=_active_platform())
    session.pop("platform", None)
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
def dashboard():
    adapter = _adapter()
    if not adapter:
        return redirect(url_for("login"))
    err = None
    accounts: list[dict] = []
    try:
        accounts = adapter.get_accounts()
    except Exception as e:
        err = str(e)
    return render_template("dashboard.html", accounts=accounts, error=err)


# NewsBreak Inventory / Platforms field — added 2026-04 per
# https://advertising-api.newsbreak.com/hc/en-us/articles/45950972191629-Platforms
# These constants mirror the validator on NewsBreak's side; we enforce
# them client-side AND server-side so the user gets a clear message
# instead of a generic "platforms[] is invalid" 400 from /ad-set/create.
_NB_PLATFORM_VALUES = {
    "APP_AND_WEB_UNLIMITED",
    "NEWSBREAK",
    "SCOOPZ",
    "PREMIUM_PARTNERS_ALL",
    "PREMIUM_PARTNERS_GAMING",
    "PREMIUM_PARTNERS_NON_GAMING",
}
_NB_PLATFORM_PREMIUM_VALUES = {
    "PREMIUM_PARTNERS_ALL",
    "PREMIUM_PARTNERS_GAMING",
    "PREMIUM_PARTNERS_NON_GAMING",
}


def _build_newsbreak_platforms(form) -> list[str] | None:
    """Translate the launch-form `platforms_mode`/`platforms[]`/`platforms_premium`
    triplet into the JSON list NewsBreak expects on `/ad-set/create`.

    Returns:
        - ``None`` to mean "omit the field" (NewsBreak then defaults to
          ``["APP_AND_WEB_UNLIMITED"]``, identical to pre-Apr-2026 behavior).
        - A validated list of enum strings otherwise.

    Rules from NewsBreak's docs:
      1. APP_AND_WEB_UNLIMITED is exclusive — must appear alone.
      2. At most one PREMIUM_PARTNERS_* value may appear.
      3. Duplicate entries are rejected.
      4. Unknown values return 400.
    """
    mode = (form.get("platforms_mode") or "all").strip().lower()
    if mode != "custom":
        # Default = omit the field; lets the API pick its default.
        return None

    raw = [str(v).strip() for v in form.getlist("platforms")]
    premium = (form.get("platforms_premium") or "").strip()

    out: list[str] = []
    seen: set[str] = set()
    for v in raw:
        if not v or v not in _NB_PLATFORM_VALUES or v in seen:
            continue
        # Reject any premium value smuggled in as a checkbox — those go
        # through the dedicated single-select to enforce the at-most-one
        # rule. Skip silently rather than 500 the launch.
        if v in _NB_PLATFORM_PREMIUM_VALUES or v == "APP_AND_WEB_UNLIMITED":
            continue
        seen.add(v)
        out.append(v)

    if premium and premium in _NB_PLATFORM_PREMIUM_VALUES and premium not in seen:
        out.append(premium)

    if not out:
        # User picked "Custom" but cleared everything — be lenient and
        # fall back to the API default rather than failing the launch.
        return None
    return out


@app.route("/public/creative/<path:filename>")
def public_creative(filename):
    """Serve a prepared creative image without auth (Outbrain fetches it)."""
    directory = storage.public_creative_dir()
    return send_from_directory(directory, filename)


def _outbrain_public_base_url() -> str:
    """Best-effort public base URL for image hosting Outbrain/MediaGo can reach."""
    base = (
        _cfg_val("OUTBRAIN_PUBLIC_BASE_URL", "")
        or _cfg_val("MEDIAGO_PUBLIC_BASE_URL", "")
        or _cfg_val("AGENT_BASE_URL", "")
        or os.environ.get("RENDER_EXTERNAL_URL", "")
    ).strip()
    if base:
        return base.rstrip("/")
    # Fall back to the inbound request's host (works locally / single host).
    try:
        return request.url_root.rstrip("/")
    except Exception:
        return ""


def _host_creative_image(image_bytes: bytes, filename: str) -> str:
    """Persist a creative image and return its public, unauthenticated URL."""
    ext = os.path.splitext(filename)[1].lstrip(".") or "jpg"
    name = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(storage.public_creative_dir(), name)
    with open(path, "wb") as f:
        f.write(image_bytes)
    base = _outbrain_public_base_url()
    rel = url_for("public_creative", filename=name)
    return f"{base}{rel}" if base else rel


@app.route("/launch", methods=["GET", "POST"])
def launch():
    adapter = _adapter()
    if not adapter:
        return redirect(url_for("login"))
    platform = adapter.platform
    try:
        accounts = adapter.get_accounts()
    except Exception:
        accounts = []
    account_options = {str(a.get("id")): a.get("name", a.get("id")) for a in accounts if a.get("id")}

    # MediaGo native launches go through a dedicated handler.
    if platform == "mediago":
        from bulk_launcher_mediago import mediago_bulk_launch

        if request.method == "GET":
            return render_template(
                "launch.html",
                accounts=account_options,
                campaigns=[],
                pixels=storage.list_pixels(platform=platform),
                events=storage.list_events(platform=platform),
                offers=storage.list_offers(platform=platform),
            )
        exclusions = []
        account_id = (request.form.get("account_id") or "").strip()
        if account_id:
            exclusions = storage.load_site_exclusions(account_id, platform=platform)
        result = mediago_bulk_launch(
            adapter,
            form=request.form,
            files=request.files,
            host_image=_host_creative_image,
            exclusions=exclusions,
        )
        _studio_link_launch_if_any(platform, request.form.get("studio_gen_id"), result)
        _mark_library_used_if_any(platform, request.form, result)
        return render_template(
            "launch.html",
            accounts=account_options,
            campaigns=[],
            pixels=storage.list_pixels(platform=platform),
            events=storage.list_events(platform=platform),
            offers=storage.list_offers(platform=platform),
            result=result,
        )

    # Outbrain / Teads launches go through a dedicated handler.
    if platform == "outbrain":
        from bulk_launcher_outbrain import outbrain_bulk_launch

        campaigns_for_pick: list[dict] = []
        if account_options:
            try:
                first_acct = next(iter(account_options))
                campaigns_for_pick = adapter.get_campaigns(first_acct)
            except Exception:
                campaigns_for_pick = []

        if request.method == "GET":
            return render_template(
                "launch.html",
                accounts=account_options,
                campaigns=campaigns_for_pick,
                pixels=storage.list_pixels(platform=platform),
                events=storage.list_events(platform=platform),
                offers=storage.list_offers(platform=platform),
            )
        result = outbrain_bulk_launch(
            adapter,
            form=request.form,
            files=request.files,
            host_image=_host_creative_image,
        )
        _studio_link_launch_if_any(platform, request.form.get("studio_gen_id"), result)
        return render_template(
            "launch.html",
            accounts=account_options,
            campaigns=campaigns_for_pick,
            pixels=storage.list_pixels(platform=platform),
            events=storage.list_events(platform=platform),
            offers=storage.list_offers(platform=platform),
            result=result,
        )

    # SmartNews launches go through a dedicated handler.
    if platform == "smartnews":
        from bulk_launcher_smartnews import (
            creative_pair_from_square,
            smartnews_bulk_launch,
        )

        if request.method == "GET":
            return render_template(
                "launch.html",
                accounts=account_options,
                campaigns=[],
                pixels=storage.list_pixels(platform=platform),
                events=storage.list_events(platform=platform),
                offers=storage.list_offers(platform=platform),
            )
        # POST: delegate to SmartNews launcher (v3 API).
        result = smartnews_bulk_launch(
            adapter,
            form=request.form,
            files=request.files,
            pair_builder=creative_pair_from_square,
        )
        _studio_link_launch_if_any(platform, request.form.get("studio_gen_id"), result)
        _mark_library_used_if_any(platform, request.form, result)
        return render_template(
            "launch.html",
            accounts=account_options,
            campaigns=[],
            pixels=storage.list_pixels(platform=platform),
            events=storage.list_events(platform=platform),
            offers=storage.list_offers(platform=platform),
            result=result,
        )

    # NewsBreak path uses the legacy NewsBreakClient directly.
    cl = _client()
    if not cl:
        return redirect(url_for("login"))

    if request.method == "GET":
        return render_template(
            "launch.html",
            accounts=account_options,
            campaigns=[],
            pixels=storage.list_pixels(platform=platform),
            events=storage.list_events(platform=platform),
            offers=storage.list_offers(platform=platform),
        )

    # POST
    ad_account_id = request.form.get("ad_account_id", "").strip()
    campaign_mode = request.form.get("campaign_mode", "existing")
    campaign_id = request.form.get("campaign_id", "").strip() or None
    campaign_name = request.form.get("campaign_name", "API Campaign").strip()
    grouping = request.form.get("grouping", "groups_of_n")
    group_size = int(request.form.get("group_size") or 5)

    files = request.files.getlist("creatives")
    headlines = request.form.getlist("headline")
    bodies = request.form.getlist("body")
    landings = request.form.getlist("landing_url")

    creatives = []
    for i, f in enumerate(files):
        if not f or not f.filename:
            continue
        creatives.append(
            creative_from_upload(
                f,
                headlines[i] if i < len(headlines) else "",
                bodies[i] if i < len(bodies) else "",
                landings[i] if i < len(landings) else "",
            )
        )

    campaign_payload = None
    if campaign_mode == "new":
        campaign_payload = {
            "adAccountId": ad_account_id,
            "name": campaign_name,
            "objective": "WEB_CONVERSION",
        }

    bid_type = (request.form.get("bid_type") or "MAX_CONVERSION").strip().upper()
    needs_bid = bid_type in {"CPC", "CPM", "TARGET_CPA"}
    needs_roas = bid_type == "TARGET_ROAS"

    pixel_ref = (request.form.get("pixel_ref") or "").strip()
    pixel_id = (request.form.get("pixel_id") or "").strip()
    if pixel_ref and pixel_ref != "__manual__":
        for p in storage.list_pixels():
            if p.get("id") == pixel_ref:
                pixel_id = (p.get("pixel_id") or "").strip()
                break

    event_ref = (request.form.get("event_ref") or "").strip()
    tracking_id = ""
    custom_event_name = (request.form.get("custom_event_name") or "").strip()
    manual_event = (request.form.get("conversion_event") or "").strip().upper()
    if event_ref.startswith("nb:"):
        tracking_id = event_ref[3:].strip()
    elif event_ref and event_ref != "__manual__":
        for e in storage.list_events():
            if e.get("id") == event_ref:
                tracking_id = (e.get("tracking_id") or e.get("event_type") or "").strip()
                break
    else:
        tracking_id = custom_event_name if manual_event == "CUSTOM" else manual_event

    def _to_epoch(dt_local_str: str) -> Optional[int]:
        if not dt_local_str:
            return None
        try:
            dt = datetime.fromisoformat(dt_local_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            return None

    start_epoch = _to_epoch(request.form.get("start_time", ""))
    end_epoch = _to_epoch(request.form.get("end_time", ""))
    if start_epoch is None:
        start_epoch = int(datetime.now(timezone.utc).timestamp())
    # NewsBreak Create Ad Set requires endTime, but the UI's "no end date"
    # checkbox really just sends a far-future timestamp under the hood.
    # If the user left the box checked (no_end_date), honour that;
    # otherwise fall back to the supplied datetime or start + 30d.
    no_end_date = (request.form.get("no_end_date") or "").strip().lower() in {"1", "on", "true", "yes"}
    # NewsBreak's own Ad Manager UI uses int32-max (Jan 19, 2038) as the
    # "no end date" sentinel. Matching that avoids any int32 storage
    # overflow on their side.
    FAR_FUTURE = 2_147_483_647
    if no_end_date:
        end_epoch = FAR_FUTURE
    elif end_epoch is None:
        end_epoch = start_epoch + 30 * 24 * 3600

    ad_set_base: dict = {
        "name_prefix": (request.form.get("ad_set_name") or "").strip(),
        "budgetType": request.form.get("budget_type", "DAILY"),
        "budget": int(float(request.form.get("budget_dollars", "50") or 50) * 100),
        "bidType": bid_type,
        "startTime": start_epoch,
        "endTime": end_epoch,
    }
    if needs_bid:
        ad_set_base["bidRate"] = int(float(request.form.get("bid_dollars", "1") or 1) * 100)
    elif needs_roas:
        ad_set_base["roas"] = float(request.form.get("bid_dollars") or 2.0)

    if tracking_id:
        ad_set_base["trackingId"] = tracking_id

    _AGE_ALIASES = {
        "18-24": "18-30",
        "18-30": "18-30",
        "25-30": "18-30",
        "25-34": "18-30",
        "31-44": "31-44",
        "35-44": "31-44",
        "45-54": "45-64",
        "45-64": "45-64",
        "55-64": "45-64",
        "55+": "45-64",
        "65+": "65+",
        "65 or more": "65+",
    }
    _AGE_VALID = {"18-30", "31-44", "45-64", "65+"}
    raw_ages = [a.replace(" +", "+").strip() for a in request.form.getlist("age_groups") if a]
    age_groups = []
    for a in raw_ages:
        mapped = _AGE_ALIASES.get(a, a)
        if mapped in _AGE_VALID and mapped not in age_groups:
            age_groups.append(mapped)
    gender = (request.form.get("gender") or "").strip().lower()
    targeting: dict = {"location": {"positive": ["all"]}}
    if age_groups:
        targeting["ageGroup"] = {"positive": age_groups}
    if gender in {"male", "female"}:
        targeting["gender"] = {"positive": [gender]}
    ad_set_base["targeting"] = targeting

    ad_set_base["_ad_account_id_for_upload"] = ad_account_id
    ad_set_base["_brand_name"] = (request.form.get("brand_name") or "Advertiser").strip() or "Advertiser"
    ad_set_base["_cta"] = (request.form.get("cta") or "Learn More").strip() or "Learn More"

    # NewsBreak `platforms` field (added 2026-04 — see
    # https://advertising-api.newsbreak.com/hc/en-us/articles/45950972191629-Platforms).
    # Lives at the ad-set level alongside `targeting`, accepts a list of
    # enum strings (NEWSBREAK, SCOOPZ, PREMIUM_PARTNERS_*, APP_AND_WEB_UNLIMITED).
    # We collect it from the form here and pass through verbatim;
    # bulk_launcher.bulk_launch spreads ad_set_base into the JSON body.
    #
    # Earlier attempts to use `placements` / `trafficPlatforms` were silently
    # stripped — those fields never shipped publicly. The new `platforms`
    # field IS supported and goes into the create_ad_set payload.
    if platform == "newsbreak":
        platforms_value = _build_newsbreak_platforms(request.form)
        if platforms_value is not None:
            ad_set_base["platforms"] = platforms_value

    ad_set_base = {k: v for k, v in ad_set_base.items() if v is not None}

    normalized_grouping = (
        "all_in_one" if grouping == "all_in_one"
        else ("isolate" if grouping == "isolate" else "groups_of_n")
    )
    app.logger.info(
        "bulk_launch.params grouping_raw=%r grouping=%r group_size_raw=%r group_size=%d creatives=%d",
        grouping,
        normalized_grouping,
        request.form.get("group_size"),
        group_size,
        len(creatives),
    )

    result = bulk_launch(
        cl,
        ad_account_id=ad_account_id,
        campaign_mode=campaign_mode,
        campaign_id=campaign_id,
        campaign_payload=campaign_payload,
        ad_set_base=ad_set_base,
        creatives=creatives,
        grouping=normalized_grouping,
        group_size=group_size,
    )

    try:
        app.logger.info(
            "bulk_launch: account=%s success=%s campaign_id=%s ad_sets=%d errors=%s",
            ad_account_id,
            result.get("success"),
            result.get("campaign_id"),
            len(result.get("ad_sets") or []),
            result.get("errors") or [],
        )
    except Exception:
        pass

    _studio_link_launch_if_any(platform, request.form.get("studio_gen_id"), result)
    _mark_library_used_if_any(platform, request.form, result)

    return render_template(
        "launch.html",
        accounts=account_options,
        campaigns=[],
        pixels=storage.list_pixels(platform=platform),
        events=storage.list_events(platform=platform),
        offers=storage.list_offers(platform=platform),
        result=result,
    )


@app.route("/scaling")
def scaling():
    adapter = _adapter()
    if not adapter:
        return redirect(url_for("login"))
    try:
        accounts = adapter.get_accounts()
    except Exception:
        accounts = []
    account_options = {str(a.get("id")): a.get("name", a.get("id")) for a in accounts if a.get("id")}
    return render_template(
        "scaling.html",
        accounts=account_options,
        supports_ad_set=adapter.supports_ad_set_scope,
    )


@app.route("/sources")
def sources_page():
    adapter = _adapter()
    if not adapter:
        return redirect(url_for("login"))
    try:
        accounts = adapter.get_accounts()
    except Exception:
        accounts = []
    account_options = {str(a.get("id")): a.get("name", a.get("id")) for a in accounts if a.get("id")}
    return render_template(
        "sources.html",
        accounts=account_options,
        supports_site_block=hasattr(adapter, "block_sites"),
        supports_site_report=hasattr(adapter, "fetch_site_report_rows"),
    )


@app.route("/api/sources/report")
def api_sources_report():
    adapter = _adapter()
    if not adapter:
        return jsonify({"error": "not logged in"}), 401
    account_id = (request.args.get("account_id") or "").strip()
    if not account_id:
        return jsonify({"error": "account_id is required"}), 400
    try:
        days = int(request.args.get("days") or 7)
    except (TypeError, ValueError):
        days = 7
    days = max(1, min(days, 31))
    end = date.today()
    start = end - timedelta(days=days - 1)
    if not hasattr(adapter, "fetch_site_report_rows"):
        return jsonify(
            {
                "ok": False,
                "error": f"{adapter.label} does not expose a site-dimension report yet.",
                "rows": [],
                "exclusions": storage.load_site_exclusions(
                    account_id, platform=adapter.platform
                ),
            }
        )
    try:
        raw = adapter.fetch_site_report_rows(account_id, start, end)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "rows": []}), 502
    from platforms.mediago import score_source_rows

    scored = score_source_rows(raw)
    exclusions = storage.load_site_exclusions(account_id, platform=adapter.platform)
    excluded_ids = {str(x.get("site_id")) for x in exclusions}
    for row in scored:
        row["excluded"] = str(row.get("site_id")) in excluded_ids
    return jsonify(
        {
            "ok": True,
            "account_id": account_id,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "rows": scored,
            "exclusions": exclusions,
        }
    )


@app.route("/api/sources/exclusions", methods=["GET", "POST"])
def api_sources_exclusions():
    adapter = _adapter()
    if not adapter:
        return jsonify({"error": "not logged in"}), 401
    if request.method == "GET":
        account_id = (request.args.get("account_id") or "").strip()
        if not account_id:
            return jsonify({"error": "account_id is required"}), 400
        return jsonify(
            {
                "ok": True,
                "account_id": account_id,
                "sites": storage.load_site_exclusions(account_id, platform=adapter.platform),
            }
        )
    data = request.get_json(silent=True) or {}
    account_id = str(data.get("account_id") or "").strip()
    if not account_id:
        return jsonify({"error": "account_id is required"}), 400
    sites = data.get("sites") or []
    if not isinstance(sites, list):
        return jsonify({"error": "sites must be a list"}), 400
    saved = storage.save_site_exclusions(account_id, sites, platform=adapter.platform)
    return jsonify({"ok": True, "account_id": account_id, "sites": saved})


@app.route("/api/sources/apply", methods=["POST"])
def api_sources_apply():
    adapter = _adapter()
    if not adapter:
        return jsonify({"error": "not logged in"}), 401
    if not hasattr(adapter, "block_sites"):
        return jsonify(
            {"ok": False, "error": f"{adapter.label} cannot push a site blocklist via API."}
        ), 400
    data = request.get_json(silent=True) or {}
    account_id = str(data.get("account_id") or "").strip()
    if not account_id:
        return jsonify({"error": "account_id is required"}), 400
    sites = data.get("sites")
    if sites is None:
        sites = storage.load_site_exclusions(account_id, platform=adapter.platform)
    campaign_id = str(data.get("campaign_id") or "").strip() or None
    block = data.get("block", True)
    try:
        results = adapter.block_sites(
            account_id, sites, campaign_id=campaign_id, block=bool(block)
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 502
    storage.save_site_exclusions(account_id, sites, platform=adapter.platform)
    return jsonify(
        {
            "ok": True,
            "account_id": account_id,
            "campaign_id": campaign_id,
            "applied": len(sites or []),
            "results": results,
        }
    )


@app.route("/rules")
def rules_page():
    adapter = _adapter()
    if not adapter:
        return redirect(url_for("login"))
    try:
        accounts = adapter.get_accounts()
    except Exception:
        accounts = []
    account_options = {str(a.get("id")): a.get("name", a.get("id")) for a in accounts if a.get("id")}
    account_id = request.args.get("account_id", "")
    platform = _active_platform()
    rules = storage.load_rules(account_id, platform=platform) if account_id else []
    audit = storage.read_audit_tail(account_id, platform=platform) if account_id else []
    visible_templates = {
        tid: t
        for tid, t in RULE_TEMPLATES.items()
        if platform in (t.get("supported_platforms") or ["newsbreak"])
    }
    return render_template(
        "rules.html",
        accounts=account_options,
        account_id=account_id,
        rules=rules,
        audit=audit,
        templates=visible_templates,
    )


# --- JSON API ---


@app.route("/api/accounts")
def api_accounts():
    adapter = _adapter()
    if not adapter:
        return jsonify({"error": "unauthorized"}), 401
    try:
        return jsonify({"accounts": adapter.get_accounts(), "platform": adapter.platform})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/campaigns")
def api_campaigns():
    adapter = _adapter()
    if not adapter:
        return jsonify({"error": "unauthorized"}), 401
    aid = request.args.get("ad_account_id", "")
    if not aid:
        return jsonify({"error": "ad_account_id required"}), 400
    try:
        return jsonify({"campaigns": adapter.get_campaigns(aid)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/report/batch")
def api_report_batch():
    """Fetch spend/conversions/value for every ad account on the current token."""
    adapter = _adapter()
    if not adapter:
        return jsonify({"error": "unauthorized"}), 401
    try:
        raw_accounts = adapter.get_accounts()
    except Exception:
        raw_accounts = []
    accounts = {str(a.get("id")): a.get("name", a.get("id")) for a in raw_accounts if a.get("id")}
    if not accounts:
        return jsonify({"accounts": [], "totals": {"spend": 0, "conversions": 0, "value": 0}})
    days = int(request.args.get("days", "7"))
    end = date.today()
    start = end - timedelta(days=max(1, days))

    by_account: dict = {}
    if adapter.platform == "newsbreak":
        cl = _client()
        try:
            ids: list = []
            for aid in accounts.keys():
                try:
                    ids.append(int(aid))
                except (TypeError, ValueError):
                    ids.append(aid)
            payload = {
                "name": f"dashboard_batch_{start.isoformat()}_{end.isoformat()}",
                "timezone": "UTC",
                "dateRange": "FIXED",
                "startDate": start.isoformat(),
                "endDate": end.isoformat(),
                "filter": "AD_ACCOUNT",
                "filterIds": ids,
                "dimensions": ["AD_ACCOUNT"],
                "metrics": ["COST", "IMPRESSION", "CLICK", "CONVERSION", "VALUE"],
            }
            raw = cl.get_integrated_report(payload)
        except Exception as e:
            return jsonify({"error": str(e), "accounts": [], "totals": {"spend": 0, "conversions": 0, "value": 0}}), 400
        rows = normalize_report_rows(raw)
        for r in rows:
            aid = str(
                r.get("adAccountId")
                or r.get("ad_account_id")
                or r.get("accountId")
                or r.get("AD_ACCOUNT")
                or ""
            )
            if not aid:
                continue
            bucket = by_account.setdefault(aid, {"spend": 0.0, "conversions": 0.0, "value": 0.0})
            bucket["spend"] += float(r.get("spend") or 0)
            bucket["conversions"] += float(r.get("conversions") or 0)
            bucket["value"] += float(r.get("value") or r.get("conversionValue") or 0)
    else:
        # Fallback: iterate per-account via adapter
        for aid in accounts.keys():
            try:
                rows = adapter.fetch_report_rows(aid, "campaign", start, end)
            except Exception:
                rows = []
            bucket = by_account.setdefault(str(aid), {"spend": 0.0, "conversions": 0.0, "value": 0.0})
            for r in rows:
                bucket["spend"] += float(r.get("spend") or 0)
                bucket["conversions"] += float(r.get("conversions") or 0)
                bucket["value"] += float(r.get("value") or 0)

    out: list = []
    totals = {"spend": 0.0, "conversions": 0.0, "value": 0.0}
    for aid, name in accounts.items():
        b = by_account.get(str(aid), {"spend": 0.0, "conversions": 0.0, "value": 0.0})
        cpa = (b["spend"] / b["conversions"]) if b["conversions"] > 0 else None
        roas = (b["value"] / b["spend"]) if b["spend"] > 0 else None
        out.append(
            {
                "ad_account_id": str(aid),
                "name": name,
                "spend": round(b["spend"], 2),
                "conversions": int(round(b["conversions"])),
                "value": round(b["value"], 2),
                "cpa": round(cpa, 2) if cpa is not None else None,
                "roas": round(roas, 4) if roas is not None else None,
            }
        )
        totals["spend"] += b["spend"]
        totals["conversions"] += b["conversions"]
        totals["value"] += b["value"]

    blended_roas = (totals["value"] / totals["spend"]) if totals["spend"] > 0 else None
    return jsonify(
        {
            "accounts": out,
            "totals": {
                "spend": round(totals["spend"], 2),
                "conversions": int(round(totals["conversions"])),
                "value": round(totals["value"], 2),
                "roas": round(blended_roas, 4) if blended_roas is not None else None,
            },
            "days": days,
        }
    )


@app.route("/api/report")
def api_report():
    adapter = _adapter()
    if not adapter:
        return jsonify({"error": "unauthorized"}), 401
    aid = request.args.get("ad_account_id", "")
    level = request.args.get("level", "ad").lower()
    days = int(request.args.get("days", "7"))
    if not aid:
        return jsonify({"error": "ad_account_id required"}), 400

    # SmartNews AMv1 collapses ad_set to campaign
    if level == "ad_set" and not adapter.supports_ad_set_scope:
        level = "campaign"

    end = date.today()
    start = end - timedelta(days=max(1, days))

    if adapter.platform == "newsbreak":
        cl = _client()
        dim = "AD"
        if level == "ad_set":
            dim = "AD_SET"
        elif level == "campaign":
            dim = "CAMPAIGN"
        payload = build_report_payload(aid, start, end, dim)
        try:
            raw = cl.get_integrated_report(payload)
            rows = normalize_report_rows(raw)
            return jsonify({"rows": rows, "raw": raw, "platform": adapter.platform})
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    try:
        rows = adapter.fetch_report_rows(aid, level, start, end)
        return jsonify({"rows": rows, "platform": adapter.platform})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


def _normalize_newsbreak_events(raw: Any, ad_account_id: str) -> list:
    """Flatten the /event/getList response into a predictable list of dicts."""
    rows = unwrap_list_response(raw)
    if not rows and isinstance(raw, dict):
        data = raw.get("data")
        if isinstance(data, dict):
            inner = data.get("list") or data.get("rows") or data.get("events") or data.get("records")
            if isinstance(inner, list):
                rows = [x for x in inner if isinstance(x, dict)]
            elif not rows and any(k in data for k in ("id", "name", "eventType")):
                rows = [data]
    out = []
    for r in rows:
        tid = (
            r.get("id")
            or r.get("eventId")
            or r.get("trackingId")
            or r.get("tracking_id")
        )
        if tid is None:
            continue
        out.append(
            {
                "tracking_id": str(tid),
                "name": r.get("name") or r.get("eventName") or f"Event {tid}",
                "event_type": r.get("eventType") or r.get("event_type") or "",
                "pixel_id": str(r.get("pixelId") or r.get("pixel_id") or "") or None,
                "tracking_type": r.get("trackingType") or r.get("tracking_type") or "",
                "status": r.get("status") or "",
                "ad_account_id": ad_account_id,
                "source": "newsbreak",
                "raw": r,
            }
        )
    return out


@app.route("/api/newsbreak/events")
def api_newsbreak_events():
    cl = _client()
    if not cl:
        return jsonify({"error": "unauthorized"}), 401
    aid = request.args.get("ad_account_id", "").strip()
    if not aid:
        return jsonify({"error": "ad_account_id required"}), 400
    try:
        raw = cl.get_events(aid)
        return jsonify({"events": _normalize_newsbreak_events(raw, aid), "raw": raw})
    except Exception as e:
        return jsonify({"error": str(e), "events": []}), 400


@app.route("/api/newsbreak/sync-events", methods=["POST"])
def api_newsbreak_sync_events():
    """Pull events from every ad account we know about and upsert them into the local catalog."""
    if _active_platform() != "newsbreak":
        return jsonify({"error": "Switch to NewsBreak to sync events"}), 400
    cl = _client()
    if not cl:
        return jsonify({"error": "unauthorized"}), 401
    accounts = _ad_accounts_for_current_token()
    if not accounts:
        return jsonify({"error": "no ad accounts available"}), 400

    existing = {e.get("tracking_id"): e for e in storage.list_events(platform="newsbreak") if e.get("tracking_id")}
    existing_pixels = {p.get("pixel_id"): p for p in storage.list_pixels(platform="newsbreak") if p.get("pixel_id")}
    added = 0
    updated = 0
    pixels_added = 0
    per_account: list = []
    errors: list = []

    for acc_id, acc_name in accounts.items():
        try:
            raw = cl.get_events(acc_id)
            items = _normalize_newsbreak_events(raw, acc_id)
        except Exception as e:
            errors.append({"ad_account_id": acc_id, "error": str(e)})
            continue
        imported = 0
        for ev in items:
            tid = ev["tracking_id"]
            prior = existing.get(tid)
            merged = {
                "name": ev["name"],
                "tracking_id": tid,
                "event_type": ev.get("event_type") or (prior.get("event_type") if prior else ""),
                "pixel_id": ev.get("pixel_id") or (prior.get("pixel_id") if prior else None),
                "tracking_type": ev.get("tracking_type"),
                "ad_account_id": acc_id,
                "ad_account_name": acc_name,
                "source": "newsbreak",
                "is_custom": False,
            }
            if prior:
                merged["id"] = prior["id"]
                storage.upsert_event(merged, platform="newsbreak")
                updated += 1
            else:
                storage.upsert_event(merged, platform="newsbreak")
                added += 1
            imported += 1
            existing[tid] = {**(prior or {}), **merged}

            pid = ev.get("pixel_id")
            if pid and pid not in existing_pixels:
                storage.upsert_pixel(
                    {
                        "name": f"{acc_name} pixel {pid}",
                        "pixel_id": pid,
                        "ad_account_id": acc_id,
                        "ad_account_name": acc_name,
                        "source": "newsbreak",
                    },
                    platform="newsbreak",
                )
                existing_pixels[pid] = {"pixel_id": pid}
                pixels_added += 1
        per_account.append({"ad_account_id": acc_id, "name": acc_name, "imported": imported})

    return jsonify(
        {
            "ok": True,
            "added": added,
            "updated": updated,
            "pixels_added": pixels_added,
            "accounts": per_account,
            "errors": errors,
        }
    )


@app.route("/api/smartnews/sync-events", methods=["POST"])
def api_smartnews_sync_events():
    """Seed SmartNews built-in conversion events *and* import pixels from every ad account.

    SmartNews v3 exposes a real pixel list per ad account (``/ad_accounts/{id}/pixels``)
    — we call that here for every account the operator can see so the Offers
    screen's pixel dropdown isn't empty. Built-in conversion events (purchase,
    addToCart, viewContent, …) are also seeded on the same click since they're
    account-independent.
    """
    if _active_platform() != "smartnews":
        return jsonify({"error": "Switch to SmartNews to seed built-in events"}), 400
    adapter = _adapter()
    if not adapter:
        return jsonify({"error": "unauthorized"}), 401

    # -------- Built-in conversion events (shared across accounts) --------
    existing = {e.get("tracking_id"): e for e in storage.list_events(platform="smartnews") if e.get("tracking_id")}
    added = 0
    updated = 0

    events = adapter.list_events("")
    for ev in events:
        tid = ev.get("tracking_id") or ev.get("event_type")
        if not tid:
            continue
        prior = existing.get(tid)
        merged = {
            "name": ev.get("name") or tid,
            "tracking_id": tid,
            "event_type": ev.get("event_type") or tid,
            "pixel_id": None,
            "tracking_type": ev.get("tracking_type") or "builtin",
            "ad_account_id": "",
            "ad_account_name": "SmartNews built-in",
            "source": "smartnews",
            "is_custom": False,
        }
        if prior:
            merged["id"] = prior["id"]
            storage.upsert_event(merged, platform="smartnews")
            updated += 1
        else:
            storage.upsert_event(merged, platform="smartnews")
            added += 1
        existing[tid] = {**(prior or {}), **merged}

    # -------- Pixels (per ad account via SmartNews v3) --------
    existing_pixels = {
        str(p.get("pixel_id")): p
        for p in storage.list_pixels(platform="smartnews")
        if p.get("pixel_id")
    }
    pixels_added = 0
    pixels_updated = 0
    per_account: list = []
    errors: list = []

    try:
        raw_accounts = adapter.get_accounts() or []
    except Exception as e:  # pragma: no cover - network failure
        app.logger.warning("smartnews sync: get_accounts failed err=%s", e)
        raw_accounts = []
        errors.append({"ad_account_id": "-", "error": f"get_accounts: {e}"})

    for acc in raw_accounts:
        acc_id = str(acc.get("id") or acc.get("ad_account_id") or "")
        acc_name = acc.get("name") or acc.get("ad_account_name") or acc_id
        if not acc_id:
            continue
        try:
            pixels = adapter.list_pixels(acc_id) or []
        except Exception as e:  # pragma: no cover - network failure
            errors.append({"ad_account_id": acc_id, "error": str(e)})
            continue
        imported = 0
        for px in pixels:
            pid = str(
                px.get("pixel_tag_id")
                or px.get("pixel_id")
                or px.get("id")
                or ""
            )
            if not pid:
                continue
            pname = (
                px.get("name")
                or px.get("pixel_tag_name")
                or f"{acc_name} pixel {pid}"
            )
            prior = existing_pixels.get(pid)
            merged = {
                "name": pname,
                "pixel_id": pid,
                "ad_account_id": acc_id,
                "ad_account_name": acc_name,
                "source": "smartnews",
            }
            if prior:
                merged["id"] = prior["id"]
                storage.upsert_pixel(merged, platform="smartnews")
                pixels_updated += 1
            else:
                storage.upsert_pixel(merged, platform="smartnews")
                pixels_added += 1
            existing_pixels[pid] = {**(prior or {}), **merged}
            imported += 1
        per_account.append({"ad_account_id": acc_id, "name": acc_name, "imported": imported})

    return jsonify(
        {
            "ok": True,
            "added": added,
            "updated": updated,
            "pixels_added": pixels_added,
            "pixels_updated": pixels_updated,
            "accounts": per_account,
            "errors": errors,
        }
    )


@app.route("/api/smartnews/pixels/<account_id>")
def api_smartnews_pixels(account_id: str):
    """Return the list of tracking pixels for a SmartNews ad account."""
    if _active_platform() != "smartnews":
        return jsonify({"error": "Switch to SmartNews to list pixels"}), 400
    adapter = _adapter()
    if not adapter:
        return jsonify({"error": "unauthorized"}), 401
    try:
        pixels = adapter.list_pixels(account_id) or []
    except Exception as e:
        app.logger.warning("smartnews list_pixels failed account=%s err=%s", account_id, e)
        return jsonify({"error": str(e)}), 502
    return jsonify({"ok": True, "account_id": account_id, "pixels": pixels})


@app.route("/api/mediago/pixels/<account_id>")
def api_mediago_pixels(account_id: str):
    """Return conversion pixels for a MediaGo ad account (``GET /manage/v1/account``)."""
    if _active_platform() != "mediago":
        return jsonify({"error": "Switch to MediaGo to list pixels"}), 400
    adapter = _adapter()
    if not adapter:
        return jsonify({"error": "unauthorized"}), 401
    try:
        pixels = adapter.list_pixels(account_id) or []
    except Exception as e:
        app.logger.warning("mediago list_pixels failed account=%s err=%s", account_id, e)
        return jsonify({"error": str(e)}), 502
    return jsonify({"ok": True, "account_id": account_id, "pixels": pixels})


@app.route("/api/mediago/sync-events", methods=["POST"])
def api_mediago_sync_events():
    """Import MediaGo conversion pixels + events from every ad account.

    MediaGo does not expose a numeric pixel id. ``GET /manage/v1/account``
    returns a ``pixels`` array of conversion trackers (``conversion_name``,
    ``category``, ``status``). Campaign create then uses ``optimization_type``
    (1–10 / -1) as the conversion goal for that pixel.
    """
    if _active_platform() != "mediago":
        return jsonify({"error": "Switch to MediaGo to sync conversion pixels"}), 400
    adapter = _adapter()
    if not adapter:
        return jsonify({"error": "unauthorized"}), 401

    existing_events = {
        e.get("tracking_id"): e
        for e in storage.list_events(platform="mediago")
        if e.get("tracking_id")
    }
    existing_pixels = {
        str(p.get("pixel_id")): p
        for p in storage.list_pixels(platform="mediago")
        if p.get("pixel_id")
    }
    added = 0
    updated = 0
    pixels_added = 0
    pixels_updated = 0
    per_account: list = []
    errors: list = []

    try:
        raw_accounts = adapter.get_accounts() or []
    except Exception as e:
        app.logger.warning("mediago sync: get_accounts failed err=%s", e)
        raw_accounts = []
        errors.append({"ad_account_id": "-", "error": f"get_accounts: {e}"})

    for acc in raw_accounts:
        acc_id = str(acc.get("id") or acc.get("account_id") or "")
        acc_name = acc.get("name") or acc.get("account_name") or acc_id
        if not acc_id:
            continue
        try:
            pixels = adapter.list_pixels(acc_id) or []
        except Exception as e:
            errors.append({"ad_account_id": acc_id, "error": str(e)})
            continue
        imported = 0
        for px in pixels:
            pid = str(px.get("pixel_id") or px.get("conversion_name") or "")
            if not pid:
                continue
            pname = px.get("name") or px.get("category") or pid
            opt = str(px.get("optimization_type") or "") or None
            prior_px = existing_pixels.get(pid)
            merged_px = {
                "name": pname,
                "pixel_id": pid,
                "conversion_name": px.get("conversion_name") or pid,
                "optimization_type": opt,
                "ad_account_id": acc_id,
                "ad_account_name": acc_name,
                "source": "mediago",
            }
            if prior_px:
                merged_px["id"] = prior_px["id"]
                storage.upsert_pixel(merged_px, platform="mediago")
                pixels_updated += 1
            else:
                storage.upsert_pixel(merged_px, platform="mediago")
                pixels_added += 1
            existing_pixels[pid] = {**(prior_px or {}), **merged_px}

            prior_ev = existing_events.get(pid)
            merged_ev = {
                "name": pname,
                "tracking_id": pid,
                "event_type": pid,
                "pixel_id": pid,
                "optimization_type": opt,
                "ad_account_id": acc_id,
                "ad_account_name": acc_name,
                "source": "mediago",
                "is_custom": False,
            }
            if prior_ev:
                merged_ev["id"] = prior_ev["id"]
                storage.upsert_event(merged_ev, platform="mediago")
                updated += 1
            else:
                storage.upsert_event(merged_ev, platform="mediago")
                added += 1
            existing_events[pid] = {**(prior_ev or {}), **merged_ev}
            imported += 1
        per_account.append({"ad_account_id": acc_id, "name": acc_name, "imported": imported})

    return jsonify(
        {
            "ok": True,
            "added": added,
            "updated": updated,
            "pixels_added": pixels_added,
            "pixels_updated": pixels_updated,
            "accounts": per_account,
            "errors": errors,
        }
    )


@app.route("/api/adsets")
def api_adsets():
    adapter = _adapter()
    if not adapter:
        return jsonify({"error": "unauthorized"}), 401
    cid = request.args.get("campaign_id", "")
    account_id = request.args.get("ad_account_id", "")
    if not cid:
        return jsonify({"error": "campaign_id required"}), 400
    try:
        return jsonify({"ad_sets": adapter.get_ad_groups(account_id, cid)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/ads")
def api_ads():
    adapter = _adapter()
    if not adapter:
        return jsonify({"error": "unauthorized"}), 401
    parent_id = request.args.get("ad_set_id") or request.args.get("campaign_id") or ""
    account_id = request.args.get("ad_account_id", "")
    if not parent_id:
        return jsonify({"error": "ad_set_id (or campaign_id) required"}), 400
    try:
        return jsonify({"ads": adapter.get_ads(account_id, parent_id)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


def _resolve_status_target(data):
    """Return (level, entity_id) from request payload, supporting campaign/ad_set/ad."""
    level = (data.get("level") or "").strip().lower()
    if level == "campaign" or data.get("campaign_id") and not data.get("ad_id") and not data.get("ad_set_id"):
        return "campaign", data.get("campaign_id") or data.get("ad_id")
    if level == "ad_set" or data.get("ad_set_id") and not data.get("ad_id"):
        return "ad_set", data.get("ad_set_id") or data.get("ad_id")
    return "ad", data.get("ad_id")


@app.route("/api/ad/pause", methods=["POST"])
def api_ad_pause():
    adapter = _adapter()
    if not adapter:
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(force=True, silent=True) or {}
    level, entity_id = _resolve_status_target(data)
    account_id = data.get("ad_account_id")
    if not entity_id:
        return jsonify({"error": "entity id required"}), 400
    try:
        adapter.update_status(level, str(entity_id), enabled=False, account_id=account_id)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/ad/enable", methods=["POST"])
def api_ad_enable():
    adapter = _adapter()
    if not adapter:
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(force=True, silent=True) or {}
    level, entity_id = _resolve_status_target(data)
    account_id = data.get("ad_account_id")
    if not entity_id:
        return jsonify({"error": "entity id required"}), 400
    try:
        adapter.update_status(level, str(entity_id), enabled=True, account_id=account_id)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/adset/budget", methods=["POST"])
def api_adset_budget():
    adapter = _adapter()
    if not adapter:
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(force=True, silent=True) or {}
    entity_id = data.get("ad_set_id") or data.get("campaign_id")
    budget_dollars = float(data.get("budget_dollars", 0))
    budget_type = data.get("budget_type", "DAILY")
    account_id = data.get("ad_account_id")
    if not entity_id:
        return jsonify({"error": "ad_set_id or campaign_id required"}), 400
    level = "ad_set" if adapter.supports_ad_set_scope else "campaign"
    try:
        cents = int(budget_dollars * 100)
        adapter.update_budget(
            level,
            str(entity_id),
            budget_cents=cents,
            budget_type=budget_type,
            account_id=account_id,
        )
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/adset/budget_delta", methods=["POST"])
def api_adset_budget_delta():
    adapter = _adapter()
    if not adapter:
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(force=True, silent=True) or {}
    requested_level = (data.get("level") or "").strip().lower()
    if requested_level in ("ad_set", "campaign"):
        level = requested_level
    else:
        level = "ad_set" if adapter.supports_ad_set_scope else "campaign"
    if level == "campaign":
        entity_id = data.get("campaign_id") or data.get("ad_set_id")
    else:
        entity_id = data.get("ad_set_id") or data.get("campaign_id")
    pct = float(data.get("percent", 0))
    budget_type = data.get("budget_type", "DAILY")
    current_cents = data.get("current_budget_cents")
    account_id = data.get("ad_account_id")
    if not entity_id or current_cents is None:
        return jsonify({"error": "entity id and current_budget_cents required"}), 400
    try:
        cur = float(current_cents)
        new_cents = max(0, int(cur * (1 + pct / 100.0)))
        adapter.update_budget(
            level,
            str(entity_id),
            budget_cents=new_cents,
            budget_type=budget_type,
            account_id=account_id,
        )
        return jsonify({"ok": True, "new_budget_cents": new_cents})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/rules/save", methods=["POST"])
def api_rules_save():
    adapter = _adapter()
    if not adapter:
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(force=True, silent=True) or {}
    account_id = data.get("account_id")
    rule = data.get("rule")
    if not account_id or not rule:
        return jsonify({"error": "account_id and rule required"}), 400
    rule.setdefault("platform", adapter.platform)
    storage.upsert_rule(account_id, rule, platform=adapter.platform)
    return jsonify({"ok": True})


@app.route("/api/rules/list", methods=["GET"])
def api_rules_list():
    adapter = _adapter()
    if adapter is None:
        return jsonify({"error": "unauthorized"}), 401
    account_id = request.args.get("account_id", "")
    if not account_id:
        return jsonify({"error": "account_id required"}), 400
    return jsonify({"rules": storage.load_rules(account_id, platform=adapter.platform)})


@app.route("/api/rules/patch", methods=["POST"])
def api_rules_patch():
    adapter = _adapter()
    if adapter is None:
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(force=True, silent=True) or {}
    account_id = data.get("account_id")
    rule_id = data.get("rule_id")
    if not account_id or not rule_id:
        return jsonify({"error": "account_id and rule_id required"}), 400
    rules = storage.load_rules(account_id, platform=adapter.platform)
    for r in rules:
        if r.get("id") == rule_id:
            if "enabled" in data:
                r["enabled"] = bool(data["enabled"])
            if "dry_run" in data:
                r["dry_run"] = bool(data["dry_run"])
            storage.save_rules(account_id, rules, platform=adapter.platform)
            return jsonify({"ok": True})
    return jsonify({"error": "rule not found"}), 404


@app.route("/api/rules/from_template", methods=["POST"])
def api_rules_from_template():
    adapter = _adapter()
    if adapter is None:
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(force=True, silent=True) or {}
    account_id = data.get("account_id")
    template_id = data.get("template_id")
    if not account_id or not template_id:
        return jsonify({"error": "account_id and template_id required"}), 400
    try:
        rule = instantiate_template(template_id, account_id, platform=adapter.platform)
        rule["id"] = str(uuid.uuid4())
        storage.upsert_rule(account_id, rule, platform=adapter.platform)
        return jsonify({"ok": True, "rule": rule})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/rules/delete", methods=["POST"])
def api_rules_delete():
    adapter = _adapter()
    if not adapter:
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(force=True, silent=True) or {}
    account_id = data.get("account_id")
    rule_id = data.get("rule_id")
    if not account_id or not rule_id:
        return jsonify({"error": "account_id and rule_id required"}), 400
    storage.delete_rule(account_id, rule_id, platform=adapter.platform)
    return jsonify({"ok": True})


@app.route("/api/rules/run", methods=["POST"])
def api_rules_run():
    adapter = _adapter()
    if not adapter:
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(force=True, silent=True) or {}
    account_id = data.get("account_id")
    if not account_id:
        return jsonify({"error": "account_id required"}), 400
    rules = storage.load_rules(account_id, platform=adapter.platform)

    def audit(entry):
        storage.append_audit(account_id, entry, platform=adapter.platform)

    try:
        results = run_rules_for_account(adapter, account_id, rules, audit=audit)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/scheduler/run", methods=["POST"])
def api_scheduler_run():
    """Manual trigger (same as cron)."""
    if _adapter() is None:
        return jsonify({"error": "unauthorized"}), 401
    try:
        run_scheduled_rules()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# --- Catalog: pixels / events / offers ---


@app.route("/settings")
def settings_page():
    if not _effective_token():
        return redirect(url_for("login"))
    platform = _active_platform()
    return render_template(
        "settings.html",
        pixels=storage.list_pixels(platform=platform),
        events=storage.list_events(platform=platform),
        offers=storage.list_offers(platform=platform),
    )


def _auth_required():
    return _effective_token() is not None


@app.route("/api/pixels", methods=["GET"])
def api_pixels_list():
    if not _auth_required():
        return jsonify({"error": "unauthorized"}), 401
    return jsonify({"pixels": storage.list_pixels(platform=_active_platform())})


@app.route("/api/pixels", methods=["POST"])
def api_pixels_save():
    if not _auth_required():
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(force=True, silent=True) or {}
    name = (data.get("name") or "").strip()
    pixel_id = (data.get("pixel_id") or "").strip()
    if not name or not pixel_id:
        return jsonify({"error": "name and pixel_id required"}), 400
    saved = storage.upsert_pixel(
        {
            "id": data.get("id"),
            "name": name,
            "pixel_id": pixel_id,
            "notes": (data.get("notes") or "").strip(),
        },
        platform=_active_platform(),
    )
    return jsonify({"ok": True, "pixel": saved})


@app.route("/api/pixels/<item_id>", methods=["DELETE"])
def api_pixels_delete(item_id):
    if not _auth_required():
        return jsonify({"error": "unauthorized"}), 401
    ok = storage.delete_pixel(item_id, platform=_active_platform())
    return jsonify({"ok": ok})


@app.route("/api/events", methods=["GET"])
def api_events_list():
    if not _auth_required():
        return jsonify({"error": "unauthorized"}), 401
    return jsonify({"events": storage.list_events(platform=_active_platform())})


@app.route("/api/events", methods=["POST"])
def api_events_save():
    if not _auth_required():
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(force=True, silent=True) or {}
    name = (data.get("name") or "").strip()
    tracking_id = (data.get("tracking_id") or "").strip()
    event_type = (data.get("event_type") or tracking_id).strip()
    if not name:
        return jsonify({"error": "name required"}), 400
    if not tracking_id and not event_type:
        return jsonify({"error": "tracking_id or event_type required"}), 400
    saved = storage.upsert_event(
        {
            "id": data.get("id"),
            "name": name,
            "event_type": event_type,
            "tracking_id": tracking_id or event_type,
            "is_custom": bool(data.get("is_custom")),
        },
        platform=_active_platform(),
    )
    return jsonify({"ok": True, "event": saved})


@app.route("/api/events/<item_id>", methods=["DELETE"])
def api_events_delete(item_id):
    if not _auth_required():
        return jsonify({"error": "unauthorized"}), 401
    ok = storage.delete_event(item_id, platform=_active_platform())
    return jsonify({"ok": ok})


@app.route("/api/offers", methods=["GET"])
def api_offers_list():
    if not _auth_required():
        return jsonify({"error": "unauthorized"}), 401
    return jsonify({"offers": storage.list_offers(platform=_active_platform())})


@app.route("/api/offers", methods=["POST"])
def api_offers_save():
    if not _auth_required():
        return jsonify({"error": "unauthorized"}), 401
    platform = _active_platform()
    data = request.get_json(force=True, silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400

    def _num(v):
        try:
            return float(v) if v not in (None, "") else None
        except (TypeError, ValueError):
            return None

    payout = _num(data.get("payout"))
    target_cpa = _num(data.get("target_cpa"))
    if payout is not None and target_cpa is None:
        target_cpa = round(payout * 0.8, 2)

    raw_accounts = data.get("ad_account_ids") or []
    if isinstance(raw_accounts, str):
        raw_accounts = [x.strip() for x in raw_accounts.split(",")]
    ad_account_ids = [str(x).strip() for x in raw_accounts if str(x).strip()]

    pixel_id = (data.get("pixel_id") or "").strip()
    incoming_pixels = data.get("pixels") if isinstance(data.get("pixels"), dict) else None
    existing_offer = None
    if data.get("id"):
        for o in storage.list_offers(platform=platform):
            if o.get("id") == data.get("id"):
                existing_offer = o
                break
    pixels_map = storage.merge_offer_platform_pixels(
        existing_offer, platform, pixel_id, incoming_pixels
    )

    saved = storage.upsert_offer(
        {
            "id": data.get("id"),
            "name": name,
            "brand_name": (data.get("brand_name") or "").strip(),
            "cta": (data.get("cta") or "").strip(),
            "landing_url": (data.get("landing_url") or "").strip(),
            "headline": (data.get("headline") or "").strip(),
            "body": (data.get("body") or "").strip(),
            "pixel_id": pixel_id or pixels_map.get(platform, ""),
            "pixels": pixels_map,
            "event_id": (data.get("event_id") or "").strip(),
            "payout": payout,
            "target_cpa": target_cpa,
            "utm_parameters": (data.get("utm_parameters") or "").strip(),
            "notes": (data.get("notes") or "").strip(),
            "ad_account_ids": ad_account_ids,
        },
        platform=platform,
    )
    return jsonify({"ok": True, "offer": saved})


@app.route("/api/offers/<item_id>", methods=["DELETE"])
def api_offers_delete(item_id):
    if not _auth_required():
        return jsonify({"error": "unauthorized"}), 401
    ok = storage.delete_offer(item_id, platform=_active_platform())
    return jsonify({"ok": ok})


# -------------------------------------------------------------------
# AI Ad Studio routes
# -------------------------------------------------------------------
try:
    from ai_studio import analyzer as _studio_analyzer
    from ai_studio import image_gen as _studio_image_gen
    from ai_studio import library as _studio_library
    from ai_studio import pipeline as _studio_pipeline
    from ai_studio import winners as _studio_winners
    from ai_studio.research import (
        bandit as _studio_bandit,
        discover as _studio_discover,
        lifecycle as _studio_lifecycle,
    )
    _AI_STUDIO_AVAILABLE = True
except Exception as _studio_exc:  # noqa: BLE001
    app.logger.warning("AI Ad Studio import failed: %s", _studio_exc)
    _AI_STUDIO_AVAILABLE = False


def _studio_required():
    if not _auth_required():
        return jsonify({"error": "unauthorized"}), 401
    if not _AI_STUDIO_AVAILABLE:
        return jsonify({"error": "ai_studio module unavailable"}), 503
    return None


@app.route("/studio")
def studio_page():
    if not _effective_token():
        return redirect(url_for("login"))
    platform = _active_platform()
    return render_template(
        "studio.html",
        platform=platform,
        offers=storage.list_offers(platform=platform),
        catalog=[
            {
                "id": s.id,
                "name": s.name,
                "description": s.description,
            }
            for s in (
                _studio_pipeline.prompt_gen.STYLE_CATALOG
                if _AI_STUDIO_AVAILABLE
                else []
            )
        ],
    )


@app.route("/api/studio/winners", methods=["GET"])
def api_studio_winners():
    guard = _studio_required()
    if guard is not None:
        return guard
    # "all" (default) returns the cross-platform pool so SmartNews + NewsBreak
    # winners feed one shared database. ?scope=platform falls back to just the
    # active surface for debugging / per-platform views.
    scope = (request.args.get("scope") or "all").lower()
    active = _active_platform()
    if scope == "platform":
        winners = storage.list_winners(platform=active)
        for w in winners:
            w.setdefault("source_platform", active)
    else:
        winners = storage.list_all_winners()
    for w in winners:
        local = w.get("image_local_path")
        if local and os.path.exists(local):
            w["local_image_url"] = url_for(
                "winner_image",
                platform=w.get("source_platform") or active,
                filename=os.path.basename(local),
            )
    return jsonify({
        "winners": winners,
        "scope": scope,
        "active_platform": active,
    })


@app.route("/studio/winner-image/<platform>/<path:filename>")
def winner_image(platform, filename):
    guard = _studio_required()
    if guard is not None:
        return guard
    directory = storage.winner_image_dir(platform)
    return send_from_directory(directory, filename)


@app.route("/api/studio/refresh-winners", methods=["POST"])
def api_studio_refresh_winners():
    guard = _studio_required()
    if guard is not None:
        return guard
    body = request.get_json(silent=True) or {}
    def _num(key, default):
        try:
            return type(default)(body[key]) if key in body and body[key] not in (None, "") else default
        except (TypeError, ValueError):
            return default
    days = _num("days", 14)
    min_spend = _num("min_spend", 20.0)
    min_conv = _num("min_conv", 3.0)
    cpa_factor = _num("cpa_factor", 1.0)
    adapter = _adapter()
    if not adapter:
        return jsonify({"error": "no adapter configured"}), 400
    try:
        summary = _studio_winners.refresh_winners(
            adapter,
            platform=_active_platform(),
            days=days,
            min_spend=min_spend,
            min_conv=min_conv,
            cpa_factor=cpa_factor,
        )
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("refresh_winners failed")
        return jsonify({"error": str(exc)}), 500
    return jsonify({"ok": True, "summary": summary})


@app.route("/api/studio/winners/refetch-creatives", methods=["POST"])
def api_studio_refetch_winner_creatives():
    """Surgical backfill: re-resolve creative metadata for already-stored
    winners that came back from ``refresh_winners`` without an image (the
    NewsBreak creative resolver fails for some ads when the per-account
    index doesn't surface ``ad_id → ad_group_id``). Iterates every
    configured platform unless the caller pins one explicitly.
    """
    guard = _studio_required()
    if guard is not None:
        return guard
    body = request.get_json(silent=True) or {}
    requested_platform = body.get("platform")
    only_imageless = body.get("only_imageless", True)
    raw_ad_ids = body.get("ad_ids")
    ad_ids = [str(x) for x in raw_ad_ids] if isinstance(raw_ad_ids, list) else None

    if requested_platform:
        platforms = [normalize_platform(requested_platform)]
    else:
        platforms = list(PLATFORMS)

    summaries: List[Dict[str, Any]] = []
    totals = {"scanned": 0, "updated": 0, "still_imageless": 0, "missing_account": 0}
    errors: List[Dict[str, Any]] = []
    for plat in platforms:
        adapter = _adapter(plat)
        if not adapter:
            errors.append({"platform": plat, "error": "no adapter configured"})
            continue
        try:
            summary = _studio_winners.refetch_creatives(
                adapter,
                platform=plat,
                only_imageless=bool(only_imageless),
                ad_ids=ad_ids,
            )
        except Exception as exc:  # noqa: BLE001
            app.logger.exception("refetch_creatives failed for platform=%s", plat)
            errors.append({"platform": plat, "error": str(exc)})
            continue
        summaries.append(summary)
        for k in totals:
            totals[k] += int(summary.get(k, 0) or 0)
        errors.extend(summary.get("errors", []))

    return jsonify({
        "ok": True,
        "summary": {**totals, "errors": errors, "per_platform": summaries},
    })


@app.route("/api/studio/winners/cleanup-imageless", methods=["POST"])
def api_studio_cleanup_imageless_winners():
    """Hard-delete winners that have neither a cached local image nor a
    remote ``image_url`` — the rows that show up as a generic placeholder
    icon + the bare word "Ad" in the winners table. Iterates every
    configured platform unless one is pinned in the body.
    """
    guard = _studio_required()
    if guard is not None:
        return guard
    body = request.get_json(silent=True) or {}
    requested_platform = body.get("platform")
    if requested_platform:
        platforms = [normalize_platform(requested_platform)]
    else:
        platforms = list(PLATFORMS)
    per_platform: List[Dict[str, Any]] = []
    deleted = 0
    kept = 0
    for plat in platforms:
        try:
            result = _studio_winners.cleanup_imageless(platform=plat)
        except Exception as exc:  # noqa: BLE001
            app.logger.exception("cleanup_imageless failed for platform=%s", plat)
            per_platform.append({"platform": plat, "error": str(exc)})
            continue
        deleted += int(result.get("deleted", 0) or 0)
        kept += int(result.get("kept", 0) or 0)
        per_platform.append(result)
    return jsonify({"ok": True, "deleted": deleted, "kept": kept, "per_platform": per_platform})


@app.route("/api/studio/winners/<platform>/<ad_id>", methods=["DELETE"])
def api_studio_delete_winner(platform, ad_id):
    """Delete a single winner from ``winners.json``. Same caveat as the
    bulk cleanup: the row will come back on the next ``refresh_winners``
    pass if it still clears spend / conv thresholds — but at least the
    user can clear out one row at a time without losing the rest.
    """
    guard = _studio_required()
    if guard is not None:
        return guard
    plat = normalize_platform(platform)
    if not ad_id:
        return jsonify({"ok": False, "error": "missing ad_id"}), 400
    removed = storage.delete_winner(ad_id, platform=plat)
    return jsonify({"ok": True, "removed": removed, "ad_id": ad_id, "platform": plat})


@app.route("/api/studio/insights/<offer_id>", methods=["GET"])
def api_studio_insights(offer_id):
    guard = _studio_required()
    if guard is not None:
        return guard
    platform = _active_platform()
    fresh = request.args.get("fresh", "").lower() in {"1", "true", "yes"}
    if fresh:
        try:
            insights = _studio_analyzer.analyze_offer(
                offer_id, platform=platform, force=True
            )
        except Exception as exc:  # noqa: BLE001
            app.logger.exception("analyze_offer failed")
            return jsonify({"error": str(exc)}), 500
    else:
        insights = storage.load_insights(offer_id, platform=platform)
        if not insights:
            try:
                insights = _studio_analyzer.analyze_offer(
                    offer_id, platform=platform
                )
            except Exception as exc:  # noqa: BLE001
                app.logger.exception("analyze_offer failed")
                return jsonify({"error": str(exc)}), 500
    return jsonify({"insights": insights})


def _library_row_to_image_payload(row, *, platform):
    """Build the same {b64, mime, model, ...} shape :func:`generate_ads`
    returns, but for an item drained from the prebuilt library.

    We expose the file via ``/studio/library-image/...`` (``url`` field)
    instead of inlining base64 — keeps the JSON payload small when the
    operator clicks Generate and we drain 10 items at once.
    """
    filename = row.get("filename") or ""
    url = ""
    if filename:
        url = url_for(
            "library_image",
            platform=row.get("platform") or platform,
            filename=filename,
        )
    return {
        "style_id": row.get("style_id"),
        "style_name": row.get("style_name"),
        "is_candidate": bool(row.get("is_candidate")),
        "url": url,
        "mime": row.get("mime") or "image/png",
        "model": row.get("model"),
        "ms": row.get("ms"),
        "aspect": row.get("aspect"),
        "error": None,
        "from_library": True,
        "library_id": row.get("library_id"),
    }


def _library_row_to_prompt_payload(row):
    """Mirror the ``prompts`` field shape from ``generate_ads``."""
    return {
        "style_id": row.get("style_id"),
        "style_name": row.get("style_name"),
        "is_candidate": bool(row.get("is_candidate")),
        "prompt": row.get("prompt"),
        "cta_label": None,
        "cta_color": None,
        "angle": row.get("headline"),
        "from_library": True,
    }


@app.route("/api/studio/generate", methods=["POST"])
def api_studio_generate():
    guard = _studio_required()
    if guard is not None:
        return guard
    data = request.get_json(force=True, silent=True) or {}
    offer_id = str(data.get("offer_id") or "").strip()
    if not offer_id:
        return jsonify({"error": "offer_id required"}), 400
    platform = normalize_platform(data.get("platform") or _active_platform())
    count = int(data.get("count") or 10)
    model_image = (data.get("model_image") or "nano-banana-2").strip()
    model_analyzer = (data.get("model_analyzer") or "").strip() or None
    style_mix = data.get("style_mix") or None
    research_ratio = data.get("research_ratio")
    try:
        research_ratio = float(research_ratio) if research_ratio is not None else None
    except (TypeError, ValueError):
        research_ratio = None

    use_library = bool(data.get("use_library", True))

    # ------------------------------------------------------------------
    # Step 1: drain the prebuilt library FIFO. This makes the user-facing
    # "Generate" button feel near-instant when the daily topup has had a
    # chance to run. We still fall through to fresh rendering for the
    # gap if there aren't enough library items.
    # ------------------------------------------------------------------
    library_rows = []
    if use_library and not style_mix:
        try:
            library_rows = storage.consume_library_items(
                offer_id, count, platform=platform
            )
        except Exception as exc:  # noqa: BLE001
            app.logger.exception("studio/generate: library drain failed: %s", exc)

    fresh_count = max(0, count - len(library_rows))
    fresh_result = None
    if fresh_count > 0:
        try:
            fresh_result = _studio_pipeline.generate_ads(
                offer_id,
                platform=platform,
                count=fresh_count,
                model_image=model_image,
                model_analyzer=model_analyzer,
                style_mix=style_mix,
                research_ratio=research_ratio,
            )
        except Exception as exc:  # noqa: BLE001
            app.logger.exception("studio/generate failed")
            # If we already drained library items but fresh failed, hand
            # back what we have so the operator isn't blocked. Otherwise
            # surface the real error.
            if not library_rows:
                return jsonify({"error": str(exc)}), 500
            fresh_result = {
                "gen_id": str(uuid.uuid4()),
                "offer_id": str(offer_id),
                "platform": platform,
                "aspect": library_rows[0].get("aspect", "1:1"),
                "allocation": [],
                "prompts": [],
                "images": [],
            }

    # Compose payload: library items first, fresh renders after.
    base = fresh_result or {
        "gen_id": str(uuid.uuid4()),
        "offer_id": str(offer_id),
        "platform": platform,
        "aspect": (library_rows[0].get("aspect") if library_rows else "1:1"),
        "allocation": [],
        "prompts": [],
        "images": [],
    }

    images_payload = [_library_row_to_image_payload(r, platform=platform) for r in library_rows]
    prompts_payload = [_library_row_to_prompt_payload(r) for r in library_rows]
    for img, prompt in zip(base["images"], base["prompts"]):
        images_payload.append(
            {
                "style_id": img.get("style_id"),
                "style_name": img.get("style_name"),
                "is_candidate": img.get("is_candidate"),
                "b64": img.get("b64"),
                "mime": img.get("mime"),
                "model": img.get("model"),
                "ms": img.get("ms"),
                "aspect": img.get("aspect"),
                "error": img.get("error"),
                "from_library": False,
            }
        )
        prompts_payload.append(
            {
                "style_id": prompt.get("style_id"),
                "style_name": prompt.get("style_name"),
                "is_candidate": prompt.get("is_candidate"),
                "prompt": prompt.get("prompt"),
                "cta_label": prompt.get("cta_label"),
                "cta_color": prompt.get("cta_color"),
                "angle": prompt.get("angle"),
                "from_library": False,
            }
        )

    thin = {
        "gen_id": base["gen_id"],
        "offer_id": str(offer_id),
        "platform": platform,
        "aspect": base.get("aspect", "1:1"),
        "allocation": base.get("allocation", []),
        "prompts": prompts_payload,
        "images": images_payload,
        "library_consumed": len(library_rows),
        "freshly_rendered": len(base.get("images") or []),
    }
    return jsonify(thin)


# ---- Prebuilt library -------------------------------------------------


@app.route("/api/studio/library/status", methods=["GET"])
def api_studio_library_status():
    guard = _studio_required()
    if guard is not None:
        return guard
    platform = normalize_platform(request.args.get("platform") or _active_platform())
    counts = storage.library_counts(platform=platform)
    return jsonify(
        {
            "active_platform": platform,
            "counts": counts,
            "target_per_offer": _studio_library.DEFAULT_TARGET_PER_OFFER,
        }
    )


@app.route("/api/studio/library/list", methods=["GET"])
def api_studio_library_list():
    """List the prebuilt library items so the Research → Prebuilt images
    mode can render them as a thumbnail grid.

    Each row carries an ``image_url`` pointing at the existing
    /studio/library-image/<platform>/<filename> route so the browser
    can render the bytes without an extra round-trip.

    Query params:
      platform : "newsbreak" | "smartnews" | "all" (default).
                 Empty / unknown values are treated as "all" so the
                 browser sees every prebuilt image regardless of which
                 ad-platform is currently selected in the session.
      offer_id : optional exact-match filter
      include_consumed : 1/true/yes to include rows already burned by Generate
    """
    guard = _studio_required()
    if guard is not None:
        return guard
    raw_platform = (request.args.get("platform") or "").strip().lower()
    if raw_platform in ("", "all"):
        targets = list(PLATFORMS)  # newsbreak + smartnews + …
    else:
        targets = [normalize_platform(raw_platform)]
    offer_id = (request.args.get("offer_id") or "").strip() or None
    include_consumed = (request.args.get("include_consumed") or "").lower() in (
        "1", "true", "yes",
    )
    out = []
    per_platform_counts: Dict[str, int] = {}
    per_platform_usage: Dict[str, Dict[str, int]] = {}
    total_unused = 0
    total_used = 0
    for plat in targets:
        try:
            # Always list everything so used/unused counts stay accurate
            # regardless of the include_consumed toggle; filter below.
            rows = storage.list_library_items(
                platform=plat,
                offer_id=offer_id,
                include_consumed=True,
            )
        except Exception as exc:  # noqa: BLE001
            app.logger.exception("studio/library/list failed for %s", plat)
            continue
        kept = 0
        n_used = 0
        n_unused = 0
        for row in rows:
            filename = (row.get("filename") or "").strip()
            if not filename:
                # Stale rows where the render failed before we could patch
                # the filename in. Skip — they have nothing to show.
                continue
            is_consumed = bool(row.get("consumed_at"))
            if is_consumed:
                n_used += 1
            else:
                n_unused += 1
            if is_consumed and not include_consumed:
                continue
            item = dict(row)
            item["platform"] = plat
            item["image_url"] = url_for(
                "library_image", platform=plat, filename=filename
            )
            out.append(item)
            kept += 1
        per_platform_counts[plat] = kept
        per_platform_usage[plat] = {"unused": n_unused, "used": n_used}
        total_unused += n_unused
        total_used += n_used
    out.sort(key=lambda r: str(r.get("created_at") or ""), reverse=True)
    return jsonify(
        {
            "platforms": targets,
            "platform_counts": per_platform_counts,
            "platform_usage": per_platform_usage,
            "unused_count": total_unused,
            "used_count": total_used,
            "items": out,
            "count": len(out),
        }
    )


@app.route("/api/studio/library/topup", methods=["POST"])
def api_studio_library_topup():
    """On-demand topup for one offer × active platform.

    Lets the operator force a topup right before clicking Generate
    (instead of waiting for the daily cron). Capped per-call by
    AD_STUDIO_LIBRARY_MAX_BATCH so a stuck button can't run away.
    """
    guard = _studio_required()
    if guard is not None:
        return guard
    data = request.get_json(force=True, silent=True) or {}
    offer_id = str(data.get("offer_id") or "").strip()
    if not offer_id:
        return jsonify({"error": "offer_id required"}), 400
    platform = normalize_platform(data.get("platform") or _active_platform())
    model_image = (data.get("model_image") or "nano-banana-2").strip()
    target = data.get("target")
    try:
        target = int(target) if target is not None else None
    except (TypeError, ValueError):
        target = None
    try:
        result = _studio_library.topup_offer(
            offer_id,
            platform=platform,
            target=target,
            model_image=model_image,
        )
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("studio/library/topup failed")
        return jsonify({"error": str(exc)}), 500
    return jsonify(result)


@app.route("/api/studio/library/mark", methods=["POST"])
def api_studio_library_mark():
    """Manually mark library items used / unused (housekeeping toggles).

    Body: {library_ids: [...], consumed: bool, platform: "newsbreak"|...}
    Marking used stamps ``used_in.via = "manual"`` so it's distinguishable
    from launch-driven consumption; marking unused clears the trail.
    """
    guard = _studio_required()
    if guard is not None:
        return guard
    data = request.get_json(force=True, silent=True) or {}
    ids = [str(x).strip() for x in (data.get("library_ids") or []) if str(x).strip()]
    if not ids:
        return jsonify({"error": "library_ids required"}), 400
    consumed = bool(data.get("consumed", True))
    platform = normalize_platform(data.get("platform") or _active_platform())
    used_in = None
    if consumed:
        used_in = {
            "platform": platform,
            "via": "manual",
            "launched_at": None,
        }
    try:
        updated = storage.set_library_consumed(
            ids, consumed, platform=platform, used_in=used_in
        )
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("studio/library/mark failed")
        return jsonify({"error": str(exc)}), 500
    return jsonify(
        {
            "ok": True,
            "updated": len(updated),
            "consumed": consumed,
            "library_ids": [r.get("library_id") for r in updated],
        }
    )


_LIBRARY_EXPORT_FIELDS = [
    "library_id",
    "platform",
    "offer_id",
    "style_id",
    "style_name",
    "aspect",
    "headline",
    "prompt",
    "model",
    "status",
    "created_at",
    "consumed_at",
    "used_via",
    "used_campaign_id",
]


@app.route("/api/studio/library/export", methods=["GET"])
def api_studio_library_export():
    """Download the banked library prompts as CSV or JSON.

    Lets the operator feed the stashed ideas (full image-generation prompt
    + headline/style metadata) into other tools — e.g. generating the same
    concepts for Facebook. Honors the same filters as the Research view:

      format   : csv (default) | json
      status   : unused (default) | used | all
      platform : "" / "all" (default) | newsbreak | smartnews | ...
      offer_id : optional exact-match filter
    """
    guard = _studio_required()
    if guard is not None:
        return guard
    import csv
    import io
    import json as _json

    fmt = (request.args.get("format") or "csv").strip().lower()
    if fmt not in ("csv", "json"):
        return jsonify({"error": "format must be csv or json"}), 400
    status = (request.args.get("status") or "unused").strip().lower()
    if status not in ("unused", "used", "all"):
        return jsonify({"error": "status must be unused, used or all"}), 400
    raw_platform = (request.args.get("platform") or "").strip().lower()
    if raw_platform in ("", "all"):
        targets = list(PLATFORMS)
    else:
        targets = [normalize_platform(raw_platform)]
    offer_id = (request.args.get("offer_id") or "").strip() or None

    rows: List[Dict[str, Any]] = []
    for plat in targets:
        try:
            items = storage.list_library_items(
                platform=plat, offer_id=offer_id, include_consumed=True
            )
        except Exception:  # noqa: BLE001
            app.logger.exception("studio/library/export: list failed for %s", plat)
            continue
        for it in items:
            consumed = bool(it.get("consumed_at"))
            if status == "unused" and consumed:
                continue
            if status == "used" and not consumed:
                continue
            used_in = it.get("used_in") or {}
            rows.append(
                {
                    "library_id": it.get("library_id"),
                    "platform": plat,
                    "offer_id": it.get("offer_id"),
                    "style_id": it.get("style_id"),
                    "style_name": it.get("style_name"),
                    "aspect": it.get("aspect"),
                    "headline": it.get("headline"),
                    "prompt": it.get("prompt"),
                    "model": it.get("model"),
                    "status": "used" if consumed else "unused",
                    "created_at": it.get("created_at"),
                    "consumed_at": it.get("consumed_at"),
                    "used_via": used_in.get("via"),
                    "used_campaign_id": used_in.get("campaign_id"),
                }
            )
    rows.sort(key=lambda r: str(r.get("created_at") or ""), reverse=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filename = f"library-prompts-{status}-{stamp}.{fmt}"
    if fmt == "json":
        payload = _json.dumps(
            {
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "platforms": targets,
                "count": len(rows),
                "items": rows,
            },
            indent=2,
            default=str,
        )
        return Response(
            payload,
            mimetype="application/json",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_LIBRARY_EXPORT_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.route("/studio/library-image/<platform>/<path:filename>")
def library_image(platform, filename):
    guard = _studio_required()
    if guard is not None:
        return guard
    directory = storage.library_image_dir(platform)
    return send_from_directory(directory, filename)


@app.route("/api/studio/link-launch", methods=["POST"])
def api_studio_link_launch():
    guard = _studio_required()
    if guard is not None:
        return guard
    data = request.get_json(force=True, silent=True) or {}
    gen_id = (data.get("gen_id") or "").strip()
    ad_ids = data.get("ad_ids") or []
    if not gen_id:
        return jsonify({"error": "gen_id required"}), 400
    updated = storage.update_generation(
        gen_id,
        {"launched_ad_ids": [str(x) for x in ad_ids]},
        platform=_active_platform(),
    )
    return jsonify({"ok": bool(updated), "generation": updated})


# ---- Research ---------------------------------------------------------


@app.route("/api/studio/research/candidates", methods=["GET"])
def api_studio_research_candidates():
    guard = _studio_required()
    if guard is not None:
        return guard
    return jsonify(
        {"candidates": storage.list_style_candidates(platform=_active_platform())}
    )


_CANDIDATE_EXPORT_FIELDS = [
    "style_id",
    "platform",
    "name",
    "description",
    "visual_cues",
    "prompt_template",
    "source",
    "status",
    "offer_id",
    "trials",
    "wins",
    "impressions",
    "spend",
    "conversions",
    "cpa",
    "ctr",
    "created_at",
    "updated_at",
    "last_trial_at",
]


def _candidate_matches_source(cand: Dict[str, Any], chip: str) -> bool:
    """Mirror the Research tab's fuzzy source-chip matching."""
    raw = str(cand.get("source") or "").lower()
    anchor = str((cand.get("source_meta") or {}).get("anchor_platform") or "").lower()
    if chip == "gethookd":
        return raw in ("gethookd", "scrape")
    if chip == "cluster_winners":
        return raw in ("cluster_winners", "cluster")
    if chip == "meta":
        return raw == "meta" or (raw == "public_scout" and anchor in ("meta", "mixed"))
    if chip == "tiktok":
        return raw == "tiktok" or (raw == "public_scout" and anchor in ("tiktok", "mixed"))
    return chip in raw


@app.route("/api/studio/research/candidates/export", methods=["GET"])
def api_studio_research_candidates_export():
    """Download the banked style ideas (concepts) as CSV or JSON.

    These are the raw idea records from the Research tab — angle name,
    description, visual cues, and the reusable image-generation prompt
    template — independent of whether any image was ever rendered from
    them, so they can be fed to other generators (Facebook, etc.).

      format   : csv (default) | json
      status   : "" / all (default) | candidate | testing | promoted | archived
      source   : "" / all (default) | cluster_winners | gethookd | brainstorm | meta | tiktok | scholar | upload
      platform : "" (default = active platform) | all | newsbreak | ...
    """
    guard = _studio_required()
    if guard is not None:
        return guard
    import csv
    import io
    import json as _json

    fmt = (request.args.get("format") or "csv").strip().lower()
    if fmt not in ("csv", "json"):
        return jsonify({"error": "format must be csv or json"}), 400
    status = (request.args.get("status") or "all").strip().lower() or "all"
    source = (request.args.get("source") or "all").strip().lower() or "all"
    raw_platform = (request.args.get("platform") or "").strip().lower()
    if raw_platform == "all":
        targets = list(PLATFORMS)
    elif raw_platform:
        targets = [normalize_platform(raw_platform)]
    else:
        targets = [_active_platform()]

    rows: List[Dict[str, Any]] = []
    for plat in targets:
        try:
            cands = storage.list_style_candidates(platform=plat)
        except Exception:  # noqa: BLE001
            app.logger.exception("research/candidates/export: list failed for %s", plat)
            continue
        for c in cands:
            sid = str(c.get("style_id") or c.get("id") or "")
            # "catalog:" rows are lifecycle bookkeeping mirrors, not ideas.
            if sid.startswith("catalog:"):
                continue
            if status != "all" and str(c.get("status") or "candidate").lower() != status:
                continue
            if source != "all" and not _candidate_matches_source(c, source):
                continue
            meta = c.get("source_meta") or {}
            cues = c.get("visual_cues") or []
            rows.append(
                {
                    "style_id": sid,
                    "platform": plat,
                    "name": c.get("name"),
                    "description": c.get("description"),
                    "visual_cues": " | ".join(str(x) for x in cues) if fmt == "csv" else cues,
                    "prompt_template": c.get("prompt_template"),
                    "source": c.get("source"),
                    "status": c.get("status") or "candidate",
                    "offer_id": meta.get("offer_id"),
                    "trials": c.get("trials") or 0,
                    "wins": c.get("wins") or 0,
                    "impressions": c.get("impressions") or 0,
                    "spend": c.get("spend") or 0,
                    "conversions": c.get("conversions") or 0,
                    "cpa": c.get("cpa"),
                    "ctr": c.get("ctr"),
                    "created_at": c.get("created_at"),
                    "updated_at": c.get("updated_at"),
                    "last_trial_at": c.get("last_trial_at"),
                }
            )
    rows.sort(key=lambda r: str(r.get("created_at") or ""), reverse=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filename = f"style-ideas-{status}-{stamp}.{fmt}"
    if fmt == "json":
        payload = _json.dumps(
            {
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "source": source,
                "platforms": targets,
                "count": len(rows),
                "items": rows,
            },
            indent=2,
            default=str,
        )
        return Response(
            payload,
            mimetype="application/json",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CANDIDATE_EXPORT_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.route("/api/studio/research/candidates/import", methods=["POST"])
def api_studio_research_candidates_import():
    """Restore banked style ideas from a previous ``/candidates/export`` file.

    Accepts a multipart ``file`` (the CSV or JSON produced by the export
    route) or a raw JSON body ``{"items": [...]}``. The merge is strictly
    additive by ``style_id``:
      * unknown ids are inserted,
      * existing records only gain fields they are currently missing
        (so lifecycle stamps like ``status=promoted`` are never undone),
      * nothing is ever deleted.
    """
    guard = _studio_required()
    if guard is not None:
        return guard
    import csv
    import io
    import json as _json

    items: List[Dict[str, Any]] = []
    up = request.files.get("file")
    if up is not None:
        raw = up.read().decode("utf-8-sig", errors="replace")
        stripped = raw.lstrip()
        if (up.filename or "").lower().endswith(".json") or stripped[:1] in ("{", "["):
            try:
                payload = _json.loads(raw)
            except ValueError:
                return jsonify({"error": "file is not valid JSON"}), 400
            items = payload.get("items") if isinstance(payload, dict) else payload
        else:
            items = list(csv.DictReader(io.StringIO(raw)))
    else:
        data = request.get_json(force=True, silent=True) or {}
        items = data.get("items") or []
    if not isinstance(items, list) or not items:
        return jsonify({"error": "no items to import"}), 400

    content_fields = ("name", "description", "visual_cues", "prompt_template", "source")
    inserted = updated = skipped = 0
    existing_by_platform: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in items:
        if not isinstance(row, dict):
            skipped += 1
            continue
        sid = str(row.get("style_id") or "").strip()
        if not sid or sid.startswith("catalog:"):
            skipped += 1
            continue
        raw_plat = str(row.get("platform") or "").strip().lower()
        plat = normalize_platform(raw_plat) if raw_plat else _active_platform()
        if plat not in existing_by_platform:
            existing_by_platform[plat] = {
                str(c.get("style_id") or c.get("id") or ""): c
                for c in storage.list_style_candidates(platform=plat)
            }
        existing = existing_by_platform[plat].get(sid)

        patch: Dict[str, Any] = {"style_id": sid}
        for field in content_fields:
            val = row.get(field)
            if field == "visual_cues" and isinstance(val, str):
                val = [x.strip() for x in val.split("|") if x.strip()]
            if val in (None, "", []):
                continue
            if existing is not None and existing.get(field) not in (None, "", []):
                continue  # fill gaps only — never overwrite live data
            patch[field] = val
        if existing is None:
            patch["status"] = str(row.get("status") or "candidate")
            if row.get("created_at"):
                patch["created_at"] = row["created_at"]
            if row.get("offer_id"):
                patch["source_meta"] = {"offer_id": row["offer_id"]}
        elif len(patch) == 1:
            skipped += 1
            continue

        saved = storage.upsert_style_candidate(patch, platform=plat)
        existing_by_platform[plat][sid] = saved
        if existing is None:
            inserted += 1
        else:
            updated += 1
    return jsonify(
        {"ok": True, "inserted": inserted, "updated": updated, "skipped": skipped}
    )


@app.route("/api/studio/research/discover", methods=["POST"])
def api_studio_research_discover():
    guard = _studio_required()
    if guard is not None:
        return guard
    data = request.get_json(force=True, silent=True) or {}
    mode = (data.get("mode") or "").strip()
    platform = normalize_platform(data.get("platform") or _active_platform())
    offer_id = (data.get("offer_id") or "").strip() or None
    try:
        if mode == "cluster_winners":
            out = _studio_discover.discover_from_winners(platform)
        elif mode == "gethookd":
            out = _studio_discover.discover_from_gethookd(
                platform=platform,
                keywords=data.get("keywords") or [],
                filters=data.get("filters") or None,
                limit=int(data.get("limit") or 50),
            )
        elif mode == "brainstorm":
            if not offer_id:
                return jsonify({"error": "offer_id required for brainstorm"}), 400
            offer = None
            for o in storage.list_offers(platform=platform):
                if str(o.get("id")) == offer_id:
                    offer = o
                    break
            if not offer:
                return jsonify({"error": "offer not found"}), 404
            out = _studio_discover.discover_from_brainstorm(
                offer,
                platform=platform,
                count=int(data.get("count") or 5),
            )
        elif mode == "all":
            out = _studio_discover.discover_all(
                platform,
                offer_id=offer_id,
                gethookd_keywords=data.get("keywords") or None,
                gethookd_filters=data.get("filters") or None,
            )
        else:
            return jsonify({"error": f"unknown mode: {mode}"}), 400
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("research/discover %s failed", mode)
        return jsonify({"error": str(exc)}), 500
    return jsonify({"ok": True, "mode": mode, "candidates": out})


@app.route("/api/studio/research/gethookd/authcheck", methods=["GET"])
def api_studio_research_gethookd_authcheck():
    guard = _studio_required()
    if guard is not None:
        return guard
    import requests as _req
    key = os.environ.get("GETHOOKD_API_KEY", "")
    if not key:
        return jsonify({"error": "GETHOOKD_API_KEY not configured"}), 400
    base = os.environ.get("GETHOOKD_BASE_URL", "https://app.gethookd.ai/api/v1")
    try:
        resp = _req.get(
            f"{base}/authcheck",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        payload = resp.json() if resp.content else {}
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500
    return jsonify({"ok": resp.ok, "status": resp.status_code, "payload": payload})


@app.route("/api/studio/research/promote", methods=["POST"])
def api_studio_research_promote():
    guard = _studio_required()
    if guard is not None:
        return guard
    data = request.get_json(force=True, silent=True) or {}
    style_id = (data.get("style_id") or "").strip()
    if not style_id:
        return jsonify({"error": "style_id required"}), 400
    updated = storage.upsert_style_candidate(
        {
            "style_id": style_id,
            "status": "promoted",
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        },
        platform=_active_platform(),
    )
    return jsonify({"ok": True, "candidate": updated})


@app.route("/api/studio/research/archive", methods=["POST"])
def api_studio_research_archive():
    guard = _studio_required()
    if guard is not None:
        return guard
    data = request.get_json(force=True, silent=True) or {}
    style_id = (data.get("style_id") or "").strip()
    if not style_id:
        return jsonify({"error": "style_id required"}), 400
    updated = storage.upsert_style_candidate(
        {
            "style_id": style_id,
            "status": "archived",
            "archived_at": datetime.now(timezone.utc).isoformat(),
        },
        platform=_active_platform(),
    )
    return jsonify({"ok": True, "candidate": updated})


@app.route("/api/studio/research/upload-refs", methods=["POST"])
def api_studio_research_upload_refs():
    guard = _studio_required()
    if guard is not None:
        return guard
    offer_id = (request.form.get("offer_id") or "").strip()
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "no files"}), 400
    platform = _active_platform()
    save_dir = os.path.join(
        storage._catalog_dir(platform), "research_uploads", offer_id or "misc"
    )
    os.makedirs(save_dir, exist_ok=True)
    saved_paths = []
    for f in files:
        if not f.filename:
            continue
        name = f"{uuid.uuid4().hex[:8]}_{os.path.basename(f.filename)}"
        full = os.path.join(save_dir, name)
        f.save(full)
        saved_paths.append(full)
    try:
        out = _studio_discover.discover_from_uploads(
            offer_id, platform=platform, image_paths=saved_paths
        )
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("discover_from_uploads failed")
        return jsonify({"error": str(exc), "saved_paths": saved_paths}), 500
    return jsonify({"ok": True, "candidates": out, "saved_paths": saved_paths})


@app.route("/api/studio/research/runs", methods=["GET"])
def api_studio_research_runs():
    guard = _studio_required()
    if guard is not None:
        return guard
    return jsonify(
        {"runs": storage.list_research_runs(platform=_active_platform(), limit=200)}
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5055)), debug=True)

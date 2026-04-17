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
from typing import Optional

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
    session,
    url_for,
)

load_dotenv()

import storage
from bulk_launcher import bulk_launch, creative_from_upload
from newsbreak_api import NewsBreakAPIError, NewsBreakClient, unwrap_list_response
from rules_engine import (
    RULE_TEMPLATES,
    build_report_payload,
    instantiate_template,
    normalize_report_rows,
    run_rules_for_account,
)
from scheduler import run_scheduled_rules, start_scheduler

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
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

storage.ensure_dirs()


def _user_id() -> str:
    if "uid" not in session:
        session["uid"] = str(uuid.uuid4())
    return session["uid"]


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


def _effective_token() -> dict | None:
    """Prefer env-configured token so Render redeploys don't force re-login."""
    env_tok = _env_access_token()
    if env_tok:
        return {"access_token": env_tok, "org_ids": _env_org_ids()}
    return storage.load_token(_user_id())


def _client() -> NewsBreakClient | None:
    tok = _effective_token()
    if not tok:
        return None
    return NewsBreakClient(tok["access_token"])


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
    if _basic_auth_ok():
        return None
    return Response(
        "Authentication required",
        401,
        {"WWW-Authenticate": 'Basic realm="NewsBreak Launcher"', "Content-Type": "text/plain"},
    )


@app.before_request
def _start_jobs():
    if not app.config.get("_scheduler_started"):
        start_scheduler(15)
        app.config["_scheduler_started"] = True


@app.route("/")
def index():
    if _effective_token():
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if _env_access_token():
        return redirect(url_for("dashboard"))
    err = None
    if request.method == "POST":
        token = (request.form.get("access_token") or "").strip()
        org_ids = _org_ids_from_form()
        if not token:
            err = "Access token is required."
        elif not org_ids:
            err = "At least one Organization ID is required (comma-separated). Find it in NewsBreak Ad Manager or API docs."
        else:
            try:
                client = NewsBreakClient(token)
                client.get_ad_accounts(org_ids)
                storage.save_token(_user_id(), token, org_ids)
                return redirect(url_for("dashboard"))
            except NewsBreakAPIError as e:
                err = str(e)
            except Exception as e:
                err = f"Login failed: {e}"
    return render_template("login.html", error=err, default_org_ids=_cfg_val("NEWSBREAK_DEFAULT_ORG_IDS", ""))


@app.route("/logout")
def logout():
    storage.delete_token(_user_id())
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
def dashboard():
    cl = _client()
    if not cl:
        return redirect(url_for("login"))
    err = None
    accounts: list[dict] = []
    try:
        tok = _effective_token() or {}
        raw = cl.get_ad_accounts(tok.get("org_ids") or [])
        accounts = _flatten_ad_accounts(raw)
    except Exception as e:
        err = str(e)
    return render_template("dashboard.html", accounts=accounts, error=err)


@app.route("/launch", methods=["GET", "POST"])
def launch():
    cl = _client()
    if not cl:
        return redirect(url_for("login"))
    tok = _effective_token() or {}
    accounts = _flatten_ad_accounts(cl.get_ad_accounts(tok.get("org_ids") or []))
    account_options = {str(a.get("id")): a.get("name", a.get("id")) for a in accounts if a.get("id")}

    if request.method == "GET":
        return render_template(
            "launch.html",
            accounts=account_options,
            campaigns=[],
            pixels=storage.list_pixels(),
            events=storage.list_events(),
            offers=storage.list_offers(),
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

    # Inventory (NewsBreak-only vs Unlimited). Public API field is
    # `trafficPlatforms` with enum values ["APP", "WEB"] per the Ad Set
    # targeting structure docs. (The Nova Ad Manager UI internally uses
    # ["NEWSBREAK", "NBWEB"] which the public API silently drops as
    # unknown enum values — do not use those here.)
    nb_only = (request.form.get("nb_only_inventory") or "").strip().lower() in {"1", "on", "true", "yes"}
    if nb_only:
        ad_set_base["trafficPlatforms"] = ["APP", "WEB"]

    ad_set_base = {k: v for k, v in ad_set_base.items() if v is not None}

    result = bulk_launch(
        cl,
        ad_account_id=ad_account_id,
        campaign_mode=campaign_mode,
        campaign_id=campaign_id,
        campaign_payload=campaign_payload,
        ad_set_base=ad_set_base,
        creatives=creatives,
        grouping="all_in_one" if grouping == "all_in_one" else ("isolate" if grouping == "isolate" else "groups_of_n"),
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

    return render_template(
        "launch.html",
        accounts=account_options,
        campaigns=[],
        pixels=storage.list_pixels(),
        events=storage.list_events(),
        offers=storage.list_offers(),
        result=result,
    )


@app.route("/scaling")
def scaling():
    cl = _client()
    if not cl:
        return redirect(url_for("login"))
    tok = _effective_token() or {}
    accounts = _flatten_ad_accounts(cl.get_ad_accounts(tok.get("org_ids") or []))
    account_options = {str(a.get("id")): a.get("name", a.get("id")) for a in accounts if a.get("id")}
    return render_template("scaling.html", accounts=account_options)


@app.route("/rules")
def rules_page():
    cl = _client()
    if not cl:
        return redirect(url_for("login"))
    tok = _effective_token() or {}
    accounts = _flatten_ad_accounts(cl.get_ad_accounts(tok.get("org_ids") or []))
    account_options = {str(a.get("id")): a.get("name", a.get("id")) for a in accounts if a.get("id")}
    account_id = request.args.get("account_id", "")
    rules = storage.load_rules(account_id) if account_id else []
    audit = storage.read_audit_tail(account_id) if account_id else []
    return render_template(
        "rules.html",
        accounts=account_options,
        account_id=account_id,
        rules=rules,
        audit=audit,
        templates=RULE_TEMPLATES,
    )


# --- JSON API ---


@app.route("/api/accounts")
def api_accounts():
    cl = _client()
    if not cl:
        return jsonify({"error": "unauthorized"}), 401
    tok = _effective_token() or {}
    try:
        raw = cl.get_ad_accounts(tok.get("org_ids") or [])
        return jsonify({"accounts": _flatten_ad_accounts(raw)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/campaigns")
def api_campaigns():
    cl = _client()
    if not cl:
        return jsonify({"error": "unauthorized"}), 401
    aid = request.args.get("ad_account_id", "")
    if not aid:
        return jsonify({"error": "ad_account_id required"}), 400
    try:
        raw = cl.get_campaigns(aid)
        return jsonify({"campaigns": unwrap_list_response(raw)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/report/batch")
def api_report_batch():
    """Fetch spend/conversions/value for every ad account on the current token in ONE call."""
    cl = _client()
    if not cl:
        return jsonify({"error": "unauthorized"}), 401
    accounts = _ad_accounts_for_current_token()
    if not accounts:
        return jsonify({"accounts": [], "totals": {"spend": 0, "conversions": 0, "value": 0}})
    days = int(request.args.get("days", "7"))
    end = date.today()
    start = end - timedelta(days=max(1, days))
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
    by_account: dict = {}
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
    cl = _client()
    if not cl:
        return jsonify({"error": "unauthorized"}), 401
    aid = request.args.get("ad_account_id", "")
    level = request.args.get("level", "ad").lower()
    days = int(request.args.get("days", "7"))
    if not aid:
        return jsonify({"error": "ad_account_id required"}), 400
    dim = "AD"
    if level == "ad_set":
        dim = "AD_SET"
    elif level == "campaign":
        dim = "CAMPAIGN"
    end = date.today()
    start = end - timedelta(days=max(1, days))

    payload = build_report_payload(aid, start, end, dim)
    try:
        raw = cl.get_integrated_report(payload)
        rows = normalize_report_rows(raw)
        return jsonify({"rows": rows, "raw": raw})
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
    cl = _client()
    if not cl:
        return jsonify({"error": "unauthorized"}), 401
    accounts = _ad_accounts_for_current_token()
    if not accounts:
        return jsonify({"error": "no ad accounts available"}), 400

    existing = {e.get("tracking_id"): e for e in storage.list_events() if e.get("tracking_id")}
    existing_pixels = {p.get("pixel_id"): p for p in storage.list_pixels() if p.get("pixel_id")}
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
                storage.upsert_event(merged)
                updated += 1
            else:
                storage.upsert_event(merged)
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
                    }
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


@app.route("/api/adsets")
def api_adsets():
    cl = _client()
    if not cl:
        return jsonify({"error": "unauthorized"}), 401
    cid = request.args.get("campaign_id", "")
    if not cid:
        return jsonify({"error": "campaign_id required"}), 400
    try:
        raw = cl.get_ad_sets(cid)
        return jsonify({"ad_sets": unwrap_list_response(raw)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/ads")
def api_ads():
    cl = _client()
    if not cl:
        return jsonify({"error": "unauthorized"}), 401
    asid = request.args.get("ad_set_id", "")
    if not asid:
        return jsonify({"error": "ad_set_id required"}), 400
    try:
        raw = cl.get_ads(asid)
        return jsonify({"ads": unwrap_list_response(raw)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/ad/pause", methods=["POST"])
def api_ad_pause():
    cl = _client()
    if not cl:
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(force=True, silent=True) or {}
    ad_id = data.get("ad_id")
    if not ad_id:
        return jsonify({"error": "ad_id required"}), 400
    try:
        cl.update_ad_status(str(ad_id), "OFF")
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/ad/enable", methods=["POST"])
def api_ad_enable():
    cl = _client()
    if not cl:
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(force=True, silent=True) or {}
    ad_id = data.get("ad_id")
    if not ad_id:
        return jsonify({"error": "ad_id required"}), 400
    try:
        cl.update_ad_status(str(ad_id), "ON")
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/adset/budget", methods=["POST"])
def api_adset_budget():
    cl = _client()
    if not cl:
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(force=True, silent=True) or {}
    ad_set_id = data.get("ad_set_id")
    budget_dollars = float(data.get("budget_dollars", 0))
    budget_type = data.get("budget_type", "DAILY")
    if not ad_set_id:
        return jsonify({"error": "ad_set_id required"}), 400
    try:
        cents = int(budget_dollars * 100)
        cl.update_ad_set(str(ad_set_id), {"budget": cents, "budgetType": budget_type})
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/adset/budget_delta", methods=["POST"])
def api_adset_budget_delta():
    cl = _client()
    if not cl:
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(force=True, silent=True) or {}
    ad_set_id = data.get("ad_set_id")
    pct = float(data.get("percent", 0))
    budget_type = data.get("budget_type", "DAILY")
    current_cents = data.get("current_budget_cents")
    if not ad_set_id or current_cents is None:
        return jsonify({"error": "ad_set_id and current_budget_cents required"}), 400
    try:
        cur = float(current_cents)
        new_cents = max(0, int(cur * (1 + pct / 100.0)))
        cl.update_ad_set(str(ad_set_id), {"budget": new_cents, "budgetType": budget_type})
        return jsonify({"ok": True, "new_budget_cents": new_cents})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/rules/save", methods=["POST"])
def api_rules_save():
    cl = _client()
    if not cl:
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(force=True, silent=True) or {}
    account_id = data.get("account_id")
    rule = data.get("rule")
    if not account_id or not rule:
        return jsonify({"error": "account_id and rule required"}), 400
    storage.upsert_rule(account_id, rule)
    return jsonify({"ok": True})


@app.route("/api/rules/list", methods=["GET"])
def api_rules_list():
    if _client() is None:
        return jsonify({"error": "unauthorized"}), 401
    account_id = request.args.get("account_id", "")
    if not account_id:
        return jsonify({"error": "account_id required"}), 400
    return jsonify({"rules": storage.load_rules(account_id)})


@app.route("/api/rules/patch", methods=["POST"])
def api_rules_patch():
    if _client() is None:
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(force=True, silent=True) or {}
    account_id = data.get("account_id")
    rule_id = data.get("rule_id")
    if not account_id or not rule_id:
        return jsonify({"error": "account_id and rule_id required"}), 400
    rules = storage.load_rules(account_id)
    for r in rules:
        if r.get("id") == rule_id:
            if "enabled" in data:
                r["enabled"] = bool(data["enabled"])
            if "dry_run" in data:
                r["dry_run"] = bool(data["dry_run"])
            storage.save_rules(account_id, rules)
            return jsonify({"ok": True})
    return jsonify({"error": "rule not found"}), 404


@app.route("/api/rules/from_template", methods=["POST"])
def api_rules_from_template():
    if _client() is None:
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(force=True, silent=True) or {}
    account_id = data.get("account_id")
    template_id = data.get("template_id")
    if not account_id or not template_id:
        return jsonify({"error": "account_id and template_id required"}), 400
    try:
        rule = instantiate_template(template_id, account_id)
        rule["id"] = str(uuid.uuid4())
        storage.upsert_rule(account_id, rule)
        return jsonify({"ok": True, "rule": rule})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/rules/delete", methods=["POST"])
def api_rules_delete():
    cl = _client()
    if not cl:
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(force=True, silent=True) or {}
    account_id = data.get("account_id")
    rule_id = data.get("rule_id")
    if not account_id or not rule_id:
        return jsonify({"error": "account_id and rule_id required"}), 400
    storage.delete_rule(account_id, rule_id)
    return jsonify({"ok": True})


@app.route("/api/rules/run", methods=["POST"])
def api_rules_run():
    cl = _client()
    if not cl:
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(force=True, silent=True) or {}
    account_id = data.get("account_id")
    if not account_id:
        return jsonify({"error": "account_id required"}), 400
    rules = storage.load_rules(account_id)

    def audit(entry):
        storage.append_audit(account_id, entry)

    try:
        results = run_rules_for_account(cl, account_id, rules, audit=audit)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/scheduler/run", methods=["POST"])
def api_scheduler_run():
    """Manual trigger (same as cron)."""
    if _client() is None:
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
    return render_template(
        "settings.html",
        pixels=storage.list_pixels(),
        events=storage.list_events(),
        offers=storage.list_offers(),
    )


def _auth_required():
    return _effective_token() is not None


@app.route("/api/pixels", methods=["GET"])
def api_pixels_list():
    if not _auth_required():
        return jsonify({"error": "unauthorized"}), 401
    return jsonify({"pixels": storage.list_pixels()})


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
        }
    )
    return jsonify({"ok": True, "pixel": saved})


@app.route("/api/pixels/<item_id>", methods=["DELETE"])
def api_pixels_delete(item_id):
    if not _auth_required():
        return jsonify({"error": "unauthorized"}), 401
    ok = storage.delete_pixel(item_id)
    return jsonify({"ok": ok})


@app.route("/api/events", methods=["GET"])
def api_events_list():
    if not _auth_required():
        return jsonify({"error": "unauthorized"}), 401
    return jsonify({"events": storage.list_events()})


@app.route("/api/events", methods=["POST"])
def api_events_save():
    if not _auth_required():
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(force=True, silent=True) or {}
    name = (data.get("name") or "").strip()
    event_type = (data.get("event_type") or "").strip()
    if not name or not event_type:
        return jsonify({"error": "name and event_type required"}), 400
    saved = storage.upsert_event(
        {
            "id": data.get("id"),
            "name": name,
            "event_type": event_type,
            "is_custom": bool(data.get("is_custom")),
        }
    )
    return jsonify({"ok": True, "event": saved})


@app.route("/api/events/<item_id>", methods=["DELETE"])
def api_events_delete(item_id):
    if not _auth_required():
        return jsonify({"error": "unauthorized"}), 401
    ok = storage.delete_event(item_id)
    return jsonify({"ok": ok})


@app.route("/api/offers", methods=["GET"])
def api_offers_list():
    if not _auth_required():
        return jsonify({"error": "unauthorized"}), 401
    return jsonify({"offers": storage.list_offers()})


@app.route("/api/offers", methods=["POST"])
def api_offers_save():
    if not _auth_required():
        return jsonify({"error": "unauthorized"}), 401
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

    saved = storage.upsert_offer(
        {
            "id": data.get("id"),
            "name": name,
            "brand_name": (data.get("brand_name") or "").strip(),
            "cta": (data.get("cta") or "").strip(),
            "landing_url": (data.get("landing_url") or "").strip(),
            "headline": (data.get("headline") or "").strip(),
            "body": (data.get("body") or "").strip(),
            "pixel_id": (data.get("pixel_id") or "").strip(),
            "event_id": (data.get("event_id") or "").strip(),
            "payout": payout,
            "target_cpa": target_cpa,
            "utm_parameters": (data.get("utm_parameters") or "").strip(),
            "notes": (data.get("notes") or "").strip(),
        }
    )
    return jsonify({"ok": True, "offer": saved})


@app.route("/api/offers/<item_id>", methods=["DELETE"])
def api_offers_delete(item_id):
    if not _auth_required():
        return jsonify({"error": "unauthorized"}), 401
    ok = storage.delete_offer(item_id)
    return jsonify({"ok": ok})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5055)), debug=True)

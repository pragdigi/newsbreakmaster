"""
NewsBreak Ads Launcher — Flask application.
"""
from __future__ import annotations

import os
import secrets
import uuid
from datetime import date, timedelta

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


def _client() -> NewsBreakClient | None:
    tok = storage.load_token(_user_id())
    if not tok:
        return None
    return NewsBreakClient(tok["access_token"])


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
    if storage.load_token(_user_id()):
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
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
        tok = storage.load_token(_user_id())
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
    tok = storage.load_token(_user_id())
    accounts = _flatten_ad_accounts(cl.get_ad_accounts(tok.get("org_ids") or []))
    account_options = {str(a.get("id")): a.get("name", a.get("id")) for a in accounts if a.get("id")}

    if request.method == "GET":
        return render_template(
            "launch.html",
            accounts=account_options,
            campaigns=[],
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
            "objective": "CONVERSION",
        }

    bid_type = (request.form.get("bid_type") or "MAX_CONVERSION").strip().upper()
    needs_bid = bid_type in {"CPC", "CPM", "TARGET_CPA", "TARGET_ROAS"}

    pixel_id = (request.form.get("pixel_id") or "").strip()
    conversion_event = (request.form.get("conversion_event") or "PURCHASE").strip().upper()
    custom_event_name = (request.form.get("custom_event_name") or "").strip()

    ad_set_base: dict = {
        "name_prefix": request.form.get("ad_set_name", "Bulk ad set").strip() or "Bulk ad set",
        "adAccountId": ad_account_id,
        "budgetType": request.form.get("budget_type", "DAILY"),
        "budget": int(float(request.form.get("budget_dollars", "50") or 50) * 100),
        "bidType": bid_type,
        "startTime": request.form.get("start_time") or None,
        "endTime": request.form.get("end_time") or None,
    }
    if needs_bid:
        if bid_type == "TARGET_ROAS":
            ad_set_base["targetRoas"] = float(request.form.get("bid_dollars") or 2.0)
        else:
            ad_set_base["bidAmount"] = int(float(request.form.get("bid_dollars", "1") or 1) * 100)

    if pixel_id:
        ad_set_base["pixelId"] = pixel_id
        event_value = custom_event_name if (conversion_event == "CUSTOM" and custom_event_name) else conversion_event
        ad_set_base["conversionEvent"] = event_value
        ad_set_base["optimizationGoal"] = "CONVERSION"

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

    return render_template(
        "launch.html",
        accounts=account_options,
        campaigns=[],
        result=result,
    )


@app.route("/scaling")
def scaling():
    cl = _client()
    if not cl:
        return redirect(url_for("login"))
    tok = storage.load_token(_user_id())
    accounts = _flatten_ad_accounts(cl.get_ad_accounts(tok.get("org_ids") or []))
    account_options = {str(a.get("id")): a.get("name", a.get("id")) for a in accounts if a.get("id")}
    return render_template("scaling.html", accounts=account_options)


@app.route("/rules")
def rules_page():
    cl = _client()
    if not cl:
        return redirect(url_for("login"))
    tok = storage.load_token(_user_id())
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
    tok = storage.load_token(_user_id())
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5055)), debug=True)

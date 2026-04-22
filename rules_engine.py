"""
Rule templates and evaluation against normalised report rows.

Rules operate on canonical rows provided by a platform adapter
(``AdPlatformAdapter.fetch_report_rows``). Action execution is delegated back
to the adapter via ``adapter.update_status`` / ``adapter.update_budget``.

Each rule dict may carry a ``platform`` field ("newsbreak" | "smartnews"); if
absent, rules default to "newsbreak" for backwards compatibility.
"""
from __future__ import annotations

import copy
from datetime import date, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from newsbreak_api import NewsBreakClient, unwrap_list_response  # re-exported for legacy callers

RULE_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "kill_no_conversions": {
        "id": "kill_no_conversions",
        "name": "Kill zero-conversion spenders",
        "description": "Pause ads with spend above threshold and 0 conversions in the window",
        "category": "optimization",
        "supported_platforms": ["newsbreak", "smartnews"],
        "defaults": {
            "scope": "ad",
            "conditions": [
                {"metric": "spend", "op": ">", "value": 25.0, "window_days": 7},
                {"metric": "conversions", "op": "==", "value": 0, "window_days": 7, "logic": "AND"},
            ],
            "action": {"type": "pause"},
            "enabled": False,
            "dry_run": True,
        },
    },
    "cut_high_cpa": {
        "id": "cut_high_cpa",
        "name": "Cut high CPA",
        "description": "Pause ads when CPA exceeds max and enough spend",
        "category": "optimization",
        "supported_platforms": ["newsbreak", "smartnews"],
        "defaults": {
            "scope": "ad",
            "conditions": [
                {"metric": "cpa", "op": ">", "value": 50.0, "window_days": 3},
                {"metric": "spend", "op": ">", "value": 25.0, "window_days": 3, "logic": "AND"},
            ],
            "action": {"type": "pause"},
            "enabled": False,
            "dry_run": True,
        },
    },
    "scale_winners": {
        "id": "scale_winners",
        "name": "Scale low CPA",
        "description": "Increase ad set budget when CPA is below target and spend proves volume",
        "category": "scaling",
        "supported_platforms": ["newsbreak"],
        "defaults": {
            "scope": "ad_set",
            "conditions": [
                {"metric": "cpa", "op": "<", "value": 30.0, "window_days": 3},
                {"metric": "spend", "op": ">", "value": 50.0, "window_days": 3, "logic": "AND"},
            ],
            "action": {"type": "increase_budget", "value": 20},
            "enabled": False,
            "dry_run": True,
        },
    },
    "pause_losers": {
        "id": "pause_losers",
        "name": "Pause low ROAS",
        "description": "Pause when ROAS below 1 and minimum spend met",
        "category": "optimization",
        "supported_platforms": ["newsbreak", "smartnews"],
        "defaults": {
            "scope": "ad",
            "conditions": [
                {"metric": "roas", "op": "<", "value": 1.0, "window_days": 7},
                {"metric": "spend", "op": ">", "value": 30.0, "window_days": 7, "logic": "AND"},
            ],
            "action": {"type": "pause"},
            "enabled": False,
            "dry_run": True,
        },
    },

    # --- SmartNews-first templates ---
    "sn_kill_no_purchases": {
        "id": "sn_kill_no_purchases",
        "name": "Kill ads with no purchases (SmartNews)",
        "description": "Pause creatives spending ¥3000+ with 0 purchases in the window",
        "category": "optimization",
        "supported_platforms": ["smartnews"],
        "defaults": {
            "platform": "smartnews",
            "scope": "ad",
            "conditions": [
                {"metric": "spend", "op": ">", "value": 3000.0, "window_days": 3},
                {"metric": "purchase", "op": "==", "value": 0, "window_days": 3, "logic": "AND"},
            ],
            "action": {"type": "pause"},
            "enabled": False,
            "dry_run": True,
        },
    },
    "sn_scale_campaign": {
        "id": "sn_scale_campaign",
        "name": "Scale low-CPA campaign (SmartNews)",
        "description": "Bump campaign daily budget +20% when CPA is below target",
        "category": "scaling",
        "supported_platforms": ["smartnews"],
        "defaults": {
            "platform": "smartnews",
            "scope": "campaign",
            "conditions": [
                {"metric": "cpa", "op": "<", "value": 2000.0, "window_days": 3},
                {"metric": "spend", "op": ">", "value": 5000.0, "window_days": 3, "logic": "AND"},
            ],
            "action": {"type": "increase_budget", "value": 20},
            "enabled": False,
            "dry_run": True,
        },
    },
}


OPS: Dict[str, Callable[[float, float], bool]] = {
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: abs(a - b) < 1e-9,
    "!=": lambda a, b: abs(a - b) >= 1e-9,
}


def _metric_from_row(row: Dict[str, Any], metric: str) -> Optional[float]:
    m = (metric or "").lower()
    # events dict first (platform-normalized events)
    events = row.get("events") or {}
    if isinstance(events, dict) and m in events:
        try:
            return float(events[m])
        except (TypeError, ValueError):
            pass
    keys = {
        "spend": ("spend", "cost", "COST", "amountSpent"),
        "cpa": ("cpa", "CPA", "costPerConversion"),
        "roas": ("roas", "ROAS", "returnOnAdSpend"),
        "conversions": ("conversions", "conversion", "CONVERSION", "purchases"),
        "ctr": ("ctr", "CTR"),
        "impressions": ("impressions", "IMPRESSION", "impression"),
        "clicks": ("clicks", "CLICK", "click"),
        "value": ("value", "VALUE", "conversionValue", "revenue"),
        # SmartNews-style event names (camelCase passthrough too)
        "purchase": ("purchase", "purchases"),
        "add_to_cart": ("add_to_cart", "addToCart"),
        "initiate_checkout": ("initiate_checkout", "initiateCheckout"),
        "view_content": ("view_content", "viewContent"),
    }
    for k in keys.get(m, (metric,)):
        if k in row and row[k] is not None:
            try:
                return float(row[k])
            except (TypeError, ValueError):
                pass
    return None


def evaluate_conditions(row: Dict[str, Any], conditions: List[Dict[str, Any]]) -> bool:
    if not conditions:
        return False
    prev: Optional[bool] = None
    for i, cond in enumerate(conditions):
        metric = cond.get("metric", "")
        op = cond.get("op", ">")
        val = float(cond.get("value", 0))
        actual = _metric_from_row(row, metric)
        if actual is None:
            return False
        ok = OPS.get(op, OPS[">"])(actual, val)
        logic = cond.get("logic", "AND").upper()
        if i == 0:
            prev = ok
        else:
            if logic == "OR":
                prev = bool(prev or ok)
            else:
                prev = bool(prev and ok)
    return bool(prev)


# --- NewsBreak-specific report payload helper (still used by app.py directly) ---
def build_report_payload(
    ad_account_id: str,
    start: date,
    end: date,
    dimension: str,
    metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if metrics is None:
        metrics = ["COST", "IMPRESSION", "CLICK", "CTR", "CONVERSION", "CPA", "VALUE"]
    try:
        filter_id: Any = int(ad_account_id)
    except (TypeError, ValueError):
        filter_id = ad_account_id
    return {
        "name": f"report_{dimension.lower()}_{start.isoformat()}_{end.isoformat()}",
        "timezone": "UTC",
        "dateRange": "FIXED",
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
        "filter": "AD_ACCOUNT",
        "filterIds": [filter_id],
        "dimensions": [dimension],
        "metrics": metrics,
    }


def normalize_report_rows(raw: Any) -> List[Dict[str, Any]]:
    """Flatten integrated report response into numeric-friendly rows.

    NewsBreak-specific; kept here because app.py's dashboard/report endpoints
    still use the raw envelope pattern. SmartNews normalisation lives in the
    SmartNews adapter.
    """
    rows = unwrap_list_response(raw)
    if not rows and isinstance(raw, dict):
        data = raw.get("data")
        if isinstance(data, dict):
            inner = data.get("rows") or data.get("list") or data.get("records") or data.get("items") or data.get("reportData") or data.get("result")
            rows = unwrap_list_response(inner)
        if not rows:
            inner = raw.get("result") or raw.get("rows") or raw.get("reportData")
            rows = unwrap_list_response(inner)
    out: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        spend = r.get("cost") or r.get("COST") or r.get("spend")
        conv = r.get("conversion") or r.get("CONVERSION") or r.get("conversions")
        cpa = r.get("cpa") or r.get("CPA")
        roas = r.get("roas") or r.get("ROAS")
        value = (
            r.get("value")
            or r.get("VALUE")
            or r.get("conversionValue")
            or r.get("revenue")
        )

        def _f(x: Any) -> Optional[float]:
            if x is None:
                return None
            try:
                return float(x)
            except (TypeError, ValueError):
                return None

        def _money(raw_val: Any) -> Optional[float]:
            f = _f(raw_val)
            if f is None:
                return None
            if f > 1_000_000:
                return f / 1_000_000.0
            is_fractional = False
            if isinstance(raw_val, float) and not raw_val.is_integer():
                is_fractional = True
            elif isinstance(raw_val, str) and "." in raw_val:
                is_fractional = True
            if is_fractional:
                return f
            return f / 100.0

        spend_f = _money(spend)
        value_f = _money(value)
        cpa_f = _money(cpa)

        conv_f = _f(conv) or 0.0
        computed_roas = None
        if spend_f and spend_f > 0 and value_f is not None:
            computed_roas = value_f / spend_f
        out.append(
            {
                **r,
                "spend": spend_f,
                "conversions": conv_f,
                "conversionValue": value_f,
                "value": value_f,
                "cpa": cpa_f,
                "roas": _f(roas) if _f(roas) is not None else computed_roas,
                "ad_id": r.get("adId") or r.get("ad_id") or r.get("id"),
                "ad_set_id": r.get("adSetId") or r.get("ad_set_id"),
                "campaign_id": r.get("campaignId") or r.get("campaign_id"),
            }
        )
    return out


# --- Adapter-based rule execution ---
def apply_action(
    adapter: Any,
    rule: Dict[str, Any],
    row: Dict[str, Any],
) -> Tuple[str, Optional[str]]:
    """Apply a rule's action via the platform adapter.

    ``adapter`` is a platforms.AdPlatformAdapter instance. ``row`` must be a
    canonical row (see platforms.base module docstring) produced by the
    adapter's ``fetch_report_rows``.
    """
    action = rule.get("action") or {}
    atype = action.get("type", "pause")
    scope = rule.get("scope", "ad")
    account_id = rule.get("account_id")

    # Resolve canonical entity id for this scope.
    def _row_id(scope_name: str) -> Optional[str]:
        if scope_name == "ad":
            return row.get("ad_id") or row.get("id")
        if scope_name == "ad_set":
            return row.get("ad_set_id") or row.get("id")
        if scope_name == "campaign":
            return row.get("campaign_id") or row.get("id")
        return row.get("id")

    entity_id = _row_id(scope)
    if not entity_id:
        return "skip", f"no id for scope={scope}"

    if atype == "pause":
        try:
            adapter.update_status(scope, str(entity_id), enabled=False, account_id=account_id)
            return "ok", f"paused {scope} {entity_id}"
        except Exception as e:
            return "err", str(e)

    if atype == "enable":
        try:
            adapter.update_status(scope, str(entity_id), enabled=True, account_id=account_id)
            return "ok", f"enabled {scope} {entity_id}"
        except Exception as e:
            return "err", str(e)

    if atype in ("increase_budget", "decrease_budget"):
        pct = float(action.get("value", 10))
        if scope not in ("ad_set", "campaign"):
            return "skip", "budget change requires ad_set or campaign scope"
        # NewsBreak ad-set budget is in cents on the row; SmartNews campaign budget likewise.
        current = row.get("budget") or row.get("dailyBudget")
        try:
            current_f = float(current) if current is not None else 0.0
        except (TypeError, ValueError):
            current_f = 0.0
        if current_f <= 0:
            return "skip", "could not read current budget from row"
        factor = 1 + pct / 100.0 if atype == "increase_budget" else 1 - pct / 100.0
        new_cents = max(0, int(current_f * factor))
        try:
            adapter.update_budget(
                scope,
                str(entity_id),
                budget_cents=new_cents,
                budget_type=row.get("budget_type") or "DAILY",
                account_id=account_id,
            )
            return "ok", f"budget {scope} {entity_id} -> {new_cents}"
        except Exception as e:
            return "err", str(e)

    return "skip", f"unknown action {atype}"


def _window_from_conditions(conditions: List[Dict[str, Any]]) -> int:
    vals: List[int] = []
    for c in conditions or []:
        try:
            vals.append(int(c.get("window_days") or 7))
        except (TypeError, ValueError):
            continue
    return max(vals) if vals else 7


def run_rules_for_account(
    adapter_or_client: Any,
    ad_account_id: str,
    rules: List[Dict[str, Any]],
    *,
    audit: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """Evaluate enabled rules against ``ad_account_id``.

    ``adapter_or_client`` may be a platform adapter (new path) or a legacy
    ``NewsBreakClient`` (old path). Legacy clients are auto-wrapped in a
    NewsBreakAdapter so any existing callers keep working.
    """
    adapter = _coerce_adapter(adapter_or_client)
    results: List[Dict[str, Any]] = []
    for rule in rules:
        if not rule.get("enabled"):
            continue
        scope = rule.get("scope", "ad")
        # SmartNews collapses ad_set scope to campaign.
        if scope == "ad_set" and not adapter.supports_ad_set_scope:
            scope = "campaign"
        rule_platform = rule.get("platform") or adapter.platform
        if rule_platform != adapter.platform:
            continue
        window = _window_from_conditions(rule.get("conditions", []))
        end = date.today()
        start = end - timedelta(days=max(1, window))
        try:
            rows = adapter.fetch_report_rows(ad_account_id, scope, start, end)
        except Exception as e:
            if audit:
                audit({"rule_id": rule.get("id"), "error": str(e), "scope": scope})
            continue
        for row in rows:
            if not evaluate_conditions(row, rule.get("conditions", [])):
                continue
            entry = {
                "rule_id": rule.get("id"),
                "rule_name": rule.get("name"),
                "scope": scope,
                "platform": adapter.platform,
                "row": row,
                "dry_run": rule.get("dry_run", True),
            }
            if rule.get("dry_run"):
                entry["result"] = "dry_run"
                results.append(entry)
                if audit:
                    audit({**entry, "action": "would_run"})
                continue
            try:
                rule_with_acct = {**rule, "scope": scope, "account_id": ad_account_id}
                status, msg = apply_action(adapter, rule_with_acct, row)
                entry["result"] = status
                entry["message"] = msg
            except Exception as e:
                entry["result"] = "error"
                entry["message"] = str(e)
            results.append(entry)
            if audit:
                audit(entry)
    return results


def _coerce_adapter(obj: Any) -> Any:
    if hasattr(obj, "fetch_report_rows") and hasattr(obj, "update_status"):
        return obj
    if isinstance(obj, NewsBreakClient):
        from platforms.newsbreak import NewsBreakAdapter
        return NewsBreakAdapter(obj, [])
    raise TypeError(f"Cannot use {type(obj).__name__} as platform adapter")


def instantiate_template(
    template_id: str,
    account_id: str,
    overrides: Optional[Dict[str, Any]] = None,
    *,
    platform: Optional[str] = None,
) -> Dict[str, Any]:
    tpl = RULE_TEMPLATES.get(template_id)
    if not tpl:
        raise ValueError(f"Unknown template {template_id}")
    rule = copy.deepcopy(tpl["defaults"])
    rule["id"] = template_id
    rule["name"] = tpl["name"]
    rule["account_id"] = account_id
    if platform:
        rule["platform"] = platform
    rule.setdefault("platform", "newsbreak")
    if overrides:
        rule.update(overrides)
    return rule


def fetch_report_for_rules(
    client: NewsBreakClient,
    ad_account_id: str,
    scope: str,
    window_days: int,
) -> List[Dict[str, Any]]:
    """Legacy helper — kept so standalone NewsBreak scripts / tests keep working.

    Prefer ``adapter.fetch_report_rows`` for new code.
    """
    end = date.today()
    start = end - timedelta(days=max(1, window_days))
    dim = "AD"
    if scope == "ad_set":
        dim = "AD_SET"
    elif scope == "campaign":
        dim = "CAMPAIGN"
    payload = build_report_payload(ad_account_id, start, end, dim)
    raw = client.get_integrated_report(payload)
    return normalize_report_rows(raw)

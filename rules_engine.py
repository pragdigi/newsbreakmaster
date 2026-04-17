"""
Rule templates and evaluation against report rows.
"""
from __future__ import annotations

import copy
from datetime import date, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from newsbreak_api import NewsBreakClient, unwrap_list_response

RULE_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "kill_no_conversions": {
        "id": "kill_no_conversions",
        "name": "Kill zero-conversion spenders",
        "description": "Pause ads with spend above threshold and 0 conversions in the window",
        "category": "optimization",
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
    m = metric.lower()
    keys = {
        "spend": ("spend", "cost", "COST", "amountSpent"),
        "cpa": ("cpa", "CPA", "costPerConversion"),
        "roas": ("roas", "ROAS", "returnOnAdSpend"),
        "conversions": ("conversions", "conversion", "CONVERSION", "purchases"),
        "ctr": ("ctr", "CTR"),
        "impressions": ("impressions", "IMPRESSION"),
        "clicks": ("clicks", "CLICK"),
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


def build_report_payload(
    ad_account_id: str,
    start: date,
    end: date,
    dimension: str,
    metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    dimension: AD | AD_SET | CAMPAIGN (NewsBreak enums — uppercase)
    NewsBreak `getIntegratedReport` shape:
      { name, timezone, dateRange: FIXED, startDate, endDate,
        filter: "AD_ACCOUNT", filterIds: [<int>],
        dimensions: [...], metrics: [...] }
    """
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
    """Flatten integrated report response into numeric-friendly rows."""
    rows = unwrap_list_response(raw)
    if not rows and isinstance(raw, dict):
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

        spend_f = _f(spend)
        # Some APIs return cost in micro-units — heuristic: if huge, divide
        if spend_f is not None and spend_f > 1_000_000:
            spend_f = spend_f / 1_000_000.0

        value_f = _f(value)
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
                "cpa": _f(cpa),
                "roas": _f(roas) if _f(roas) is not None else computed_roas,
                "ad_id": r.get("adId") or r.get("ad_id") or r.get("id"),
                "ad_set_id": r.get("adSetId") or r.get("ad_set_id"),
                "campaign_id": r.get("campaignId") or r.get("campaign_id"),
            }
        )
    return out


def fetch_report_for_rules(
    client: NewsBreakClient,
    ad_account_id: str,
    scope: str,
    window_days: int,
) -> List[Dict[str, Any]]:
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


def apply_action(
    client: NewsBreakClient,
    rule: Dict[str, Any],
    row: Dict[str, Any],
) -> Tuple[str, Optional[str]]:
    """Returns (status, message)."""
    action = rule.get("action") or {}
    atype = action.get("type", "pause")
    scope = rule.get("scope", "ad")

    if atype == "pause":
        if scope == "ad":
            aid = row.get("ad_id") or row.get("adId")
            if not aid:
                return "skip", "no ad id"
            client.update_ad_status(str(aid), "OFF")
            return "ok", f"paused ad {aid}"
        if scope == "ad_set":
            asid = row.get("ad_set_id") or row.get("adSetId")
            if not asid:
                return "skip", "no ad set id"
            client.update_ad_set_status(str(asid), "OFF")
            return "ok", f"paused ad set {asid}"
        cid = row.get("campaign_id") or row.get("campaignId")
        if not cid:
            return "skip", "no campaign id"
        # If API supports campaign pause — try update_campaign
        try:
            client.update_campaign(str(cid), {"status": "OFF"})
            return "ok", f"paused campaign {cid}"
        except Exception as e:
            return "err", str(e)

    if atype in ("increase_budget", "decrease_budget"):
        pct = float(action.get("value", 10))
        if scope != "ad_set":
            return "skip", "budget change only for ad_set scope in v1"
        asid = row.get("ad_set_id") or row.get("adSetId")
        if not asid:
            return "skip", "no ad set id"
        # Need current budget — may be on row or fetch ad set
        budget_cents = row.get("budget") or row.get("dailyBudget")
        try:
            current = float(budget_cents)
        except (TypeError, ValueError):
            current = 0.0
        if current <= 0:
            return "skip", "could not read budget from report row"
        factor = 1 + pct / 100.0 if atype == "increase_budget" else 1 - pct / 100.0
        new_cents = max(0, int(current * factor))
        client.update_ad_set(str(asid), {"budget": new_cents, "budgetType": row.get("budgetType") or "DAILY"})
        return "ok", f"budget {asid} -> {new_cents}"

    return "skip", f"unknown action {atype}"


def run_rules_for_account(
    client: NewsBreakClient,
    ad_account_id: str,
    rules: List[Dict[str, Any]],
    *,
    audit: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """Evaluate enabled rules; optionally append audit entries."""
    results: List[Dict[str, Any]] = []
    for rule in rules:
        if not rule.get("enabled"):
            continue
        scope = rule.get("scope", "ad")
        window = max(c.get("window_days", 7) for c in rule.get("conditions", []) or [{}])
        rows = fetch_report_for_rules(client, ad_account_id, scope, window)
        for row in rows:
            if not evaluate_conditions(row, rule.get("conditions", [])):
                continue
            entry = {
                "rule_id": rule.get("id"),
                "rule_name": rule.get("name"),
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
                status, msg = apply_action(client, rule, row)
                entry["result"] = status
                entry["message"] = msg
            except Exception as e:
                entry["result"] = "error"
                entry["message"] = str(e)
            results.append(entry)
            if audit:
                audit(entry)
    return results


def instantiate_template(template_id: str, account_id: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    tpl = RULE_TEMPLATES.get(template_id)
    if not tpl:
        raise ValueError(f"Unknown template {template_id}")
    rule = copy.deepcopy(tpl["defaults"])
    rule["id"] = template_id
    rule["name"] = tpl["name"]
    rule["account_id"] = account_id
    if overrides:
        rule.update(overrides)
    return rule

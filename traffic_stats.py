"""Read-only spend / purchase / revenue pull for the Slack stats blender.

Reuses the NewsBreak, MediaGo, and SmartNews clients in this project.
Credentials come from the environment (this folder's .env locally).
Stdout is JSON only. This module never prints secrets and never writes ads.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

NATIVE_SOURCES = ("newsbreak", "mediago", "smartnews")

_SECRET_RE = re.compile(
    r"(?i)(access[_ ]?token|client[_ ]?secret|api[_ ]?key|api[_ ]?token|authorization|bearer|basic)\s*[:=]\s*\S+"
)


def _redact(msg: Any) -> str:
    text = _SECRET_RE.sub(lambda m: m.group(1) + "=(redacted)", str(msg or ""))
    return text[:500]


def _money(n: Any) -> float:
    return round(float(n or 0), 2)


def _cpa(spend: float, purchases: float) -> Optional[float]:
    if not purchases:
        return None
    return round(spend / purchases, 2)


def _roas(revenue: Optional[float], spend: float) -> Optional[float]:
    if revenue is None or not spend:
        return None
    return round(revenue / spend, 2)


def _csv_ids(name: str) -> List[str]:
    raw = os.environ.get(name, "") or ""
    return [part.strip() for part in raw.split(",") if part.strip()]


def _env_set(name: str) -> bool:
    return bool((os.environ.get(name) or "").strip())


def missing_env(source: str) -> List[str]:
    """Env var names that must be set before this source can be pulled."""
    source = (source or "").strip().lower()
    if source == "newsbreak":
        missing = []
        if not _env_set("NEWSBREAK_ACCESS_TOKEN"):
            missing.append("NEWSBREAK_ACCESS_TOKEN")
        if not _csv_ids("NEWSBREAK_DEFAULT_ORG_IDS"):
            missing.append("NEWSBREAK_DEFAULT_ORG_IDS")
        return missing
    if source == "mediago":
        return [] if _env_set("MEDIAGO_API_TOKEN") else ["MEDIAGO_API_TOKEN"]
    if source == "smartnews":
        missing = []
        if not _env_set("SMARTNEWS_CLIENT_ID"):
            missing.append("SMARTNEWS_CLIENT_ID")
        if not (_env_set("SMARTNEWS_CLIENT_SECRET") or _env_set("SMARTNEWS_API_KEY")):
            missing.append("SMARTNEWS_CLIENT_SECRET")
        return missing
    return [f"unknown source {source}"]


def load_local_env() -> None:
    """Fill gaps from newsbreakmaster/.env. Existing process env wins."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path(__file__).resolve().parent / ".env")


def _account_id(acc: Dict[str, Any]) -> str:
    for key in ("id", "adAccountId", "ad_account_id", "account_id", "accountId"):
        if acc.get(key) not in (None, ""):
            return str(acc.get(key))
    return ""


def _account_name(acc: Dict[str, Any], aid: str) -> str:
    for key in ("name", "account_name", "adAccountName", "ad_account_name"):
        if acc.get(key):
            return str(acc.get(key))
    return aid


def _row_account_id(row: Dict[str, Any]) -> str:
    for key in (
        "adAccountId",
        "ad_account_id",
        "accountId",
        "account_id",
        "AD_ACCOUNT",
        "id",
    ):
        if row.get(key) not in (None, ""):
            return str(row.get(key))
    return ""


def _num(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _purchases_from_row(row: Dict[str, Any], *, prefer_purchase: bool) -> tuple[float, str, bool]:
    """Return (count, metric, metric_was_explicit).

    MediaGo always returns ``cv_purchase``, usually 0. A zero column is not a
    purchase: ``select_mediago_purchase`` keeps a non-zero purchase-like
    ``cv_*`` and otherwise the ``conversion`` total.
    """
    events = row.get("events") if isinstance(row.get("events"), dict) else {}
    if prefer_purchase:
        from platforms.mediago import select_mediago_purchase

        count, metric = select_mediago_purchase(row)
        if metric != "conversion":
            return count, metric, True
        if "purchase" in events:
            return _num(events.get("purchase")) or 0.0, "purchase", True
        return count, "conversion", False
    if "purchase" in events:
        return _num(events.get("purchase")) or 0.0, "purchase", True
    return _num(row.get("conversions")) or 0.0, "conversion", False


def _revenue_from_row(row: Dict[str, Any]) -> tuple[Optional[float], bool]:
    raw = row.get("raw") if isinstance(row.get("raw"), dict) else {}
    for src in (row, raw):
        for key in ("value", "conversionValue", "conversion_value", "revenue", "VALUE", "cv_purchase_value"):
            if key in src and src.get(key) is not None:
                return _num(src.get(key)) or 0.0, True
        for key, val in src.items():
            if isinstance(key, str) and key.lower().endswith("_value") and val not in (None, ""):
                n = _num(val)
                if n:
                    return n, True
    roas = _num(row.get("roas") if row.get("roas") is not None else raw.get("roas"))
    spend = _num(row.get("spend"))
    # A reported ROAS of 0 is not a revenue figure. MediaGo sends roas=0 when
    # the report has no conversion value.
    if roas and spend:
        return roas * spend, True
    return None, False


def _mediago_purchase_label(metric: str) -> str:
    """Slack label, e.g. ``(mediago conversion)`` or ``(mediago cv_purchase)``."""
    name = metric or "conversion"
    return f"(mediago {name})"


def rollup_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    prefer_purchase: bool,
    revenue_supported: bool,
) -> Dict[str, Any]:
    spend = 0.0
    purchases = 0.0
    revenue = 0.0
    saw_revenue = False
    saw_explicit_purchase = False
    metric = "purchase" if prefer_purchase else "conversion"
    for row in rows:
        spend += _num(row.get("spend")) or 0.0
        count, row_metric, explicit = _purchases_from_row(row, prefer_purchase=prefer_purchase)
        purchases += count
        if explicit:
            saw_explicit_purchase = True
            metric = row_metric
        elif not saw_explicit_purchase:
            metric = row_metric
        if revenue_supported:
            rev, ok = _revenue_from_row(row)
            if ok and rev is not None:
                saw_revenue = True
                revenue += rev
    spend_m = _money(spend)
    purch = int(round(purchases))
    rev_m = _money(revenue) if saw_revenue else None
    return {
        "spend": spend_m,
        "purchases": purch,
        "purchases_available": True,
        "purchases_metric": metric if saw_explicit_purchase or not prefer_purchase else metric,
        "revenue": rev_m,
        "revenue_available": bool(revenue_supported and saw_revenue),
        "cpa": _cpa(spend_m, purch),
        "roas": _roas(rev_m, spend_m),
    }


def _base_result(source: str, start: date, end: date) -> Dict[str, Any]:
    return {
        "source": source,
        "ok": False,
        "skipped": False,
        "since": start.isoformat(),
        "until": end.isoformat(),
        "currency": "USD",
        "spend": 0.0,
        "purchases": None,
        "purchases_available": False,
        "purchases_metric": None,
        "revenue": None,
        "revenue_available": False,
        "cpa": None,
        "roas": None,
        "accounts": [],
        "missing_env": [],
        "gaps": [],
        "error": None,
    }


def _skipped(source: str, start: date, end: date, missing: List[str]) -> Dict[str, Any]:
    row = _base_result(source, start, end)
    row["skipped"] = True
    row["missing_env"] = missing
    row["error"] = "missing credentials: " + ", ".join(missing)
    return row


def _failed(source: str, start: date, end: date, exc: BaseException) -> Dict[str, Any]:
    row = _base_result(source, start, end)
    row["error"] = _redact(exc)
    return row


def _apply_totals(result: Dict[str, Any], totals: Dict[str, Any], accounts: List[Dict[str, Any]]) -> Dict[str, Any]:
    result.update(
        {
            "ok": True,
            "spend": totals["spend"],
            "purchases": totals["purchases"],
            "purchases_available": totals["purchases_available"],
            "purchases_metric": totals["purchases_metric"],
            "revenue": totals["revenue"],
            "revenue_available": totals["revenue_available"],
            "cpa": totals["cpa"],
            "roas": totals["roas"],
            "accounts": accounts,
        }
    )
    return result


def fetch_newsbreak(start: date, end: date) -> Dict[str, Any]:
    missing = missing_env("newsbreak")
    if missing:
        return _skipped("newsbreak", start, end, missing)
    try:
        from platforms import get_adapter
        from rules_engine import build_report_payload, normalize_report_rows

        org_ids = _csv_ids("NEWSBREAK_DEFAULT_ORG_IDS")
        adapter = get_adapter(
            "newsbreak",
            access_token=os.environ.get("NEWSBREAK_ACCESS_TOKEN", "").strip(),
            org_ids=org_ids,
        )
        raw_accounts = adapter.get_accounts() or []
        accounts = []
        ids: List[Any] = []
        names: Dict[str, str] = {}
        for acc in raw_accounts:
            if not isinstance(acc, dict):
                continue
            aid = _account_id(acc)
            if not aid:
                continue
            names[aid] = _account_name(acc, aid)
            try:
                ids.append(int(aid))
            except (TypeError, ValueError):
                ids.append(aid)
        result = _base_result("newsbreak", start, end)
        result["gaps"] = [
            "Purchases are NewsBreak CONVERSION (all conversion events, not purchase-only).",
            "Revenue is NewsBreak VALUE.",
            "Report timezone is UTC, matching the existing integrated-report client.",
        ]
        result["timezone"] = "UTC"
        if not ids:
            result["ok"] = True
            result["purchases"] = 0
            result["purchases_available"] = True
            result["purchases_metric"] = "CONVERSION"
            result["revenue"] = 0.0
            result["revenue_available"] = True
            result["gaps"].append("No NewsBreak ad accounts were returned for the configured org ids.")
            return result

        payload = build_report_payload("0", start, end, "AD_ACCOUNT")
        payload["filterIds"] = ids
        payload["dimensions"] = ["AD_ACCOUNT"]
        payload["metrics"] = ["COST", "IMPRESSION", "CLICK", "CONVERSION", "VALUE"]
        raw = adapter.client.get_integrated_report(payload)
        rows = normalize_report_rows(raw)
        by_account: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            aid = _row_account_id(row)
            by_account.setdefault(aid or "unknown", []).append(row)
        for aid, name in names.items():
            bucket = rollup_rows(by_account.get(aid, []), prefer_purchase=False, revenue_supported=True)
            accounts.append({"id": aid, "name": name, **{k: bucket[k] for k in ("spend", "purchases", "revenue", "cpa", "roas")}})
        # Rows that didn't match a discovered account still count in the total.
        totals = rollup_rows(rows, prefer_purchase=False, revenue_supported=True)
        totals["purchases_metric"] = "CONVERSION"
        if not totals["revenue_available"]:
            result["gaps"].append("NewsBreak returned no VALUE for this window.")
        return _apply_totals(result, totals, accounts)
    except Exception as exc:  # noqa: BLE001 — surface a redacted error, do not crash the blend
        return _failed("newsbreak", start, end, exc)


def fetch_mediago(start: date, end: date) -> Dict[str, Any]:
    missing = missing_env("mediago")
    if missing:
        return _skipped("mediago", start, end, missing)
    try:
        from platforms import get_adapter

        adapter = get_adapter(
            "mediago",
            api_token=os.environ.get("MEDIAGO_API_TOKEN", "").strip(),
            auth_level=(os.environ.get("MEDIAGO_AUTH_LEVEL") or "auto").strip() or "auto",
            account_ids=_csv_ids("MEDIAGO_DEFAULT_ACCOUNT_IDS"),
        )
        raw_accounts = adapter.get_accounts() or []
        result = _base_result("mediago", start, end)
        result["gaps"] = [
            "Purchases use a non-zero purchase-like cv_* column (cv_purchase first). "
            "MediaGo returns every cv_* column as 0 when unused, so a zero cv_purchase "
            "falls through to conversion, the total the dashboards already use.",
            "Revenue is included only when a row has a value field or a positive ROAS. MediaGo daily reports often omit revenue.",
            "Dates use the MediaGo report timezone parameter est (US Eastern), on the same calendar dates.",
        ]
        result["timezone"] = "est"
        rows: List[Dict[str, Any]] = []
        accounts = []
        errors = []
        for acc in raw_accounts:
            if not isinstance(acc, dict):
                continue
            aid = _account_id(acc)
            if not aid:
                continue
            try:
                account_rows = adapter.fetch_report_rows(aid, "campaign", start, end) or []
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{_account_name(acc, aid)}: {_redact(exc)}")
                continue
            rows.extend(account_rows)
            bucket = rollup_rows(account_rows, prefer_purchase=True, revenue_supported=True)
            accounts.append(
                {
                    "id": aid,
                    "name": _account_name(acc, aid),
                    "spend": bucket["spend"],
                    "purchases": bucket["purchases"],
                    "revenue": bucket["revenue"],
                    "cpa": bucket["cpa"],
                    "roas": bucket["roas"],
                }
            )
        if not raw_accounts and not errors:
            result["ok"] = True
            result["gaps"].append("MediaGo returned no ad accounts.")
            return result
        if not accounts and errors:
            result["error"] = "; ".join(errors)
            return result
        totals = rollup_rows(rows, prefer_purchase=True, revenue_supported=True)
        chosen = totals.get("purchases_metric") or "conversion"
        if chosen == "conversion":
            result["gaps"].append(
                "cv_purchase was zero or absent; purchases are MediaGo conversion."
            )
        totals["purchases_metric"] = _mediago_purchase_label(chosen)
        if not totals["revenue_available"]:
            result["gaps"].append("No MediaGo revenue/ROAS on these rows, so revenue is omitted from the blend.")
        if errors:
            result["gaps"].append("Some MediaGo accounts failed: " + "; ".join(errors))
        return _apply_totals(result, totals, accounts)
    except Exception as exc:  # noqa: BLE001
        return _failed("mediago", start, end, exc)


def fetch_smartnews(start: date, end: date) -> Dict[str, Any]:
    missing = missing_env("smartnews")
    if missing:
        return _skipped("smartnews", start, end, missing)
    try:
        from platforms import get_adapter

        adapter = get_adapter(
            "smartnews",
            client_id=os.environ.get("SMARTNEWS_CLIENT_ID", "").strip(),
            client_secret=(
                os.environ.get("SMARTNEWS_CLIENT_SECRET")
                or os.environ.get("SMARTNEWS_API_KEY")
                or ""
            ).strip(),
            account_ids=_csv_ids("SMARTNEWS_DEFAULT_ACCOUNT_IDS"),
        )
        raw_accounts = adapter.get_accounts() or []
        result = _base_result("smartnews", start, end)
        result["gaps"] = [
            "Purchases are metrics_count_purchase.",
            "SmartNews insights used here have spend and purchase count, not purchase value or ROAS.",
        ]
        result["timezone"] = "UTC"
        currencies = set()
        rows: List[Dict[str, Any]] = []
        accounts = []
        errors = []
        for acc in raw_accounts:
            if not isinstance(acc, dict):
                continue
            aid = _account_id(acc)
            if not aid:
                continue
            cur = str(acc.get("currency") or getattr(adapter, "currency", "USD") or "USD").upper()
            currencies.add(cur)
            try:
                account_rows = adapter.fetch_report_rows(aid, "campaign", start, end) or []
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{_account_name(acc, aid)}: {_redact(exc)}")
                continue
            rows.extend(account_rows)
            bucket = rollup_rows(account_rows, prefer_purchase=True, revenue_supported=False)
            accounts.append(
                {
                    "id": aid,
                    "name": _account_name(acc, aid),
                    "currency": cur,
                    "spend": bucket["spend"],
                    "purchases": bucket["purchases"],
                    "revenue": None,
                    "cpa": bucket["cpa"],
                    "roas": None,
                }
            )
        if len(currencies) == 1:
            result["currency"] = next(iter(currencies))
        elif len(currencies) > 1:
            result["currency"] = "MIXED"
            result["gaps"].append("SmartNews accounts are not a single currency: " + ", ".join(sorted(currencies)) + ".")
        if not raw_accounts and not errors:
            result["ok"] = True
            result["purchases"] = 0
            result["purchases_available"] = True
            result["purchases_metric"] = "count_purchase"
            result["gaps"].append("SmartNews returned no ad accounts.")
            return result
        if not accounts and errors:
            result["error"] = "; ".join(errors)
            return result
        totals = rollup_rows(rows, prefer_purchase=True, revenue_supported=False)
        totals["purchases_metric"] = "count_purchase"
        totals["revenue"] = None
        totals["revenue_available"] = False
        totals["roas"] = None
        if errors:
            result["gaps"].append("Some SmartNews accounts failed: " + "; ".join(errors))
        return _apply_totals(result, totals, accounts)
    except Exception as exc:  # noqa: BLE001
        return _failed("smartnews", start, end, exc)


FETCHERS = {
    "newsbreak": fetch_newsbreak,
    "mediago": fetch_mediago,
    "smartnews": fetch_smartnews,
}


def fetch_sources(sources: Iterable[str], start: date, end: date) -> Dict[str, Any]:
    rows = []
    for source in sources:
        key = (source or "").strip().lower()
        fn = FETCHERS.get(key)
        if not fn:
            rows.append(
                {
                    **_base_result(key or "unknown", start, end),
                    "error": f"unsupported source {key}",
                }
            )
            continue
        rows.append(fn(start, end))
    return {
        "since": start.isoformat(),
        "until": end.isoformat(),
        "sources": rows,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only native traffic stats (JSON on stdout).")
    parser.add_argument("--source", default="all", help="newsbreak, mediago, smartnews, or all")
    parser.add_argument("--since", required=True, help="YYYY-MM-DD")
    parser.add_argument("--until", required=True, help="YYYY-MM-DD")
    args = parser.parse_args(argv)
    load_local_env()
    start = date.fromisoformat(args.since)
    end = date.fromisoformat(args.until)
    if end < start:
        raise SystemExit("--until is before --since")
    requested = NATIVE_SOURCES if args.source.strip().lower() == "all" else [part.strip().lower() for part in args.source.split(",") if part.strip()]
    payload = fetch_sources(requested, start, end)
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(_redact(exc) + "\n")
        raise SystemExit(1)

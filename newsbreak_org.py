"""NewsBreak organization + ad-account list helpers."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from newsbreak_api import NewsBreakClient


def _truthy_env(name: str, default: str = "1") -> bool:
    raw = (os.environ.get(name, default) or default).strip().lower()
    return raw not in ("0", "false", "no", "off")


def parse_csv_ids(raw: str) -> List[str]:
    return [x.strip() for x in (raw or "").split(",") if x.strip()]


def get_admin_org_ids(client: NewsBreakClient) -> List[str]:
    """Org IDs visible to this API token (GET /org/admin-orgs)."""
    body = client._request("GET", "/org/admin-orgs")
    data = body.get("data") if isinstance(body, dict) else None
    rows = []
    if isinstance(data, dict):
        rows = data.get("list") or data.get("groups") or []
    elif isinstance(data, list):
        rows = data
    out: List[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        oid = str(row.get("id") or row.get("organizationId") or "").strip()
        if oid and oid not in seen:
            seen.add(oid)
            out.append(oid)
    return out


def resolve_newsbreak_org_ids(
    access_token: str,
    explicit_org_ids: Optional[List[str]] = None,
) -> List[str]:
    """
    Merge NEWSBREAK_DEFAULT_ORG_IDS with /org/admin-orgs when auto-discover is on
    (default on). Ensures newly created ad accounts under any visible org appear
    in newsbreakmaster account dropdowns.
    """
    explicit = list(explicit_org_ids or [])
    if not access_token:
        return explicit
    if not _truthy_env("NEWSBREAK_AUTO_DISCOVER_ORGS", "1"):
        return explicit
    try:
        client = NewsBreakClient(access_token)
        discovered = get_admin_org_ids(client)
    except Exception:
        return explicit
    merged: List[str] = []
    seen: set[str] = set()
    for oid in explicit + discovered:
        if oid and oid not in seen:
            seen.add(oid)
            merged.append(oid)
    return merged or explicit


def sort_newsbreak_accounts(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pin Tinnito / Cardian (and numbered variants) to the top of account lists."""

    def sort_key(row: Dict[str, Any]) -> tuple:
        name = (row.get("name") or "").lower()
        if "tinnito" in name:
            return (0, name)
        if "cardian" in name:
            return (1, name)
        return (2, name)

    return sorted(rows, key=sort_key)

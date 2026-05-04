"""HMAC-signed client for the creative_brief_tool ad-library scraper.

Mirrors the signing protocol in ``newsbreakmaster.agent_api`` so the
existing pattern is reused exactly:

    canonical = METHOD + "\\n" + PATH + "\\n" + TIMESTAMP + "\\n"
              + sha256(body).hexdigest()
    signature = HMAC-SHA256(SCRAPER_HMAC_SECRET, canonical).hexdigest()

Headers:
    X-Scraper-Key       (matches SCRAPER_PUBLIC_KEY, default "default")
    X-Scraper-Timestamp (unix seconds)
    X-Scraper-Signature (hex)

Configuration (env):
    SCRAPER_BASE_URL          (e.g. https://ai-creatives.onrender.com)
    SCRAPER_HMAC_SECRET       (must match server)
    SCRAPER_PUBLIC_KEY        (default: "default")
    SCRAPER_TIMEOUT           (seconds, default 90)

When ``SCRAPER_BASE_URL`` is unset every call returns ``None`` so callers
can degrade to the direct-fetch path. ``ScraperUnavailable`` is raised
only on configured-but-unreachable failures, never for "not configured".
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import requests

logger = logging.getLogger(__name__)


SCRAPER_BASE_URL = (os.environ.get("SCRAPER_BASE_URL") or "").strip().rstrip("/")
SCRAPER_HMAC_SECRET = (os.environ.get("SCRAPER_HMAC_SECRET") or "").strip()
SCRAPER_PUBLIC_KEY = (os.environ.get("SCRAPER_PUBLIC_KEY") or "default").strip()
SCRAPER_TIMEOUT = int(os.environ.get("SCRAPER_TIMEOUT", "90"))


class ScraperUnavailable(Exception):
    """Raised when the scraper service is configured but the call fails."""


def is_configured() -> bool:
    """True iff both SCRAPER_BASE_URL and SCRAPER_HMAC_SECRET are set."""
    return bool(SCRAPER_BASE_URL and SCRAPER_HMAC_SECRET)


def _sign(method: str, path: str, body: bytes) -> Dict[str, str]:
    ts = str(int(time.time()))
    digest = hashlib.sha256(body or b"").hexdigest()
    msg = f"{method.upper()}\n{path}\n{ts}\n{digest}".encode("utf-8")
    sig = hmac.new(SCRAPER_HMAC_SECRET.encode("utf-8"), msg, hashlib.sha256).hexdigest()
    return {
        "X-Scraper-Key": SCRAPER_PUBLIC_KEY,
        "X-Scraper-Timestamp": ts,
        "X-Scraper-Signature": sig,
        "Content-Type": "application/json",
    }


def _post(path: str, payload: Dict[str, Any], *, timeout: int) -> Dict[str, Any]:
    if not is_configured():
        raise ScraperUnavailable("scraper not configured (SCRAPER_BASE_URL/SCRAPER_HMAC_SECRET)")
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = _sign("POST", path, body)
    url = SCRAPER_BASE_URL + path
    try:
        resp = requests.post(url, data=body, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise ScraperUnavailable(f"network error: {exc}") from exc
    if resp.status_code == 401:
        raise ScraperUnavailable(
            f"scraper rejected our signature ({resp.status_code}): {resp.text[:200]}"
        )
    if resp.status_code >= 500:
        raise ScraperUnavailable(
            f"scraper {resp.status_code}: {resp.text[:300]}"
        )
    if resp.status_code >= 400:
        raise ScraperUnavailable(
            f"scraper {resp.status_code}: {resp.text[:300]}"
        )
    try:
        return resp.json() or {}
    except json.JSONDecodeError as exc:
        raise ScraperUnavailable(f"non-JSON response: {resp.text[:200]}") from exc


def fetch_meta(
    queries: Sequence[str],
    *,
    country: str = "US",
    limit_per_query: int = 25,
    timeout: int = SCRAPER_TIMEOUT,
) -> Optional[List[Dict[str, Any]]]:
    """Fetch Meta Ad Library cards via the scraper service.

    Returns None if the scraper is not configured (caller should fall
    back to the direct-fetch path). Returns a list (possibly empty) on
    success. Raises ``ScraperUnavailable`` on configured-but-failing.
    """
    if not is_configured():
        return None
    queries = [str(q).strip() for q in queries if str(q).strip()]
    if not queries:
        return []
    body = {
        "queries": list(queries),
        "country": country,
        "limit_per_query": int(limit_per_query),
    }
    payload = _post("/api/scraper/meta-ad-library", body, timeout=timeout)
    cards = payload.get("cards") or []
    per_query = payload.get("per_query") or {}
    logger.info(
        "scraper_client.meta queries=%d -> %d cards (per_query=%s)",
        len(queries),
        len(cards),
        per_query,
    )
    return cards


def fetch_tiktok(
    queries: Sequence[str],
    *,
    region: str = "US",
    period: int = 30,
    limit_per_query: int = 25,
    timeout: int = SCRAPER_TIMEOUT,
) -> Optional[List[Dict[str, Any]]]:
    """Fetch TikTok Creative Center cards via the scraper service."""
    if not is_configured():
        return None
    queries = [str(q).strip() for q in queries if str(q).strip()]
    if not queries:
        return []
    body = {
        "queries": list(queries),
        "region": region,
        "period": int(period),
        "limit_per_query": int(limit_per_query),
    }
    payload = _post("/api/scraper/tiktok-creative-center", body, timeout=timeout)
    cards = payload.get("cards") or []
    per_query = payload.get("per_query") or {}
    logger.info(
        "scraper_client.tiktok queries=%d -> %d cards (per_query=%s)",
        len(queries),
        len(cards),
        per_query,
    )
    return cards


__all__ = ["fetch_meta", "fetch_tiktok", "is_configured", "ScraperUnavailable"]

"""TikTok Creative Center "Top Ads" scraper — no auth required.

The Creative Center exposes a public JSON API at
``ads.tiktok.com/business/creativecenter/inspiration/topads/`` that
the official inspiration page consumes. We hit the same endpoint with
a browser-like User-Agent. No login or developer token needed.

Returns the same normalized shape as ``meta_ad_library.fetch`` so the
``discover_public`` orchestrator can treat both sources uniformly.

Rate-limit notes
----------------
TikTok caps anonymous access fairly generously (a few hundred queries
per IP per day), but we still use a single session with jitter between
calls to stay below the radar. If we hit a 403 we degrade silently.
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Sequence

import requests

logger = logging.getLogger(__name__)


DEFAULT_REGION = os.environ.get("TIKTOK_CREATIVE_REGION", "US").upper()
DEFAULT_TIMEOUT = int(os.environ.get("TIKTOK_CREATIVE_TIMEOUT", "30"))
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# Top-ads listing endpoint (publicly consumable). Keyword search uses
# the search endpoint instead. Both return the same item shape.
_TOP_ADS_URL = "https://ads.tiktok.com/creative_radar_api/v1/top_ads/v2/list"
_SEARCH_URL = "https://ads.tiktok.com/creative_radar_api/v1/top_ads/v2/list"


_SESSION: Optional[requests.Session] = None


def _session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        s.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
                "Accept-Language": "en-US,en;q=0.9",
                "Origin": "https://ads.tiktok.com",
                "Referer": (
                    "https://ads.tiktok.com/business/creativecenter/inspiration/"
                    "topads/pc/en"
                ),
            }
        )
        _SESSION = s
    return _SESSION


def _normalize_card(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    advertiser = str(item.get("brand_name") or item.get("advertiser_name") or "").strip()
    headline = str(item.get("title") or item.get("ad_title") or "").strip()
    body = str(item.get("ad_desc") or item.get("description") or "").strip()
    landing_url = str(item.get("landing_url") or item.get("url") or "").strip() or None

    image_urls: List[str] = []
    cover = item.get("cover_url") or item.get("cover_image")
    if isinstance(cover, str) and cover.startswith("http"):
        image_urls.append(cover)
    images = item.get("image_list") or []
    if isinstance(images, list):
        for img in images:
            if isinstance(img, str) and img.startswith("http"):
                image_urls.append(img)
            elif isinstance(img, dict):
                u = img.get("url") or img.get("image_url")
                if isinstance(u, str) and u.startswith("http"):
                    image_urls.append(u)

    video_url = None
    v = item.get("video_url") or item.get("video_info") or {}
    if isinstance(v, dict):
        video_url = v.get("video_url") or v.get("play_addr")
    elif isinstance(v, str):
        video_url = v

    started = item.get("first_show_date") or item.get("start_time")
    if isinstance(started, (int, float)):
        try:
            from datetime import datetime, timezone

            started = datetime.fromtimestamp(int(started), tz=timezone.utc).isoformat()
        except Exception:  # noqa: BLE001
            started = ""

    record_id = str(item.get("id") or item.get("ad_id") or item.get("item_id") or "").strip()
    if not record_id:
        record_id = f"tt:{advertiser}:{abs(hash(headline + body)) % (10**12)}"
    else:
        record_id = f"tt:{record_id}"

    if not advertiser and not headline and not body:
        return None

    return {
        "id": record_id,
        "source": "tiktok",
        "advertiser": advertiser,
        "headline": headline,
        "body": body,
        "image_urls": image_urls[:4],
        "video_url": video_url if isinstance(video_url, str) else None,
        "landing_url": landing_url,
        "started_at": started or None,
        "raw": {
            "id": item.get("id"),
            "industry": item.get("industry"),
            "objective": item.get("objective"),
            "ctr": item.get("ctr"),
            "like": item.get("like"),
        },
    }


def _fetch_page(
    *,
    keyword: str,
    region: str,
    page: int,
    limit: int,
    timeout: int,
) -> List[Dict[str, Any]]:
    params = {
        "period": 30,
        "page": page,
        "limit": limit,
        "region_code": region,
        "industry": "",
        "objective": "",
        "like": "",
        "country_code": region,
    }
    if keyword:
        params["keyword"] = keyword
    try:
        resp = _session().get(_SEARCH_URL, params=params, timeout=timeout)
    except requests.RequestException as exc:
        logger.warning("tiktok_creative network error %r: %s", keyword, exc)
        return []
    if resp.status_code != 200:
        logger.warning(
            "tiktok_creative %s for keyword=%r region=%s: %s",
            resp.status_code,
            keyword,
            region,
            resp.text[:200],
        )
        return []
    try:
        payload = resp.json() or {}
    except json.JSONDecodeError:
        logger.warning("tiktok_creative non-JSON for %r", keyword)
        return []
    data = payload.get("data") or {}
    materials = data.get("materials") or data.get("items") or []
    if not isinstance(materials, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in materials:
        norm = _normalize_card(item)
        if norm:
            out.append(norm)
    return out


def fetch(
    query: str,
    *,
    limit: int = 30,
    region: str = DEFAULT_REGION,
    timeout: int = DEFAULT_TIMEOUT,
) -> List[Dict[str, Any]]:
    """Fetch up to ``limit`` top-performing TikTok ads for ``query``.

    Empty ``query`` returns the global top-ads list (no keyword filter).
    Returns ``[]`` on any failure.
    """
    query = (query or "").strip()
    page_size = min(limit, 50)
    rows: List[Dict[str, Any]] = []
    page = 1
    while len(rows) < limit and page <= 4:
        chunk = _fetch_page(
            keyword=query,
            region=region,
            page=page,
            limit=page_size,
            timeout=timeout,
        )
        if not chunk:
            break
        rows.extend(chunk)
        page += 1
    if not rows:
        logger.info(
            "tiktok_creative: 0 cards for query=%r region=%s",
            query,
            region,
        )
    # De-dupe preserving order.
    seen: set = set()
    deduped: List[Dict[str, Any]] = []
    for r in rows:
        rid = r.get("id")
        if not rid or rid in seen:
            continue
        seen.add(rid)
        deduped.append(r)
        if len(deduped) >= limit:
            break
    return deduped


def fetch_many(
    queries: Sequence[str],
    *,
    limit_per_query: int = 25,
    region: str = DEFAULT_REGION,
    timeout: int = DEFAULT_TIMEOUT,
    sleep_jitter: float = 1.0,
) -> List[Dict[str, Any]]:
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for q in queries:
        rows = fetch(q, limit=limit_per_query, region=region, timeout=timeout)
        for r in rows:
            rid = r.get("id")
            if not rid or rid in seen:
                continue
            seen.add(rid)
            out.append(r)
        if sleep_jitter > 0 and len(queries) > 1:
            time.sleep(random.uniform(0.3, sleep_jitter))
    return out


__all__ = ["fetch", "fetch_many"]

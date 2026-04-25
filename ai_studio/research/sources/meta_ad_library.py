"""Facebook / Meta Ad Library scraper — no API token required.

Hits the public ``facebook.com/ads/library`` page with a browser-like
User-Agent and parses the inline JSON blob Facebook embeds in the
initial HTML. This is the same JSON the page's own React client
consumes, so we get headlines, body copy, advertiser names, and image
URLs without an API key.

Caveats
-------
1. **Fragile**: Facebook reshapes this HTML constantly. We keep the
   parser defensive — multiple regex/JSON paths, graceful empty-list
   fallback. Expect to revisit this every few months.
2. **Rate limited**: aggressive scraping triggers a checkpoint. We
   sleep a small amount between queries and use a single session with
   browser-like cookies.
3. **Country-locked**: the public library is per-country. Default
   ``country="US"`` matches NewsBreak / SmartNews ad surface.
4. **No auth**: when FB changes their page so the JSON is no longer
   inlined, this returns ``[]`` and the orchestrator logs a warning —
   not an exception. The optional ``FB_AD_LIBRARY_TOKEN`` Graph API
   path is still present as an opt-in fallback for when a token is
   configured.

We intentionally keep the dependency surface to ``requests`` only — no
playwright / selenium — so this works on the existing Render plan
without extra build steps.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Sequence

import requests

logger = logging.getLogger(__name__)


META_TOKEN = (os.environ.get("FB_AD_LIBRARY_TOKEN") or "").strip()
META_GRAPH_VERSION = os.environ.get("FB_AD_LIBRARY_GRAPH_VERSION", "v20.0")
DEFAULT_COUNTRY = os.environ.get("META_AD_LIB_COUNTRY", "US").upper()
DEFAULT_TIMEOUT = int(os.environ.get("META_AD_LIB_TIMEOUT", "30"))
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# Reusable session keeps cookies / connection-pool warm.
_SESSION: Optional[requests.Session] = None


def _session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        s.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;q=0.9,"
                    "image/avif,image/webp,*/*;q=0.8"
                ),
                "Sec-Ch-Ua": '"Chromium";v="124", "Not.A/Brand";v="24"',
                "Sec-Ch-Ua-Mobile": "?0",
                "Sec-Ch-Ua-Platform": '"Windows"',
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Upgrade-Insecure-Requests": "1",
            }
        )
        _SESSION = s
    return _SESSION


# ----------------------------------------------------------------------
# HTML scrape path (no token)
# ----------------------------------------------------------------------


# These two regexes find embedded JSON blobs on the Ad Library page.
# FB encodes the initial Relay store under one of a few keys; we try
# them in order. The third regex is a generic catch-all for the older
# adNode shape.
_RE_JSON_BLOBS = [
    re.compile(r'\{"data":\s*\{[^<]*?"ad_archive_search_v2"[^<]*?\}\s*\}'),
    re.compile(r'\{"require":\[\["ScheduledServerJS"[^<]*?\}\s*\]\]\}'),
    re.compile(
        r'\{"node":\s*\{"snapshot":\s*\{[^<]*?"link_url"[^<]*?\}[^<]*?\}\s*\}'
    ),
]


def _find_json_blobs(html: str) -> List[str]:
    out: List[str] = []
    for rx in _RE_JSON_BLOBS:
        out.extend(rx.findall(html))
    return out


def _safe_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        return None


def _walk_for_ad_cards(node: Any, out: List[Dict[str, Any]]) -> None:
    """Walk a parsed JSON tree and accumulate ad-card snapshots.

    Facebook uses a ``snapshot`` shape with ``link_url``, ``page_name``,
    ``cards`` / ``body`` / ``creation_time`` / image-url fields. We
    look for any dict that has both ``page_name`` and (``link_url`` or
    ``cards``) and treat it as a card.
    """
    if isinstance(node, dict):
        keys = node.keys()
        if "page_name" in keys and ("link_url" in keys or "cards" in keys or "body" in keys):
            out.append(node)
        for v in node.values():
            _walk_for_ad_cards(v, out)
    elif isinstance(node, list):
        for item in node:
            _walk_for_ad_cards(item, out)


def _normalize_html_card(card: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    advertiser = str(card.get("page_name") or "").strip()
    if not advertiser:
        return None

    # Body / headline can live in several places.
    body = ""
    body_obj = card.get("body") or {}
    if isinstance(body_obj, dict):
        body = str(body_obj.get("text") or body_obj.get("markup") or "").strip()
    elif isinstance(body_obj, str):
        body = body_obj.strip()

    headline = ""
    title_obj = card.get("title")
    if isinstance(title_obj, dict):
        headline = str(title_obj.get("text") or "").strip()
    elif isinstance(title_obj, str):
        headline = title_obj.strip()
    if not headline:
        # Carousels stash the headline on the first card.
        cards = card.get("cards") or []
        if isinstance(cards, list) and cards:
            first = cards[0] if isinstance(cards[0], dict) else {}
            headline = str(first.get("title") or "").strip()
            if not body:
                body = str(first.get("body") or "").strip()

    image_urls: List[str] = []
    for k in ("original_image_url", "resized_image_url", "image_url"):
        v = card.get(k)
        if isinstance(v, str) and v.startswith("http"):
            image_urls.append(v)
    images = card.get("images") or []
    if isinstance(images, list):
        for img in images:
            if isinstance(img, dict):
                u = img.get("original_image_url") or img.get("resized_image_url") or img.get("url")
                if isinstance(u, str) and u.startswith("http"):
                    image_urls.append(u)
    cards_inner = card.get("cards") or []
    if isinstance(cards_inner, list):
        for c in cards_inner:
            if not isinstance(c, dict):
                continue
            for k in ("original_image_url", "resized_image_url", "image_url"):
                v = c.get(k)
                if isinstance(v, str) and v.startswith("http"):
                    image_urls.append(v)

    # De-dupe preserving order.
    seen: set = set()
    image_urls = [u for u in image_urls if not (u in seen or seen.add(u))]

    landing_url = ""
    for k in ("link_url", "caption", "display_format_url"):
        v = card.get(k)
        if isinstance(v, str) and v.startswith("http"):
            landing_url = v
            break

    started = ""
    ts = card.get("creation_time") or card.get("start_date")
    if isinstance(ts, (int, float)):
        try:
            from datetime import datetime, timezone

            started = datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
        except Exception:  # noqa: BLE001
            started = ""

    record_id = str(card.get("id") or card.get("ad_archive_id") or card.get("snapshot_id") or "").strip()
    if not record_id:
        record_id = f"meta:{advertiser}:{abs(hash(headline + body)) % (10**12)}"
    else:
        record_id = f"meta:{record_id}"

    return {
        "id": record_id,
        "source": "meta",
        "advertiser": advertiser,
        "headline": headline,
        "body": body,
        "image_urls": image_urls[:4],
        "video_url": None,
        "landing_url": landing_url or None,
        "started_at": started or None,
        "raw": {
            "page_name": advertiser,
            "ad_archive_id": card.get("ad_archive_id"),
            "id": card.get("id"),
        },
    }


def _fetch_html(query: str, *, country: str, timeout: int) -> str:
    url = "https://www.facebook.com/ads/library/"
    params = {
        "active_status": "all",
        "ad_type": "all",
        "country": country,
        "q": query,
        "search_type": "keyword_unordered",
        "media_type": "all",
    }
    try:
        resp = _session().get(url, params=params, timeout=timeout)
    except requests.RequestException as exc:
        logger.warning("meta_ad_library: request error for %r: %s", query, exc)
        return ""
    if resp.status_code != 200:
        logger.warning(
            "meta_ad_library: %s for query=%r (len=%s)",
            resp.status_code,
            query,
            len(resp.text or ""),
        )
        return ""
    return resp.text or ""


def _scrape_html(query: str, *, country: str, limit: int, timeout: int) -> List[Dict[str, Any]]:
    html = _fetch_html(query, country=country, timeout=timeout)
    if not html:
        return []
    blobs = _find_json_blobs(html)
    cards: List[Dict[str, Any]] = []
    for blob in blobs:
        parsed = _safe_json(blob)
        if parsed is None:
            continue
        _walk_for_ad_cards(parsed, cards)
        if len(cards) >= limit * 4:
            break

    if not cards:
        # Last-ditch: scan every JSON-looking object that mentions
        # page_name anywhere. We anchor on the literal substring so the
        # regex stays bounded, then walk backwards to the opening brace
        # and forwards to the matching closing brace using a small
        # depth-counting loop. Slower than the targeted regexes above
        # but resilient to FB shuffling key order.
        for needle in re.finditer(r'"page_name"\s*:', html):
            start = html.rfind("{", 0, needle.start())
            if start < 0:
                continue
            depth = 0
            end = start
            for j in range(start, min(start + 20000, len(html))):
                ch = html[j]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = j + 1
                        break
            if end <= start:
                continue
            parsed = _safe_json(html[start:end])
            if parsed:
                cards.append(parsed)
            if len(cards) >= limit * 4:
                break

    out: List[Dict[str, Any]] = []
    seen_ids: set = set()
    for c in cards:
        norm = _normalize_html_card(c)
        if not norm:
            continue
        if norm["id"] in seen_ids:
            continue
        seen_ids.add(norm["id"])
        out.append(norm)
        if len(out) >= limit:
            break
    return out


# ----------------------------------------------------------------------
# Optional Graph API path (used only when FB_AD_LIBRARY_TOKEN is set)
# ----------------------------------------------------------------------


def _scrape_graph_api(
    query: str,
    *,
    country: str,
    limit: int,
    timeout: int,
) -> List[Dict[str, Any]]:
    if not META_TOKEN:
        return []
    url = f"https://graph.facebook.com/{META_GRAPH_VERSION}/ads_archive"
    params = {
        "ad_reached_countries": f"['{country}']",
        "search_terms": query,
        "ad_type": "ALL",
        "ad_active_status": "ALL",
        "limit": min(limit, 50),
        "fields": (
            "id,page_name,ad_creative_bodies,ad_creative_link_titles,"
            "ad_creative_link_descriptions,ad_snapshot_url,ad_delivery_start_time,"
            "publisher_platforms,languages"
        ),
        "access_token": META_TOKEN,
    }
    try:
        resp = _session().get(url, params=params, timeout=timeout)
    except requests.RequestException as exc:
        logger.warning("meta_ad_library graph error %r: %s", query, exc)
        return []
    if resp.status_code != 200:
        logger.warning(
            "meta_ad_library graph %s for %r: %s",
            resp.status_code,
            query,
            resp.text[:200],
        )
        return []
    payload = resp.json() or {}
    rows = payload.get("data") or []
    out: List[Dict[str, Any]] = []
    for row in rows:
        bodies = row.get("ad_creative_bodies") or []
        titles = row.get("ad_creative_link_titles") or []
        descs = row.get("ad_creative_link_descriptions") or []
        out.append(
            {
                "id": f"meta:{row.get('id')}",
                "source": "meta",
                "advertiser": row.get("page_name") or "",
                "headline": (titles[0] if titles else "") or "",
                "body": (bodies[0] if bodies else "") or "",
                "image_urls": [],  # graph endpoint doesn't return media without extra perms
                "video_url": None,
                "landing_url": row.get("ad_snapshot_url"),
                "started_at": row.get("ad_delivery_start_time"),
                "raw": {
                    "id": row.get("id"),
                    "publisher_platforms": row.get("publisher_platforms"),
                    "descriptions": descs[:2],
                },
            }
        )
        if len(out) >= limit:
            break
    return out


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def fetch(
    query: str,
    *,
    limit: int = 30,
    country: str = DEFAULT_COUNTRY,
    timeout: int = DEFAULT_TIMEOUT,
    prefer_graph: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Fetch up to ``limit`` ads matching ``query`` from Meta Ad Library.

    Returns ``[]`` on any failure (network, parse, captcha checkpoint).
    Set ``prefer_graph=True`` to force the Graph API path when a token
    is configured. By default we try the HTML scrape first (richer
    creative metadata) and fall back to Graph if it returns nothing.
    """
    query = (query or "").strip()
    if not query:
        return []

    if prefer_graph is None:
        prefer_graph = bool(META_TOKEN)

    rows: List[Dict[str, Any]] = []
    if prefer_graph and META_TOKEN:
        rows = _scrape_graph_api(query, country=country, limit=limit, timeout=timeout)

    if not rows:
        rows = _scrape_html(query, country=country, limit=limit, timeout=timeout)

    if not rows and not prefer_graph and META_TOKEN:
        rows = _scrape_graph_api(query, country=country, limit=limit, timeout=timeout)

    if not rows:
        logger.info(
            "meta_ad_library: 0 cards for query=%r country=%s (token=%s)",
            query,
            country,
            "yes" if META_TOKEN else "no",
        )
    return rows


def fetch_many(
    queries: Sequence[str],
    *,
    limit_per_query: int = 25,
    country: str = DEFAULT_COUNTRY,
    timeout: int = DEFAULT_TIMEOUT,
    sleep_jitter: float = 1.5,
) -> List[Dict[str, Any]]:
    """Fan out across ``queries`` with a small jitter between requests.

    De-duplicates by ``id`` and trims to ``limit_per_query * len(queries)``
    so a single noisy query can't dominate the result set.
    """
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for q in queries:
        rows = fetch(q, limit=limit_per_query, country=country, timeout=timeout)
        for r in rows:
            rid = r.get("id")
            if not rid or rid in seen:
                continue
            seen.add(rid)
            out.append(r)
        if sleep_jitter > 0 and len(queries) > 1:
            time.sleep(random.uniform(0.4, sleep_jitter))
    return out


__all__ = ["fetch", "fetch_many"]

"""Style discovery — four modes that emit :class:`StyleCandidate` entries.

1. ``discover_from_winners``      — cluster own winners that don't match the catalog.
2. ``discover_from_gethookd``     — pull high-performing competitor ads from
                                    GetHookd (Meta + TikTok + Google supply).
3. ``discover_from_brainstorm``   — pure LLM ideation for new styles.
4. ``discover_from_uploads``      — extract style templates from user-uploaded
                                    reference screenshots (vision-LLM).

All four return ``[StyleCandidate, ...]`` dicts and persist them via
:func:`storage.upsert_style_candidate`. Every run is logged to
``research_runs.jsonl``.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

import storage

from .. import prompt_gen

logger = logging.getLogger(__name__)

GETHOOKD_BASE_URL = os.environ.get("GETHOOKD_BASE_URL", "https://app.gethookd.ai/api/v1")
GETHOOKD_API_KEY = os.environ.get("GETHOOKD_API_KEY", "")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_GENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

GEMINI_TEXT_MODEL = os.environ.get("AD_STUDIO_GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_VISION_MODEL = os.environ.get("AD_STUDIO_GEMINI_VISION_MODEL", "gemini-3.1-pro-preview")
CLAUDE_MODEL = os.environ.get("AD_STUDIO_CLAUDE_MODEL", "claude-opus-4-7")

CATALOG_IDS = [s.id for s in prompt_gen.STYLE_CATALOG]
CLUSTER_MIN_SIZE = int(os.environ.get("AD_STUDIO_CLUSTER_MIN_SIZE", "3"))

# High-performance filter for GetHookd — keep only strong signals.
HIGH_PERFORMANCE_TIERS = {"high", "very high", "top", "trending"}


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------

def _slugify(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return text[:48] or f"style_{uuid.uuid4().hex[:8]}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_json(text: str) -> Any:
    if not text:
        return None
    m = re.search(r"\{.*\}|\[.*\]", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None


def _base_candidate(
    *,
    style_id: str,
    name: str,
    description: str,
    visual_cues: Sequence[str],
    prompt_template: str,
    source: str,
    source_meta: Optional[Dict[str, Any]] = None,
    reference_image_paths: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    now = _now_iso()
    return {
        "style_id": _slugify(style_id),
        "name": name,
        "description": description,
        "visual_cues": list(visual_cues or []),
        "prompt_template": prompt_template,
        "reference_image_paths": list(reference_image_paths or []),
        "source": source,
        "source_meta": source_meta or {},
        "status": "candidate",
        "trials": 0,
        "wins": 0,
        "impressions": 0,
        "spend": 0.0,
        "conversions": 0,
        "cpa": None,
        "ctr": None,
        "thompson_alpha": 1,
        "thompson_beta": 1,
        "created_at": now,
        "last_trial_at": None,
    }


def _call_gemini_text(prompt: str, *, timeout: int = 90) -> str:
    if not GEMINI_API_KEY:
        return ""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_TEXT_MODEL}:generateContent"
    )
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    resp = requests.post(url, params={"key": GEMINI_API_KEY}, json=body, timeout=timeout)
    if resp.status_code >= 400:
        logger.warning("gemini text call failed: %s %s", resp.status_code, resp.text[:400])
        return ""
    data = resp.json()
    try:
        return (data["candidates"][0]["content"]["parts"][0].get("text")) or ""
    except Exception:  # noqa: BLE001
        return ""


def _call_claude_text(prompt: str, *, timeout: int = 90) -> str:
    if not ANTHROPIC_API_KEY:
        return ""
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": CLAUDE_MODEL,
            "max_tokens": 3000,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=timeout,
    )
    if resp.status_code >= 400:
        logger.warning("claude call failed: %s %s", resp.status_code, resp.text[:400])
        return ""
    try:
        data = resp.json()
        blocks = data.get("content") or []
        for b in blocks:
            if b.get("type") == "text":
                return b.get("text") or ""
    except Exception:  # noqa: BLE001
        pass
    return ""


def _call_gemini_vision(prompt: str, *, image_paths: Sequence[str], timeout: int = 120) -> str:
    if not GEMINI_API_KEY or not image_paths:
        return ""
    parts: List[Dict[str, Any]] = [{"text": prompt}]
    for p in image_paths:
        try:
            with open(p, "rb") as f:
                data_b64 = base64.b64encode(f.read()).decode("ascii")
        except Exception as exc:  # noqa: BLE001
            logger.warning("cannot read %s: %s", p, exc)
            continue
        mime = "image/png"
        if p.lower().endswith((".jpg", ".jpeg")):
            mime = "image/jpeg"
        elif p.lower().endswith(".webp"):
            mime = "image/webp"
        parts.append({"inlineData": {"mimeType": mime, "data": data_b64}})
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_VISION_MODEL}:generateContent"
    )
    body = {"contents": [{"role": "user", "parts": parts}]}
    resp = requests.post(url, params={"key": GEMINI_API_KEY}, json=body, timeout=timeout)
    if resp.status_code >= 400:
        logger.warning("gemini vision failed: %s %s", resp.status_code, resp.text[:400])
        return ""
    data = resp.json()
    try:
        return (data["candidates"][0]["content"]["parts"][0].get("text")) or ""
    except Exception:  # noqa: BLE001
        return ""


def _log_run(
    *,
    mode: str,
    platform: str,
    offer_id: Optional[str],
    inputs: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    raw: Optional[str] = None,
) -> str:
    run = storage.append_research_run(
        {
            "run_id": str(uuid.uuid4()),
            "mode": mode,
            "offer_id": offer_id,
            "inputs": inputs,
            "candidates_emitted": [c.get("style_id") for c in candidates],
            "raw_llm_output": (raw or "")[:4000] if raw else None,
        },
        platform=platform,
    )
    return run.get("run_id", "")


# ----------------------------------------------------------------------
# 1. Cluster-own-winners
# ----------------------------------------------------------------------

_CLUSTERING_PROMPT = """You are clustering winning direct-response ad creatives into reusable visual style templates.

Catalog styles (already known, do NOT re-emit these):
{catalog_ids}

I will give you N winning ads below, each with an index, headline, description, and image URL. Your job:
  1. For each ad, decide whether it fits one of the catalog styles. If yes, tag it with that id; if no, tag it "NEW".
  2. For all "NEW" ads, cluster them into groups of visually similar ads (minimum group size: {min_size}).
  3. For each cluster, invent a short style name, a 1-sentence description, 3-6 visual cues, and a reusable prompt template with {{headline}} and {{cta_label}} placeholders (the template must end with "Square format.").

Return STRICT JSON with this exact shape and nothing else:

{{
  "classifications": [ {{ "index": int, "style_id": str }} ],
  "clusters": [
    {{ "name": str, "description": str, "visual_cues": [str],
       "prompt_template": str, "ad_indices": [int] }}
  ]
}}

Winning ads:
{winners_block}
"""


def _format_winners_for_cluster(winners: Sequence[Dict[str, Any]]) -> str:
    rows = []
    for i, w in enumerate(winners):
        rows.append(
            f"[{i}] headline={w.get('headline','')!r} "
            f"description={w.get('description','')!r} "
            f"sponsored={w.get('sponsored_name','')!r} "
            f"image={w.get('image_url','')!r}"
        )
    return "\n".join(rows) or "(none)"


def discover_from_winners(
    platform: str,
    *,
    model: str = "gemini-3.1-pro",
    min_size: int = CLUSTER_MIN_SIZE,
) -> List[Dict[str, Any]]:
    """Classify own winners vs the catalog; cluster the rest into candidates.

    Reads from ALL platforms so a winner discovered on SmartNews can seed a
    style candidate usable on NewsBreak and vice-versa.
    """
    winners = storage.list_all_winners()
    if not winners:
        _log_run(
            mode="cluster_winners",
            platform=platform,
            offer_id=None,
            inputs={"model": model, "min_size": min_size, "winners_count": 0},
            candidates=[],
        )
        return []

    prompt = _CLUSTERING_PROMPT.format(
        catalog_ids=", ".join(CATALOG_IDS),
        min_size=min_size,
        winners_block=_format_winners_for_cluster(winners),
    )
    raw = _call_gemini_text(prompt) or _call_claude_text(prompt)
    parsed = _extract_json(raw) or {}

    emitted: List[Dict[str, Any]] = []
    clusters = (parsed.get("clusters") or []) if isinstance(parsed, dict) else []
    for cluster in clusters:
        indices = cluster.get("ad_indices") or []
        if not isinstance(indices, list) or len(indices) < min_size:
            continue
        name = str(cluster.get("name") or "New style").strip()
        cand = _base_candidate(
            style_id=name,
            name=name,
            description=str(cluster.get("description") or "").strip(),
            visual_cues=[str(v) for v in (cluster.get("visual_cues") or [])],
            prompt_template=str(cluster.get("prompt_template") or "").strip(),
            source="cluster",
            source_meta={"cluster_size": len(indices), "ad_indices": indices},
        )
        saved = storage.upsert_style_candidate(cand, platform=platform)
        emitted.append(saved)

    _log_run(
        mode="cluster_winners",
        platform=platform,
        offer_id=None,
        inputs={"model": model, "min_size": min_size, "winners_count": len(winners)},
        candidates=emitted,
        raw=raw,
    )
    return emitted


# ----------------------------------------------------------------------
# 2. GetHookd
# ----------------------------------------------------------------------


def normalize_gethookd_ad(ad: Dict[str, Any]) -> Dict[str, Any]:
    """Port of the JS ``normalizeGethookdAd`` helper into Python."""
    brand = ad.get("brand") or {}
    media = ad.get("media") or []
    if not isinstance(media, list):
        media = []
    return {
        "id": ad.get("id"),
        "external_id": str(ad.get("external_id") or ""),
        "title": str(ad.get("title") or ""),
        "body": str(ad.get("body") or ""),
        "platform": str(ad.get("platform") or ""),
        "display_format": str(ad.get("display_format") or ""),
        "cta_type": str(ad.get("cta_type") or ""),
        "cta_text": str(ad.get("cta_text") or ""),
        "landing_page": str(ad.get("landing_page") or ""),
        "days_active": int(ad.get("days_active") or 0),
        "performance_score": ad.get("performance_score"),
        "performance_score_title": str(ad.get("performance_score_title") or ""),
        "share_url": str(ad.get("share_url") or ""),
        "start_date": str(ad.get("start_date") or ""),
        "end_date": str(ad.get("end_date") or ""),
        "active_in_library": int(ad.get("active_in_library") or 0),
        "used_count": int(ad.get("used_count") or 0),
        "media": [
            {
                "url": str(m.get("url") or ""),
                "thumbnail_url": str(m.get("thumbnail_url") or ""),
                "type": str(m.get("type") or ""),
            }
            for m in media
            if isinstance(m, dict)
        ],
        "brand": {
            "external_id": str(brand.get("external_id") or ""),
            "name": str(brand.get("name") or "Unknown Brand"),
            "logo_url": str(brand.get("logo_url") or ""),
            "active_ads": int(brand.get("active_ads") or 0),
        },
    }


# GetHookd renamed their search parameter at some point; the production API
# now rejects ``q`` with "Unrecognized parameter(s): q". We don't have access
# to current docs, so we probe the most likely names and remember the first
# one the server accepts. Order goes from most likely to least.
_GETHOOKD_QUERY_PARAM_CANDIDATES: Tuple[str, ...] = (
    "query",
    "search",
    "keyword",
    "keywords",
    "term",
    "q",
)
_gethookd_query_param: Optional[str] = os.environ.get("GETHOOKD_QUERY_PARAM") or None


def _gethookd_query_key_to_try() -> List[str]:
    if _gethookd_query_param:
        return [_gethookd_query_param]
    return list(_GETHOOKD_QUERY_PARAM_CANDIDATES)


def _do_gethookd_request(params: Dict[str, Any], *, timeout: int) -> "requests.Response":
    return requests.get(
        f"{GETHOOKD_BASE_URL}/explore",
        params={k: v for k, v in params.items() if v not in (None, "")},
        headers={
            "Authorization": f"Bearer {GETHOOKD_API_KEY}",
            "Content-Type": "application/json",
        },
        timeout=timeout,
    )


def _fetch_gethookd(params: Dict[str, Any], *, timeout: int = 60) -> List[Dict[str, Any]]:
    """Fetch /explore, automatically discovering the correct query-param name.

    If the request includes a ``__query`` virtual key, we try each candidate
    parameter name (``query``, ``search``, etc.) until the API stops
    returning a 422 "Unrecognized parameter(s)" error. The discovered name
    is cached on the module so subsequent calls hit the right name on the
    first try, and can be pinned via ``GETHOOKD_QUERY_PARAM`` env var.
    """
    global _gethookd_query_param

    if not GETHOOKD_API_KEY:
        raise RuntimeError("GETHOOKD_API_KEY not configured")

    query_value = params.pop("__query", None)
    candidates = _gethookd_query_key_to_try() if query_value else [None]

    last_err: Optional[str] = None
    for key in candidates:
        attempt = dict(params)
        if key and query_value:
            attempt[key] = query_value
        resp = _do_gethookd_request(attempt, timeout=timeout)
        if resp.status_code == 422 and key:
            body = resp.text[:400]
            if "Unrecognized parameter" in body and key in body:
                last_err = f"422 (param {key!r} rejected)"
                continue
        if resp.status_code >= 400:
            raise RuntimeError(
                f"GetHookd /explore {resp.status_code}: {resp.text[:400]}"
            )
        if key and key != _gethookd_query_param:
            logger.info("gethookd: query param resolved to %r", key)
            _gethookd_query_param = key
        payload = resp.json() or {}
        rows = payload.get("data") or []
        return [normalize_gethookd_ad(ad) for ad in rows if isinstance(ad, dict)]

    raise RuntimeError(
        f"GetHookd /explore: all query param candidates rejected ({last_err})"
    )


_GETHOOKD_CLUSTER_PROMPT = """You are categorising competitor ads into reusable visual style templates.

Catalog styles (already known):
{catalog_ids}

Below are competitor ads that are already known to be high-performing. For each:
  1. If its visual style clearly matches a catalog id, tag it.
  2. Otherwise mark it "NEW" and help me cluster unknowns together (min cluster size: {min_size}).
  3. For each cluster, invent a short name, 1-sentence description, 3-6 visual cues, and a reusable prompt template with {{headline}} and {{cta_label}} placeholders. Template must end with "Square format."

Return STRICT JSON:

{{
  "classifications": [ {{ "index": int, "style_id": str }} ],
  "clusters": [
    {{ "name": str, "description": str, "visual_cues": [str],
       "prompt_template": str, "ad_indices": [int] }}
  ]
}}

Ads:
{ads_block}
"""


def _format_gethookd_for_cluster(ads: Sequence[Dict[str, Any]]) -> str:
    rows = []
    for i, ad in enumerate(ads):
        thumb = ""
        if ad.get("media"):
            thumb = ad["media"][0].get("thumbnail_url") or ad["media"][0].get("url") or ""
        rows.append(
            f"[{i}] brand={ad['brand']['name']!r} platform={ad['platform']!r} "
            f"format={ad['display_format']!r} title={ad['title']!r} "
            f"body={ad['body'][:180]!r} score={ad['performance_score_title']!r} "
            f"thumb={thumb!r}"
        )
    return "\n".join(rows) or "(none)"


def discover_from_gethookd(
    *,
    platform: str,
    keywords: Sequence[str] = (),
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 50,
    min_size: int = CLUSTER_MIN_SIZE,
) -> List[Dict[str, Any]]:
    """Pull high-performance competitor ads via GetHookd, cluster unknowns."""
    if not GETHOOKD_API_KEY:
        logger.info("GETHOOKD_API_KEY not configured — skipping gethookd discovery")
        _log_run(
            mode="gethookd",
            platform=platform,
            offer_id=None,
            inputs={"keywords": list(keywords), "filters": filters or {}, "limit": limit},
            candidates=[],
        )
        return []

    keywords = list(keywords or [])
    all_ads: List[Dict[str, Any]] = []
    queries: List[str] = keywords or [""]
    per_page = max(1, min(limit, 50))
    for q in queries:
        params: Dict[str, Any] = {
            "per_page": per_page,
            "page": 1,
        }
        if q:
            # Translated by _fetch_gethookd into whatever query param name
            # GetHookd currently accepts (query / search / keyword / ...).
            params["__query"] = q
        if filters:
            for k, v in filters.items():
                if v not in (None, ""):
                    params[k] = v
        try:
            rows = _fetch_gethookd(params)
        except Exception as exc:  # noqa: BLE001
            logger.warning("gethookd fetch failed for %r: %s", q, exc)
            continue
        all_ads.extend(rows)

    # De-dupe by id or external_id
    seen: set = set()
    unique: List[Dict[str, Any]] = []
    for ad in all_ads:
        key = ad.get("id") or ad.get("external_id") or ad.get("share_url")
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        unique.append(ad)

    # Keep only high-performance tiers
    high = [
        ad for ad in unique
        if (ad.get("performance_score_title") or "").strip().lower() in HIGH_PERFORMANCE_TIERS
    ]
    if not high:
        high = unique[:limit]  # fallback: use whatever we got
    high = high[:limit]

    if not high:
        _log_run(
            mode="gethookd",
            platform=platform,
            offer_id=None,
            inputs={"keywords": keywords, "filters": filters or {}, "limit": limit},
            candidates=[],
        )
        return []

    prompt = _GETHOOKD_CLUSTER_PROMPT.format(
        catalog_ids=", ".join(CATALOG_IDS),
        min_size=min_size,
        ads_block=_format_gethookd_for_cluster(high),
    )
    raw = _call_gemini_text(prompt) or _call_claude_text(prompt)
    parsed = _extract_json(raw) or {}

    emitted: List[Dict[str, Any]] = []
    clusters = (parsed.get("clusters") or []) if isinstance(parsed, dict) else []
    for cluster in clusters:
        indices = cluster.get("ad_indices") or []
        if not isinstance(indices, list) or len(indices) < min_size:
            continue
        brands = sorted({high[i]["brand"]["name"] for i in indices if 0 <= i < len(high)})
        share_urls = [high[i].get("share_url") for i in indices if 0 <= i < len(high)]
        thumbs = []
        for i in indices[:5]:
            if 0 <= i < len(high) and high[i].get("media"):
                t = high[i]["media"][0].get("thumbnail_url") or high[i]["media"][0].get("url")
                if t:
                    thumbs.append(t)
        name = str(cluster.get("name") or "New style").strip()
        cand = _base_candidate(
            style_id=name,
            name=name,
            description=str(cluster.get("description") or "").strip(),
            visual_cues=[str(v) for v in (cluster.get("visual_cues") or [])],
            prompt_template=str(cluster.get("prompt_template") or "").strip(),
            source="scrape",
            source_meta={
                "cluster_size": len(indices),
                "brands": brands,
                "share_urls": share_urls,
                "thumbnails": thumbs,
                "query": keywords,
            },
            reference_image_paths=thumbs,
        )
        saved = storage.upsert_style_candidate(cand, platform=platform)
        emitted.append(saved)

    _log_run(
        mode="gethookd",
        platform=platform,
        offer_id=None,
        inputs={
            "keywords": keywords,
            "filters": filters or {},
            "limit": limit,
            "total_fetched": len(unique),
            "high_performance": len(high),
        },
        candidates=emitted,
        raw=raw,
    )
    return emitted


# ----------------------------------------------------------------------
# 3. Pure LLM brainstorm
# ----------------------------------------------------------------------

_BRAINSTORM_PROMPT = """You are a senior direct-response creative director brainstorming fresh visual styles for static ad images.

Context:
- Offer: {offer_name}  (brand={brand})
- Headline: {headline}
- Body: {body}
- Existing catalog styles (avoid direct duplicates): {catalog_ids}
- Current winners digest (may be empty): {insights_hint}

Task:
Propose {count} brand-new ad-image STYLES (not copy variants) that could outperform the catalog for this offer.
For each style, provide: name, 1-sentence description, 3-6 concrete visual cues, and a reusable prompt template with {{headline}} and {{cta_label}} placeholders. Template must end with "Square format."

Return STRICT JSON:

[
  {{ "name": str, "description": str, "visual_cues": [str], "prompt_template": str }}
]
"""


def discover_from_brainstorm(
    offer: Dict[str, Any],
    *,
    platform: str,
    model: str = "claude-opus-4-7",
    count: int = 5,
) -> List[Dict[str, Any]]:
    """Pure-LLM ideation of new styles, no reference images."""
    insights = None
    try:
        insights = storage.load_insights(str(offer.get("id")), platform=platform)
    except Exception:  # noqa: BLE001
        insights = None
    hint = ""
    if insights:
        hint = json.dumps(
            {
                "top_hooks": insights.get("top_hooks") or [],
                "emotional_triggers": insights.get("emotional_triggers") or [],
                "mechanisms": insights.get("mechanisms") or [],
            },
            default=str,
        )

    prompt = _BRAINSTORM_PROMPT.format(
        offer_name=offer.get("name") or "(unnamed)",
        brand=offer.get("brand_name") or offer.get("name") or "(unknown)",
        headline=offer.get("headline") or "",
        body=offer.get("body") or offer.get("description") or "",
        catalog_ids=", ".join(CATALOG_IDS),
        insights_hint=hint or "(no insights yet)",
        count=count,
    )
    raw = ""
    if model.startswith("claude"):
        raw = _call_claude_text(prompt) or _call_gemini_text(prompt)
    else:
        raw = _call_gemini_text(prompt) or _call_claude_text(prompt)
    parsed = _extract_json(raw) or []
    if not isinstance(parsed, list):
        parsed = []

    emitted: List[Dict[str, Any]] = []
    for item in parsed[:count]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        cand = _base_candidate(
            style_id=name,
            name=name,
            description=str(item.get("description") or "").strip(),
            visual_cues=[str(v) for v in (item.get("visual_cues") or [])],
            prompt_template=str(item.get("prompt_template") or "").strip(),
            source="brainstorm",
            source_meta={"offer_id": offer.get("id"), "model": model},
        )
        saved = storage.upsert_style_candidate(cand, platform=platform)
        emitted.append(saved)

    _log_run(
        mode="brainstorm",
        platform=platform,
        offer_id=str(offer.get("id") or ""),
        inputs={"model": model, "count": count},
        candidates=emitted,
        raw=raw,
    )
    return emitted


# ----------------------------------------------------------------------
# 4. User uploads
# ----------------------------------------------------------------------

_UPLOAD_PROMPT = """You are analysing reference ad screenshots uploaded by a direct-response marketer.

Catalog styles (avoid duplicating): {catalog_ids}

For EACH attached image, extract one reusable visual STYLE template. If multiple images share the same style, group them into a single entry.
For each entry, provide: name, 1-sentence description, 3-6 visual cues, and a reusable prompt template with {{headline}} and {{cta_label}} placeholders. Template must end with "Square format."

Return STRICT JSON:

[
  {{ "name": str, "description": str, "visual_cues": [str], "prompt_template": str,
     "ad_indices": [int] }}
]
"""


def discover_from_uploads(
    offer_id: str,
    *,
    platform: str,
    image_paths: Sequence[str],
) -> List[Dict[str, Any]]:
    """Extract style templates from user-uploaded reference screenshots."""
    image_paths = [p for p in image_paths if p and os.path.exists(p)]
    if not image_paths:
        _log_run(
            mode="upload",
            platform=platform,
            offer_id=offer_id,
            inputs={"image_count": 0},
            candidates=[],
        )
        return []

    prompt = _UPLOAD_PROMPT.format(catalog_ids=", ".join(CATALOG_IDS))
    raw = _call_gemini_vision(prompt, image_paths=image_paths)
    parsed = _extract_json(raw) or []
    if not isinstance(parsed, list):
        parsed = []

    emitted: List[Dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        indices = item.get("ad_indices") or []
        refs = [image_paths[i] for i in indices if isinstance(i, int) and 0 <= i < len(image_paths)]
        if not refs:
            refs = list(image_paths)
        cand = _base_candidate(
            style_id=name,
            name=name,
            description=str(item.get("description") or "").strip(),
            visual_cues=[str(v) for v in (item.get("visual_cues") or [])],
            prompt_template=str(item.get("prompt_template") or "").strip(),
            source="upload",
            source_meta={"offer_id": offer_id, "image_count": len(refs)},
            reference_image_paths=refs,
        )
        saved = storage.upsert_style_candidate(cand, platform=platform)
        emitted.append(saved)

    _log_run(
        mode="upload",
        platform=platform,
        offer_id=offer_id,
        inputs={"image_count": len(image_paths)},
        candidates=emitted,
        raw=raw,
    )
    return emitted


# ----------------------------------------------------------------------
# Fan-out
# ----------------------------------------------------------------------

def discover_all(
    platform: str,
    *,
    offer_id: Optional[str] = None,
    gethookd_keywords: Optional[Sequence[str]] = None,
    gethookd_filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run every available discovery mode for ``platform`` and return per-mode results."""
    results: Dict[str, List[Dict[str, Any]]] = {
        "cluster_winners": [],
        "gethookd": [],
        "brainstorm": [],
        "upload": [],
    }
    try:
        results["cluster_winners"] = discover_from_winners(platform)
    except Exception as exc:  # noqa: BLE001
        logger.warning("discover_from_winners failed: %s", exc)
    try:
        if GETHOOKD_API_KEY:
            results["gethookd"] = discover_from_gethookd(
                platform=platform,
                keywords=gethookd_keywords or [],
                filters=gethookd_filters,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("discover_from_gethookd failed: %s", exc)
    try:
        if offer_id:
            for o in storage.list_offers(platform=platform):
                if str(o.get("id")) == str(offer_id):
                    results["brainstorm"] = discover_from_brainstorm(o, platform=platform)
                    break
    except Exception as exc:  # noqa: BLE001
        logger.warning("discover_from_brainstorm failed: %s", exc)
    # uploads are never run implicitly — always user-triggered.
    return results


__all__ = [
    "discover_from_winners",
    "discover_from_gethookd",
    "discover_from_brainstorm",
    "discover_from_uploads",
    "discover_all",
    "normalize_gethookd_ad",
]

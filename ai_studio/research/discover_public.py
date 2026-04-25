"""Public-libraries discovery agent — Meta Ad Library + TikTok Creative Center.

This is the second discovery agent (alongside ``discover.py`` which uses
GetHookd). It hits *public* ad libraries — no API tokens required — and
clusters the high-volume competitor ads into reusable style candidates.

Flow per offer:
  1. Reuse ``discover.derive_keywords_for_offer`` to pick 3–5 search
     phrases (LLM-derived, with regex fallback).
  2. Fan out across registered sources (Meta + TikTok by default) and
     collect normalized ad cards.
  3. Send the merged pool to Gemini/Claude for clustering into NEW
     style candidates that don't duplicate the existing catalog.
  4. Persist via ``storage.upsert_style_candidate`` with
     ``source="public_scout"`` and platform-specific source_meta.

Failure isolation: any one source erroring out (FB checkpoints, TikTok
rate limit, LLM timeout) is logged and skipped; the orchestrator never
raises out of a discovery sweep.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import storage

from . import discover as _disc
from .sources import meta_ad_library, tiktok_creative

logger = logging.getLogger(__name__)


CATALOG_IDS = _disc.CATALOG_IDS
CLUSTER_MIN_SIZE = _disc.CLUSTER_MIN_SIZE


_PUBLIC_CLUSTER_PROMPT = """You are categorising competitor ads from public ad libraries
(Meta Ad Library, TikTok Creative Center) into reusable visual style templates for
direct-response static ad creatives.

Catalog styles (already known — DO NOT re-emit duplicates of these):
{catalog_ids}

Below are ads pulled from public libraries. For each ad I provide source, advertiser,
headline, body, landing url, and (when available) image urls.

Your job:
  1. Decide if each ad's visual style clearly matches a catalog id (tag with that id).
  2. Otherwise mark "NEW" and cluster unknowns together. Minimum cluster size: {min_size}.
  3. For each cluster, invent: short style name (snake_case-friendly), 1-sentence
     description, 3-6 visual cues, and a reusable prompt template with {{headline}}
     and {{cta_label}} placeholders. Template must end with "Square format."
  4. Capture which platform (meta/tiktok) the cluster is most strongly anchored in.

Return STRICT JSON:

{{
  "classifications": [ {{ "index": int, "style_id": str }} ],
  "clusters": [
    {{
      "name": str,
      "description": str,
      "visual_cues": [str],
      "prompt_template": str,
      "ad_indices": [int],
      "anchor_platform": str
    }}
  ]
}}

Ads:
{ads_block}
"""


def _format_ads_for_cluster(ads: Sequence[Dict[str, Any]]) -> str:
    rows: List[str] = []
    for i, ad in enumerate(ads):
        thumb = ""
        imgs = ad.get("image_urls") or []
        if isinstance(imgs, list) and imgs:
            thumb = imgs[0]
        rows.append(
            f"[{i}] source={ad.get('source','')} advertiser={ad.get('advertiser','')!r} "
            f"headline={ad.get('headline','')!r} "
            f"body={(ad.get('body') or '')[:200]!r} "
            f"landing={ad.get('landing_url') or ''!r} thumb={thumb!r}"
        )
    return "\n".join(rows) or "(none)"


def _gather_from_sources(
    *,
    keywords: Sequence[str],
    sources: Sequence[str],
    limit_per_query: int,
    country: str,
) -> List[Dict[str, Any]]:
    """Pull ads from each enabled source. Best-effort, never raises."""
    all_ads: List[Dict[str, Any]] = []
    sources = [s.strip().lower() for s in (sources or []) if s and s.strip()]
    if not sources:
        sources = ["meta", "tiktok"]
    if "meta" in sources:
        try:
            rows = meta_ad_library.fetch_many(
                keywords,
                limit_per_query=limit_per_query,
                country=country,
            )
            logger.info(
                "public_scout: meta returned %s cards across %s queries",
                len(rows),
                len(keywords),
            )
            all_ads.extend(rows)
        except Exception as exc:  # noqa: BLE001
            logger.warning("public_scout: meta source failed: %s", exc)
    if "tiktok" in sources:
        try:
            rows = tiktok_creative.fetch_many(
                keywords,
                limit_per_query=limit_per_query,
                region=country,
            )
            logger.info(
                "public_scout: tiktok returned %s cards across %s queries",
                len(rows),
                len(keywords),
            )
            all_ads.extend(rows)
        except Exception as exc:  # noqa: BLE001
            logger.warning("public_scout: tiktok source failed: %s", exc)

    seen: set = set()
    unique: List[Dict[str, Any]] = []
    for ad in all_ads:
        rid = ad.get("id")
        if not rid or rid in seen:
            continue
        seen.add(rid)
        unique.append(ad)
    return unique


def discover_from_public(
    *,
    platform: str,
    keywords: Sequence[str],
    sources: Optional[Sequence[str]] = None,
    limit_per_query: int = 25,
    country: str = "US",
    min_size: int = CLUSTER_MIN_SIZE,
) -> List[Dict[str, Any]]:
    """Single-shot discovery: pull `keywords` from public sources, cluster, persist."""
    keywords = [k for k in (keywords or []) if k and str(k).strip()]
    if not keywords:
        _disc._log_run(
            mode="public_scout",
            platform=platform,
            offer_id=None,
            inputs={"keywords": [], "sources": list(sources or [])},
            candidates=[],
        )
        return []

    ads = _gather_from_sources(
        keywords=keywords,
        sources=sources or ["meta", "tiktok"],
        limit_per_query=limit_per_query,
        country=country,
    )
    if not ads:
        _disc._log_run(
            mode="public_scout",
            platform=platform,
            offer_id=None,
            inputs={
                "keywords": list(keywords),
                "sources": list(sources or []),
                "country": country,
            },
            candidates=[],
        )
        return []

    prompt = _PUBLIC_CLUSTER_PROMPT.format(
        catalog_ids=", ".join(CATALOG_IDS),
        min_size=min_size,
        ads_block=_format_ads_for_cluster(ads),
    )
    raw = _disc._call_gemini_text(prompt) or _disc._call_claude_text(prompt)
    parsed = _disc._extract_json(raw) or {}
    if not isinstance(parsed, dict):
        parsed = {}

    emitted: List[Dict[str, Any]] = []
    clusters = parsed.get("clusters") or []
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        indices = cluster.get("ad_indices") or []
        if not isinstance(indices, list) or len(indices) < min_size:
            continue
        members = [ads[i] for i in indices if isinstance(i, int) and 0 <= i < len(ads)]
        advertisers = sorted({m.get("advertiser") for m in members if m.get("advertiser")})
        landing = [m.get("landing_url") for m in members if m.get("landing_url")]
        thumbs: List[str] = []
        for m in members[:6]:
            for u in (m.get("image_urls") or []):
                thumbs.append(u)
                if len(thumbs) >= 6:
                    break
            if len(thumbs) >= 6:
                break
        anchor = str(cluster.get("anchor_platform") or "").lower().strip()
        if anchor not in {"meta", "tiktok", "mixed"}:
            sources_in = {m.get("source") for m in members}
            anchor = (
                "meta" if sources_in == {"meta"}
                else "tiktok" if sources_in == {"tiktok"}
                else "mixed"
            )

        name = str(cluster.get("name") or "New public style").strip()
        cand = _disc._base_candidate(
            style_id=name,
            name=name,
            description=str(cluster.get("description") or "").strip(),
            visual_cues=[str(v) for v in (cluster.get("visual_cues") or [])],
            prompt_template=str(cluster.get("prompt_template") or "").strip(),
            source="public_scout",
            source_meta={
                "cluster_size": len(indices),
                "advertisers": advertisers,
                "landing_urls": [u for u in landing if u][:6],
                "thumbnails": thumbs,
                "queries": list(keywords),
                "anchor_platform": anchor,
            },
            reference_image_paths=thumbs,
        )
        saved = storage.upsert_style_candidate(cand, platform=platform)
        emitted.append(saved)

    _disc._log_run(
        mode="public_scout",
        platform=platform,
        offer_id=None,
        inputs={
            "keywords": list(keywords),
            "sources": list(sources or []),
            "country": country,
            "total_ads": len(ads),
        },
        candidates=emitted,
        raw=raw,
    )
    return emitted


def discover_all_public(
    platform: str,
    *,
    offer_id: Optional[str] = None,
    scan_all_offers: bool = False,
    keywords_per_offer: int = 4,
    limit_per_query: int = 20,
    country: str = "US",
    sources: Optional[Sequence[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Per-offer public-libraries sweep.

    Mirrors ``discover.discover_all`` but uses the public sources only.
    Designed for the every-N-hour public scout job and the
    ``/api/agent/run-public-scout`` endpoint.
    """
    results: Dict[str, List[Dict[str, Any]]] = {"public_scout": []}

    offers_to_scan: List[Dict[str, Any]] = []
    if offer_id:
        for o in storage.list_offers(platform=platform):
            if str(o.get("id")) == str(offer_id):
                offers_to_scan = [o]
                break
    elif scan_all_offers:
        offers_to_scan = list(storage.list_offers(platform=platform) or [])

    if not offers_to_scan:
        logger.info(
            "discover_all_public: no offers for platform=%s — running an empty sweep",
            platform,
        )
        return results

    for o in offers_to_scan:
        try:
            keywords = _disc.derive_keywords_for_offer(o, count=keywords_per_offer)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "derive_keywords_for_offer failed for offer=%s: %s",
                o.get("id"),
                exc,
            )
            keywords = _disc._heuristic_keywords(o, count=keywords_per_offer)
        if not keywords:
            continue
        logger.info(
            "public_scout keywords for offer=%s (%s): %s",
            o.get("id"),
            o.get("name"),
            keywords,
        )
        try:
            rows = discover_from_public(
                platform=platform,
                keywords=keywords,
                sources=sources,
                limit_per_query=limit_per_query,
                country=country,
            )
            results["public_scout"].extend(rows)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "discover_from_public failed for offer=%s: %s", o.get("id"), exc
            )
    return results


__all__ = ["discover_from_public", "discover_all_public"]

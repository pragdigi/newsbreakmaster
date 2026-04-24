"""Winner detection: scan report rows for ad-level proven performers.

A "winner" is an ad that simultaneously clears three thresholds on a recent
time window:
  1. Minimum spend (default $20 — skip tiny experiments).
  2. Minimum conversions (default 3 — skip statistical noise).
  3. CPA below its offer's ``target_cpa * cpa_factor`` (default factor 1.0 =
     at or under target).

Winners are upserted into ``storage/catalog/<platform>/winners.json`` with
enriched creative copy and image URLs pulled from the adapter. Ads that
fall out of the threshold on the next refresh flip to ``proven=false``
rather than being deleted, so the audit trail survives regressions.

Also runs the generation feedback cross-reference: when a winner's
``ad_id`` matches a ``launched_ad_ids`` entry in ``generations.jsonl``,
the generation row is patched with ``becomes_winner=true`` so the bandit
has outcome data to learn from.
"""
from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlsplit

import storage

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - requests is a hard dep but be defensive
    requests = None  # type: ignore

logger = logging.getLogger(__name__)


def _guess_image_ext(url: str, content_type: Optional[str]) -> str:
    if content_type:
        ct = content_type.split(";", 1)[0].strip().lower()
        if ct == "image/png":
            return "png"
        if ct in ("image/jpeg", "image/jpg"):
            return "jpg"
        if ct == "image/webp":
            return "webp"
        if ct == "image/gif":
            return "gif"
    path = urlsplit(url).path.lower()
    for e in ("png", "jpg", "jpeg", "webp", "gif"):
        if path.endswith("." + e):
            return "jpg" if e == "jpeg" else e
    return "jpg"


def _cache_winner_image(ad_id: str, image_url: Optional[str], *, platform: str) -> Optional[str]:
    """Download the ad's creative once and save it locally for later visual
    context (multimodal analyzer, winners UI, etc.). Returns the local path
    or ``None`` if the fetch fails. Idempotent: keeps the existing file when
    one is already on disk for this ad_id.
    """
    if not image_url or not ad_id or requests is None:
        return None
    # Reuse any existing cached file for this ad (ignore extension).
    existing_dir = storage.winner_image_dir(platform)
    try:
        for fn in os.listdir(existing_dir):
            if fn.rsplit(".", 1)[0] == str(ad_id):
                return os.path.join(existing_dir, fn)
    except FileNotFoundError:
        pass
    try:
        resp = requests.get(image_url, timeout=15, stream=True)
        resp.raise_for_status()
        ext = _guess_image_ext(image_url, resp.headers.get("Content-Type"))
        path = storage.winner_image_path(ad_id, platform=platform, ext=ext)
        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)
        return path
    except Exception as exc:  # noqa: BLE001
        logger.debug("winner image cache failed ad=%s err=%s", ad_id, exc)
        return None

# Defaults — tweak per offer via env overrides
DEFAULT_WINDOW_DAYS = int(os.environ.get("AD_STUDIO_WINNERS_WINDOW_DAYS", "14"))
DEFAULT_MIN_SPEND = float(os.environ.get("AD_STUDIO_WINNERS_MIN_SPEND", "20"))
DEFAULT_MIN_CONV = float(os.environ.get("AD_STUDIO_WINNERS_MIN_CONV", "3"))
DEFAULT_CPA_FACTOR = float(os.environ.get("AD_STUDIO_WINNERS_CPA_FACTOR", "1.0"))


def _target_cpa_lookup(platform: str) -> Dict[str, float]:
    """Map pixel_id → target_cpa across all saved offers for quick tagging."""
    lookup: Dict[str, float] = {}
    for o in storage.list_offers(platform=platform):
        cpa = o.get("target_cpa")
        if cpa is None:
            continue
        pid = str(o.get("pixel_id") or "").strip()
        if pid:
            lookup[pid] = float(cpa)
    return lookup


def _offer_lookup_by_landing(platform: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for o in storage.list_offers(platform=platform):
        url = (o.get("landing_url") or "").strip().lower()
        if url:
            out[url] = o
    return out


def _guess_offer_for_row(
    row: Dict[str, Any],
    *,
    platform: str,
    offers_by_landing: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Best-effort offer attribution via landing URL match.

    Falls back to None if no saved offer looks like the right fit — winners
    still get tracked but won't be routable back into the studio's
    per-offer insights refresher until a human fixes it.
    """
    raw = row.get("raw") or {}
    meta = row.get("metadata") or raw.get("metadata") or {}
    landing = (
        row.get("landing_page_url")
        or meta.get("landing_page_url")
        or (raw.get("creative") or {}).get("landing_page_url")
        or ""
    )
    landing = str(landing).strip().lower()
    if not landing:
        return None
    # Strip UTMs before matching.
    base = landing.split("?")[0]
    for key, offer in offers_by_landing.items():
        key_base = key.split("?")[0]
        if key_base == base:
            return offer
    return None


def _score(cpa: Optional[float], target_cpa: Optional[float], conversions: float) -> float:
    """Simple deterministic score: lower CPA + more conv = higher number."""
    if not cpa or cpa <= 0:
        return float(conversions or 0)
    edge = 1.0
    if target_cpa and target_cpa > 0:
        edge = max(0.1, target_cpa / cpa)
    return round(edge * (conversions or 0), 3)


# Keys that, on any platform we integrate with, tend to hold an image URL.
# NewsBreak uses ``assetUrl`` / ``assetMediaUrl`` nested under ``creative.content``;
# SmartNews uses ``url`` / ``file_url`` nested under ``image_creative_info.media_files``;
# GetHookd/Meta/TikTok discovery payloads use the snake/camel variants.
_IMAGE_URL_KEYS = (
    "assetUrl", "asset_url",
    "url", "file_url", "fileUrl",
    "preview_url", "previewUrl",
    "image_url", "imageUrl",
    "thumbnailUrl", "thumbnail_url",
    "display_url", "displayUrl",
    "mediaUrl", "media_url",
    "cover_url", "coverUrl",
    "src",
)


def _first_image_url(*candidates: Any, _depth: int = 0) -> Optional[str]:
    """Walk arbitrary nested dicts/lists and return the first URL that looks
    like an image. Handles SmartNews ``image_creative_info.media_files``,
    NewsBreak's ``creative.content.assetUrl``, and the flatter
    ``imageUrl`` / ``image_url`` shapes from normalized rows.

    Bounded recursion (depth<=6) so pathological payloads can't loop.
    """
    if _depth > 6:
        return None
    for c in candidates:
        if not c:
            continue
        if isinstance(c, str):
            if c.startswith(("http://", "https://")):
                return c
            continue
        if isinstance(c, dict):
            # Pass 1 — prefer the well-known image keys directly on this dict.
            for k in _IMAGE_URL_KEYS:
                v = c.get(k)
                if isinstance(v, str) and v.startswith(("http://", "https://")):
                    return v
            # Pass 2 — descend into nested containers (creative.content, etc.).
            for v in c.values():
                if isinstance(v, (dict, list, tuple)):
                    got = _first_image_url(v, _depth=_depth + 1)
                    if got:
                        return got
            continue
        if isinstance(c, (list, tuple)):
            for item in c:
                got = _first_image_url(item, _depth=_depth + 1)
                if got:
                    return got
    return None


def _first_str(source: Any, *keys: str) -> str:
    """Fetch the first non-empty string value from ``source`` walking the
    given keys in order. ``source`` may be ``None`` or non-dict; returns
    empty string in that case.
    """
    if not isinstance(source, dict):
        return ""
    for k in keys:
        v = source.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _build_account_index(
    adapter,
    account_id: str,
    cache: Dict[tuple, Dict[str, str]],
) -> Dict[str, str]:
    """Walk the full campaign → ad_group → ad tree for an account and build
    an ``ad_id → ad_group_id`` index. One network burst per account per
    refresh, then every row lookup is O(1).

    NewsBreak's ``/report`` at AD dim doesn't include ``adSetId`` or
    ``campaignId`` on each row, so this is the only reliable way to resolve
    the parent group needed by ``_creative_for_ad``.
    """
    key = ("account", str(account_id))
    mapping = cache.get(key)
    if mapping is not None:
        return mapping
    mapping = {}
    try:
        campaigns = adapter.get_campaigns(account_id) or []
    except Exception as e:
        logger.warning(
            "winners.enrich: get_campaigns failed account=%s err=%s",
            account_id, e,
        )
        cache[key] = mapping
        return mapping
    campaign_ids: List[str] = []
    for c in campaigns:
        cid = str(
            c.get("id") or c.get("campaign_id") or c.get("campaignId") or ""
        )
        if cid:
            campaign_ids.append(cid)

    # If get_campaigns returned nothing (or far fewer than we expected),
    # try pulling campaign IDs from the report so we can still walk the
    # ad-group tree. NewsBreak's /campaign/getList can omit campaigns
    # depending on status filters, so this is a useful safety net.
    if not campaign_ids or len(campaign_ids) < 2:
        try:
            from datetime import date as _d, timedelta as _td

            today = _d.today()
            rep = adapter.fetch_report_rows(
                account_id, "campaign", today - _td(days=90), today
            ) or []
            for r in rep:
                cid = str(
                    r.get("id")
                    or r.get("campaign_id")
                    or r.get("campaignId")
                    or ""
                )
                if cid and cid not in campaign_ids:
                    campaign_ids.append(cid)
        except Exception as e:
            logger.debug(
                "winners.enrich: campaign-report fallback failed account=%s err=%s",
                account_id, e,
            )

    total_groups = 0
    group_ids: List[str] = []
    for cid in campaign_ids:
        try:
            groups = adapter.get_ad_groups(account_id, cid) or []
        except Exception as e:
            logger.debug(
                "winners.enrich: get_ad_groups failed campaign=%s err=%s", cid, e,
            )
            continue
        for g in groups:
            gid = str(
                g.get("id")
                or g.get("ad_set_id")
                or g.get("adSetId")
                or g.get("ad_group_id")
                or ""
            )
            if not gid:
                continue
            total_groups += 1
            if gid not in group_ids:
                group_ids.append(gid)

    # Defense-in-depth: if walking campaigns → ad_groups didn't surface
    # any groups, hit the AD_SET dim report to recover group IDs. This is
    # what unblocks NewsBreak accounts where /campaign/getList returns
    # filtered results (status-based) but the report still shows spend.
    if not group_ids:
        try:
            from datetime import date as _d, timedelta as _td

            today = _d.today()
            ad_set_rep = adapter.fetch_report_rows(
                account_id, "ad_set", today - _td(days=90), today
            ) or []
            for r in ad_set_rep:
                gid = str(
                    r.get("id")
                    or r.get("ad_set_id")
                    or r.get("adSetId")
                    or ""
                )
                if gid and gid not in group_ids:
                    group_ids.append(gid)
            logger.info(
                "winners.enrich: ad_set-report fallback account=%s groups=%d",
                account_id, len(group_ids),
            )
        except Exception as e:
            logger.debug(
                "winners.enrich: ad_set-report fallback failed account=%s err=%s",
                account_id, e,
            )

    for gid in group_ids:
        try:
            ads = adapter.get_ads(account_id, gid) or []
        except Exception as e:
            logger.debug(
                "winners.enrich: get_ads failed group=%s err=%s", gid, e,
            )
            continue
        for a in ads:
            aid_candidate = str(
                a.get("id") or a.get("ad_id") or a.get("adId") or ""
            )
            if aid_candidate:
                mapping[aid_candidate] = gid
    logger.info(
        "winners.enrich: built account index account=%s campaigns=%d groups=%d ads=%d",
        account_id, len(campaign_ids), len(group_ids), len(mapping),
    )
    cache[key] = mapping
    return mapping


def _discover_group_id_via_campaign(
    adapter,
    account_id: str,
    campaign_id: str,
    ad_id: str,
    cache: Dict[tuple, Dict[str, str]],
) -> Optional[str]:
    """When the report row does expose campaign_id but not ad_set_id, walk
    just that campaign once and cache per (account, campaign).
    """
    key = ("campaign", str(account_id), str(campaign_id))
    mapping = cache.get(key)
    if mapping is None:
        mapping = {}
        try:
            groups = adapter.get_ad_groups(account_id, campaign_id) or []
        except Exception as e:
            logger.warning(
                "winners.enrich: get_ad_groups failed account=%s campaign=%s err=%s",
                account_id, campaign_id, e,
            )
            cache[key] = mapping
            return None
        for g in groups:
            gid = str(
                g.get("id")
                or g.get("ad_set_id")
                or g.get("adSetId")
                or g.get("ad_group_id")
                or ""
            )
            if not gid:
                continue
            try:
                ads = adapter.get_ads(account_id, gid) or []
            except Exception as e:
                logger.debug(
                    "winners.enrich: get_ads failed group=%s err=%s", gid, e,
                )
                continue
            for a in ads:
                aid_candidate = str(
                    a.get("id") or a.get("ad_id") or a.get("adId") or ""
                )
                if aid_candidate:
                    mapping[aid_candidate] = gid
        logger.info(
            "winners.enrich: built campaign index account=%s campaign=%s groups=%d ads=%d",
            account_id, campaign_id, len(groups), len(mapping),
        )
        cache[key] = mapping
    return mapping.get(str(ad_id))


def _creative_for_ad(adapter, account_id: str, ad_group_id: Optional[str], ad_id: str) -> Dict[str, Any]:
    """Pull headline/description/image_url for a single ad by hitting the
    adapter's ``get_ads`` with the known parent id and filtering to ``ad_id``.

    Fails open: returns ``{}`` if the adapter can't satisfy the request.
    Handles both SmartNews (``creative.image_creative_info.media_files``) and
    NewsBreak (``creative.content.assetUrl``) response shapes.
    """
    if not ad_group_id:
        logger.info(
            "winners.enrich: skip ad_id=%s — no parent ad_set_id on report row",
            ad_id,
        )
        return {}
    try:
        ads = adapter.get_ads(account_id, ad_group_id) or []
    except Exception as e:
        logger.warning(
            "winners.enrich: get_ads failed account=%s group=%s err=%s",
            account_id, ad_group_id, e,
        )
        return {}
    logger.info(
        "winners.enrich: get_ads account=%s group=%s returned %d ads (looking for %s)",
        account_id, ad_group_id, len(ads), ad_id,
    )
    for a in ads:
        if str(a.get("id") or a.get("ad_id") or a.get("adId") or "") != str(ad_id):
            continue
        raw = a.get("raw") or a
        creative = a.get("creative") or raw.get("creative") or {}
        # SmartNews shape: ``creative.image_creative_info``
        img_info = creative.get("image_creative_info") or {}
        # NewsBreak shape: ``creative.content.{assetUrl,headline,description,...}``
        content = creative.get("content") or {}
        # Walk everything we know about — _first_image_url handles nested dicts.
        image_url = _first_image_url(
            content,
            img_info,
            creative,
            raw.get("creatives"),
            raw.get("media"),
            raw,
            a,
        )
        headline = (
            _first_str(img_info, "headline")
            or _first_str(content, "headline", "title")
            or _first_str(creative, "headline", "title")
            or _first_str(a, "headline", "name")
            or _first_str(raw, "headline", "title")
        )
        description = (
            _first_str(img_info, "description")
            or _first_str(content, "description", "body")
            or _first_str(creative, "description", "body")
            or _first_str(a, "body", "description")
            or _first_str(raw, "description", "body")
        )
        sponsored_name = (
            _first_str(img_info, "sponsored_name")
            or _first_str(content, "brandName", "brand_name", "sponsored_name", "sponsoredName")
            or _first_str(creative, "sponsored_name", "sponsoredName", "brandName")
            or _first_str(raw, "sponsoredName", "sponsored_name", "brandName")
        )
        landing_page_url = (
            _first_str(a, "landing_page_url", "landingPageUrl")
            or _first_str(content, "clickThroughUrl", "click_through_url", "landing_page_url", "landingPageUrl")
            or _first_str(creative, "landing_page_url", "landingPageUrl", "clickThroughUrl")
            or _first_str(raw, "landing_page_url", "landingPageUrl")
        )
        cta_label = (
            _first_str(a, "cta_label", "ctaLabel")
            or _first_str(content, "callToAction", "call_to_action", "cta_label", "ctaLabel")
            or _first_str(creative, "callToAction", "cta_label", "ctaLabel")
            or _first_str(raw, "callToAction", "cta_label", "ctaLabel")
        )
        if not image_url:
            try:
                sample_keys = {
                    "ad_keys": sorted(list(a.keys()))[:15],
                    "creative_keys": sorted(list(creative.keys()))[:15] if isinstance(creative, dict) else None,
                    "content_keys": sorted(list(content.keys()))[:15] if isinstance(content, dict) else None,
                }
                logger.warning(
                    "winners.enrich: no image_url for ad_id=%s — keys=%s",
                    ad_id, sample_keys,
                )
            except Exception:
                pass
        else:
            logger.info(
                "winners.enrich: ad_id=%s image_url=%s headline=%r",
                ad_id, image_url, (headline or "")[:60],
            )
        return {
            "headline": headline,
            "description": description,
            "sponsored_name": sponsored_name or None,
            "landing_page_url": landing_page_url or None,
            "cta_label": cta_label or None,
            "image_url": image_url,
        }
    logger.warning(
        "winners.enrich: ad_id=%s not in %d ads returned by get_ads(group=%s)",
        ad_id, len(ads), ad_group_id,
    )
    return {}


def _cross_reference_generations(platform: str, winning_ad_ids: Iterable[str]) -> int:
    """Patch any matching generation row with ``becomes_winner=true``."""
    targets = {str(x) for x in winning_ad_ids if x}
    if not targets:
        return 0
    flipped = 0
    for row in storage.list_generations(platform=platform, limit=0):
        if row.get("becomes_winner"):
            continue
        launched = row.get("launched_ad_ids") or []
        if any(str(x) in targets for x in launched):
            storage.update_generation(
                row.get("gen_id"),
                {"becomes_winner": True},
                platform=platform,
            )
            flipped += 1
    return flipped


def refresh_winners(
    adapter,
    *,
    platform: Optional[str] = None,
    days: int = DEFAULT_WINDOW_DAYS,
    cpa_factor: float = DEFAULT_CPA_FACTOR,
    min_spend: float = DEFAULT_MIN_SPEND,
    min_conv: float = DEFAULT_MIN_CONV,
    account_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Scan ad-level report rows across all accounts and upsert winners.

    Returns a summary dict:
        {"added", "updated", "demoted", "considered", "accounts", "errors"}
    """
    plat = platform or getattr(adapter, "platform", None) or storage.DEFAULT_PLATFORM
    today = date.today()
    start = today - timedelta(days=max(1, days))

    target_cpa_by_pixel = _target_cpa_lookup(plat)
    offers_by_landing = _offer_lookup_by_landing(plat)
    existing = {
        str(w.get("ad_id") or w.get("id")): w for w in storage.list_winners(platform=plat)
    }

    account_list: List[str]
    if account_ids is None:
        try:
            raw_accounts = adapter.get_accounts() or []
        except Exception as e:  # pragma: no cover - network
            return {
                "added": 0,
                "updated": 0,
                "demoted": 0,
                "considered": 0,
                "accounts": [],
                "errors": [{"stage": "get_accounts", "error": str(e)}],
            }
        account_list = [str(a.get("id")) for a in raw_accounts if a.get("id")]
    else:
        account_list = [str(a) for a in account_ids]

    added = 0
    updated = 0
    considered = 0
    errors: List[Dict[str, Any]] = []
    per_account: List[Dict[str, Any]] = []
    current_winner_ids: set[str] = set()
    # cache: (account_id, campaign_id) → {ad_id: ad_group_id} to avoid
    # re-fetching ad groups + ads for every row in the same campaign.
    _ad_group_cache: Dict[tuple, Dict[str, str]] = {}

    for aid in account_list:
        try:
            rows = adapter.fetch_report_rows(aid, "ad", start, today) or []
        except Exception as e:  # pragma: no cover - network
            errors.append({"ad_account_id": aid, "stage": "fetch_report_rows", "error": str(e)})
            continue

        acct_added = 0
        acct_updated = 0
        for row in rows:
            considered += 1
            ad_id = str(row.get("ad_id") or row.get("id") or "")
            if not ad_id:
                continue
            spend = float(row.get("spend") or 0.0)
            conv = float(row.get("conversions") or 0.0)
            cpa = row.get("cpa")
            if spend < min_spend:
                continue
            if conv < min_conv:
                continue

            offer = _guess_offer_for_row(row, platform=plat, offers_by_landing=offers_by_landing)
            # Pick target_cpa in priority order: offer → pixel lookup → None
            target_cpa: Optional[float] = None
            if offer and offer.get("target_cpa") is not None:
                try:
                    target_cpa = float(offer["target_cpa"])
                except (TypeError, ValueError):
                    target_cpa = None
            if target_cpa is None:
                raw = row.get("raw") or {}
                meta = row.get("metadata") or raw.get("metadata") or {}
                pid = str(
                    meta.get("pixel_id")
                    or meta.get("website_tracking_tag_id")
                    or raw.get("website_tracking_tag_id")
                    or ""
                )
                if pid and pid in target_cpa_by_pixel:
                    target_cpa = target_cpa_by_pixel[pid]

            if target_cpa is not None and cpa is not None:
                if float(cpa) > target_cpa * cpa_factor:
                    continue
            elif target_cpa is None and cpa is not None and cpa > 0:
                # No target available → still accept if spend + conv thresholds
                # clear, but record target=None so analyzer can weight it lower.
                pass

            group_id = (
                row.get("ad_set_id")
                or row.get("parent_id")
                or row.get("ad_group_id")
                or row.get("adSetId")
            )
            campaign_id = (
                row.get("campaign_id")
                or row.get("campaignId")
                or (row.get("raw") or {}).get("campaignId")
            )
            if not group_id and campaign_id:
                group_id = _discover_group_id_via_campaign(
                    adapter, aid, str(campaign_id), str(ad_id), _ad_group_cache
                )
                if group_id:
                    logger.info(
                        "winners.enrich: resolved ad_id=%s → ad_group_id=%s via campaign=%s",
                        ad_id, group_id, campaign_id,
                    )
            if not group_id:
                # Fallback for NewsBreak: report rows don't carry campaign_id
                # either, so walk the whole account tree once and resolve
                # ad_id → ad_group_id from a global index.
                account_index = _build_account_index(adapter, aid, _ad_group_cache)
                group_id = account_index.get(str(ad_id))
                if group_id:
                    logger.info(
                        "winners.enrich: resolved ad_id=%s → ad_group_id=%s via account index",
                        ad_id, group_id,
                    )
                else:
                    logger.info(
                        "winners.enrich: ad=%s not found in account=%s index (size=%d)",
                        ad_id, aid, len(account_index),
                    )
            creative = _creative_for_ad(adapter, aid, group_id, ad_id)

            # Download+cache the creative image (best-effort, never fatal)
            local_img = _cache_winner_image(
                ad_id, creative.get("image_url"), platform=plat
            )

            winner = {
                "ad_id": ad_id,
                "ad_account_id": aid,
                "offer_id": (offer or {}).get("id"),
                "headline": creative.get("headline") or row.get("name") or "",
                "description": creative.get("description") or "",
                "image_url": creative.get("image_url"),
                "image_local_path": local_img,
                "sponsored_name": creative.get("sponsored_name"),
                "landing_url": creative.get("landing_page_url") or row.get("landing_page_url"),
                "metrics": {
                    "spend": round(spend, 2),
                    "conversions": conv,
                    "cpa": round(float(cpa), 2) if cpa is not None else None,
                    "roas": row.get("roas"),
                    "ctr": round(float(row.get("ctr") or 0.0), 3),
                    "impressions": int(row.get("impressions") or 0),
                    "clicks": int(row.get("clicks") or 0),
                },
                "target_cpa": target_cpa,
                "window_days": days,
                "score": _score(cpa, target_cpa, conv),
                "proven": True,
                "marked_at": today.isoformat(),
            }

            prior = existing.get(ad_id)
            storage.upsert_winner(winner, platform=plat)
            if prior:
                acct_updated += 1
            else:
                acct_added += 1
            current_winner_ids.add(ad_id)

        per_account.append(
            {"ad_account_id": aid, "added": acct_added, "updated": acct_updated}
        )
        added += acct_added
        updated += acct_updated

    # Demote anyone who was in the list but didn't clear thresholds this run.
    demoted = 0
    for ad_id, prior in list(existing.items()):
        if ad_id in current_winner_ids:
            continue
        if prior.get("proven") is False:
            continue
        patch = {**prior, "proven": False, "marked_at": today.isoformat()}
        storage.upsert_winner(patch, platform=plat)
        demoted += 1

    # Feedback loop: mark generations that produced winners.
    feedback_flipped = _cross_reference_generations(plat, current_winner_ids)

    return {
        "added": added,
        "updated": updated,
        "demoted": demoted,
        "considered": considered,
        "accounts": per_account,
        "accounts_scanned": len(account_list),
        "winners_found": added + updated,
        "errors": errors,
        "generations_linked": feedback_flipped,
        "window_days": days,
        "platform": plat,
    }


__all__ = ["refresh_winners"]

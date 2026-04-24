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

import storage

logger = logging.getLogger(__name__)

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


def _first_image_url(*candidates: Any) -> Optional[str]:
    """Walk a handful of candidate shapes and return the first URL-looking
    string we can find. Handles SmartNews ``image_creative_info.media_files``
    and NewsBreak's flatter ``imageUrl`` / ``image_url`` / ``creatives[*]``.
    """
    for c in candidates:
        if not c:
            continue
        if isinstance(c, str) and c.startswith(("http://", "https://")):
            return c
        if isinstance(c, dict):
            for k in (
                "url", "file_url", "preview_url", "image_url", "imageUrl",
                "thumbnailUrl", "thumbnail_url", "display_url", "displayUrl",
            ):
                v = c.get(k)
                if isinstance(v, str) and v.startswith(("http://", "https://")):
                    return v
        if isinstance(c, (list, tuple)):
            for item in c:
                got = _first_image_url(item)
                if got:
                    return got
    return None


def _creative_for_ad(adapter, account_id: str, ad_group_id: Optional[str], ad_id: str) -> Dict[str, Any]:
    """Pull headline/description/image_url for a single ad by hitting the
    adapter's ``get_ads`` with the known parent id and filtering to ``ad_id``.

    Fails open: returns ``{}`` if the adapter can't satisfy the request.
    Handles both SmartNews (``creative.image_creative_info.media_files``) and
    NewsBreak (``imageUrl`` / ``creatives[*].imageUrl``) response shapes.
    """
    if not ad_group_id:
        return {}
    try:
        ads = adapter.get_ads(account_id, ad_group_id) or []
    except Exception as e:  # pragma: no cover - network
        logger.debug("get_ads failed account=%s group=%s err=%s", account_id, ad_group_id, e)
        return {}
    for a in ads:
        if str(a.get("id") or a.get("ad_id") or a.get("adId") or "") != str(ad_id):
            continue
        raw = a.get("raw") or a
        creative = a.get("creative") or raw.get("creative") or {}
        img_info = creative.get("image_creative_info") or {}
        image_url = _first_image_url(
            img_info.get("media_files"),
            creative.get("media_files"),
            creative.get("creatives"),
            raw.get("creatives"),
            raw.get("media"),
            raw.get("imageUrl"),
            raw.get("image_url"),
            a.get("imageUrl"),
            a.get("image_url"),
        )
        return {
            "headline": (
                img_info.get("headline")
                or creative.get("headline")
                or a.get("headline")
                or raw.get("headline")
                or raw.get("title")
                or a.get("name")
                or ""
            ),
            "description": (
                img_info.get("description")
                or creative.get("description")
                or a.get("body")
                or raw.get("description")
                or raw.get("body")
                or ""
            ),
            "sponsored_name": (
                img_info.get("sponsored_name")
                or creative.get("sponsored_name")
                or raw.get("sponsoredName")
                or raw.get("sponsored_name")
            ),
            "landing_page_url": (
                a.get("landing_page_url")
                or raw.get("landing_page_url")
                or raw.get("landingPageUrl")
                or a.get("landingPageUrl")
            ),
            "cta_label": a.get("cta_label") or raw.get("cta_label") or raw.get("ctaLabel"),
            "image_url": image_url,
        }
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

            # Enrich with creative copy (best effort)
            creative = _creative_for_ad(
                adapter,
                aid,
                row.get("ad_set_id") or row.get("parent_id") or row.get("ad_group_id"),
                ad_id,
            )

            winner = {
                "ad_id": ad_id,
                "ad_account_id": aid,
                "offer_id": (offer or {}).get("id"),
                "headline": creative.get("headline") or row.get("name") or "",
                "description": creative.get("description") or "",
                "image_url": creative.get("image_url"),
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

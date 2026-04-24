"""Lifecycle reconciler for style candidates.

Run nightly (called from :func:`scheduler.run_ad_studio_nightly`).

For each platform we:
  1. Roll up outcomes from ``generations.jsonl`` + ``winners.json`` to
     compute per-style trials / wins / spend / CPA windows.
  2. Promote candidates with enough trials and a CPA edge over the catalog
     median (``status="promoted"``, stamps ``promoted_at``).
  3. Archive candidates that have clearly underperformed.
  4. Flag catalog styles in the bottom CPA quartile over the last 90 days
     so humans can decide whether to retire them (we never auto-remove).
"""
from __future__ import annotations

import logging
import os
import statistics
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PROMOTE_MIN_TRIALS = int(os.environ.get("AD_STUDIO_PROMOTE_MIN_TRIALS", "20"))
ARCHIVE_MIN_TRIALS = int(os.environ.get("AD_STUDIO_ARCHIVE_MIN_TRIALS", "15"))
PROMOTE_CPA_EDGE = float(os.environ.get("AD_STUDIO_PROMOTE_CPA_EDGE", "0.10"))  # 10% better
ARCHIVE_NO_WIN_TRIALS = int(os.environ.get("AD_STUDIO_ARCHIVE_NO_WIN_TRIALS", "30"))


def _per_style_cpa(platform: str) -> Dict[str, Dict[str, float]]:
    """Roll up ``winners.json`` + ``generations.jsonl`` into per-style stats.

    Returns ``{style_id: {trials, wins, spend, conversions, cpa}}``.
    """
    import storage

    winners = {str(w.get("ad_id")): w for w in storage.list_winners(platform=platform)}
    gens = storage.list_generations(platform=platform, limit=5000)
    stats: Dict[str, Dict[str, float]] = {}

    for gen in gens:
        style_ids = gen.get("style_ids") or []
        launched = gen.get("launched_ad_ids") or []
        is_winner_flag = bool(gen.get("becomes_winner"))
        # When the generation knows its launched ads, attribute per style.
        for sid, ad_id in zip(style_ids, launched + [None] * (len(style_ids) - len(launched))):
            if not sid:
                continue
            bucket = stats.setdefault(
                sid, {"trials": 0, "wins": 0, "spend": 0.0, "conversions": 0, "cpa": None}
            )
            bucket["trials"] += 1
            if ad_id and str(ad_id) in winners:
                w = winners[str(ad_id)]
                metrics = (w.get("metrics") or {})
                bucket["wins"] += 1
                bucket["spend"] += float(metrics.get("spend") or 0)
                bucket["conversions"] += int(metrics.get("conversions") or 0)
            elif is_winner_flag and not launched:
                # Older gens that were stamped winner without per-ad attribution.
                bucket["wins"] += 1

    for sid, bucket in stats.items():
        conv = bucket["conversions"]
        spend = bucket["spend"]
        bucket["cpa"] = (spend / conv) if conv else None
    return stats


def _catalog_median_cpa(stats: Dict[str, Dict[str, float]], catalog_ids: List[str]) -> Optional[float]:
    vals = [stats[s]["cpa"] for s in catalog_ids if s in stats and stats[s].get("cpa") is not None]
    if not vals:
        return None
    return float(statistics.median(vals))


def _catalog_bottom_quartile(stats: Dict[str, Dict[str, float]], catalog_ids: List[str]) -> List[str]:
    rows = [(s, stats[s]["cpa"]) for s in catalog_ids if s in stats and stats[s].get("cpa") is not None]
    if len(rows) < 4:
        return []
    rows.sort(key=lambda x: x[1], reverse=True)  # worst CPA first (higher = worse)
    cutoff = max(1, len(rows) // 4)
    return [s for s, _ in rows[:cutoff]]


def reconcile(platform: str) -> Dict[str, Any]:
    """Run the promotion / archive / demote-flag pass for ``platform``."""
    import storage
    from .. import prompt_gen as _pg

    now = datetime.now(timezone.utc).isoformat()
    stats = _per_style_cpa(platform)
    catalog_ids = [s.id for s in _pg.STYLE_CATALOG]
    median_cpa = _catalog_median_cpa(stats, catalog_ids)
    bottom = set(_catalog_bottom_quartile(stats, catalog_ids))

    promoted: List[str] = []
    archived: List[str] = []
    flagged_catalog: List[str] = []

    for cand in storage.list_style_candidates(platform=platform):
        sid = cand.get("style_id") or cand.get("id")
        if not sid:
            continue
        status = cand.get("status") or "candidate"
        if status in ("promoted", "archived"):
            continue
        s = stats.get(sid) or {}
        trials = int(cand.get("trials") or s.get("trials") or 0)
        wins = int(cand.get("wins") or s.get("wins") or 0)
        cpa = s.get("cpa")

        should_archive = False
        should_promote = False

        if trials >= ARCHIVE_NO_WIN_TRIALS and wins == 0:
            should_archive = True
        elif trials >= ARCHIVE_MIN_TRIALS and median_cpa is not None and cpa is not None:
            if cpa > median_cpa * (1 + PROMOTE_CPA_EDGE):
                should_archive = True
        if (
            trials >= PROMOTE_MIN_TRIALS
            and median_cpa is not None
            and cpa is not None
            and cpa <= median_cpa * (1 - PROMOTE_CPA_EDGE)
        ):
            should_promote = True

        if should_promote:
            storage.upsert_style_candidate(
                {"style_id": sid, "status": "promoted", "promoted_at": now},
                platform=platform,
            )
            promoted.append(sid)
        elif should_archive:
            storage.upsert_style_candidate(
                {"style_id": sid, "status": "archived", "archived_at": now},
                platform=platform,
            )
            archived.append(sid)
        else:
            # Keep as testing once we have any trials under its belt.
            if trials > 0 and status == "candidate":
                storage.upsert_style_candidate(
                    {"style_id": sid, "status": "testing"},
                    platform=platform,
                )

    # Flag underperforming catalog styles (no auto-remove).
    for sid in catalog_ids:
        flag = sid in bottom
        entry = storage.upsert_style_candidate(
            {
                "style_id": f"catalog:{sid}",
                "name": sid,
                "status": "catalog",
                "demotion_flagged": flag,
                "updated_at": now,
            },
            platform=platform,
        )
        if flag:
            flagged_catalog.append(sid)

    summary = {
        "platform": platform,
        "promoted": promoted,
        "archived": archived,
        "flagged_catalog": flagged_catalog,
        "catalog_median_cpa": median_cpa,
        "reconciled_at": now,
    }
    logger.info("ai_studio.lifecycle.reconcile: %s", summary)
    return summary


__all__ = [
    "reconcile",
    "PROMOTE_MIN_TRIALS",
    "ARCHIVE_MIN_TRIALS",
    "PROMOTE_CPA_EDGE",
]

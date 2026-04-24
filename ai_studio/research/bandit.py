"""Thompson-sampling bandit allocator for ad styles.

Allocates ``n`` style slots across the baseline 10-style catalog and any
active style candidates from :func:`storage.list_style_candidates`. Each
style is sampled from ``Beta(alpha, beta)`` where alpha/beta are updated
by the ``becomes_winner`` outcomes stored on candidate entries.

Guarantees:
  - Catalog floor: within any rolling ``FLOOR_WINDOW`` batches, every
    catalog style gets at least one slot. This prevents starvation of
    proven styles when the bandit gets momentarily biased.
  - Candidate cold-start boost: any candidate with ``trials < COLD_START_TRIALS``
    gets a synthetic ``alpha += 2`` prior to encourage early exploration.
  - Candidate budget cap: total candidate slots never exceed
    ``MAX_CANDIDATE_RATIO * n`` unless the caller overrides via
    ``research_ratio``.

All thresholds are tunable via env vars.
"""
from __future__ import annotations

import logging
import os
import random
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

FLOOR_WINDOW = int(os.environ.get("AD_STUDIO_BANDIT_FLOOR_WINDOW", "30"))
COLD_START_TRIALS = int(os.environ.get("AD_STUDIO_BANDIT_COLD_START", "5"))
MAX_CANDIDATE_RATIO = float(os.environ.get("AD_STUDIO_BANDIT_CANDIDATE_MAX", "0.4"))
CATALOG_PRIOR_ALPHA = float(os.environ.get("AD_STUDIO_BANDIT_CATALOG_ALPHA", "3"))
CATALOG_PRIOR_BETA = float(os.environ.get("AD_STUDIO_BANDIT_CATALOG_BETA", "2"))


def _recent_allocation_counts(platform: str, *, window: int = FLOOR_WINDOW) -> Dict[str, int]:
    """Count how often each style_id appeared in the last ``window`` generations."""
    try:
        import storage
        rows = storage.list_generations(platform=platform, limit=window)
    except Exception:  # noqa: BLE001
        return {}
    counts: Dict[str, int] = {}
    for r in rows:
        for sid in r.get("style_ids") or []:
            if not sid:
                continue
            counts[sid] = counts.get(sid, 0) + 1
    return counts


def _candidate_priors(candidate: Dict[str, Any]) -> tuple:
    trials = int(candidate.get("trials") or 0)
    wins = int(candidate.get("wins") or 0)
    alpha = float(candidate.get("thompson_alpha") or (1 + wins))
    beta = float(candidate.get("thompson_beta") or (1 + max(0, trials - wins)))
    if trials < COLD_START_TRIALS:
        alpha += 2.0
    return alpha, beta


def _catalog_priors(style_id: str, win_count: int = 0, trial_count: int = 0) -> tuple:
    # Catalog styles start with a gently optimistic prior so they dominate
    # cold-start batches. Once real outcomes accrue we use Laplace updates.
    alpha = CATALOG_PRIOR_ALPHA + win_count
    beta = CATALOG_PRIOR_BETA + max(0, trial_count - win_count)
    return alpha, beta


def _sample_beta(alpha: float, beta: float, rng: random.Random) -> float:
    # random.betavariate requires both > 0
    return rng.betavariate(max(alpha, 1e-3), max(beta, 1e-3))


def allocate(
    catalog_styles: Sequence[str],
    candidates: Sequence[Dict[str, Any]],
    *,
    n: int = 10,
    research_ratio: Optional[float] = None,
    platform: Optional[str] = None,
    seed: Optional[int] = None,
    catalog_stats: Optional[Dict[str, Dict[str, int]]] = None,
) -> List[Dict[str, Any]]:
    """Allocate ``n`` slots and return ordered ``[{style_id, is_candidate}, ...]``.

    Parameters
    ----------
    catalog_styles : sequence of str
        Ordered list of catalog style ids (e.g. from ``prompt_gen.STYLE_CATALOG``).
    candidates : sequence of dict
        Active entries from :func:`storage.list_style_candidates`. Only rows
        with ``status`` in ``{None, "candidate", "testing"}`` are considered.
    n : int
        Number of slots to fill.
    research_ratio : float or None
        Explicit override for the candidate budget cap (0..1). ``None`` uses
        :data:`MAX_CANDIDATE_RATIO`.
    platform : str or None
        If supplied, used to read recent generations for the catalog floor.
    seed : int or None
        Deterministic seed for Thompson sampling — useful for tests.
    catalog_stats : dict or None
        Optional ``{style_id: {trials, wins}}`` stats to feed catalog priors.
        When omitted, catalog priors are constant (cold-start dominant).
    """
    rng = random.Random(seed)
    catalog_styles = list(catalog_styles or [])
    active_candidates = [
        c for c in (candidates or [])
        if (c.get("status") in (None, "candidate", "testing"))
        and (c.get("style_id") or c.get("id"))
    ]

    if n <= 0 or not catalog_styles:
        return []

    ratio = float(research_ratio) if research_ratio is not None else MAX_CANDIDATE_RATIO
    ratio = max(0.0, min(1.0, ratio))
    max_candidate_slots = int(round(ratio * n))
    if not active_candidates:
        max_candidate_slots = 0

    # --- Catalog floor pass: ensure every catalog style appears within
    # FLOOR_WINDOW batches. Any style missing from recent history gets a
    # guaranteed slot here.
    recent = _recent_allocation_counts(platform or "", window=FLOOR_WINDOW) if platform else {}
    floor_picks: List[Dict[str, Any]] = []
    floor_budget = max(0, n - max_candidate_slots)
    for sid in catalog_styles:
        if len(floor_picks) >= floor_budget:
            break
        if recent.get(sid, 0) == 0:
            floor_picks.append({"style_id": sid, "is_candidate": False})

    remaining = n - len(floor_picks)
    if remaining <= 0:
        return floor_picks[:n]

    # --- Thompson sampling for the rest ----------------------------------
    arms: List[Dict[str, Any]] = []
    for sid in catalog_styles:
        stats = (catalog_stats or {}).get(sid) or {}
        a, b = _catalog_priors(
            sid, win_count=int(stats.get("wins") or 0), trial_count=int(stats.get("trials") or 0)
        )
        arms.append({"style_id": sid, "is_candidate": False, "alpha": a, "beta": b})
    for c in active_candidates:
        a, b = _candidate_priors(c)
        arms.append(
            {
                "style_id": c.get("style_id") or c.get("id"),
                "is_candidate": True,
                "alpha": a,
                "beta": b,
            }
        )

    candidate_slots_used = 0
    picks: List[Dict[str, Any]] = list(floor_picks)

    # Sample with replacement — same style can win multiple slots if strong.
    # Enforce candidate budget cap by re-sampling if it would exceed.
    safety = 0
    while len(picks) < n and safety < n * 20:
        safety += 1
        scored = [
            (
                _sample_beta(arm["alpha"], arm["beta"], rng),
                arm,
            )
            for arm in arms
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        for _, arm in scored:
            if arm["is_candidate"] and candidate_slots_used >= max_candidate_slots:
                continue
            picks.append({"style_id": arm["style_id"], "is_candidate": arm["is_candidate"]})
            if arm["is_candidate"]:
                candidate_slots_used += 1
            break
    # Sanity fallback
    while len(picks) < n:
        sid = catalog_styles[len(picks) % len(catalog_styles)]
        picks.append({"style_id": sid, "is_candidate": False})

    return picks[:n]


def record_outcome(
    style_id: str,
    *,
    platform: str,
    won: bool,
    spent: float = 0.0,
    conversions: int = 0,
) -> Optional[Dict[str, Any]]:
    """Increment a candidate's trials/wins counters after a launched ad
    either becomes a winner or drops out of the winners window.

    Catalog styles are tracked implicitly via their performance in
    ``winners.json`` and ``generations.jsonl`` — this helper only updates
    candidate-style rows.
    """
    import storage
    cands = storage.list_style_candidates(platform=platform)
    target: Optional[Dict[str, Any]] = None
    for c in cands:
        if str(c.get("style_id") or c.get("id")) == str(style_id):
            target = c
            break
    if not target:
        return None
    trials = int(target.get("trials") or 0) + 1
    wins = int(target.get("wins") or 0) + (1 if won else 0)
    alpha = float(target.get("thompson_alpha") or 1) + (1 if won else 0)
    beta = float(target.get("thompson_beta") or 1) + (0 if won else 1)
    patch = {
        "style_id": target.get("style_id") or target.get("id"),
        "trials": trials,
        "wins": wins,
        "thompson_alpha": alpha,
        "thompson_beta": beta,
        "spend": float(target.get("spend") or 0) + float(spent or 0),
        "conversions": int(target.get("conversions") or 0) + int(conversions or 0),
    }
    return storage.upsert_style_candidate(patch, platform=platform)


__all__ = [
    "allocate",
    "record_outcome",
    "FLOOR_WINDOW",
    "COLD_START_TRIALS",
    "MAX_CANDIDATE_RATIO",
]

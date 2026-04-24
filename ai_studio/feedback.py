"""Feedback loop — links launched ads back to their generation batch.

When the bulk launcher successfully creates ads from an AI Studio batch it
calls :func:`link_launch` with the resulting ``ad_ids``. The next winners
refresh can then stamp ``becomes_winner=True`` on that generation row
(see :func:`ai_studio.winners._cross_reference_generations`), which in turn
updates bandit priors via :func:`ai_studio.research.bandit.record_outcome`.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

import storage

logger = logging.getLogger(__name__)


def link_launch(
    gen_id: str,
    ad_ids: Iterable[Any],
    *,
    platform: str,
) -> Optional[Dict[str, Any]]:
    """Record the launched ad IDs on a generation row.

    Existing IDs are preserved (we append new ones de-duplicated) so a
    single batch can be used across multiple launches without losing
    provenance.
    """
    gen_id = str(gen_id).strip()
    if not gen_id:
        return None
    new_ids = {str(x) for x in ad_ids if x}
    current = storage.list_generations(platform=platform, limit=5000)
    existing: List[str] = []
    for row in current:
        if str(row.get("gen_id")) == gen_id:
            existing = [str(x) for x in (row.get("launched_ad_ids") or [])]
            break
    merged = list(dict.fromkeys(existing + list(new_ids)))
    return storage.update_generation(
        gen_id,
        {"launched_ad_ids": merged},
        platform=platform,
    )


__all__ = ["link_launch"]

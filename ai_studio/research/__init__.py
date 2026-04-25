"""AI Ad Studio research sub-module.

Discovers, tests, and promotes new ad styles beyond the baseline 10-style
catalog.

Public surface:
  - :func:`ai_studio.research.discover.discover_all`
  - :func:`ai_studio.research.discover_public.discover_all_public`
  - :func:`ai_studio.research.scholar.study_all`
  - :func:`ai_studio.research.bandit.allocate`
  - :func:`ai_studio.research.lifecycle.reconcile`
"""
from __future__ import annotations

from . import bandit, discover, discover_public, lifecycle, scholar, sources  # noqa: F401
from .discover import discover_all  # noqa: F401
from .discover_public import discover_all_public  # noqa: F401
from .scholar import study_all as scholar_study_all  # noqa: F401
from .bandit import allocate  # noqa: F401
from .lifecycle import reconcile  # noqa: F401

__all__ = [
    "bandit",
    "discover",
    "discover_public",
    "scholar",
    "sources",
    "lifecycle",
    "discover_all",
    "discover_all_public",
    "scholar_study_all",
    "allocate",
    "reconcile",
]

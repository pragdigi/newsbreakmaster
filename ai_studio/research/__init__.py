"""AI Ad Studio research sub-module.

Discovers, tests, and promotes new ad styles beyond the baseline 10-style
catalog.

Public surface:
  - :func:`ai_studio.research.discover.discover_all`
  - :func:`ai_studio.research.bandit.allocate`
  - :func:`ai_studio.research.lifecycle.reconcile`
"""
from __future__ import annotations

from . import bandit, discover, lifecycle  # noqa: F401
from .discover import discover_all  # noqa: F401
from .bandit import allocate  # noqa: F401
from .lifecycle import reconcile  # noqa: F401

__all__ = [
    "bandit",
    "discover",
    "lifecycle",
    "discover_all",
    "allocate",
    "reconcile",
]

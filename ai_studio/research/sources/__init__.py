"""Public ad-library source scrapers used by ``discover_public``.

Each source exposes a single ``fetch(query, *, limit, **opts)`` function
that returns a list of normalized ad-card dicts:

    {
      "id":            str,            # source-stable id
      "source":        "meta" | "tiktok" | ...
      "advertiser":    str,
      "headline":      str,
      "body":          str,
      "image_urls":    [str, ...],
      "video_url":     str | None,
      "landing_url":   str | None,
      "started_at":    str | None,     # ISO 8601 if known
      "raw":           dict,           # original payload (small subset only)
    }

These return ``[]`` on any error — they're best-effort scrapers and the
upstream pages change shape frequently. The orchestrator must tolerate
empty results without failing the whole discovery run.
"""

from __future__ import annotations

from . import meta_ad_library, tiktok_creative  # noqa: F401

__all__ = ["meta_ad_library", "tiktok_creative"]

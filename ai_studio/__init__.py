"""AI Ad Studio for newsbreakmaster.

The studio learns from your winning ads, distills their patterns with a
large-language-model analyzer, and generates fresh on-brand image prompts
across proven direct-response styles. Image rendering happens in parallel
through Gemini Nano Banana 2 and OpenAI's ``gpt-image-2`` with cross-model
fallback.

Public entry points:
  - :mod:`ai_studio.winners`    refresh_winners(), is_winner()
  - :mod:`ai_studio.analyzer`   analyze_offer()
  - :mod:`ai_studio.prompt_gen` generate_prompts(), STYLE_CATALOG
  - :mod:`ai_studio.image_gen`  render_batch()
  - :mod:`ai_studio.pipeline`   generate_ads()
  - :mod:`ai_studio.feedback`   link_launch()
  - :mod:`ai_studio.research`   discover_all(), bandit.allocate(), lifecycle.reconcile()
"""
from __future__ import annotations

__all__ = [
    "winners",
    "analyzer",
    "prompt_gen",
    "image_gen",
    "pipeline",
    "feedback",
    "research",
]

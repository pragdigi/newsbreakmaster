"""End-to-end orchestrator for the AI Ad Studio.

``generate_ads`` runs:

    offer → analyzer → bandit allocation → prompt_gen → image_gen → log

and writes an append-only row to ``generations.jsonl`` so the feedback loop
can later stamp ``becomes_winner=True`` once a linked ad wins.

The function is synchronous but image rendering happens in a thread pool
inside :mod:`ai_studio.image_gen`, so a 10-image batch typically returns
within the latency of the slowest model call.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Sequence

import storage

from . import analyzer, concept_gen, image_gen, prompt_gen

logger = logging.getLogger(__name__)


# Platform → target creative aspect ratio. SmartNews v3 accepts square
# (1:1) and 16:9 images; NewsBreak's native feed shows 16:9 landscape.
_PLATFORM_ASPECT = {
    "smartnews": "1:1",
    "newsbreak": "16:9",
}


def _aspect_for_platform(platform: str) -> str:
    return _PLATFORM_ASPECT.get((platform or "").strip().lower(), "1:1")


def _collect_recent_prompts(platform: str, *, limit: int) -> List[str]:
    """Pull the last ``limit`` prompts from the platform's generation log.

    Used as anti-repetition memory for the concept LLM so successive batches
    don't keep producing the same scenes. Pulls a small extra buffer in case
    older rows were image-only (no prompts persisted).
    """
    out: List[str] = []
    try:
        rows = storage.list_generations(platform=platform, limit=max(limit * 2, limit))
    except Exception:  # noqa: BLE001
        return out
    for row in rows[-limit * 2 :]:
        prompts = row.get("prompts") or []
        if isinstance(prompts, list):
            for p in prompts:
                if isinstance(p, str) and p.strip():
                    out.append(p.strip())
    # Keep order, only the most recent ``limit`` items survive.
    if len(out) > limit:
        out = out[-limit:]
    return out


def _load_offer(offer_id: str, *, platform: str) -> Optional[Dict[str, Any]]:
    for o in storage.list_offers(platform=platform):
        if str(o.get("id")) == str(offer_id) or str(o.get("offer_id")) == str(offer_id):
            return o
    return None


def _allocate_styles(
    *,
    platform: str,
    count: int,
    style_mix: Optional[Sequence[str]],
    research_ratio: Optional[float],
) -> List[Dict[str, Any]]:
    """Allocate ``count`` style slots across catalog + candidates.

    Lazy-imports the bandit to avoid a circular dep during early wiring and
    to let the pipeline keep working even if the research module is not yet
    available.
    """
    if style_mix:
        # Explicit override — one slot per listed style.
        return [
            {"style_id": sid, "is_candidate": False}
            for sid in style_mix
        ][:count]

    try:
        from .research import bandit as _bandit  # type: ignore
        from .research import discover as _discover  # noqa: F401
    except Exception:  # noqa: BLE001
        _bandit = None

    catalog_ids = [s.id for s in prompt_gen.STYLE_CATALOG]
    candidates = [
        c for c in storage.list_style_candidates(platform=platform)
        if c.get("status") in (None, "candidate", "testing")
    ]

    if _bandit is not None:
        try:
            return _bandit.allocate(
                catalog_styles=catalog_ids,
                candidates=candidates,
                n=count,
                research_ratio=research_ratio,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("bandit.allocate failed, falling back to catalog-only: %s", exc)

    return [{"style_id": catalog_ids[i % len(catalog_ids)], "is_candidate": False} for i in range(count)]


def generate_ads(
    offer_id: str,
    *,
    platform: str,
    count: int = 10,
    model_image: str = "nano-banana-2",
    model_analyzer: Optional[str] = None,
    style_mix: Optional[Sequence[str]] = None,
    research_ratio: Optional[float] = None,
    fallback: bool = True,
    render: bool = True,
) -> Dict[str, Any]:
    """Produce a batch of on-brand static ad images for ``offer_id``.

    Returns a dict:

        {
          "gen_id": str,
          "offer_id": str,
          "platform": str,
          "prompts": [ {style_id, prompt, ...}, ... ],
          "images":  [ {style_id, b64, mime, model, is_candidate, ...}, ... ],
          "insights": {...},
          "allocation": [ {style_id, is_candidate}, ... ],
        }
    """
    offer = _load_offer(offer_id, platform=platform)
    if offer is None:
        raise ValueError(f"Offer not found: offer_id={offer_id} platform={platform}")

    aspect = _aspect_for_platform(platform)

    insights = analyzer.analyze_offer(
        str(offer_id),
        platform=platform,
        model=model_analyzer,
    )

    allocation = _allocate_styles(
        platform=platform,
        count=count,
        style_mix=style_mix,
        research_ratio=research_ratio,
    )
    if len(allocation) < count:
        catalog_ids = [s.id for s in prompt_gen.STYLE_CATALOG]
        i = 0
        while len(allocation) < count:
            allocation.append(
                {"style_id": catalog_ids[i % len(catalog_ids)], "is_candidate": False}
            )
            i += 1

    catalog_index = {s.id: s for s in prompt_gen.STYLE_CATALOG}
    candidates_index = {
        c.get("style_id") or c.get("id"): c
        for c in storage.list_style_candidates(platform=platform)
    }

    # ------------------------------------------------------------------
    # Step 1: try the LLM concept generator. It writes one fully-formed
    # image prompt per allocated slot, with explicit anti-repetition memory
    # built from the platform's recent generation log. If it succeeds we
    # use those prompts directly; if it fails we fall back to prompt_gen.
    # ------------------------------------------------------------------
    recent_prompts = _collect_recent_prompts(platform, limit=concept_gen.RECENT_PROMPT_MEMORY)
    llm_concepts = concept_gen.generate_concepts(
        offer,
        insights,
        allocation=allocation,
        aspect=aspect,
        recent_prompts=recent_prompts,
        model=model_analyzer,
        platform=platform,
    )

    prompts: List[Dict[str, Any]] = []
    for i, slot in enumerate(allocation):
        sid = slot.get("style_id")
        is_candidate = bool(slot.get("is_candidate"))

        if llm_concepts and i < len(llm_concepts):
            base = dict(llm_concepts[i])
            # Candidate-slot prompt template overrides take precedence over
            # the LLM concept (operator opted into a hand-tuned template).
            if is_candidate and sid in candidates_index:
                cand = candidates_index[sid]
                tpl = (cand.get("prompt_template") or "").strip()
                if tpl:
                    base["prompt"] = prompt_gen._retune_aspect(tpl, aspect)
                base["style_id"] = sid
                base["style_name"] = cand.get("name") or sid
                base["concept_source"] = "candidate_template"
            base["is_candidate"] = is_candidate
            base["aspect"] = aspect
            prompts.append(base)
            continue

        # Fallback: hardcoded scene templates (only when LLM is unreachable).
        if is_candidate and sid in candidates_index:
            cand = candidates_index[sid]
            base = prompt_gen.generate_prompts(
                offer, insights, count=1, style_mix=[sid if sid in catalog_index else "product_showcase"],
                seed=None, aspect=aspect,
            )[0]
            tpl = (cand.get("prompt_template") or "").strip()
            if tpl:
                base["prompt"] = prompt_gen._retune_aspect(tpl, aspect)
            base["style_id"] = sid
            base["style_name"] = cand.get("name") or sid
            base["is_candidate"] = True
            base["aspect"] = aspect
            base["concept_source"] = "candidate_template"
            prompts.append(base)
        else:
            base = prompt_gen.generate_prompts(
                offer,
                insights,
                count=1,
                style_mix=[sid] if sid in catalog_index else None,
                seed=None,
                aspect=aspect,
            )[0]
            base["is_candidate"] = False
            base["concept_source"] = base.get("concept_source") or "fallback"
            prompts.append(base)

    images: List[Dict[str, Any]] = []
    if render:
        rendered = image_gen.render_batch(
            prompts,
            model=model_image,
            fallback=fallback,
            aspect=aspect,
        )
        for i, row in enumerate(rendered):
            row["is_candidate"] = prompts[i].get("is_candidate", False)
        images = rendered

    gen_id = str(uuid.uuid4())
    storage.append_generation(
        {
            "gen_id": gen_id,
            "offer_id": str(offer_id),
            "platform": platform,
            "aspect": aspect,
            "model_image": model_image,
            "model_analyzer": model_analyzer or analyzer.DEFAULT_ANALYZER_MODEL,
            "prompts": [p["prompt"] for p in prompts],
            "style_ids": [p.get("style_id") for p in prompts],
            "concept_sources": [p.get("concept_source", "fallback") for p in prompts],
            "headlines": [p.get("headline") or p.get("angle") for p in prompts],
            "is_candidate_mask": [bool(p.get("is_candidate")) for p in prompts],
            "image_errors": [img.get("error") for img in images],
            "image_providers": [img.get("model") for img in images],
            "launched_ad_ids": [],
            "becomes_winner": False,
        },
        platform=platform,
    )

    return {
        "gen_id": gen_id,
        "offer_id": str(offer_id),
        "platform": platform,
        "aspect": aspect,
        "prompts": prompts,
        "images": images,
        "insights": insights,
        "allocation": allocation,
    }


__all__ = ["generate_ads"]

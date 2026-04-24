"""Fresh-concept generator for the AI Ad Studio.

The original ``prompt_gen`` module produces prompts from a fixed set of
hand-written scene templates (marble countertop, "Sarah M. with 412 likes",
etc.). After a few batches every "product_showcase" image starts to feel
the same: the only things that change are decor accents, palette, and
camera angle — the underlying *scene* is frozen in code.

This module replaces that with an LLM-driven concept step. For each batch
we ask Gemini 3.1 Pro (or Claude Opus 4.7 as fallback) to write ``count``
*genuinely distinct* ad concepts: a fresh scene description + headline +
CTA per slot. The model also receives a "recent prompts" memory so it can
explicitly avoid repeating ideas from the last few batches.

Output shape per concept (designed to slot directly into ``image_gen``):

    {
      "style_id":     str,
      "style_name":   str,
      "prompt":       str,            # ready-to-render image prompt
      "headline":     str,            # ad copy headline
      "cta_label":    str,
      "cta_color":    str,
      "angle":        str,            # short angle line (for storage/log)
      "aspect":       str,            # "1:1" | "16:9"
      "variation_id": int,            # used as image-model seed
      "concept_source": "llm"|"fallback"
    }

When the LLM is unreachable / returns garbage we fall back to
``prompt_gen.generate_prompts`` so the pipeline never hard-fails.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
from typing import Any, Dict, List, Optional, Sequence

from . import prompt_gen

logger = logging.getLogger(__name__)


DEFAULT_CONCEPT_MODEL = os.environ.get("AD_STUDIO_CONCEPT_MODEL", "gemini-3.1-pro")
GEMINI_CONCEPT_MODEL = os.environ.get(
    "AD_STUDIO_GEMINI_CONCEPT_MODEL", "gemini-3.1-pro-preview"
)
CLAUDE_CONCEPT_MODEL = os.environ.get(
    "AD_STUDIO_CLAUDE_CONCEPT_MODEL", "claude-opus-4-7"
)

# How many of the most recent prompts to feed back as "do not repeat".
RECENT_PROMPT_MEMORY = 30


_STYLE_BRIEFS: Dict[str, str] = {
    "product_showcase": (
        "Editorial top-down photo of the offer's product / ingredient / device on "
        "a clean surface (NOT always marble — vary: wood, linen, ceramic tile, "
        "stone, paper, fabric). Composition shows 2–3 supporting objects that "
        "hint at the angle. Bold sans-serif text overlay carries the headline. "
        "A button at the bottom shows the CTA in white on a coloured pill."
    ),
    "organic_native_photo": (
        "Casual phone-shot scene from someone's daily life — kitchen, desk, "
        "car interior, garden, gym, bathroom counter, notebook. A handwritten "
        "note (sticky note OR napkin OR notebook page OR receipt back) carries "
        "the headline in pen. Looks unposed, like a candid camera-roll photo. "
        "No graphic-design overlays, no logo."
    ),
    "medical_illustration": (
        "Flat or semi-flat vector illustration with a stylised diagram (anatomy, "
        "cellular, mechanical, supply-chain) showing the mechanism named in the "
        "concept. Highlight one component with a contrasting accent colour. "
        "Annotation arrows + minimal labels. Headline as bold display text on a "
        "soft tinted background. Coloured CTA pill at the bottom."
    ),
    "fake_social_proof": (
        "Mocked Facebook-style post with a NEW set of believable commenter "
        "names, ages, copy, and like counts each time. Avoid 'Sarah M. 412 "
        "likes' — use varied first names, last initials, and 3-figure like "
        "counts. Each comment reads like a real reaction to the headline."
    ),
    "bold_text_urgency": (
        "Pure-typography ad on a saturated background (black, deep navy, oxblood, "
        "forest green — vary). Oversize headline as bold display type, an alert "
        "label above (\"WARNING\" / \"NOTICE\" / \"ATTENTION\" / \"HEADS UP\"), "
        "and a chunky CTA pill in a contrasting accent. No imagery, only type."
    ),
    "ugc_selfie": (
        "Selfie-style photo of a real-feeling person mid-moment (kitchen, car, "
        "porch, gym, hallway). Vary age 35–70, ethnicity, gender, outfit, room. "
        "Sincere unposed expression. Caption overlay across the lower third "
        "carries the headline in white with a light drop shadow."
    ),
    "before_after_split": (
        "Two-panel comparison — left = problem state (cooler tones, lower "
        "energy), right = relieved state (warmer tones, lighter mood). Vary "
        "the SUBJECT and the PROBLEM each batch (sleep, joints, energy, focus, "
        "skin, weight). Small BEFORE / AFTER labels and a single centred caption."
    ),
    "news_headline": (
        "Mock health-news article screenshot with a serif headline. Vary the "
        "publication's look (broadsheet, magazine, blog, newsletter). Include a "
        "thumbnail image, byline ('By Dr. ...'), date, and a single small "
        "highlight pull-quote. Layout looks like a real reading view."
    ),
    "text_message": (
        "iMessage / SMS-style screen with a short two-person conversation. Each "
        "concept uses a DIFFERENT pair of personas (friend-to-friend, parent-to-"
        "child, spouses, coworkers, siblings). Use natural varied dialogue. "
        "Final blue bubble carries the headline; link-preview tile follows."
    ),
    "listicle": (
        "Clean infographic with the headline up top and 3–5 numbered steps / "
        "tips / signs. Each step has a small icon (vary across the batch — "
        "circles, squares, hexagons, hand-drawn). Coloured CTA pill at the "
        "bottom. Soft background colour, generous whitespace."
    ),
}


SYSTEM_PROMPT = (
    "You are a senior direct-response creative director. You write image-generation "
    "prompts for static social ads. Your job is to invent distinctly different "
    "scenes for the same offer — never two prompts in one batch should describe "
    "the same setting, props, character, copy, or composition. You have full "
    "creative latitude to invent new angles, scenes, props, characters, and copy "
    "as long as you stay truthful to the offer's category and never fabricate "
    "specific medical, financial, or legal claims."
)


def _format_brief(style_id: str) -> str:
    return _STYLE_BRIEFS.get(style_id, "Editorial direct-response social ad.")


def _format_recent_prompts(recent: Sequence[str], *, limit: int = RECENT_PROMPT_MEMORY) -> str:
    if not recent:
        return "(none)"
    rows: List[str] = []
    for i, p in enumerate(list(recent)[-limit:], 1):
        snippet = (p or "").strip().replace("\n", " ")
        if len(snippet) > 220:
            snippet = snippet[:217] + "..."
        rows.append(f"{i}. {snippet}")
    return "\n".join(rows)


def _build_user_prompt(
    *,
    offer: Dict[str, Any],
    insights: Optional[Dict[str, Any]],
    allocation: Sequence[Dict[str, Any]],
    aspect: str,
    recent_prompts: Sequence[str],
) -> str:
    insights = insights or {}
    angles = insights.get("suggested_angles") or []
    hooks = insights.get("top_hooks") or []
    mechanisms = insights.get("mechanisms") or []
    triggers = insights.get("emotional_triggers") or []

    slots_block = []
    for i, slot in enumerate(allocation, 1):
        sid = (slot.get("style_id") or "product_showcase").strip()
        slots_block.append(
            f"  Slot {i}: style_id=\"{sid}\"\n    brief: {_format_brief(sid)}"
        )
    slots_text = "\n".join(slots_block)

    return f"""Offer: {offer.get('name') or '(unnamed)'}
Brand: {offer.get('brand_name') or '(none)'}
Default headline: {offer.get('headline') or '(none)'}
Default body: {offer.get('body') or '(none)'}
Default CTA: {offer.get('cta') or 'Learn More'}
Landing URL: {offer.get('landing_url') or '(none)'}

Winner-derived insights for this offer:
- top_hooks: {", ".join(map(str, hooks)) or '(none)'}
- mechanisms: {", ".join(map(str, mechanisms)) or '(none)'}
- emotional_triggers: {", ".join(map(str, triggers)) or '(none)'}
- suggested_angles: {", ".join(map(str, angles)) or '(none)'}

Target aspect ratio: {aspect}

Slot allocation (write one concept per slot, in order):
{slots_text}

Prompts emitted in the last {RECENT_PROMPT_MEMORY} batches — DO NOT REPEAT ANY
of these scenes, characters, copy, props, or compositions. Invent fresh ideas:
{_format_recent_prompts(recent_prompts)}

Hard rules:
1. The {len(allocation)} concepts in this batch MUST be visually different from
   each other. No shared scene, character archetype, prop combination, headline
   hook, or colour palette across slots.
2. Vary characters: different ages, genders, ethnicities, body types, settings.
3. Vary scene location: kitchen, car, office, garden, gym, bathroom, porch,
   bedroom, park, waiting room, etc. — pick what fits each style brief.
4. Vary props and accents. No two slots may both use a sticky note, both use
   a marble counter, both use a coffee mug, etc.
5. Vary headline copy across slots — do not reuse the same hook in two slots.
6. The "prompt" field must be a fully formed image-generation prompt, ready
   to send to Gemini Imagen / nano-banana / GPT-image. Describe scene,
   composition, lighting, colour palette, framing, and ALL visible text. End
   each prompt with the appropriate aspect sentinel:
     - "Square format." when aspect=1:1
     - "16:9 landscape format, wide horizontal composition." when aspect=16:9
7. Each prompt: 90–180 words.
8. Headlines: 4–14 words, direct-response voice, no emoji.

Return a SINGLE JSON object with this exact shape and nothing else:

{{
  "concepts": [
    {{
      "style_id":  "<the slot's style_id>",
      "scene":     "<one-sentence scene tagline>",
      "prompt":    "<full image prompt, ending with the aspect sentinel>",
      "headline":  "<4-14 word ad headline>",
      "cta_label": "<short CTA text>",
      "angle":     "<3-8 word angle summary>",
      "palette":   "<short colour-palette hint>"
    }},
    ... ({len(allocation)} concepts total, one per slot, in slot order)
  ]
}}

Output ONLY that JSON object. No prose, no markdown fences."""


# ---------------------------------------------------------------------------
# LLM callers (mirror analyzer.py to keep deps light)
# ---------------------------------------------------------------------------


def _call_gemini(prompt: str, *, api_key: str, model: str) -> Optional[str]:
    import requests

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 1.0,
            "topP": 0.95,
            "responseMimeType": "application/json",
        },
    }
    try:
        r = requests.post(url, json=body, timeout=120)
    except requests.RequestException as e:  # pragma: no cover - network
        logger.warning("concept_gen.gemini: network error %s", e)
        return None
    if r.status_code != 200:
        logger.warning(
            "concept_gen.gemini status=%s body=%s", r.status_code, r.text[:500]
        )
        return None
    data = r.json()
    try:
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts).strip()
    except (KeyError, IndexError, TypeError):
        logger.warning("concept_gen.gemini unexpected shape: %s", str(data)[:300])
        return None


def _call_claude(prompt: str, *, api_key: str, model: str) -> Optional[str]:
    import requests

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": 4096,
        "temperature": 1.0,
        "system": SYSTEM_PROMPT + "\n\nReturn only JSON.",
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        r = requests.post(url, json=body, headers=headers, timeout=120)
    except requests.RequestException as e:  # pragma: no cover - network
        logger.warning("concept_gen.claude: network error %s", e)
        return None
    if r.status_code != 200:
        logger.warning(
            "concept_gen.claude status=%s body=%s", r.status_code, r.text[:500]
        )
        return None
    data = r.json()
    try:
        return "".join(
            b.get("text", "") for b in data["content"] if b.get("type") == "text"
        ).strip()
    except (KeyError, TypeError):
        logger.warning("concept_gen.claude unexpected shape: %s", str(data)[:300])
        return None


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def _ensure_aspect_suffix(prompt: str, aspect: str) -> str:
    return prompt_gen._retune_aspect(prompt or "", aspect)


def _normalize_concept(
    raw: Dict[str, Any],
    *,
    fallback_style_id: str,
    fallback_cta_label: str,
    fallback_cta_color: str,
    aspect: str,
    rng: random.Random,
) -> Dict[str, Any]:
    sid = str(raw.get("style_id") or fallback_style_id).strip() or fallback_style_id
    style_def = prompt_gen.STYLE_BY_ID.get(sid)
    style_name = style_def.name if style_def else sid
    cta_color = (style_def.default_cta_color if style_def else fallback_cta_color) or "#DC2626"

    prompt_text = str(raw.get("prompt") or "").strip()
    if not prompt_text:
        # Caller will detect this and fall back.
        return {}
    prompt_text = _ensure_aspect_suffix(prompt_text, aspect)

    cta_label = str(raw.get("cta_label") or fallback_cta_label or "Learn More").strip()
    headline = str(raw.get("headline") or "").strip()
    angle = str(raw.get("angle") or headline or "").strip()
    palette = str(raw.get("palette") or "").strip()

    return {
        "style_id": sid,
        "style_name": style_name,
        "prompt": prompt_text,
        "cta_label": cta_label,
        "cta_color": cta_color,
        "headline": headline,
        "angle": angle,
        "palette": palette,
        "aspect": aspect,
        "variation_id": rng.randrange(10_000_000),
        "concept_source": "llm",
    }


def generate_concepts(
    offer: Dict[str, Any],
    insights: Optional[Dict[str, Any]],
    *,
    allocation: Sequence[Dict[str, Any]],
    aspect: str = "1:1",
    recent_prompts: Optional[Sequence[str]] = None,
    model: Optional[str] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Ask the configured LLM for ``len(allocation)`` fresh ad concepts.

    Returns ``None`` (not [] ) when the LLM is unavailable or its output
    can't be parsed, so the caller knows to fall back to ``prompt_gen``.
    """
    if not allocation:
        return []

    gemini_key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_GENAI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or ""
    ).strip()
    anthropic_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not gemini_key and not anthropic_key:
        return None

    user_prompt = _build_user_prompt(
        offer=offer,
        insights=insights,
        allocation=allocation,
        aspect=aspect,
        recent_prompts=recent_prompts or [],
    )

    preference = (model or DEFAULT_CONCEPT_MODEL).lower()
    order: List[str] = (
        ["claude", "gemini"] if preference.startswith("claude") else ["gemini", "claude"]
    )

    raw_text: Optional[str] = None
    used_model: Optional[str] = None
    for backend in order:
        if backend == "gemini" and gemini_key:
            raw_text = _call_gemini(user_prompt, api_key=gemini_key, model=GEMINI_CONCEPT_MODEL)
            if raw_text:
                used_model = GEMINI_CONCEPT_MODEL
                break
        if backend == "claude" and anthropic_key:
            raw_text = _call_claude(user_prompt, api_key=anthropic_key, model=CLAUDE_CONCEPT_MODEL)
            if raw_text:
                used_model = CLAUDE_CONCEPT_MODEL
                break

    if not raw_text:
        return None

    parsed = _extract_json(raw_text)
    if not parsed or not isinstance(parsed.get("concepts"), list):
        logger.warning("concept_gen: model %s returned unparsable JSON", used_model)
        return None

    rng = random.Random()
    out: List[Dict[str, Any]] = []
    raw_list = parsed["concepts"]
    for i, slot in enumerate(allocation):
        sid = (slot.get("style_id") or "product_showcase").strip()
        style_def = prompt_gen.STYLE_BY_ID.get(sid)
        fallback_cta_label = style_def.default_cta_label if style_def else "Learn More"
        fallback_cta_color = style_def.default_cta_color if style_def else "#DC2626"
        # If the model returned fewer concepts than slots, recycle by index.
        raw_concept = raw_list[i] if i < len(raw_list) else raw_list[i % len(raw_list)]
        if not isinstance(raw_concept, dict):
            return None
        normalized = _normalize_concept(
            raw_concept,
            fallback_style_id=sid,
            fallback_cta_label=fallback_cta_label,
            fallback_cta_color=fallback_cta_color,
            aspect=aspect,
            rng=rng,
        )
        if not normalized:
            logger.warning(
                "concept_gen: model %s returned empty prompt at slot %d", used_model, i
            )
            return None
        out.append(normalized)

    logger.info(
        "concept_gen: %s emitted %d fresh concepts (aspect=%s, recent=%d)",
        used_model,
        len(out),
        aspect,
        len(recent_prompts or []),
    )
    return out


__all__ = ["generate_concepts", "RECENT_PROMPT_MEMORY"]

"""Copywriting Scholar — deep-research agent for ad ideation.

This agent doesn't scrape anything. It *reasons* about ad concepts
through the lens of established direct-response copywriting frameworks
and persuasion psychology, then emits new style candidates that drop
into the same catalog the bandit rotates through.

Each run picks ONE framework lens for ONE offer (LLM-randomised) so the
catalog gets diverse, framework-grounded ideas over time instead of the
same generic "edgy hook" pattern. Frameworks rotate via the persisted
``last_lens_used`` field on a per-offer key in ``research_runs.jsonl``.

Models: Claude Opus 4.7 primary (sharpest persuasion writing), Gemini
3.1 Pro fallback. Configurable via ``AD_STUDIO_SCHOLAR_*`` env vars.
"""
from __future__ import annotations

import logging
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import storage

from . import discover as _disc

logger = logging.getLogger(__name__)


SCHOLAR_PRIMARY_MODEL = os.environ.get("AD_STUDIO_SCHOLAR_PRIMARY", "claude-opus-4-7")
SCHOLAR_FALLBACK_MODEL = os.environ.get("AD_STUDIO_SCHOLAR_FALLBACK", "gemini-3.1-pro-preview")
SCHOLAR_MAX_TOKENS = int(os.environ.get("AD_STUDIO_SCHOLAR_MAX_TOKENS", "4000"))
SCHOLAR_DEFAULT_CONCEPT_COUNT = int(os.environ.get("AD_STUDIO_SCHOLAR_COUNT", "3"))


# ----------------------------------------------------------------------
# Frameworks the scholar rotates through
# ----------------------------------------------------------------------


class Lens:
    """A copywriting framework the scholar can apply to an offer."""

    __slots__ = ("id", "name", "kind", "system_brief", "scaffold")

    def __init__(self, id: str, name: str, kind: str, system_brief: str, scaffold: str):
        self.id = id
        self.name = name
        self.kind = kind
        self.system_brief = system_brief
        self.scaffold = scaffold


LENSES: List[Lens] = [
    Lens(
        id="schwartz_awareness",
        name="Eugene Schwartz — 5 Levels of Awareness",
        kind="framework",
        system_brief=(
            "Eugene Schwartz argued every prospect occupies one of five awareness "
            "levels: (1) Most Aware — knows your product, just needs the offer; "
            "(2) Product Aware — knows your product exists but not why it's right; "
            "(3) Solution Aware — knows the kind of solution but not yours; "
            "(4) Problem Aware — feels the problem, hasn't searched yet; "
            "(5) Unaware — doesn't even consciously have the problem framed. "
            "Cold ad traffic is almost always (4) or (5). Headlines for those "
            "audiences must START in the audience's existing world: a moment, "
            "a frustration, a curiosity gap — NOT a benefit list. Pair this with "
            "Schwartz's '5 Stages of Sophistication' (the market has heard it all; "
            "claim novelty / mechanism / unique angle to break through)."
        ),
        scaffold=(
            "Pick ONE awareness level (4 or 5) and ONE sophistication stage (3-5). "
            "Engineer the visual + headline so a prospect at that level *recognises "
            "themselves* in the first half-second of seeing the ad."
        ),
    ),
    Lens(
        id="cialdini_persuasion",
        name="Cialdini — 7 Principles of Influence",
        kind="framework",
        system_brief=(
            "Robert Cialdini's seven principles of influence: Reciprocity, "
            "Commitment & Consistency, Social Proof, Authority, Liking, Scarcity, "
            "Unity. Direct-response ads typically over-rely on Scarcity and "
            "Social Proof and under-use Authority and Unity. The strongest cold "
            "ads layer two or three of these without making any of them obvious."
        ),
        scaffold=(
            "Pick TWO complementary principles (e.g. Authority + Unity, or "
            "Social Proof + Liking) and design a single static ad that lands "
            "both within ~2 seconds of viewing. Be specific about how the "
            "image carries each principle visually."
        ),
    ),
    Lens(
        id="halbert_kennedy",
        name="Halbert / Kennedy — Direct-Response Pattern Library",
        kind="framework",
        system_brief=(
            "Gary Halbert and Dan Kennedy's school of direct-response: "
            "  • Open with a 'reason why' that sounds like a story, not a pitch. "
            "  • Pair a startling claim with concrete proof (testimonial, "
            "    numbered evidence, before/after). "
            "  • Use 'specificity beats hyperbole' — '37 lbs in 4 months' beats "
            "    'lose weight fast'. "
            "  • Close with an irresistible offer + risk reversal."
        ),
        scaffold=(
            "Compose ONE static ad scene that reads like a Halbert lead: a "
            "specific, almost over-told moment that pulls the eye, with the "
            "headline doing the 'reason why' lift. The visual must feel "
            "candid and reportorial, not graphic-design'd."
        ),
    ),
    Lens(
        id="pas_aida_4cs",
        name="PAS / AIDA / 4Cs / Before-After-Bridge",
        kind="framework",
        system_brief=(
            "Workhorse copy frameworks: "
            "  • PAS — Problem, Agitate, Solution. "
            "  • AIDA — Attention, Interest, Desire, Action. "
            "  • 4Cs — Clear, Concise, Compelling, Credible. "
            "  • Before / After / Bridge — show the present pain, the desired "
            "    future, and the bridge that gets there. "
            "Static ads usually only have room for ONE turn of the framework — "
            "pick the punchiest beat and lean into it."
        ),
        scaffold=(
            "Pick ONE of these four frameworks and execute its single most "
            "potent beat in a static ad. Be explicit which beat you're "
            "compressing into the visual + headline."
        ),
    ),
    Lens(
        id="sugarman_slippery",
        name="Joe Sugarman — Slippery Slide & Curiosity Gaps",
        kind="framework",
        system_brief=(
            "Joe Sugarman's central rule: 'The sole purpose of the first sentence "
            "is to get them to read the second.' Build curiosity gaps that the "
            "reader can't help but want closed: incomplete claims, partial "
            "reveals, unexpected pairings, almost-too-personal moments. The "
            "static ad's headline should function like Sugarman's first sentence "
            "— it doesn't sell the product, it sells the click."
        ),
        scaffold=(
            "Engineer a curiosity-gap headline + visual that makes the user "
            "*need* to know what comes next. The visual must hint at, but not "
            "complete, the reveal."
        ),
    ),
    Lens(
        id="ogilvy_long_copy",
        name="Ogilvy — Long-Copy Hooks Compressed to Static",
        kind="framework",
        system_brief=(
            "Ogilvy taught that great long-copy ads earn the read with three "
            "moves: a specific concrete promise, an unexpected detail that "
            "creates trust, and a 'reason why' that proves the promise isn't "
            "puffery. Compressing this to a static ad: the visual must be the "
            "'unexpected detail' (the thing that makes a sceptic pause), the "
            "headline carries the specific promise, and a tiny supporting line "
            "delivers the reason why."
        ),
        scaffold=(
            "Compose a 3-element static ad: specific-promise headline, "
            "unexpected-detail image, and a single reason-why line under "
            "the headline. Keep the design editorial, NOT graphic-design'd."
        ),
    ),
    Lens(
        id="contrarian_proof",
        name="Contrarian Claim + Concrete Proof",
        kind="psychology",
        system_brief=(
            "Pattern from health / supplement / DTC creative analysis: ads that "
            "declare a contrarian claim ('Doctors got this wrong for 30 years') "
            "and immediately back it with a specific, almost-too-detailed piece "
            "of proof outperform safer 'benefit list' creative by 2-4x. The "
            "claim must be precise (numbered, dated, named) and the proof must "
            "be visualisable in a single static frame."
        ),
        scaffold=(
            "Write ONE contrarian claim + ONE piece of concrete proof. The "
            "static ad must feature the proof — a chart, a side-by-side, a "
            "photographed object, an annotated diagram — as the dominant visual."
        ),
    ),
    Lens(
        id="curiosity_native",
        name="Native-Feed Curiosity (NewsBreak / Local-News Voice)",
        kind="psychology",
        system_brief=(
            "Local-news / native-feed surfaces (NewsBreak, MSN, Outbrain, "
            "Taboola) reward ads that read as organic content. The most "
            "scroll-stopping native ads use: "
            "  • Local specificity ('Florida woman', 'Phoenix retiree') "
            "  • Reportorial framing (matter-of-fact, not enthusiastic) "
            "  • A small candid photo, not a polished studio shot "
            "  • A serif headline, news metadata strip, and reading-time "
            "    estimate as visual chrome."
        ),
        scaffold=(
            "Design a static ad that *looks* like a local-news article card. "
            "It should be indistinguishable at a glance from organic feed "
            "content. Specify the local geography, the byline voice, and the "
            "visual chrome (timestamp, share icon, etc.)."
        ),
    ),
    Lens(
        id="pattern_disrupt_visual",
        name="Pattern-Disrupt Visual",
        kind="visual",
        system_brief=(
            "Direct-response ad analysis shows the strongest cold-traffic "
            "creatives violate one of three feed conventions: scale "
            "(unexpectedly large or small object), proximity (something "
            "uncomfortably close to the camera), or context (an everyday "
            "object in a strange place). The pattern-disrupt isn't gimmicky "
            "— it's specific to the offer's mechanism."
        ),
        scaffold=(
            "Pick the SINGLE pattern-disrupt that fits this offer's mechanism "
            "and write the ad around it. The disrupt must visualise the "
            "mechanism, not just be 'weird for weird's sake'."
        ),
    ),
    Lens(
        id="emotion_specificity",
        name="Specific-Emotion Hook (vs Generic 'Pain')",
        kind="psychology",
        system_brief=(
            "Most direct-response ads name emotions generically ('frustrated', "
            "'tired'). Top creatives instead name a *specific moment* that "
            "evokes the emotion — '3am, lying awake counting ceiling tiles', "
            "'leaving the gym in tears at 6am'. The reader feels recognised "
            "instead of pitched."
        ),
        scaffold=(
            "Pick ONE specific moment that the offer's audience has lived. "
            "The visual must put the viewer inside that moment; the headline "
            "must name it with surgical precision."
        ),
    ),
]


# ----------------------------------------------------------------------
# Lens selection (anti-repetition by offer)
# ----------------------------------------------------------------------


def _recent_lens_ids_for_offer(platform: str, offer_id: str, *, window: int = 8) -> List[str]:
    """Pull the last N lenses applied to this offer from research_runs.jsonl."""
    try:
        rows = storage.list_research_runs(platform=platform, limit=window * 4)
    except Exception:  # noqa: BLE001
        return []
    out: List[str] = []
    for r in rows:
        if r.get("mode") != "scholar":
            continue
        if str(r.get("offer_id") or "") != str(offer_id):
            continue
        lens_id = (r.get("inputs") or {}).get("lens_id")
        if isinstance(lens_id, str):
            out.append(lens_id)
        if len(out) >= window:
            break
    return out


def _pick_lens(platform: str, offer_id: str, *, rng: Optional[random.Random] = None) -> Lens:
    rng = rng or random.Random()
    used = set(_recent_lens_ids_for_offer(platform, offer_id))
    fresh = [l for l in LENSES if l.id not in used]
    pool = fresh if fresh else LENSES
    return rng.choice(pool)


# ----------------------------------------------------------------------
# LLM call
# ----------------------------------------------------------------------


_SCHOLAR_USER_TEMPLATE = """Offer:
  name: {name}
  brand: {brand}
  headline: {headline}
  body: {body}
  category hints: {categories}

Existing catalog styles (do NOT duplicate, but you may evolve):
{catalog_ids}

Recently used Scholar lenses for this offer (avoid repeating their angles):
{recent_lens_ids}

Active lens for this run:
  id:   {lens_id}
  name: {lens_name}
  kind: {lens_kind}

Lens system brief:
{lens_brief}

Scaffold for THIS run (must follow):
{lens_scaffold}

Task:
Produce {count} brand-new static-ad STYLE CANDIDATES for this offer, all written
through the lens above. Each candidate must be a *style*, not a copy variant —
i.e. a reusable visual + headline framework that could be re-used for several
different headlines and CTAs over time.

For each candidate, write:
  • name              short snake_case-friendly name
  • description       1 sentence, what makes this style distinctive
  • visual_cues       3-6 concrete visual elements
  • prompt_template   reusable image prompt with {{headline}}, {{cta_label}},
                      {{brand_name}} placeholders. Must end with "Square format."
  • framework_note    one sentence linking THIS candidate to the lens above
  • copy_seed         a sample headline that demonstrates the style

Return STRICT JSON, no commentary:

[
  {{
    "name": str,
    "description": str,
    "visual_cues": [str],
    "prompt_template": str,
    "framework_note": str,
    "copy_seed": str
  }}
]
"""


def _build_user_prompt(
    *,
    offer: Dict[str, Any],
    lens: Lens,
    count: int,
    recent_lens_ids: Sequence[str],
) -> str:
    cats = offer.get("categories") or []
    if isinstance(cats, list):
        cats_text = ", ".join(str(c) for c in cats) or "(none)"
    else:
        cats_text = str(cats) or "(none)"
    return _SCHOLAR_USER_TEMPLATE.format(
        name=offer.get("name") or "(unnamed)",
        brand=offer.get("brand_name") or offer.get("name") or "(unknown)",
        headline=offer.get("headline") or "",
        body=(offer.get("body") or offer.get("description") or "")[:1000],
        categories=cats_text,
        catalog_ids=", ".join(_disc.CATALOG_IDS),
        recent_lens_ids=", ".join(recent_lens_ids) or "(none)",
        lens_id=lens.id,
        lens_name=lens.name,
        lens_kind=lens.kind,
        lens_brief=lens.system_brief,
        lens_scaffold=lens.scaffold,
        count=count,
    )


_SCHOLAR_SYSTEM_PROMPT = (
    "You are the Copywriting Scholar — a deep-research agent embedded in an "
    "AI Ad Studio. You design static-ad STYLE CANDIDATES for direct-response "
    "marketers, drawing on classical copywriting frameworks (Schwartz, "
    "Halbert, Kennedy, Sugarman, Ogilvy), persuasion psychology (Cialdini, "
    "specificity-of-emotion research), and pattern-disrupt visual theory. "
    "Every output must be: (a) practically renderable as a static image with "
    "an image-generation model, (b) on-brand for the offer's audience and "
    "mechanism, (c) NOT a near-duplicate of the existing catalog. You always "
    "respond with strict JSON, no preamble."
)


def _call_claude(prompt: str, *, timeout: int = 120) -> str:
    if not _disc.ANTHROPIC_API_KEY:
        return ""
    import requests

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": _disc.ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": SCHOLAR_PRIMARY_MODEL,
                "max_tokens": SCHOLAR_MAX_TOKENS,
                "system": _SCHOLAR_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("scholar: claude transport error: %s", exc)
        return ""
    if resp.status_code >= 400:
        logger.warning(
            "scholar: claude %s — %s",
            resp.status_code,
            resp.text[:400],
        )
        return ""
    try:
        data = resp.json()
        for b in data.get("content") or []:
            if b.get("type") == "text":
                return b.get("text") or ""
    except Exception:  # noqa: BLE001
        pass
    return ""


def _call_gemini(prompt: str, *, timeout: int = 120) -> str:
    if not _disc.GEMINI_API_KEY:
        return ""
    import requests

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{SCHOLAR_FALLBACK_MODEL}:generateContent"
    )
    body = {
        "systemInstruction": {"parts": [{"text": _SCHOLAR_SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.85, "maxOutputTokens": SCHOLAR_MAX_TOKENS},
    }
    try:
        resp = requests.post(
            url, params={"key": _disc.GEMINI_API_KEY}, json=body, timeout=timeout
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("scholar: gemini transport error: %s", exc)
        return ""
    if resp.status_code >= 400:
        logger.warning("scholar: gemini %s — %s", resp.status_code, resp.text[:400])
        return ""
    try:
        data = resp.json()
        return (data["candidates"][0]["content"]["parts"][0].get("text")) or ""
    except Exception:  # noqa: BLE001
        return ""


def _scholar_call(prompt: str) -> str:
    """Opus first, Gemini fallback. Returns raw text or ''."""
    text = _call_claude(prompt)
    if text:
        return text
    return _call_gemini(prompt)


# ----------------------------------------------------------------------
# Public entry points
# ----------------------------------------------------------------------


def study_offer(
    offer: Dict[str, Any],
    *,
    platform: str,
    count: int = SCHOLAR_DEFAULT_CONCEPT_COUNT,
    lens_id: Optional[str] = None,
    seed: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Lens]:
    """Run the Scholar against ``offer`` for ``platform``.

    Returns ``(emitted_candidates, lens_used)``. Persists candidates via
    ``storage.upsert_style_candidate`` and logs the run to research_runs.
    """
    rng = random.Random(seed)
    lens = next((l for l in LENSES if l.id == lens_id), None) if lens_id else None
    if lens is None:
        lens = _pick_lens(platform, str(offer.get("id") or ""), rng=rng)

    recent = _recent_lens_ids_for_offer(platform, str(offer.get("id") or ""))
    user_prompt = _build_user_prompt(
        offer=offer,
        lens=lens,
        count=max(1, int(count)),
        recent_lens_ids=recent,
    )
    raw = _scholar_call(user_prompt)
    parsed = _disc._extract_json(raw) or []
    if isinstance(parsed, dict):
        parsed = parsed.get("candidates") or parsed.get("styles") or []
    if not isinstance(parsed, list):
        parsed = []

    emitted: List[Dict[str, Any]] = []
    for item in parsed[: max(1, int(count))]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        cand = _disc._base_candidate(
            style_id=name,
            name=name,
            description=str(item.get("description") or "").strip(),
            visual_cues=[str(v) for v in (item.get("visual_cues") or [])],
            prompt_template=str(item.get("prompt_template") or "").strip(),
            source="scholar",
            source_meta={
                "offer_id": offer.get("id"),
                "lens_id": lens.id,
                "lens_name": lens.name,
                "lens_kind": lens.kind,
                "framework_note": str(item.get("framework_note") or "").strip(),
                "copy_seed": str(item.get("copy_seed") or "").strip(),
                "model_primary": SCHOLAR_PRIMARY_MODEL,
                "model_fallback": SCHOLAR_FALLBACK_MODEL,
            },
        )
        saved = storage.upsert_style_candidate(cand, platform=platform)
        emitted.append(saved)

    _disc._log_run(
        mode="scholar",
        platform=platform,
        offer_id=str(offer.get("id") or ""),
        inputs={
            "lens_id": lens.id,
            "lens_name": lens.name,
            "count": int(count),
            "model_primary": SCHOLAR_PRIMARY_MODEL,
            "model_fallback": SCHOLAR_FALLBACK_MODEL,
        },
        candidates=emitted,
        raw=raw,
    )
    return emitted, lens


def study_all(
    platform: str,
    *,
    offer_id: Optional[str] = None,
    scan_all_offers: bool = False,
    count_per_offer: int = SCHOLAR_DEFAULT_CONCEPT_COUNT,
) -> Dict[str, List[Dict[str, Any]]]:
    """Apply the Scholar to one offer (``offer_id``) or every saved offer."""
    out: Dict[str, List[Dict[str, Any]]] = {"scholar": []}

    offers: List[Dict[str, Any]] = []
    if offer_id:
        for o in storage.list_offers(platform=platform):
            if str(o.get("id")) == str(offer_id):
                offers = [o]
                break
    elif scan_all_offers:
        offers = list(storage.list_offers(platform=platform) or [])

    if not offers:
        logger.info("scholar.study_all: no matching offers for platform=%s", platform)
        return out

    for o in offers:
        try:
            emitted, lens = study_offer(o, platform=platform, count=count_per_offer)
            logger.info(
                "scholar: offer=%s lens=%s emitted=%s",
                o.get("id"),
                lens.id,
                len(emitted),
            )
            out["scholar"].extend(emitted)
        except Exception as exc:  # noqa: BLE001
            logger.warning("scholar.study_offer failed for offer=%s: %s", o.get("id"), exc)
    return out


__all__ = ["study_offer", "study_all", "LENSES", "Lens"]

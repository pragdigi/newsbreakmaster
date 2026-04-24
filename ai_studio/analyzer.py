"""AI offer analyzer: distill winning ads into a structured ``AdInsights`` JSON.

Primary model: Gemini 3.1 Pro (Google AI Studio REST API).
Fallback:       Claude Opus 4.7 (Anthropic Messages API).
Last resort:    Heuristic baseline that mines headline/description text from
                winners without any LLM — guarantees ``analyze_offer`` never
                hard-fails when API keys are missing.

Output shape (matches the plan's :class:`AdInsights` spec):

    {
      "offer_id": str,
      "generated_at": iso8601,
      "model": str,
      "top_hooks": [str],
      "emotional_triggers": [str],
      "mechanisms": [str],
      "winning_style_mix": {style_id: weight, ...},   # 0..1 weights
      "suggested_angles": [str],
      "raw_llm_output": str | None,
    }
"""
from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import storage

logger = logging.getLogger(__name__)

DEFAULT_ANALYZER_MODEL = os.environ.get("AD_STUDIO_ANALYZER_MODEL", "gemini-3.1-pro")
GEMINI_ANALYZER_MODEL = os.environ.get("AD_STUDIO_GEMINI_MODEL", "gemini-3.1-pro-preview")
CLAUDE_ANALYZER_MODEL = os.environ.get("AD_STUDIO_CLAUDE_MODEL", "claude-opus-4-7")

# Shared with prompt_gen — keep the ten canonical styles here so the LLM
# learns to classify weights in terms our prompt generator understands.
CATALOG_STYLE_IDS = [
    "product_showcase",
    "organic_native_photo",
    "medical_illustration",
    "fake_social_proof",
    "bold_text_urgency",
    "ugc_selfie",
    "before_after_split",
    "news_headline",
    "text_message",
    "listicle",
]

SYSTEM_PROMPT = (
    "You are a direct-response ad strategist. You analyse winning ads for a single "
    "offer, extract repeatable patterns, and emit a compact JSON digest that will "
    "steer the next batch of AI-generated static ad creatives. Be specific, "
    "concrete, and actionable. Do not invent facts about the offer."
)

INSTRUCTION_TEMPLATE = """Offer: {offer_name}
Brand: {brand_name}
Landing URL: {landing_url}
Headline template: {headline}
Body template: {body}
CTA: {cta}
Target CPA: ${target_cpa}
Payout: ${payout}

Winning ads for this offer (top {winner_count} by score):
{winners_block}

Baseline winners across the whole account (last 20):
{baseline_block}

Catalog of ad styles you may cite in ``winning_style_mix`` (pick the top 3–5):
{style_catalog}

Return a single JSON object with EXACTLY these keys and nothing else:
  top_hooks:           3–6 short strings — the recurring attention-grabbing claims
  emotional_triggers:  3–5 short strings (e.g. fear, relief, urgency, curiosity)
  mechanisms:          3–5 short strings — the method / ingredient / device / "trick"
                        named in the creative
  winning_style_mix:   object mapping style_id → weight 0..1 (weights must sum to 1.0)
  suggested_angles:    5–8 fresh angle headlines we should test next — each 4–12 words,
                        direct-response voice, no emoji

Only output valid JSON, no prose."""


def _format_winners_block(rows: List[Dict[str, Any]], *, max_n: int = 15) -> str:
    out_lines: List[str] = []
    for i, w in enumerate(rows[:max_n], 1):
        metrics = w.get("metrics") or {}
        out_lines.append(
            f"{i}. [CPA ${metrics.get('cpa') or '—'} · conv {int(metrics.get('conversions') or 0)} · "
            f"spend ${metrics.get('spend') or 0} · score {w.get('score') or 0}]"
        )
        out_lines.append(
            f"   headline: {(w.get('headline') or '').strip()[:140]}"
        )
        if w.get("description"):
            out_lines.append(
                f"   body: {(w.get('description') or '').strip()[:220]}"
            )
        if w.get("sponsored_name"):
            out_lines.append(f"   brand: {w['sponsored_name']}")
    return "\n".join(out_lines) or "(none)"


def _format_style_catalog() -> str:
    return ", ".join(CATALOG_STYLE_IDS)


def _default_empty_insights(offer_id: str) -> Dict[str, Any]:
    return {
        "offer_id": str(offer_id),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "heuristic",
        "top_hooks": [],
        "emotional_triggers": [],
        "mechanisms": [],
        "winning_style_mix": {},
        "suggested_angles": [],
        "raw_llm_output": None,
    }


def _heuristic_insights(
    offer: Dict[str, Any], winners: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Zero-LLM fallback — pull the most common n-grams from winning headlines."""
    text = " ".join(
        [
            f"{w.get('headline') or ''} {w.get('description') or ''}"
            for w in winners
        ]
    ).lower()
    words = re.findall(r"[a-z][a-z'-]{3,}", text)
    stop = {
        "this","that","with","from","your","will","have","about","been","they",
        "them","were","what","when","where","which","just","more","than","into",
        "their","there","these","those","here","very","like","only","even","some",
        "most","also","many","much","most","other","while","because","after","before",
    }
    filtered = [w for w in words if w not in stop]
    top = [w for w, _ in Counter(filtered).most_common(6)]

    triggers: List[str] = []
    if "warning" in filtered or "alert" in filtered:
        triggers.append("fear/urgency")
    if any(k in filtered for k in ("simple","easy","minutes")):
        triggers.append("relief/simplicity")
    if any(k in filtered for k in ("discovered","trick","secret")):
        triggers.append("curiosity")
    if any(k in filtered for k in ("doctor","study","science")):
        triggers.append("authority/trust")
    if not triggers:
        triggers = ["curiosity"]

    mix_count = min(4, max(2, len(winners) // 3 or 1))
    winning_mix = {sid: round(1.0 / mix_count, 3) for sid in CATALOG_STYLE_IDS[:mix_count]}

    out = _default_empty_insights(offer.get("id") or "")
    out.update(
        {
            "top_hooks": top,
            "emotional_triggers": triggers,
            "mechanisms": top[:3],
            "winning_style_mix": winning_mix,
            "suggested_angles": [
                (offer.get("headline") or offer.get("name") or "Learn the simple method")
            ],
            "model": "heuristic",
        }
    )
    return out


# ---------------------------------------------------------------------------
# LLM callers
# ---------------------------------------------------------------------------


def _call_gemini(prompt: str, *, api_key: str, model: str) -> Optional[str]:
    """Direct REST call to the Gemini generateContent endpoint."""
    import requests

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.35,
            "responseMimeType": "application/json",
        },
    }
    try:
        r = requests.post(url, json=body, timeout=120)
    except requests.RequestException as e:  # pragma: no cover - network
        logger.warning("gemini analyze: network error %s", e)
        return None
    if r.status_code != 200:
        logger.warning("gemini analyze status=%s body=%s", r.status_code, r.text[:500])
        return None
    data = r.json()
    try:
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts).strip()
    except (KeyError, IndexError, TypeError):
        logger.warning("gemini analyze unexpected shape: %s", str(data)[:300])
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
        "max_tokens": 2048,
        "temperature": 0.35,
        "system": SYSTEM_PROMPT + "\n\nReturn only JSON.",
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        r = requests.post(url, json=body, headers=headers, timeout=120)
    except requests.RequestException as e:  # pragma: no cover - network
        logger.warning("claude analyze: network error %s", e)
        return None
    if r.status_code != 200:
        logger.warning("claude analyze status=%s body=%s", r.status_code, r.text[:500])
        return None
    data = r.json()
    try:
        return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text").strip()
    except (KeyError, TypeError):
        logger.warning("claude analyze unexpected shape: %s", str(data)[:300])
        return None


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    # Strip ```json fences if any.
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


def _normalize_insights(parsed: Dict[str, Any], *, offer_id: str, model: str, raw: Optional[str]) -> Dict[str, Any]:
    out = _default_empty_insights(offer_id)
    out["model"] = model
    out["raw_llm_output"] = raw

    def _list_of_str(v: Any, limit: int = 8) -> List[str]:
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()][:limit]
        if isinstance(v, str):
            return [x.strip() for x in re.split(r"[\n,;]+", v) if x.strip()][:limit]
        return []

    out["top_hooks"] = _list_of_str(parsed.get("top_hooks"), 6)
    out["emotional_triggers"] = _list_of_str(parsed.get("emotional_triggers"), 5)
    out["mechanisms"] = _list_of_str(parsed.get("mechanisms"), 5)
    out["suggested_angles"] = _list_of_str(parsed.get("suggested_angles"), 8)

    mix = parsed.get("winning_style_mix") or {}
    if isinstance(mix, dict):
        filtered: Dict[str, float] = {}
        for k, v in mix.items():
            key = str(k).strip()
            try:
                w = float(v)
            except (TypeError, ValueError):
                continue
            if not key or w < 0:
                continue
            filtered[key] = w
        total = sum(filtered.values()) or 1.0
        out["winning_style_mix"] = {k: round(w / total, 4) for k, w in filtered.items()}
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_offer(
    offer_id: str,
    *,
    platform: str,
    model: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Build and persist an :class:`AdInsights` digest for one offer.

    ``model`` picks a preferred backend (``gemini-3.1-pro`` | ``claude-opus-4-7``).
    When ``force=False`` and a cached digest exists, returns the cache as-is.
    """
    existing = storage.load_insights(offer_id, platform=platform)
    if existing and not force:
        return existing

    offer = next(
        (o for o in storage.list_offers(platform=platform) if str(o.get("id")) == str(offer_id)),
        None,
    )
    if not offer:
        # Graceful degradation: still emit something so the UI renders.
        out = _default_empty_insights(offer_id)
        out["raw_llm_output"] = f"no offer with id={offer_id}"
        storage.save_insights(offer_id, out, platform=platform)
        return out

    # Read winners across ALL platforms so SmartNews + NewsBreak performance
    # feed the same analyzer. Writes remain platform-scoped; only reads merge.
    all_winners = storage.list_all_winners()
    offer_winners = [w for w in all_winners if str(w.get("offer_id")) == str(offer_id) and w.get("proven")]
    offer_winners.sort(key=lambda w: w.get("score") or 0, reverse=True)

    baseline_winners = [w for w in all_winners if w.get("proven")]
    baseline_winners.sort(key=lambda w: w.get("score") or 0, reverse=True)
    baseline_winners = baseline_winners[:20]

    if not offer_winners and not baseline_winners:
        # No data to learn from; emit empty digest.
        out = _default_empty_insights(offer_id)
        storage.save_insights(offer_id, out, platform=platform)
        return out

    prompt = INSTRUCTION_TEMPLATE.format(
        offer_name=offer.get("name") or "",
        brand_name=offer.get("brand_name") or "",
        landing_url=offer.get("landing_url") or "",
        headline=offer.get("headline") or "",
        body=offer.get("body") or "",
        cta=offer.get("cta") or "Learn More",
        target_cpa=offer.get("target_cpa") if offer.get("target_cpa") is not None else "—",
        payout=offer.get("payout") if offer.get("payout") is not None else "—",
        winner_count=len(offer_winners),
        winners_block=_format_winners_block(offer_winners),
        baseline_block=_format_winners_block(baseline_winners),
        style_catalog=_format_style_catalog(),
    )

    preference = (model or DEFAULT_ANALYZER_MODEL).lower()
    gemini_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_GENAI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    anthropic_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()

    order: List[str] = []
    if preference.startswith("claude"):
        order = ["claude", "gemini"]
    else:
        order = ["gemini", "claude"]

    raw_text: Optional[str] = None
    used_model: Optional[str] = None
    for backend in order:
        if backend == "gemini" and gemini_key:
            raw_text = _call_gemini(prompt, api_key=gemini_key, model=GEMINI_ANALYZER_MODEL)
            if raw_text:
                used_model = GEMINI_ANALYZER_MODEL
                break
        if backend == "claude" and anthropic_key:
            raw_text = _call_claude(prompt, api_key=anthropic_key, model=CLAUDE_ANALYZER_MODEL)
            if raw_text:
                used_model = CLAUDE_ANALYZER_MODEL
                break

    if not raw_text:
        # Last-resort heuristic.
        out = _heuristic_insights(offer, offer_winners or baseline_winners)
        out["offer_id"] = str(offer_id)
        storage.save_insights(offer_id, out, platform=platform)
        return out

    parsed = _extract_json(raw_text)
    if not parsed:
        logger.warning("analyzer: failed to parse JSON from %s output", used_model)
        out = _heuristic_insights(offer, offer_winners or baseline_winners)
        out["model"] = f"{used_model}+fallback-heuristic"
        out["raw_llm_output"] = raw_text[:4000]
        storage.save_insights(offer_id, out, platform=platform)
        return out

    normalized = _normalize_insights(parsed, offer_id=offer_id, model=used_model or "unknown", raw=raw_text[:4000])
    storage.save_insights(offer_id, normalized, platform=platform)
    return normalized


__all__ = ["analyze_offer", "CATALOG_STYLE_IDS"]

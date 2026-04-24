"""Image rendering — parallel Nano Banana 2 + gpt-image-2 with fallback.

Design goals:
  - Direct Python HTTP calls (no Node bridge, no SDK lock-in).
  - Parallel dispatch via ``concurrent.futures.ThreadPoolExecutor``.
  - Cross-model fallback: if Nano Banana fails (no image, rate limit, 5xx),
    retry on OpenAI gpt-image, and vice-versa. Configurable per call.
  - Returns a list of ``{style_id, prompt, b64, mime, model, ms, error?}``
    dicts with base64 payloads the caller can persist or stream to the UI.

Env:
  - ``GEMINI_API_KEY`` — same key shared with the landscape outpainter.
  - ``OPENAI_API_KEY`` — for ``gpt-image-2``.
  - ``NANO_BANANA_MODEL`` — default ``gemini-3.1-flash-image-preview``.
  - ``GPT_IMAGE_MODEL`` — default ``gpt-image-2``.
  - ``AD_STUDIO_IMAGE_PARALLEL`` — max workers (default 5).
  - ``AD_STUDIO_IMAGE_TIMEOUT`` — per-request timeout seconds (default 180).
"""
from __future__ import annotations

import base64
import concurrent.futures
import logging
import os
import random
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

NANO_BANANA_MODEL = os.environ.get("NANO_BANANA_MODEL", "gemini-3.1-flash-image-preview")
GPT_IMAGE_MODEL = os.environ.get("GPT_IMAGE_MODEL", "gpt-image-2")
# Keep parallelism conservative on smaller plans; a generate batch of 10
# at parallel=5 with ~1MB base64 strings can OOM a 512MB Starter worker.
DEFAULT_PARALLEL = int(os.environ.get("AD_STUDIO_IMAGE_PARALLEL", "3"))
# Per-request timeout — kept short so cross-model fallback can actually run
# before gunicorn's worker timeout kills the request.
DEFAULT_TIMEOUT = int(os.environ.get("AD_STUDIO_IMAGE_TIMEOUT", "90"))

# Public model aliases accepted by render_batch(model=...).
ALIAS_NANO = {"nano-banana-2", "nano-banana", "gemini", "gemini-image"}
ALIAS_GPT = {"gpt-image-2", "gpt-image", "openai", "openai-image"}

# Aspect-ratio → provider-specific size settings.
#   Gemini "aspectRatio" accepts "1:1", "16:9", "9:16", "4:3", "3:4".
#   OpenAI Images (gpt-image-*) accepts "1024x1024", "1792x1024", "1024x1792".
_ASPECT_NANO = {
    "1:1": "1:1",
    "16:9": "16:9",
    "9:16": "9:16",
    "4:3": "4:3",
    "3:4": "3:4",
}
_ASPECT_GPT_SIZE = {
    "1:1": "1024x1024",
    "16:9": "1792x1024",
    "9:16": "1024x1792",
    # 4:3 / 3:4 aren't natively supported by gpt-image; bucket to nearest.
    "4:3": "1792x1024",
    "3:4": "1024x1792",
}


def _normalize_aspect(aspect: Optional[str]) -> str:
    a = (aspect or "1:1").strip().lower().replace(" ", "")
    if a in ("landscape", "169"):
        return "16:9"
    if a in ("square", "11"):
        return "1:1"
    return a if a in _ASPECT_NANO else "1:1"


class ImageGenerationError(RuntimeError):
    """Raised when both primary and fallback providers fail."""


# ----------------------------------------------------------------------
# Provider calls
# ----------------------------------------------------------------------

def _call_nano_banana(
    prompt: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    aspect: str = "1:1",
    seed: Optional[int] = None,
    temperature: float = 1.0,
) -> Tuple[str, str]:
    """Call Gemini Flash Image (Nano Banana 2). Returns (b64, mime).

    ``seed`` and ``temperature`` are injected into ``generationConfig`` to
    force variation across batch items that share the same offer/insights.
    When ``seed`` is None a fresh random value is picked per call so
    consecutive Generate clicks don't collapse to the same image.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_GENAI_API_KEY")
    if not api_key:
        raise ImageGenerationError("GEMINI_API_KEY not configured")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{NANO_BANANA_MODEL}:generateContent"
    )
    effective_seed = seed if seed is not None else random.randrange(1, 2**31 - 1)
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "responseModalities": ["IMAGE", "TEXT"],
            "temperature": float(temperature),
            "seed": int(effective_seed),
            "imageConfig": {
                "aspectRatio": _ASPECT_NANO.get(_normalize_aspect(aspect), "1:1"),
                "imageSize": "1K",
            },
        },
    }
    resp = requests.post(
        url,
        params={"key": api_key},
        json=body,
        timeout=timeout,
        headers={"Content-Type": "application/json"},
    )
    if resp.status_code >= 400:
        raise ImageGenerationError(
            f"Nano Banana HTTP {resp.status_code}: {resp.text[:500]}"
        )
    data = resp.json()
    candidates = data.get("candidates") or []
    for cand in candidates:
        for part in (cand.get("content") or {}).get("parts", []):
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                mime = inline.get("mimeType") or inline.get("mime_type") or "image/png"
                return inline["data"], mime
    # No image → capture block reason if present
    block = None
    try:
        block = (
            (data.get("promptFeedback") or {}).get("blockReason")
            or (candidates[0] if candidates else {}).get("finishReason")
        )
    except Exception:
        pass
    raise ImageGenerationError(f"Nano Banana returned no image (reason={block})")


def _call_gpt_image(
    prompt: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    aspect: str = "1:1",
) -> Tuple[str, str]:
    """Call OpenAI Images API (gpt-image-2). Returns (b64, mime)."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ImageGenerationError("OPENAI_API_KEY not configured")

    url = "https://api.openai.com/v1/images/generations"
    body = {
        "model": GPT_IMAGE_MODEL,
        "prompt": prompt,
        "n": 1,
        "size": _ASPECT_GPT_SIZE.get(_normalize_aspect(aspect), "1024x1024"),
    }
    resp = requests.post(
        url,
        json=body,
        timeout=timeout,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    if resp.status_code >= 400:
        raise ImageGenerationError(
            f"gpt-image HTTP {resp.status_code}: {resp.text[:500]}"
        )
    data = resp.json()
    rows = data.get("data") or []
    if not rows:
        raise ImageGenerationError("gpt-image returned no data[]")
    row = rows[0]
    if row.get("b64_json"):
        return row["b64_json"], "image/png"
    if row.get("url"):
        # Fetch the hosted result and re-encode to base64 for consistency.
        fetched = requests.get(row["url"], timeout=timeout)
        if fetched.status_code >= 400:
            raise ImageGenerationError(
                f"gpt-image url fetch failed: {fetched.status_code}"
            )
        b64 = base64.b64encode(fetched.content).decode("ascii")
        mime = fetched.headers.get("content-type", "image/png")
        return b64, mime
    raise ImageGenerationError("gpt-image payload missing b64_json and url")


# ----------------------------------------------------------------------
# Dispatch + fallback
# ----------------------------------------------------------------------

def _model_alias(name: str) -> str:
    name = (name or "").strip().lower()
    if name in ALIAS_NANO:
        return "nano"
    if name in ALIAS_GPT:
        return "gpt"
    # Unknown → default to nano (primary)
    return "nano"


def _render_one(
    prompt: Dict[str, Any],
    *,
    primary: str,
    fallback: bool,
    timeout: int,
    aspect: str = "1:1",
) -> Dict[str, Any]:
    """Render a single prompt dict with cross-model fallback."""
    prompt_text = prompt["prompt"]
    # Per-prompt aspect override (set by prompt_gen) takes precedence.
    effective_aspect = _normalize_aspect(prompt.get("aspect") or aspect)
    providers: List[Tuple[str, Callable[..., Tuple[str, str]]]] = []
    if primary == "nano":
        providers.append(("nano-banana-2", _call_nano_banana))
        if fallback:
            providers.append(("gpt-image-2", _call_gpt_image))
    else:
        providers.append(("gpt-image-2", _call_gpt_image))
        if fallback:
            providers.append(("nano-banana-2", _call_nano_banana))

    # Per-prompt seed so each batch item varies even when multiple prompts
    # share the same style template. Falls back to the prompt's own
    # variation_id when one is provided by prompt_gen.
    prompt_seed = prompt.get("variation_id")
    if not isinstance(prompt_seed, int):
        prompt_seed = random.randrange(1, 2**31 - 1)

    errors: List[str] = []
    for model_name, fn in providers:
        started = time.time()
        try:
            kwargs = {"timeout": timeout, "aspect": effective_aspect}
            if model_name == "nano-banana-2":
                kwargs["seed"] = prompt_seed
                kwargs["temperature"] = 1.0
            b64, mime = fn(prompt_text, **kwargs)
            elapsed_ms = int((time.time() - started) * 1000)
            return {
                "style_id": prompt.get("style_id"),
                "style_name": prompt.get("style_name"),
                "prompt": prompt_text,
                "b64": b64,
                "mime": mime,
                "model": model_name,
                "ms": elapsed_ms,
                "aspect": effective_aspect,
                "cta_label": prompt.get("cta_label"),
                "cta_color": prompt.get("cta_color"),
                "angle": prompt.get("angle"),
            }
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = int((time.time() - started) * 1000)
            errors.append(f"{model_name} [{elapsed_ms}ms] {exc}")
            logger.warning("image_gen: %s failed: %s", model_name, exc)

    return {
        "style_id": prompt.get("style_id"),
        "style_name": prompt.get("style_name"),
        "prompt": prompt_text,
        "b64": None,
        "mime": None,
        "model": None,
        "ms": 0,
        "aspect": effective_aspect,
        "error": "; ".join(errors) or "unknown error",
        "cta_label": prompt.get("cta_label"),
        "cta_color": prompt.get("cta_color"),
        "angle": prompt.get("angle"),
    }


def render_batch(
    prompts: Iterable[Dict[str, Any]],
    *,
    model: str = "nano-banana-2",
    parallel: int = DEFAULT_PARALLEL,
    fallback: bool = True,
    timeout: int = DEFAULT_TIMEOUT,
    aspect: str = "1:1",
) -> List[Dict[str, Any]]:
    """Render a batch of prompts in parallel.

    Parameters
    ----------
    prompts : iterable of dict
        Output of :func:`ai_studio.prompt_gen.generate_prompts`. Must include
        ``prompt`` keys; other fields (style_id, style_name, cta_label,
        cta_color, angle) are propagated into each result row.
    model : str
        Primary provider alias. ``"nano-banana-2"`` or ``"gpt-image-2"``.
    parallel : int
        Max concurrent renders.
    fallback : bool
        When True, retry on the other provider if the primary fails.
    timeout : int
        Per-request timeout in seconds.
    """
    primary = _model_alias(model)
    prompt_list = list(prompts)
    if not prompt_list:
        return []

    workers = max(1, min(parallel, len(prompt_list)))
    results: List[Optional[Dict[str, Any]]] = [None] * len(prompt_list)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _render_one,
                p,
                primary=primary,
                fallback=fallback,
                timeout=timeout,
                aspect=aspect,
            ): i
            for i, p in enumerate(prompt_list)
        }
        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:  # noqa: BLE001
                logger.exception("image_gen: worker crashed")
                results[idx] = {
                    "style_id": prompt_list[idx].get("style_id"),
                    "style_name": prompt_list[idx].get("style_name"),
                    "prompt": prompt_list[idx].get("prompt"),
                    "b64": None,
                    "mime": None,
                    "model": None,
                    "ms": 0,
                    "error": f"worker crashed: {exc}",
                }
    return [r for r in results if r is not None]


__all__ = [
    "NANO_BANANA_MODEL",
    "GPT_IMAGE_MODEL",
    "ImageGenerationError",
    "render_batch",
]

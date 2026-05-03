"""Prebuilt-ad library — pre-render a stash of ads per offer × platform.

The daily scheduler hook calls :func:`topup_all` to walk every saved offer
on every platform, ensure ``AD_STUDIO_LIBRARY_TARGET_PER_OFFER`` ads are
ready, and stash any missing ones to disk. When the operator hits
"Generate" in the UI, the API drains from the library first (FIFO,
oldest images first) and only renders fresh images for the gap. That
makes the "Generate" button feel near-instant on the typical
"give me 10 ads for offer X" workflow.

Storage layout (under ``catalog/<platform>/``)::

    ad_library.jsonl       jsonl row per stashed image (metadata)
    library_images/<id>.<ext>   the actual rendered file

A library row looks like::

    {
      "library_id":  "<uuid>",
      "platform":    "smartnews" | "newsbreak",
      "offer_id":    "<offer.id>",
      "style_id":    "value_stack",
      "style_name":  "Value Stack",
      "is_candidate": false,
      "concept_source": "llm" | "fallback" | "candidate_template",
      "model":       "nano-banana-2",
      "mime":        "image/png",
      "ms":          12345,
      "filename":    "<library_id>.png",
      "headline":    "...",
      "prompt":      "...",
      "aspect":      "1:1" | "16:9",
      "created_at":  "...",
      "consumed_at": null | "...",
      "source_gen_id": "<the gen_id the rendering pipeline assigned>"
    }
"""
from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, List, Optional

import storage

from . import pipeline

logger = logging.getLogger(__name__)


# Default per-offer target. Configurable via env so an operator can tune
# the daily image-API spend without touching code. 10 matches what the
# Generate UI defaults to so a single click usually drains the whole
# stash without any fresh renders.
DEFAULT_TARGET_PER_OFFER = int(os.environ.get("AD_STUDIO_LIBRARY_TARGET_PER_OFFER", "10"))

# Cap on how many fresh renders the topup may issue per offer per call.
# Acts as a circuit breaker so a single bad config (e.g. target=200)
# can't blow through the entire daily image-API budget in one tick.
MAX_TOPUP_BATCH = int(os.environ.get("AD_STUDIO_LIBRARY_MAX_BATCH", "10"))


def _ext_for_mime(mime: str) -> str:
    m = (mime or "").lower()
    if "jpeg" in m or "jpg" in m:
        return "jpg"
    if "webp" in m:
        return "webp"
    return "png"


def topup_offer(
    offer_id: str,
    *,
    platform: str,
    target: Optional[int] = None,
    model_image: str = "nano-banana-2",
) -> Dict[str, Any]:
    """Ensure the library has ``target`` unconsumed items for this offer.

    Returns a small summary dict ``{added, total, errors, target}``. Any
    rendering errors are swallowed (logged) so a bad-prompt slot doesn't
    take down the whole topup; the missing slots will be retried on the
    next tick.
    """
    target = int(target if target is not None else DEFAULT_TARGET_PER_OFFER)
    if target <= 0:
        return {"added": 0, "total": 0, "errors": [], "target": 0}

    counts = storage.library_counts(platform=platform)
    have = int(counts.get(str(offer_id), 0))
    need = max(0, target - have)
    if need <= 0:
        return {"added": 0, "total": have, "errors": [], "target": target}

    need = min(need, MAX_TOPUP_BATCH)

    try:
        batch = pipeline.generate_ads(
            offer_id,
            platform=platform,
            count=need,
            model_image=model_image,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("library.topup_offer: pipeline.generate_ads failed offer=%s platform=%s: %s", offer_id, platform, exc)
        return {"added": 0, "total": have, "errors": [str(exc)], "target": target}

    images = batch.get("images") or []
    prompts = batch.get("prompts") or []
    aspect = batch.get("aspect") or "1:1"
    src_gen_id = batch.get("gen_id")

    added = 0
    errors: List[str] = []
    for i, img in enumerate(images):
        b64 = img.get("b64")
        if not b64:
            errors.append(img.get("error") or f"slot {i}: empty payload")
            continue
        prompt_meta = prompts[i] if i < len(prompts) else {}
        mime = img.get("mime") or "image/png"
        ext = _ext_for_mime(mime)
        # Use append_library_item to allocate the library_id, then write
        # the file under that id. We append BEFORE the file write so the
        # row + file land together; if the file write fails we patch the
        # row to ``consumed_at=<error>`` so it's never served.
        row = storage.append_library_item(
            {
                "offer_id": str(offer_id),
                "style_id": img.get("style_id") or prompt_meta.get("style_id"),
                "style_name": img.get("style_name") or prompt_meta.get("style_name"),
                "is_candidate": bool(img.get("is_candidate")),
                "concept_source": prompt_meta.get("concept_source"),
                "model": img.get("model"),
                "mime": mime,
                "ms": img.get("ms"),
                "headline": prompt_meta.get("headline") or prompt_meta.get("angle"),
                "prompt": prompt_meta.get("prompt"),
                "aspect": aspect,
                "source_gen_id": src_gen_id,
            },
            platform=platform,
        )
        filename = f"{row['library_id']}.{ext}"
        full_path = storage.library_image_path(filename, platform=platform)
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "wb") as f:
                f.write(base64.b64decode(b64))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{row['library_id']}: write failed: {exc}")
            # Burn the row by marking consumed so it doesn't show up.
            storage.consume_library_items(str(offer_id), 1, platform=platform)
            continue
        # Patch the row with the filename so the consumer can serve it.
        # We do a tiny re-read + rewrite; library file stays small.
        _patch_library_row(row["library_id"], {"filename": filename}, platform=platform)
        added += 1

    total = have + added
    logger.info(
        "library.topup_offer offer=%s platform=%s target=%s have=%s added=%s errors=%s",
        offer_id, platform, target, have, added, len(errors),
    )
    return {"added": added, "total": total, "errors": errors, "target": target}


def _patch_library_row(library_id: str, patch: Dict[str, Any], *, platform: str) -> None:
    """Tiny in-place patcher for a single library row. Called rarely, so
    the brute-force rewrite is fine."""
    import json
    import shutil

    path = storage._library_file(platform)
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    changed = False
    for r in rows:
        if str(r.get("library_id")) == str(library_id):
            r.update(patch)
            changed = True
            break
    if not changed:
        return
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    shutil.move(tmp, path)


def topup_all(*, target_per_offer: Optional[int] = None, model_image: str = "nano-banana-2") -> Dict[str, Any]:
    """Walk every offer on every platform and top up its library.

    Each (platform, offer) failure is isolated — we log and keep going so
    one bad offer can't starve the rest. Returns a per-platform summary
    dict the scheduler logs at INFO.
    """
    target = int(target_per_offer if target_per_offer is not None else DEFAULT_TARGET_PER_OFFER)
    summary: Dict[str, Any] = {"target_per_offer": target, "platforms": {}}
    for platform in ("newsbreak", "smartnews"):
        per_platform = {"offers": 0, "added": 0, "skipped": 0, "errors": 0}
        try:
            offers = storage.list_offers(platform=platform) or []
        except Exception as exc:  # noqa: BLE001
            logger.exception("library.topup_all: list_offers failed platform=%s: %s", platform, exc)
            summary["platforms"][platform] = {"error": str(exc)}
            continue
        for offer in offers:
            offer_id = str(offer.get("id") or offer.get("offer_id") or "").strip()
            if not offer_id:
                continue
            per_platform["offers"] += 1
            try:
                res = topup_offer(
                    offer_id,
                    platform=platform,
                    target=target,
                    model_image=model_image,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("library.topup_all: topup_offer failed platform=%s offer=%s: %s", platform, offer_id, exc)
                per_platform["errors"] += 1
                continue
            per_platform["added"] += int(res.get("added") or 0)
            if not res.get("added"):
                per_platform["skipped"] += 1
            if res.get("errors"):
                per_platform["errors"] += len(res["errors"])
        summary["platforms"][platform] = per_platform
        logger.info(
            "library.topup_all platform=%s offers=%s added=%s skipped=%s errors=%s",
            platform,
            per_platform["offers"],
            per_platform["added"],
            per_platform["skipped"],
            per_platform["errors"],
        )
    return summary


__all__ = ["topup_offer", "topup_all", "DEFAULT_TARGET_PER_OFFER"]

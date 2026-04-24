"""
Bulk upload creatives and create campaign / ad sets / ads per grouping strategy.
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from newsbreak_api import NewsBreakClient

logger = logging.getLogger(__name__)


def _log_json(label: str, obj: Any) -> None:
    """Emit a single JSON line to stderr so Render's log viewer can ingest it."""
    try:
        logger.info("%s %s", label, json.dumps(obj, default=str)[:4000])
    except Exception:
        logger.info("%s %r", label, obj)


# NewsBreak's /ad/uploadAssets endpoint silently rejects formats outside
# of {jpg, jpeg, png, gif, mp4, mov, webm} with a generic
# "Fail to create media: unexpected error occurred". Transparently
# convert .webp / .heic / .heif / .avif / .bmp / .tiff images to PNG
# before upload so the user doesn't have to pre-process them.
_IMAGE_EXT_OK = {".jpg", ".jpeg", ".png", ".gif"}
_VIDEO_EXT_OK = {".mp4", ".mov", ".webm", ".m4v"}
_IMAGE_EXT_CONVERT = {".webp", ".heic", ".heif", ".avif", ".bmp", ".tif", ".tiff"}


def _normalize_upload(file_bytes: bytes, filename: str) -> Tuple[bytes, str, bool]:
    """
    Returns (bytes, filename, converted). If the input is an unsupported
    image format, converts to PNG in memory and rewrites the filename.
    Non-image formats and already-supported formats pass through unchanged.
    """
    ext = os.path.splitext(filename or "")[1].lower()
    if ext in _IMAGE_EXT_OK or ext in _VIDEO_EXT_OK:
        return file_bytes, filename, False
    if ext not in _IMAGE_EXT_CONVERT:
        return file_bytes, filename, False
    try:
        from PIL import Image
    except Exception as e:
        logger.warning("pillow import failed, cannot convert %s: %s", ext, e)
        return file_bytes, filename, False
    try:
        with Image.open(io.BytesIO(file_bytes)) as im:
            if im.mode not in ("RGB", "RGBA"):
                im = im.convert("RGBA" if "A" in im.mode else "RGB")
            buf = io.BytesIO()
            im.save(buf, format="PNG", optimize=True)
            new_bytes = buf.getvalue()
        new_filename = (os.path.splitext(filename)[0] or "asset") + ".png"
        logger.info(
            "normalize_upload converted filename=%r (%d bytes) -> %r (%d bytes)",
            filename,
            len(file_bytes),
            new_filename,
            len(new_bytes),
        )
        return new_bytes, new_filename, True
    except Exception as e:
        logger.warning("normalize_upload failed filename=%r err=%s", filename, e)
        return file_bytes, filename, False


def _name_from_filename(filename: str, fallback: str = "") -> str:
    """Strip extension, replace separators with spaces, title-case, cap length.

    e.g. ``summer_promo-01.jpg`` -> ``Summer Promo 01``.
    """
    if not filename:
        return fallback
    stem = os.path.splitext(os.path.basename(filename))[0]
    stem = re.sub(r"[_\-\.]+", " ", stem).strip()
    stem = re.sub(r"\s+", " ", stem)
    if not stem:
        return fallback
    # Preserve existing casing if it already has spaces / mixed case,
    # otherwise title-case to make dashboard names readable.
    if stem == stem.lower():
        stem = stem.title()
    return stem[:120]


def _first_id(obj: Any, *keys: str) -> Optional[str]:
    if not isinstance(obj, dict):
        return None
    for k in keys:
        v = obj.get(k)
        if v is not None:
            return str(v)
    return None


def _extract_id(resp: Any, *keys: str) -> Optional[str]:
    if resp is None:
        return None
    rid = _first_id(resp, *keys) if isinstance(resp, dict) else None
    if rid:
        return rid
    if isinstance(resp, dict):
        data = resp.get("data")
        if isinstance(data, dict):
            return _first_id(data, *keys)
    return None


def group_creatives(
    creatives: List[Dict[str, Any]],
    grouping: str,
    group_size: int = 5,
) -> List[List[Dict[str, Any]]]:
    """
    grouping: all_in_one | isolate | groups_of_n
    """
    if not creatives:
        return []
    if grouping == "all_in_one":
        return [creatives]
    if grouping == "isolate":
        return [[c] for c in creatives]
    # groups_of_n
    n = max(1, group_size)
    return [creatives[i : i + n] for i in range(0, len(creatives), n)]


def bulk_launch(
    client: NewsBreakClient,
    *,
    ad_account_id: str,
    campaign_mode: str,
    campaign_id: Optional[str],
    campaign_payload: Optional[Dict[str, Any]],
    ad_set_base: Dict[str, Any],
    creatives: List[Dict[str, Any]],
    grouping: str,
    group_size: int,
) -> Dict[str, Any]:
    """
    creatives: list of dicts with keys: file_bytes, filename, headline, body, landing_url, media_name (optional)
    campaign_payload: used when campaign_mode == 'new'
    ad_set_base: shared fields for create_ad_set (targeting, schedule, bid, budget, name prefix, etc.)
    """
    result: Dict[str, Any] = {
        "success": True,
        "campaign_id": None,
        "ad_sets": [],
        "errors": [],
    }

    cid = campaign_id
    if campaign_mode == "new" and campaign_payload:
        try:
            cr = client.create_campaign(campaign_payload)
            cid = _extract_id(cr, "id", "campaignId")
            result["campaign_raw"] = cr
        except Exception as e:
            result["success"] = False
            result["errors"].append(f"create_campaign: {e}")
            return result

    if not cid:
        result["success"] = False
        result["errors"].append("No campaign id")
        return result

    result["campaign_id"] = cid

    groups = group_creatives(creatives, grouping, group_size)
    logger.info(
        "bulk_launch.groups grouping=%s group_size=%s creatives=%d groups=%d sizes=%s",
        grouping,
        group_size,
        len(creatives),
        len(groups),
        [len(g) for g in groups],
    )
    if not groups:
        result["success"] = False
        result["errors"].append(
            "No creatives were submitted — the form POST had an empty file list. "
            "Drop at least one image/video in the launch form and try again."
        )
        return result

    base_name = (ad_set_base.get("name_prefix") or "").strip()
    brand_name = ad_set_base.get("_brand_name", "Advertiser")
    cta = ad_set_base.get("_cta", "Learn More")

    def _guess_creative_type(filename: str) -> str:
        f = (filename or "").lower()
        if f.endswith(".gif"):
            return "GIF"
        if f.endswith((".mp4", ".mov", ".webm", ".m4v")):
            return "VIDEO"
        return "IMAGE"

    def _extract_asset_url(resp: Any) -> Optional[str]:
        if not isinstance(resp, dict):
            return None
        for k in ("assetUrl", "url", "cdnUrl"):
            v = resp.get(k)
            if isinstance(v, str) and v:
                return v
        data = resp.get("data")
        if isinstance(data, dict):
            for k in ("assetUrl", "url", "cdnUrl"):
                v = data.get(k)
                if isinstance(v, str) and v:
                    return v
        return None

    def _upload_with_retry(c: Dict[str, Any], max_attempts: int = 3) -> Any:
        """
        NewsBreak's /ad/uploadAssets returns 400 "Fail to create media:
        unexpected error occurred" for both unsupported formats AND
        genuine transient errors. We normalize unsupported image
        formats to PNG up-front (see _normalize_upload) and then retry
        on the same generic error in case it's truly transient.
        Non-transient errors with specific messages re-raise immediately.
        """
        raw_bytes = c.get("file_bytes") or b""
        raw_filename = c.get("filename") or "asset.bin"
        file_bytes, filename, converted = _normalize_upload(raw_bytes, raw_filename)
        media_name = filename if converted else c.get("media_name")
        logger.info(
            "upload_asset.request filename=%r size=%d media_name=%r converted=%s ad_account=%s",
            filename,
            len(file_bytes),
            media_name,
            converted,
            ad_account_id,
        )
        last_err: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                raw = client.upload_asset(
                    file_bytes,
                    filename,
                    ad_account_id,
                    media_name=media_name,
                )
                if attempt > 1:
                    logger.info("upload_asset.retry_ok filename=%r attempt=%d", filename, attempt)
                return raw
            except Exception as e:
                last_err = e
                msg = str(e)
                transient = (
                    "unexpected error occurred" in msg
                    or "502" in msg
                    or "503" in msg
                    or "504" in msg
                    or "timed out" in msg.lower()
                )
                if not transient or attempt == max_attempts:
                    raise
                delay = 0.5 * (2 ** (attempt - 1))
                logger.warning(
                    "upload_asset.retry filename=%r attempt=%d/%d delay=%.1fs err=%s",
                    filename,
                    attempt,
                    max_attempts,
                    delay,
                    msg[:200],
                )
                time.sleep(delay)
        # unreachable — loop either returns or raises
        if last_err:
            raise last_err
        raise RuntimeError("upload_asset: no attempts made")

    for gi, group in enumerate(groups):
        uploaded: List[Dict[str, Any]] = []
        for c in group:
            try:
                raw = _upload_with_retry(c)
                asset_url = _extract_asset_url(raw)
                if not asset_url:
                    result["errors"].append(f"upload_asset missing assetUrl: {raw}")
                    continue
                reused = bool(isinstance(raw, dict) and (raw.get("data") or {}).get("reused"))
                if reused:
                    logger.info(
                        "upload_asset.reused filename=%r asset_url=%s",
                        c.get("filename", "?"),
                        asset_url,
                    )
                uploaded.append(
                    {
                        "asset_url": asset_url,
                        "creative_type": _guess_creative_type(c.get("filename", "")),
                        "filename": c.get("filename", ""),
                        "headline": c.get("headline", ""),
                        "body": c.get("body", ""),
                        "landing_url": c.get("landing_url", ""),
                    }
                )
            except Exception as e:
                fname = c.get("filename", "?")
                size = len(c.get("file_bytes") or b"")
                logger.warning(
                    "upload_asset.error filename=%r size=%d err=%s",
                    fname,
                    size,
                    str(e)[:500],
                )
                result["errors"].append(f"upload {fname} ({size} bytes): {e}")

        if not uploaded:
            continue

        # Derive a human-readable ad-set name:
        #   - user-supplied prefix wins (numbered when >1 group)
        #   - single-creative groups use the creative's filename
        #   - multi-creative groups use "Ad set N" since the first
        #     filename is no longer representative of the whole group
        if base_name:
            ad_set_name = f"{base_name} {gi + 1}" if len(groups) > 1 else base_name
        elif len(group) == 1:
            ad_set_name = _name_from_filename(group[0].get("filename", ""), fallback=f"Ad set {gi + 1}")
        else:
            ad_set_name = f"Ad set {gi + 1}"

        ad_set_payload = {
            **{k: v for k, v in ad_set_base.items() if not k.startswith("_") and k != "name_prefix"},
            "name": ad_set_name,
            "campaignId": cid,
        }

        _log_json("create_ad_set.request", ad_set_payload)
        try:
            aset = client.create_ad_set(ad_set_payload)
            _log_json("create_ad_set.response", aset)
            ad_set_id = _extract_id(aset, "id", "adSetId")
        except Exception as e:
            logger.warning("create_ad_set.error %s", e)
            result["errors"].append(f"create_ad_set: {e}")
            continue

        ads_out = []
        for i, row in enumerate(uploaded):
            creative = {
                "type": row["creative_type"],
                "headline": row["headline"] or ad_set_name,
                "assetUrl": row["asset_url"],
                "description": row["body"] or row["headline"] or ad_set_name,
                "callToAction": cta,
                "brandName": brand_name,
            }
            if row["landing_url"]:
                creative["clickThroughUrl"] = row["landing_url"]
            ad_name = _name_from_filename(
                row.get("filename", ""),
                fallback=f"{ad_set_name} — Ad {i + 1}",
            )
            ad_payload = {
                "adSetId": ad_set_id,
                "name": ad_name,
                "creative": creative,
            }
            _log_json("create_ad.request", ad_payload)
            try:
                ad_resp = client.create_ad(ad_payload)
                _log_json("create_ad.response", ad_resp)
                ads_out.append({"raw": ad_resp, "payload": ad_payload})
            except Exception as e:
                logger.warning("create_ad.error %s", e)
                result["errors"].append(f"create_ad: {e}")

        result["ad_sets"].append({"ad_set_id": ad_set_id, "ads": ads_out})

    if not result["ad_sets"]:
        result["success"] = False
        if not result["errors"]:
            result["errors"].append(
                "Campaign was created but no ad sets were produced. This usually means "
                "every creative upload failed silently; re-check the file types and try again."
            )
    return result


def creative_from_upload(file_storage, headline: str, body: str, landing_url: str) -> Dict[str, Any]:
    """Build creative dict from Werkzeug FileStorage."""
    data = file_storage.read()
    return {
        "file_bytes": data,
        "filename": file_storage.filename or "upload.bin",
        "headline": headline,
        "body": body,
        "landing_url": landing_url,
        "media_name": file_storage.filename,
    }


def creative_from_bytes(
    file_bytes: bytes,
    filename: str,
    headline: str,
    body: str,
    landing_url: str,
) -> Dict[str, Any]:
    return {
        "file_bytes": file_bytes,
        "filename": filename,
        "headline": headline,
        "body": body,
        "landing_url": landing_url,
        "media_name": filename,
    }

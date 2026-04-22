"""
SmartNews bulk launcher.

SmartNews AMv1 hierarchy: Account -> Campaign -> Creative. Each creative needs
an image **triplet** (300x300 "a", 600x500 "b", 1280x720 "c"). The launcher
collects the three uploaded image files per ad, resizes/reformats them into
the required dimensions, uploads each to SmartNews, creates a campaign,
creates one creative per uploaded triplet, then calls ``submit_review`` so
the creatives enter the approval queue.

This module intentionally mirrors the public shape of the NewsBreak launcher
(``bulk_launcher.py``): it is called from ``app.py``'s ``/launch`` route with
``form`` and ``files`` from Flask and a ``triplet_builder`` factory.
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)


_IMAGE_SPECS = {
    "a": (300, 300),
    "b": (600, 500),
    "c": (1280, 720),
}

_SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".gif"}
_CONVERT_EXTS = {".webp", ".heic", ".heif", ".avif", ".bmp", ".tif", ".tiff"}
_MAX_IMAGE_BYTES = 500 * 1024  # SmartNews /images/upload limit


def _log_json(label: str, obj: Any) -> None:
    try:
        logger.info("%s %s", label, json.dumps(obj, default=str)[:4000])
    except Exception:
        logger.info("%s %r", label, obj)


def _load_image(file_bytes: bytes):
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - pillow always installed here
        raise RuntimeError("Pillow is required for SmartNews image preparation") from exc
    return Image, Image.open(io.BytesIO(file_bytes))


def _resize_to(file_bytes: bytes, target_w: int, target_h: int) -> bytes:
    """Resize (cover+crop) an image to exactly (target_w, target_h) and
    encode as JPEG <= 500KB."""
    Image, im = _load_image(file_bytes)
    with im:
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        src_w, src_h = im.size
        target_ratio = target_w / target_h
        src_ratio = src_w / src_h
        if src_ratio > target_ratio:
            new_h = src_h
            new_w = int(src_h * target_ratio)
            left = (src_w - new_w) // 2
            box = (left, 0, left + new_w, src_h)
        else:
            new_w = src_w
            new_h = int(src_w / target_ratio)
            top = (src_h - new_h) // 2
            box = (0, top, src_w, top + new_h)
        cropped = im.crop(box).resize((target_w, target_h), Image.LANCZOS)
        if cropped.mode != "RGB":
            cropped = cropped.convert("RGB")
        quality = 88
        while True:
            buf = io.BytesIO()
            cropped.save(buf, format="JPEG", quality=quality, optimize=True)
            out = buf.getvalue()
            if len(out) <= _MAX_IMAGE_BYTES or quality <= 40:
                return out
            quality -= 8


def _read_upload(f: Any) -> Tuple[bytes, str]:
    data = f.read()
    name = getattr(f, "filename", None) or "image.jpg"
    return data, name


def creative_triplet_from_uploads(
    file_a: Any,
    file_b: Optional[Any] = None,
    file_c: Optional[Any] = None,
) -> Dict[str, Tuple[bytes, str]]:
    """Return three (bytes, filename) pairs, one per required SmartNews image
    size. If only ``file_a`` is provided it will be resized three times."""
    raw_a, fn_a = _read_upload(file_a)
    raw_b, fn_b = (_read_upload(file_b) if file_b else (raw_a, fn_a))
    raw_c, fn_c = (_read_upload(file_c) if file_c else (raw_a, fn_a))
    out: Dict[str, Tuple[bytes, str]] = {}
    for key, (w, h), (data, name) in (
        ("a", _IMAGE_SPECS["a"], (raw_a, fn_a)),
        ("b", _IMAGE_SPECS["b"], (raw_b, fn_b)),
        ("c", _IMAGE_SPECS["c"], (raw_c, fn_c)),
    ):
        resized = _resize_to(data, w, h)
        base = os.path.splitext(os.path.basename(name or "image"))[0]
        out[key] = (resized, f"{base}_{w}x{h}.jpg")
    return out


# ----------------------------------------------------------------------
# Launch flow
# ----------------------------------------------------------------------
def _form_list(form: Mapping[str, Any], key: str) -> List[str]:
    if hasattr(form, "getlist"):
        return [v for v in form.getlist(key) if v]
    val = form.get(key)
    if isinstance(val, list):
        return [v for v in val if v]
    if val is None:
        return []
    return [val]


def _files_for_prefix(files: Mapping[str, Any], prefix: str) -> List[Tuple[str, Any]]:
    """Return list of (ad_index_or_name, file_storage) pairs for a given prefix.

    Accepted keys:
      <prefix>_<n>, <prefix>_<n>_a, <prefix>_<n>_b, <prefix>_<n>_c
    """
    keys = sorted(files.keys())
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)(?:_([abc]))?$")
    bucket: Dict[str, Dict[str, Any]] = {}
    for k in keys:
        m = pat.match(k)
        if not m:
            continue
        idx, sub = m.group(1), m.group(2) or "a"
        f = files.get(k)
        if not f or not getattr(f, "filename", None):
            continue
        bucket.setdefault(idx, {})[sub] = f
    out: List[Tuple[str, Any]] = []
    for idx in sorted(bucket.keys(), key=lambda s: int(s)):
        entry = bucket[idx]
        out.append((idx, entry))
    return out


def _parse_schedule(form: Mapping[str, Any]) -> Tuple[str, str]:
    now = datetime.now(timezone.utc)
    start_raw = (form.get("start_time") or "").strip()
    end_raw = (form.get("end_time") or "").strip()
    start = _parse_iso(start_raw) or now
    end = _parse_iso(end_raw) or (start + timedelta(days=30))
    return _iso(start), _iso(end)


def _parse_iso(s: str) -> Optional[datetime]:
    if not s:
        return None
    s = s.strip()
    try:
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _int_or_none(v: Any) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _int_req(v: Any, *, field: str) -> int:
    got = _int_or_none(v)
    if got is None:
        raise ValueError(f"{field} is required")
    return got


def smartnews_bulk_launch(
    adapter: Any,
    *,
    form: Mapping[str, Any],
    files: Mapping[str, Any],
    triplet_builder=creative_triplet_from_uploads,
) -> Dict[str, Any]:
    """Create a SmartNews campaign + creatives from form data.

    Expected form fields (all values are JPY integers):
      account_id        (required)
      campaign_name     (required)
      sponsored_name    (required)
      action_type       (APP_INSTALL | WEBSITE_CONVERSION, default WEBSITE_CONVERSION)
      bid_amount        (required)
      daily_budget      (required)
      total_budget      (required)
      target_cpa        (optional)
      ad_category_id    (required; e.g. "3")
      start_time        (ISO 8601, optional)
      end_time          (ISO 8601, optional)
      link_url          (required for each ad when actionType=WEBSITE_CONVERSION)
      tracking_url      (optional, per-ad)
      ad_title_<n>, ad_text_<n>, ad_name_<n>
      creative_<n>_(a|b|c)  (uploaded files — a is mandatory)

    Returns a summary dict with ``campaign_id``, ``creatives`` (list of
    {name, creative_id, images}), and any errors encountered.
    """
    errors: List[Dict[str, Any]] = []
    account_id = (form.get("account_id") or "").strip()
    if not account_id:
        return {"ok": False, "error": "account_id is required"}

    creatives_meta = _files_for_prefix(files, "creative")
    if not creatives_meta:
        return {"ok": False, "error": "at least one creative image (creative_0 / creative_0_a) is required"}

    try:
        campaign_payload = _build_campaign_payload(form)
    except ValueError as e:
        return {"ok": False, "error": str(e)}

    try:
        created = adapter.create_campaign(account_id, campaign_payload)
    except Exception as e:
        _log_json("sn_create_campaign_failed", {"error": str(e), "payload": campaign_payload})
        return {"ok": False, "error": f"create_campaign failed: {e}"}

    campaign_id = str(created.get("campaignId") or created.get("id") or "")
    if not campaign_id:
        return {"ok": False, "error": f"campaign create returned no id: {created}"}

    creative_results: List[Dict[str, Any]] = []
    action_type = (form.get("action_type") or "WEBSITE_CONVERSION").upper()
    default_link = (form.get("link_url") or "").strip()
    default_tracking = (form.get("tracking_url") or "").strip()

    for idx, bucket in creatives_meta:
        try:
            triplet = triplet_builder(
                bucket.get("a"),
                bucket.get("b"),
                bucket.get("c"),
            )
        except Exception as e:
            errors.append({"ad": idx, "stage": "image_prep", "error": str(e)})
            continue

        uploaded: Dict[str, Dict[str, Any]] = {}
        try:
            for slot, (data, fname) in triplet.items():
                resp = adapter.upload_asset(account_id, data, fname)
                uploaded[slot] = resp
        except Exception as e:
            errors.append({"ad": idx, "stage": "image_upload", "error": str(e)})
            continue

        title = (form.get(f"ad_title_{idx}") or form.get("ad_title") or "").strip()
        text = (form.get(f"ad_text_{idx}") or form.get("ad_text") or "").strip()
        ad_name = (form.get(f"ad_name_{idx}") or f"Ad {idx}").strip()
        link_url = (form.get(f"link_url_{idx}") or default_link).strip()
        tracking_url = (form.get(f"tracking_url_{idx}") or default_tracking).strip()

        creative_payload: Dict[str, Any] = {
            "name": ad_name,
            "title": title,
            "text": text,
            "imageset": {
                slot: {"imageId": str(uploaded[slot].get("imageId") or uploaded[slot].get("id"))}
                for slot in ("a", "b", "c")
                if slot in uploaded
            },
        }
        if action_type == "APP_INSTALL" or not link_url:
            pass
        if link_url:
            creative_payload["linkUrl"] = link_url
        if tracking_url:
            creative_payload["trackingUrl"] = tracking_url

        try:
            cr = adapter.create_creative(campaign_id, creative_payload)
        except Exception as e:
            errors.append(
                {"ad": idx, "stage": "create_creative", "error": str(e), "payload": creative_payload}
            )
            continue

        creative_results.append(
            {
                "ad": idx,
                "name": ad_name,
                "creative_id": str(cr.get("creativeId") or cr.get("id") or ""),
                "images": {k: v.get("imageId") for k, v in uploaded.items()},
            }
        )

    # Submit the whole campaign for review if any creatives were created.
    submission: Any = None
    if creative_results:
        try:
            submission = adapter.submit_review(campaign_id)
        except Exception as e:
            errors.append({"stage": "submit_review", "error": str(e)})

    return {
        "ok": not errors or bool(creative_results),
        "platform": "smartnews",
        "campaign_id": campaign_id,
        "creatives": creative_results,
        "submit_review": submission,
        "errors": errors,
    }


def _build_campaign_payload(form: Mapping[str, Any]) -> Dict[str, Any]:
    name = (form.get("campaign_name") or "").strip()
    if not name:
        raise ValueError("campaign_name is required")
    sponsored = (form.get("sponsored_name") or "").strip()
    if not sponsored:
        raise ValueError("sponsored_name is required")
    action_type = (form.get("action_type") or "WEBSITE_CONVERSION").strip().upper()
    bid_amount = _int_req(form.get("bid_amount"), field="bid_amount")
    daily_budget = _int_req(form.get("daily_budget"), field="daily_budget")
    total_budget = _int_req(form.get("total_budget"), field="total_budget")
    ad_category_id = (form.get("ad_category_id") or "").strip()
    if not ad_category_id:
        raise ValueError("ad_category_id is required")
    start_time, end_time = _parse_schedule(form)

    payload: Dict[str, Any] = {
        "actionType": action_type,
        "name": name,
        "sponsoredName": sponsored,
        "bidAmount": bid_amount,
        "dailyBudget": daily_budget,
        "totalBudget": total_budget,
        "startTime": start_time,
        "endTime": end_time,
        "adCategoryId": ad_category_id,
    }
    target_cpa = _int_or_none(form.get("target_cpa"))
    if target_cpa is not None:
        payload["targetCpa"] = target_cpa

    publishers = _form_list(form, "publishers")
    devices = _form_list(form, "devices")
    genders = _form_list(form, "genders")
    city_keys = _form_list(form, "cities")
    targeting: Dict[str, Any] = {}
    if publishers:
        targeting["publishers"] = publishers
    if devices:
        targeting["devices"] = devices
    if genders:
        targeting["genders"] = genders
    if city_keys:
        targeting["cities"] = [{"key": k} for k in city_keys]
    if targeting:
        payload["targeting"] = targeting

    if action_type == "APP_INSTALL":
        application = (form.get("app_application") or "").strip()
        if application:
            app_spec: Dict[str, Any] = {"application": application}
            urlscheme = (form.get("app_urlscheme") or "").strip()
            if urlscheme:
                app_spec["urlscheme"] = urlscheme
            payload["appSpec"] = app_spec
        tracking_type = (form.get("tracking_type") or "").strip()
        if tracking_type:
            payload["trackingSpec"] = {"trackingType": tracking_type}

    return payload


__all__ = [
    "creative_triplet_from_uploads",
    "smartnews_bulk_launch",
]

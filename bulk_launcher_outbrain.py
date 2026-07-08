"""
Outbrain (Amplify) bulk launcher.

Hierarchy: Marketer → Budget + Campaign → PromotedLink. The launcher:

  1. Regenerates each uploaded creative to the chosen Outbrain/Teads format
     using Gemini "nano banana" (same approach as the SmartNews launcher):
       - ``1:1``  → 1200x1200 (Smartfeed / Carousel card)
       - ``16:9`` → 1200x675  (wide / Teads native)
     1:1 needs no AI (the source is already square — just cover-resize);
     16:9 is outpainted from the square so nothing is cropped.
  2. Creates a Budget under the marketer (MONTHLY+runForever for evergreen, or
     a CAMPAIGN budget bounded by the campaign end date).
  3. Creates one Campaign referencing that budget.
  4. Hosts each prepared image at a public URL (Outbrain fetches + caches it)
     and creates one PromotedLink per ad via the JSON ``imageUrl`` contract.

Image bytes are NOT uploaded to Outbrain directly — the Amelia-proxied token
500s on the multipart byte-upload endpoint, but the documented ``imageUrl``
path (Outbrain fetches a public URL) works reliably. The caller supplies a
``host_image(bytes, filename) -> public_url`` callback.

Expected form fields (see ``templates/launch.html`` outbrain block):

    account_id          required  (marketer id)
    campaign_mode       new|existing            (default new)
    campaign_id         required when mode=existing
    campaign_name       required when mode=new
    objective           default Traffic
    budget_amount_usd   required when mode=new   (total budget)
    daily_target_usd    optional                 (pacing daily cap)
    cpc_usd             default 0.30             (max CPC bid)
    start_time/end_time ISO 8601 (optional)
    creative_format     "1:1" | "16:9"           (default 1:1)
    platforms[]         DESKTOP|MOBILE|TABLET     (default all three)
    locations[]         Outbrain location ids     (default United States)
    language            default en
    suffix_tracking_code optional UTM string
    landing_page_url    default landing for every ad
    headline_<n>        required per ad (<=100 chars)
    landing_page_url_<n> per-ad landing override
    cta_label_<n>       default LEARN_MORE
    creative_<n>        file field, source image (required)
"""
from __future__ import annotations

import io
import logging
import os
import re
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

# US country id (from /locations/search?term=United States) — the safe default.
_DEFAULT_LOCATION_US = "fc4deb5112fb4415a9edacdf4aafb0d8"

# Outbrain still-image specs (max 2.5 MB, JPG/PNG).
_MAX_IMAGE_BYTES = 2 * 1024 * 1024 + 512 * 1024  # 2.5 MiB
_FORMAT_DIMS: Dict[str, Tuple[int, int]] = {
    "1:1": (1200, 1200),
    "16:9": (1200, 675),
}

# Outbrain call-to-action enum (subset; NONE means no button).
_CTA_VALUES = {
    "NONE", "READ_MORE", "LEARN_MORE", "SHOP_NOW", "SIGN_UP", "DOWNLOAD",
    "BOOK_NOW", "APPLY_NOW", "GET_QUOTE", "SUBSCRIBE", "GET_OFFER",
    "REGISTER", "WATCH_MORE", "TRY_NOW", "ORDER_NOW", "DONATE", "VISIT_SITE",
}

_OBJECTIVE_VALUES = {
    "Traffic", "Conversions", "Awareness", "AppInstall", "LeadGeneration",
    "Engagement", "VideoViews",
}


# ----------------------------------------------------------------------
# Image pipeline (nano banana regenerate to Outbrain/Teads format)
# ----------------------------------------------------------------------
def _load_image(file_bytes: bytes):
    from PIL import Image, ImageFilter

    return Image, ImageFilter, Image.open(io.BytesIO(file_bytes))


def _encode_jpeg(im, *, max_bytes: int = _MAX_IMAGE_BYTES) -> bytes:
    if im.mode != "RGB":
        im = im.convert("RGB")
    quality = 90
    while True:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        out = buf.getvalue()
        if len(out) <= max_bytes or quality <= 40:
            return out
        quality -= 8


def _resize_cover(file_bytes: bytes, target_w: int, target_h: int) -> bytes:
    Image, _ImageFilter, im = _load_image(file_bytes)
    with im:
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        sw, sh = im.size
        tr, sr = target_w / target_h, sw / sh
        if sr > tr:
            new_w = int(sh * tr)
            left = (sw - new_w) // 2
            box = (left, 0, left + new_w, sh)
        else:
            new_h = int(sw / tr)
            top = (sh - new_h) // 2
            box = (0, top, sw, top + new_h)
        out = im.crop(box).resize((target_w, target_h), Image.LANCZOS)
        return _encode_jpeg(out)


def _local_blur_fill(square_bytes: bytes, *, target: Tuple[int, int]) -> bytes:
    """Deterministic fallback: center the source on a blurred cover background."""
    Image, ImageFilter, im = _load_image(square_bytes)
    with im:
        tw, th = target
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        bg_src = im.copy()
        sw, sh = bg_src.size
        tr, sr = tw / th, sw / sh
        if sr > tr:
            new_w = int(sh * tr)
            left = (sw - new_w) // 2
            bg = bg_src.crop((left, 0, left + new_w, sh))
        else:
            new_h = int(sw / tr)
            top = (sh - new_h) // 2
            bg = bg_src.crop((0, top, sw, top + new_h))
        bg = bg.resize((tw, th), Image.LANCZOS).filter(ImageFilter.GaussianBlur(radius=18))
        fg_h = min(th, im.size[1])
        scale = fg_h / im.size[1]
        fg_w = int(im.size[0] * scale)
        fg = im.resize((fg_w, fg_h), Image.LANCZOS)
        x = (tw - fg_w) // 2
        y = (th - fg_h) // 2
        canvas = bg.copy()
        canvas.paste(fg, (x, y))
        return _encode_jpeg(canvas)


_GEMINI_IMAGE_MODELS = (
    "gemini-3.1-flash-image-preview",
    "gemini-2.5-flash-image-preview",
)
_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
_GEMINI_TIMEOUT_S = 90
_WIDE_PROMPT = (
    "You are an expert creative director. Extend this square advertising image "
    "horizontally into a 16:9 landscape (1200x675) creative. Naturally continue "
    "the scene on the LEFT and RIGHT edges so the original subject, text, and "
    "typography stay exactly as-is and fully centered. Match the existing "
    "lighting, colors, textures, grain, and background seamlessly. Do NOT crop, "
    "warp, rotate, re-color, re-type, or add new text, logos, watermarks, "
    "borders, or foreground subjects. The result must read as one cohesive photo."
)


def _gemini_api_key() -> Optional[str]:
    return (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GOOGLE_GENAI_API_KEY")
    )


def _gemini_outpaint_wide(square_bytes: bytes, *, target: Tuple[int, int], api_key: str) -> Optional[bytes]:
    import base64

    try:
        import requests
    except Exception:  # pragma: no cover
        return None

    b64 = base64.b64encode(square_bytes).decode("ascii")
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": _WIDE_PROMPT},
                    {"inlineData": {"mimeType": "image/jpeg", "data": b64}},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.4,
            "responseModalities": ["IMAGE"],
            "imageConfig": {"aspectRatio": "16:9"},
        },
    }
    for model in _GEMINI_IMAGE_MODELS:
        url = f"{_GEMINI_BASE}/{model}:generateContent?key={api_key}"
        try:
            resp = requests.post(url, json=body, timeout=_GEMINI_TIMEOUT_S)
        except Exception as exc:
            logger.warning("Outbrain gemini outpaint failed on %s: %s", model, exc)
            continue
        if resp.status_code == 404:
            continue
        if resp.status_code != 200:
            logger.warning("Outbrain gemini HTTP %s on %s: %s", resp.status_code, model, resp.text[:300])
            continue
        try:
            data = resp.json()
        except Exception:
            continue
        img_bytes = None
        for cand in data.get("candidates") or []:
            for part in ((cand.get("content") or {}).get("parts") or []):
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    try:
                        img_bytes = base64.b64decode(inline["data"])
                    except Exception:
                        img_bytes = None
                    if img_bytes:
                        break
            if img_bytes:
                break
        if not img_bytes:
            continue
        # Normalize to exact target resolution + JPEG under the size cap.
        Image, _ImageFilter, im = _load_image(img_bytes)
        with im:
            tw, th = target
            if im.mode != "RGB":
                im = im.convert("RGB")
            sw, sh = im.size
            tr, sr = tw / th, sw / sh
            if sr > tr:
                new_w = int(sh * tr)
                left = (sw - new_w) // 2
                im = im.crop((left, 0, left + new_w, sh))
            elif sr < tr:
                new_h = int(sw / tr)
                top = (sh - new_h) // 2
                im = im.crop((0, top, sw, top + new_h))
            im = im.resize((tw, th), Image.LANCZOS)
            return _encode_jpeg(im)
    return None


def prepare_creative(file_obj: Any, *, fmt: str = "1:1") -> Tuple[bytes, str]:
    """Return ``(jpeg_bytes, filename)`` for the chosen Outbrain format.

    ``1:1`` cover-resizes the source square to 1200x1200. ``16:9`` outpaints
    the square to 1200x675 with nano banana (local blur-fill fallback).
    """
    data = file_obj.read() if hasattr(file_obj, "read") else bytes(file_obj)
    filename = getattr(file_obj, "filename", None) or "creative.jpg"
    base = os.path.splitext(os.path.basename(filename))[0] or "creative"
    fmt = (fmt or "1:1").strip()
    target = _FORMAT_DIMS.get(fmt, _FORMAT_DIMS["1:1"])

    square = _resize_cover(data, 1200, 1200)
    if fmt == "16:9":
        out = None
        key = _gemini_api_key()
        if key:
            try:
                out = _gemini_outpaint_wide(square, target=target, api_key=key)
            except Exception as exc:  # pragma: no cover
                logger.warning("Outbrain outpaint raised; using blur fill: %s", exc)
        if not out:
            out = _local_blur_fill(square, target=target)
        return out, f"{base}_1200x675.jpg"
    return square, f"{base}_1200x1200.jpg"


# ----------------------------------------------------------------------
# Form helpers
# ----------------------------------------------------------------------
def _form_list(form: Mapping[str, Any], key: str) -> List[str]:
    if hasattr(form, "getlist"):
        return [v for v in form.getlist(key) if v]
    val = form.get(key)
    if isinstance(val, list):
        return [v for v in val if v]
    return [val] if val else []


def _files_for_prefix(files: Mapping[str, Any], prefix: str) -> List[Tuple[str, Any]]:
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    pairs: List[Tuple[int, Any]] = []
    for k in sorted(files.keys()):
        m = pat.match(k)
        if not m:
            continue
        f = files.get(k)
        if not f or not getattr(f, "filename", None):
            continue
        pairs.append((int(m.group(1)), f))
    pairs.sort(key=lambda t: t[0])
    return [(str(i), f) for i, f in pairs]


def _usd(v: Any) -> Optional[float]:
    if v in (None, ""):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f >= 0 else None


def _parse_iso(s: str) -> Optional[datetime]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _norm_cta(v: Any, default: str = "LEARN_MORE") -> Optional[str]:
    cta = (str(v or "").strip().upper()) or default
    if cta == "NONE":
        return None
    return cta if cta in _CTA_VALUES else default


# ----------------------------------------------------------------------
# Payload builders
# ----------------------------------------------------------------------
def _build_budget_payload(form: Mapping[str, Any], *, campaign_name: str) -> Dict[str, Any]:
    amount = _usd(form.get("budget_amount_usd")) or _usd(form.get("budget_usd"))
    if amount is None:
        raise ValueError("budget_amount_usd is required")
    start = _parse_iso(form.get("start_time")) or datetime.now(timezone.utc)
    end = _parse_iso(form.get("end_time"))
    # Unique budget name (Outbrain rejects duplicates per marketer).
    name = f"{campaign_name} budget {uuid.uuid4().hex[:8]}"
    payload: Dict[str, Any] = {
        "name": name[:100],
        "amount": round(amount, 2),
        "pacing": (form.get("budget_pacing") or "AUTOMATIC").strip().upper(),
        "startDate": start.date().strftime("%Y-%m-%d"),
    }
    daily = _usd(form.get("daily_target_usd"))
    if daily is not None:
        payload["dailyTarget"] = round(daily, 2)
    if end is not None:
        payload["type"] = "CAMPAIGN"
        payload["endDate"] = end.date().strftime("%Y-%m-%d")
        payload["runForever"] = False
    else:
        # Evergreen: monthly budget that runs forever.
        payload["type"] = "MONTHLY"
        payload["runForever"] = True
    return payload


def _build_campaign_payload(form: Mapping[str, Any], *, budget_id: str) -> Dict[str, Any]:
    name = (form.get("campaign_name") or "").strip()
    if not name:
        raise ValueError("campaign_name is required")
    cpc = _usd(form.get("cpc_usd"))
    if cpc is None:
        cpc = 0.30
    objective = (form.get("objective") or "Traffic").strip()
    if objective not in _OBJECTIVE_VALUES:
        objective = "Traffic"

    platforms = [p for p in _form_list(form, "platforms") if p in ("DESKTOP", "MOBILE", "TABLET")]
    if not platforms:
        platforms = ["DESKTOP", "MOBILE", "TABLET"]
    locations = _form_list(form, "locations") or [_DEFAULT_LOCATION_US]
    language = (form.get("language") or "en").strip() or "en"

    targeting: Dict[str, Any] = {
        "platform": platforms,
        "locations": locations,
        "language": language,
    }
    payload: Dict[str, Any] = {
        "name": name,
        "budgetId": budget_id,
        "cpc": round(float(cpc), 2),
        "enabled": False,  # always create paused; operator enables after review
        "objective": objective,
        "targeting": targeting,
    }
    suffix = (form.get("suffix_tracking_code") or "").strip()
    if suffix:
        payload["suffixTrackingCode"] = suffix
    return payload


# ----------------------------------------------------------------------
# Launch flow
# ----------------------------------------------------------------------
def outbrain_bulk_launch(
    adapter: Any,
    *,
    form: Mapping[str, Any],
    files: Mapping[str, Any],
    host_image: Callable[[bytes, str], str],
    creative_builder: Callable[..., Tuple[bytes, str]] = prepare_creative,
    log_progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    def _progress(msg: str) -> None:
        if log_progress:
            try:
                log_progress(msg)
            except Exception:
                pass

    errors: List[Dict[str, Any]] = []
    account_id = (form.get("account_id") or "").strip()
    if not account_id:
        return {"ok": False, "error": "account_id (marketer) is required"}

    ads = _files_for_prefix(files, "creative")
    # Library-drained launches may pass image URLs instead of files; the app
    # layer materializes those into ``creative_<n>`` files before calling us.
    if not ads:
        return {"ok": False, "error": "at least one creative image is required (creative_0)"}

    fmt = (form.get("creative_format") or "1:1").strip()
    if fmt not in _FORMAT_DIMS:
        fmt = "1:1"

    default_landing = (form.get("landing_page_url") or "").strip()
    default_cta = (form.get("cta_label") or "LEARN_MORE").strip().upper()

    # -------- Pre-flight copy validation --------
    preflight: List[Dict[str, Any]] = []
    for idx, _ in ads:
        text = (form.get(f"headline_{idx}") or form.get("headline") or "").strip()
        landing = (form.get(f"landing_page_url_{idx}") or default_landing).strip()
        errs: List[str] = []
        if not text:
            errs.append("headline required")
        elif len(text) > 100:
            errs.append(f"headline too long ({len(text)}>100 chars)")
        if not landing:
            errs.append("landing_page_url required")
        if errs:
            preflight.append({"ad": idx, "stage": "validate", "error": "; ".join(errs)})
    if preflight:
        return {
            "ok": False,
            "platform": "outbrain",
            "error": "ad copy validation failed — nothing was created in Outbrain",
            "errors": preflight,
        }

    # -------- Campaign (new or existing) --------
    campaign_mode = (form.get("campaign_mode") or "new").strip().lower()
    reused_campaign = False
    budget_id = None
    if campaign_mode == "existing":
        campaign_id = (form.get("campaign_id") or "").strip()
        if not campaign_id:
            return {"ok": False, "error": "campaign_mode=existing requires campaign_id"}
        reused_campaign = True
    else:
        campaign_name = (form.get("campaign_name") or "").strip()
        if not campaign_name:
            return {"ok": False, "error": "campaign_name is required"}
        try:
            budget_payload = _build_budget_payload(form, campaign_name=campaign_name)
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        try:
            _progress("Creating budget…")
            created_budget = adapter.create_budget(account_id, budget_payload)
            budget_id = str(created_budget.get("id") or "")
        except Exception as e:
            logger.warning("ob_create_budget_failed: %s payload=%s", e, budget_payload)
            return {"ok": False, "error": f"create_budget failed: {e}"}
        if not budget_id:
            return {"ok": False, "error": f"budget create returned no id: {created_budget}"}

        try:
            campaign_payload = _build_campaign_payload(form, budget_id=budget_id)
        except ValueError as e:
            return {"ok": False, "error": str(e), "budget_id": budget_id}
        try:
            _progress("Creating campaign…")
            created_campaign = adapter.create_campaign(account_id, campaign_payload)
        except Exception as e:
            logger.warning("ob_create_campaign_failed: %s payload=%s", e, campaign_payload)
            return {"ok": False, "error": f"create_campaign failed: {e}", "budget_id": budget_id}
        campaign_id = str(created_campaign.get("id") or "")
        if not campaign_id:
            return {"ok": False, "error": f"campaign create returned no id: {created_campaign}"}

    # -------- Promoted links (ads) --------
    ad_results: List[Dict[str, Any]] = []
    total = len(ads)
    for n, (idx, creative_file) in enumerate(ads, start=1):
        _progress(f"Preparing creative {n}/{total} ({fmt})…")
        try:
            img_bytes, fname = creative_builder(creative_file, fmt=fmt)
        except Exception as e:
            errors.append({"ad": idx, "stage": "image_prep", "error": str(e)})
            continue
        try:
            public_url = host_image(img_bytes, fname)
        except Exception as e:
            errors.append({"ad": idx, "stage": "image_host", "error": str(e)})
            continue
        if not public_url:
            errors.append({"ad": idx, "stage": "image_host", "error": "no public url produced"})
            continue

        text = (form.get(f"headline_{idx}") or form.get("headline") or "").strip()
        landing = (form.get(f"landing_page_url_{idx}") or default_landing).strip()
        cta = _norm_cta(form.get(f"cta_label_{idx}") or default_cta)

        payload: Dict[str, Any] = {
            "text": text[:100],
            "url": landing,
            "enabled": True,
            "imageUrl": public_url,
        }
        if cta:
            payload["callToAction"] = cta

        try:
            _progress(f"Creating ad {n}/{total}…")
            created = adapter.client.create_promoted_link(campaign_id, payload)
        except Exception as e:
            errors.append({"ad": idx, "stage": "create_promoted_link", "error": str(e)})
            continue
        ad_results.append(
            {
                "ad": idx,
                "ad_id": str(created.get("id") or ""),
                "promoted_link_id": str(created.get("id") or ""),
                "text": text,
                "image_url": public_url,
                "cached_image_url": created.get("cachedImageUrl"),
            }
        )

    return {
        "ok": bool(ad_results) and not (campaign_mode == "new" and not ad_results),
        "platform": "outbrain",
        "campaign_id": campaign_id,
        "campaign_reused": reused_campaign,
        "budget_id": budget_id,
        "creative_format": fmt,
        "ads": ad_results,
        "errors": errors,
        "note": "Campaign created PAUSED — review and enable it in Outbrain when ready.",
    }


__all__ = ["outbrain_bulk_launch", "prepare_creative"]

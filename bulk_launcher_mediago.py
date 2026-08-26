"""
MediaGo native bulk launcher.

MediaGo has no ad-set layer: one Campaign owns up to 10 native ads
(``creative_type=native``). Each ad is ``{asset_name, img, headline}``.
Images are hosted at a public URL (same ``/public/creative/`` path as
Outbrain) because MediaGo fetches ``img`` rather than accepting uploads.

Native format (default): 1200×628 (1.91:1). Optional 1:1 (1200×1200).
Display / banner / video sizes are not generated.

Expected form fields (see ``templates/launch.html`` mediago block):

    account_id          required
    campaign_name       required
    objective           lead | conversions | awareness   (default conversions)
    charge_type         cpc | smart_bid | max_cv         (default max_cv)
    daily_cap_usd       required, min $20                (default 100)
    spend_limit_usd     optional (defaults to daily_cap * 30)
    spend_mode / pacing 0=accelerate (default), 1=standard
    cpc_usd             required when charge_type is cpc/smart_bid
    target_cpa_usd      required for lead/conversions    (default 40)
    brand_name          required, <=30 chars
    landing_page        required (also landing_url / copy-variant aliases)
    utm_tracking        optional
    product_type        default Health & Fitness
    language            default en
    location_region     default US
    platform_*          mobile/desktop/tablet on, xbox off
    campaign_status     0 paused (default) | 1 on
    creative_format     "1.91:1" | "1:1"
    apply_site_exclusions  "1" (default) to push persisted site blocks
    headline_<n>        required per ad, <=80 chars
    creative_<n>        file field
"""
from __future__ import annotations

import io
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_MAX_IMAGE_BYTES = 200 * 1024  # MediaGo native typically caps ~200 KB
_MAX_ADS_PER_CAMPAIGN = 10
_HEADLINE_MAX = 80
_BRAND_MAX = 30
_MIN_DAILY_CAP = 20.0
_DEFAULT_DAILY_CAP = 100.0
_DEFAULT_TARGET_CPA = 40.0
_DEFAULT_PRODUCT = "Health & Fitness"
_DEFAULT_CHARGE = "max_cv"
_DEFAULT_OBJECTIVE = "conversions"
_NO_END = "2030-01-01 00:00:00"
_DEFAULT_PLATFORMS = ("Mobile", "Desktop", "Tablet")
_ALL_PLATFORMS = ("Mobile", "Desktop", "Tablet", "Xbox")
_LANDING_KEYS = (
    "landing_page",
    "landing_page_url",
    "landing_url",
    "lander_url",
    "copy_variant_url",
    "copy_url",
)

_FORMAT_DIMS: Dict[str, Tuple[int, int]] = {
    "1.91:1": (1200, 628),
    "16:9": (1200, 628),
    "1:1": (1200, 1200),
}

_OBJECTIVES = {"lead", "conversions", "awareness"}
_CHARGE_TYPES = {"cpc", "smart_bid", "max_cv"}
_PRODUCT_TYPES = {
    "E-commerce", "Lead Gen", "Health & Fitness", "Finance & Insurance",
    "Real Estate", "Auto", "Careers", "Technology & Computing", "Education",
    "Arts & Entertainment", "Style & Fashion", "Family & Parenting", "Food",
    "Hobbies & Interests", "Home & Garden", "Law Gov't & Politics", "News",
    "Pets", "Religion & Spirituality", "Science", "Sports", "Travel",
    "Society", "Others",
}

_DEFAULT_UTM = (
    "utm_source=mediago&utm_medium=referral"
    "&utm_campaign=${CAMPAIGN_NAME}&utm_content=${AD_TITLE}"
    "&utm_term=${CONTENT_NAME}"
)

_GEMINI_IMAGE_MODELS = (
    "gemini-3.1-flash-image-preview",
    "gemini-2.5-flash-image-preview",
)
_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
_GEMINI_TIMEOUT_S = 90
_WIDE_PROMPT = (
    "You are an expert creative director. Extend this square advertising image "
    "horizontally into a 1.91:1 landscape (1200x628) native ad. Naturally "
    "continue the scene on the LEFT and RIGHT edges so the original subject, "
    "text, and typography stay exactly as-is and fully centered. Match the "
    "existing lighting, colors, textures, grain, and background seamlessly. "
    "Do NOT crop, warp, rotate, re-color, re-type, or add new text, logos, "
    "watermarks, borders, or foreground subjects."
)


# ----------------------------------------------------------------------
# Image pipeline (mirror SmartNews / Outbrain)
# ----------------------------------------------------------------------
def _load_image(file_bytes: bytes):
    from PIL import Image, ImageFilter

    return Image, ImageFilter, Image.open(io.BytesIO(file_bytes))


def _encode_jpeg(im, *, max_bytes: int = _MAX_IMAGE_BYTES) -> bytes:
    if im.mode != "RGB":
        im = im.convert("RGB")
    quality = 88
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
            logger.warning("MediaGo gemini outpaint failed on %s: %s", model, exc)
            continue
        if resp.status_code in (404,):
            continue
        if resp.status_code != 200:
            logger.warning("MediaGo gemini HTTP %s on %s: %s", resp.status_code, model, resp.text[:300])
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


def prepare_native_creative(file_obj: Any, *, fmt: str = "1.91:1") -> Tuple[bytes, str]:
    """Return ``(jpeg_bytes, filename)`` in the MediaGo native size."""
    data = file_obj.read() if hasattr(file_obj, "read") else bytes(file_obj)
    filename = getattr(file_obj, "filename", None) or "creative.jpg"
    base = os.path.splitext(os.path.basename(filename))[0] or "creative"
    fmt = (fmt or "1.91:1").strip()
    target = _FORMAT_DIMS.get(fmt, _FORMAT_DIMS["1.91:1"])

    square = _resize_cover(data, 1200, 1200)
    if fmt in ("1.91:1", "16:9"):
        out = None
        key = _gemini_api_key()
        if key:
            try:
                out = _gemini_outpaint_wide(square, target=target, api_key=key)
            except Exception as exc:  # pragma: no cover
                logger.warning("MediaGo outpaint raised; using blur fill: %s", exc)
        if not out:
            out = _local_blur_fill(square, target=target)
        return out, f"{base}_1200x628.jpg"
    return square, f"{base}_1200x1200.jpg"


# ----------------------------------------------------------------------
# Form helpers
# ----------------------------------------------------------------------
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


def _parse_dt(s: str) -> Optional[datetime]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _fmt_dt(dt: datetime) -> str:
    if dt.tzinfo:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _all_hours_dayparting() -> List[List[int]]:
    return [[1] * 24 for _ in range(7)]


def _chunks(items: Sequence[Any], n: int) -> List[Sequence[Any]]:
    return [items[i : i + n] for i in range(0, len(items), n)]


def _clean_landing(url: str) -> str:
    s = (url or "").strip()
    if not s:
        return ""
    s = re.sub(r"[?&=]+$", "", s)
    return s


def _resolve_landing(form: Mapping[str, Any]) -> str:
    for key in _LANDING_KEYS:
        raw = _clean_landing(str(form.get(key) or ""))
        if raw:
            return raw
    return ""


def _truthy(v: Any) -> bool:
    return str(v or "").strip().lower() in ("1", "on", "true", "yes", "checked")


def _spend_mode(form: Mapping[str, Any]) -> int:
    """MediaGo ``spend_mode``: 0 = accelerate, 1 = standard/uniform."""
    raw = form.get("spend_mode")
    if raw in (None, ""):
        raw = form.get("pacing")
    s = str(raw or "").strip().lower()
    if s in ("1", "standard", "uniform", "even"):
        return 1
    return 0


def _campaign_status(form: Mapping[str, Any]) -> int:
    raw = form.get("campaign_status")
    if raw in (None, ""):
        raw = form.get("status")
    s = str(raw or "").strip().lower()
    if s in ("1", "on", "true", "active", "enabled"):
        return 1
    return 0


def _platform_targeting(form: Mapping[str, Any]) -> Dict[str, Any]:
    selected: List[str] = []
    for key, label in (
        ("platform_mobile", "Mobile"),
        ("platform_desktop", "Desktop"),
        ("platform_tablet", "Tablet"),
        ("platform_xbox", "Xbox"),
    ):
        if _truthy(form.get(key)):
            selected.append(label)
    raw = form.get("platforms") or form.get("platform_targeting")
    if not selected and raw:
        if isinstance(raw, (list, tuple)):
            parts = [str(x) for x in raw]
        else:
            parts = str(raw).split(",")
        for part in parts:
            label = part.strip().title()
            if label == "Ios":
                label = "IOS"
            if label in _ALL_PLATFORMS and label not in selected:
                selected.append(label)
    if not selected:
        selected = list(_DEFAULT_PLATFORMS)
    if set(selected) >= set(_ALL_PLATFORMS):
        return {"type": "ALL", "value": []}
    return {"type": "INCLUDE", "value": selected}


def _location(form: Mapping[str, Any]) -> List[Dict[str, Any]]:
    region = (form.get("location_region") or "US").strip().upper() or "US"
    mode = (form.get("location_mode") or "all").strip().lower()
    option = "zipcode" if mode in ("zip", "zipcode", "zip_code") else "state"
    raw = form.get("location_value") or form.get("location_values") or ""
    if isinstance(raw, (list, tuple)):
        values = [str(x).strip() for x in raw if str(x).strip()]
    else:
        values = [x.strip() for x in str(raw).split(",") if x.strip()]
    if mode in ("state", "zip", "zipcode", "zip_code") and values:
        return [{"type": "INCLUDE", "option": option, "value": values, "region": region}]
    return [{"type": "ALL", "option": "state", "value": [], "region": region}]


# ----------------------------------------------------------------------
# Payload
# ----------------------------------------------------------------------
def build_campaign_payload(
    form: Mapping[str, Any],
    ads: List[Dict[str, str]],
    *,
    campaign_name: Optional[str] = None,
) -> Dict[str, Any]:
    name = (campaign_name or form.get("campaign_name") or "").strip()
    if not name:
        raise ValueError("campaign_name is required")
    brand = (form.get("brand_name") or "").strip()
    if not brand:
        raise ValueError("brand_name is required")
    landing = _resolve_landing(form)
    if not landing:
        raise ValueError("landing_page is required")
    daily = _usd(form.get("daily_cap_usd") or form.get("daily_cap"))
    if daily is None:
        daily = _DEFAULT_DAILY_CAP
    if daily < _MIN_DAILY_CAP:
        raise ValueError(f"daily_cap_usd must be at least ${_MIN_DAILY_CAP:.0f}")
    spend_limit = _usd(form.get("spend_limit_usd") or form.get("spend_limit")) or round(daily * 30, 2)
    charge = (form.get("charge_type") or _DEFAULT_CHARGE).strip().lower()
    if charge not in _CHARGE_TYPES:
        charge = _DEFAULT_CHARGE
    objective = (form.get("objective") or _DEFAULT_OBJECTIVE).strip().lower()
    if objective not in _OBJECTIVES:
        objective = _DEFAULT_OBJECTIVE
    product = (form.get("product_type") or _DEFAULT_PRODUCT).strip()
    if product not in _PRODUCT_TYPES:
        product = _DEFAULT_PRODUCT

    start = _parse_dt(form.get("start_time") or "") or datetime.now(timezone.utc)
    end = _parse_dt(form.get("end_time") or "")
    end_s = _fmt_dt(end) if end else _NO_END

    language = (form.get("language") or "en").strip() or "en"
    utm = (form.get("utm_tracking") or form.get("utm_parameters") or "").strip() or _DEFAULT_UTM
    spend_mode = _spend_mode(form)

    payload: Dict[str, Any] = {
        "campaign_name": name,
        "creative_type": "native",
        "status": _campaign_status(form),
        "day_parting": _all_hours_dayparting(),
        "dp_timezone": (form.get("dp_timezone") or "EST").strip() or "EST",
        "start_time": _fmt_dt(start),
        "end_time": end_s,
        "daily_cap": round(daily, 2),
        "spend_limit": round(spend_limit, 2),
        "spend_mode": spend_mode,
        "charge_type": charge,
        "audience": {"type": "ALL", "value": []},
        "language": language,
        "location": _location(form),
        "platform_targeting": _platform_targeting(form),
        "os_targeting": {"type": "ALL", "value": []},
        "browser_targeting": {"type": "ALL", "value": []},
        "product_type": product,
        "objective": objective,
        "landing_page": landing,
        "utm_tracking": utm,
        "brand_name": brand[:_BRAND_MAX],
        "ad": ads[:_MAX_ADS_PER_CAMPAIGN],
    }
    if charge in ("cpc", "smart_bid"):
        cpc = _usd(form.get("cpc_usd") or form.get("cpc")) or 0.50
        payload["cpc"] = round(min(cpc, 5.0), 3)
    if objective in ("lead", "conversions"):
        # Official create-campaign field is ``target_cpa`` (USD). Required when
        # objective is Lead Generation or Online Purchases.
        # https://apidoc.mediago.io/346347754e0
        tcpa = _usd(form.get("target_cpa_usd") or form.get("target_cpa"))
        if tcpa is None or tcpa <= 0:
            tcpa = _DEFAULT_TARGET_CPA
        payload["target_cpa"] = round(tcpa, 2)
        if charge == "max_cv" and daily >= tcpa * 30:
            raise ValueError(
                "When bid strategy is max conversions, daily_cap must be less than target_cpa × 30"
            )
        from platforms.mediago import optimization_type_for_conversion

        opt = (form.get("optimization_type") or "").strip()
        if not opt:
            opt = optimization_type_for_conversion(
                form.get("mediago_pixel") or form.get("pixel_id") or ""
            )
        payload["optimization_type"] = opt or "-1"
    return payload


# ----------------------------------------------------------------------
# Launch
# ----------------------------------------------------------------------
def mediago_bulk_launch(
    adapter: Any,
    *,
    form: Mapping[str, Any],
    files: Mapping[str, Any],
    host_image: Callable[[bytes, str], str],
    creative_builder: Callable[..., Tuple[bytes, str]] = prepare_native_creative,
    exclusions: Optional[Sequence[Dict[str, Any]]] = None,
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
        return {"ok": False, "error": "account_id is required"}

    ads_files = _files_for_prefix(files, "creative")
    if not ads_files:
        return {"ok": False, "error": "at least one creative image is required (creative_0)"}

    fmt = (form.get("creative_format") or "1.91:1").strip()
    if fmt not in _FORMAT_DIMS:
        fmt = "1.91:1"

    preflight: List[Dict[str, Any]] = []
    for idx, _ in ads_files:
        text = (form.get(f"headline_{idx}") or form.get("headline") or "").strip()
        errs: List[str] = []
        if not text:
            errs.append("headline required")
        elif len(text) > _HEADLINE_MAX:
            errs.append(f"headline too long ({len(text)}>{_HEADLINE_MAX} chars)")
        if errs:
            preflight.append({"ad": idx, "stage": "validate", "error": "; ".join(errs)})
    brand = (form.get("brand_name") or "").strip()
    if not brand:
        preflight.append({"stage": "validate", "error": "brand_name is required"})
    landing = _resolve_landing(form)
    if not landing:
        preflight.append({"stage": "validate", "error": "landing_page is required"})
    if preflight:
        return {
            "ok": False,
            "platform": "mediago",
            "error": "validation failed — nothing was created in MediaGo",
            "errors": preflight,
        }

    prepared: List[Dict[str, str]] = []
    total = len(ads_files)
    for n, (idx, creative_file) in enumerate(ads_files, start=1):
        _progress(f"Preparing native creative {n}/{total} ({fmt})…")
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
        asset = (form.get(f"ad_name_{idx}") or form.get(f"asset_name_{idx}") or f"Ad {idx}").strip()
        prepared.append(
            {
                "idx": idx,
                "asset_name": asset[:80] or f"Ad {idx}",
                "img": public_url,
                "headline": text[:_HEADLINE_MAX],
            }
        )

    if not prepared:
        return {
            "ok": False,
            "platform": "mediago",
            "error": "no creatives could be prepared",
            "errors": errors,
        }

    base_name = (form.get("campaign_name") or "").strip() or "MediaGo native"
    created_campaigns: List[Dict[str, Any]] = []
    all_ads: List[Dict[str, Any]] = []
    apply_exclusions = (form.get("apply_site_exclusions") or "1").strip() not in ("0", "false", "off")

    for batch_i, batch in enumerate(_chunks(prepared, _MAX_ADS_PER_CAMPAIGN)):
        suffix = f" ({batch_i + 1})" if len(prepared) > _MAX_ADS_PER_CAMPAIGN else ""
        name = (base_name + suffix)[:200]
        ads_payload = [
            {"asset_name": a["asset_name"], "img": a["img"], "headline": a["headline"]}
            for a in batch
        ]
        try:
            payload = build_campaign_payload(form, ads_payload, campaign_name=name)
        except ValueError as e:
            return {"ok": False, "platform": "mediago", "error": str(e), "errors": errors}
        try:
            _progress(f"Creating native campaign {batch_i + 1}…")
            created = adapter.create_campaign(account_id, payload)
        except Exception as e:
            logger.warning("mg_create_campaign_failed: %s", e)
            errors.append({"stage": "create_campaign", "error": str(e), "batch": batch_i})
            continue
        campaign_id = str(
            (created or {}).get("campaign_id") or (created or {}).get("id") or ""
        )
        created_campaigns.append(
            {"campaign_id": campaign_id, "name": name, "raw": created}
        )
        for a in batch:
            all_ads.append(
                {
                    "ad": a["idx"],
                    "headline": a["headline"],
                    "image_url": a["img"],
                    "campaign_id": campaign_id,
                }
            )
        if apply_exclusions and exclusions and campaign_id:
            try:
                _progress(f"Applying {len(exclusions)} site exclusions to {campaign_id}…")
                adapter.block_sites(account_id, exclusions, campaign_id=campaign_id, block=True)
            except Exception as e:
                errors.append(
                    {
                        "stage": "site_exclusions",
                        "campaign_id": campaign_id,
                        "error": str(e),
                    }
                )

    first_id = created_campaigns[0]["campaign_id"] if created_campaigns else None
    created_status = _campaign_status(form)
    if created_status == 1:
        note = "Campaign created ACTIVE (native). It can start spending immediately."
    else:
        note = "Campaign created PAUSED (native). Review in MediaGo, then enable."
    return {
        "ok": bool(created_campaigns),
        "platform": "mediago",
        "campaign_id": first_id,
        "campaigns": created_campaigns,
        "creative_format": fmt,
        "ads": all_ads,
        "errors": errors,
        "status": created_status,
        "note": note,
    }


__all__ = ["mediago_bulk_launch", "prepare_native_creative", "build_campaign_payload"]

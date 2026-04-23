"""
SmartNews v3 bulk launcher.

v3 hierarchy: Account → Campaign → Ad Group → Ad. Each ad references media
files by id (``media_file_ids``). The launcher:

  1. Uploads one 1:1 square image per ad (required).
  2. Auto-generates a 1.91:1 landscape variant from the same square using
     Gemini 2.5 Flash Image ("nano banana") outpainting when a Google AI key
     is configured. Falls back to a local Pillow blur-fill so the pipeline
     always produces a valid asset even without the AI service.
  3. Creates one Campaign, one Ad Group under it, then N Ads.
  4. Optionally auto-submits every created ad to SmartNews moderation
     (``submission_status=SUBMITTED``) — default ON.

Budgets on the form are passed in as **cents** (integer, smallest currency
unit). The adapter converts to SmartNews ``_micro`` units per account
currency.

Expected form fields (see ``templates/launch_smartnews.html``):

    account_id        required
    campaign_name     required
    objective         default TRAFFIC          (campaign-level)
    daily_budget_cents required                (campaign or ad-group)
    start_time / end_time  ISO 8601 (optional)
    ad_group_name     required                 (one ad group per launch)
    landing_page_url  required for each ad when objective is WEB
    sponsored_name    required
    headline_<n>      required per ad
    description_<n>   required per ad
    ad_name_<n>       default "Ad <n>"
    cta_label_<n>     default BOOK_NOW
    creative_<n>      file field, 1:1 square image (required)
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

# Far-future sentinel used when the operator wants "no end date". SmartNews
# v3 always requires ``end_date_time`` on a campaign, but it happily accepts
# a date far in the future — same trick NewsBreak uses for evergreen runs.
_NO_END_DATE_SENTINEL = datetime(2099, 12, 31, 23, 59, tzinfo=timezone.utc)


# ----------------------------------------------------------------------
# Image pipeline
# ----------------------------------------------------------------------
_MAX_IMAGE_BYTES = 5 * 1024 * 1024  # v3 allows 5 MiB


def _load_image(file_bytes: bytes):
    try:
        from PIL import Image, ImageFilter
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for SmartNews image preparation") from exc
    return Image, ImageFilter, Image.open(io.BytesIO(file_bytes))


def _encode_jpeg(im, *, max_bytes: int = _MAX_IMAGE_BYTES) -> bytes:
    """Encode a PIL image as JPEG, shrinking quality until it fits."""
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
    """Cover-crop resize to (target_w, target_h) and re-encode as JPEG."""
    Image, _ImageFilter, im = _load_image(file_bytes)
    with im:
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        sw, sh = im.size
        tr, sr = target_w / target_h, sw / sh
        if sr > tr:
            new_h = sh
            new_w = int(sh * tr)
            left = (sw - new_w) // 2
            box = (left, 0, left + new_w, sh)
        else:
            new_w = sw
            new_h = int(sw / tr)
            top = (sh - new_h) // 2
            box = (0, top, sw, top + new_h)
        out = im.crop(box).resize((target_w, target_h), Image.LANCZOS)
        return _encode_jpeg(out)


def _local_blur_fill_landscape(square_bytes: bytes, *, target: Tuple[int, int] = (1200, 628)) -> bytes:
    """Deterministic local fallback: center the square on a blurred background.

    Used when no AI service is available so the pipeline still produces a
    valid 1.91:1 asset. This is NOT visually great for creatives with text —
    prefer the Gemini outpainting path whenever an API key is configured.
    """
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

        fg_w = min(th, im.size[0])
        fg_h = fg_w
        fg = im.resize((fg_w, fg_h), Image.LANCZOS)
        x = (tw - fg_w) // 2
        y = (th - fg_h) // 2

        canvas = bg.copy()
        canvas.paste(fg, (x, y))
        return _encode_jpeg(canvas)


# Gemini image-edit model. Nano Banana 2 is what creative_brief_tool calls
# ``gemini-3.1-flash-image-preview``; we prefer it when present and fall back
# to the stable 2.5 flash image preview model.
_GEMINI_IMAGE_MODELS = (
    "gemini-3.1-flash-image-preview",
    "gemini-2.5-flash-image-preview",
    "gemini-2.0-flash-exp-image-generation",
)
_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
_GEMINI_TIMEOUT_S = 90
_OUTPAINT_PROMPT = (
    "You are an expert creative director. Extend this 1:1 square image "
    "horizontally into a 1.91:1 landscape (1200x628) advertising creative. "
    "Naturally continue the scene on the LEFT and RIGHT edges so that the "
    "original subject, text, and typography stay exactly as-is and fully "
    "centered. Match the existing lighting, colors, textures, grain, and "
    "background seamlessly. Do NOT crop, warp, rotate, re-color, re-type, "
    "or add new text, logos, watermarks, borders, or foreground subjects. "
    "The final image must read as a single cohesive photograph."
)


def _gemini_api_key() -> Optional[str]:
    return (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GOOGLE_GENAI_API_KEY")
    )


def _extract_image_from_gemini_response(data: Any) -> Optional[bytes]:
    """Pull the first inline image out of a Gemini generateContent response."""
    if not isinstance(data, dict):
        return None
    for cand in data.get("candidates") or []:
        content = cand.get("content") if isinstance(cand, dict) else None
        parts = (content or {}).get("parts") or []
        for part in parts:
            if not isinstance(part, dict):
                continue
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                try:
                    return base64.b64decode(inline["data"])
                except Exception:  # pragma: no cover
                    continue
    return None


def _gemini_outpaint_to_landscape(
    square_bytes: bytes,
    *,
    target: Tuple[int, int],
    api_key: str,
) -> Optional[bytes]:
    """Call Gemini to outpaint the square into 1.91:1. Returns None on failure."""
    try:
        import requests  # local import so the module stays importable offline
    except Exception as exc:  # pragma: no cover
        logger.warning("requests unavailable for Gemini outpainting: %s", exc)
        return None

    b64 = base64.b64encode(square_bytes).decode("ascii")
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": _OUTPAINT_PROMPT},
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
            logger.warning("Gemini outpaint request failed on %s: %s", model, exc)
            continue
        if resp.status_code == 404:
            # Model not available to this project; try the next one.
            continue
        if resp.status_code != 200:
            logger.warning(
                "Gemini outpaint HTTP %s on %s: %s", resp.status_code, model, resp.text[:500]
            )
            continue
        try:
            data = resp.json()
        except Exception:
            logger.warning("Gemini outpaint returned non-JSON on %s", model)
            continue
        img_bytes = _extract_image_from_gemini_response(data)
        if not img_bytes:
            logger.warning(
                "Gemini outpaint returned no image data on %s (response=%s)",
                model,
                json.dumps(data)[:600],
            )
            continue
        # Normalize to target resolution & JPEG to stay within the 5 MiB limit.
        _Image, _ImageFilter, im = _load_image(img_bytes)
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
            im = im.resize((tw, th), _Image.LANCZOS)
            return _encode_jpeg(im)
    return None


def ai_expand_square_to_landscape(square_bytes: bytes, *, target: Tuple[int, int] = (1200, 628)) -> bytes:
    """Expand a 1:1 image into a 1.91:1 landscape.

    Preference order:
      1. **Gemini outpainting** (``gemini-3.1-flash-image-preview`` →
         ``gemini-2.5-flash-image-preview``) when ``GEMINI_API_KEY`` /
         ``GOOGLE_API_KEY`` is configured.
      2. **Local blur-fill fallback** — centers the square on a blurred
         enlargement of itself. Always produces a valid asset.

    The fallback is only chosen when the AI path is unavailable OR fails;
    we never silently ship a blurred letterbox if the AI key is present and
    produced a valid image.
    """
    api_key = _gemini_api_key()
    if api_key:
        try:
            out = _gemini_outpaint_to_landscape(square_bytes, target=target, api_key=api_key)
            if out:
                return out
        except Exception as exc:  # pragma: no cover
            logger.warning("Gemini outpaint raised; falling back to local blur: %s", exc)
    return _local_blur_fill_landscape(square_bytes, target=target)


def creative_pair_from_square(square_file: Any) -> Dict[str, Tuple[bytes, str]]:
    """Return (1:1, 1.91:1) image pair from a single uploaded square file.

    Returns a dict ``{"square": (bytes, filename), "landscape": (bytes, filename)}``.
    """
    data = square_file.read()
    filename = getattr(square_file, "filename", None) or "creative.jpg"
    base = os.path.splitext(os.path.basename(filename))[0]

    square_bytes = _resize_cover(data, 1080, 1080)
    landscape_bytes = ai_expand_square_to_landscape(square_bytes, target=(1200, 628))
    return {
        "square": (square_bytes, f"{base}_1080x1080.jpg"),
        "landscape": (landscape_bytes, f"{base}_1200x628.jpg"),
    }


# ----------------------------------------------------------------------
# Form parsing helpers
# ----------------------------------------------------------------------
def _log_json(label: str, obj: Any) -> None:
    try:
        logger.info("%s %s", label, json.dumps(obj, default=str)[:4000])
    except Exception:
        logger.info("%s %r", label, obj)


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
    """Return [(ad_index, file)] for every ``<prefix>_<n>`` upload with a filename."""
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


def _int_or_none(v: Any) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _usd_to_cents(v: Any) -> Optional[int]:
    """Convert a USD dollar amount (e.g. '50', '1.50', 2.5) to integer cents."""
    if v is None or v == "":
        return None
    try:
        dollars = float(v)
    except (TypeError, ValueError):
        return None
    if dollars < 0:
        return None
    return int(round(dollars * 100))


def _budget_cents_from_form(form: Mapping[str, Any], *, usd_key: str, cents_key: str) -> Optional[int]:
    """Prefer the USD field, fall back to the explicit cents field."""
    cents = _usd_to_cents(form.get(usd_key))
    if cents is not None:
        return cents
    return _int_or_none(form.get(cents_key))


def _int_req(v: Any, *, field: str) -> int:
    got = _int_or_none(v)
    if got is None:
        raise ValueError(f"{field} is required")
    return got


def _form_bool(v: Any) -> bool:
    """Truthy for HTML form values ('1', 'on', 'true', 'yes', True, 1)."""
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "on", "true", "yes", "y")


def _iso(dt: datetime) -> str:
    """Format UTC ISO-8601 with minute precision.

    SmartNews v3 rejects timestamps that include sub-minute precision
    (``"The start date time can't have a unit smaller than a minute."``),
    so we always floor to ``HH:MM:00Z``.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc).replace(second=0, microsecond=0)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _parse_schedule(form: Mapping[str, Any]) -> Tuple[str, str]:
    """Return (start_iso, end_iso) honoring the "no end date" convention.

    If ``end_time`` is blank we ship the far-future ``_NO_END_DATE_SENTINEL``
    (2099-12-31T23:59Z) — same evergreen trick the NewsBreak launcher uses —
    so SmartNews happily schedules the campaign forever.
    """
    now = datetime.now(timezone.utc)
    start = _parse_iso((form.get("start_time") or "").strip()) or now
    end_raw = (form.get("end_time") or "").strip()
    end = _parse_iso(end_raw) if end_raw else _NO_END_DATE_SENTINEL
    if end <= start:
        # Protect against operators picking an end date before start.
        end = max(_NO_END_DATE_SENTINEL, start + timedelta(days=30))
    return _iso(start), _iso(end)


# ----------------------------------------------------------------------
# Launch flow
# ----------------------------------------------------------------------
def smartnews_bulk_launch(
    adapter: Any,
    *,
    form: Mapping[str, Any],
    files: Mapping[str, Any],
    pair_builder=creative_pair_from_square,
) -> Dict[str, Any]:
    errors: List[Dict[str, Any]] = []
    account_id = (form.get("account_id") or "").strip()
    if not account_id:
        return {"ok": False, "error": "account_id is required"}

    ads = _files_for_prefix(files, "creative")
    if not ads:
        return {"ok": False, "error": "at least one 1:1 creative image is required (creative_0)"}

    # -------- Campaign --------
    try:
        campaign_payload = _build_campaign_payload(form)
    except ValueError as e:
        return {"ok": False, "error": str(e)}

    try:
        created_campaign = adapter.create_campaign(account_id, campaign_payload)
    except Exception as e:
        _log_json("sn_create_campaign_failed", {"error": str(e), "payload": campaign_payload})
        return {"ok": False, "error": f"create_campaign failed: {e}"}

    campaign_id = str(
        created_campaign.get("campaign_id") or created_campaign.get("id") or ""
    )
    if not campaign_id:
        return {"ok": False, "error": f"campaign create returned no id: {created_campaign}"}

    # -------- Ad Group --------
    try:
        ad_group_payload = _build_ad_group_payload(form)
    except ValueError as e:
        errors.append({"stage": "build_ad_group", "error": str(e)})
        return {
            "ok": False,
            "platform": "smartnews",
            "campaign_id": campaign_id,
            "errors": errors,
        }
    try:
        created_group = adapter.create_ad_group(account_id, campaign_id, ad_group_payload)
    except Exception as e:
        _log_json("sn_create_ad_group_failed", {"error": str(e), "payload": ad_group_payload})
        errors.append({"stage": "create_ad_group", "error": str(e)})
        return {
            "ok": False,
            "platform": "smartnews",
            "campaign_id": campaign_id,
            "errors": errors,
        }
    ad_group_id = str(created_group.get("ad_group_id") or created_group.get("id") or "")

    # -------- Ads --------
    ad_results: List[Dict[str, Any]] = []
    default_landing = (form.get("landing_page_url") or "").strip()
    default_cta = (form.get("cta_label") or "LEARN_MORE").strip().upper()
    default_sponsored = (form.get("sponsored_name") or "").strip()
    show_cta = _form_bool(form.get("show_cta"))

    for idx, square_file in ads:
        try:
            pair = pair_builder(square_file)
        except Exception as e:
            errors.append({"ad": idx, "stage": "image_prep", "error": str(e)})
            continue

        uploaded_media_ids: List[int] = []
        try:
            for slot in ("square", "landscape"):
                data, fname = pair[slot]
                resp = adapter.upload_asset(account_id, data, fname, media_type="IMAGE")
                mid = (
                    resp.get("media_file_id")
                    or resp.get("mediaFileId")
                    or resp.get("id")
                )
                if mid is not None:
                    uploaded_media_ids.append(int(mid))
        except Exception as e:
            errors.append({"ad": idx, "stage": "image_upload", "error": str(e)})
            continue
        if not uploaded_media_ids:
            errors.append({"ad": idx, "stage": "image_upload", "error": "no media_file_id returned"})
            continue

        headline = (form.get(f"headline_{idx}") or form.get("headline") or "").strip()
        description = (form.get(f"description_{idx}") or form.get("description") or "").strip()
        ad_name = (form.get(f"ad_name_{idx}") or f"Ad {idx}").strip()
        landing = (
            form.get(f"landing_page_url_{idx}")
            or form.get(f"landing_url_{idx}")
            or default_landing
        ).strip()
        cta = (form.get(f"cta_label_{idx}") or default_cta).strip().upper() or "LEARN_MORE"
        sponsored = (form.get(f"sponsored_name_{idx}") or default_sponsored).strip()

        if not headline or not description:
            errors.append(
                {"ad": idx, "stage": "validate", "error": "headline and description required"}
            )
            continue

        ad_payload: Dict[str, Any] = {
            "name": ad_name,
            "landing_page_url": landing,
            "configured_status": "ACTIVE",
            "creative": {
                "format": "IMAGE",
                "image_creative_info": {
                    "headline": headline,
                    "description": description,
                    "sponsored_name": sponsored,
                    "media_file_ids": uploaded_media_ids,
                },
            },
        }
        if show_cta:
            ad_payload["cta_label"] = cta

        try:
            created_ad = adapter.create_ad(account_id, ad_group_id, ad_payload)
        except Exception as e:
            errors.append({"ad": idx, "stage": "create_ad", "error": str(e), "payload": ad_payload})
            continue

        ad_results.append(
            {
                "ad": idx,
                "name": ad_name,
                "ad_id": str(created_ad.get("ad_id") or created_ad.get("id") or ""),
                "media_file_ids": uploaded_media_ids,
            }
        )

    # -------- Auto-submit for review --------
    # Ads are created with submission_status=BEFORE_SUBMISSION. Unless the
    # operator explicitly opts out, flip each to SUBMITTED so reviewers can
    # pick them up immediately — that matches the "publish" affordance the
    # Ads Manager surfaces and the user's expectation when they click Launch.
    auto_submit_raw = form.get("auto_submit")
    auto_submit = True if auto_submit_raw is None else _form_bool(auto_submit_raw)
    submitted_ad_ids: List[str] = []
    if auto_submit and ad_results and hasattr(adapter, "submit_ad_for_review"):
        for entry in ad_results:
            ad_id = entry.get("ad_id")
            if not ad_id:
                continue
            try:
                adapter.submit_ad_for_review(account_id, ad_id)
                entry["submission_status"] = "SUBMITTED"
                submitted_ad_ids.append(ad_id)
            except Exception as e:
                entry["submission_status"] = "BEFORE_SUBMISSION"
                errors.append(
                    {"ad": entry.get("ad"), "stage": "submit_for_review", "error": str(e)}
                )

    return {
        "ok": not errors or bool(ad_results),
        "platform": "smartnews",
        "campaign_id": campaign_id,
        "ad_group_id": ad_group_id,
        "ads": ad_results,
        "submitted_ad_ids": submitted_ad_ids,
        "auto_submit": auto_submit,
        "errors": errors,
    }


# ----------------------------------------------------------------------
# Payload builders
# ----------------------------------------------------------------------
def _build_campaign_payload(form: Mapping[str, Any]) -> Dict[str, Any]:
    name = (form.get("campaign_name") or "").strip()
    if not name:
        raise ValueError("campaign_name is required")
    objective = (form.get("objective") or "SALES").strip().upper()
    daily_budget_cents = _budget_cents_from_form(
        form, usd_key="daily_budget_usd", cents_key="daily_budget_cents"
    )
    if daily_budget_cents is None:
        daily_budget_cents = _int_or_none(form.get("daily_budget"))
    spending_limit_cents = _budget_cents_from_form(
        form, usd_key="spending_limit_usd", cents_key="spending_limit_cents"
    )

    start, end = _parse_schedule(form)

    delivery_type = (form.get("delivery_type") or "STANDARD").strip().upper() or "STANDARD"
    if delivery_type not in ("STANDARD", "ACCELERATED"):
        delivery_type = "STANDARD"

    payload: Dict[str, Any] = {
        "name": name,
        "objective": objective,
        "start_date_time": start,
        "end_date_time": end,
        "configured_status": "ACTIVE",
        "billing_event": (form.get("billing_event") or "CLICK").strip().upper(),
        "optimization_goal": (form.get("optimization_goal") or "OFFSITE_CONVERSIONS").strip().upper(),
        "bid_strategy": (form.get("bid_strategy") or "LOWEST_COST_WITHOUT_CAP").strip().upper(),
        "delivery_type": delivery_type,
        "click_destination_type": (form.get("click_destination_type") or "WEB_VIEW").strip().upper(),
    }
    if daily_budget_cents is not None:
        payload["daily_budget_cents"] = daily_budget_cents  # adapter converts to micros
    if spending_limit_cents is not None:
        payload["spending_limit_cents"] = spending_limit_cents

    opt_event = (form.get("optimization_event") or "").strip().upper()
    if opt_event:
        payload["optimization_event"] = opt_event

    tracking_tag = (form.get("website_tracking_tag") or "").strip()
    if tracking_tag:
        payload["website_tracking_tag"] = tracking_tag

    # Target cost (USD) — required when bid_strategy is TARGET_COST; the
    # adapter converts target_cost_cents → target_cost_micro.
    target_cost_cents = _budget_cents_from_form(
        form, usd_key="target_cost_usd", cents_key="target_cost_cents"
    )
    if target_cost_cents is not None:
        payload["target_cost_cents"] = target_cost_cents

    return payload


def _build_ad_group_payload(form: Mapping[str, Any]) -> Dict[str, Any]:
    name = (form.get("ad_group_name") or "").strip()
    if not name:
        raise ValueError("ad_group_name is required")

    payload: Dict[str, Any] = {
        "name": name,
        "configured_status": "ACTIVE",
    }

    # Budget may live on the ad group instead of the campaign.
    daily_budget_cents = _budget_cents_from_form(
        form, usd_key="ad_group_daily_budget_usd", cents_key="ad_group_daily_budget_cents"
    )
    if daily_budget_cents is not None:
        payload["daily_budget_cents"] = daily_budget_cents

    # NOTE: ``bid_amount_*`` was removed from the launch form in favor of the
    # campaign-level Target CPA (``target_cost_usd``). SmartNews v3's
    # TARGET_COST bid strategy expects ``target_cost_micro`` on the campaign,
    # so a separate ad-group bid field was redundant and confusing.

    # Audience: very minimal for now — just ages/genders/locations if provided.
    audience: Dict[str, Any] = {}
    ages = _form_list(form, "ages")
    if ages:
        audience["ages"] = ages
    genders = _form_list(form, "genders")
    if genders:
        audience["genders"] = genders
    location_ids = [
        int(v) for v in _form_list(form, "locations") if str(v).strip().isdigit()
    ]
    if location_ids:
        audience["locations"] = location_ids
    os_type = (form.get("os_type") or "").strip().upper()
    if os_type in ("IOS", "ANDROID"):
        audience["operating_system"] = {"type": os_type}
    if audience:
        payload["audience"] = audience

    return payload


__all__ = [
    "creative_pair_from_square",
    "ai_expand_square_to_landscape",
    "smartnews_bulk_launch",
]

"""
MediaGo adapter — wraps :class:`mediago_api.MediaGoClient` behind the
``AdPlatformAdapter`` contract.

Hierarchy is Account → Campaign → Ad (no ad-set), so
``supports_ad_set_scope`` is False. Money is USD floats; our internal
integer "cents" is converted at the edge.

Campaigns are ``creative_type=native`` by default (MediaGo API default).
Display is opt-in via ``creative_type=display`` plus an allowed banner size.
"""
from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from typing import Any, BinaryIO, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from mediago_api import MediaGoClient

logger = logging.getLogger(__name__)

DEFAULT_CUT_OVER_MULTIPLE = 1.5
DEFAULT_CUT_OVER_SPEND_FLOOR = 20.0
DEFAULT_CUT_NO_CONV_SPEND_FLOOR = 10.0

# Create-campaign ``creative_type``: native (API default) or display.
# Display ``img`` must be one of these exact pixel sizes (GIF allowed).
# https://apidoc.mediago.io/346347754e0
DISPLAY_SIZES: Dict[str, Tuple[int, int]] = {
    "300x250": (300, 250),
    "728x90": (728, 90),
    "160x600": (160, 600),
    "320x50": (320, 50),
    "300x600": (300, 600),
    "970x250": (970, 250),
    "300x50": (300, 50),
    "320x100": (320, 100),
    "336x280": (336, 280),
}
DEFAULT_DISPLAY_SIZE = "300x250"
DISPLAY_SIZE_LABELS: Dict[str, str] = {
    "300x250": "300×250 Medium Rectangle",
    "728x90": "728×90 Leaderboard",
    "160x600": "160×600 Wide Skyscraper",
    "320x50": "320×50 Mobile Banner",
    "300x600": "300×600 Half Page",
    "970x250": "970×250 Billboard",
    "300x50": "300×50 Mobile Banner",
    "320x100": "320×100 Large Mobile",
    "336x280": "336×280 Large Rectangle",
}


def normalize_creative_type(value: Any) -> str:
    """Return ``display`` only when explicitly requested; else ``native``."""
    return "display" if str(value or "").strip().lower() == "display" else "native"


def normalize_display_size(value: Any) -> str:
    key = (
        str(value or "")
        .strip()
        .lower()
        .replace("×", "x")
        .replace(" ", "")
    )
    return key if key in DISPLAY_SIZES else DEFAULT_DISPLAY_SIZE


def generation_aspect_for_display_size(size: Any) -> str:
    """Nearest studio/Gemini aspect for a MediaGo display pixel size."""
    w, h = DISPLAY_SIZES[normalize_display_size(size)]
    ratio = w / float(h)
    if ratio >= 1.6:
        return "16:9"
    if ratio <= 0.6:
        return "9:16"
    if ratio >= 1.15:
        return "4:3"
    if ratio <= 0.87:
        return "3:4"
    return "1:1"


def _num(v: Any) -> Optional[float]:
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _cents_to_usd(cents: Optional[int]) -> Optional[float]:
    if cents is None:
        return None
    return round(int(cents) / 100.0, 2)


def _usd_to_cents(amount: Any) -> Optional[int]:
    n = _num(amount)
    if n is None:
        return None
    return int(round(n * 100))


def _usd_label(n: Optional[float]) -> str:
    if n is None:
        return "—"
    v = float(n)
    if abs(v - round(v)) < 0.005:
        return f"${int(round(v))}"
    return f"${v:.2f}"


def _vs_target_fields(cpa: Optional[float], target_cpa: Optional[float]) -> Dict[str, Any]:
    vs_target = None
    vs_target_ratio = None
    over_target = False
    if target_cpa and target_cpa > 0 and cpa is not None:
        vs_target = cpa - target_cpa
        vs_target_ratio = cpa / target_cpa
        over_target = cpa > target_cpa
    return {
        "target_cpa": target_cpa,
        "vs_target": vs_target,
        "vs_target_ratio": vs_target_ratio,
        "over_target": over_target,
    }


def score_source_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    min_spend: float = 1.0,
    target_cpa: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Aggregate site rows and score them vs the account average.

    Weight is ``account_cpa / site_cpa`` when both have conversions (1.0 =
    average, higher is better). Sites with spend but no conversions get
    weight 0 and ``flag="no_conv"``. Among sites meeting ``min_spend``,
    the bottom quartile by weight is flagged ``bottom_quartile``.

    When ``target_cpa`` is set, each row also gets ``vs_target`` /
    ``over_target`` against that campaign target.
    """
    buckets: Dict[str, Dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        sid = str(raw.get("site_id") or raw.get("id") or "").strip()
        if not sid:
            continue
        b = buckets.setdefault(
            sid,
            {
                "site_id": sid,
                "site_name": raw.get("site_name") or raw.get("domain_name") or sid,
                "spend": 0.0,
                "clicks": 0,
                "impressions": 0,
                "conversions": 0.0,
                "purchases": 0.0,
                "leads": 0.0,
                "revenue": 0.0,
            },
        )
        name = raw.get("site_name") or raw.get("domain_name")
        if name:
            b["site_name"] = name
        b["spend"] += _num(raw.get("spend")) or 0.0
        b["clicks"] += int(_num(raw.get("click") or raw.get("clicks")) or 0)
        b["impressions"] += int(_num(raw.get("impression") or raw.get("impressions")) or 0)
        b["conversions"] += _num(raw.get("conversion") or raw.get("conversions")) or 0.0
        b["purchases"] += _num(raw.get("cv_purchase")) or 0.0
        b["leads"] += _num(raw.get("cv_lead")) or 0.0
        spend_row = _num(raw.get("spend")) or 0.0
        rev = _num(raw.get("value") or raw.get("revenue"))
        roas = _num(raw.get("roas"))
        if rev:
            b["revenue"] += rev
        elif roas and spend_row:
            b["revenue"] += roas * spend_row

    total_spend = sum(b["spend"] for b in buckets.values())
    total_conv = sum(b["conversions"] for b in buckets.values())
    total_clicks = sum(b["clicks"] for b in buckets.values())
    account_cpa = (total_spend / total_conv) if total_conv > 0 else None
    account_cpc = (total_spend / total_clicks) if total_clicks > 0 else None

    scored: List[Dict[str, Any]] = []
    for b in buckets.values():
        spend = b["spend"]
        conv = b["conversions"]
        clicks = b["clicks"]
        cpa = (spend / conv) if conv > 0 else None
        cpc = (spend / clicks) if clicks > 0 else None
        roas = (b["revenue"] / spend) if spend and b["revenue"] else None
        flag = ""
        if spend >= min_spend and conv <= 0:
            weight = 0.0
            flag = "no_conv"
        elif account_cpa and cpa:
            weight = account_cpa / cpa
        elif account_cpc and cpc:
            weight = account_cpc / cpc
        else:
            weight = 1.0
        scored.append(
            {
                **b,
                "cpa": cpa,
                "cpc": cpc,
                "roas": roas,
                "weight": round(float(weight), 4),
                "spend_share": (spend / total_spend) if total_spend else 0.0,
                "flag": flag,
                "account_cpa": account_cpa,
                "account_cpc": account_cpc,
                **_vs_target_fields(cpa, target_cpa),
            }
        )

    eligible = [r for r in scored if r["spend"] >= min_spend]
    eligible.sort(key=lambda r: r["weight"])
    cutoff_n = max(1, len(eligible) // 4) if eligible else 0
    bottom_ids = {r["site_id"] for r in eligible[:cutoff_n]}
    for r in scored:
        if r["site_id"] in bottom_ids and r["flag"] != "no_conv":
            r["flag"] = "bottom_quartile"
        elif r["site_id"] in bottom_ids and r["flag"] == "no_conv":
            r["flag"] = "no_conv,bottom_quartile"

    scored.sort(key=lambda r: r["spend"], reverse=True)
    return scored


def recommend_sites_to_cut(
    rows: Iterable[Dict[str, Any]],
    target_cpa: Optional[float],
    *,
    over_multiple: float = DEFAULT_CUT_OVER_MULTIPLE,
    over_spend_floor: float = DEFAULT_CUT_OVER_SPEND_FLOOR,
    no_conv_spend_floor: float = DEFAULT_CUT_NO_CONV_SPEND_FLOOR,
) -> List[Dict[str, Any]]:
    """Deterministic cut list: high CPA vs target, or spend with no conversions.

    Always computed in-process so recommendations work without Gemini.
    Does not apply blocks — callers must confirm.
    """
    target = _num(target_cpa)
    recs: List[Dict[str, Any]] = []
    seen = set()
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        sid = str(raw.get("site_id") or "").strip()
        if not sid or sid in seen:
            continue
        spend = _num(raw.get("spend")) or 0.0
        conv = _num(raw.get("conversions") or raw.get("conversion")) or 0.0
        cpa = _num(raw.get("cpa"))
        if cpa is None and conv > 0 and spend:
            cpa = spend / conv
        name = raw.get("site_name") or raw.get("domain_name") or sid
        rule = ""
        reason = ""
        if conv <= 0 and spend >= no_conv_spend_floor:
            rule = "no_conv"
            reason = f"0 conversions, {_usd_label(spend)} spend"
        elif (
            target
            and target > 0
            and cpa is not None
            and cpa > target * over_multiple
            and spend >= over_spend_floor
        ):
            rule = "high_cpa"
            reason = (
                f"CPA {_usd_label(cpa)} vs target {_usd_label(target)}, "
                f"{_usd_label(spend)} spend"
            )
        if not rule:
            continue
        seen.add(sid)
        recs.append(
            {
                "site_id": sid,
                "site_name": name,
                "domain_name": name,
                "rule": rule,
                "reason": reason,
                "cpa": cpa,
                "spend": spend,
                "conversions": conv,
                "target_cpa": target,
            }
        )
    recs.sort(key=lambda r: r["spend"], reverse=True)
    return recs


def gemini_cut_note(
    recs: Sequence[Dict[str, Any]],
    target_cpa: Optional[float],
    *,
    call_gemini: Optional[Callable[[str], Optional[str]]] = None,
) -> Optional[str]:
    """Optional short ranked note. Returns None without a key or on failure."""
    if not recs:
        return None
    caller = call_gemini
    if caller is None:
        key = (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_GENAI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or ""
        ).strip()
        if not key:
            return None
        caller = lambda prompt, _key=key: _gemini_cut_text(prompt, _key)
    lines = []
    for r in recs[:12]:
        lines.append(
            f"- {r.get('site_name') or r.get('site_id')}: {r.get('reason') or r.get('rule')}"
        )
    prompt = (
        "You are helping a media buyer cut losing publishers. "
        f"Campaign target CPA is {_usd_label(target_cpa)}. "
        "Here are rule-based cut candidates:\n"
        + "\n".join(lines)
        + "\nWrite 1-3 short sentences ranking which to cut first and why. "
        "Do not invent sites. Do not tell the user to auto-block."
    )
    try:
        text = caller(prompt)
    except Exception as exc:
        logger.warning("gemini cut note failed: %s", exc)
        return None
    if not text:
        return None
    return str(text).strip() or None


def _gemini_cut_text(prompt: str, api_key: str) -> Optional[str]:
    import requests

    model = (
        os.environ.get("AD_STUDIO_GEMINI_FLASH_MODEL")
        or os.environ.get("AD_STUDIO_GEMINI_MODEL")
        or "gemini-3.1-flash"
    )
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 220},
    }
    try:
        resp = requests.post(url, params={"key": api_key}, json=body, timeout=20)
    except requests.RequestException as exc:
        logger.warning("gemini cut note network: %s", exc)
        return None
    if resp.status_code != 200:
        logger.warning("gemini cut note HTTP %s %s", resp.status_code, resp.text[:300])
        return None
    try:
        parts = resp.json()["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts).strip()
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def rollup_campaign_source_stats(
    campaigns: Iterable[Dict[str, Any]],
    report_rows: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge campaign objects (target CPA) with live report spend/conv."""
    buckets: Dict[str, Dict[str, Any]] = {}
    for c in campaigns or []:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id") or c.get("campaign_id") or "").strip()
        if not cid:
            continue
        buckets[cid] = {
            "id": cid,
            "name": c.get("name") or c.get("campaign_name") or "",
            "status": c.get("status") or "",
            "target_cpa": _num(c.get("target_cpa")),
            "spend": 0.0,
            "clicks": 0,
            "conversions": 0.0,
        }
    for r in report_rows or []:
        if not isinstance(r, dict):
            continue
        cid = str(r.get("campaign_id") or r.get("id") or "").strip()
        if not cid:
            continue
        b = buckets.setdefault(
            cid,
            {
                "id": cid,
                "name": r.get("name") or r.get("campaign_name") or "",
                "status": r.get("status") or "",
                "target_cpa": _num(r.get("target_cpa")),
                "spend": 0.0,
                "clicks": 0,
                "conversions": 0.0,
            },
        )
        if r.get("name") and not b.get("name"):
            b["name"] = r.get("name")
        b["spend"] += _num(r.get("spend")) or 0.0
        b["clicks"] += int(_num(r.get("clicks") or r.get("click")) or 0)
        b["conversions"] += _num(r.get("conversions") or r.get("conversion")) or 0.0
    out: List[Dict[str, Any]] = []
    for b in buckets.values():
        spend = b["spend"]
        conv = b["conversions"]
        cpa = (spend / conv) if conv > 0 else None
        out.append({**b, "cpa": cpa, **_vs_target_fields(cpa, b.get("target_cpa"))})
    out.sort(key=lambda r: r["spend"], reverse=True)
    return out


_OPTIMIZATION_EVENTS = {
    "1": "view_content",
    "2": "app_install",
    "3": "complete_registration",
    "4": "add_to_cart",
    "5": "add_payment_info",
    "6": "search",
    "7": "start_checkout",
    "8": "purchase",
    "9": "add_to_wishlist",
    "10": "lead",
    "-1": "default",
}

# Inverse of ``_OPTIMIZATION_EVENTS`` plus human labels from the account pixel list.
_CONVERSION_NAME_TO_OPTIMIZATION = {
    "view_content": "1",
    "view content": "1",
    "app_install": "2",
    "app install": "2",
    "complete_registration": "3",
    "complete registration": "3",
    "add_to_cart": "4",
    "add to cart": "4",
    "add_payment_info": "5",
    "add payment info": "5",
    "search": "6",
    "start_checkout": "7",
    "start checkout": "7",
    "initiate_checkout": "7",
    "purchase": "8",
    "add_to_wishlist": "9",
    "add to wishlist": "9",
    "lead": "10",
    "default": "-1",
    "default_optimization": "-1",
}


def optimization_type_for_conversion(name: Any) -> str:
    """Map a MediaGo conversion name / pixel to campaign ``optimization_type``.

    Create-campaign docs: "1" View Content … "8" Purchase … "10" Lead, "-1" default.
    Account pixels identify conversions by ``conversion_name`` (e.g. ``purchase``),
    not a numeric pixel id.
    """
    raw = str(name or "").strip()
    if not raw:
        return "-1"
    if raw in _OPTIMIZATION_EVENTS:
        return raw
    key = raw.lower().replace("-", "_")
    if key in _CONVERSION_NAME_TO_OPTIMIZATION:
        return _CONVERSION_NAME_TO_OPTIMIZATION[key]
    spaced = key.replace("_", " ")
    if spaced in _CONVERSION_NAME_TO_OPTIMIZATION:
        return _CONVERSION_NAME_TO_OPTIMIZATION[spaced]
    return "-1"


def normalize_account_pixel(px: Dict[str, Any], account_id: str = "") -> Dict[str, Any]:
    """Flatten a ``GET /manage/v1/account`` pixel row into catalog shape."""
    conv = str(
        px.get("conversion_name")
        or px.get("category")
        or px.get("pixel_id")
        or ""
    ).strip()
    if not conv:
        return {}
    name = str(px.get("category") or conv).strip() or conv
    opt = optimization_type_for_conversion(
        px.get("optimization_type") or conv or px.get("category")
    )
    status_raw = px.get("status")
    return {
        "pixel_id": conv,
        "name": name,
        "conversion_name": conv,
        "optimization_type": opt,
        "include_in_total_conversion": px.get("include_in_total_conversion"),
        "status": status_raw,
        "ad_account_id": str(account_id) if account_id else "",
        "source": "mediago",
        "raw": px,
    }


class MediaGoAdapter:
    platform = "mediago"
    label = "MediaGo"
    currency = "USD"
    supports_ad_set_scope = False

    def __init__(
        self,
        client: MediaGoClient,
        account_ids: Optional[Sequence[Union[int, str]]] = None,
        default_currency: Optional[str] = None,
    ):
        self.client = client
        self.account_ids = [str(x) for x in (account_ids or [])]
        if default_currency:
            self.currency = default_currency.upper()
        self._account_pixels: Dict[str, List[Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Auth / accounts
    # ------------------------------------------------------------------
    def verify(self) -> None:
        self.client.verify()

    def get_accounts(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for acc in self.client.get_accounts():
            aid = acc.get("account_id") or acc.get("id")
            if not aid:
                continue
            aid = str(aid)
            if self.account_ids and aid not in self.account_ids:
                continue
            pixels = acc.get("pixels") if isinstance(acc.get("pixels"), list) else []
            self._account_pixels[aid] = pixels
            out.append(
                {
                    "id": aid,
                    "name": acc.get("account_name") or acc.get("name") or f"MediaGo {aid}",
                    "currency": self.currency,
                    "pixels": pixels,
                    "raw": acc,
                }
            )
        return out

    # ------------------------------------------------------------------
    # Hierarchy reads
    # ------------------------------------------------------------------
    def get_campaigns(self, account_id: str) -> List[Dict[str, Any]]:
        rows = self.client.list_campaigns(account_id)
        return [self._normalize_campaign(c, account_id) for c in rows]

    def get_ad_groups(self, account_id: str, campaign_id: str) -> List[Dict[str, Any]]:
        return []

    def get_ads(self, account_id: str, parent_id: str) -> List[Dict[str, Any]]:
        rows = self.client.list_ads(account_id, campaign_ids=[parent_id])
        return [self._normalize_ad(a, account_id, parent_id) for a in rows]

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def create_campaign(self, account_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = self.client.create_campaign(account_id, self._prepare_campaign_payload(payload))
        if isinstance(body, dict):
            return body
        return {"raw": body}

    def update_status(
        self,
        level: str,
        entity_id: str,
        enabled: bool,
        *,
        account_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not account_id:
            raise ValueError("MediaGo update_status requires account_id")
        lvl = (level or "").lower()
        if lvl in ("campaign", "ad_set", "ad_group"):
            body = self.client.set_campaign_status(account_id, [entity_id], enabled)
        elif lvl == "ad":
            body = self.client.set_ad_status(account_id, [entity_id], enabled)
        else:
            raise ValueError(f"Unknown level {level}")
        return body if isinstance(body, dict) else {"raw": body}

    def update_budget(
        self,
        level: str,
        entity_id: str,
        *,
        budget_cents: int,
        budget_type: str = "DAILY",
        account_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not account_id:
            raise ValueError("MediaGo update_budget requires account_id")
        details = self.client.get_campaign_detail(account_id, entity_id)
        current = details[0] if details else {}
        daily = _cents_to_usd(budget_cents) or 20.0
        payload: Dict[str, Any] = {
            "campaign_id": str(entity_id),
            "status": current.get("status", 1),
            "day_parting": current.get("day_parting") or _all_hours_dayparting(),
            "end_time": current.get("end_time") or "2030-01-01 00:00:00",
            "daily_cap": daily,
            "spend_limit": current.get("spend_limit") or max(daily * 30, daily),
            "spend_mode": current.get("spend_mode", 0),
        }
        if current.get("cpc") is not None:
            payload["cpc"] = current.get("cpc")
        body = self.client.update_campaign(account_id, payload)
        return body if isinstance(body, dict) else {"raw": body}

    def block_sites(
        self,
        account_id: str,
        sites: Sequence[Dict[str, Any]],
        *,
        campaign_id: Optional[str] = None,
        block: bool = True,
    ) -> List[Any]:
        """Apply a site block/unblock in chunks of 100 (API limit)."""
        results: List[Any] = []
        chunk: List[Dict[str, Any]] = []
        for s in sites:
            sid = s.get("site_id")
            name = s.get("domain_name") or s.get("site_name") or ""
            if sid in (None, "", 0, "0"):
                continue
            try:
                sid_int = int(sid)
            except (TypeError, ValueError):
                continue
            if sid_int == 0:
                continue
            chunk.append({"site_id": sid_int, "domain_name": name})
            if len(chunk) >= 100:
                results.append(self._flush_block(account_id, chunk, campaign_id, block))
                chunk = []
        if chunk:
            results.append(self._flush_block(account_id, chunk, campaign_id, block))
        return results

    def _flush_block(
        self,
        account_id: str,
        chunk: List[Dict[str, Any]],
        campaign_id: Optional[str],
        block: bool,
    ) -> Any:
        if campaign_id:
            return self.client.block_campaign_sites(
                account_id, campaign_id, chunk, block=block
            )
        return self.client.block_account_sites(account_id, chunk, block=block)

    def fetch_site_report_rows(
        self,
        account_id: str,
        start: date,
        end: date,
        *,
        timezone: str = "est",
    ) -> List[Dict[str, Any]]:
        """Pull site-dimension rows, day-by-day (account API max window is 1 day)."""
        rows: List[Dict[str, Any]] = []
        day = start
        # Inclusive end; cap the loop at 31 days to stay polite to QPS=10.
        guard = 0
        while day <= end and guard < 31:
            ds = day.strftime("%Y-%m-%d")
            rows.extend(
                list(
                    self.client.account_site_report(
                        account_id, ds, ds, timezone=timezone
                    )
                )
            )
            day += timedelta(days=1)
            guard += 1
        return rows

    def fetch_campaign_site_report_rows(
        self,
        account_id: str,
        campaign_id: str,
        start: date,
        end: date,
        *,
        timezone: str = "est",
    ) -> List[Dict[str, Any]]:
        """Pull campaign site rows in ≤7-day windows (API max), cap 31 days."""
        rows: List[Dict[str, Any]] = []
        cursor = start
        consumed = 0
        while cursor <= end and consumed < 31:
            remaining = min(7, 31 - consumed)
            chunk_end = min(cursor + timedelta(days=remaining - 1), end)
            rows.extend(
                list(
                    self.client.campaign_site_report(
                        account_id,
                        campaign_id,
                        cursor.strftime("%Y-%m-%d"),
                        chunk_end.strftime("%Y-%m-%d"),
                        timezone=timezone,
                    )
                )
            )
            consumed += (chunk_end - cursor).days + 1
            cursor = chunk_end + timedelta(days=1)
        return rows

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def fetch_report_rows(
        self,
        account_id: str,
        scope: str,
        start: date,
        end: date,
    ) -> List[Dict[str, Any]]:
        start_s = start.strftime("%Y-%m-%d")
        end_s = end.strftime("%Y-%m-%d")
        lvl = (scope or "campaign").lower()
        if lvl in ("ad", "ads"):
            raw = list(self.client.ad_daily_report(account_id, start_s, end_s))
            return [self._canonicalize_report_row(r, "ad") for r in raw]
        raw = list(self.client.campaign_daily_report(account_id, start_s, end_s))
        return [self._canonicalize_report_row(r, "campaign") for r in raw]

    # ------------------------------------------------------------------
    # Events / pixels
    # ------------------------------------------------------------------
    def list_pixels(self, account_id: str) -> List[Dict[str, Any]]:
        """Conversion pixels for ``account_id`` (MediaGo has no numeric pixel id).

        Hits ``GET /manage/v1/account`` so client-level tokens still see the
        ``pixels`` array. Falls back to whatever ``get_accounts`` cached.
        """
        aid = str(account_id or "")
        raw: List[Dict[str, Any]] = []
        try:
            raw = self.client.list_account_pixels(aid) or []
        except Exception:
            raw = []
        if not raw:
            cached = self._account_pixels.get(aid)
            if cached:
                raw = list(cached)
            else:
                try:
                    for acc in self.get_accounts():
                        if acc["id"] == aid:
                            raw = list(acc.get("pixels") or [])
                            break
                except Exception:
                    raw = []
        out: List[Dict[str, Any]] = []
        seen = set()
        for px in raw:
            if not isinstance(px, dict):
                continue
            n = normalize_account_pixel(px, aid)
            pid = n.get("pixel_id")
            if not pid or pid in seen:
                continue
            seen.add(pid)
            out.append(n)
        return out

    def list_events(self, account_id: str) -> List[Dict[str, Any]]:
        pixels = self.list_pixels(account_id)
        out: List[Dict[str, Any]] = []
        for px in pixels:
            name = px.get("conversion_name") or px.get("name") or "event"
            out.append(
                {
                    "tracking_id": name,
                    "name": px.get("name") or name,
                    "event_type": name,
                    "pixel_id": px.get("pixel_id") or name,
                    "optimization_type": px.get("optimization_type") or "-1",
                    "tracking_type": "pixel",
                    "status": "ACTIVE" if px.get("status") else "PAUSED",
                    "ad_account_id": account_id,
                    "source": "mediago",
                    "raw": px.get("raw") or px,
                }
            )
        if out:
            return out
        return [
            {
                "tracking_id": key,
                "name": label,
                "event_type": key,
                "pixel_id": key,
                "optimization_type": optimization_type_for_conversion(key),
                "source": "mediago",
                "ad_account_id": account_id,
            }
            for key, label in (
                ("purchase", "Purchase"),
                ("lead", "Lead"),
                ("view_content", "View Content"),
                ("add_to_cart", "Add to Cart"),
            )
        ]

    def upload_asset(
        self,
        account_id: str,
        file_obj: Union[BinaryIO, bytes],
        filename: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "MediaGo creatives take a public image URL (img), "
            "not a binary upload. Host the file and pass the URL."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _prepare_campaign_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(payload)
        out["creative_type"] = normalize_creative_type(out.get("creative_type"))
        if "daily_budget_cents" in out and "daily_cap" not in out:
            out["daily_cap"] = _cents_to_usd(out.pop("daily_budget_cents"))
        if "spend_limit_cents" in out and "spend_limit" not in out:
            out["spend_limit"] = _cents_to_usd(out.pop("spend_limit_cents"))
        if "cpc_cents" in out and "cpc" not in out:
            out["cpc"] = _cents_to_usd(out.pop("cpc_cents"))
        return out

    @staticmethod
    def _normalize_campaign(c: Dict[str, Any], account_id: str) -> Dict[str, Any]:
        cid = str(c.get("campaign_id") or c.get("id") or "")
        status_raw = c.get("status")
        enabled = status_raw in (1, "1", True, "active", "ACTIVE")
        return {
            "id": cid,
            "name": c.get("campaign_name") or c.get("name") or "",
            "ad_account_id": str(c.get("account_id") or account_id),
            "status": "on" if enabled else "off",
            "enable": enabled,
            "daily_budget": c.get("daily_cap"),
            "daily_budget_cents": _usd_to_cents(c.get("daily_cap")),
            "objective": c.get("objective"),
            "creative_type": c.get("creative_type") or "native",
            "target_cpa": _num(c.get("target_cpa")),
            "ads": c.get("ads") or [],
            "raw": c,
        }

    @staticmethod
    def _normalize_ad(a: Dict[str, Any], account_id: str, campaign_id: str) -> Dict[str, Any]:
        return {
            "id": str(a.get("ad_id") or a.get("id") or ""),
            "name": a.get("ad_name") or a.get("asset_name") or a.get("name") or "",
            "ad_account_id": str(account_id),
            "campaign_id": str(a.get("campaign_id") or campaign_id),
            "status": "",
            "width": a.get("width"),
            "height": a.get("height"),
            "raw": a,
        }

    def _canonicalize_report_row(self, r: Dict[str, Any], scope: str) -> Dict[str, Any]:
        spend = _num(r.get("spend")) or 0.0
        clicks = int(_num(r.get("click") or r.get("clicks")) or 0)
        impressions = int(_num(r.get("impression") or r.get("impressions")) or 0)
        conversions = _num(r.get("conversion") or r.get("conversions") or r.get("cv_purchase")) or 0.0
        purchases = _num(r.get("cv_purchase")) or 0.0
        cpa = _num(r.get("cpa"))
        if cpa is None and spend and conversions:
            cpa = spend / conversions
        ctr = (clicks / impressions * 100.0) if impressions else 0.0
        entity_id = r.get("id") or r.get("campaign_id") or r.get("ad_id")
        name = r.get("name") or r.get("campaign_name") or r.get("ad_name") or "—"
        status_raw = r.get("status")
        status = ""
        if status_raw in (1, "1"):
            status = "on"
        elif status_raw in (0, "0"):
            status = "off"
        events: Dict[str, float] = {}
        for key, dest in (
            ("cv_purchase", "purchase"),
            ("cv_lead", "lead"),
            ("cv_view_content", "view_content"),
            ("cv_add_to_cart", "add_to_cart"),
            ("cv_add_to_car", "add_to_cart"),
            ("cv_start_checkout", "initiate_checkout"),
        ):
            v = _num(r.get(key))
            if v:
                events[dest] = v
        return {
            **r,
            "scope": scope,
            "id": str(entity_id) if entity_id is not None else None,
            "name": name,
            "parent_id": str(r.get("campaign_id") or "") if scope == "ad" else None,
            "status": status,
            "spend": spend,
            "impressions": impressions,
            "clicks": clicks,
            "ctr": ctr,
            "conversions": conversions or purchases,
            "cpa": cpa,
            "roas": _num(r.get("roas")),
            "value": None,
            "events": events,
            "campaign_id": str(r.get("id") or r.get("campaign_id") or "") or None,
            "ad_set_id": None,
            "ad_id": str(r.get("ad_id") or "") or None if scope == "ad" else None,
            "budget": None,
            "budget_type": "DAILY",
            "raw": r,
        }


def _all_hours_dayparting() -> List[List[int]]:
    return [[1] * 24 for _ in range(7)]


__all__ = [
    "MediaGoAdapter",
    "DISPLAY_SIZES",
    "DEFAULT_DISPLAY_SIZE",
    "DISPLAY_SIZE_LABELS",
    "normalize_creative_type",
    "normalize_display_size",
    "generation_aspect_for_display_size",
    "score_source_rows",
    "recommend_sites_to_cut",
    "gemini_cut_note",
    "rollup_campaign_source_stats",
    "normalize_account_pixel",
    "optimization_type_for_conversion",
]

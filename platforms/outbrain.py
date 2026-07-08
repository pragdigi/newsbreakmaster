"""
Outbrain (Amplify) adapter — wraps :class:`outbrain_api.OutbrainClient`
behind the ``AdPlatformAdapter`` contract.

Outbrain's hierarchy is Marketer → (Budget) + Campaign → PromotedLink. There
is no ad-set / ad-group layer, so ``supports_ad_set_scope`` is False and the
ad-set scope collapses onto the campaign (same shape as SmartNews AMv1).

Money: Outbrain works in plain currency floats (``cpc: 0.55`` dollars,
``budget.amount: 500`` dollars); our internal "cents" integers are converted
at the edge with ``cents_to_amount`` / ``amount_to_cents``.
"""
from __future__ import annotations

from datetime import date
from typing import Any, BinaryIO, Dict, List, Optional, Sequence, Union

from outbrain_api import (
    OutbrainAPIError,
    OutbrainClient,
    amount_to_cents,
    cents_to_amount,
)


def _num(v: Any) -> Optional[float]:
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _get_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    """Outbrain report rows nest stats under ``metrics``/``summary`` or inline."""
    for key in ("metrics", "summary", "metric"):
        v = row.get(key)
        if isinstance(v, dict):
            return v
    return row


def _get_meta(row: Dict[str, Any]) -> Dict[str, Any]:
    md = row.get("metadata")
    if isinstance(md, dict):
        return md
    return row


class OutbrainAdapter:
    platform = "outbrain"
    label = "Outbrain / Teads"
    currency = "USD"
    supports_ad_set_scope = False

    def __init__(
        self,
        client: OutbrainClient,
        marketer_ids: Optional[Sequence[Union[int, str]]] = None,
        default_currency: Optional[str] = None,
    ):
        self.client = client
        self.marketer_ids = [str(x) for x in (marketer_ids or [])]
        if default_currency:
            self.currency = default_currency.upper()
        self._marketer_currency: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Auth / accounts
    # ------------------------------------------------------------------
    def verify(self) -> None:
        self.client.verify()

    def get_accounts(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for m in self.client.get_marketers():
            mid = m.get("id")
            if not mid:
                continue
            mid = str(mid)
            if self.marketer_ids and mid not in self.marketer_ids:
                continue
            currency = (m.get("currency") or self.currency).upper()
            self._marketer_currency[mid] = currency
            out.append(
                {
                    "id": mid,
                    "name": m.get("name") or f"Outbrain {mid}",
                    "currency": currency,
                    "enabled": m.get("enabled"),
                    "raw": m,
                }
            )
        return out

    def _currency_for(self, marketer_id: Union[int, str]) -> str:
        return self._marketer_currency.get(str(marketer_id), self.currency)

    # ------------------------------------------------------------------
    # Hierarchy reads
    # ------------------------------------------------------------------
    def get_campaigns(self, account_id: str) -> List[Dict[str, Any]]:
        rows = list(self.client.iter_campaigns(account_id))
        return [self._normalize_campaign(c, account_id) for c in rows]

    def get_ad_groups(self, account_id: str, campaign_id: str) -> List[Dict[str, Any]]:
        # Outbrain has no ad-set layer.
        return []

    def get_ads(self, account_id: str, parent_id: str) -> List[Dict[str, Any]]:
        """``parent_id`` is a campaign id (no ad-set layer on Outbrain)."""
        rows = list(self.client.iter_promoted_links(parent_id))
        return [self._normalize_promoted_link(p, account_id, parent_id) for p in rows]

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def create_budget(self, account_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.client.create_budget(account_id, self._prepare_budget_payload(payload))

    def create_campaign(self, account_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        # ``account_id`` is implied by the budgetId on Outbrain's global
        # create endpoint; kept in the signature for contract parity.
        return self.client.create_campaign(self._prepare_campaign_payload(payload, account_id))

    def update_status(
        self,
        level: str,
        entity_id: str,
        enabled: bool,
        *,
        account_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        lvl = (level or "").lower()
        body = {"enabled": bool(enabled)}
        if lvl in ("campaign", "ad_set", "ad_group"):
            return self.client.update_campaign(entity_id, body)
        if lvl == "ad":
            return self.client.update_promoted_link(entity_id, body)
        raise ValueError(f"Unknown level {level}")

    def update_budget(
        self,
        level: str,
        entity_id: str,
        *,
        budget_cents: int,
        budget_type: str = "DAILY",
        account_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Adjust spend.

        Outbrain keeps budgets as standalone objects referenced by a
        campaign's ``budgetId``. For ``level`` campaign/ad_set we resolve the
        campaign's budget and update it (``dailyTarget`` for DAILY,
        ``amount`` for TOTAL). Pass ``level="budget"`` to update a budget id
        directly.
        """
        amount = cents_to_amount(int(budget_cents))
        bt = (budget_type or "DAILY").upper()
        body: Dict[str, Any] = {}
        if bt in ("TOTAL", "LIFETIME"):
            body["amount"] = amount
        else:
            body["dailyTarget"] = amount

        lvl = (level or "").lower()
        if lvl == "budget":
            return self.client.update_budget(entity_id, body)
        if lvl in ("campaign", "ad_set", "ad_group"):
            camp = self.client.get_campaign(entity_id)
            budget_id = camp.get("budgetId") or (camp.get("budget") or {}).get("id")
            if not budget_id:
                raise OutbrainAPIError(
                    f"campaign {entity_id} has no budgetId to update"
                )
            return self.client.update_budget(str(budget_id), body)
        raise ValueError(f"Outbrain update_budget got unsupported level {level}")

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
        s = (scope or "").lower()
        date_from = start.strftime("%Y-%m-%d")
        date_to = end.strftime("%Y-%m-%d")
        if s in ("ad", "ads", "promoted_link"):
            # Per-promoted-link reporting needs a campaign id; the rules
            # engine drives this per-campaign, so without one we return the
            # campaign rollup as ads are not addressable account-wide here.
            rows: List[Dict[str, Any]] = []
            for camp in self.client.iter_campaigns(account_id):
                cid = str(camp.get("id") or "")
                if not cid:
                    continue
                for r in self.client.report_promoted_links(
                    cid, date_from=date_from, date_to=date_to
                ):
                    rows.append(self._canonicalize_report_row(r, "ad", account_id, cid))
            return rows
        # campaign / ad_set both map to the campaign rollup
        rows = self.client.report_campaigns(
            account_id, date_from=date_from, date_to=date_to
        )
        canonical = "ad_set" if s in ("ad_set", "ad_group") else "campaign"
        return [self._canonicalize_report_row(r, canonical, account_id, None) for r in rows]

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def list_events(self, account_id: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for c in self.client.list_conversions(account_id):
            cid = c.get("id") or c.get("conversionId")
            if not cid:
                continue
            out.append(
                {
                    "tracking_id": str(cid),
                    "name": c.get("name") or f"Conversion {cid}",
                    "event_type": c.get("conversionEvent") or c.get("type") or "conversion",
                    "pixel_id": None,
                    "tracking_type": "outbrain_conversion",
                    "status": "ACTIVE" if c.get("enabled", True) else "OFF",
                    "ad_account_id": str(account_id),
                    "source": "outbrain",
                    "raw": c,
                }
            )
        return out

    def search_locations(self, term: str, *, limit: int = 10) -> List[Dict[str, Any]]:
        return self.client.search_locations(term, limit=limit)

    def upload_asset(
        self,
        account_id: str,
        file_obj: Union[BinaryIO, bytes],
        filename: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Outbrain has no standalone asset store — images attach to a
        promoted link at creation time. The bulk launcher uploads bytes
        directly via ``create_promoted_link_with_image``; this stub exists for
        contract parity and returns the raw bytes echoed back.
        """
        data = file_obj if isinstance(file_obj, (bytes, bytearray)) else file_obj.read()
        return {"bytes": bytes(data), "filename": filename}

    # ------------------------------------------------------------------
    # Payload helpers
    # ------------------------------------------------------------------
    def _prepare_budget_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(payload)
        for key_in, key_out in (
            ("amount_cents", "amount"),
            ("daily_target_cents", "dailyTarget"),
        ):
            if key_in in out and out[key_in] is not None:
                out[key_out] = cents_to_amount(out.pop(key_in))
        return out

    def _prepare_campaign_payload(self, payload: Dict[str, Any], account_id: str) -> Dict[str, Any]:
        out = dict(payload)
        # cpc may arrive as cents → convert to dollar float.
        if "cpc_cents" in out and out["cpc_cents"] is not None:
            out["cpc"] = cents_to_amount(out.pop("cpc_cents"))
        return out

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------
    @staticmethod
    def _status_from_enabled(row: Dict[str, Any]) -> str:
        enabled = row.get("enabled")
        if enabled is True:
            return "on"
        if enabled is False:
            return "off"
        return ""

    def _normalize_campaign(self, c: Dict[str, Any], account_id: str) -> Dict[str, Any]:
        cur = self._currency_for(account_id)
        budget = c.get("budget") if isinstance(c.get("budget"), dict) else {}
        amount = _num(budget.get("amount"))
        daily = _num(budget.get("dailyTarget") or budget.get("dailyAmount"))
        budget_cents = None
        budget_type = None
        if daily is not None:
            budget_cents = int(round(daily * 100))
            budget_type = "DAILY"
        elif amount is not None:
            budget_cents = int(round(amount * 100))
            budget_type = "TOTAL"
        on_air = c.get("onAirStatus") or c.get("liveStatus") or {}
        approval = ""
        if isinstance(on_air, dict):
            approval = on_air.get("status") or on_air.get("onAirStatus") or ""
        return {
            "id": str(c.get("id") or ""),
            "name": c.get("name") or "",
            "ad_account_id": str(account_id),
            "status": self._status_from_enabled(c),
            "enable": bool(c.get("enabled")),
            "approval_status": approval,
            "cpc": _num(c.get("cpc")),
            "budget_id": c.get("budgetId") or budget.get("id"),
            "daily_budget_cents": budget_cents if budget_type == "DAILY" else None,
            "daily_budget": daily,
            "spending_limit_cents": int(round(amount * 100)) if amount is not None else None,
            "objective": c.get("objective"),
            "currency": cur,
            "raw": c,
        }

    def _normalize_promoted_link(self, p: Dict[str, Any], account_id: str, campaign_id: str) -> Dict[str, Any]:
        image = p.get("imageMetadata") if isinstance(p.get("imageMetadata"), dict) else {}
        return {
            "id": str(p.get("id") or ""),
            "name": p.get("text") or p.get("name") or "",
            "ad_account_id": str(account_id),
            "campaign_id": str(p.get("campaignId") or campaign_id),
            "status": self._status_from_enabled(p),
            "enable": bool(p.get("enabled")),
            "approval_status": (p.get("status") or p.get("onAirStatus") or ""),
            "landing_page_url": p.get("url"),
            "text": p.get("text"),
            "cta_label": (p.get("callToAction") or None),
            "image_url": (
                p.get("imageUrl")
                or p.get("cachedImageUrl")
                or image.get("url")
            ),
            "raw": p,
        }

    def _canonicalize_report_row(
        self,
        r: Dict[str, Any],
        scope: str,
        account_id: str,
        campaign_id: Optional[str],
    ) -> Dict[str, Any]:
        meta = _get_meta(r)
        metrics = _get_metrics(r)

        entity_id = (
            meta.get("id")
            or r.get("id")
            or r.get("campaignId")
            or r.get("promotedLinkId")
        )
        name = meta.get("name") or r.get("name") or r.get("campaignName") or ""

        spend = _num(metrics.get("spend")) or 0.0
        impressions = int(_num(metrics.get("impressions")) or 0)
        clicks = int(_num(metrics.get("clicks")) or 0)
        ctr = _num(metrics.get("ctr"))
        if ctr is None and impressions:
            ctr = clicks / impressions * 100.0
        elif ctr is not None and ctr < 1:
            ctr = ctr * 100.0

        conversions = (
            _num(metrics.get("conversions"))
            or _num(metrics.get("totalConversions"))
            or 0.0
        )
        value = _num(metrics.get("sumValue")) or _num(metrics.get("conversionValue"))
        cpa = _num(metrics.get("cpa"))
        if (cpa is None or cpa == 0) and spend and conversions:
            cpa = spend / conversions
        roas = _num(metrics.get("roas"))
        if roas is None and value and spend:
            roas = value / spend

        if scope == "campaign":
            parent_id = str(account_id)
            cid = str(entity_id) if entity_id is not None else None
            ad_id = None
        elif scope == "ad_set":
            parent_id = str(account_id)
            cid = str(entity_id) if entity_id is not None else None
            ad_id = None
        else:  # ad / promoted link
            parent_id = str(campaign_id or "")
            cid = str(campaign_id or "") or None
            ad_id = str(entity_id) if entity_id is not None else None

        return {
            **r,
            "scope": scope,
            "id": str(entity_id) if entity_id is not None else None,
            "name": name,
            "parent_id": parent_id,
            "status": "",
            "spend": spend,
            "impressions": impressions,
            "clicks": clicks,
            "ctr": float(ctr or 0.0),
            "conversions": conversions,
            "cpa": cpa,
            "roas": roas,
            "value": value,
            "events": {"conversion": conversions} if conversions else {},
            "campaign_id": cid,
            "ad_set_id": cid if scope in ("ad_set", "ad") else None,
            "ad_id": ad_id,
            "budget": None,
            "budget_type": None,
            "raw": r,
        }


__all__ = ["OutbrainAdapter"]

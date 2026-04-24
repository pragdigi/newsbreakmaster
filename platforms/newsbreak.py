"""
NewsBreak adapter: wraps NewsBreakClient and normalises output.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, BinaryIO, Dict, List, Optional, Union

from newsbreak_api import NewsBreakAPIError, NewsBreakClient, unwrap_list_response


class NewsBreakAdapter:
    platform = "newsbreak"
    label = "NewsBreak"
    currency = "USD"
    supports_ad_set_scope = True

    def __init__(self, client: NewsBreakClient, org_ids: Optional[List[str]] = None):
        self.client = client
        self.org_ids = list(org_ids or [])

    # --- auth ---
    def verify(self) -> None:
        self.client.get_ad_accounts(self.org_ids or [])

    # --- accounts ---
    def get_accounts(self) -> List[Dict[str, Any]]:
        raw = self.client.get_ad_accounts(self.org_ids or [])
        return _flatten_ad_accounts(raw)

    # --- campaigns / ad sets / ads ---
    def get_campaigns(self, account_id: str) -> List[Dict[str, Any]]:
        raw = self.client.get_campaigns(account_id)
        return unwrap_list_response(raw)

    def get_ad_groups(self, account_id: str, campaign_id: str) -> List[Dict[str, Any]]:
        raw = self.client.get_ad_sets(campaign_id)
        return unwrap_list_response(raw)

    def get_ads(self, account_id: str, parent_id: str) -> List[Dict[str, Any]]:
        # parent_id is ad_set_id on NewsBreak; /ad/getList also requires
        # adAccountId or the endpoint rejects the request with HTTP 400.
        raw = self.client.get_ads(parent_id, ad_account_id=account_id)
        return unwrap_list_response(raw)

    # --- writes ---
    def create_campaign(self, account_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = {"adAccountId": account_id, **payload}
        return self.client.create_campaign(body)

    def update_status(
        self,
        level: str,
        entity_id: str,
        enabled: bool,
        *,
        account_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        status = "ON" if enabled else "OFF"
        if level == "ad":
            return self.client.update_ad_status(str(entity_id), status)
        if level == "ad_set":
            return self.client.update_ad_set_status(str(entity_id), status)
        if level == "campaign":
            return self.client.update_campaign(str(entity_id), {"status": status})
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
        if level != "ad_set":
            raise ValueError("NewsBreak budget changes target ad-set scope only")
        return self.client.update_ad_set(
            str(entity_id),
            {"budget": int(budget_cents), "budgetType": budget_type},
        )

    # --- reporting ---
    def fetch_report_rows(
        self,
        account_id: str,
        scope: str,
        start: date,
        end: date,
    ) -> List[Dict[str, Any]]:
        from rules_engine import build_report_payload, normalize_report_rows

        if scope == "ad_set":
            dimensions: List[str] = ["AD_SET", "CAMPAIGN"]
        elif scope == "campaign":
            dimensions = ["CAMPAIGN"]
        elif scope == "ad_account":
            dimensions = ["AD_ACCOUNT"]
        else:
            # AD scope: request the full hierarchy so each row carries
            # campaignId + adSetId. Without this, NewsBreak only emits
            # adId + metrics and there's no way to resolve the parent
            # ad set for creative enrichment or winner discovery.
            dimensions = ["AD", "AD_SET", "CAMPAIGN"]

        payload = build_report_payload(account_id, start, end, dimensions[0])
        payload["dimensions"] = dimensions
        try:
            raw = self.client.get_integrated_report(payload)
        except NewsBreakAPIError:
            if len(dimensions) == 1:
                raise
            # Some tenants may reject multi-dim payloads; retry with just
            # the primary dimension so reporting still works even if
            # enrichment has to fall back to the account index path.
            fallback_payload = build_report_payload(
                account_id, start, end, dimensions[0]
            )
            raw = self.client.get_integrated_report(fallback_payload)
        rows = normalize_report_rows(raw)
        return [_canonicalize_row(r, scope) for r in rows]

    # --- events ---
    def list_events(self, account_id: str) -> List[Dict[str, Any]]:
        raw = self.client.get_events(account_id)
        rows = unwrap_list_response(raw)
        if not rows and isinstance(raw, dict):
            data = raw.get("data")
            if isinstance(data, dict):
                inner = data.get("list") or data.get("rows") or data.get("events") or data.get("records")
                if isinstance(inner, list):
                    rows = [x for x in inner if isinstance(x, dict)]
        out: List[Dict[str, Any]] = []
        for r in rows:
            tid = r.get("id") or r.get("eventId") or r.get("trackingId") or r.get("tracking_id")
            if tid is None:
                continue
            out.append(
                {
                    "tracking_id": str(tid),
                    "name": r.get("name") or r.get("eventName") or f"Event {tid}",
                    "event_type": r.get("eventType") or r.get("event_type") or "",
                    "pixel_id": str(r.get("pixelId") or r.get("pixel_id") or "") or None,
                    "tracking_type": r.get("trackingType") or r.get("tracking_type") or "",
                    "status": r.get("status") or "",
                    "ad_account_id": account_id,
                    "source": "newsbreak",
                    "raw": r,
                }
            )
        return out

    # --- assets ---
    def upload_asset(
        self,
        account_id: str,
        file_obj: Union[BinaryIO, bytes],
        filename: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self.client.upload_asset(
            file_obj,
            filename,
            account_id,
            media_name=kwargs.get("media_name"),
            save_to_media_library=kwargs.get("save_to_media_library", True),
        )


def _flatten_ad_accounts(api_response: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not isinstance(api_response, dict):
        return rows
    data = api_response.get("data") or api_response.get("result") or api_response
    groups = data if isinstance(data, list) else data.get("groups") or data.get("list") or []
    if isinstance(groups, dict):
        groups = [groups]
    if not isinstance(groups, list):
        return rows
    for g in groups:
        if not isinstance(g, dict):
            continue
        org_id = g.get("id") or g.get("organizationId")
        accounts = g.get("adAccounts") or g.get("ad_accounts") or []
        if isinstance(accounts, dict):
            accounts = [accounts]
        for a in accounts or []:
            if isinstance(a, dict):
                rows.append({**a, "_org_id": org_id})
    return rows


def _canonicalize_row(r: Dict[str, Any], scope: str) -> Dict[str, Any]:
    """Augment a NewsBreak normalized report row with canonical keys."""
    campaign_id = r.get("campaign_id") or r.get("campaignId")
    ad_set_id = r.get("ad_set_id") or r.get("adSetId")
    ad_id = r.get("ad_id") or r.get("adId")

    if scope == "ad":
        entity_id = ad_id
        parent_id = ad_set_id
        name = r.get("adName") or r.get("ad_name") or r.get("name") or "Ad"
    elif scope == "ad_set":
        entity_id = ad_set_id
        parent_id = campaign_id
        name = r.get("adSetName") or r.get("ad_set_name") or r.get("name") or "Ad set"
    elif scope == "campaign":
        entity_id = campaign_id
        parent_id = None
        name = r.get("campaignName") or r.get("campaign_name") or r.get("name") or "Campaign"
    else:
        entity_id = r.get("id")
        parent_id = None
        name = r.get("name") or ""

    status_raw = str(r.get("status") or r.get("onlineStatus") or "").lower()

    impressions = _num(r.get("impression") or r.get("IMPRESSION") or r.get("impressions"))
    clicks = _num(r.get("click") or r.get("CLICK") or r.get("clicks"))
    ctr = _num(r.get("ctr") or r.get("CTR"))
    if ctr is None and impressions:
        ctr = (clicks or 0) / impressions * 100.0

    events: Dict[str, float] = {}
    for k, v in r.items():
        lk = str(k).lower()
        if lk in {"add_to_cart", "initiate_checkout", "purchase", "view_content", "complete_payment"}:
            fv = _num(v)
            if fv is not None:
                events[lk] = fv

    return {
        **r,
        "scope": scope,
        "id": str(entity_id) if entity_id is not None else None,
        "name": name,
        "parent_id": str(parent_id) if parent_id is not None else None,
        "status": status_raw,
        "impressions": int(impressions or 0),
        "clicks": int(clicks or 0),
        "ctr": float(ctr or 0.0),
        "events": events,
        "budget": _num(r.get("dailyBudget") or r.get("budget")),
        "budget_type": r.get("budgetType") or r.get("budget_type"),
        "raw": r,
    }


def _num(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


__all__ = ["NewsBreakAdapter", "NewsBreakAPIError"]

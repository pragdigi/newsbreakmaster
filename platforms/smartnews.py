"""
SmartNews adapter — wraps SmartNewsClient behind the AdPlatformAdapter contract.

SmartNews AMv1 hierarchy is Account -> Campaign -> Creative, so the generic
``ad_set`` scope is collapsed onto ``campaign`` for all interactions.
All monetary values are in JPY (integer) and budgets are passed through
verbatim (no cents conversion).
"""
from __future__ import annotations

from datetime import date
from typing import Any, BinaryIO, Dict, List, Optional, Union

from smartnews_api import SmartNewsClient, unwrap_data, unwrap_list


class SmartNewsAdapter:
    platform = "smartnews"
    label = "SmartNews"
    currency = "JPY"
    supports_ad_set_scope = False

    def __init__(self, client: SmartNewsClient, account_ids: Optional[List[str]] = None):
        self.client = client
        self.account_ids = list(account_ids or [])

    # ------------------------------------------------------------------
    # Auth / accounts
    # ------------------------------------------------------------------
    def verify(self) -> None:
        """Make a lightweight call to validate the API key."""
        self.client.get_accounts()

    def get_accounts(self) -> List[Dict[str, Any]]:
        body = self.client.get_accounts()
        out: List[Dict[str, Any]] = []
        for acc in unwrap_list(body):
            aid = acc.get("accountId") or acc.get("id")
            if not aid:
                continue
            if self.account_ids and str(aid) not in [str(x) for x in self.account_ids]:
                continue
            out.append(
                {
                    "id": str(aid),
                    "name": acc.get("name") or acc.get("accountName") or f"SmartNews {aid}",
                    "currency": self.currency,
                    "raw": acc,
                }
            )
        return out

    # ------------------------------------------------------------------
    # Campaigns / creatives
    # ------------------------------------------------------------------
    def get_campaigns(self, account_id: str) -> List[Dict[str, Any]]:
        body = self.client.get_campaigns(account_id)
        return [self._normalize_campaign(c, account_id) for c in unwrap_list(body)]

    def get_ad_groups(self, account_id: str, campaign_id: str) -> List[Dict[str, Any]]:
        # SmartNews AMv1 has no ad-group layer; surface the campaign itself so
        # the UI can render a single placeholder row if it expects one.
        body = self.client.get_campaign(campaign_id)
        inner = unwrap_data(body)
        if isinstance(inner, dict):
            return [self._normalize_campaign(inner, account_id)]
        return []

    def get_ads(self, account_id: str, parent_id: str) -> List[Dict[str, Any]]:
        body = self.client.get_creatives(parent_id)
        return [self._normalize_creative(c, parent_id) for c in unwrap_list(body)]

    def create_campaign(self, account_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = self.client.create_campaign(account_id, payload)
        inner = unwrap_data(body)
        return inner if isinstance(inner, dict) else {"raw": body}

    # ------------------------------------------------------------------
    # Status / budget
    # ------------------------------------------------------------------
    def update_status(
        self,
        level: str,
        entity_id: str,
        enabled: bool,
        *,
        account_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        lvl = (level or "").lower()
        if lvl == "ad_set":
            lvl = "campaign"
        if lvl == "campaign":
            body = self.client.update_campaign_enable(str(entity_id), bool(enabled))
        elif lvl == "ad":
            body = self.client.update_creative_enable(str(entity_id), bool(enabled))
        else:
            raise ValueError(f"Unknown level {level}")
        inner = unwrap_data(body)
        return inner if isinstance(inner, dict) else {"raw": body}

    def update_budget(
        self,
        level: str,
        entity_id: str,
        *,
        budget_cents: int,
        budget_type: str = "DAILY",
        account_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """SmartNews budgets are whole JPY — callers still pass ``budget_cents``
        for API consistency, so we treat the value as JPY (no ×100 conversion)."""
        lvl = (level or "").lower()
        if lvl == "ad_set":
            lvl = "campaign"
        if lvl != "campaign":
            raise ValueError(
                f"SmartNews only supports campaign-level budget updates (got {level})"
            )
        amount = int(budget_cents)
        bt = (budget_type or "DAILY").upper()
        payload: Dict[str, Any] = {}
        if bt == "TOTAL" or bt == "LIFETIME":
            payload["totalBudget"] = amount
        else:
            payload["dailyBudget"] = amount
        body = self.client.update_campaign(str(entity_id), payload)
        inner = unwrap_data(body)
        return inner if isinstance(inner, dict) else {"raw": body}

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
        lvl = (scope or "").lower()
        if lvl == "ad_set":
            lvl = "campaign"
        if lvl not in ("account", "campaign", "ad", "creative"):
            lvl = "campaign"

        level_param = {
            "account": "account",
            "campaign": "campaign",
            "ad": "creative",
            "creative": "creative",
        }[lvl]
        since = start.isoformat()
        until = end.isoformat()
        body = self.client.get_account_insights(
            account_id,
            since=since,
            until=until,
            level=level_param,
        )
        rows = unwrap_list(body)
        out: List[Dict[str, Any]] = []
        canonical_scope = "ad" if lvl in ("ad", "creative") else lvl
        for r in rows:
            out.append(_canonicalize_insights_row(r, canonical_scope))
        return out

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def list_events(self, account_id: str) -> List[Dict[str, Any]]:
        return _builtin_events(account_id)

    def upload_asset(
        self,
        account_id: str,
        file_obj: Union[BinaryIO, bytes],
        filename: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        body = self.client.upload_image(account_id, file_obj, filename)
        inner = unwrap_data(body)
        return inner if isinstance(inner, dict) else {"raw": body}

    # ------------------------------------------------------------------
    # Extra SmartNews-specific helpers used by the launcher
    # ------------------------------------------------------------------
    def submit_review(self, campaign_id: str) -> Any:
        return self.client.submit_review(campaign_id)

    def create_creative(self, campaign_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = self.client.create_creative(campaign_id, payload)
        inner = unwrap_data(body)
        return inner if isinstance(inner, dict) else {"raw": body}

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_campaign(c: Dict[str, Any], account_id: str) -> Dict[str, Any]:
        return {
            "id": str(c.get("campaignId") or c.get("id") or ""),
            "name": c.get("name") or "",
            "ad_account_id": str(c.get("accountId") or account_id),
            "status": "ON" if c.get("enable") else "OFF",
            "enable": bool(c.get("enable")),
            "approval_status": c.get("approvalStatus") or "",
            "daily_budget": c.get("dailyBudget"),
            "total_budget": c.get("totalBudget"),
            "bid_amount": c.get("bidAmount"),
            "target_cpa": c.get("targetCpa"),
            "action_type": c.get("actionType"),
            "start_time": c.get("startTime"),
            "end_time": c.get("endTime"),
            "raw": c,
        }

    @staticmethod
    def _normalize_creative(c: Dict[str, Any], campaign_id: str) -> Dict[str, Any]:
        return {
            "id": str(c.get("creativeId") or c.get("id") or ""),
            "name": c.get("name") or "",
            "campaign_id": str(campaign_id),
            "status": "ON" if c.get("enable") else "OFF",
            "enable": bool(c.get("enable")),
            "approval_status": c.get("approvalStatus") or "",
            "title": c.get("title"),
            "text": c.get("text"),
            "imageset": c.get("imageset"),
            "tracking_url": c.get("trackingUrl"),
            "raw": c,
        }


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------
def _builtin_events(account_id: str) -> List[Dict[str, Any]]:
    """SmartNews Insights event names.

    SmartNews doesn't expose a discoverable pixel/event catalogue; these are
    the built-in conversion buckets every account reports against.
    """
    names = [
        ("purchase", "Purchase"),
        ("addToCart", "Add to Cart"),
        ("initiateCheckout", "Initiate Checkout"),
        ("viewContent", "View Content"),
        ("completeRegistration", "Complete Registration"),
        ("lead", "Lead"),
        ("search", "Search"),
        ("addPaymentInfo", "Add Payment Info"),
        ("startTrial", "Start Trial"),
        ("subscribe", "Subscribe"),
        ("install", "Install"),
        ("signUp", "Sign Up"),
        ("contact", "Contact"),
        ("submitForm", "Submit Form"),
        ("download", "Download"),
        ("login", "Login"),
    ]
    return [
        {
            "tracking_id": key,
            "name": label,
            "event_type": key,
            "pixel_id": None,
            "tracking_type": "builtin",
            "status": "ACTIVE",
            "ad_account_id": account_id,
            "source": "smartnews",
            "raw": {"key": key, "label": label},
        }
        for key, label in names
    ]


_EVENT_KEY_ALIASES = {
    "addToCart": "add_to_cart",
    "initiateCheckout": "initiate_checkout",
    "viewContent": "view_content",
    "completeRegistration": "complete_registration",
    "addPaymentInfo": "add_payment_info",
    "startTrial": "start_trial",
    "signUp": "sign_up",
    "submitForm": "submit_form",
    "findLocation": "find_location",
    "visitCart": "visit_cart",
    "addToWishList": "add_to_wish_list",
    "customizeProduct": "customize_product",
    "d1Retention": "d1_retention",
}

_EVENT_CANDIDATES = (
    "purchase",
    "addToCart",
    "initiateCheckout",
    "viewContent",
    "completeRegistration",
    "lead",
    "search",
    "subscribe",
    "addPaymentInfo",
    "startTrial",
    "install",
    "signUp",
    "contact",
    "submitForm",
    "download",
    "login",
    "donate",
    "findLocation",
    "share",
    "booking",
    "visitCart",
    "addToWishList",
    "customizeProduct",
    "d1Retention",
)


def _canonicalize_insights_row(r: Dict[str, Any], scope: str) -> Dict[str, Any]:
    if scope == "campaign":
        entity_id = r.get("campaignId") or r.get("id")
        parent_id = r.get("accountId")
        name = r.get("campaignName") or r.get("name") or "Campaign"
    elif scope == "ad":
        entity_id = r.get("creativeId") or r.get("id")
        parent_id = r.get("campaignId")
        name = r.get("creativeName") or r.get("name") or "Creative"
    elif scope == "account":
        entity_id = r.get("accountId")
        parent_id = None
        name = r.get("accountName") or "Account"
    else:
        entity_id = r.get("id")
        parent_id = None
        name = r.get("name") or ""

    spend = _num(r.get("spend"))
    clicks = int(_num(r.get("clicks")) or 0)
    impressions = int(_num(r.get("impressions")) or 0)
    conversions = _num(r.get("conversions")) or 0.0
    cpa = _num(r.get("cpa"))
    if (cpa is None or cpa == 0) and spend and conversions:
        cpa = spend / conversions
    ctr = _num(r.get("ctr"))
    if ctr is None and impressions:
        ctr = clicks / impressions * 100.0

    events: Dict[str, float] = {}
    for k in _EVENT_CANDIDATES:
        v = _num(r.get(k))
        if v is None:
            continue
        events[k] = v
        alias = _EVENT_KEY_ALIASES.get(k)
        if alias and alias not in events:
            events[alias] = v

    # SmartNews doesn't expose revenue/ROAS today; leave them None so the rule
    # engine just skips those rules on SmartNews.
    return {
        **r,
        "scope": scope,
        "id": str(entity_id) if entity_id is not None else None,
        "name": name,
        "parent_id": str(parent_id) if parent_id is not None else None,
        "status": "",
        "spend": spend,
        "impressions": impressions,
        "clicks": clicks,
        "ctr": float(ctr or 0.0),
        "conversions": conversions,
        "cpa": cpa,
        "roas": None,
        "value": None,
        "events": events,
        "campaign_id": str(r.get("campaignId") or "") or None,
        "ad_set_id": None,
        "ad_id": str(r.get("creativeId") or "") or None if scope == "ad" else None,
        "budget": None,
        "budget_type": "DAILY",
        "raw": r,
    }


def _num(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


__all__ = ["SmartNewsAdapter"]

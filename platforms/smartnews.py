"""
SmartNews adapter — wraps the v3 Marketing API client behind the
``AdPlatformAdapter`` contract.

The v3 hierarchy is Account → Campaign → Ad Group → Ad (3 real levels, matching
the in-product UI), so ``supports_ad_set_scope`` is True. Monetary values are
converted between our internal "cents" integer (1/100 USD, or whole JPY for
zero-decimal currencies) and SmartNews' ``_micro`` integers (×1,000,000 the
human amount) at the edge.

Accounts can be auto-discovered via the Business Manager
``/api/bm/v1/developer_apps/me/ad_accounts`` endpoint when no explicit
``account_ids`` are provided by the caller.
"""
from __future__ import annotations

import os
from datetime import date, datetime, timezone
from typing import Any, BinaryIO, Dict, Iterable, List, Optional, Sequence, Union

from smartnews_api import (
    SmartNewsClient,
    cents_to_micro,
    from_micro,
    micro_to_cents,
    unwrap_data,
    unwrap_list,
)


DEFAULT_INSIGHT_FIELDS: Dict[str, List[str]] = {
    "campaigns": [
        "metadata_name",
        "metadata_configured_status",
        "metadata_campaign_id",
        "metadata_campaign_name",
        "metadata_objective",
        "metadata_daily_budget_amount",
        "metadata_spending_limit",
        "metadata_optimization_event",
        "metadata_optimization_goal",
        "metadata_delivery_status",
        "metrics_viewable_impression",
        "metrics_click",
        "metrics_ctr",
        "metrics_cpc",
        "metrics_cpm",
        "metrics_budget_spent",
        "metrics_count_purchase",
        "metrics_cpa_purchase",
        "metrics_cvr_purchase",
        "metrics_count_add_to_cart",
        "metrics_cpa_add_to_cart",
        "metrics_count_initiate_checkout",
        "metrics_cpa_initiate_checkout",
        "metrics_count_submit_form",
        "metrics_count_subscribe",
        "metrics_count_complete_registration",
        "metrics_count_sign_up",
        "metrics_count_view_content",
        "metrics_count_lead",
    ],
    "ad_groups": [
        "metadata_name",
        "metadata_configured_status",
        "metadata_campaign_id",
        "metadata_campaign_name",
        "metadata_ad_group_id",
        "metadata_ad_group_name",
        "metadata_daily_budget_amount",
        "metadata_delivery_status",
        "metrics_viewable_impression",
        "metrics_click",
        "metrics_ctr",
        "metrics_cpc",
        "metrics_cpm",
        "metrics_budget_spent",
        "metrics_count_purchase",
        "metrics_cpa_purchase",
        "metrics_count_add_to_cart",
        "metrics_count_initiate_checkout",
        "metrics_count_lead",
    ],
    "ads": [
        "metadata_name",
        "metadata_configured_status",
        "metadata_campaign_id",
        "metadata_campaign_name",
        "metadata_ad_group_id",
        "metadata_ad_group_name",
        "metadata_moderation_status",
        "metadata_submission_status",
        "metrics_viewable_impression",
        "metrics_click",
        "metrics_ctr",
        "metrics_cpc",
        "metrics_cpm",
        "metrics_budget_spent",
        "metrics_count_purchase",
        "metrics_cpa_purchase",
        "metrics_count_add_to_cart",
        "metrics_count_initiate_checkout",
        "metrics_count_lead",
    ],
}


class SmartNewsAdapter:
    platform = "smartnews"
    label = "SmartNews"
    currency = "USD"  # default; overridden per-account after discovery
    supports_ad_set_scope = True

    def __init__(
        self,
        client: SmartNewsClient,
        account_ids: Optional[Sequence[Union[int, str]]] = None,
        default_currency: Optional[str] = None,
    ):
        self.client = client
        self.account_ids = [str(x) for x in (account_ids or [])]
        # SmartNews v3 does not expose currency on any account-metadata
        # endpoint, so we rely on a configurable fallback (``USD`` by
        # default — most of our advertisers are US-based). Override via
        # the ``SMARTNEWS_DEFAULT_CURRENCY`` env var or the
        # ``default_currency`` kwarg.
        env_cur = os.getenv("SMARTNEWS_DEFAULT_CURRENCY")
        if default_currency:
            self.currency = default_currency.upper()
        elif env_cur:
            self.currency = env_cur.upper()
        # else: keep the class-level default
        self._account_currency: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Auth / accounts
    # ------------------------------------------------------------------
    def verify(self) -> None:
        """Exchange the OAuth credentials for a token (raises on failure)."""
        self.client._get_token(force_refresh=True)

    def get_accounts(self) -> List[Dict[str, Any]]:
        body = self.client.get_developer_app_ad_accounts()
        out: List[Dict[str, Any]] = []
        for acc in unwrap_list(body):
            aid = acc.get("ad_account_id") or acc.get("id") or acc.get("accountId")
            if aid is None:
                continue
            aid = str(aid)
            if self.account_ids and aid not in self.account_ids:
                continue
            currency = (
                acc.get("currency")
                or acc.get("currency_code")
                or self.currency
            ).upper()
            self._account_currency[aid] = currency
            out.append(
                {
                    "id": aid,
                    "name": acc.get("name") or acc.get("ad_account_name") or f"SmartNews {aid}",
                    "currency": currency,
                    "raw": acc,
                }
            )
        return out

    def _currency_for(self, account_id: Union[int, str]) -> str:
        return self._account_currency.get(str(account_id), self.currency)

    # ------------------------------------------------------------------
    # Hierarchy reads
    # ------------------------------------------------------------------
    def get_campaigns(self, account_id: str) -> List[Dict[str, Any]]:
        rows = list(self.client.iter_campaigns(account_id))
        return [self._normalize_campaign(c, account_id) for c in rows]

    def get_ad_groups(self, account_id: str, campaign_id: str) -> List[Dict[str, Any]]:
        rows = list(self.client.iter_ad_groups_by_campaign(account_id, campaign_id))
        return [self._normalize_ad_group(g, account_id, campaign_id) for g in rows]

    def get_ads(self, account_id: str, parent_id: str) -> List[Dict[str, Any]]:
        """``parent_id`` is an ``ad_group_id`` in the v3 world."""
        rows = list(self.client.iter_ads_by_ad_group(account_id, parent_id))
        return [self._normalize_ad(a, account_id, parent_id) for a in rows]

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def create_campaign(self, account_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = self.client.create_campaign(account_id, self._prepare_campaign_payload(payload, account_id))
        inner = unwrap_data(body)
        return inner if isinstance(inner, dict) else {"raw": body}

    def create_ad_group(
        self,
        account_id: str,
        campaign_id: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        body = self.client.create_ad_group(
            account_id, campaign_id, self._prepare_ad_group_payload(payload, account_id)
        )
        inner = unwrap_data(body)
        return inner if isinstance(inner, dict) else {"raw": body}

    def create_ad(
        self,
        account_id: str,
        ad_group_id: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        body = self.client.create_ad(account_id, ad_group_id, payload)
        inner = unwrap_data(body)
        return inner if isinstance(inner, dict) else {"raw": body}

    def submit_ad_for_review(self, account_id: str, ad_id: str) -> Dict[str, Any]:
        """Flip ``submission_status`` on an ad from BEFORE_SUBMISSION to SUBMITTED."""
        body = self.client.submit_ad_for_review(account_id, ad_id)
        inner = unwrap_data(body)
        return inner if isinstance(inner, dict) else {"raw": body}

    def update_status(
        self,
        level: str,
        entity_id: str,
        enabled: bool,
        *,
        account_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not account_id:
            raise ValueError("SmartNews v3 update_status requires account_id")
        lvl = (level or "").lower()
        status = "ACTIVE" if enabled else "PAUSED"
        payload = {"configured_status": status}
        if lvl == "campaign":
            body = self.client.update_campaign(account_id, entity_id, payload)
        elif lvl in ("ad_set", "ad_group"):
            body = self.client.update_ad_group(account_id, entity_id, payload)
        elif lvl == "ad":
            body = self.client.update_ad(account_id, entity_id, payload)
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
        """Set a daily or total (spending-limit) budget.

        SmartNews v3 keeps budgets on the Campaign by default. Callers that
        want to move the lever to an Ad Group can pass ``level="ad_group"``
        and this adapter will PATCH the ``daily_budget_amount_micro`` on the
        ad group instead.
        """
        if not account_id:
            raise ValueError("SmartNews v3 update_budget requires account_id")
        lvl = (level or "").lower()
        if lvl == "ad_set":
            lvl = "ad_group"
        cur = self._currency_for(account_id)
        micro = cents_to_micro(int(budget_cents), currency=cur)
        bt = (budget_type or "DAILY").upper()
        payload: Dict[str, Any] = {}
        if bt in ("TOTAL", "LIFETIME"):
            payload["spending_limit_micro"] = micro
        else:
            payload["daily_budget_amount_micro"] = micro

        if lvl == "campaign":
            body = self.client.update_campaign(account_id, entity_id, payload)
        elif lvl == "ad_group":
            body = self.client.update_ad_group(account_id, entity_id, payload)
        else:
            raise ValueError(
                f"SmartNews supports campaign/ad_group budget updates (got {level})"
            )
        inner = unwrap_data(body)
        return inner if isinstance(inner, dict) else {"raw": body}

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    _LAYER_MAP = {
        "campaign": "campaigns",
        "campaigns": "campaigns",
        "ad_set": "ad_groups",
        "ad_group": "ad_groups",
        "ad_groups": "ad_groups",
        "ad": "ads",
        "ads": "ads",
    }

    _CANONICAL_SCOPE = {
        "campaigns": "campaign",
        "ad_groups": "ad_set",
        "ads": "ad",
    }

    def fetch_report_rows(
        self,
        account_id: str,
        scope: str,
        start: date,
        end: date,
    ) -> List[Dict[str, Any]]:
        layer = self._LAYER_MAP.get((scope or "").lower(), "campaigns")
        fields = DEFAULT_INSIGHT_FIELDS[layer]
        since_dt = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
        until_dt = datetime.combine(end, datetime.max.time().replace(microsecond=0), tzinfo=timezone.utc)

        params = {
            "since": since_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "until": until_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "fields": ",".join(fields),
        }
        rows = list(self.client.paginate(
            f"/api/ma/v3/ad_accounts/{account_id}/insights/{layer}",
            params=params,
        ))
        canonical = self._CANONICAL_SCOPE[layer]
        return [self._canonicalize_insights_row(r, canonical) for r in rows]

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def list_events(self, account_id: str) -> List[Dict[str, Any]]:
        return _builtin_events(account_id)

    def list_pixels(self, account_id: str) -> List[Dict[str, Any]]:
        try:
            body = self.client.list_pixels(account_id)
        except Exception:
            return []
        return unwrap_list(body)

    def upload_asset(
        self,
        account_id: str,
        file_obj: Union[BinaryIO, bytes],
        filename: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        body = self.client.create_media_file(
            account_id,
            file_obj,
            filename,
            media_type=kwargs.get("media_type", "IMAGE"),
            mime_type=kwargs.get("mime_type", "image/jpeg"),
        )
        inner = unwrap_data(body)
        return inner if isinstance(inner, dict) else {"raw": body}

    # ------------------------------------------------------------------
    # Payload helpers — convert our internal "budget_cents" etc to micros
    # ------------------------------------------------------------------
    def _prepare_campaign_payload(self, payload: Dict[str, Any], account_id: str) -> Dict[str, Any]:
        out = dict(payload)
        cur = self._currency_for(account_id)
        for key_in, key_out in (
            ("daily_budget_cents", "daily_budget_amount_micro"),
            ("spending_limit_cents", "spending_limit_micro"),
            ("bid_amount_cents", "bid_amount_micro"),
            ("target_cost_cents", "target_cost_micro"),
            ("budget_auto_target_cpa_cents", "budget_auto_target_cpa_micro"),
        ):
            if key_in in out and out[key_in] is not None:
                out[key_out] = cents_to_micro(out.pop(key_in), currency=cur)
        return out

    def _prepare_ad_group_payload(self, payload: Dict[str, Any], account_id: str) -> Dict[str, Any]:
        out = dict(payload)
        cur = self._currency_for(account_id)
        for key_in, key_out in (
            ("daily_budget_cents", "daily_budget_amount_micro"),
            ("bid_amount_cents", "bid_amount_micro"),
            ("target_cost_cents", "target_cost_micro"),
        ):
            if key_in in out and out[key_in] is not None:
                out[key_out] = cents_to_micro(out.pop(key_in), currency=cur)
        return out

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _status_from_configured(row: Dict[str, Any]) -> str:
        status = (row.get("configured_status") or "").upper()
        if status == "ACTIVE":
            return "on"
        if status in ("PAUSED", "DELETED"):
            return "off"
        return ""

    def _normalize_campaign(self, c: Dict[str, Any], account_id: str) -> Dict[str, Any]:
        cur = self._currency_for(account_id)
        daily_micro = c.get("daily_budget_amount_micro")
        return {
            "id": str(c.get("campaign_id") or c.get("id") or ""),
            "name": c.get("name") or "",
            "ad_account_id": str(c.get("ad_account_id") or account_id),
            "status": self._status_from_configured(c),
            "enable": (c.get("configured_status") or "").upper() == "ACTIVE",
            "approval_status": (c.get("delivery_status") or {}).get("status") or "",
            "daily_budget_cents": micro_to_cents(daily_micro, currency=cur) if daily_micro else None,
            "daily_budget": from_micro(daily_micro) if daily_micro else None,
            "spending_limit_cents": micro_to_cents(c.get("spending_limit_micro"), currency=cur)
            if c.get("spending_limit_micro")
            else None,
            "objective": c.get("objective"),
            "optimization_event": c.get("optimization_event"),
            "optimization_goal": c.get("optimization_goal"),
            "start_time": c.get("start_date_time"),
            "end_time": c.get("end_date_time"),
            "ready_for_delivery": c.get("ready_for_delivery"),
            "raw": c,
        }

    def _normalize_ad_group(self, g: Dict[str, Any], account_id: str, campaign_id: str) -> Dict[str, Any]:
        cur = self._currency_for(account_id)
        daily_micro = g.get("daily_budget_amount_micro")
        return {
            "id": str(g.get("ad_group_id") or g.get("id") or ""),
            "name": g.get("name") or "",
            "ad_account_id": str(g.get("ad_account_id") or account_id),
            "campaign_id": str(g.get("campaign_id") or campaign_id),
            "status": self._status_from_configured(g),
            "enable": (g.get("configured_status") or "").upper() == "ACTIVE",
            "daily_budget_cents": micro_to_cents(daily_micro, currency=cur) if daily_micro else None,
            "daily_budget": from_micro(daily_micro) if daily_micro else None,
            "approval_status": (g.get("delivery_status") or {}).get("status") or "",
            "raw": g,
        }

    def _normalize_ad(self, a: Dict[str, Any], account_id: str, ad_group_id: str) -> Dict[str, Any]:
        return {
            "id": str(a.get("ad_id") or a.get("id") or ""),
            "name": a.get("name") or "",
            "ad_account_id": str(a.get("ad_account_id") or account_id),
            "ad_group_id": str(a.get("ad_group_id") or ad_group_id),
            "campaign_id": str(a.get("campaign_id") or ""),
            "status": self._status_from_configured(a),
            "enable": (a.get("configured_status") or "").upper() == "ACTIVE",
            "approval_status": a.get("moderation_status") or "",
            "submission_status": a.get("submission_status"),
            "landing_page_url": a.get("landing_page_url"),
            "cta_label": a.get("cta_label"),
            "creative": a.get("creative"),
            "raw": a,
        }

    def _canonicalize_insights_row(self, r: Dict[str, Any], scope: str) -> Dict[str, Any]:
        meta = r.get("metadata") or {}
        metrics = r.get("metrics") or {}
        cur = self._currency_for(meta.get("ad_account_id") or "") if meta.get("ad_account_id") else self.currency

        entity_id = r.get("id")
        if scope == "campaign":
            name = meta.get("name") or meta.get("campaign_name") or "Campaign"
            parent_id = meta.get("ad_account_id")
            campaign_id = str(entity_id) if entity_id is not None else None
            ad_set_id = None
            ad_id = None
        elif scope == "ad_set":
            name = meta.get("name") or meta.get("ad_group_name") or "Ad Group"
            parent_id = str(meta.get("campaign_id") or "")
            campaign_id = str(meta.get("campaign_id") or "") or None
            ad_set_id = str(entity_id) if entity_id is not None else None
            ad_id = None
        else:  # scope == "ad"
            name = meta.get("name") or "Ad"
            parent_id = str(meta.get("ad_group_id") or "")
            campaign_id = str(meta.get("campaign_id") or "") or None
            ad_set_id = str(meta.get("ad_group_id") or "") or None
            ad_id = str(entity_id) if entity_id is not None else None

        spend = _num(metrics.get("budget_spent"))
        clicks = int(_num(metrics.get("click")) or 0)
        impressions = int(_num(metrics.get("viewable_impression")) or 0)
        ctr_val = _num(metrics.get("ctr"))
        if ctr_val is None and impressions:
            ctr_val = clicks / impressions * 100.0
        elif ctr_val is not None and ctr_val < 1:
            # v3 returns ctr as a fraction string like "0.03" → convert to percent
            ctr_val = ctr_val * 100.0

        purchase = _num(metrics.get("count_purchase")) or 0.0
        cpa_purchase = _num(metrics.get("cpa_purchase"))
        if (cpa_purchase is None or cpa_purchase == 0) and spend and purchase:
            cpa_purchase = spend / purchase

        events: Dict[str, float] = {}
        for metric_key, event_key in _COUNT_METRIC_TO_EVENT.items():
            v = _num(metrics.get(metric_key))
            if v is None:
                continue
            events[event_key] = v

        status = ""
        conf = (meta.get("configured_status") or "").upper()
        if conf == "ACTIVE":
            status = "on"
        elif conf in ("PAUSED", "DELETED"):
            status = "off"

        budget_cents = None
        budget_type = None
        dba = meta.get("daily_budget_amount")
        if dba not in (None, "", "0"):
            try:
                dollars = float(dba)
                budget_cents = (
                    int(round(dollars)) if cur.upper() == "JPY" else int(round(dollars * 100))
                )
                budget_type = "DAILY"
            except (TypeError, ValueError):
                pass

        return {
            **r,
            "scope": scope,
            "id": str(entity_id) if entity_id is not None else None,
            "name": name,
            "parent_id": str(parent_id) if parent_id is not None else None,
            "status": status,
            "spend": spend or 0.0,
            "impressions": impressions,
            "clicks": clicks,
            "ctr": float(ctr_val or 0.0),
            "conversions": purchase,
            "cpa": cpa_purchase,
            "roas": None,
            "value": None,
            "events": events,
            "campaign_id": campaign_id,
            "ad_set_id": ad_set_id,
            "ad_id": ad_id,
            "budget": budget_cents,
            "budget_type": budget_type,
            "metadata": meta,
            "metrics": metrics,
            "raw": r,
        }


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------
_COUNT_METRIC_TO_EVENT = {
    "count_purchase": "purchase",
    "count_add_to_cart": "add_to_cart",
    "count_initiate_checkout": "initiate_checkout",
    "count_submit_form": "submit_form",
    "count_subscribe": "subscribe",
    "count_complete_registration": "complete_registration",
    "count_contact": "contact",
    "count_sign_up": "sign_up",
    "count_view_content": "view_content",
    "count_add_payment_info": "add_payment_info",
    "count_add_to_wish_list": "add_to_wish_list",
    "count_visit_cart": "visit_cart",
    "count_customize_product": "customize_product",
    "count_search": "search",
    "count_booking": "booking",
    "count_download": "download",
    "count_start_trial": "start_trial",
    "count_share": "share",
    "count_login": "login",
    "count_donate": "donate",
    "count_find_location": "find_location",
    "count_time_spent": "time_spent",
    "count_install": "install",
    "count_d1_retention": "d1_retention",
    "count_skan_install": "skan_install",
    "count_lead": "lead",
}


def _builtin_events(account_id: str) -> List[Dict[str, Any]]:
    """SmartNews Insights event catalogue — matches the v3 metrics prefixes."""
    labels = {
        "purchase": "Purchase",
        "add_to_cart": "Add to Cart",
        "initiate_checkout": "Initiate Checkout",
        "view_content": "View Content",
        "complete_registration": "Complete Registration",
        "lead": "Lead",
        "search": "Search",
        "add_payment_info": "Add Payment Info",
        "start_trial": "Start Trial",
        "subscribe": "Subscribe",
        "install": "Install",
        "sign_up": "Sign Up",
        "contact": "Contact",
        "submit_form": "Submit Form",
        "download": "Download",
        "login": "Login",
        "booking": "Booking",
        "donate": "Donate",
        "share": "Share",
        "visit_cart": "Visit Cart",
        "add_to_wish_list": "Add to Wish List",
        "customize_product": "Customize Product",
        "find_location": "Find Location",
        "d1_retention": "D1 Retention",
    }
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
        for key, label in labels.items()
    ]


def _num(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


__all__ = ["SmartNewsAdapter"]

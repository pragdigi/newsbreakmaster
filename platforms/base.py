"""
Platform adapter Protocol.

Adapters wrap a platform-specific HTTP client (NewsBreakClient, SmartNewsClient, …)
and expose a small, canonical surface the rest of the app + rules engine depends on.

Normalised report row shape consumed by the rules engine:

    {
      "scope":        "ad" | "ad_set" | "campaign",
      "id":           str,                # canonical entity id for this row
      "name":         str,
      "parent_id":    str | None,
      "status":       "on" | "off" | "pending" | "",
      "spend":        float (dollars/native currency),
      "impressions":  int,
      "clicks":       int,
      "ctr":          float (percent),
      "conversions":  float,
      "cpa":          float | None,
      "roas":         float | None,
      "value":        float | None,       # revenue / conversion value
      "events":       dict[str, float],   # { add_to_cart, initiate_checkout, purchase, … }
      "campaign_id":  str | None,
      "ad_set_id":    str | None,
      "ad_id":        str | None,
      "budget":       float | None,       # current budget in cents/subunit
      "budget_type":  "DAILY" | "TOTAL" | None,
      "raw":          dict,               # original platform row (keep for debugging)
    }
"""
from __future__ import annotations

from datetime import date
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Protocol, Union


class AdPlatformAdapter(Protocol):
    platform: str                 # "newsbreak" | "smartnews"
    label: str                    # "NewsBreak" | "SmartNews"
    currency: str                 # "USD" | "JPY"
    supports_ad_set_scope: bool   # True for NewsBreak, False for SmartNews AMv1

    # --- auth helpers ---
    def verify(self) -> None:
        """Raise if the current credentials are invalid."""
        ...

    # --- account discovery ---
    def get_accounts(self) -> List[Dict[str, Any]]:
        """Return flat list of ad accounts visible to this token.

        Each dict must include at least ``id`` and ``name``. Other keys are
        passed through to the UI for debugging / labelling.
        """
        ...

    # --- campaign / ad set / ad reads ---
    def get_campaigns(self, account_id: str) -> List[Dict[str, Any]]:
        ...

    def get_ad_groups(self, account_id: str, campaign_id: str) -> List[Dict[str, Any]]:
        """Ad-sets on NewsBreak; on SmartNews AMv1 returns [] (campaign IS the ad set)."""
        ...

    def get_ads(self, account_id: str, parent_id: str) -> List[Dict[str, Any]]:
        """``parent_id`` is an ad-set id on NewsBreak, a campaign id on SmartNews."""
        ...

    # --- writes ---
    def create_campaign(self, account_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def update_status(self, level: str, entity_id: str, enabled: bool, *, account_id: Optional[str] = None) -> Dict[str, Any]:
        """level ∈ {ad, ad_set, campaign}. For SmartNews ad_set collapses to campaign."""
        ...

    def update_budget(
        self,
        level: str,
        entity_id: str,
        *,
        budget_cents: int,
        budget_type: str = "DAILY",
        account_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        ...

    # --- reporting ---
    def fetch_report_rows(
        self,
        account_id: str,
        scope: str,
        start: date,
        end: date,
    ) -> List[Dict[str, Any]]:
        """Return normalised rows (see module doc)."""
        ...

    # --- events / pixels catalog bootstrap ---
    def list_events(self, account_id: str) -> List[Dict[str, Any]]:
        ...

    # --- assets ---
    def upload_asset(
        self,
        account_id: str,
        file_obj: Union[BinaryIO, bytes],
        filename: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...


# --- convenience: no-op fallback for unsupported operations ---
class UnsupportedOperationError(RuntimeError):
    pass


def unsupported(platform: str, op: str) -> Callable[..., Any]:
    def _raise(*_a: Any, **_k: Any) -> Any:
        raise UnsupportedOperationError(f"{platform} adapter does not support {op}")
    return _raise

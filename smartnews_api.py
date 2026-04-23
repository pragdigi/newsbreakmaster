"""
SmartNews Marketing API v3 client.

Docs: https://ads.smartnews.com/developers
Base: https://ads.smartnews.com

Endpoints covered:
    /api/oauth/v1/access_tokens                     (OAuth client_credentials)
    /api/bm/v1/developer_apps/me/ad_accounts        (ad account discovery)
    /api/ma/v3/ad_accounts/{id}/...                 (Marketing API — campaigns,
                                                     ad groups, ads, insights,
                                                     media files, pixels, etc.)

Authentication
--------------
Clients are authenticated via OAuth2 client_credentials grant. The caller
provides ``client_id`` (the developer app id) and ``client_secret`` (the
developer app secret). This client exchanges them for a short-lived Bearer
access_token on first use and refreshes on expiry.

Monetary values
---------------
All create/update payloads use the ``_micro`` notation — integers scaled by
1,000,000 regardless of currency (so ¥120 JPY = 120_000_000 and $1.50 USD =
1_500_000). Read responses contain both ``..._micro`` integers and human-
readable strings like ``daily_budget_amount: "123.4"``.

Pagination
----------
v3 list endpoints use page-based pagination via ``page`` (1-indexed) and
``page_size`` (default 100, max typically 200). Callers that want the full
listing can use the ``paginate`` helper.

Backward compatibility shims (``unwrap_data`` / ``unwrap_list``) are preserved
so existing callers/tests that expect v1-style envelopes keep working.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any, BinaryIO, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import requests

BASE_URL = "https://ads.smartnews.com"
OAUTH_PATH = "/api/oauth/v1/access_tokens"
OAUTH_REVOKE_PATH = "/api/oauth/v1/access_tokens/revoke"
DEV_ACCOUNTS_PATH = "/api/bm/v1/developer_apps/me/ad_accounts"
MA_BASE = "/api/ma/v3"


class SmartNewsAPIError(Exception):
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        body: Any = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class SmartNewsAuthError(SmartNewsAPIError):
    """Raised when OAuth client credentials are rejected."""


class SmartNewsClient:
    """Thin wrapper around SmartNews Marketing API v3 + OAuth."""

    def __init__(
        self,
        client_id: Union[int, str],
        client_secret: str,
        *,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        base_url: str = BASE_URL,
    ):
        if not client_id or not client_secret:
            raise ValueError("SmartNewsClient requires client_id and client_secret")
        self.client_id = int(client_id) if str(client_id).isdigit() else client_id
        self.client_secret = str(client_secret)
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.base_url = base_url.rstrip("/")

        self._session = requests.Session()
        self._session.headers.update(
            {
                "Accept": "application/json",
                "Accept-Language": "en-US",
                "User-Agent": "newsbreakmaster/smartnews-v3 (+https://newsbreakmaster.onrender.com)",
            }
        )

        self._lock = threading.Lock()
        self._access_token: Optional[str] = None
        self._access_token_expires_at: float = 0.0

    # ------------------------------------------------------------------
    # OAuth / token management
    # ------------------------------------------------------------------
    def _fetch_access_token(self) -> None:
        """Exchange client_credentials for a short-lived Bearer access token."""
        url = self.base_url + OAUTH_PATH
        try:
            resp = self._session.post(
                url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise SmartNewsAuthError(f"OAuth request failed: {e}") from e

        try:
            body = resp.json() if resp.text else {}
        except json.JSONDecodeError:
            body = {"raw": resp.text}

        if resp.status_code >= 400:
            err = body.get("error") if isinstance(body, dict) else None
            msg = (
                (err.get("message") if isinstance(err, dict) else None)
                or (body.get("message") if isinstance(body, dict) else None)
                or resp.reason
            )
            raise SmartNewsAuthError(
                f"SmartNews OAuth error ({resp.status_code}): {msg}",
                status_code=resp.status_code,
                body=body,
            )

        token = (body or {}).get("access_token")
        expires_in = int((body or {}).get("expires_in") or 3600)
        if not token:
            raise SmartNewsAuthError(
                "OAuth response missing access_token", status_code=resp.status_code, body=body
            )
        # Refresh ~60s before actual expiry to avoid racing with the edge.
        self._access_token = token
        self._access_token_expires_at = time.time() + max(60, expires_in - 60)

    def _get_token(self, force_refresh: bool = False) -> str:
        with self._lock:
            if (
                force_refresh
                or not self._access_token
                or time.time() >= self._access_token_expires_at
            ):
                self._fetch_access_token()
            assert self._access_token
            return self._access_token

    @property
    def access_token(self) -> Optional[str]:
        return self._access_token

    def revoke_tokens(self) -> None:
        """Invalidate every active access token for this developer app."""
        url = self.base_url + OAUTH_REVOKE_PATH
        self._session.post(
            url,
            data={"client_id": self.client_id, "client_secret": self.client_secret},
            timeout=self.timeout,
        )
        with self._lock:
            self._access_token = None
            self._access_token_expires_at = 0.0

    # ------------------------------------------------------------------
    # Low-level HTTP
    # ------------------------------------------------------------------
    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Any] = None,
        form: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        allow_empty: bool = False,
        _retry_on_401: bool = True,
    ) -> Any:
        url = path if path.startswith("http") else self.base_url + path
        last_err: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            headers = {"Authorization": f"Bearer {self._get_token()}"}
            if json_body is not None and not files:
                headers["Content-Type"] = "application/json"

            try:
                resp = self._session.request(
                    method,
                    url,
                    params=params,
                    json=json_body if json_body is not None and not files else None,
                    data=form if form is not None else None,
                    files=files,
                    headers=headers,
                    timeout=self.timeout,
                )
            except requests.RequestException as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise SmartNewsAPIError(str(e)) from e

            # Transparent re-auth on 401 in case the cached token expired
            # between the expiry check and the actual request.
            if resp.status_code == 401 and _retry_on_401:
                self._get_token(force_refresh=True)
                _retry_on_401 = False
                continue

            if resp.status_code == 429 or (500 <= resp.status_code < 600):
                last_err = SmartNewsAPIError(
                    f"SmartNews transient error {resp.status_code}",
                    status_code=resp.status_code,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise last_err

            text = resp.text
            try:
                body = resp.json() if text else {}
            except json.JSONDecodeError:
                body = {"raw": text}

            if resp.status_code >= 400:
                err = body.get("error") if isinstance(body, dict) else None
                msg = (
                    (err.get("message") if isinstance(err, dict) else None)
                    or (body.get("message") if isinstance(body, dict) else None)
                    or text
                    or resp.reason
                )
                # Append per-field reasons (e.g. VALIDATION_ERROR) so the
                # caller does not have to dig through logs to see exactly
                # which field SmartNews rejected.
                if isinstance(err, dict):
                    field_errs = err.get("error_fields") or err.get("fields") or []
                    parts: list[str] = []
                    if isinstance(field_errs, list):
                        for fe in field_errs:
                            if not isinstance(fe, dict):
                                continue
                            name = fe.get("field_name") or fe.get("field") or "?"
                            reason = fe.get("reason") or fe.get("message") or ""
                            parts.append(f"{name}: {reason}".strip(": "))
                    if parts:
                        msg = f"{msg} [{'; '.join(parts)}]"
                raise SmartNewsAPIError(
                    f"SmartNews API error ({resp.status_code}): {msg}",
                    status_code=resp.status_code,
                    body=body,
                )

            if not body and not allow_empty:
                return {}
            return body

        if last_err:
            raise SmartNewsAPIError(str(last_err)) from last_err
        raise SmartNewsAPIError("Unknown request error")

    # Convenience
    def get(self, path: str, **kwargs: Any) -> Any:
        return self._request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> Any:
        return self._request("POST", path, **kwargs)

    def patch(self, path: str, **kwargs: Any) -> Any:
        return self._request("PATCH", path, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> Any:
        return self._request("DELETE", path, allow_empty=True, **kwargs)

    # ------------------------------------------------------------------
    # Pagination helper
    # ------------------------------------------------------------------
    def paginate(
        self,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        page_size: int = 100,
        max_pages: int = 50,
    ) -> Iterator[Dict[str, Any]]:
        base_params = dict(params or {})
        base_params.setdefault("page_size", page_size)
        page = 1
        while page <= max_pages:
            q = dict(base_params)
            q["page"] = page
            body = self.get(path, params=q)
            rows = _extract_list(body)
            if not rows:
                return
            for row in rows:
                yield row
            if len(rows) < q["page_size"]:
                return
            page += 1

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def get_developer_app_ad_accounts(self) -> Any:
        """Ad accounts the developer app can manage (auto-discovery)."""
        return self.get(DEV_ACCOUNTS_PATH)

    # ------------------------------------------------------------------
    # Campaigns
    # ------------------------------------------------------------------
    def list_campaigns(self, ad_account_id: Union[int, str], **params: Any) -> Any:
        return self.get(f"{MA_BASE}/ad_accounts/{ad_account_id}/campaigns", params=params)

    def iter_campaigns(self, ad_account_id: Union[int, str], **params: Any) -> Iterator[Dict[str, Any]]:
        return self.paginate(
            f"{MA_BASE}/ad_accounts/{ad_account_id}/campaigns",
            params=params,
        )

    def get_campaign(self, ad_account_id: Union[int, str], campaign_id: Union[int, str]) -> Any:
        return self.get(f"{MA_BASE}/ad_accounts/{ad_account_id}/campaigns/{campaign_id}")

    def create_campaign(
        self,
        ad_account_id: Union[int, str],
        payload: Dict[str, Any],
    ) -> Any:
        return self.post(
            f"{MA_BASE}/ad_accounts/{ad_account_id}/campaigns",
            json_body=payload,
        )

    def update_campaign(
        self,
        ad_account_id: Union[int, str],
        campaign_id: Union[int, str],
        payload: Dict[str, Any],
    ) -> Any:
        return self.patch(
            f"{MA_BASE}/ad_accounts/{ad_account_id}/campaigns/{campaign_id}",
            json_body=payload,
        )

    def delete_campaign(
        self,
        ad_account_id: Union[int, str],
        campaign_id: Union[int, str],
    ) -> Any:
        return self.delete(f"{MA_BASE}/ad_accounts/{ad_account_id}/campaigns/{campaign_id}")

    # ------------------------------------------------------------------
    # Ad groups
    # ------------------------------------------------------------------
    def list_ad_groups_by_campaign(
        self,
        ad_account_id: Union[int, str],
        campaign_id: Union[int, str],
        **params: Any,
    ) -> Any:
        return self.get(
            f"{MA_BASE}/ad_accounts/{ad_account_id}/campaigns/{campaign_id}/ad_groups",
            params=params,
        )

    def iter_ad_groups_by_campaign(
        self,
        ad_account_id: Union[int, str],
        campaign_id: Union[int, str],
        **params: Any,
    ) -> Iterator[Dict[str, Any]]:
        return self.paginate(
            f"{MA_BASE}/ad_accounts/{ad_account_id}/campaigns/{campaign_id}/ad_groups",
            params=params,
        )

    def list_ad_groups_by_account(
        self,
        ad_account_id: Union[int, str],
        **params: Any,
    ) -> Any:
        return self.get(f"{MA_BASE}/ad_accounts/{ad_account_id}/ad_groups", params=params)

    def get_ad_group(
        self,
        ad_account_id: Union[int, str],
        ad_group_id: Union[int, str],
    ) -> Any:
        return self.get(f"{MA_BASE}/ad_accounts/{ad_account_id}/ad_groups/{ad_group_id}")

    def create_ad_group(
        self,
        ad_account_id: Union[int, str],
        campaign_id: Union[int, str],
        payload: Dict[str, Any],
    ) -> Any:
        return self.post(
            f"{MA_BASE}/ad_accounts/{ad_account_id}/campaigns/{campaign_id}/ad_groups",
            json_body=payload,
        )

    def update_ad_group(
        self,
        ad_account_id: Union[int, str],
        ad_group_id: Union[int, str],
        payload: Dict[str, Any],
    ) -> Any:
        return self.patch(
            f"{MA_BASE}/ad_accounts/{ad_account_id}/ad_groups/{ad_group_id}",
            json_body=payload,
        )

    def delete_ad_group(
        self,
        ad_account_id: Union[int, str],
        ad_group_id: Union[int, str],
    ) -> Any:
        return self.delete(f"{MA_BASE}/ad_accounts/{ad_account_id}/ad_groups/{ad_group_id}")

    # ------------------------------------------------------------------
    # Ads
    # ------------------------------------------------------------------
    def list_ads_by_ad_group(
        self,
        ad_account_id: Union[int, str],
        ad_group_id: Union[int, str],
        **params: Any,
    ) -> Any:
        return self.get(
            f"{MA_BASE}/ad_accounts/{ad_account_id}/ad_groups/{ad_group_id}/ads",
            params=params,
        )

    def iter_ads_by_ad_group(
        self,
        ad_account_id: Union[int, str],
        ad_group_id: Union[int, str],
        **params: Any,
    ) -> Iterator[Dict[str, Any]]:
        return self.paginate(
            f"{MA_BASE}/ad_accounts/{ad_account_id}/ad_groups/{ad_group_id}/ads",
            params=params,
        )

    def list_ads_by_account(
        self,
        ad_account_id: Union[int, str],
        **params: Any,
    ) -> Any:
        return self.get(f"{MA_BASE}/ad_accounts/{ad_account_id}/ads", params=params)

    def get_ad(self, ad_account_id: Union[int, str], ad_id: Union[int, str]) -> Any:
        return self.get(f"{MA_BASE}/ad_accounts/{ad_account_id}/ads/{ad_id}")

    def create_ad(
        self,
        ad_account_id: Union[int, str],
        ad_group_id: Union[int, str],
        payload: Dict[str, Any],
    ) -> Any:
        return self.post(
            f"{MA_BASE}/ad_accounts/{ad_account_id}/ad_groups/{ad_group_id}/ads",
            json_body=payload,
        )

    def update_ad(
        self,
        ad_account_id: Union[int, str],
        ad_id: Union[int, str],
        payload: Dict[str, Any],
    ) -> Any:
        return self.patch(
            f"{MA_BASE}/ad_accounts/{ad_account_id}/ads/{ad_id}",
            json_body=payload,
        )

    def delete_ad(
        self,
        ad_account_id: Union[int, str],
        ad_id: Union[int, str],
    ) -> Any:
        return self.delete(f"{MA_BASE}/ad_accounts/{ad_account_id}/ads/{ad_id}")

    # ------------------------------------------------------------------
    # Insights
    # ------------------------------------------------------------------
    def get_insights(
        self,
        ad_account_id: Union[int, str],
        layer: str,
        **params: Any,
    ) -> Any:
        """Insights for a given layer: ``CAMPAIGN``, ``AD_GROUP`` or ``AD``."""
        return self.get(
            f"{MA_BASE}/ad_accounts/{ad_account_id}/insights/{layer}",
            params=params,
        )

    def iter_insights(
        self,
        ad_account_id: Union[int, str],
        layer: str,
        **params: Any,
    ) -> Iterator[Dict[str, Any]]:
        return self.paginate(
            f"{MA_BASE}/ad_accounts/{ad_account_id}/insights/{layer}",
            params=params,
        )

    def get_aggregated_insights(
        self,
        ad_account_id: Union[int, str],
        layer: str,
        **params: Any,
    ) -> Any:
        return self.get(
            f"{MA_BASE}/ad_accounts/{ad_account_id}/aggregated_insights/{layer}",
            params=params,
        )

    # ------------------------------------------------------------------
    # Media files
    # ------------------------------------------------------------------
    def list_media_files(self, ad_account_id: Union[int, str], **params: Any) -> Any:
        return self.get(f"{MA_BASE}/ad_accounts/{ad_account_id}/media_files", params=params)

    def create_media_file(
        self,
        ad_account_id: Union[int, str],
        file_obj: Union[BinaryIO, bytes],
        filename: str,
        *,
        media_type: str = "IMAGE",
        mime_type: str = "image/jpeg",
    ) -> Any:
        """Upload an image/video and register it as a media file."""
        files = {"file": (filename, file_obj, mime_type)}
        form = {"media_type": media_type}
        return self.post(
            f"{MA_BASE}/ad_accounts/{ad_account_id}/media_files",
            form=form,
            files=files,
        )

    # ------------------------------------------------------------------
    # Pixels / taxonomies / audiences (read-only helpers used by the UI)
    # ------------------------------------------------------------------
    def list_pixels(self, ad_account_id: Union[int, str]) -> Any:
        return self.get(f"{MA_BASE}/ad_accounts/{ad_account_id}/pixels")

    def get_pixel(
        self,
        ad_account_id: Union[int, str],
        pixel_tag_id: Union[int, str],
    ) -> Any:
        return self.get(f"{MA_BASE}/ad_accounts/{ad_account_id}/pixels/{pixel_tag_id}")

    def list_locations(self, **params: Any) -> Any:
        return self.get(f"{MA_BASE}/locations", params=params)

    def list_interests(self, **params: Any) -> Any:
        return self.get(f"{MA_BASE}/iab_interest_categories", params=params)

    def list_custom_audiences(
        self,
        ad_account_id: Union[int, str],
        **params: Any,
    ) -> Any:
        return self.get(
            f"{MA_BASE}/ad_accounts/{ad_account_id}/custom_audiences",
            params=params,
        )


# ----------------------------------------------------------------------
# Response-shape helpers (kept API-compatible with the old v1 client)
# ----------------------------------------------------------------------


def _extract_list(body: Any) -> List[Dict[str, Any]]:
    if isinstance(body, list):
        return [x for x in body if isinstance(x, dict)]
    if isinstance(body, dict):
        for key in ("data", "items", "results", "ad_accounts", "campaigns", "ad_groups", "ads"):
            inner = body.get(key)
            if isinstance(inner, list):
                return [x for x in inner if isinstance(x, dict)]
            if isinstance(inner, dict):
                rows = inner.get("data") or inner.get("items") or inner.get("results")
                if isinstance(rows, list):
                    return [x for x in rows if isinstance(x, dict)]
    return []


def unwrap_list(body: Any) -> List[Dict[str, Any]]:
    """Back-compat with callers of the old v1 client."""
    return _extract_list(body)


def unwrap_data(body: Any) -> Any:
    if isinstance(body, dict) and "data" in body:
        return body["data"]
    return body


# ----------------------------------------------------------------------
# Money helpers
# ----------------------------------------------------------------------
MICRO_FACTOR = 1_000_000


def to_micro(amount: Union[int, float, str]) -> int:
    """Convert a human-readable amount (e.g. 1.50 USD, 120 JPY) to ``_micro``."""
    if amount is None:
        return 0
    return int(round(float(amount) * MICRO_FACTOR))


def from_micro(amount_micro: Union[int, float, str, None]) -> float:
    """Convert a ``_micro`` integer back to the human-readable amount."""
    if amount_micro is None:
        return 0.0
    try:
        return float(amount_micro) / MICRO_FACTOR
    except (TypeError, ValueError):
        return 0.0


def cents_to_micro(amount_cents: Union[int, float, None], *, currency: str = "USD") -> int:
    """Convert our internal "cents" integer to SmartNews ``_micro`` units.

    For USD-like currencies ("cents" = 1/100 of the unit), micros = cents * 10_000.
    For JPY ("cents" = whole yen, zero-decimal), micros = cents * 1_000_000.
    """
    if amount_cents is None:
        return 0
    cur = (currency or "USD").upper()
    if cur == "JPY":
        return int(round(float(amount_cents) * MICRO_FACTOR))
    return int(round(float(amount_cents) * 10_000))


def micro_to_cents(amount_micro: Union[int, float, None], *, currency: str = "USD") -> int:
    if amount_micro is None:
        return 0
    cur = (currency or "USD").upper()
    if cur == "JPY":
        return int(round(float(amount_micro) / MICRO_FACTOR))
    return int(round(float(amount_micro) / 10_000))

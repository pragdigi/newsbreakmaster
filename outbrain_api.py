"""
Outbrain Amplify API v0.1 client (also powers Teads native demand via Outbrain).

Docs:  https://amplifyv01.docs.apiary.io/
Base:  https://api.outbrain.com/amplify/v0.1

Hierarchy
---------
    User  ->  Marketer (≈ ad account)  ->  Budget
                                       ->  Campaign  ->  PromotedLink (≈ ad)

There is no ad-set / ad-group layer: a Campaign directly owns its
PromotedLinks, so the adapter collapses the ad-set scope onto the campaign
(same shape as the SmartNews AMv1 adapter).

Authentication
--------------
Every request carries an ``OB-TOKEN-V1: <token>`` header. A token can be
supplied directly (the common case — Outbrain only lets you mint a token via
``/login`` twice per hour, so long-lived tokens are pasted/env-configured) or
minted on demand from a username + password via HTTP Basic auth against
``GET /login``. When both are configured we prefer the supplied token and fall
back to a fresh login only after a 401.

Money
-----
Outbrain works in plain currency floats (e.g. ``cpc: 0.55`` dollars,
``budget.amount: 500`` dollars). The adapter converts our internal integer
"cents" to/from these floats at the edge.

Pagination
----------
List endpoints use ``limit`` (max 50) + ``offset`` and wrap rows in a named
collection (``marketers`` / ``budgets`` / ``campaigns`` / ``promotedLinks``)
alongside ``count`` and ``totalCount``. Use :meth:`OutbrainClient.paginate`.
"""
from __future__ import annotations

import base64
import logging
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.outbrain.com/amplify/v0.1"
TOKEN_HEADER = "OB-TOKEN-V1"
PAGE_LIMIT = 50  # Outbrain caps list pages at 50 rows.


class OutbrainAPIError(Exception):
    def __init__(self, message: str, status_code: Optional[int] = None, body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class OutbrainAuthError(OutbrainAPIError):
    """Raised when the token is missing/expired and login is impossible."""


# ----------------------------------------------------------------------
# Envelope helpers (mirror smartnews_api.unwrap_* so adapters read alike)
# ----------------------------------------------------------------------
def unwrap_list(body: Any, *keys: str) -> List[Dict[str, Any]]:
    """Pull the row collection out of an Outbrain list envelope.

    Tries the explicit ``keys`` first, then the common collection names.
    """
    if isinstance(body, list):
        return [r for r in body if isinstance(r, dict)]
    if not isinstance(body, dict):
        return []
    for k in (*keys, "marketers", "budgets", "campaigns", "promotedLinks", "results", "data"):
        v = body.get(k)
        if isinstance(v, list):
            return [r for r in v if isinstance(r, dict)]
    return []


class OutbrainClient:
    """Thin wrapper around the Outbrain Amplify v0.1 REST API."""

    def __init__(
        self,
        token: Optional[str] = None,
        *,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        base_url: str = BASE_URL,
    ):
        if not token and not (username and password):
            raise ValueError(
                "OutbrainClient requires a token or username+password credentials"
            )
        self.username = username
        self.password = password
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.base_url = base_url.rstrip("/")

        self._session = requests.Session()
        self._session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "newsbreakmaster/outbrain (+https://newsbreakmaster.onrender.com)",
            }
        )
        self._lock = threading.Lock()
        self._token: Optional[str] = (token or "").strip() or None

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------
    def login(self) -> str:
        """Mint a fresh OB-TOKEN-V1 from username + password (HTTP Basic).

        Outbrain rate-limits ``/login`` to ~twice per hour per user, so this
        is only called when no token is available or a cached one 401s.
        """
        if not (self.username and self.password):
            raise OutbrainAuthError("Outbrain login requires username + password")
        creds = base64.b64encode(
            f"{self.username}:{self.password}".encode("utf-8")
        ).decode("ascii")
        try:
            resp = self._session.get(
                self.base_url + "/login",
                headers={"Authorization": f"Basic {creds}"},
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise OutbrainAuthError(f"Outbrain login request failed: {e}") from e
        try:
            body = resp.json() if resp.text else {}
        except ValueError:
            body = {}
        if resp.status_code >= 400:
            msg = (body.get("message") if isinstance(body, dict) else None) or resp.reason
            raise OutbrainAuthError(
                f"Outbrain login failed ({resp.status_code}): {msg}",
                status_code=resp.status_code,
                body=body,
            )
        token = (body or {}).get(TOKEN_HEADER) or (body or {}).get("token")
        if not token:
            raise OutbrainAuthError("Outbrain login response missing OB-TOKEN-V1", body=body)
        with self._lock:
            self._token = token
        return token

    def _get_token(self, force_refresh: bool = False) -> str:
        with self._lock:
            tok = self._token
        if tok and not force_refresh:
            return tok
        if self.username and self.password:
            return self.login()
        if tok:
            return tok
        raise OutbrainAuthError("No Outbrain token available and no login credentials")

    @property
    def token(self) -> Optional[str]:
        return self._token

    def verify(self) -> None:
        """Raise if the current credentials cannot list marketers."""
        self.get("/marketers", params={"limit": 1})

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
        files: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        allow_empty: bool = False,
        _retry_on_401: bool = True,
    ) -> Any:
        url = path if path.startswith("http") else self.base_url + path
        last_err: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            headers = {TOKEN_HEADER: self._get_token()}
            if json_body is not None and not files:
                headers["Content-Type"] = "application/json"
            try:
                resp = self._session.request(
                    method,
                    url,
                    params=params,
                    json=json_body if (json_body is not None and not files) else None,
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=self.timeout,
                )
            except requests.RequestException as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise OutbrainAPIError(str(e)) from e

            # Transparent re-auth on 401 when we can mint a fresh token.
            if resp.status_code == 401 and _retry_on_401 and self.username and self.password:
                try:
                    self._get_token(force_refresh=True)
                except OutbrainAuthError:
                    pass
                _retry_on_401 = False
                continue

            if resp.status_code == 429 or (500 <= resp.status_code < 600):
                # Honour Retry-After when present, else exponential-ish backoff.
                last_err = OutbrainAPIError(
                    f"Outbrain transient error {resp.status_code}",
                    status_code=resp.status_code,
                )
                if attempt < self.max_retries:
                    retry_after = resp.headers.get("Retry-After")
                    delay = self.retry_delay * (attempt + 1)
                    if retry_after and str(retry_after).isdigit():
                        delay = max(delay, float(retry_after))
                    time.sleep(delay)
                    continue
                raise last_err

            text = resp.text
            try:
                body = resp.json() if text else {}
            except ValueError:
                body = {"raw": text}

            if resp.status_code >= 400:
                msg = (
                    (body.get("message") if isinstance(body, dict) else None)
                    or text
                    or resp.reason
                )
                if isinstance(body, dict):
                    verrs = body.get("validationErrors") or body.get("validation_errors")
                    if isinstance(verrs, list) and verrs:
                        msg = f"{msg} [{'; '.join(str(v) for v in verrs)}]"
                req_id = resp.headers.get("AMPLIFY-REQUEST-ID")
                if req_id:
                    msg = f"{msg} (request-id {req_id})"
                raise OutbrainAPIError(
                    f"Outbrain API error ({resp.status_code}): {msg}",
                    status_code=resp.status_code,
                    body=body,
                )

            if not body and not allow_empty:
                return {}
            return body

        if last_err:
            raise OutbrainAPIError(str(last_err)) from last_err
        raise OutbrainAPIError("Unknown request error")

    def get(self, path: str, **kwargs: Any) -> Any:
        return self._request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> Any:
        return self._request("POST", path, **kwargs)

    def put(self, path: str, **kwargs: Any) -> Any:
        return self._request("PUT", path, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> Any:
        return self._request("DELETE", path, allow_empty=True, **kwargs)

    # ------------------------------------------------------------------
    # Pagination
    # ------------------------------------------------------------------
    def paginate(
        self,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        collection: Optional[str] = None,
        limit: int = PAGE_LIMIT,
        max_rows: int = 5000,
    ) -> Iterator[Dict[str, Any]]:
        params = dict(params or {})
        offset = 0
        seen = 0
        limit = max(1, min(int(limit), PAGE_LIMIT))
        while True:
            params["limit"] = limit
            params["offset"] = offset
            body = self.get(path, params=params)
            rows = unwrap_list(body, *( (collection,) if collection else () ))
            if not rows:
                break
            for r in rows:
                yield r
                seen += 1
                if seen >= max_rows:
                    return
            total = body.get("totalCount") if isinstance(body, dict) else None
            offset += len(rows)
            if total is not None and offset >= int(total):
                break
            if len(rows) < limit:
                break

    # ------------------------------------------------------------------
    # Marketers (ad accounts)
    # ------------------------------------------------------------------
    def get_marketers(self) -> List[Dict[str, Any]]:
        return list(self.paginate("/marketers", collection="marketers"))

    def get_marketer(self, marketer_id: str) -> Dict[str, Any]:
        return self.get(f"/marketers/{marketer_id}")

    # ------------------------------------------------------------------
    # Budgets
    # ------------------------------------------------------------------
    def get_budgets(self, marketer_id: str) -> List[Dict[str, Any]]:
        return list(
            self.paginate(f"/marketers/{marketer_id}/budgets", collection="budgets")
        )

    def create_budget(self, marketer_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.post(f"/marketers/{marketer_id}/budgets", json_body=payload)

    def update_budget(self, budget_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.put(f"/budgets/{budget_id}", json_body=payload)

    # ------------------------------------------------------------------
    # Campaigns
    # ------------------------------------------------------------------
    _CAMPAIGN_EXTRA = "Locations,PlatformTargeting,CampaignOptimization,Scheduling"

    def iter_campaigns(
        self, marketer_id: str, *, extra_fields: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        params = {"extraFields": extra_fields or self._CAMPAIGN_EXTRA}
        yield from self.paginate(
            f"/marketers/{marketer_id}/campaigns",
            params=params,
            collection="campaigns",
        )

    def get_campaign(self, campaign_id: str, *, extra_fields: Optional[str] = None) -> Dict[str, Any]:
        params = {"extraFields": extra_fields or self._CAMPAIGN_EXTRA}
        return self.get(f"/campaigns/{campaign_id}", params=params)

    def create_campaign(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Outbrain wants the global endpoint for creation (the marketer-scoped
        # POST returns 405). The marketer is inferred from the budgetId.
        return self.post("/campaigns", json_body=payload)

    def update_campaign(self, campaign_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.put(f"/campaigns/{campaign_id}", json_body=payload)

    # ------------------------------------------------------------------
    # Promoted Links (ads)
    # ------------------------------------------------------------------
    _PL_EXTRA = "ImageMetaData"

    def iter_promoted_links(
        self, campaign_id: str, *, extra_fields: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        params = {"extraFields": extra_fields or self._PL_EXTRA}
        yield from self.paginate(
            f"/campaigns/{campaign_id}/promotedLinks",
            params=params,
            collection="promotedLinks",
        )

    def create_promoted_link(self, campaign_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Create a promoted link referencing a hosted ``imageUrl``."""
        return self.post(f"/campaigns/{campaign_id}/promotedLinks", json_body=payload)

    def create_promoted_link_with_image(
        self,
        campaign_id: str,
        *,
        image_bytes: bytes,
        filename: str,
        url: str,
        text: str,
        enabled: bool = True,
        call_to_action: Optional[str] = None,
        mime_type: str = "image/jpeg",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a promoted link by uploading local image bytes (multipart).

        Outbrain's documented contract for byte uploads is a ``multipart/
        form-data`` POST with text fields (``url``, ``enabled``, ``text``,
        optional ``callToAction``) plus an ``image`` file part.
        """
        data: Dict[str, Any] = {
            "url": url,
            "text": text,
            "enabled": "true" if enabled else "false",
        }
        if call_to_action:
            data["callToAction"] = call_to_action
        if extra:
            for k, v in extra.items():
                data[k] = v
        files = {"image": (filename, image_bytes, mime_type)}
        return self.post(
            f"/campaigns/{campaign_id}/promotedLinks",
            files=files,
            data=data,
        )

    def update_promoted_link(self, promoted_link_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.put(f"/promotedLinks/{promoted_link_id}", json_body=payload)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def report_campaigns(
        self,
        marketer_id: str,
        *,
        date_from: str,
        date_to: str,
        include_conversions: bool = True,
        limit: int = PAGE_LIMIT,
    ) -> List[Dict[str, Any]]:
        params = {
            "from": date_from,
            "to": date_to,
            "includeConversions": "true" if include_conversions else "false",
        }
        return list(
            self.paginate(
                f"/reports/marketers/{marketer_id}/campaigns",
                params=params,
                collection="results",
                limit=limit,
            )
        )

    def report_promoted_links(
        self,
        campaign_id: str,
        *,
        date_from: str,
        date_to: str,
        include_conversions: bool = True,
        limit: int = PAGE_LIMIT,
    ) -> List[Dict[str, Any]]:
        params = {
            "from": date_from,
            "to": date_to,
            "includeConversions": "true" if include_conversions else "false",
        }
        return list(
            self.paginate(
                f"/reports/campaigns/{campaign_id}/promotedLinks",
                params=params,
                collection="results",
                limit=limit,
            )
        )

    # ------------------------------------------------------------------
    # Misc lookups
    # ------------------------------------------------------------------
    def search_locations(self, term: str, *, limit: int = 10) -> List[Dict[str, Any]]:
        body = self.get("/locations/search", params={"term": term, "limit": limit})
        if isinstance(body, list):
            return [r for r in body if isinstance(r, dict)]
        return unwrap_list(body, "locations", "results")

    def list_conversions(self, marketer_id: str) -> List[Dict[str, Any]]:
        try:
            body = self.get(f"/marketers/{marketer_id}/conversions")
        except OutbrainAPIError:
            return []
        return unwrap_list(body, "conversions", "results")


# ----------------------------------------------------------------------
# Money helpers (Outbrain works in currency floats; we store integer cents)
# ----------------------------------------------------------------------
def cents_to_amount(cents: Optional[int]) -> Optional[float]:
    if cents is None:
        return None
    return round(int(cents) / 100.0, 2)


def amount_to_cents(amount: Any) -> Optional[int]:
    if amount in (None, ""):
        return None
    try:
        return int(round(float(amount) * 100))
    except (TypeError, ValueError):
        return None


__all__ = [
    "OutbrainClient",
    "OutbrainAPIError",
    "OutbrainAuthError",
    "unwrap_list",
    "cents_to_amount",
    "amount_to_cents",
    "BASE_URL",
]

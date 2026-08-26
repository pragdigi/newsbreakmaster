"""
MediaGo (mediago.io) Marketing API client.

Official docs: https://apidoc.mediago.io
Base:          https://api.mediago.io

Auth
----
The AM issues a Base64 API token. Exchange it once for a short-lived
Bearer access token (expires ~3600s). A *new* access token invalidates
the previous one, so this client caches and reuses the token until it
is about to expire.

Two token classes exist:

  * Account-level  ``POST /data/v1/authentication``
  * Client-level   ``POST /data/v1/client/authentication``
    — lists every ad account under the client and requires an
      ``Account-Id`` header on subsequent manage/report calls.

``auth_level="auto"`` tries client first, then account.

Hierarchy
---------
    Client  →  Account  →  Campaign  →  Ad

There is no ad-set layer. Native creatives are URLs + headline (max 80
chars) + brand_name (max 30). Site blocklists live at account and
campaign scope.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.mediago.io"
TOKEN_REFRESH_SKEW_S = 60


class MediaGoAPIError(Exception):
    def __init__(self, message: str, status_code: Optional[int] = None, body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class MediaGoAuthError(MediaGoAPIError):
    """Raised when the API token cannot be exchanged for an access token."""


def unwrap_list(body: Any, *keys: str) -> List[Dict[str, Any]]:
    """Pull a row collection out of a MediaGo envelope."""
    if isinstance(body, list):
        return [r for r in body if isinstance(r, dict)]
    if not isinstance(body, dict):
        return []
    for k in (*keys, "results", "result", "accounts", "data", "blocked_sites"):
        v = body.get(k)
        if isinstance(v, list):
            return [r for r in v if isinstance(r, dict)]
        if isinstance(v, dict):
            for inner in ("accounts", "blocked_sites", "results", "result"):
                rows = v.get(inner)
                if isinstance(rows, list):
                    return [r for r in rows if isinstance(r, dict)]
    return []


def unwrap_data(body: Any) -> Any:
    if isinstance(body, dict) and "data" in body:
        return body.get("data")
    return body


def basic_authorization(api_token: str) -> str:
    """Build the ``Authorization: Basic …`` header from an AM-issued token."""
    t = (api_token or "").strip()
    if not t:
        raise MediaGoAuthError("MediaGo API token is empty")
    if t.lower().startswith("basic "):
        return t
    return f"Basic {t}"


class MediaGoClient:
    """Thin wrapper around the MediaGo REST API."""

    def __init__(
        self,
        api_token: str,
        *,
        auth_level: str = "auto",
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        base_url: str = BASE_URL,
    ):
        if not (api_token or "").strip():
            raise ValueError("MediaGoClient requires api_token")
        self.api_token = api_token.strip()
        level = (auth_level or "auto").strip().lower()
        if level not in ("auto", "client", "account"):
            level = "auto"
        self.auth_level_pref = level
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.base_url = base_url.rstrip("/")

        self._session = requests.Session()
        self._session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "newsbreakmaster/mediago (+https://newsbreakmaster.onrender.com)",
            }
        )
        self._lock = threading.Lock()
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._resolved_level: Optional[str] = None
        self._client_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------
    @property
    def resolved_auth_level(self) -> Optional[str]:
        return self._resolved_level

    @property
    def client_id(self) -> Optional[str]:
        return self._client_id

    def authenticate(self, *, force: bool = False) -> str:
        """Exchange the API token for a Bearer access token."""
        with self._lock:
            if (
                not force
                and self._access_token
                and time.time() < self._token_expires_at
            ):
                return self._access_token

        preferred = self.auth_level_pref
        if preferred == "account":
            order = ("account",)
        elif preferred == "client":
            order = ("client",)
        else:
            order = ("client", "account")

        last_err: Optional[Exception] = None
        for level in order:
            try:
                return self._authenticate_level(level)
            except MediaGoAuthError as exc:
                last_err = exc
                logger.info("MediaGo %s-level auth failed: %s", level, exc)
                continue
        raise last_err or MediaGoAuthError("MediaGo authentication failed")

    def _authenticate_level(self, level: str) -> str:
        path = (
            "/data/v1/client/authentication"
            if level == "client"
            else "/data/v1/authentication"
        )
        url = self.base_url + path
        headers = {
            "Authorization": basic_authorization(self.api_token),
            "Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
        }
        try:
            resp = self._session.post(url, headers=headers, timeout=self.timeout)
        except requests.RequestException as e:
            raise MediaGoAuthError(f"MediaGo auth request failed: {e}") from e
        try:
            body = resp.json() if resp.text else {}
        except ValueError:
            body = {}
        if resp.status_code >= 400:
            msg = (
                (body.get("error") if isinstance(body, dict) else None)
                or (body.get("errmsg") if isinstance(body, dict) else None)
                or resp.reason
            )
            raise MediaGoAuthError(
                f"MediaGo {level} auth failed ({resp.status_code}): {msg}",
                status_code=resp.status_code,
                body=body,
            )
        token = (body or {}).get("access_token")
        if not token:
            raise MediaGoAuthError(
                f"MediaGo {level} auth response missing access_token", body=body
            )
        expires = int((body or {}).get("expires_in") or 3600)
        with self._lock:
            self._access_token = token
            self._token_expires_at = time.time() + max(30, expires - TOKEN_REFRESH_SKEW_S)
            self._resolved_level = level
            self._client_id = str((body or {}).get("client_id") or "") or None
        return token

    def _get_token(self, force_refresh: bool = False) -> str:
        return self.authenticate(force=force_refresh)

    def verify(self) -> None:
        """Raise if the current credentials cannot list accounts."""
        self.get_accounts()

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
        account_id: Optional[str] = None,
        _retry_on_401: bool = True,
    ) -> Any:
        url = path if path.startswith("http") else self.base_url + path
        last_err: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            headers = {
                "Authorization": f"Bearer {self._get_token()}",
            }
            if json_body is not None:
                headers["Content-Type"] = "application/json"
            else:
                headers["Content-Type"] = "application/x-www-form-urlencoded;charset=utf-8"
            if account_id and (self._resolved_level == "client" or self.auth_level_pref == "client"):
                headers["Account-Id"] = str(account_id)
            try:
                resp = self._session.request(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    headers=headers,
                    timeout=self.timeout,
                )
            except requests.RequestException as e:
                last_err = MediaGoAPIError(f"MediaGo request failed: {e}")
                time.sleep(self.retry_delay * (attempt + 1))
                continue

            if resp.status_code in (401, 403) and _retry_on_401:
                self._get_token(force_refresh=True)
                return self._request(
                    method,
                    path,
                    params=params,
                    json_body=json_body,
                    account_id=account_id,
                    _retry_on_401=False,
                )

            try:
                body = resp.json() if resp.text else {}
            except ValueError:
                body = {"raw": resp.text}

            if resp.status_code >= 400:
                msg = (
                    (body.get("error") if isinstance(body, dict) else None)
                    or (body.get("errmsg") if isinstance(body, dict) else None)
                    or (body.get("message") if isinstance(body, dict) else None)
                    or (body.get("msg") if isinstance(body, dict) else None)
                    or resp.reason
                )
                raise MediaGoAPIError(
                    f"MediaGo {method} {path} failed ({resp.status_code}): {msg}",
                    status_code=resp.status_code,
                    body=body,
                )
            if isinstance(body, dict):
                errno = body.get("errno")
                code = body.get("code")
                if errno not in (None, 0) or (isinstance(code, int) and code not in (0, 200)):
                    msg = body.get("errmsg") or body.get("error") or body.get("message") or "error"
                    raise MediaGoAPIError(
                        f"MediaGo {method} {path} rejected: {msg}",
                        status_code=resp.status_code,
                        body=body,
                    )
            return body

        raise last_err or MediaGoAPIError(f"MediaGo {method} {path} failed after retries")

    def get(
        self,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        account_id: Optional[str] = None,
    ) -> Any:
        return self._request("GET", path, params=params, account_id=account_id)

    def post(
        self,
        path: str,
        json_body: Any,
        *,
        account_id: Optional[str] = None,
    ) -> Any:
        return self._request("POST", path, json_body=json_body, account_id=account_id)

    def paginate_report(
        self,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        account_id: Optional[str] = None,
        page_size: int = 100,
        rows_keys: Sequence[str] = ("results", "result"),
    ) -> Iterator[Dict[str, Any]]:
        page = 1
        seen = 0
        while True:
            q = dict(params or {})
            q["page_size"] = page_size
            q["current_page"] = page
            body = self.get(path, params=q, account_id=account_id)
            rows: List[Dict[str, Any]] = []
            if isinstance(body, dict):
                for k in rows_keys:
                    v = body.get(k)
                    if isinstance(v, list):
                        rows = [r for r in v if isinstance(r, dict)]
                        break
            elif isinstance(body, list):
                rows = [r for r in body if isinstance(r, dict)]
            if not rows:
                break
            for r in rows:
                yield r
            seen += len(rows)
            total = 0
            if isinstance(body, dict):
                try:
                    total = int(body.get("total") or 0)
                except (TypeError, ValueError):
                    total = 0
            if total and seen >= total:
                break
            if len(rows) < page_size:
                break
            page += 1

    # ------------------------------------------------------------------
    # Accounts
    # ------------------------------------------------------------------
    def get_client_accounts(self) -> List[Dict[str, Any]]:
        body = self.get("/data/v1/client/accounts")
        return unwrap_list(body, "accounts")

    def get_authorized_accounts(self, *, account_id: Optional[str] = None) -> List[Dict[str, Any]]:
        body = self.get("/manage/v1/account", params={"auth_level": "rw"}, account_id=account_id)
        return unwrap_list(body)

    def get_accounts(self) -> List[Dict[str, Any]]:
        """List ad accounts visible to this token (client or account level)."""
        self._get_token()
        if self._resolved_level == "client":
            rows = self.get_client_accounts()
            if rows:
                return rows
        return self.get_authorized_accounts()

    def list_account_pixels(self, account_id: str) -> List[Dict[str, Any]]:
        """Conversion pixels for one account from ``GET /manage/v1/account``.

        MediaGo has no standalone pixel-id object. Each account returns a
        ``pixels`` array of conversion trackers (``conversion_name``,
        ``category``, ``status``, ``include_in_total_conversion``). Client-level
        ``/data/v1/client/accounts`` often omits this array, so we always hit
        the manage endpoint with ``Account-Id``.
        """
        aid = str(account_id or "").strip()
        if not aid:
            return []
        rows = self.get_authorized_accounts(account_id=aid)
        matched: Optional[Dict[str, Any]] = None
        for acc in rows:
            if str(acc.get("account_id") or acc.get("id") or "") == aid:
                matched = acc
                break
        if matched is None and len(rows) == 1:
            matched = rows[0]
        if not matched:
            return []
        pixels = matched.get("pixels")
        if not isinstance(pixels, list):
            return []
        return [p for p in pixels if isinstance(p, dict)]

    # ------------------------------------------------------------------
    # Campaigns / ads
    # ------------------------------------------------------------------
    def list_campaigns(
        self,
        account_id: str,
        *,
        page_type: str = "0",
        page_size: int = 50,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"auth_level": "rw", "page_type": page_type}
        if page_type == "1":
            params["page_size"] = page_size
            params["current_page"] = 1
        body = self.get("/manage/v1/campaign", params=params, account_id=account_id)
        return unwrap_list(body)

    def get_campaign_detail(
        self, account_id: str, campaign_ids: Union[str, Sequence[str]]
    ) -> List[Dict[str, Any]]:
        if isinstance(campaign_ids, (list, tuple)):
            ids = ",".join(str(x) for x in campaign_ids if x)
        else:
            ids = str(campaign_ids)
        body = self.get(
            "/manage/v1/campaign/detail",
            params={"campaign_ids": ids},
            account_id=account_id,
        )
        return unwrap_list(body)

    def create_campaign(self, account_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = dict(payload)
        body.setdefault("account_id", str(account_id))
        return self.post("/manage/v1/campaign/create", body, account_id=account_id)

    def update_campaign(self, account_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.post("/manage/v1/campaign/update", payload, account_id=account_id)

    def set_campaign_status(
        self, account_id: str, campaign_ids: Sequence[str], enabled: bool
    ) -> Any:
        return self.post(
            "/manage/v1/campaign/status",
            {"campaign_ids": [str(x) for x in campaign_ids], "status": 1 if enabled else 0},
            account_id=account_id,
        )

    def list_ads(
        self, account_id: str, campaign_ids: Optional[Sequence[str]] = None
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"auth_level": "rw"}
        if campaign_ids:
            params["campaign_ids"] = ",".join(str(x) for x in campaign_ids)
        body = self.get("/manage/v1/campaign/ad", params=params, account_id=account_id)
        return unwrap_list(body)

    def set_ad_status(self, account_id: str, ad_ids: Sequence[str], enabled: bool) -> Any:
        return self.post(
            "/manage/v1/campaign/ad/status",
            {"ad_ids": [str(x) for x in ad_ids], "status": 1 if enabled else 0},
            account_id=account_id,
        )

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------
    def campaign_daily_report(
        self,
        account_id: str,
        start_date: str,
        end_date: str,
        *,
        timezone: str = "est",
        campaign_ids: Optional[Sequence[str]] = None,
    ) -> Iterator[Dict[str, Any]]:
        params: Dict[str, Any] = {
            "start_date": start_date,
            "end_date": end_date,
            "timezone": timezone,
            "sort_field": "spend",
            "sort_val": "desc",
        }
        if campaign_ids:
            params["campaign_ids"] = ",".join(str(x) for x in campaign_ids)
        yield from self.paginate_report(
            "/data/v1/report/day/list",
            params=params,
            account_id=account_id,
        )

    def ad_daily_report(
        self,
        account_id: str,
        start_date: str,
        end_date: str,
        *,
        timezone: str = "est",
    ) -> Iterator[Dict[str, Any]]:
        params = {
            "start_date": start_date,
            "end_date": end_date,
            "timezone": timezone,
            "sort_field": "spend",
            "sort_val": "desc",
        }
        yield from self.paginate_report(
            "/data/v1/report/ad/day/list",
            params=params,
            account_id=account_id,
        )

    def account_site_report(
        self,
        account_id: str,
        start_date: str,
        end_date: str,
        *,
        timezone: str = "est",
    ) -> Iterator[Dict[str, Any]]:
        """Site dimension — API max window is 1 day; caller should loop dates."""
        params = {
            "account_id": str(account_id),
            "start_date": start_date,
            "end_date": end_date,
            "timezone": timezone,
            "sort_field": "spend",
            "sort_val": "desc",
        }
        yield from self.paginate_report(
            "/data/v1/report/site/day/accountList",
            params=params,
            account_id=account_id,
            page_size=1000,
            rows_keys=("result", "results"),
        )

    def campaign_site_report(
        self,
        account_id: str,
        campaign_id: str,
        start_date: str,
        end_date: str,
        *,
        timezone: str = "est",
    ) -> Iterator[Dict[str, Any]]:
        """Site dimension for one campaign — API max window is 7 days."""
        params = {
            "campaign_id": str(campaign_id),
            "start_date": start_date,
            "end_date": end_date,
            "timezone": timezone,
            "sort_field": "spend",
            "sort_val": "desc",
        }
        yield from self.paginate_report(
            "/manage/v1/report/site/day/list",
            params=params,
            account_id=account_id,
        )

    # ------------------------------------------------------------------
    # Site blocklists
    # ------------------------------------------------------------------
    def get_account_block_list(self, account_id: str) -> List[Dict[str, Any]]:
        body = self.get(
            "/manage/v1/account/domain/block",
            params={"account_id": str(account_id)},
            account_id=account_id,
        )
        return unwrap_list(body, "blocked_sites")

    def block_account_sites(
        self,
        account_id: str,
        sites: Sequence[Dict[str, Any]],
        *,
        block: bool = True,
    ) -> Any:
        return self.post(
            "/manage/v1/account/domain/block",
            {
                "account_id": str(account_id),
                "op": "0" if block else "1",
                "block_sites": list(sites),
            },
            account_id=account_id,
        )

    def get_campaign_block_list(self, account_id: str, campaign_id: str) -> List[Dict[str, Any]]:
        body = self.get(
            "/manage/v1/campaign/domain/block",
            params={"account_id": str(account_id), "campaign_id": str(campaign_id)},
            account_id=account_id,
        )
        return unwrap_list(body, "blocked_sites")

    def block_campaign_sites(
        self,
        account_id: str,
        campaign_id: str,
        sites: Sequence[Dict[str, Any]],
        *,
        block: bool = True,
    ) -> Any:
        payload_sites = []
        for s in sites:
            row = {
                "site_id": int(s["site_id"]),
                "domain_name": s.get("domain_name") or s.get("site_name") or "",
                "campaign_id": int(campaign_id),
            }
            payload_sites.append(row)
        return self.post(
            "/manage/v1/campaign/domain/block",
            {
                "account_id": str(account_id),
                "op": "0" if block else "1",
                "block_sites": payload_sites,
            },
            account_id=account_id,
        )


__all__ = [
    "MediaGoAPIError",
    "MediaGoAuthError",
    "MediaGoClient",
    "BASE_URL",
    "unwrap_list",
    "unwrap_data",
    "basic_authorization",
]

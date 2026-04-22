"""
SmartNews Ads Advertising API client.
Docs: https://github.com/smartnews/smartnews-ads-advertising-api-spec/blob/v1.0/README.md

Authentication: X-Auth-Api header.
Base URL:       https://ads.smartnews.com/api

Hierarchy (AMv1 — what this client targets):
    Account -> Campaign -> Creative

AMv2 data (3-level Campaign -> AdGroup -> Ad) is surfaced by SmartNews through
the same endpoints via an ``amV2`` object embedded in responses; this client
returns the raw payloads and leaves interpretation to the adapter layer.

All money amounts are in JPY (integer units).
"""
from __future__ import annotations

import json
import time
from typing import Any, BinaryIO, Dict, List, Optional, Union

import requests

BASE_URL = "https://ads.smartnews.com/api"
API_VERSION = "v1.0"
AUDIENCE_VERSION = "v2.0"


class SmartNewsAPIError(Exception):
    def __init__(self, message: str, status_code: Optional[int] = None, body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class SmartNewsClient:
    """Thin wrapper around SmartNews Ads Management + Insights APIs."""

    def __init__(self, api_key: str, timeout: int = 60):
        self.api_key = api_key
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "X-Auth-Api": api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

    # ------------------------------------------------------------------
    # Low-level request helpers
    # ------------------------------------------------------------------
    def _url(self, path: str, *, version: str = API_VERSION) -> str:
        p = path if path.startswith("/") else f"/{path}"
        if p.startswith(f"/{version}/") or p.startswith(f"/{version}?"):
            return f"{BASE_URL}{p}"
        return f"{BASE_URL}/{version}{p}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Any = None,
        data: Any = None,
        files: Any = None,
        headers: Optional[Dict[str, str]] = None,
        retries: int = 4,
        version: str = API_VERSION,
    ) -> Any:
        url = self._url(path, version=version)
        req_headers = dict(self._session.headers)
        if headers:
            req_headers.update(headers)
        if files is not None:
            req_headers["Content-Type"] = None  # let requests set multipart boundary

        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                resp = self._session.request(
                    method,
                    url,
                    params=params,
                    json=json_body if files is None else None,
                    data=data,
                    files=files,
                    headers=req_headers,
                    timeout=self.timeout,
                )
                if resp.status_code == 429 or (500 <= resp.status_code < 600):
                    if attempt < retries:
                        time.sleep(2**attempt)
                        continue
                text = resp.text
                try:
                    body = resp.json() if text else {}
                except json.JSONDecodeError:
                    body = {"raw": text}
                if resp.status_code >= 400:
                    err = body.get("error") if isinstance(body, dict) else None
                    msg = (
                        (err or {}).get("message")
                        if isinstance(err, dict)
                        else None
                    ) or body.get("message") if isinstance(body, dict) else None
                    msg = msg or text or resp.reason
                    raise SmartNewsAPIError(
                        f"SmartNews API error ({resp.status_code}): {msg}",
                        status_code=resp.status_code,
                        body=body,
                    )
                return body
            except SmartNewsAPIError:
                raise
            except requests.RequestException as e:
                last_err = e
                if attempt < retries:
                    time.sleep(2**attempt)
                    continue
                raise SmartNewsAPIError(str(e)) from e
        if last_err:
            raise SmartNewsAPIError(str(last_err)) from last_err
        raise SmartNewsAPIError("Unknown request error")

    # ------------------------------------------------------------------
    # Accounts
    # ------------------------------------------------------------------
    def get_accounts(self) -> Any:
        """GET /v1.0/accounts — list advertiser accounts the API key has access to."""
        return self._request("GET", "/accounts")

    # ------------------------------------------------------------------
    # Campaigns
    # ------------------------------------------------------------------
    def get_campaigns(self, account_id: str) -> Any:
        return self._request("GET", f"/accounts/{account_id}/campaigns")

    def get_campaign(self, campaign_id: str) -> Any:
        return self._request("GET", f"/campaigns/{campaign_id}")

    def create_campaign(self, account_id: str, payload: Dict[str, Any]) -> Any:
        return self._request("POST", f"/accounts/{account_id}/campaigns", json_body=payload)

    def update_campaign(self, campaign_id: str, payload: Dict[str, Any]) -> Any:
        return self._request("POST", f"/campaigns/{campaign_id}/update", json_body=payload)

    def update_campaign_enable(self, campaign_id: str, enable: bool) -> Any:
        """Request body is a bare JSON boolean."""
        return self._request(
            "POST",
            f"/campaigns/{campaign_id}/update_enable",
            json_body=bool(enable),
        )

    def submit_review(self, campaign_id: str) -> Any:
        return self._request("POST", f"/campaigns/{campaign_id}/submit_review")

    def pullback_pending_review(self, campaign_id: str) -> Any:
        return self._request("POST", f"/campaigns/{campaign_id}/pullback_pending_review")

    # ------------------------------------------------------------------
    # Creatives
    # ------------------------------------------------------------------
    def get_creatives(self, campaign_id: str) -> Any:
        return self._request("GET", f"/campaigns/{campaign_id}/creatives")

    def get_creative(self, creative_id: str) -> Any:
        return self._request("GET", f"/creatives/{creative_id}")

    def create_creative(self, campaign_id: str, payload: Dict[str, Any]) -> Any:
        return self._request(
            "POST",
            f"/campaigns/{campaign_id}/creatives",
            json_body=payload,
        )

    def update_creative(self, creative_id: str, payload: Dict[str, Any]) -> Any:
        return self._request("POST", f"/creatives/{creative_id}/update", json_body=payload)

    def update_creative_enable(self, creative_id: str, enable: bool) -> Any:
        return self._request(
            "POST",
            f"/creatives/{creative_id}/update_enable",
            json_body=bool(enable),
        )

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------
    def upload_image(
        self,
        account_id: str,
        file_obj: Union[BinaryIO, bytes],
        filename: str,
    ) -> Any:
        """POST /v1.0/accounts/{accountId}/images/upload — multipart upload.

        SmartNews enforces a ≤500KB upload size per file.
        """
        if isinstance(file_obj, bytes):
            files = {"file": (filename, file_obj)}
        else:
            files = {"file": (filename, file_obj)}
        return self._request(
            "POST",
            f"/accounts/{account_id}/images/upload",
            files=files,
        )

    # ------------------------------------------------------------------
    # Taxonomies
    # ------------------------------------------------------------------
    def get_publishers(self) -> Any:
        return self._request("GET", "/publishers")

    def get_adcategories(self) -> Any:
        return self._request("GET", "/adcategories")

    def get_genres(self) -> Any:
        return self._request("GET", "/genres")

    def get_cities(self) -> Any:
        return self._request("GET", "/cities")

    # ------------------------------------------------------------------
    # Insights
    # ------------------------------------------------------------------
    def get_account_insights(
        self,
        account_id: str,
        *,
        since: Optional[str] = None,
        until: Optional[str] = None,
        date_preset: Optional[str] = None,
        breakdowns: Optional[str] = None,
        level: Optional[str] = None,
        fields_presets: Optional[str] = None,
    ) -> Any:
        params = self._insights_params(
            since=since,
            until=until,
            date_preset=date_preset,
            breakdowns=breakdowns,
            level=level,
            fields_presets=fields_presets,
        )
        return self._request("GET", f"/accounts/{account_id}/insights", params=params)

    def get_campaign_insights(
        self,
        campaign_id: str,
        *,
        since: Optional[str] = None,
        until: Optional[str] = None,
        date_preset: Optional[str] = None,
        breakdowns: Optional[str] = None,
        level: Optional[str] = None,
        fields_presets: Optional[str] = None,
    ) -> Any:
        params = self._insights_params(
            since=since,
            until=until,
            date_preset=date_preset,
            breakdowns=breakdowns,
            level=level,
            fields_presets=fields_presets,
        )
        return self._request("GET", f"/campaigns/{campaign_id}/insights", params=params)

    def get_creative_insights(
        self,
        creative_id: str,
        *,
        since: Optional[str] = None,
        until: Optional[str] = None,
        date_preset: Optional[str] = None,
        breakdowns: Optional[str] = None,
        level: Optional[str] = None,
        fields_presets: Optional[str] = None,
    ) -> Any:
        params = self._insights_params(
            since=since,
            until=until,
            date_preset=date_preset,
            breakdowns=breakdowns,
            level=level,
            fields_presets=fields_presets,
        )
        return self._request("GET", f"/creatives/{creative_id}/insights", params=params)

    @staticmethod
    def _insights_params(**kwargs: Any) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in kwargs.items():
            if v is None or v == "":
                continue
            out[k] = v
        return out

    # ------------------------------------------------------------------
    # Custom audiences (v2.0)
    # ------------------------------------------------------------------
    def upload_idlist(self, account_id: str, file_obj: Union[BinaryIO, bytes], filename: str) -> Any:
        if isinstance(file_obj, bytes):
            files = {"file": (filename, file_obj)}
        else:
            files = {"file": (filename, file_obj)}
        data = {"accountId": account_id}
        return self._request(
            "POST",
            "/audiences/idlist",
            data=data,
            files=files,
            version=AUDIENCE_VERSION,
        )

    def register_audience(
        self,
        account_id: str,
        *,
        name: str,
        idlist_id: str,
        description: Optional[str] = None,
    ) -> Any:
        payload: Dict[str, Any] = {
            "name": name,
            "idlistId": idlist_id,
            "accountId": account_id,
            "type": "idlist",
        }
        if description:
            payload["description"] = description
        return self._request(
            "POST",
            "/audiences",
            json_body=payload,
            version=AUDIENCE_VERSION,
        )

    def get_audiences(self, account_id: str, audience_ids: Optional[List[str]] = None) -> Any:
        params: Dict[str, Any] = {"accountId": account_id}
        if audience_ids:
            params["audienceIds"] = ",".join(audience_ids)
        return self._request(
            "GET",
            "/audiences",
            params=params,
            version=AUDIENCE_VERSION,
        )


def unwrap_data(body: Any) -> Any:
    """SmartNews wraps responses as ``{"data": ...}``. Return the inner node."""
    if isinstance(body, dict) and "data" in body:
        return body["data"]
    return body


def unwrap_list(body: Any) -> List[Dict[str, Any]]:
    """Return the ``data`` array if present, else an empty list."""
    inner = unwrap_data(body)
    if isinstance(inner, list):
        return [x for x in inner if isinstance(x, dict)]
    if isinstance(inner, dict):
        return [inner]
    return []

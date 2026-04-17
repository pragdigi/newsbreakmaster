"""
NewsBreak Advertising API client.
Docs: https://advertising-api.newsbreak.com/hc/en-us/categories/37825505060237-API-Reference
"""
from __future__ import annotations

import json
import time
from typing import Any, BinaryIO, Dict, List, Optional, Union

import requests

BASE_URL = "https://business.newsbreak.com/business-api/v1"


class NewsBreakAPIError(Exception):
    def __init__(self, message: str, status_code: Optional[int] = None, body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class NewsBreakClient:
    """Thin wrapper around NewsBreak Business API v1."""

    def __init__(self, access_token: str, timeout: int = 60):
        self.access_token = access_token
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Access-Token": access_token,
                "Content-Type": "application/json",
            }
        )

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
    ) -> Any:
        url = f"{BASE_URL}{path}" if path.startswith("/") else f"{BASE_URL}/{path}"
        req_headers = dict(self._session.headers)
        if headers:
            req_headers.update(headers)
        if files is not None:
            # multipart: drop JSON Content-Type so requests sets boundary
            req_headers.pop("Content-Type", None)

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
                # Rate limit / transient
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
                    msg = body.get("message") or body.get("msg") or text or resp.reason
                    raise NewsBreakAPIError(
                        f"NewsBreak API error ({resp.status_code}): {msg}",
                        status_code=resp.status_code,
                        body=body,
                    )
                return body
            except NewsBreakAPIError:
                raise
            except requests.RequestException as e:
                last_err = e
                if attempt < retries:
                    time.sleep(2**attempt)
                    continue
                raise NewsBreakAPIError(str(e)) from e
        if last_err:
            raise NewsBreakAPIError(str(last_err)) from last_err
        raise NewsBreakAPIError("Unknown request error")

    # --- Ad accounts ---
    def get_ad_accounts(self, org_ids: List[str]) -> Any:
        """GET /ad-account/getGroupsByOrgIds — orgIds repeated query params."""
        if not org_ids:
            raise NewsBreakAPIError("At least one organization id (orgIds) is required")
        params = [("orgIds", oid) for oid in org_ids]
        url = f"{BASE_URL}/ad-account/getGroupsByOrgIds"
        return self._raw_get_with_multi_params(url, params)

    def _raw_get_with_multi_params(self, url: str, param_pairs: List[tuple]) -> Any:
        last_err: Optional[Exception] = None
        for attempt in range(5):
            try:
                resp = self._session.get(url, params=param_pairs, timeout=self.timeout)
                if resp.status_code == 429 or (500 <= resp.status_code < 600):
                    time.sleep(2**attempt)
                    continue
                text = resp.text
                try:
                    body = resp.json() if text else {}
                except json.JSONDecodeError:
                    body = {"raw": text}
                if resp.status_code >= 400:
                    raise NewsBreakAPIError(
                        f"NewsBreak API error ({resp.status_code}): {body}",
                        status_code=resp.status_code,
                        body=body,
                    )
                return body
            except NewsBreakAPIError:
                raise
            except requests.RequestException as e:
                last_err = e
                time.sleep(2**attempt)
        raise NewsBreakAPIError(str(last_err) if last_err else "request failed")

    # --- Campaigns ---
    def get_campaigns(
        self,
        ad_account_id: str,
        *,
        campaign_ids: Optional[List[str]] = None,
        status: Optional[str] = None,
        page_no: int = 1,
        page_size: int = 100,
    ) -> Any:
        params: Dict[str, Any] = {
            "adAccountId": ad_account_id,
            "pageNo": page_no,
            "pageSize": page_size,
        }
        if campaign_ids:
            params["campaignIds"] = campaign_ids
        if status:
            params["status"] = status
        return self._request("GET", "/campaign/getList", params=params)

    def create_campaign(self, payload: Dict[str, Any]) -> Any:
        return self._request("POST", "/campaign/create", json_body=payload)

    def update_campaign(self, campaign_id: str, payload: Dict[str, Any]) -> Any:
        return self._request("PUT", f"/campaign/update/{campaign_id}", json_body=payload)

    # --- Ad sets ---
    def get_ad_sets(self, campaign_id: str) -> Any:
        return self._request("GET", "/ad-set/list", params={"campaignId": campaign_id})

    def create_ad_set(self, payload: Dict[str, Any]) -> Any:
        return self._request("POST", "/ad-set/create", json_body=payload)

    def update_ad_set(self, ad_set_id: str, payload: Dict[str, Any]) -> Any:
        return self._request("PUT", f"/ad-set/update/{ad_set_id}", json_body=payload)

    def update_ad_set_status(self, ad_set_id: str, status: str) -> Any:
        """status: typically ON or OFF"""
        return self._request(
            "PUT",
            f"/ad-set/updateStatus/{ad_set_id}",
            json_body={"status": status},
        )

    # --- Ads ---
    def get_ads(self, ad_set_id: str) -> Any:
        return self._request("GET", "/ad/list", params={"adSetId": ad_set_id})

    def create_ad(self, payload: Dict[str, Any]) -> Any:
        return self._request("POST", "/ad/create", json_body=payload)

    def update_ad(self, ad_id: str, payload: Dict[str, Any]) -> Any:
        return self._request("PUT", f"/ad/update/{ad_id}", json_body=payload)

    def update_ad_status(self, ad_id: str, status: str) -> Any:
        return self._request(
            "PUT",
            f"/ad/updateStatus/{ad_id}",
            json_body={"status": status},
        )

    # --- Assets ---
    def upload_asset(
        self,
        file_obj: Union[BinaryIO, bytes],
        filename: str,
        ad_account_id: str,
        *,
        media_name: Optional[str] = None,
        save_to_media_library: bool = True,
    ) -> Any:
        """
        POST /ad/uploadAssets — multipart form.
        NewsBreak expects binary asset + adAccountId.
        """
        if isinstance(file_obj, bytes):
            files = {"asset": (filename, file_obj)}
        else:
            files = {"asset": (filename, file_obj)}
        data = {
            "adAccountId": ad_account_id,
            "saveToMediaLibrary": str(save_to_media_library).lower(),
        }
        if media_name:
            data["mediaName"] = media_name
        return self._request(
            "POST",
            "/ad/uploadAssets",
            data=data,
            files=files,
        )

    # --- Reporting ---
    def get_integrated_report(self, payload: Dict[str, Any]) -> Any:
        """POST /reports/getIntegratedReport — synchronous report."""
        return self._request("POST", "/reports/getIntegratedReport", json_body=payload)

    # --- Events / pixels ---
    def get_events(self, ad_account_id: str) -> Any:
        """GET /event/getList/{adAccountId} — returns tracking events configured in Ad Manager."""
        if not ad_account_id:
            raise NewsBreakAPIError("ad_account_id required")
        return self._request("GET", f"/event/getList/{ad_account_id}")


def unwrap_list_response(data: Any, keys: tuple = ("data", "list", "records", "items")) -> List[Dict[str, Any]]:
    """Normalize various API response shapes to a list of dicts."""
    if data is None:
        return []
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if not isinstance(data, dict):
        return []
    for k in keys:
        inner = data.get(k)
        if isinstance(inner, list):
            return [x for x in inner if isinstance(x, dict)]
    # Single object
    if "id" in data or "campaignId" in data:
        return [data]
    return []

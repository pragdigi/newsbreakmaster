"""
Bulk upload creatives and create campaign / ad sets / ads per grouping strategy.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from newsbreak_api import NewsBreakClient


def _first_id(obj: Any, *keys: str) -> Optional[str]:
    if not isinstance(obj, dict):
        return None
    for k in keys:
        v = obj.get(k)
        if v is not None:
            return str(v)
    return None


def _extract_id(resp: Any, *keys: str) -> Optional[str]:
    if resp is None:
        return None
    rid = _first_id(resp, *keys) if isinstance(resp, dict) else None
    if rid:
        return rid
    if isinstance(resp, dict):
        data = resp.get("data")
        if isinstance(data, dict):
            return _first_id(data, *keys)
    return None


def group_creatives(
    creatives: List[Dict[str, Any]],
    grouping: str,
    group_size: int = 5,
) -> List[List[Dict[str, Any]]]:
    """
    grouping: all_in_one | isolate | groups_of_n
    """
    if not creatives:
        return []
    if grouping == "all_in_one":
        return [creatives]
    if grouping == "isolate":
        return [[c] for c in creatives]
    # groups_of_n
    n = max(1, group_size)
    return [creatives[i : i + n] for i in range(0, len(creatives), n)]


def bulk_launch(
    client: NewsBreakClient,
    *,
    ad_account_id: str,
    campaign_mode: str,
    campaign_id: Optional[str],
    campaign_payload: Optional[Dict[str, Any]],
    ad_set_base: Dict[str, Any],
    creatives: List[Dict[str, Any]],
    grouping: str,
    group_size: int,
) -> Dict[str, Any]:
    """
    creatives: list of dicts with keys: file_bytes, filename, headline, body, landing_url, media_name (optional)
    campaign_payload: used when campaign_mode == 'new'
    ad_set_base: shared fields for create_ad_set (targeting, schedule, bid, budget, name prefix, etc.)
    """
    result: Dict[str, Any] = {
        "success": True,
        "campaign_id": None,
        "ad_sets": [],
        "errors": [],
    }

    cid = campaign_id
    if campaign_mode == "new" and campaign_payload:
        try:
            cr = client.create_campaign(campaign_payload)
            cid = _extract_id(cr, "id", "campaignId")
            result["campaign_raw"] = cr
        except Exception as e:
            result["success"] = False
            result["errors"].append(f"create_campaign: {e}")
            return result

    if not cid:
        result["success"] = False
        result["errors"].append("No campaign id")
        return result

    result["campaign_id"] = cid

    groups = group_creatives(creatives, grouping, group_size)
    base_name = ad_set_base.get("name_prefix", "Ad set")
    brand_name = ad_set_base.get("_brand_name", "Advertiser")
    cta = ad_set_base.get("_cta", "Learn More")

    def _guess_creative_type(filename: str) -> str:
        f = (filename or "").lower()
        if f.endswith(".gif"):
            return "GIF"
        if f.endswith((".mp4", ".mov", ".webm", ".m4v")):
            return "VIDEO"
        return "IMAGE"

    def _extract_asset_url(resp: Any) -> Optional[str]:
        if not isinstance(resp, dict):
            return None
        for k in ("assetUrl", "url", "cdnUrl"):
            v = resp.get(k)
            if isinstance(v, str) and v:
                return v
        data = resp.get("data")
        if isinstance(data, dict):
            for k in ("assetUrl", "url", "cdnUrl"):
                v = data.get(k)
                if isinstance(v, str) and v:
                    return v
        return None

    for gi, group in enumerate(groups):
        uploaded: List[Dict[str, Any]] = []
        for c in group:
            try:
                raw = client.upload_asset(
                    c.get("file_bytes") or b"",
                    c.get("filename") or "asset.bin",
                    ad_account_id,
                    media_name=c.get("media_name"),
                )
                asset_url = _extract_asset_url(raw)
                if not asset_url:
                    result["errors"].append(f"upload_asset missing assetUrl: {raw}")
                    continue
                uploaded.append(
                    {
                        "asset_url": asset_url,
                        "creative_type": _guess_creative_type(c.get("filename", "")),
                        "headline": c.get("headline", ""),
                        "body": c.get("body", ""),
                        "landing_url": c.get("landing_url", ""),
                    }
                )
            except Exception as e:
                result["errors"].append(f"upload: {e}")

        if not uploaded:
            continue

        ad_set_name = f"{base_name} {gi + 1}" if len(groups) > 1 else base_name
        ad_set_payload = {
            **{k: v for k, v in ad_set_base.items() if not k.startswith("_") and k != "name_prefix"},
            "name": ad_set_name,
            "campaignId": cid,
        }

        try:
            aset = client.create_ad_set(ad_set_payload)
            ad_set_id = _extract_id(aset, "id", "adSetId")
        except Exception as e:
            result["errors"].append(f"create_ad_set: {e}")
            continue

        ads_out = []
        for i, row in enumerate(uploaded):
            creative = {
                "type": row["creative_type"],
                "headline": row["headline"] or ad_set_name,
                "assetUrl": row["asset_url"],
                "description": row["body"] or row["headline"] or ad_set_name,
                "callToAction": cta,
                "brandName": brand_name,
            }
            if row["landing_url"]:
                creative["clickThroughUrl"] = row["landing_url"]
            ad_payload = {
                "adSetId": ad_set_id,
                "name": f"{ad_set_name} — Ad {i + 1}",
                "creative": creative,
            }
            try:
                ad_resp = client.create_ad(ad_payload)
                ads_out.append({"raw": ad_resp, "payload": ad_payload})
            except Exception as e:
                result["errors"].append(f"create_ad: {e}")

        result["ad_sets"].append({"ad_set_id": ad_set_id, "ads": ads_out})

    if result["errors"] and not result["ad_sets"]:
        result["success"] = False
    return result


def creative_from_upload(file_storage, headline: str, body: str, landing_url: str) -> Dict[str, Any]:
    """Build creative dict from Werkzeug FileStorage."""
    data = file_storage.read()
    return {
        "file_bytes": data,
        "filename": file_storage.filename or "upload.bin",
        "headline": headline,
        "body": body,
        "landing_url": landing_url,
        "media_name": file_storage.filename,
    }


def creative_from_bytes(
    file_bytes: bytes,
    filename: str,
    headline: str,
    body: str,
    landing_url: str,
) -> Dict[str, Any]:
    return {
        "file_bytes": file_bytes,
        "filename": filename,
        "headline": headline,
        "body": body,
        "landing_url": landing_url,
        "media_name": filename,
    }

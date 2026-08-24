"""
Platform adapter registry and factory.

Use ``get_adapter(platform, **credentials)`` from app / scheduler code
instead of instantiating NewsBreakClient / SmartNewsClient directly.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import AdPlatformAdapter, UnsupportedOperationError

PLATFORMS: List[str] = ["newsbreak", "smartnews", "outbrain", "mediago"]
DEFAULT_PLATFORM = "newsbreak"

PLATFORM_LABELS: Dict[str, str] = {
    "newsbreak": "NewsBreak",
    "smartnews": "SmartNews",
    "outbrain": "Outbrain / Teads",
    "mediago": "MediaGo",
}

PLATFORM_CURRENCIES: Dict[str, str] = {
    "newsbreak": "USD",
    "smartnews": "JPY",
    "outbrain": "USD",
    "mediago": "USD",
}


def normalize_platform(platform: Optional[str]) -> str:
    p = (platform or "").strip().lower()
    if p not in PLATFORMS:
        return DEFAULT_PLATFORM
    return p


def get_adapter(platform: str, **credentials: Any) -> AdPlatformAdapter:
    """Build an adapter for the named platform.

    Kwargs depend on the platform:
      - newsbreak: access_token (str, required), org_ids (list[str], optional)
      - smartnews: client_id (int/str, required), client_secret (str, required),
                   account_ids (list[str], optional). ``api_key`` is no longer
                   accepted — SmartNews Marketing API v3 uses OAuth
                   client_credentials, not a shared key.
      - outbrain:  access_token (OB-TOKEN-V1 str) OR username+password, plus
                   marketer_ids (list[str], optional).
      - mediago:   api_token (AM-issued Base64 token), optional auth_level
                   ("auto" | "client" | "account"), optional account_ids.
    """
    p = normalize_platform(platform)
    if p == "newsbreak":
        from newsbreak_api import NewsBreakClient

        from .newsbreak import NewsBreakAdapter

        token = credentials.get("access_token") or credentials.get("api_key")
        if not token:
            raise ValueError("newsbreak adapter requires access_token")
        client = NewsBreakClient(token)
        return NewsBreakAdapter(client, credentials.get("org_ids") or [])

    if p == "smartnews":
        from smartnews_api import SmartNewsClient

        from .smartnews import SmartNewsAdapter

        client_id = credentials.get("client_id")
        client_secret = credentials.get("client_secret")
        if not client_id or not client_secret:
            raise ValueError(
                "smartnews adapter requires client_id + client_secret "
                "(OAuth client_credentials)"
            )
        client = SmartNewsClient(client_id, client_secret)
        return SmartNewsAdapter(client, credentials.get("account_ids") or [])

    if p == "outbrain":
        from outbrain_api import OutbrainClient

        from .outbrain import OutbrainAdapter

        token = credentials.get("access_token") or credentials.get("api_key")
        username = credentials.get("username")
        password = credentials.get("password")
        if not token and not (username and password):
            raise ValueError(
                "outbrain adapter requires access_token (OB-TOKEN-V1) "
                "or username+password"
            )
        client = OutbrainClient(token, username=username, password=password)
        return OutbrainAdapter(
            client,
            credentials.get("marketer_ids")
            or credentials.get("account_ids")
            or [],
        )

    if p == "mediago":
        from mediago_api import MediaGoClient

        from .mediago import MediaGoAdapter

        api_token = (
            credentials.get("api_token")
            or credentials.get("access_token")
            or ""
        )
        if not api_token:
            raise ValueError("mediago adapter requires api_token")
        client = MediaGoClient(
            api_token,
            auth_level=credentials.get("auth_level") or "auto",
        )
        return MediaGoAdapter(
            client,
            credentials.get("account_ids")
            or credentials.get("org_ids")
            or [],
        )

    raise ValueError(f"Unknown platform {platform}")


__all__ = [
    "AdPlatformAdapter",
    "UnsupportedOperationError",
    "PLATFORMS",
    "DEFAULT_PLATFORM",
    "PLATFORM_LABELS",
    "PLATFORM_CURRENCIES",
    "normalize_platform",
    "get_adapter",
]

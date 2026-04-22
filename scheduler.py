"""Background APScheduler job: evaluate automation rules for every platform."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from apscheduler.schedulers.background import BackgroundScheduler

import storage
from platforms import PLATFORMS, get_adapter
from rules_engine import run_rules_for_account

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


def _env_credentials_for(platform: str) -> List[Dict[str, Any]]:
    """Env-configured credentials per platform (preferred — survives redeploys)."""
    if platform == "newsbreak":
        tok = (os.environ.get("NEWSBREAK_ACCESS_TOKEN") or "").strip()
        raw = os.environ.get("NEWSBREAK_DEFAULT_ORG_IDS", "")
        org_ids = [x.strip() for x in raw.split(",") if x.strip()]
        if tok and org_ids:
            return [{"uid": "env", "access_token": tok, "org_ids": org_ids}]
        return []
    if platform == "smartnews":
        key = (os.environ.get("SMARTNEWS_API_KEY") or "").strip()
        raw = os.environ.get("SMARTNEWS_DEFAULT_ACCOUNT_IDS", "")
        account_ids = [x.strip() for x in raw.split(",") if x.strip()]
        if key:
            return [{"uid": "env", "api_key": key, "account_ids": account_ids}]
        return []
    return []


def _file_credentials_for(platform: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for uid in storage.list_token_user_ids(platform=platform):
        tok = storage.load_token(uid, platform=platform)
        if not tok:
            continue
        entry: Dict[str, Any] = {"uid": uid}
        if platform == "newsbreak":
            entry["access_token"] = tok.get("access_token")
            entry["org_ids"] = tok.get("org_ids") or []
        else:  # smartnews
            entry["api_key"] = tok.get("access_token") or tok.get("api_key")
            entry["account_ids"] = tok.get("account_ids") or tok.get("org_ids") or []
        out.append(entry)
    return out


def _resolve_accounts(adapter) -> List[str]:
    """Return a flat list of ad account ids for an adapter."""
    try:
        accounts = adapter.get_accounts() or []
    except Exception:
        accounts = []
    ids: List[str] = []
    for a in accounts:
        aid = a.get("id") if isinstance(a, dict) else None
        if aid:
            ids.append(str(aid))
    return list(dict.fromkeys(ids))


def _run_for_platform(platform: str) -> None:
    creds = _env_credentials_for(platform) or _file_credentials_for(platform)
    for cred in creds:
        uid = cred["uid"]
        try:
            if platform == "newsbreak":
                adapter = get_adapter(
                    "newsbreak",
                    access_token=cred.get("access_token"),
                    org_ids=cred.get("org_ids") or [],
                )
            elif platform == "smartnews":
                adapter = get_adapter(
                    "smartnews",
                    api_key=cred.get("api_key"),
                    account_ids=cred.get("account_ids") or [],
                )
            else:
                continue
        except Exception as e:
            logger.warning("Scheduler skip %s/%s: %s", platform, uid, e)
            continue

        account_ids = _resolve_accounts(adapter)
        for aid in account_ids:
            rules = storage.load_rules(aid, platform=platform)
            if not any(r.get("enabled") for r in rules):
                continue

            def audit(entry: Dict[str, Any], _aid=aid, _uid=uid, _p=platform) -> None:
                storage.append_audit(
                    _aid,
                    {**entry, "user_id": _uid, "platform": _p},
                    platform=_p,
                )

            try:
                run_rules_for_account(
                    adapter,
                    aid,
                    rules,
                    audit=audit,
                )
            except Exception as e:
                logger.exception(
                    "Rules run failed platform=%s account=%s: %s", platform, aid, e
                )
                storage.append_audit(
                    aid,
                    {
                        "error": str(e),
                        "user_id": uid,
                        "platform": platform,
                        "scope": "scheduler",
                    },
                    platform=platform,
                )


def run_scheduled_rules() -> None:
    for platform in PLATFORMS:
        try:
            _run_for_platform(platform)
        except Exception as e:
            logger.exception("Scheduler failed for platform %s: %s", platform, e)


def start_scheduler(interval_minutes: int = 15) -> BackgroundScheduler:
    global _scheduler
    if _scheduler is not None:
        return _scheduler
    sched = BackgroundScheduler(daemon=True)
    sched.add_job(
        run_scheduled_rules,
        "interval",
        minutes=interval_minutes,
        id="newsbreak_rules",
        replace_existing=True,
    )
    sched.start()
    _scheduler = sched
    logger.info("Scheduler started: rules every %s min (all platforms)", interval_minutes)
    return sched


def shutdown_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None

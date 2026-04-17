"""Background APScheduler job: evaluate automation rules."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from apscheduler.schedulers.background import BackgroundScheduler

import storage
from newsbreak_api import NewsBreakClient
from rules_engine import run_rules_for_account

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


def _flatten_ad_accounts(api_response: Any) -> List[str]:
    """Extract ad account ids from getGroupsByOrgIds response."""
    ids: List[str] = []
    if not isinstance(api_response, dict):
        return ids
    data = api_response.get("data") or api_response.get("result") or api_response
    groups = data if isinstance(data, list) else data.get("groups") or data.get("list") or []
    if isinstance(groups, dict):
        groups = [groups]
    if not isinstance(groups, list):
        return ids
    for g in groups:
        if not isinstance(g, dict):
            continue
        accounts = g.get("adAccounts") or g.get("ad_accounts") or []
        if isinstance(accounts, dict):
            accounts = [accounts]
        for a in accounts or []:
            if isinstance(a, dict):
                aid = a.get("id") or a.get("adAccountId")
                if aid:
                    ids.append(str(aid))
    return list(dict.fromkeys(ids))


def _env_credentials() -> List[Dict[str, Any]]:
    """Env-configured credentials (preferred — survives redeploys)."""
    tok = (os.environ.get("NEWSBREAK_ACCESS_TOKEN") or "").strip()
    raw = os.environ.get("NEWSBREAK_DEFAULT_ORG_IDS", "")
    org_ids = [x.strip() for x in raw.split(",") if x.strip()]
    if tok and org_ids:
        return [{"uid": "env", "access_token": tok, "org_ids": org_ids}]
    return []


def _file_credentials() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for uid in storage.list_token_user_ids():
        tok = storage.load_token(uid)
        if not tok:
            continue
        out.append({"uid": uid, "access_token": tok.get("access_token"), "org_ids": tok.get("org_ids") or []})
    return out


def run_scheduled_rules() -> None:
    creds = _env_credentials() or _file_credentials()
    for cred in creds:
        uid = cred["uid"]
        token = cred.get("access_token")
        org_ids = cred.get("org_ids") or []
        if not token or not org_ids:
            logger.warning("Scheduler skip user %s: missing token or org_ids", uid)
            continue
        try:
            client = NewsBreakClient(token)
            acc_resp = client.get_ad_accounts(org_ids)
            account_ids = _flatten_ad_accounts(acc_resp)
        except Exception as e:
            logger.exception("Scheduler API error for user %s: %s", uid, e)
            continue

        for aid in account_ids:
            rules = storage.load_rules(aid)
            if not any(r.get("enabled") for r in rules):
                continue

            def audit(entry: Dict[str, Any]) -> None:
                storage.append_audit(aid, {**entry, "user_id": uid})

            try:
                run_rules_for_account(
                    client,
                    aid,
                    rules,
                    audit=audit,
                )
            except Exception as e:
                logger.exception("Rules run failed account %s: %s", aid, e)
                storage.append_audit(
                    aid,
                    {"error": str(e), "user_id": uid, "scope": "scheduler"},
                )


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
    logger.info("Scheduler started: rules every %s min", interval_minutes)
    return sched


def shutdown_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None

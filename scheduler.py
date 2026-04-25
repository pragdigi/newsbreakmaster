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
        client_id = (os.environ.get("SMARTNEWS_CLIENT_ID") or "").strip()
        client_secret = (
            os.environ.get("SMARTNEWS_CLIENT_SECRET")
            or os.environ.get("SMARTNEWS_API_KEY")  # legacy fallback
            or ""
        ).strip()
        raw = os.environ.get("SMARTNEWS_DEFAULT_ACCOUNT_IDS", "")
        account_ids = [x.strip() for x in raw.split(",") if x.strip()]
        if client_id and client_secret:
            return [
                {
                    "uid": "env",
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "account_ids": account_ids,
                }
            ]
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
            entry["client_id"] = tok.get("client_id")
            entry["client_secret"] = tok.get("client_secret") or tok.get("access_token")
            entry["account_ids"] = tok.get("account_ids") or tok.get("org_ids") or []
            if not entry["client_id"] or not entry["client_secret"]:
                continue
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
                    client_id=cred.get("client_id"),
                    client_secret=cred.get("client_secret"),
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


def _adapters_for_platform(platform: str):
    """Yield (adapter, uid) pairs for every configured account on a platform."""
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
                    client_id=cred.get("client_id"),
                    client_secret=cred.get("client_secret"),
                    account_ids=cred.get("account_ids") or [],
                )
            else:
                continue
        except Exception as e:
            logger.warning("AI studio scheduler skip %s/%s: %s", platform, uid, e)
            continue
        yield adapter, uid


def run_ad_studio_nightly(*, mode: str = "full") -> None:
    """AI Ad Studio maintenance: refresh winners + discover styles + reconcile lifecycle.

    Runs across every platform with configured credentials. Each hook is
    isolated behind try/except so a single failure doesn't kill the others.

    ``mode``:
      * ``"full"``  — runs the full nightly pass (winners + discover + lifecycle).
      * ``"scout"`` — runs only the discovery pass (cheap, ~15s/offer). Used by
        the every-6h scout job so the AI keeps scanning GetHookd / brainstorming
        new style angles per saved offer in the background, without re-pulling
        the full winners report on each tick.
    """
    if mode == "scout":
        for platform in PLATFORMS:
            for adapter, uid in _adapters_for_platform(platform):
                try:
                    from ai_studio.research import discover_all

                    discovered = discover_all(
                        platform=platform,
                        scan_all_offers=True,
                        keywords_per_offer=int(
                            os.environ.get("AD_STUDIO_SCOUT_KEYWORDS_PER_OFFER", "3")
                        ),
                        gethookd_limit_per_offer=int(
                            os.environ.get("AD_STUDIO_SCOUT_GETHOOKD_LIMIT", "20")
                        ),
                        brainstorm_count=int(
                            os.environ.get("AD_STUDIO_SCOUT_BRAINSTORM_COUNT", "2")
                        ),
                    )
                    logger.info(
                        "ai_studio.scout platform=%s uid=%s modes=%s candidates=%s",
                        platform,
                        uid,
                        list(discovered.keys()),
                        sum(len(v or []) for v in discovered.values()),
                    )
                except Exception as e:
                    logger.exception(
                        "ai_studio.scout failed platform=%s uid=%s: %s",
                        platform,
                        uid,
                        e,
                    )
        return

    for platform in PLATFORMS:
        for adapter, uid in _adapters_for_platform(platform):
            # 1) Winners refresher — produces winners.json and flips
            #    becomes_winner=true on any matching generation row.
            try:
                from ai_studio.winners import refresh_winners

                summary = refresh_winners(adapter, platform=platform)
                logger.info(
                    "ai_studio.winners platform=%s uid=%s added=%s updated=%s demoted=%s linked=%s",
                    platform,
                    uid,
                    summary.get("added"),
                    summary.get("updated"),
                    summary.get("demoted"),
                    summary.get("generations_linked"),
                )
            except Exception as e:
                logger.exception(
                    "ai_studio.winners failed platform=%s uid=%s: %s", platform, uid, e
                )

            # 2) Research discovery — scan catalog for candidate styles
            #    across every available mode. ``scan_all_offers=True`` makes
            #    the nightly pass derive search keywords per saved offer and
            #    call GetHookd / brainstorm for each, instead of running
            #    with an empty query.
            try:
                from ai_studio.research import discover_all

                discovered = discover_all(
                    platform=platform,
                    scan_all_offers=True,
                    keywords_per_offer=int(os.environ.get("AD_STUDIO_NIGHTLY_KEYWORDS_PER_OFFER", "5")),
                    gethookd_limit_per_offer=int(os.environ.get("AD_STUDIO_NIGHTLY_GETHOOKD_LIMIT", "40")),
                    brainstorm_count=int(os.environ.get("AD_STUDIO_NIGHTLY_BRAINSTORM_COUNT", "3")),
                )
                logger.info(
                    "ai_studio.research platform=%s uid=%s modes=%s candidates=%s",
                    platform,
                    uid,
                    list(discovered.keys()),
                    sum(len(v or []) for v in discovered.values()),
                )
            except Exception as e:
                logger.exception(
                    "ai_studio.research failed platform=%s uid=%s: %s", platform, uid, e
                )

            # 3) Lifecycle reconciliation — promote/archive/demote-flag.
            try:
                from ai_studio.research.lifecycle import reconcile

                report = reconcile(platform=platform)
                logger.info(
                    "ai_studio.lifecycle platform=%s uid=%s promoted=%s archived=%s flagged=%s",
                    platform,
                    uid,
                    len(report.get("promoted", [])),
                    len(report.get("archived", [])),
                    len(report.get("demotion_flagged", [])),
                )
            except Exception as e:
                logger.exception(
                    "ai_studio.lifecycle failed platform=%s uid=%s: %s", platform, uid, e
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
    # AI Ad Studio nightly pass at 05:30 UTC — well after most US/JP spend
    # has settled into the previous 24h window.
    sched.add_job(
        run_ad_studio_nightly,
        "cron",
        hour=int(os.environ.get("AD_STUDIO_NIGHTLY_HOUR", "5")),
        minute=int(os.environ.get("AD_STUDIO_NIGHTLY_MINUTE", "30")),
        id="ai_studio_nightly",
        replace_existing=True,
    )
    # Every-N-hour scout pass — keeps GetHookd + brainstorm sweeping for new
    # ad concepts per saved offer in the background. Defaults to every 6h
    # which matches the user's original ask. Set AD_STUDIO_SCOUT_HOURS=0 to
    # disable.
    scout_hours = int(os.environ.get("AD_STUDIO_SCOUT_HOURS", "6"))
    if scout_hours > 0:
        sched.add_job(
            lambda: run_ad_studio_nightly(mode="scout"),
            "interval",
            hours=scout_hours,
            id="ai_studio_scout",
            replace_existing=True,
        )
    sched.start()
    _scheduler = sched
    logger.info(
        "Scheduler started: rules every %s min + AI studio nightly + scout every %sh (all platforms)",
        interval_minutes,
        scout_hours if scout_hours > 0 else "off",
    )
    return sched


def shutdown_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None

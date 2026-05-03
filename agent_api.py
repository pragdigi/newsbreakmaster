"""Managed-agent HTTP surface.

Exposes a narrow, signed-request-only API that an external autonomous
agent (e.g. a Claude managed agent running on
``platform.claude.com/workspaces/default/agent-quickstart``) can call to:

  * list offers, winners, and candidate styles
  * add new style candidates
  * kick off a discovery run
  * schedule a generation batch
  * read recent generations (with prompts + base64 images)

Every request is authenticated with an HMAC-SHA256 signature over:

    method + "\n" + path + "\n" + X-Agent-Timestamp + "\n" + sha256(body)

using a shared secret configured via the ``AGENT_SHARED_SECRET`` env
variable. Timestamps older than ``AGENT_MAX_CLOCK_SKEW`` seconds (default
300) are rejected. Requests must additionally include the header
``X-Agent-Key`` which matches ``AGENT_PUBLIC_KEY`` (optional extra
identifier so you can rotate secrets without rotating keys).

When no secret is configured the blueprint returns 503 so we never
accidentally expose an unsigned write surface.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from flask import Blueprint, Response, jsonify, request, send_from_directory

import storage
from platforms import DEFAULT_PLATFORM, PLATFORMS, normalize_platform

logger = logging.getLogger(__name__)

bp = Blueprint("agent_api", __name__, url_prefix="/api/agent")

AGENT_SHARED_SECRET = os.environ.get("AGENT_SHARED_SECRET", "").strip()
AGENT_PUBLIC_KEY = os.environ.get("AGENT_PUBLIC_KEY", "").strip() or "default"
AGENT_MAX_CLOCK_SKEW = int(os.environ.get("AGENT_MAX_CLOCK_SKEW", "300"))


# ---------------------------------------------------------------------------
# HMAC authentication
# ---------------------------------------------------------------------------


def _canonical_message(method: str, path: str, ts: str, body: bytes) -> bytes:
    digest = hashlib.sha256(body or b"").hexdigest()
    msg = f"{method.upper()}\n{path}\n{ts}\n{digest}"
    return msg.encode("utf-8")


def _expected_signature(secret: str, msg: bytes) -> str:
    return hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def _unauthorized(reason: str) -> Response:
    logger.warning("agent_api auth rejected: %s", reason)
    resp = jsonify({"error": "unauthorized", "reason": reason})
    resp.status_code = 401
    return resp


def _require_signed_request() -> Optional[Response]:
    if not AGENT_SHARED_SECRET:
        resp = jsonify(
            {
                "error": "agent_api disabled",
                "hint": "Set AGENT_SHARED_SECRET and (optionally) AGENT_PUBLIC_KEY in the env.",
            }
        )
        resp.status_code = 503
        return resp

    key = request.headers.get("X-Agent-Key", "").strip()
    if key and AGENT_PUBLIC_KEY and key != AGENT_PUBLIC_KEY:
        return _unauthorized("unknown key")

    ts_raw = request.headers.get("X-Agent-Timestamp", "").strip()
    sig_raw = request.headers.get("X-Agent-Signature", "").strip()
    if not ts_raw or not sig_raw:
        return _unauthorized("missing timestamp or signature")

    try:
        ts_int = int(ts_raw)
    except ValueError:
        return _unauthorized("timestamp not an integer")
    now = int(time.time())
    if abs(now - ts_int) > AGENT_MAX_CLOCK_SKEW:
        return _unauthorized(
            f"timestamp drift {now - ts_int}s exceeds {AGENT_MAX_CLOCK_SKEW}s"
        )

    body = request.get_data(cache=True) or b""
    expected = _expected_signature(
        AGENT_SHARED_SECRET,
        _canonical_message(request.method, request.path, ts_raw, body),
    )
    if not hmac.compare_digest(expected, sig_raw):
        return _unauthorized("bad signature")
    return None


def _guard():
    """Decorator helper — call at the top of each handler."""
    err = _require_signed_request()
    if err is not None:
        return err
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _platform_param(default: Optional[str] = None) -> str:
    raw = (request.args.get("platform") or "").strip().lower()
    if not raw:
        data = request.get_json(silent=True) or {}
        raw = str(data.get("platform") or default or DEFAULT_PLATFORM).strip().lower()
    return normalize_platform(raw) if raw else DEFAULT_PLATFORM


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Introspection / health
# ---------------------------------------------------------------------------


@bp.route("/health", methods=["GET"])
def agent_health():
    err = _guard()
    if err is not None:
        return err
    return jsonify(
        {
            "ok": True,
            "ts": _iso_now(),
            "platforms": list(PLATFORMS),
            "default_platform": DEFAULT_PLATFORM,
        }
    )


@bp.route("/authcheck", methods=["POST"])
def agent_authcheck():
    """Round-trip test for agents wiring up signing. Also works as a warm-up."""
    err = _guard()
    if err is not None:
        return err
    body = request.get_json(silent=True) or {}
    return jsonify({"ok": True, "echo": body, "ts": _iso_now()})


# ---------------------------------------------------------------------------
# Read surface — offers / winners / candidates / generations
# ---------------------------------------------------------------------------


@bp.route("/offers", methods=["GET"])
def agent_list_offers():
    err = _guard()
    if err is not None:
        return err
    platform_filter = (request.args.get("platform") or "").strip().lower()
    include = (
        [normalize_platform(platform_filter)]
        if platform_filter
        else list(PLATFORMS)
    )
    out: List[Dict[str, Any]] = []
    for p in include:
        try:
            rows = storage.list_offers(platform=p) or []
        except Exception as exc:  # noqa: BLE001
            logger.warning("agent_api: list_offers(%s) failed: %s", p, exc)
            continue
        for r in rows:
            r = dict(r)
            r.setdefault("platform", p)
            out.append(r)
    return jsonify({"offers": out, "count": len(out)})


@bp.route("/winners", methods=["GET"])
def agent_list_winners():
    err = _guard()
    if err is not None:
        return err
    platform_filter = (request.args.get("platform") or "").strip().lower()
    try:
        if platform_filter:
            rows = storage.list_winners(platform=normalize_platform(platform_filter))
        else:
            rows = storage.list_all_winners()
    except Exception as exc:  # noqa: BLE001
        logger.exception("agent_api: list_winners failed: %s", exc)
        return jsonify({"error": str(exc)}), 500
    limit = request.args.get("limit", type=int) or 200
    return jsonify({"winners": list(rows)[:limit], "count": len(rows)})


@bp.route("/candidates", methods=["GET"])
def agent_list_candidates():
    err = _guard()
    if err is not None:
        return err
    platform = _platform_param()
    try:
        rows = storage.list_style_candidates(platform=platform) or []
    except Exception as exc:  # noqa: BLE001
        logger.exception("agent_api: list_style_candidates failed: %s", exc)
        return jsonify({"error": str(exc)}), 500
    return jsonify({"platform": platform, "candidates": rows, "count": len(rows)})


@bp.route("/candidates", methods=["POST"])
def agent_add_candidate():
    err = _guard()
    if err is not None:
        return err
    data = request.get_json(silent=True) or {}
    platform = _platform_param()
    name = str(data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400
    style_id = str(data.get("style_id") or name).strip()
    prompt_template = str(data.get("prompt_template") or "").strip()
    if not prompt_template:
        return jsonify({"error": "prompt_template required"}), 400

    now = _iso_now()
    candidate: Dict[str, Any] = {
        "style_id": style_id,
        "name": name,
        "description": str(data.get("description") or "").strip(),
        "visual_cues": [str(v) for v in (data.get("visual_cues") or [])],
        "prompt_template": prompt_template,
        "reference_image_paths": [str(v) for v in (data.get("reference_image_paths") or [])],
        "source": str(data.get("source") or "managed_agent"),
        "source_meta": data.get("source_meta") or {},
        "status": str(data.get("status") or "candidate"),
        "trials": 0,
        "wins": 0,
        "impressions": 0,
        "spend": 0.0,
        "conversions": 0,
        "cpa": None,
        "ctr": None,
        "thompson_alpha": 1,
        "thompson_beta": 1,
        "created_at": now,
        "last_trial_at": None,
        "created_by": "agent",
    }
    try:
        saved = storage.upsert_style_candidate(candidate, platform=platform)
    except Exception as exc:  # noqa: BLE001
        logger.exception("agent_api: upsert_style_candidate failed: %s", exc)
        return jsonify({"error": str(exc)}), 500
    return jsonify({"ok": True, "candidate": saved})


@bp.route("/generations", methods=["GET"])
def agent_list_generations():
    err = _guard()
    if err is not None:
        return err
    platform = _platform_param()
    limit = request.args.get("limit", type=int) or 20
    try:
        rows = storage.list_generations(platform=platform, limit=limit) or []
    except Exception as exc:  # noqa: BLE001
        logger.exception("agent_api: list_generations failed: %s", exc)
        return jsonify({"error": str(exc)}), 500
    return jsonify({"platform": platform, "generations": rows, "count": len(rows)})


# ---------------------------------------------------------------------------
# Prebuilt-ad library — cross-app read surface so a sister app (the
# metamaster Meta launcher) can pick the same 1:1 / 16:9 ads we've
# already paid the image-API cost to render and launch them on Meta too.
# Read-only; consumption stays in newsbreakmaster's own UI.
# ---------------------------------------------------------------------------


def _library_filename_safe(name: str) -> Optional[str]:
    """Reject any filename component that would let a caller escape the
    library_images directory (path traversal / hidden files)."""
    if not name:
        return None
    base = os.path.basename(str(name))
    if base in ("", ".", "..") or base.startswith(".") or "/" in base or "\\" in base:
        return None
    return base


@bp.route("/library", methods=["GET"])
def agent_list_library():
    """List prebuilt library items.

    Query parameters:
      - ``platform``         (default: every known platform)
      - ``offer_id``         (optional filter)
      - ``aspect``           (optional filter, e.g. ``1:1``, ``16:9``)
      - ``include_consumed`` (default false — only show items still
                              available to launch with)
      - ``limit``            (default 200, max 1000)

    Each row carries a ``download_url`` that's the unsigned PATH (not a
    full URL) the caller hits with valid agent headers to fetch the
    image bytes. Returning a path keeps the caller free to map host /
    scheme however they want (proxy, CDN, local cache, etc.).
    """
    err = _guard()
    if err is not None:
        return err

    platform_filter = (request.args.get("platform") or "").strip().lower()
    targets = (
        [normalize_platform(platform_filter)]
        if platform_filter
        else list(PLATFORMS)
    )
    offer_id = (request.args.get("offer_id") or "").strip() or None
    aspect_filter = (request.args.get("aspect") or "").strip() or None
    include_consumed = (request.args.get("include_consumed") or "").lower() in (
        "1",
        "true",
        "yes",
    )
    limit = max(1, min(int(request.args.get("limit", type=int) or 200), 1000))

    out: List[Dict[str, Any]] = []
    for p in targets:
        try:
            rows = storage.list_library_items(
                platform=p,
                offer_id=offer_id,
                include_consumed=include_consumed,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("agent_api: list_library_items(%s) failed: %s", p, exc)
            continue
        for row in rows:
            if aspect_filter and str(row.get("aspect") or "") != aspect_filter:
                continue
            filename = _library_filename_safe(row.get("filename"))
            if not filename:
                # Skip stale rows (e.g. the rendering failed mid-flight
                # before the row was patched with a filename).
                continue
            item = dict(row)
            item["platform"] = p
            item["download_url"] = (
                f"/api/agent/library/image/{p}/{filename}"
            )
            out.append(item)

    out.sort(key=lambda r: str(r.get("created_at") or ""), reverse=True)
    return jsonify(
        {
            "items": out[:limit],
            "count": min(len(out), limit),
            "total": len(out),
            "platforms": targets,
        }
    )


@bp.route("/library/image/<platform>/<path:filename>", methods=["GET"])
def agent_library_image(platform: str, filename: str):
    """Serve raw library image bytes. HMAC-signed like the rest of the
    agent surface so we don't accidentally expose a public bucket."""
    err = _guard()
    if err is not None:
        return err
    safe_platform = normalize_platform(platform)
    safe_name = _library_filename_safe(filename)
    if not safe_name:
        return jsonify({"error": "invalid filename"}), 400
    directory = storage.library_image_dir(safe_platform)
    full = os.path.join(directory, safe_name)
    if not os.path.exists(full):
        return jsonify({"error": "not found"}), 404
    return send_from_directory(directory, safe_name)


# ---------------------------------------------------------------------------
# Write surface — discovery + scheduled generation
# ---------------------------------------------------------------------------


@bp.route("/discover", methods=["POST"])
def agent_discover():
    """Kick off a research discovery run synchronously.

    Body:
      {
        "platform": "newsbreak" | "smartnews",
        "mode": "cluster_winners" | "gethookd" | "brainstorm" | "all",
        "offer_id": "<optional>",
        "keywords": ["..."]  # only for gethookd
      }
    """
    err = _guard()
    if err is not None:
        return err
    data = request.get_json(silent=True) or {}
    platform = _platform_param()
    mode = str(data.get("mode") or "all").strip()
    offer_id = str(data.get("offer_id") or "").strip() or None
    keywords = data.get("keywords") or []

    from ai_studio.research import discover as _disc

    try:
        if mode == "cluster_winners":
            out = _disc.discover_from_winners(platform)
        elif mode == "gethookd":
            out = _disc.discover_from_gethookd(
                platform=platform,
                keywords=[str(k) for k in keywords],
                filters=data.get("filters") or None,
                limit=int(data.get("limit") or 40),
            )
        elif mode == "brainstorm":
            if not offer_id:
                return jsonify({"error": "offer_id required for brainstorm"}), 400
            offer = next(
                (o for o in storage.list_offers(platform=platform)
                 if str(o.get("id")) == offer_id),
                None,
            )
            if not offer:
                return jsonify({"error": "offer not found"}), 404
            out = _disc.discover_from_brainstorm(
                offer,
                platform=platform,
                count=int(data.get("count") or 5),
            )
        elif mode == "all":
            out = _disc.discover_all(
                platform=platform,
                offer_id=offer_id,
                gethookd_keywords=keywords or None,
                scan_all_offers=bool(data.get("scan_all_offers") or False),
            )
        else:
            return jsonify({"error": f"unknown mode {mode!r}"}), 400
    except Exception as exc:  # noqa: BLE001
        logger.exception("agent_api: discover failed")
        return jsonify({"error": str(exc)}), 500
    return jsonify({"ok": True, "mode": mode, "result": out})


@bp.route("/schedule-generation", methods=["POST"])
def agent_schedule_generation():
    """Enqueue an AI Studio generation batch for later execution.

    Writes to ``agent_queue.jsonl`` under the platform's storage dir. A
    future cron (or /api/agent/drain-queue) can pick these up. Useful so
    the managed agent can schedule work without blocking on the full
    ~60s image-generation cycle.
    """
    err = _guard()
    if err is not None:
        return err
    data = request.get_json(silent=True) or {}
    platform = _platform_param()
    offer_id = str(data.get("offer_id") or "").strip()
    if not offer_id:
        return jsonify({"error": "offer_id required"}), 400

    job = {
        "job_id": str(uuid.uuid4()),
        "kind": "generate",
        "platform": platform,
        "offer_id": offer_id,
        "count": int(data.get("count") or 10),
        "model_image": str(data.get("model_image") or "nano-banana-2"),
        "model_analyzer": str(data.get("model_analyzer") or "") or None,
        "research_ratio": data.get("research_ratio"),
        "style_mix": data.get("style_mix") or None,
        "note": str(data.get("note") or "").strip(),
        "enqueued_at": _iso_now(),
        "status": "queued",
    }
    try:
        storage.append_agent_job(job, platform=platform)
    except Exception as exc:  # noqa: BLE001
        logger.exception("agent_api: append_agent_job failed")
        return jsonify({"error": str(exc)}), 500
    return jsonify({"ok": True, "job": job})


@bp.route("/run-scout", methods=["POST"])
def agent_run_scout():
    """Run one in-process scout pass (discover_all per saved offer).

    Same logic the every-6h scheduler uses, but on-demand. Safe to call
    from a Render cron, an external uptime monitor, or a Claude managed
    agent — the request must still carry valid HMAC headers.
    """
    err = _guard()
    if err is not None:
        return err
    try:
        from scheduler import run_ad_studio_nightly

        run_ad_studio_nightly(mode="scout")
    except Exception as exc:  # noqa: BLE001
        logger.exception("agent_api: run-scout failed")
        return jsonify({"error": str(exc)}), 500
    return jsonify({"ok": True, "ts": _iso_now(), "mode": "scout"})


@bp.route("/run-public-scout", methods=["POST"])
def agent_run_public_scout():
    """Run one in-process public-libraries scout (Meta Ad Library + TikTok).

    No GetHookd dependency — pulls competitor ads directly from public
    libraries and clusters them into style candidates. Body (optional):

        {
          "platform": "newsbreak" | "smartnews",
          "offer_id": "...",        # restrict to one offer
          "scan_all_offers": true,  # default true
          "country": "US",
          "sources": ["meta", "tiktok"],
          "keywords_per_offer": 4,
          "limit_per_query": 20
        }
    """
    err = _guard()
    if err is not None:
        return err
    data = request.get_json(silent=True) or {}
    platform_filter = (data.get("platform") or request.args.get("platform") or "").strip().lower()
    targets = [normalize_platform(platform_filter)] if platform_filter else list(PLATFORMS)
    offer_id = str(data.get("offer_id") or "").strip() or None
    scan_all = bool(data.get("scan_all_offers", True))
    country = str(data.get("country") or "US").upper()
    sources = data.get("sources") or ["meta", "tiktok"]
    keywords_per_offer = int(data.get("keywords_per_offer") or 4)
    limit_per_query = int(data.get("limit_per_query") or 20)

    summary: Dict[str, Any] = {}
    try:
        from ai_studio.research import discover_public as _public

        for p in targets:
            try:
                discovered = _public.discover_all_public(
                    platform=p,
                    offer_id=offer_id,
                    scan_all_offers=scan_all,
                    keywords_per_offer=keywords_per_offer,
                    limit_per_query=limit_per_query,
                    country=country,
                    sources=sources,
                )
                summary[p] = {
                    "candidates": sum(len(v or []) for v in discovered.values()),
                    "modes": list(discovered.keys()),
                }
            except Exception as exc:  # noqa: BLE001
                logger.exception("agent_api: run-public-scout failed for %s", p)
                summary[p] = {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        logger.exception("agent_api: run-public-scout import failure")
        return jsonify({"error": str(exc)}), 500
    return jsonify({"ok": True, "ts": _iso_now(), "platforms": summary})


@bp.route("/run-scholar", methods=["POST"])
def agent_run_scholar():
    """Run the Copywriting Scholar against one or all offers.

    Body:
      {
        "platform": "newsbreak" | "smartnews",
        "offer_id": "...",        # restrict to one offer
        "scan_all_offers": true,  # default true if no offer_id
        "count_per_offer": 3,
        "lens_id": "schwartz_awareness"  # optional override
      }
    """
    err = _guard()
    if err is not None:
        return err
    data = request.get_json(silent=True) or {}
    platform_filter = (data.get("platform") or request.args.get("platform") or "").strip().lower()
    targets = [normalize_platform(platform_filter)] if platform_filter else list(PLATFORMS)
    offer_id = str(data.get("offer_id") or "").strip() or None
    scan_all = bool(data.get("scan_all_offers", offer_id is None))
    count_per_offer = int(data.get("count_per_offer") or 3)
    lens_id = (data.get("lens_id") or "").strip() or None

    summary: Dict[str, Any] = {}
    try:
        from ai_studio.research import scholar as _scholar

        for p in targets:
            try:
                if lens_id and offer_id:
                    offer = next(
                        (o for o in storage.list_offers(platform=p)
                         if str(o.get("id")) == str(offer_id)),
                        None,
                    )
                    if not offer:
                        summary[p] = {"error": f"offer {offer_id} not found"}
                        continue
                    emitted, lens = _scholar.study_offer(
                        offer,
                        platform=p,
                        count=count_per_offer,
                        lens_id=lens_id,
                    )
                    summary[p] = {"candidates": len(emitted), "lens": lens.id}
                else:
                    discovered = _scholar.study_all(
                        platform=p,
                        offer_id=offer_id,
                        scan_all_offers=scan_all,
                        count_per_offer=count_per_offer,
                    )
                    summary[p] = {
                        "candidates": sum(len(v or []) for v in discovered.values()),
                    }
            except Exception as exc:  # noqa: BLE001
                logger.exception("agent_api: run-scholar failed for %s", p)
                summary[p] = {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        logger.exception("agent_api: run-scholar import failure")
        return jsonify({"error": str(exc)}), 500
    return jsonify({"ok": True, "ts": _iso_now(), "platforms": summary})


@bp.route("/lenses", methods=["GET"])
def agent_list_lenses():
    """List the Copywriting Scholar's available framework lenses."""
    err = _guard()
    if err is not None:
        return err
    from ai_studio.research import scholar as _scholar

    return jsonify(
        {
            "lenses": [
                {
                    "id": l.id,
                    "name": l.name,
                    "kind": l.kind,
                    "scaffold": l.scaffold,
                }
                for l in _scholar.LENSES
            ]
        }
    )


@bp.route("/queue", methods=["GET"])
def agent_list_queue():
    err = _guard()
    if err is not None:
        return err
    platform = _platform_param()
    status = (request.args.get("status") or "").strip().lower() or None
    try:
        rows = storage.list_agent_jobs(platform=platform, status=status)
    except Exception as exc:  # noqa: BLE001
        logger.exception("agent_api: list_agent_jobs failed")
        return jsonify({"error": str(exc)}), 500
    return jsonify({"platform": platform, "jobs": rows, "count": len(rows)})


@bp.route("/drain-queue", methods=["POST"])
def agent_drain_queue():
    """Execute queued jobs in-process. Idempotent: a job already marked
    ``status=done`` is skipped. ``max_jobs`` caps per-call work so this
    endpoint can be called on a cron without stalling workers.
    """
    err = _guard()
    if err is not None:
        return err
    data = request.get_json(silent=True) or {}
    platform = _platform_param()
    max_jobs = int(data.get("max_jobs") or 3)

    try:
        queued = storage.list_agent_jobs(platform=platform, status="queued")
    except Exception as exc:  # noqa: BLE001
        logger.exception("agent_api: list_agent_jobs failed")
        return jsonify({"error": str(exc)}), 500

    from ai_studio import pipeline as _pipeline

    executed: List[Dict[str, Any]] = []
    for job in queued[:max_jobs]:
        job_id = job.get("job_id")
        try:
            storage.update_agent_job(
                job_id, {"status": "running", "started_at": _iso_now()}, platform=platform
            )
            if job.get("kind") == "generate":
                try:
                    ratio = job.get("research_ratio")
                    ratio = float(ratio) if ratio is not None else None
                except (TypeError, ValueError):
                    ratio = None
                result = _pipeline.generate_ads(
                    job.get("offer_id"),
                    platform=platform,
                    count=int(job.get("count") or 10),
                    model_image=job.get("model_image") or "nano-banana-2",
                    model_analyzer=job.get("model_analyzer") or None,
                    style_mix=job.get("style_mix") or None,
                    research_ratio=ratio,
                )
                storage.update_agent_job(
                    job_id,
                    {
                        "status": "done",
                        "finished_at": _iso_now(),
                        "gen_id": result.get("gen_id"),
                    },
                    platform=platform,
                )
                executed.append(
                    {"job_id": job_id, "status": "done", "gen_id": result.get("gen_id")}
                )
            else:
                storage.update_agent_job(
                    job_id,
                    {"status": "skipped", "finished_at": _iso_now(), "reason": "unknown kind"},
                    platform=platform,
                )
                executed.append({"job_id": job_id, "status": "skipped"})
        except Exception as exc:  # noqa: BLE001
            logger.exception("agent_api: drain job %s failed", job_id)
            storage.update_agent_job(
                job_id,
                {"status": "error", "finished_at": _iso_now(), "error": str(exc)},
                platform=platform,
            )
            executed.append({"job_id": job_id, "status": "error", "error": str(exc)})
    return jsonify({"ok": True, "platform": platform, "executed": executed})


# ---------------------------------------------------------------------------
# Bootstrap utilities (signed)
# ---------------------------------------------------------------------------


@bp.route("/sign-example", methods=["GET"])
def agent_sign_example():
    """Return a worked signing example so a human wiring an external agent
    can copy-paste the exact headers. REQUIRES a valid current signature
    to prevent leaking the secret (the body is safe because it's empty)."""
    err = _guard()
    if err is not None:
        return err
    ts = str(int(time.time()))
    sample_method = "POST"
    sample_path = "/api/agent/candidates"
    sample_body = b'{"name":"example","prompt_template":"..."}'
    msg = _canonical_message(sample_method, sample_path, ts, sample_body)
    signature = _expected_signature(AGENT_SHARED_SECRET, msg)
    return jsonify(
        {
            "headers": {
                "X-Agent-Key": AGENT_PUBLIC_KEY,
                "X-Agent-Timestamp": ts,
                "X-Agent-Signature": signature,
                "Content-Type": "application/json",
            },
            "canonical_message": msg.decode("utf-8"),
            "note": (
                "Compute: HMAC-SHA256(AGENT_SHARED_SECRET, "
                "'{method}\\n{path}\\n{ts}\\n{sha256(body_bytes)}').hex() "
                "where body_bytes is the exact bytes you POST."
            ),
        }
    )


# ---------------------------------------------------------------------------
# Helpers for signing from Python (consumed by tests + the Claude agent)
# ---------------------------------------------------------------------------


def build_agent_headers(
    *,
    method: str,
    path: str,
    body: bytes = b"",
    secret: Optional[str] = None,
    public_key: Optional[str] = None,
    timestamp: Optional[int] = None,
) -> Dict[str, str]:
    """Produce the headers an external caller needs. Exposed so tests and
    small scripts don't have to reimplement the canonical format."""
    s = (secret or AGENT_SHARED_SECRET).strip()
    if not s:
        raise RuntimeError("AGENT_SHARED_SECRET not configured")
    pk = (public_key or AGENT_PUBLIC_KEY).strip()
    ts = str(timestamp if timestamp is not None else int(time.time()))
    sig = _expected_signature(s, _canonical_message(method, path, ts, body or b""))
    return {
        "X-Agent-Key": pk,
        "X-Agent-Timestamp": ts,
        "X-Agent-Signature": sig,
        "Content-Type": "application/json",
    }


__all__ = ["bp", "build_agent_headers"]

"""JSON file persistence for tokens, rules, and audit logs.

Storage is namespaced by platform (``newsbreak`` | ``smartnews``) so the same
ad-account id or user id can coexist across platforms without collisions.

Layout under STORAGE_ROOT:

    tokens/
        newsbreak/<user_id>.json
        smartnews/<user_id>.json
    rules/
        newsbreak/<account_id>.json
        smartnews/<account_id>.json
    audit/
        newsbreak/<account_id>.jsonl
        smartnews/<account_id>.jsonl
    catalog/
        newsbreak/
            pixels.json
            events.json
            offers.json
        smartnews/
            pixels.json
            events.json
            offers.json

A one-shot migration on import moves pre-namespace files into the
``newsbreak/`` subfolder so existing installs don't lose data.
"""
from __future__ import annotations

import functools
import json
import os
import shutil
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, TypeVar

DEFAULT_PLATFORM = "newsbreak"
KNOWN_PLATFORMS = ("newsbreak", "smartnews", "outbrain", "mediago")

_LOCAL_STORAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage")


def _resolve_storage_root() -> str:
    """Prefer $NEWSBREAK_STORAGE_DIR if writable, else fall back to local ./storage."""
    configured = os.environ.get("NEWSBREAK_STORAGE_DIR", "").strip()
    if configured:
        try:
            os.makedirs(configured, exist_ok=True)
            probe = os.path.join(configured, ".write_probe")
            with open(probe, "w", encoding="utf-8") as f:
                f.write("")
            os.remove(probe)
            return configured
        except OSError:
            pass
    return _LOCAL_STORAGE


STORAGE_ROOT = _resolve_storage_root()

import logging as _logging
_log = _logging.getLogger("storage")
_configured = os.environ.get("NEWSBREAK_STORAGE_DIR", "").strip()
_is_persistent = STORAGE_ROOT.startswith("/var/data") or STORAGE_ROOT.startswith("/data")
_log.warning(
    "storage.init: STORAGE_ROOT=%s (env NEWSBREAK_STORAGE_DIR=%r, persistent_disk=%s)",
    STORAGE_ROOT, _configured, _is_persistent,
)
if not _is_persistent and _configured:
    _log.error(
        "storage.init: $NEWSBREAK_STORAGE_DIR=%r not writable — fell back to ephemeral %s. "
        "Data will be lost on every deploy. Fix the env var / disk mount.",
        _configured, STORAGE_ROOT,
    )
elif not _is_persistent:
    _log.error(
        "storage.init: $NEWSBREAK_STORAGE_DIR is NOT set — using ephemeral %s. "
        "Data WILL be wiped on every deploy. Set NEWSBREAK_STORAGE_DIR=/var/data/storage.",
        STORAGE_ROOT,
    )

TOKENS_DIR = os.path.join(STORAGE_ROOT, "tokens")
RULES_DIR = os.path.join(STORAGE_ROOT, "rules")
AUDIT_DIR = os.path.join(STORAGE_ROOT, "audit")
CATALOG_DIR = os.path.join(STORAGE_ROOT, "catalog")


def _norm_platform(platform: Optional[str]) -> str:
    p = (platform or "").strip().lower()
    if p not in KNOWN_PLATFORMS:
        return DEFAULT_PLATFORM
    return p


def _tokens_dir(platform: str) -> str:
    return os.path.join(TOKENS_DIR, _norm_platform(platform))


def _rules_dir(platform: str) -> str:
    return os.path.join(RULES_DIR, _norm_platform(platform))


def _audit_dir(platform: str) -> str:
    return os.path.join(AUDIT_DIR, _norm_platform(platform))


def _catalog_dir(platform: str) -> str:
    return os.path.join(CATALOG_DIR, _norm_platform(platform))


def _pixels_file(platform: str) -> str:
    return os.path.join(_catalog_dir(platform), "pixels.json")


def _events_file(platform: str) -> str:
    return os.path.join(_catalog_dir(platform), "events.json")


def _offers_file(platform: str) -> str:
    return os.path.join(_catalog_dir(platform), "offers.json")


def ensure_dirs() -> None:
    os.makedirs(STORAGE_ROOT, exist_ok=True)
    for base in (TOKENS_DIR, RULES_DIR, AUDIT_DIR, CATALOG_DIR):
        os.makedirs(base, exist_ok=True)
        for p in KNOWN_PLATFORMS:
            os.makedirs(os.path.join(base, p), exist_ok=True)


def _migrate_flat_to_namespaced() -> None:
    """Move flat files into ``newsbreak/`` subfolder so pre-namespace installs keep data."""
    try:
        _migrate_dir(TOKENS_DIR, ext=".json")
        _migrate_dir(RULES_DIR, ext=".json")
        _migrate_dir(AUDIT_DIR, ext=".jsonl")
        for fname in ("pixels.json", "events.json", "offers.json"):
            src = os.path.join(CATALOG_DIR, fname)
            if os.path.isfile(src):
                dst_dir = os.path.join(CATALOG_DIR, "newsbreak")
                os.makedirs(dst_dir, exist_ok=True)
                dst = os.path.join(dst_dir, fname)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
    except Exception:
        # Migration is best-effort — a failure here shouldn't break the app.
        pass


def _migrate_dir(base: str, *, ext: str) -> None:
    if not os.path.isdir(base):
        return
    nb_dir = os.path.join(base, "newsbreak")
    os.makedirs(nb_dir, exist_ok=True)
    for name in os.listdir(base):
        src = os.path.join(base, name)
        if not os.path.isfile(src):
            continue
        if not name.endswith(ext):
            continue
        dst = os.path.join(nb_dir, name)
        if not os.path.exists(dst):
            shutil.move(src, dst)


ensure_dirs()
_migrate_flat_to_namespaced()


# Serialises every read-modify-write against the JSON/JSONL stores.
# gunicorn runs a single process with several gthread workers, so an
# in-process lock is sufficient; if the deployment ever moves to
# multiple worker *processes*, this must become a cross-process file
# lock. Without this lock (pre 2026-07), two overlapping requests could
# interleave a truncate-write with a read, making the reader see partial
# JSON, fall back to [], and clobber the whole file (this wiped
# style_candidates.json during a bulk promote on 2026-07-12).
_MUTATE_LOCK = threading.RLock()

_F = TypeVar("_F", bound=Callable[..., Any])


def _locked(fn: _F) -> _F:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with _MUTATE_LOCK:
            return fn(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def _read_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def _read_json_strict(path: str, default: Any) -> Any:
    """Like :func:`_read_json` but raises on a corrupt (unparseable) file.

    Used by read-modify-write paths: silently treating a corrupt file as
    ``default`` would make the subsequent write throw away every record
    that was in the file. Better to fail the single mutating request.
    """
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, data: Any) -> None:
    """Atomically replace ``path`` with the serialised ``data``.

    Writes to a unique temp file in the same directory, then
    ``os.replace``s it over the target so concurrent readers only ever
    see the old or the new complete file — never a partial write.
    """
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=directory, prefix=os.path.basename(path) + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


# --- Tokens (per session user id, per platform) ---
def save_token(
    user_id: str,
    access_token: Any,
    org_ids: List[str],
    *,
    platform: str = DEFAULT_PLATFORM,
) -> None:
    """Persist credentials for a user/platform combo.

    ``access_token`` may be:
      - a string (legacy NewsBreak-style bearer token), or
      - a dict with platform-specific keys (e.g. SmartNews v3 OAuth
        ``{"client_id", "client_secret"}``).
    """
    path = os.path.join(_tokens_dir(platform), f"{user_id}.json")
    payload: Dict[str, Any] = {
        "org_ids": org_ids,
        "platform": _norm_platform(platform),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if isinstance(access_token, dict):
        payload.update(access_token)
    else:
        payload["access_token"] = access_token
    _write_json(path, payload)


def load_token(user_id: str, *, platform: str = DEFAULT_PLATFORM) -> Optional[Dict[str, Any]]:
    path = os.path.join(_tokens_dir(platform), f"{user_id}.json")
    data = _read_json(path, None)
    if not data:
        return None
    has_creds = bool(
        data.get("access_token")
        or data.get("api_token")
        or (data.get("client_id") and data.get("client_secret"))
    )
    return data if has_creds else None


def delete_token(user_id: str, *, platform: str = DEFAULT_PLATFORM) -> None:
    path = os.path.join(_tokens_dir(platform), f"{user_id}.json")
    if os.path.exists(path):
        os.remove(path)


# --- Rules (per ad account id, per platform) ---
def load_rules(account_id: str, *, platform: str = DEFAULT_PLATFORM) -> List[Dict[str, Any]]:
    path = os.path.join(_rules_dir(platform), f"{account_id}.json")
    data = _read_json(path, [])
    return data if isinstance(data, list) else []


def save_rules(account_id: str, rules: List[Dict[str, Any]], *, platform: str = DEFAULT_PLATFORM) -> None:
    path = os.path.join(_rules_dir(platform), f"{account_id}.json")
    _write_json(path, rules)


def upsert_rule(account_id: str, rule: Dict[str, Any], *, platform: str = DEFAULT_PLATFORM) -> None:
    rules = load_rules(account_id, platform=platform)
    rid = rule.get("id")
    rule["platform"] = _norm_platform(rule.get("platform") or platform)
    found = False
    for i, r in enumerate(rules):
        if r.get("id") == rid:
            rules[i] = rule
            found = True
            break
    if not found:
        if not rid:
            rule["id"] = str(uuid.uuid4())
        rules.append(rule)
    save_rules(account_id, rules, platform=platform)


def delete_rule(account_id: str, rule_id: str, *, platform: str = DEFAULT_PLATFORM) -> bool:
    rules = load_rules(account_id, platform=platform)
    new_rules = [r for r in rules if r.get("id") != rule_id]
    if len(new_rules) == len(rules):
        return False
    save_rules(account_id, new_rules, platform=platform)
    return True


# --- Audit log (append-only jsonl, per platform) ---
def append_audit(account_id: str, entry: Dict[str, Any], *, platform: str = DEFAULT_PLATFORM) -> None:
    p = _norm_platform(platform)
    path = os.path.join(_audit_dir(p), f"{account_id}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = json.dumps(
        {**entry, "platform": p, "ts": datetime.now(timezone.utc).isoformat()},
        default=str,
    )
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_audit_tail(account_id: str, max_lines: int = 200, *, platform: str = DEFAULT_PLATFORM) -> List[Dict[str, Any]]:
    path = os.path.join(_audit_dir(platform), f"{account_id}.jsonl")
    if not os.path.exists(path):
        return []
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(line.strip())
    lines = lines[-max_lines:]
    out: List[Dict[str, Any]] = []
    for line in lines:
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def list_accounts_with_rules(*, platform: str = DEFAULT_PLATFORM) -> List[str]:
    """Account ids that have a rules file on this platform."""
    d = _rules_dir(platform)
    if not os.path.isdir(d):
        return []
    return [f.replace(".json", "") for f in os.listdir(d) if f.endswith(".json")]


# --- Catalog: pixels, conversion events, offers (per platform) ---
def _load_catalog(path: str) -> List[Dict[str, Any]]:
    data = _read_json(path, [])
    return data if isinstance(data, list) else []


def _load_catalog_for_update(path: str) -> List[Dict[str, Any]]:
    """Catalog load used by mutators — raises instead of clobbering.

    A corrupt file must fail the mutation rather than be treated as an
    empty catalog (which would erase every existing record on save).
    """
    data = _read_json_strict(path, [])
    return data if isinstance(data, list) else []


def _save_catalog(path: str, items: List[Dict[str, Any]]) -> None:
    _write_json(path, items)


@_locked
def _upsert_catalog(path: str, item: Dict[str, Any]) -> Dict[str, Any]:
    items = _load_catalog_for_update(path)
    now = datetime.now(timezone.utc).isoformat()
    if not item.get("id"):
        item["id"] = str(uuid.uuid4())
        item["created_at"] = now
    item["updated_at"] = now
    for i, existing in enumerate(items):
        if existing.get("id") == item["id"]:
            items[i] = {**existing, **item}
            _save_catalog(path, items)
            return items[i]
    items.append(item)
    _save_catalog(path, items)
    return item


@_locked
def _delete_catalog(path: str, item_id: str) -> bool:
    items = _load_catalog_for_update(path)
    remaining = [x for x in items if x.get("id") != item_id]
    if len(remaining) == len(items):
        return False
    _save_catalog(path, remaining)
    return True


def list_pixels(*, platform: str = DEFAULT_PLATFORM) -> List[Dict[str, Any]]:
    return _load_catalog(_pixels_file(platform))


def upsert_pixel(item: Dict[str, Any], *, platform: str = DEFAULT_PLATFORM) -> Dict[str, Any]:
    return _upsert_catalog(_pixels_file(platform), item)


def delete_pixel(item_id: str, *, platform: str = DEFAULT_PLATFORM) -> bool:
    return _delete_catalog(_pixels_file(platform), item_id)


def list_events(*, platform: str = DEFAULT_PLATFORM) -> List[Dict[str, Any]]:
    return _load_catalog(_events_file(platform))


def upsert_event(item: Dict[str, Any], *, platform: str = DEFAULT_PLATFORM) -> Dict[str, Any]:
    return _upsert_catalog(_events_file(platform), item)


def delete_event(item_id: str, *, platform: str = DEFAULT_PLATFORM) -> bool:
    return _delete_catalog(_events_file(platform), item_id)


def list_offers(*, platform: str = DEFAULT_PLATFORM) -> List[Dict[str, Any]]:
    return _load_catalog(_offers_file(platform))


def merge_offer_platform_pixels(
    existing: Optional[Dict[str, Any]],
    platform: str,
    pixel_id: str,
    incoming_map: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Keep a per-platform pixel map so switching platforms cannot overwrite.

    Catalogs are already namespaced, but offers may be copied or later shared.
    ``pixel_id`` is the convenience field for the *current* platform.
    """
    out: Dict[str, str] = {}
    if existing and isinstance(existing.get("pixels"), dict):
        for k, v in existing["pixels"].items():
            if k and v not in (None, ""):
                out[str(k)] = str(v)
    if isinstance(incoming_map, dict):
        for k, v in incoming_map.items():
            if k and v not in (None, ""):
                out[str(k)] = str(v)
    plat = (platform or DEFAULT_PLATFORM).strip() or DEFAULT_PLATFORM
    pid = (pixel_id or "").strip()
    if pid:
        out[plat] = pid
    else:
        out.pop(plat, None)
    return out


def offer_pixel_ref(offer: Optional[Dict[str, Any]], platform: str) -> str:
    """Resolve the catalog pixel id / conversion name for ``platform``."""
    if not offer:
        return ""
    pixels = offer.get("pixels") if isinstance(offer.get("pixels"), dict) else {}
    plat = (platform or DEFAULT_PLATFORM).strip() or DEFAULT_PLATFORM
    return str(pixels.get(plat) or offer.get("pixel_id") or "").strip()


def merge_offer_platform_accounts(
    existing: Optional[Dict[str, Any]],
    platform: str,
    account_id: str,
    incoming_map: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Keep a per-platform ad-account map (same shape as ``pixels``).

    ``account_id`` is the convenience primary for the *current* platform.
    ``ad_account_ids`` remains the multi-select list used by the settings UI.
    """
    out: Dict[str, str] = {}
    if existing and isinstance(existing.get("accounts"), dict):
        for k, v in existing["accounts"].items():
            if k and v not in (None, ""):
                out[str(k)] = str(v)
    if isinstance(incoming_map, dict):
        for k, v in incoming_map.items():
            if k and v not in (None, ""):
                out[str(k)] = str(v)
    plat = (platform or DEFAULT_PLATFORM).strip() or DEFAULT_PLATFORM
    aid = (account_id or "").strip()
    if aid:
        out[plat] = aid
    else:
        out.pop(plat, None)
    return out


def offer_account_ids(offer: Optional[Dict[str, Any]], platform: str) -> List[str]:
    """Linked ad-account ids for ``platform`` (map first, then ``ad_account_ids``)."""
    if not offer:
        return []
    plat = (platform or DEFAULT_PLATFORM).strip() or DEFAULT_PLATFORM
    seen: List[str] = []
    accounts = offer.get("accounts") if isinstance(offer.get("accounts"), dict) else {}
    primary = str(accounts.get(plat) or "").strip()
    if primary:
        seen.append(primary)
    raw = offer.get("ad_account_ids") or []
    if isinstance(raw, str):
        raw = [x.strip() for x in raw.split(",")]
    for x in raw:
        aid = str(x).strip()
        if aid and aid not in seen:
            seen.append(aid)
    return seen


def offer_landing_url(offer: Optional[Dict[str, Any]]) -> str:
    """Resolve the offer lander / copy-variant URL."""
    if not offer:
        return ""
    for key in (
        "landing_url",
        "landing_page",
        "landing_page_url",
        "lander_url",
        "copy_variant_url",
        "copy_url",
    ):
        raw = str(offer.get(key) or "").strip()
        if raw:
            return raw
    variants = offer.get("copy_variants") or offer.get("variants") or []
    if isinstance(variants, list):
        for row in variants:
            if not isinstance(row, dict):
                continue
            raw = str(
                row.get("landing_url")
                or row.get("landing_page")
                or row.get("url")
                or ""
            ).strip()
            if raw:
                return raw
    return ""


def upsert_offer(item: Dict[str, Any], *, platform: str = DEFAULT_PLATFORM) -> Dict[str, Any]:
    return _upsert_catalog(_offers_file(platform), item)


def delete_offer(item_id: str, *, platform: str = DEFAULT_PLATFORM) -> bool:
    return _delete_catalog(_offers_file(platform), item_id)


def list_token_user_ids(*, platform: str = DEFAULT_PLATFORM) -> List[str]:
    """User ids that have saved API tokens (for scheduler)."""
    d = _tokens_dir(platform)
    if not os.path.isdir(d):
        return []
    return [f.replace(".json", "") for f in os.listdir(d) if f.endswith(".json")]


# --- AI Ad Studio: winners / insights / generations / style candidates ---
# All files live under storage/catalog/<platform>/ alongside pixels/events/offers.
#
#   winners.json           list of proven ad-level winners (offer_id-tagged)
#   ad_insights.json       per-offer AI digest cache
#   generations.jsonl      append-only generation batch log
#   style_candidates.json  research pool of candidate ad styles
#   research_runs.jsonl    append-only log of discovery runs
#   ad_library.jsonl       prebuilt-ad library log (one row per stashed image)
#   library_images/        rendered PNGs/JPEGs for the prebuilt library
def _winners_file(platform: str) -> str:
    return os.path.join(_catalog_dir(platform), "winners.json")


def _insights_file(platform: str) -> str:
    return os.path.join(_catalog_dir(platform), "ad_insights.json")


def _generations_file(platform: str) -> str:
    return os.path.join(_catalog_dir(platform), "generations.jsonl")


def _style_candidates_file(platform: str) -> str:
    return os.path.join(_catalog_dir(platform), "style_candidates.json")


def _research_runs_file(platform: str) -> str:
    return os.path.join(_catalog_dir(platform), "research_runs.jsonl")


def _library_file(platform: str) -> str:
    return os.path.join(_catalog_dir(platform), "ad_library.jsonl")


def winner_image_dir(platform: str = DEFAULT_PLATFORM) -> str:
    """Directory that holds cached winner creative images (one file per ad)."""
    path = os.path.join(_catalog_dir(platform), "winner_images")
    os.makedirs(path, exist_ok=True)
    return path


def winner_image_path(ad_id: str, *, platform: str = DEFAULT_PLATFORM, ext: str = "jpg") -> str:
    safe = "".join(c for c in str(ad_id) if c.isalnum() or c in ("-", "_")) or "unknown"
    return os.path.join(winner_image_dir(platform), f"{safe}.{ext.lstrip('.')}")


def library_image_dir(platform: str = DEFAULT_PLATFORM) -> str:
    """Directory that holds rendered images for the prebuilt-ad library."""
    path = os.path.join(_catalog_dir(platform), "library_images")
    os.makedirs(path, exist_ok=True)
    return path


def public_creative_dir() -> str:
    """Directory for publicly-servable creative images.

    Outbrain/Teads promoted links are created with an ``imageUrl`` that
    Outbrain fetches and caches, so prepared creatives are written here and
    exposed via an unauthenticated ``/public/creative/<file>`` route.
    """
    path = os.path.join(STORAGE_ROOT, "public_creatives")
    os.makedirs(path, exist_ok=True)
    return path


def library_image_path(filename: str, *, platform: str = DEFAULT_PLATFORM) -> str:
    """Resolve a library image filename to its on-disk absolute path.

    The filename is sanitised here so route handlers can pass user-derived
    values without worrying about path traversal.
    """
    safe = "".join(c for c in str(filename) if c.isalnum() or c in ("-", "_", ".")) or "unknown"
    return os.path.join(library_image_dir(platform), safe)


# Winners ---------------------------------------------------------------
def list_winners(*, platform: str = DEFAULT_PLATFORM) -> List[Dict[str, Any]]:
    return _load_catalog(_winners_file(platform))


def list_all_winners(*, platforms: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Return winners from every platform as a single combined pool.

    Each row is tagged with ``source_platform`` so callers can still attribute
    performance to the surface the ad ran on. Writes stay namespaced by
    platform (see ``upsert_winner``); this helper is read-only and exists so
    the AI Studio analyzer / prompt generator has one large signal pool that
    spans SmartNews + NewsBreak.

    Winners that have the same ``ad_id`` across platforms are kept as
    separate rows (different ad objects, even if the landing URL matches).
    """
    plats = platforms or list(KNOWN_PLATFORMS)
    out: List[Dict[str, Any]] = []
    for p in plats:
        try:
            rows = _load_catalog(_winners_file(p))
        except Exception:
            rows = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            r.setdefault("source_platform", p)
            out.append(r)
    return out


@_locked
def upsert_winner(item: Dict[str, Any], *, platform: str = DEFAULT_PLATFORM) -> Dict[str, Any]:
    """Upsert by ``ad_id`` (preferred) or fall back to generated ``id``."""
    path = _winners_file(platform)
    items = _load_catalog_for_update(path)
    now = datetime.now(timezone.utc).isoformat()
    key = str(item.get("ad_id") or item.get("id") or "")
    if not key:
        item["id"] = str(uuid.uuid4())
        item["created_at"] = now
        item["updated_at"] = now
        items.append(item)
        _save_catalog(path, items)
        return item
    for i, existing in enumerate(items):
        if str(existing.get("ad_id") or existing.get("id") or "") == key:
            merged = {**existing, **item, "updated_at": now}
            if not merged.get("id"):
                merged["id"] = str(uuid.uuid4())
            items[i] = merged
            _save_catalog(path, items)
            return merged
    item["id"] = item.get("id") or str(uuid.uuid4())
    item["created_at"] = item.get("created_at") or now
    item["updated_at"] = now
    items.append(item)
    _save_catalog(path, items)
    return item


@_locked
def delete_winner(ad_id: str, *, platform: str = DEFAULT_PLATFORM) -> bool:
    path = _winners_file(platform)
    items = _load_catalog_for_update(path)
    remaining = [x for x in items if str(x.get("ad_id")) != str(ad_id) and x.get("id") != ad_id]
    if len(remaining) == len(items):
        return False
    _save_catalog(path, remaining)
    return True


# Insights --------------------------------------------------------------
def list_insights(*, platform: str = DEFAULT_PLATFORM) -> List[Dict[str, Any]]:
    return _load_catalog(_insights_file(platform))


def load_insights(offer_id: str, *, platform: str = DEFAULT_PLATFORM) -> Optional[Dict[str, Any]]:
    key = str(offer_id)
    for it in list_insights(platform=platform):
        if str(it.get("offer_id")) == key:
            return it
    return None


@_locked
def save_insights(offer_id: str, insights: Dict[str, Any], *, platform: str = DEFAULT_PLATFORM) -> Dict[str, Any]:
    path = _insights_file(platform)
    items = _load_catalog_for_update(path)
    now = datetime.now(timezone.utc).isoformat()
    payload = {**insights, "offer_id": str(offer_id), "updated_at": now}
    if not payload.get("generated_at"):
        payload["generated_at"] = now
    for i, existing in enumerate(items):
        if str(existing.get("offer_id")) == str(offer_id):
            items[i] = payload
            _save_catalog(path, items)
            return payload
    items.append(payload)
    _save_catalog(path, items)
    return payload


# Generations log (append-only jsonl) -----------------------------------
@_locked
def append_generation(entry: Dict[str, Any], *, platform: str = DEFAULT_PLATFORM) -> Dict[str, Any]:
    path = _generations_file(platform)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    row = {
        **entry,
        "platform": _norm_platform(platform),
        "created_at": entry.get("created_at") or now,
    }
    if not row.get("gen_id"):
        row["gen_id"] = str(uuid.uuid4())
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
    return row


def list_generations(*, platform: str = DEFAULT_PLATFORM, limit: int = 500) -> List[Dict[str, Any]]:
    path = _generations_file(platform)
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if limit > 0:
        return out[-limit:]
    return out


# Prebuilt ad library (jsonl + on-disk images) -------------------------
#
# Each row represents ONE rendered image stashed for later use by a
# manual /api/studio/generate call. Rows are append-only; consumption
# rewrites the file with ``consumed_at`` stamped on the chosen rows so
# we can keep a forensic trail of which library entries fed which launch
# (useful for the bandit + analyzer when learning what library
# pre-renders convert vs. fresh ones).
@_locked
def append_library_item(entry: Dict[str, Any], *, platform: str = DEFAULT_PLATFORM) -> Dict[str, Any]:
    path = _library_file(platform)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    row = {
        **entry,
        "platform": _norm_platform(platform),
        "created_at": entry.get("created_at") or now,
        "consumed_at": entry.get("consumed_at") or None,
    }
    if not row.get("library_id"):
        row["library_id"] = str(uuid.uuid4())
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
    return row


def list_library_items(
    *,
    platform: str = DEFAULT_PLATFORM,
    offer_id: Optional[str] = None,
    include_consumed: bool = False,
) -> List[Dict[str, Any]]:
    path = _library_file(platform)
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if offer_id is not None and str(row.get("offer_id")) != str(offer_id):
                continue
            if not include_consumed and row.get("consumed_at"):
                continue
            out.append(row)
    return out


def library_counts(*, platform: str = DEFAULT_PLATFORM) -> Dict[str, int]:
    """Per-offer count of unconsumed library items for the given platform."""
    counts: Dict[str, int] = {}
    for row in list_library_items(platform=platform, include_consumed=False):
        oid = str(row.get("offer_id") or "")
        counts[oid] = counts.get(oid, 0) + 1
    return counts


@_locked
def consume_library_items(
    offer_id: str,
    n: int,
    *,
    platform: str = DEFAULT_PLATFORM,
) -> List[Dict[str, Any]]:
    """Pop up to ``n`` unconsumed items for the offer, FIFO (oldest first).

    Marks the chosen rows ``consumed_at=<now>`` and rewrites the jsonl
    file in place. Returns the consumed rows so the caller can build
    response payloads. We rewrite the entire file because the library
    file stays small in practice (a few hundred rows max — the
    background topup job caps total stock).
    """
    if n <= 0:
        return []
    path = _library_file(platform)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    chosen: List[Dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()
    for row in rows:
        if row.get("consumed_at"):
            continue
        if str(row.get("offer_id")) != str(offer_id):
            continue
        row["consumed_at"] = now
        chosen.append(row)
        if len(chosen) >= n:
            break
    if not chosen:
        return []
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    shutil.move(tmp, path)
    return chosen


@_locked
def set_library_consumed(
    library_ids: List[str],
    consumed: bool,
    *,
    platform: str = DEFAULT_PLATFORM,
    used_in: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Mark specific library rows used (``consumed_at=now``) or unused.

    Unlike :func:`consume_library_items` (FIFO by offer), this targets exact
    ``library_id`` values — used by the launch flow's "Use from library"
    picker and by the manual mark-used / mark-unused toggles in the studio
    Research tab. When marking used, ``used_in`` (e.g. ``{"platform": ...,
    "campaign_id": ...}``) is stamped on the row so there's a forensic
    trail of which launch consumed which prebuilt image. Marking unused
    clears both ``consumed_at`` and ``used_in``.

    Returns the updated rows (empty when nothing matched).
    """
    wanted = {str(x) for x in (library_ids or []) if str(x).strip()}
    if not wanted:
        return []
    path = _library_file(platform)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    now = datetime.now(timezone.utc).isoformat()
    updated: List[Dict[str, Any]] = []
    for row in rows:
        if str(row.get("library_id")) not in wanted:
            continue
        if consumed:
            row["consumed_at"] = now
            if used_in:
                row["used_in"] = used_in
        else:
            row["consumed_at"] = None
            row.pop("used_in", None)
        updated.append(row)
    if not updated:
        return []
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    shutil.move(tmp, path)
    return updated


@_locked
def update_generation(gen_id: str, patch: Dict[str, Any], *, platform: str = DEFAULT_PLATFORM) -> Optional[Dict[str, Any]]:
    """Rewrite the jsonl in place with ``patch`` applied to the matching row.

    Append-only is preserved for any append ordering — we just rewrite the
    whole file since typical generations log is small (< 10k rows).
    """
    path = _generations_file(platform)
    rows = list_generations(platform=platform, limit=0)
    updated: Optional[Dict[str, Any]] = None
    for i, r in enumerate(rows):
        if str(r.get("gen_id")) == str(gen_id):
            merged = {**r, **patch, "updated_at": datetime.now(timezone.utc).isoformat()}
            rows[i] = merged
            updated = merged
            break
    if updated is None:
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    shutil.move(tmp, path)
    return updated


# Style candidates ------------------------------------------------------
def list_style_candidates(*, platform: str = DEFAULT_PLATFORM) -> List[Dict[str, Any]]:
    return _load_catalog(_style_candidates_file(platform))


@_locked
def upsert_style_candidate(item: Dict[str, Any], *, platform: str = DEFAULT_PLATFORM) -> Dict[str, Any]:
    """Upsert by ``style_id`` (preferred) or fall back to generated ``id``."""
    path = _style_candidates_file(platform)
    items = _load_catalog_for_update(path)
    now = datetime.now(timezone.utc).isoformat()
    key = str(item.get("style_id") or item.get("id") or "")
    if not key:
        item["style_id"] = str(uuid.uuid4())
        item["id"] = item["style_id"]
        item["created_at"] = now
        item["updated_at"] = now
        items.append(item)
        _save_catalog(path, items)
        return item
    for i, existing in enumerate(items):
        if str(existing.get("style_id") or existing.get("id") or "") == key:
            merged = {**existing, **item, "updated_at": now}
            items[i] = merged
            _save_catalog(path, items)
            return merged
    item["style_id"] = item.get("style_id") or key
    item["id"] = item.get("id") or item["style_id"]
    item["created_at"] = item.get("created_at") or now
    item["updated_at"] = now
    items.append(item)
    _save_catalog(path, items)
    return item


@_locked
def delete_style_candidate(style_id: str, *, platform: str = DEFAULT_PLATFORM) -> bool:
    path = _style_candidates_file(platform)
    items = _load_catalog_for_update(path)
    remaining = [x for x in items if str(x.get("style_id")) != str(style_id) and x.get("id") != style_id]
    if len(remaining) == len(items):
        return False
    _save_catalog(path, remaining)
    return True


@_locked
def append_research_run(entry: Dict[str, Any], *, platform: str = DEFAULT_PLATFORM) -> Dict[str, Any]:
    path = _research_runs_file(platform)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    row = {
        **entry,
        "platform": _norm_platform(platform),
        "created_at": entry.get("created_at") or now,
    }
    if not row.get("run_id"):
        row["run_id"] = str(uuid.uuid4())
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
    return row


def list_research_runs(*, platform: str = DEFAULT_PLATFORM, limit: int = 200) -> List[Dict[str, Any]]:
    path = _research_runs_file(platform)
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if limit > 0:
        return out[-limit:]
    return out


# ---------------------------------------------------------------------------
# Managed-agent job queue
# ---------------------------------------------------------------------------
#
# Tiny jsonl-backed queue consumed by ``agent_api.drain-queue`` / external
# cron. Each line is one job row; status transitions rewrite the whole
# file in-place (same pattern as ``update_generation``).


def _agent_queue_file(platform: str) -> str:
    return os.path.join(_catalog_dir(platform), "agent_queue.jsonl")


@_locked
def append_agent_job(entry: Dict[str, Any], *, platform: str = DEFAULT_PLATFORM) -> Dict[str, Any]:
    path = _agent_queue_file(platform)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    row = {
        **entry,
        "platform": _norm_platform(platform),
        "enqueued_at": entry.get("enqueued_at") or now,
    }
    if not row.get("job_id"):
        row["job_id"] = str(uuid.uuid4())
    if not row.get("status"):
        row["status"] = "queued"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
    return row


def list_agent_jobs(
    *,
    platform: str = DEFAULT_PLATFORM,
    status: Optional[str] = None,
    limit: int = 500,
) -> List[Dict[str, Any]]:
    path = _agent_queue_file(platform)
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if status and str(row.get("status")) != status:
                continue
            out.append(row)
    if limit > 0:
        return out[-limit:]
    return out


@_locked
def update_agent_job(
    job_id: str,
    patch: Dict[str, Any],
    *,
    platform: str = DEFAULT_PLATFORM,
) -> Optional[Dict[str, Any]]:
    path = _agent_queue_file(platform)
    if not os.path.exists(path):
        return None
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    updated: Optional[Dict[str, Any]] = None
    for i, r in enumerate(rows):
        if str(r.get("job_id")) == str(job_id):
            merged = {**r, **patch, "updated_at": datetime.now(timezone.utc).isoformat()}
            rows[i] = merged
            updated = merged
            break
    if updated is None:
        return None
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    shutil.move(tmp, path)
    return updated


# --- Site / source exclusions (per platform + account) ---
def _site_exclusions_file(platform: str) -> str:
    return os.path.join(_catalog_dir(platform), "site_exclusions.json")


def load_site_exclusions(account_id: str, *, platform: str = DEFAULT_PLATFORM) -> List[Dict[str, Any]]:
    """Persisted publisher/site blocklist for one account on one platform."""
    data = _read_json(_site_exclusions_file(platform), {})
    if not isinstance(data, dict):
        return []
    bucket = data.get(str(account_id)) or {}
    sites = bucket.get("sites") if isinstance(bucket, dict) else bucket
    if not isinstance(sites, list):
        return []
    return [s for s in sites if isinstance(s, dict)]


@_locked
def save_site_exclusions(
    account_id: str,
    sites: List[Dict[str, Any]],
    *,
    platform: str = DEFAULT_PLATFORM,
) -> List[Dict[str, Any]]:
    path = _site_exclusions_file(platform)
    data = _read_json_strict(path, {})
    if not isinstance(data, dict):
        data = {}
    clean: List[Dict[str, Any]] = []
    seen = set()
    for s in sites or []:
        if not isinstance(s, dict):
            continue
        sid = str(s.get("site_id") or "").strip()
        if not sid or sid in seen or sid in ("0",):
            continue
        seen.add(sid)
        clean.append(
            {
                "site_id": sid,
                "domain_name": s.get("domain_name") or s.get("site_name") or "",
                "site_name": s.get("site_name") or s.get("domain_name") or "",
                "excluded_at": s.get("excluded_at") or datetime.now(timezone.utc).isoformat(),
                "reason": s.get("reason") or s.get("flag") or "",
            }
        )
    data[str(account_id)] = {
        "sites": clean,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(path, data)
    return clean

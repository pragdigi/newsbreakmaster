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

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

DEFAULT_PLATFORM = "newsbreak"
KNOWN_PLATFORMS = ("newsbreak", "smartnews")

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


def _read_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def _write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


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
        data.get("access_token") or (data.get("client_id") and data.get("client_secret"))
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


def _save_catalog(path: str, items: List[Dict[str, Any]]) -> None:
    _write_json(path, items)


def _upsert_catalog(path: str, item: Dict[str, Any]) -> Dict[str, Any]:
    items = _load_catalog(path)
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


def _delete_catalog(path: str, item_id: str) -> bool:
    items = _load_catalog(path)
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

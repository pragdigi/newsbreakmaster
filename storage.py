"""JSON file persistence for tokens, rules, and audit logs."""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

STORAGE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage")
TOKENS_DIR = os.path.join(STORAGE_ROOT, "tokens")
RULES_DIR = os.path.join(STORAGE_ROOT, "rules")
AUDIT_DIR = os.path.join(STORAGE_ROOT, "audit")


def ensure_dirs() -> None:
    for d in (STORAGE_ROOT, TOKENS_DIR, RULES_DIR, AUDIT_DIR):
        os.makedirs(d, exist_ok=True)


def _read_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def _write_json(path: str, data: Any) -> None:
    ensure_dirs()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


# --- Tokens (per session user id) ---
def save_token(user_id: str, access_token: str, org_ids: List[str]) -> None:
    ensure_dirs()
    path = os.path.join(TOKENS_DIR, f"{user_id}.json")
    _write_json(
        path,
        {
            "access_token": access_token,
            "org_ids": org_ids,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def load_token(user_id: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(TOKENS_DIR, f"{user_id}.json")
    data = _read_json(path, None)
    if not data or not data.get("access_token"):
        return None
    return data


def delete_token(user_id: str) -> None:
    path = os.path.join(TOKENS_DIR, f"{user_id}.json")
    if os.path.exists(path):
        os.remove(path)


# --- Rules (per ad account id) ---
def load_rules(account_id: str) -> List[Dict[str, Any]]:
    path = os.path.join(RULES_DIR, f"{account_id}.json")
    data = _read_json(path, [])
    return data if isinstance(data, list) else []


def save_rules(account_id: str, rules: List[Dict[str, Any]]) -> None:
    path = os.path.join(RULES_DIR, f"{account_id}.json")
    _write_json(path, rules)


def upsert_rule(account_id: str, rule: Dict[str, Any]) -> None:
    rules = load_rules(account_id)
    rid = rule.get("id")
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
    save_rules(account_id, rules)


def delete_rule(account_id: str, rule_id: str) -> bool:
    rules = load_rules(account_id)
    new_rules = [r for r in rules if r.get("id") != rule_id]
    if len(new_rules) == len(rules):
        return False
    save_rules(account_id, new_rules)
    return True


# --- Audit log (append-only jsonl) ---
def append_audit(account_id: str, entry: Dict[str, Any]) -> None:
    ensure_dirs()
    path = os.path.join(AUDIT_DIR, f"{account_id}.jsonl")
    line = json.dumps(
        {**entry, "ts": datetime.now(timezone.utc).isoformat()},
        default=str,
    )
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_audit_tail(account_id: str, max_lines: int = 200) -> List[Dict[str, Any]]:
    path = os.path.join(AUDIT_DIR, f"{account_id}.jsonl")
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


def list_accounts_with_rules() -> List[str]:
    """Account ids that have a rules file."""
    ensure_dirs()
    if not os.path.isdir(RULES_DIR):
        return []
    return [f.replace(".json", "") for f in os.listdir(RULES_DIR) if f.endswith(".json")]


def list_token_user_ids() -> List[str]:
    """User ids that have saved API tokens (for scheduler)."""
    ensure_dirs()
    if not os.path.isdir(TOKENS_DIR):
        return []
    return [f.replace(".json", "") for f in os.listdir(TOKENS_DIR) if f.endswith(".json")]

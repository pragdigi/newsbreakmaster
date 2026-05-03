"""Show the most-recently-created style candidates per source."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import time

import requests

BASE = os.environ.get("AGENT_BASE_URL", "https://newsbreakmaster.onrender.com")
SECRET = os.environ.get(
    "AGENT_SHARED_SECRET",
    "396e51dc8e7599d0477b1056f63477453b6ab33254b0fdbddb81753aa932b464",
)
KEY = os.environ.get("AGENT_PUBLIC_KEY", "default")


def sign(method: str, path: str, body: bytes) -> dict:
    ts = str(int(time.time()))
    digest = hashlib.sha256(body or b"").hexdigest()
    msg = f"{method.upper()}\n{path}\n{ts}\n{digest}".encode("utf-8")
    sig = hmac.new(SECRET.encode("utf-8"), msg, hashlib.sha256).hexdigest()
    return {
        "X-Agent-Key": KEY,
        "X-Agent-Timestamp": ts,
        "X-Agent-Signature": sig,
    }


def main() -> int:
    for platform in ("newsbreak", "smartnews"):
        path = f"/api/agent/candidates?platform={platform}"
        headers = sign("GET", "/api/agent/candidates", b"")
        r = requests.get(f"{BASE}{path}", headers=headers, timeout=30)
        if r.status_code != 200:
            print(f"{platform}: status={r.status_code} body={r.text[:200]}")
            continue
        cands = r.json().get("candidates") or []
        scholar = [c for c in cands if c.get("source") == "scholar"]
        public = [c for c in cands if c.get("source") == "public_scout"]
        print(f"\n=== {platform.upper()} ===")
        print(f"  scholar: {len(scholar)}    public_scout: {len(public)}    total: {len(cands)}")

        # Show the most-recent 5 scholar candidates with their lens.
        scholar.sort(key=lambda c: c.get("created_at") or "", reverse=True)
        print("\n  Latest Scholar candidates:")
        for c in scholar[:5]:
            meta = c.get("source_meta") or {}
            print(f"   - id={c['style_id']!r}")
            print(f"     name={c.get('name')}")
            print(f"     lens={meta.get('lens_id')} ({meta.get('lens_kind')})")
            print(f"     desc={(c.get('description') or '')[:140]}")
            print(f"     framework_note={(meta.get('framework_note') or '')[:140]}")
            tmpl = (c.get('prompt_template') or '').strip()
            print(f"     prompt_template={tmpl[:200]}{'...' if len(tmpl) > 200 else ''}")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

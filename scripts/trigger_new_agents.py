"""One-shot: wait for Render to redeploy, then trigger both new agents."""
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
        "Content-Type": "application/json",
    }


def wait_healthy(max_wait_s: int = 240) -> bool:
    """Poll signed /api/agent/health until 200 (deploy complete)."""
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        try:
            headers = sign("GET", "/api/agent/health", b"")
            r = requests.get(f"{BASE}/api/agent/health", headers=headers, timeout=10)
            if r.status_code == 200:
                return True
            # 502/503/504 = still deploying; 401 = HMAC mismatch (don't retry).
            if r.status_code == 401:
                print(f"\n  HMAC rejected ({r.text[:200]}) — abort.")
                return False
        except requests.RequestException:
            pass
        print(".", end="", flush=True)
        time.sleep(8)
    return False


def call_signed(method: str, path: str, payload: dict, *, timeout: int = 600) -> dict:
    body = json.dumps(payload).encode("utf-8") if payload else b""
    headers = sign(method, path, body)
    if method == "POST":
        r = requests.post(f"{BASE}{path}", data=body, headers=headers, timeout=timeout)
    else:
        r = requests.get(f"{BASE}{path}", headers=headers, timeout=timeout)
    out = {"status": r.status_code}
    try:
        out["json"] = r.json()
    except Exception:
        out["text"] = r.text[:1000]
    return out


def main() -> int:
    print(f"Waiting for {BASE} to be healthy...", end="", flush=True)
    if not wait_healthy():
        print("\nDeploy didn't go healthy in 4 min — check Render logs.")
        return 1
    print(" OK")

    print("\n[1/3] Listing scholar lenses to confirm new endpoints exist...")
    r = call_signed("GET", "/api/agent/lenses", None, timeout=20)
    print(f"  status={r['status']}")
    if r["status"] != 200:
        print(f"  {r}")
        return 2
    lenses = r["json"].get("lenses") or []
    print(f"  {len(lenses)} lenses available — first: {lenses[0]['id'] if lenses else '(none)'}")

    print("\n[2/3] Triggering POST /api/agent/run-public-scout (Meta Ad Library + TikTok)...")
    r = call_signed(
        "POST",
        "/api/agent/run-public-scout",
        {"scan_all_offers": True, "country": "US", "sources": ["meta", "tiktok"]},
        timeout=600,
    )
    print(f"  status={r['status']}")
    print(f"  body={json.dumps(r.get('json') or r.get('text'), indent=2)[:1500]}")

    print("\n[3/3] Triggering POST /api/agent/run-scholar (Opus + Gemini fallback)...")
    r = call_signed(
        "POST",
        "/api/agent/run-scholar",
        {"scan_all_offers": True, "count_per_offer": 3},
        timeout=900,
    )
    print(f"  status={r['status']}")
    print(f"  body={json.dumps(r.get('json') or r.get('text'), indent=2)[:1500]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

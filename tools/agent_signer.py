#!/usr/bin/env python3
"""Tiny signer for ``/api/agent/*`` requests.

Example
-------

    # Ping health:
    $ python tools/agent_signer.py GET https://your-app.onrender.com/api/agent/health

    # List offers:
    $ python tools/agent_signer.py GET https://your-app.onrender.com/api/agent/offers?platform=newsbreak

    # Add a style candidate:
    $ python tools/agent_signer.py POST https://your-app.onrender.com/api/agent/candidates \\
          --body '{"name":"bus-stop snapshot","prompt_template":"..."}'

    # Schedule a generation:
    $ python tools/agent_signer.py POST https://your-app.onrender.com/api/agent/schedule-generation \\
          --body '{"offer_id":"<uuid>","count":10,"research_ratio":0.4}'

The script reads ``AGENT_SHARED_SECRET`` (required) and ``AGENT_PUBLIC_KEY``
from your shell env. Requires ``requests``.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
import time
from urllib.parse import urlparse

try:
    import requests
except ImportError:  # pragma: no cover
    sys.stderr.write("requests is required. pip install requests\n")
    raise


def canonical(method: str, path: str, ts: str, body: bytes) -> bytes:
    digest = hashlib.sha256(body or b"").hexdigest()
    return f"{method.upper()}\n{path}\n{ts}\n{digest}".encode("utf-8")


def sign(secret: str, method: str, path: str, ts: str, body: bytes) -> str:
    return hmac.new(secret.encode("utf-8"), canonical(method, path, ts, body), hashlib.sha256).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=["GET", "POST", "DELETE", "PUT"], help="HTTP method")
    parser.add_argument("url", help="full URL of the endpoint")
    parser.add_argument("--body", default="", help="request body (for POST/PUT)")
    parser.add_argument("--secret", default=os.environ.get("AGENT_SHARED_SECRET", ""))
    parser.add_argument("--key", default=os.environ.get("AGENT_PUBLIC_KEY", "default"))
    parser.add_argument("--print-only", action="store_true", help="print headers, don't fire request")
    args = parser.parse_args()

    if not args.secret:
        sys.stderr.write("AGENT_SHARED_SECRET is required (env or --secret)\n")
        return 2

    url_parts = urlparse(args.url)
    path = url_parts.path or "/"
    body_bytes = args.body.encode("utf-8") if args.body else b""
    ts = str(int(time.time()))
    sig = sign(args.secret, args.method, path, ts, body_bytes)
    headers = {
        "X-Agent-Key": args.key or "default",
        "X-Agent-Timestamp": ts,
        "X-Agent-Signature": sig,
        "Content-Type": "application/json",
    }

    if args.print_only:
        print(json.dumps({"url": args.url, "method": args.method, "headers": headers, "body": args.body}, indent=2))
        return 0

    resp = requests.request(args.method, args.url, headers=headers, data=body_bytes, timeout=120)
    print(f"HTTP {resp.status_code}")
    ct = resp.headers.get("Content-Type", "")
    if "application/json" in ct:
        try:
            print(json.dumps(resp.json(), indent=2))
            return 0 if resp.ok else 1
        except ValueError:
            pass
    print(resp.text)
    return 0 if resp.ok else 1


if __name__ == "__main__":
    sys.exit(main())

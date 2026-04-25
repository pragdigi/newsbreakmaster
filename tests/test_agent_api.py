"""Tests for the HMAC-signed managed-agent HTTP surface."""
import importlib
import json
import os
import sys
import unittest


class AgentApiTests(unittest.TestCase):
    def setUp(self):
        os.environ["AGENT_SHARED_SECRET"] = "test-secret-123"
        os.environ["AGENT_PUBLIC_KEY"] = "default"
        os.environ["AGENT_MAX_CLOCK_SKEW"] = "300"
        # Reimport agent_api + app so env vars take effect.
        for mod in ("agent_api", "app"):
            if mod in sys.modules:
                del sys.modules[mod]
        import app as _app  # noqa: F401
        import agent_api

        self.agent_api = agent_api
        self.client = _app.app.test_client()

    def _sign(self, method, path, body=b""):
        return self.agent_api.build_agent_headers(method=method, path=path, body=body)

    def test_health_returns_ok_when_signed(self):
        headers = self._sign("GET", "/api/agent/health")
        resp = self.client.get("/api/agent/health", headers=headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["ok"])
        self.assertIn("platforms", data)

    def test_rejects_missing_signature(self):
        resp = self.client.get("/api/agent/health")
        self.assertEqual(resp.status_code, 401)

    def test_rejects_bad_signature(self):
        headers = self._sign("GET", "/api/agent/health")
        headers["X-Agent-Signature"] = "00" * 32
        resp = self.client.get("/api/agent/health", headers=headers)
        self.assertEqual(resp.status_code, 401)

    def test_rejects_wrong_method_in_signature(self):
        # Signed for GET but posting.
        headers = self._sign("GET", "/api/agent/authcheck")
        resp = self.client.post(
            "/api/agent/authcheck",
            headers=headers,
            data=b"{}",
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 401)

    def test_disabled_when_secret_missing(self):
        os.environ.pop("AGENT_SHARED_SECRET", None)
        for mod in ("agent_api", "app"):
            if mod in sys.modules:
                del sys.modules[mod]
        import app as _app

        resp = _app.app.test_client().get("/api/agent/health")
        self.assertEqual(resp.status_code, 503)

    def test_authcheck_echoes_body(self):
        body = b'{"hello":"world"}'
        headers = self._sign("POST", "/api/agent/authcheck", body=body)
        resp = self.client.post(
            "/api/agent/authcheck",
            headers=headers,
            data=body,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["echo"], {"hello": "world"})


if __name__ == "__main__":
    unittest.main()

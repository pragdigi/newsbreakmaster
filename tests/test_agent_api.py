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


class AgentLibraryEndpointsTests(unittest.TestCase):
    """Cross-app library read surface used by the metamaster picker.

    We keep one storage dir alive for the entire test class to avoid
    fighting Python's module cache — once ``ai_studio.*`` has imported
    ``storage`` we can't safely swap the module out from under it
    mid-suite. Per-test setup just nukes the seeded jsonl + image files
    and re-creates them.
    """

    @classmethod
    def setUpClass(cls):
        import tempfile

        cls._tmp = tempfile.mkdtemp(prefix="agentlibtest_")
        cls._prev_storage_dir = os.environ.get("NEWSBREAK_STORAGE_DIR", "")
        os.environ["NEWSBREAK_STORAGE_DIR"] = cls._tmp
        os.environ["AGENT_SHARED_SECRET"] = "test-secret-123"
        os.environ["AGENT_PUBLIC_KEY"] = "default"
        os.environ["AGENT_MAX_CLOCK_SKEW"] = "300"
        # Wipe every cached module that touches storage so the new env
        # var actually takes effect.
        for mod in list(sys.modules):
            if mod == "storage" or mod == "agent_api" or mod == "app" or mod.startswith("ai_studio"):
                sys.modules.pop(mod, None)

        import storage as _storage
        _storage.ensure_dirs()
        cls._storage = _storage
        import app as _app
        import agent_api

        cls._app = _app.app
        cls._agent_api = agent_api
        cls._client = _app.app.test_client()
        cls._png_bytes = (
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06"
            b"\x00\x00\x00\x1f\x15\xc4\x89"
            b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )

    @classmethod
    def tearDownClass(cls):
        if cls._prev_storage_dir:
            os.environ["NEWSBREAK_STORAGE_DIR"] = cls._prev_storage_dir
        else:
            os.environ.pop("NEWSBREAK_STORAGE_DIR", None)
        import shutil

        shutil.rmtree(cls._tmp, ignore_errors=True)

    def setUp(self):
        # Reset the seeded library between tests.
        storage = self._storage
        from ai_studio import library as _lib
        for plat in ("smartnews", "newsbreak"):
            try:
                lib_file = storage._library_file(plat)
                if os.path.exists(lib_file):
                    os.remove(lib_file)
            except Exception:
                pass
            img_dir = storage.library_image_dir(plat)
            for fn in os.listdir(img_dir):
                try:
                    os.remove(os.path.join(img_dir, fn))
                except Exception:
                    pass

        sn_row = storage.append_library_item(
            {
                "offer_id": "of-sn",
                "style_name": "Square Stack",
                "style_id": "stack",
                "aspect": "1:1",
                "model": "nano-banana-2",
            },
            platform="smartnews",
        )
        self._sn_filename = f"{sn_row['library_id']}.png"
        with open(
            os.path.join(storage.library_image_dir("smartnews"), self._sn_filename),
            "wb",
        ) as f:
            f.write(self._png_bytes)
        _lib._patch_library_row(
            sn_row["library_id"], {"filename": self._sn_filename}, platform="smartnews"
        )

        nb_row = storage.append_library_item(
            {
                "offer_id": "of-nb",
                "style_name": "Wide Showcase",
                "aspect": "16:9",
            },
            platform="newsbreak",
        )
        self._nb_filename = f"{nb_row['library_id']}.png"
        _lib._patch_library_row(
            nb_row["library_id"], {"filename": self._nb_filename}, platform="newsbreak"
        )
        with open(
            os.path.join(storage.library_image_dir("newsbreak"), self._nb_filename),
            "wb",
        ) as f:
            f.write(self._png_bytes)

        self.client = self._client
        self.agent_api = self._agent_api

    def _sign(self, method, path, body=b""):
        return self.agent_api.build_agent_headers(method=method, path=path, body=body)

    def test_list_library_returns_both_platforms_by_default(self):
        # Sanity: confirm the handler reads from the same storage dir
        # the test seeded into (this catches Python module-caching bugs
        # before they cause confusing list-empty failures).
        import storage as _live_storage
        sn_rows = _live_storage.list_library_items(platform="smartnews")
        self.assertEqual(len(sn_rows), 1, f"seed missed; storage at {_live_storage.STORAGE_ROOT}")
        path = "/api/agent/library"
        headers = self._sign("GET", path)
        resp = self.client.get(path, headers=headers)
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        platforms = {item["platform"] for item in body["items"]}
        self.assertEqual(platforms, {"smartnews", "newsbreak"})
        # Each item must carry an HMAC-protected download_url path.
        for item in body["items"]:
            self.assertTrue(
                item["download_url"].startswith("/api/agent/library/image/")
            )

    def test_list_library_filters_by_platform_aspect_and_offer(self):
        path = "/api/agent/library?platform=smartnews&aspect=1:1&offer_id=of-sn"
        headers = self._sign("GET", "/api/agent/library")
        # Query string is *not* part of the signed canonical message, so the
        # base-path signature is what we send.
        resp = self.client.get(path, headers=headers)
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["items"][0]["platform"], "smartnews")
        self.assertEqual(body["items"][0]["offer_id"], "of-sn")

    def test_list_library_rejects_missing_signature(self):
        resp = self.client.get("/api/agent/library")
        self.assertEqual(resp.status_code, 401)

    def test_image_endpoint_returns_bytes(self):
        path = f"/api/agent/library/image/smartnews/{self._sn_filename}"
        headers = self._sign("GET", path)
        resp = self.client.get(path, headers=headers)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data, self._png_bytes)

    def test_image_endpoint_blocks_path_traversal(self):
        # Sign the *normalised* path because werkzeug collapses ``..``
        # before routing. The app-layer filename guard then has to reject
        # any non-basename segment that survives normalisation.
        path = "/api/agent/library/image/smartnews/..%2Fetc%2Fpasswd"
        normalised = "/api/agent/library/image/etc/passwd"
        headers = self._sign("GET", normalised)
        resp = self.client.get(path, headers=headers)
        # Acceptable safe outcomes: 400 (filename guard tripped),
        # 404 (file truly absent), or 401 (HMAC path mismatch). Anything
        # but 200 means we didn't serve the traversed file.
        self.assertNotEqual(resp.status_code, 200)
        if resp.status_code == 200:
            self.assertNotIn(b"root:", resp.data)

    def test_image_endpoint_rejects_unsigned(self):
        resp = self.client.get(
            f"/api/agent/library/image/smartnews/{self._sn_filename}"
        )
        self.assertEqual(resp.status_code, 401)


if __name__ == "__main__":
    unittest.main()

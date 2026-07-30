"""413 / oversized creative upload handling."""
from __future__ import annotations

import unittest
from unittest import mock


class UploadSizeLimitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app as app_module

        cls.app_module = app_module
        cls.app = app_module.app
        cls.app.config["TESTING"] = True
        # Keep tests fast — 1 MB ceiling instead of the production 500 MB.
        cls._orig_max = app_module.MAX_UPLOAD_BYTES
        app_module.MAX_UPLOAD_BYTES = 1 * 1024 * 1024
        cls.app.config["MAX_CONTENT_LENGTH"] = app_module.MAX_UPLOAD_BYTES

    @classmethod
    def tearDownClass(cls):
        cls.app_module.MAX_UPLOAD_BYTES = cls._orig_max
        cls.app.config["MAX_CONTENT_LENGTH"] = cls._orig_max

    def setUp(self):
        self.client = self.app.test_client()

    def test_oversize_launch_post_redirects_with_query(self):
        # before_request size check runs after basic-auth; bypass auth so we
        # exercise the upload_error redirect rather than a 401.
        with mock.patch.object(self.app_module, "_basic_auth_ok", return_value=True):
            # test_client overwrites Content-Length from the body length;
            # force the header via environ_overrides so we simulate a large
            # video upload without actually sending 2 MB of payload.
            resp = self.client.post(
                "/launch",
                data=b"x" * 64,
                content_type="application/octet-stream",
                environ_overrides={"CONTENT_LENGTH": str(2 * 1024 * 1024)},
            )
        self.assertEqual(resp.status_code, 302)
        loc = resp.headers.get("Location", "")
        self.assertIn("upload_error=too_large", loc)
        self.assertIn("upload_size_mb=2", loc)

    def test_max_upload_constant_matches_config(self):
        self.assertEqual(
            self.app.config["MAX_CONTENT_LENGTH"],
            self.app_module.MAX_UPLOAD_BYTES,
        )


if __name__ == "__main__":
    unittest.main()

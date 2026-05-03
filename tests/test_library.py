"""Unit tests for the AI Studio prebuilt-ad library.

Storage tests run against a temp dir (no fixture network calls), and the
library module is exercised with a mocked ``pipeline.generate_ads`` so the
test never invokes the real image API.
"""
from __future__ import annotations

import base64
import os
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class _TempStorage:
    """Swap ``storage.STORAGE_ROOT`` to a temp path for the duration of a test."""

    def __init__(self):
        self._tmp = None
        self._orig = None

    def __enter__(self):
        import storage
        self._tmp = tempfile.mkdtemp(prefix="libtest_")
        self._orig = {
            "STORAGE_ROOT": storage.STORAGE_ROOT,
            "TOKENS_DIR": storage.TOKENS_DIR,
            "RULES_DIR": storage.RULES_DIR,
            "AUDIT_DIR": storage.AUDIT_DIR,
            "CATALOG_DIR": storage.CATALOG_DIR,
        }
        storage.STORAGE_ROOT = self._tmp
        storage.TOKENS_DIR = os.path.join(self._tmp, "tokens")
        storage.RULES_DIR = os.path.join(self._tmp, "rules")
        storage.AUDIT_DIR = os.path.join(self._tmp, "audit")
        storage.CATALOG_DIR = os.path.join(self._tmp, "catalog")
        storage.ensure_dirs()
        return self._tmp

    def __exit__(self, *exc):
        import shutil
        import storage
        for k, v in self._orig.items():
            setattr(storage, k, v)
        try:
            shutil.rmtree(self._tmp, ignore_errors=True)
        except Exception:
            pass


def _png_b64() -> str:
    """A 1×1 transparent PNG for tests — base64 because that's what
    image_gen returns."""
    return (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lE"
        "QVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    )


class StorageLibraryTests(unittest.TestCase):
    def test_append_and_list_filters_by_offer_and_consumed(self):
        with _TempStorage():
            import storage

            row = storage.append_library_item(
                {"offer_id": "of1", "style_name": "stack"}, platform="newsbreak"
            )
            self.assertTrue(row["library_id"])
            storage.append_library_item({"offer_id": "of2"}, platform="newsbreak")

            of1 = storage.list_library_items(platform="newsbreak", offer_id="of1")
            self.assertEqual(len(of1), 1)
            self.assertEqual(of1[0]["offer_id"], "of1")

            counts = storage.library_counts(platform="newsbreak")
            self.assertEqual(counts.get("of1"), 1)
            self.assertEqual(counts.get("of2"), 1)

    def test_consume_marks_oldest_first(self):
        with _TempStorage():
            import storage

            for _ in range(3):
                storage.append_library_item(
                    {"offer_id": "of1"}, platform="newsbreak"
                )
            popped = storage.consume_library_items(
                "of1", 2, platform="newsbreak"
            )
            self.assertEqual(len(popped), 2)
            for r in popped:
                self.assertIsNotNone(r["consumed_at"])

            remaining = storage.list_library_items(
                platform="newsbreak", offer_id="of1"
            )
            self.assertEqual(len(remaining), 1)
            self.assertIsNone(remaining[0]["consumed_at"])

    def test_consume_skips_other_offers(self):
        with _TempStorage():
            import storage

            storage.append_library_item({"offer_id": "of1"}, platform="newsbreak")
            storage.append_library_item({"offer_id": "of2"}, platform="newsbreak")
            popped = storage.consume_library_items(
                "of1", 5, platform="newsbreak"
            )
            self.assertEqual(len(popped), 1)
            self.assertEqual(popped[0]["offer_id"], "of1")
            # of2 still untouched
            counts = storage.library_counts(platform="newsbreak")
            self.assertEqual(counts.get("of1", 0), 0)
            self.assertEqual(counts.get("of2", 0), 1)


class LibraryTopupTests(unittest.TestCase):
    def test_topup_offer_writes_disk_files_and_appends_rows(self):
        with _TempStorage():
            from ai_studio import library

            fake_batch = {
                "gen_id": "gen-xyz",
                "offer_id": "of1",
                "platform": "newsbreak",
                "aspect": "16:9",
                "allocation": [],
                "prompts": [
                    {"style_id": "s1", "prompt": "p1", "headline": "h1", "concept_source": "llm"},
                    {"style_id": "s2", "prompt": "p2", "headline": "h2", "concept_source": "llm"},
                ],
                "images": [
                    {"style_id": "s1", "style_name": "S1", "b64": _png_b64(), "mime": "image/png", "model": "nano-banana-2", "ms": 1234},
                    {"style_id": "s2", "style_name": "S2", "b64": _png_b64(), "mime": "image/png", "model": "nano-banana-2", "ms": 1234},
                ],
            }
            with mock.patch("ai_studio.library.pipeline.generate_ads", return_value=fake_batch) as mocked:
                res = library.topup_offer(
                    "of1",
                    platform="newsbreak",
                    target=2,
                    model_image="nano-banana-2",
                )
            mocked.assert_called_once()
            self.assertEqual(res["added"], 2)
            self.assertEqual(res["target"], 2)
            self.assertEqual(res["errors"], [])

            # Files exist on disk + rows reference them.
            import storage

            rows = storage.list_library_items(platform="newsbreak", offer_id="of1")
            self.assertEqual(len(rows), 2)
            for r in rows:
                self.assertTrue(r.get("filename"))
                self.assertTrue(os.path.exists(
                    storage.library_image_path(r["filename"], platform="newsbreak")
                ))

    def test_topup_offer_skips_when_already_full(self):
        with _TempStorage():
            from ai_studio import library
            import storage

            for _ in range(2):
                storage.append_library_item({"offer_id": "of1"}, platform="newsbreak")
            with mock.patch("ai_studio.library.pipeline.generate_ads") as mocked:
                res = library.topup_offer(
                    "of1",
                    platform="newsbreak",
                    target=2,
                    model_image="nano-banana-2",
                )
            mocked.assert_not_called()
            self.assertEqual(res["added"], 0)
            self.assertEqual(res["total"], 2)


if __name__ == "__main__":
    unittest.main()

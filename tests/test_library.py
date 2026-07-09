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

    def test_set_library_consumed_by_id_and_back(self):
        with _TempStorage():
            import storage

            a = storage.append_library_item({"offer_id": "of1"}, platform="newsbreak")
            b = storage.append_library_item({"offer_id": "of1"}, platform="newsbreak")

            used_in = {"platform": "newsbreak", "campaign_id": "c-1", "via": "launch_form"}
            updated = storage.set_library_consumed(
                [a["library_id"]], True, platform="newsbreak", used_in=used_in
            )
            self.assertEqual(len(updated), 1)
            self.assertEqual(updated[0]["library_id"], a["library_id"])
            self.assertIsNotNone(updated[0]["consumed_at"])
            self.assertEqual(updated[0]["used_in"]["campaign_id"], "c-1")

            # b untouched; unused listing only shows b.
            remaining = storage.list_library_items(platform="newsbreak", offer_id="of1")
            self.assertEqual([r["library_id"] for r in remaining], [b["library_id"]])

            # Mark back unused — consumed_at + used_in cleared.
            reverted = storage.set_library_consumed(
                [a["library_id"]], False, platform="newsbreak"
            )
            self.assertEqual(len(reverted), 1)
            self.assertIsNone(reverted[0]["consumed_at"])
            self.assertNotIn("used_in", reverted[0])
            self.assertEqual(
                len(storage.list_library_items(platform="newsbreak", offer_id="of1")), 2
            )

            # Unknown ids are a no-op.
            self.assertEqual(
                storage.set_library_consumed(["nope"], True, platform="newsbreak"), []
            )

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


class LibraryExportTests(unittest.TestCase):
    def test_export_csv_and_json_honor_status_filter(self):
        with _TempStorage():
            import app as _app
            import storage

            a = storage.append_library_item(
                {
                    "offer_id": "of1",
                    "style_id": "s1",
                    "style_name": "Stack",
                    "headline": "H1",
                    "prompt": "Full prompt text, with commas",
                    "aspect": "1:1",
                },
                platform="newsbreak",
            )
            storage.append_library_item(
                {"offer_id": "of1", "headline": "H2", "prompt": "P2"},
                platform="newsbreak",
            )
            storage.set_library_consumed(
                [a["library_id"]], True, platform="newsbreak",
                used_in={"platform": "newsbreak", "campaign_id": "c-9", "via": "launch_form"},
            )

            client = _app.app.test_client()
            with mock.patch.object(_app, "_auth_required", return_value=True), \
                 mock.patch.object(_app, "_AI_STUDIO_AVAILABLE", True):
                # Unused only (default) — one row, the not-yet-used H2.
                resp = client.get("/api/studio/library/export?format=csv&status=unused")
                self.assertEqual(resp.status_code, 200)
                self.assertIn("text/csv", resp.content_type)
                body = resp.get_data(as_text=True)
                self.assertIn("P2", body)
                self.assertNotIn("c-9", body)
                self.assertIn("attachment", resp.headers.get("Content-Disposition", ""))

                # All rows as JSON — includes prompt verbatim + used_in trail.
                resp = client.get("/api/studio/library/export?format=json&status=all")
                self.assertEqual(resp.status_code, 200)
                data = resp.get_json()
                self.assertEqual(data["count"], 2)
                by_headline = {r["headline"]: r for r in data["items"]}
                self.assertEqual(
                    by_headline["H1"]["prompt"], "Full prompt text, with commas"
                )
                self.assertEqual(by_headline["H1"]["status"], "used")
                self.assertEqual(by_headline["H1"]["used_campaign_id"], "c-9")
                self.assertEqual(by_headline["H2"]["status"], "unused")

                # Bad format rejected.
                resp = client.get("/api/studio/library/export?format=xml")
                self.assertEqual(resp.status_code, 400)


class CandidateExportTests(unittest.TestCase):
    def test_export_ideas_csv_and_json_honor_filters(self):
        with _TempStorage():
            import app as _app
            import storage

            storage.upsert_style_candidate(
                {
                    "style_id": "value-stack",
                    "name": "Value Stack",
                    "description": "Products stacked with price anchors",
                    "visual_cues": ["stacked boxes", "bold price"],
                    "prompt_template": "Photo of {{headline}} value stack. Square format.",
                    "source": "brainstorm",
                    "source_meta": {"offer_id": "of1"},
                    "status": "candidate",
                },
                platform="newsbreak",
            )
            storage.upsert_style_candidate(
                {
                    "style_id": "archived-idea",
                    "name": "Old Idea",
                    "source": "gethookd",
                    "status": "archived",
                },
                platform="newsbreak",
            )
            # Lifecycle bookkeeping mirror — must never appear in exports.
            storage.upsert_style_candidate(
                {"style_id": "catalog:value-stack", "name": "value-stack", "status": "catalog"},
                platform="newsbreak",
            )

            client = _app.app.test_client()
            with mock.patch.object(_app, "_auth_required", return_value=True), \
                 mock.patch.object(_app, "_AI_STUDIO_AVAILABLE", True), \
                 mock.patch.object(_app, "_active_platform", return_value="newsbreak"):
                # All ideas as JSON — catalog mirror excluded, cues kept as list.
                resp = client.get("/api/studio/research/candidates/export?format=json&status=all")
                self.assertEqual(resp.status_code, 200)
                data = resp.get_json()
                self.assertEqual(data["count"], 2)
                by_id = {r["style_id"]: r for r in data["items"]}
                self.assertNotIn("catalog:value-stack", by_id)
                vs = by_id["value-stack"]
                self.assertEqual(
                    vs["prompt_template"],
                    "Photo of {{headline}} value stack. Square format.",
                )
                self.assertEqual(vs["visual_cues"], ["stacked boxes", "bold price"])
                self.assertEqual(vs["offer_id"], "of1")

                # Status filter — only the archived gethookd idea.
                resp = client.get("/api/studio/research/candidates/export?format=csv&status=archived")
                self.assertEqual(resp.status_code, 200)
                self.assertIn("text/csv", resp.content_type)
                body = resp.get_data(as_text=True)
                self.assertIn("archived-idea", body)
                self.assertNotIn("value-stack", body)

                # Source filter uses the fuzzy chip matching (scrape == gethookd).
                resp = client.get("/api/studio/research/candidates/export?format=json&source=gethookd")
                self.assertEqual(resp.get_json()["count"], 1)

                resp = client.get("/api/studio/research/candidates/export?format=xml")
                self.assertEqual(resp.status_code, 400)


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

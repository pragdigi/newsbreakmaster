"""Regression tests for the 2026-07-12 style_candidates.json wipe.

Covers the three fixes:
  * concurrent read-modify-writes are serialised (no lost/clobbered rows),
  * a corrupt catalog file fails the mutation instead of being treated as
    empty (which used to erase every record on the next save),
  * the /candidates/import endpoint restores an export additively.
"""
from __future__ import annotations

import io
import json
import os
import sys
import threading
import unittest
from unittest import mock

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tests.test_ai_studio import _TempStorage  # noqa: E402


class ConcurrentUpsertTests(unittest.TestCase):
    def test_parallel_upserts_do_not_lose_rows(self):
        with _TempStorage():
            import storage

            n = 40
            errors = []

            def worker(i: int) -> None:
                try:
                    storage.upsert_style_candidate(
                        {
                            "style_id": f"cand-{i}",
                            "name": f"Idea {i}",
                            "description": "x" * 500,
                            "status": "candidate",
                        },
                        platform="newsbreak",
                    )
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(errors, [])
            rows = storage.list_style_candidates(platform="newsbreak")
            self.assertEqual(len(rows), n)
            self.assertEqual(
                {r["style_id"] for r in rows}, {f"cand-{i}" for i in range(n)}
            )

    def test_corrupt_file_fails_mutation_instead_of_wiping(self):
        with _TempStorage():
            import storage

            storage.upsert_style_candidate(
                {"style_id": "keep-me", "name": "Keeper"}, platform="newsbreak"
            )
            path = storage._style_candidates_file("newsbreak")
            with open(path, "a", encoding="utf-8") as f:
                f.write("{{{ definitely not json")

            with self.assertRaises(Exception):
                storage.upsert_style_candidate(
                    {"style_id": "new-one", "name": "New"}, platform="newsbreak"
                )
            # The corrupt file was left alone — not replaced by a 1-item list.
            with open(path, "r", encoding="utf-8") as f:
                self.assertIn("keep-me", f.read())

    def test_write_json_is_atomic_replace(self):
        with _TempStorage():
            import storage

            path = storage._style_candidates_file("newsbreak")
            storage._write_json(path, [{"a": 1}])
            storage._write_json(path, [{"a": 2}])
            with open(path, "r", encoding="utf-8") as f:
                self.assertEqual(json.load(f), [{"a": 2}])
            # No stray temp files left behind.
            leftovers = [
                x for x in os.listdir(os.path.dirname(path)) if x.endswith(".tmp")
            ]
            self.assertEqual(leftovers, [])


class CandidateImportTests(unittest.TestCase):
    def _client(self):
        import app as _app

        return _app.app.test_client(), _app

    def test_import_json_fills_stub_and_inserts_new(self):
        with _TempStorage():
            import storage

            # A stub as left behind by the wipe: promoted status but no content.
            storage.upsert_style_candidate(
                {"style_id": "stub-1", "status": "promoted"}, platform="newsbreak"
            )
            payload = {
                "items": [
                    {
                        "style_id": "stub-1",
                        "platform": "newsbreak",
                        "name": "Recovered Idea",
                        "description": "Restored from export",
                        "visual_cues": ["cue a", "cue b"],
                        "prompt_template": "Do the thing. Square format.",
                        "source": "brainstorm",
                        "status": "candidate",
                    },
                    {
                        "style_id": "brand-new",
                        "platform": "newsbreak",
                        "name": "Fresh",
                        "status": "candidate",
                        "offer_id": "of1",
                    },
                    {"style_id": "catalog:ignored", "name": "nope"},
                ]
            }
            client, _app = self._client()
            with mock.patch.object(_app, "_auth_required", return_value=True), \
                 mock.patch.object(_app, "_AI_STUDIO_AVAILABLE", True), \
                 mock.patch.object(_app, "_active_platform", return_value="newsbreak"):
                resp = client.post(
                    "/api/studio/research/candidates/import", json=payload
                )
            self.assertEqual(resp.status_code, 200)
            body = resp.get_json()
            self.assertEqual(body["inserted"], 1)
            self.assertEqual(body["updated"], 1)
            self.assertEqual(body["skipped"], 1)

            rows = {
                r["style_id"]: r
                for r in storage.list_style_candidates(platform="newsbreak")
            }
            self.assertNotIn("catalog:ignored", rows)
            stub = rows["stub-1"]
            self.assertEqual(stub["name"], "Recovered Idea")
            self.assertEqual(stub["visual_cues"], ["cue a", "cue b"])
            # The lifecycle stamp on the live record wins over the import.
            self.assertEqual(stub["status"], "promoted")
            fresh = rows["brand-new"]
            self.assertEqual(fresh["status"], "candidate")
            self.assertEqual(fresh["source_meta"], {"offer_id": "of1"})

    def test_import_csv_export_roundtrip(self):
        with _TempStorage():
            import storage

            storage.upsert_style_candidate(
                {"style_id": "stub-2", "status": "promoted"}, platform="newsbreak"
            )
            csv_text = (
                "style_id,platform,name,description,visual_cues,prompt_template,source,status\n"
                'stub-2,newsbreak,CSV Idea,Desc here,cue x | cue y,Template. Square format.,gethookd,candidate\n'
            )
            client, _app = self._client()
            with mock.patch.object(_app, "_auth_required", return_value=True), \
                 mock.patch.object(_app, "_AI_STUDIO_AVAILABLE", True), \
                 mock.patch.object(_app, "_active_platform", return_value="newsbreak"):
                resp = client.post(
                    "/api/studio/research/candidates/import",
                    data={"file": (io.BytesIO(csv_text.encode("utf-8")), "ideas.csv")},
                    content_type="multipart/form-data",
                )
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.get_json()["updated"], 1)
            row = next(
                r
                for r in storage.list_style_candidates(platform="newsbreak")
                if r["style_id"] == "stub-2"
            )
            self.assertEqual(row["name"], "CSV Idea")
            self.assertEqual(row["visual_cues"], ["cue x", "cue y"])
            self.assertEqual(row["status"], "promoted")

    def test_import_rejects_empty(self):
        with _TempStorage():
            client, _app = self._client()
            with mock.patch.object(_app, "_auth_required", return_value=True), \
                 mock.patch.object(_app, "_AI_STUDIO_AVAILABLE", True), \
                 mock.patch.object(_app, "_active_platform", return_value="newsbreak"):
                resp = client.post(
                    "/api/studio/research/candidates/import", json={"items": []}
                )
            self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main()

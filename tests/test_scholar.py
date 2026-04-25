"""Tests for the Copywriting Scholar (Agent #3).

Mocks the LLM transports so we can validate:
  - Lens picker rotates fresh lenses first
  - LLM JSON output is parsed into style candidates
  - Each candidate is persisted with source="scholar" + lens_id
  - study_all loops every offer when scan_all_offers=True
"""
from __future__ import annotations

import json
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
    def __init__(self):
        self._tmp = None
        self._orig = None

    def __enter__(self):
        import storage
        self._tmp = tempfile.mkdtemp(prefix="scholartest_")
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
        import storage
        import shutil
        for k, v in self._orig.items():
            setattr(storage, k, v)
        try:
            shutil.rmtree(self._tmp, ignore_errors=True)
        except Exception:
            pass


_SAMPLE_LLM_JSON = json.dumps(
    [
        {
            "name": "schwartz_unaware_specific_moment",
            "description": "Lands on Unaware prospects via a specific 3am moment.",
            "visual_cues": [
                "candid kitchen photo at night",
                "warm tungsten light",
                "small typewriter-style headline",
            ],
            "prompt_template": (
                "A photo for {{headline}} with {{cta_label}}. Square format."
            ),
            "framework_note": "Targets Schwartz awareness level 5 (Unaware).",
            "copy_seed": "I used to lie awake at 3am, counting ceiling tiles.",
        },
        {
            "name": "schwartz_problem_aware_proof",
            "description": "Specific-moment headline + proof object.",
            "visual_cues": [
                "single proof object centred",
                "off-white linen background",
                "serif headline above object",
            ],
            "prompt_template": (
                "Editorial photo: {{headline}} with {{cta_label}}. Square format."
            ),
            "framework_note": "Targets Schwartz awareness level 4 (Problem Aware).",
            "copy_seed": "Most tinnitus advice is wrong. Here's the proof.",
        },
    ]
)


class LensPickerTests(unittest.TestCase):
    def test_picks_fresh_lens_when_available(self):
        from ai_studio.research import scholar

        # Stub recent lens lookup so most lenses are "used", forcing the
        # picker into the "fresh" branch.
        used_ids = [l.id for l in scholar.LENSES[:-1]]  # all except the last
        with mock.patch.object(
            scholar, "_recent_lens_ids_for_offer", return_value=used_ids
        ):
            import random as _random

            lens = scholar._pick_lens(
                "newsbreak", "of1", rng=_random.Random(0)
            )
        self.assertEqual(lens.id, scholar.LENSES[-1].id)

    def test_falls_back_to_full_pool_when_all_used(self):
        from ai_studio.research import scholar

        used_ids = [l.id for l in scholar.LENSES]
        with mock.patch.object(
            scholar, "_recent_lens_ids_for_offer", return_value=used_ids
        ):
            import random as _random

            lens = scholar._pick_lens(
                "newsbreak", "of1", rng=_random.Random(0)
            )
        self.assertIn(lens.id, used_ids)


class StudyOfferTests(unittest.TestCase):
    def _seed_offer(self, platform="newsbreak", offer_id="of-1"):
        import storage
        offer = {
            "id": offer_id,
            "name": "Tinnito",
            "brand_name": "Xeviola",
            "headline": "A surprising tinnitus discovery",
            "body": "Doctors don't expect this simple ritual.",
            "categories": ["health", "tinnitus"],
        }
        storage.upsert_offer(offer, platform=platform)
        return offer

    def test_emits_candidates_with_lens_metadata(self):
        from ai_studio.research import scholar

        with _TempStorage():
            offer = self._seed_offer()
            with mock.patch.object(scholar, "_call_claude", return_value=_SAMPLE_LLM_JSON), \
                 mock.patch.object(scholar, "_call_gemini", return_value=""):
                emitted, lens = scholar.study_offer(
                    offer, platform="newsbreak", count=2,
                    lens_id="schwartz_awareness",
                )
        self.assertEqual(lens.id, "schwartz_awareness")
        self.assertEqual(len(emitted), 2)
        for c in emitted:
            self.assertEqual(c["source"], "scholar")
            self.assertEqual(c["source_meta"]["lens_id"], "schwartz_awareness")
            self.assertTrue(c["prompt_template"].endswith("Square format."))

    def test_falls_back_to_gemini_when_claude_empty(self):
        from ai_studio.research import scholar

        with _TempStorage():
            offer = self._seed_offer()
            with mock.patch.object(scholar, "_call_claude", return_value=""), \
                 mock.patch.object(scholar, "_call_gemini", return_value=_SAMPLE_LLM_JSON):
                emitted, _lens = scholar.study_offer(
                    offer, platform="newsbreak", count=2,
                )
        self.assertEqual(len(emitted), 2)

    def test_empty_when_both_models_silent(self):
        from ai_studio.research import scholar

        with _TempStorage():
            offer = self._seed_offer()
            with mock.patch.object(scholar, "_call_claude", return_value=""), \
                 mock.patch.object(scholar, "_call_gemini", return_value=""):
                emitted, _lens = scholar.study_offer(
                    offer, platform="newsbreak", count=2,
                )
        self.assertEqual(emitted, [])

    def test_study_all_iterates_offers(self):
        from ai_studio.research import scholar

        with _TempStorage():
            self._seed_offer(offer_id="of-1")
            self._seed_offer(offer_id="of-2")
            with mock.patch.object(scholar, "_call_claude", return_value=_SAMPLE_LLM_JSON), \
                 mock.patch.object(scholar, "_call_gemini", return_value=""):
                out = scholar.study_all(
                    platform="newsbreak", scan_all_offers=True, count_per_offer=2,
                )
            # 2 offers × 2 candidates each = 4 emitted total
            self.assertEqual(len(out["scholar"]), 4)


if __name__ == "__main__":
    unittest.main()

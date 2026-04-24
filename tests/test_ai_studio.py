"""Smoke tests for the AI Ad Studio.

Intentionally light: no image renders, no network calls. We use a temporary
``storage`` dir, a mock adapter, and assert on the shape of the data each
module emits. Run with ``python -m pytest tests/test_ai_studio.py``.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import date, timedelta
from unittest import mock

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class _MockAdapter:
    """Minimal adapter stub used by winners/tests."""

    platform = "newsbreak"

    def __init__(self, rows=None, ads=None, accounts=None):
        self._rows = rows or []
        self._ads = ads or {}
        self._accounts = accounts or [{"id": "acc-1", "name": "Acc 1"}]

    def get_accounts(self):
        return self._accounts

    def fetch_report_rows(self, account_id, level, start, end):
        return list(self._rows)

    def get_ads(self, account_id, ad_set_id):
        return list(self._ads.get(ad_set_id, []))


class _TempStorage:
    """Swap ``storage.STORAGE_DIR`` to a temp path for the duration of a test."""

    def __init__(self):
        self._tmp = None
        self._orig = None

    def __enter__(self):
        import storage
        self._tmp = tempfile.mkdtemp(prefix="studiotest_")
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


class PromptGenTests(unittest.TestCase):
    def test_emits_ten_prompts_each_ending_square_format(self):
        from ai_studio import prompt_gen

        offer = {
            "id": "of1",
            "name": "Tinnito",
            "brand_name": "Xeviola",
            "cta": "Learn More",
            "headline": "A surprising tinnitus discovery",
            "body": "Doctors don't want you to know this simple ritual.",
        }
        prompts = prompt_gen.generate_prompts(offer, count=10, seed=42)
        self.assertEqual(len(prompts), 10)
        for p in prompts:
            self.assertIn("prompt", p)
            self.assertTrue(
                p["prompt"].rstrip().endswith("Square format."),
                msg=f"missing suffix: {p['prompt'][-60:]!r}",
            )
            self.assertIn(p["style_id"], [s.id for s in prompt_gen.STYLE_CATALOG])

    def test_uses_insights_suggested_angles(self):
        from ai_studio import prompt_gen

        offer = {"id": "of1", "name": "Tinnito", "headline": "fallback"}
        insights = {
            "suggested_angles": ["Angle one", "Angle two"],
            "top_hooks": ["Hook A"],
            "mechanisms": ["Mech X"],
        }
        prompts = prompt_gen.generate_prompts(
            offer, insights, count=2, style_mix=["product_showcase", "ugc_selfie"], seed=1
        )
        all_text = " ".join(p["prompt"] for p in prompts)
        self.assertIn("Angle one", all_text)


class BanditTests(unittest.TestCase):
    def test_allocates_n_slots(self):
        from ai_studio.research import bandit

        picks = bandit.allocate(
            catalog_styles=["a", "b", "c"],
            candidates=[],
            n=5,
            seed=1,
        )
        self.assertEqual(len(picks), 5)
        self.assertTrue(all(p["style_id"] in {"a", "b", "c"} for p in picks))
        self.assertTrue(all(p["is_candidate"] is False for p in picks))

    def test_candidate_budget_cap_respected(self):
        from ai_studio.research import bandit

        cands = [
            {"style_id": "cand1", "status": "testing", "trials": 0, "wins": 0},
            {"style_id": "cand2", "status": "candidate", "trials": 0, "wins": 0},
        ]
        picks = bandit.allocate(
            catalog_styles=["a", "b", "c", "d"],
            candidates=cands,
            n=10,
            research_ratio=0.3,
            seed=7,
        )
        cand_count = sum(1 for p in picks if p["is_candidate"])
        self.assertLessEqual(cand_count, 3)

    def test_zero_candidates_ignores_research_ratio(self):
        from ai_studio.research import bandit

        picks = bandit.allocate(
            catalog_styles=["a", "b"],
            candidates=[],
            n=4,
            research_ratio=1.0,
            seed=1,
        )
        self.assertEqual(len(picks), 4)
        self.assertTrue(all(not p["is_candidate"] for p in picks))


class ImageGenFallbackTests(unittest.TestCase):
    def test_returns_error_when_no_keys(self):
        from ai_studio import image_gen

        with mock.patch.dict(os.environ, {
            "GEMINI_API_KEY": "",
            "GOOGLE_GENAI_API_KEY": "",
            "OPENAI_API_KEY": "",
        }, clear=False):
            out = image_gen.render_batch(
                [{"prompt": "test", "style_id": "x"}],
                fallback=True,
            )
        self.assertEqual(len(out), 1)
        self.assertIsNone(out[0].get("b64"))
        self.assertIn("error", out[0])


class WinnersTests(unittest.TestCase):
    def test_refresh_with_mock_adapter(self):
        with _TempStorage():
            import storage as _st
            from ai_studio import winners

            _st.upsert_offer(
                {
                    "id": "offer-1",
                    "name": "Tinnito",
                    "landing_url": "https://xeviola.com/pages/tinnito-lander",
                    "target_cpa": 30.0,
                    "pixel_id": "pix-1",
                },
                platform="newsbreak",
            )

            rows = [
                {
                    "ad_id": "ad-win",
                    "ad_set_id": "set-1",
                    "spend": 100.0,
                    "conversions": 5,
                    "cpa": 20.0,
                    "landing_page_url": "https://xeviola.com/pages/tinnito-lander",
                    "name": "Ad 001",
                    "ctr": 0.02,
                    "impressions": 5000,
                    "clicks": 100,
                },
                {
                    "ad_id": "ad-lose",
                    "ad_set_id": "set-1",
                    "spend": 200.0,
                    "conversions": 2,
                    "cpa": 100.0,
                    "landing_page_url": "https://xeviola.com/pages/tinnito-lander",
                    "name": "Ad 002",
                },
            ]
            adapter = _MockAdapter(rows=rows)
            summary = winners.refresh_winners(adapter, platform="newsbreak")
            self.assertEqual(summary["winners_found"], 1)
            self.assertEqual(summary["accounts_scanned"], 1)
            stored = _st.list_winners(platform="newsbreak")
            self.assertEqual(len(stored), 1)
            self.assertEqual(stored[0]["ad_id"], "ad-win")


class DiscoverTests(unittest.TestCase):
    def test_cluster_winners_without_winners_is_noop(self):
        with _TempStorage():
            from ai_studio.research import discover
            out = discover.discover_from_winners("newsbreak")
            self.assertEqual(out, [])

    def test_normalize_gethookd_ad_matches_schema(self):
        from ai_studio.research.discover import normalize_gethookd_ad
        raw = {
            "id": "hk1",
            "title": "Win big",
            "body": "body",
            "performance_score_title": "Very High",
            "media": [{"url": "https://x/y.png", "type": "image"}],
            "brand": {"name": "X", "active_ads": 12},
        }
        n = normalize_gethookd_ad(raw)
        self.assertEqual(n["id"], "hk1")
        self.assertEqual(n["performance_score_title"], "Very High")
        self.assertEqual(n["brand"]["active_ads"], 12)
        self.assertEqual(n["media"][0]["url"], "https://x/y.png")


class PipelineTests(unittest.TestCase):
    def test_generate_skips_render(self):
        with _TempStorage():
            import storage as _st
            from ai_studio import pipeline

            _st.upsert_offer(
                {
                    "id": "offer-p",
                    "name": "Tinnito",
                    "brand_name": "Xeviola",
                    "cta": "Learn More",
                    "headline": "A surprising thing",
                    "body": "A surprising body",
                },
                platform="newsbreak",
            )
            res = pipeline.generate_ads(
                "offer-p",
                platform="newsbreak",
                count=5,
                render=False,
            )
            self.assertEqual(len(res["prompts"]), 5)
            self.assertEqual(res["images"], [])
            gens = _st.list_generations(platform="newsbreak")
            self.assertEqual(len(gens), 1)
            self.assertEqual(gens[0]["offer_id"], "offer-p")


if __name__ == "__main__":
    unittest.main()

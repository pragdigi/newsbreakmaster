"""Tests for the public-libraries scout (Agent #2).

Covers the source scrapers (mocked HTTP) and the orchestrator
``discover_from_public`` (mocked LLM clustering).
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
        self._tmp = tempfile.mkdtemp(prefix="publictest_")
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


class _FakeResponse:
    def __init__(self, status_code=200, text="", payload=None):
        self.status_code = status_code
        self.text = text
        self._payload = payload

    def json(self):
        if self._payload is not None:
            return self._payload
        return json.loads(self.text)


# ----------------------------------------------------------------------
# Meta Ad Library — HTML scraper
# ----------------------------------------------------------------------


class MetaAdLibraryTests(unittest.TestCase):
    def test_returns_empty_on_non_200(self):
        from ai_studio.research.sources import meta_ad_library

        with mock.patch.object(meta_ad_library, "_session") as m:
            sess = mock.MagicMock()
            sess.get.return_value = _FakeResponse(status_code=403, text="forbidden")
            m.return_value = sess
            rows = meta_ad_library.fetch("tinnitus", limit=5)
        self.assertEqual(rows, [])

    def test_parses_inline_card_blob(self):
        from ai_studio.research.sources import meta_ad_library

        # Synthetic minimal HTML containing one card-shaped JSON object.
        card = {
            "ad_archive_id": "abc123",
            "page_name": "Acme Health",
            "title": {"text": "A Surprising Tinnitus Discovery"},
            "body": {"text": "Doctors didn't expect this simple ritual."},
            "link_url": "https://acme.example/tinnitus",
            "original_image_url": "https://scontent.example/img1.jpg",
            "creation_time": 1_700_000_000,
        }
        # The regex looks for the page_name pattern; build the embed.
        html = (
            "<html><script>"
            + json.dumps(card)
            + "</script></html>"
        )
        with mock.patch.object(meta_ad_library, "_fetch_html", return_value=html):
            rows = meta_ad_library.fetch("tinnitus", limit=5)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"], "meta")
        self.assertEqual(rows[0]["advertiser"], "Acme Health")
        self.assertIn("https://scontent.example/img1.jpg", rows[0]["image_urls"])
        self.assertEqual(rows[0]["headline"], "A Surprising Tinnitus Discovery")

    def test_empty_query_short_circuits(self):
        from ai_studio.research.sources import meta_ad_library

        rows = meta_ad_library.fetch("   ", limit=5)
        self.assertEqual(rows, [])


# ----------------------------------------------------------------------
# TikTok Creative Center
# ----------------------------------------------------------------------


class TikTokCreativeTests(unittest.TestCase):
    def test_normalises_payload(self):
        from ai_studio.research.sources import tiktok_creative

        payload = {
            "data": {
                "materials": [
                    {
                        "id": "tt-001",
                        "brand_name": "FitFix",
                        "ad_title": "Morning Ritual",
                        "ad_desc": "Start your day with a 60s morning routine.",
                        "cover_url": "https://tt.example/cover.jpg",
                        "industry": "wellness",
                        "ctr": 3.2,
                    },
                    {
                        # Missing id but should still produce a record.
                        "advertiser_name": "PaceLab",
                        "title": "Run smarter",
                    },
                ]
            }
        }
        with mock.patch.object(tiktok_creative, "_session") as m:
            sess = mock.MagicMock()
            sess.get.return_value = _FakeResponse(status_code=200, payload=payload)
            m.return_value = sess
            rows = tiktok_creative.fetch("fitness", limit=5)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["source"], "tiktok")
        self.assertEqual(rows[0]["id"], "tt:tt-001")
        self.assertEqual(rows[0]["headline"], "Morning Ritual")
        self.assertIn("https://tt.example/cover.jpg", rows[0]["image_urls"])
        self.assertTrue(rows[1]["id"].startswith("tt:"))

    def test_returns_empty_on_403(self):
        from ai_studio.research.sources import tiktok_creative

        with mock.patch.object(tiktok_creative, "_session") as m:
            sess = mock.MagicMock()
            sess.get.return_value = _FakeResponse(status_code=403, text="Forbidden")
            m.return_value = sess
            rows = tiktok_creative.fetch("fitness", limit=5)
        self.assertEqual(rows, [])


# ----------------------------------------------------------------------
# discover_public orchestrator
# ----------------------------------------------------------------------


class DiscoverPublicTests(unittest.TestCase):
    def _seed_offer(self, platform="newsbreak"):
        import storage
        offer = {
            "id": "of-test-1",
            "name": "Tinnito",
            "brand_name": "Xeviola",
            "headline": "A new tinnitus discovery",
            "body": "Try this once-a-day ritual.",
        }
        storage.upsert_offer(offer, platform=platform)
        return offer

    def test_clusters_meta_and_tiktok_into_candidates(self):
        from ai_studio.research import discover_public

        sample_meta = [
            {
                "id": "meta:1", "source": "meta", "advertiser": "BrandA",
                "headline": "Headline A", "body": "Body A",
                "image_urls": ["https://x/1.jpg"], "video_url": None,
                "landing_url": "https://a/", "started_at": None, "raw": {},
            },
            {
                "id": "meta:2", "source": "meta", "advertiser": "BrandB",
                "headline": "Headline B", "body": "Body B",
                "image_urls": ["https://x/2.jpg"], "video_url": None,
                "landing_url": "https://b/", "started_at": None, "raw": {},
            },
        ]
        sample_tiktok = [
            {
                "id": "tt:1", "source": "tiktok", "advertiser": "BrandC",
                "headline": "Headline C", "body": "Body C",
                "image_urls": ["https://x/3.jpg"], "video_url": None,
                "landing_url": None, "started_at": None, "raw": {},
            }
        ]
        cluster_payload = json.dumps(
            {
                "classifications": [],
                "clusters": [
                    {
                        "name": "story_card_proof",
                        "description": "Story-card proof layout.",
                        "visual_cues": ["candid photo", "small advertiser logo"],
                        "prompt_template": (
                            "A photo of {{headline}} with {{cta_label}}. Square format."
                        ),
                        "ad_indices": [0, 1, 2],
                        "anchor_platform": "mixed",
                    }
                ],
            }
        )

        with _TempStorage():
            with mock.patch.object(
                discover_public.meta_ad_library, "fetch_many", return_value=sample_meta
            ), mock.patch.object(
                discover_public.tiktok_creative, "fetch_many", return_value=sample_tiktok
            ), mock.patch.object(
                discover_public._disc, "_call_gemini_text", return_value=cluster_payload
            ), mock.patch.object(
                discover_public._disc, "_call_claude_text", return_value=""
            ):
                emitted = discover_public.discover_from_public(
                    platform="newsbreak",
                    keywords=["tinnitus"],
                    limit_per_query=10,
                    min_size=2,
                )
            self.assertEqual(len(emitted), 1)
            cand = emitted[0]
            self.assertEqual(cand["source"], "public_scout")
            self.assertEqual(cand["source_meta"]["anchor_platform"], "mixed")
            self.assertGreater(len(cand["source_meta"]["thumbnails"]), 0)
            self.assertIn("BrandA", cand["source_meta"]["advertisers"])

    def test_returns_empty_when_all_sources_yield_nothing(self):
        from ai_studio.research import discover_public

        with _TempStorage():
            with mock.patch.object(
                discover_public.meta_ad_library, "fetch_many", return_value=[]
            ), mock.patch.object(
                discover_public.tiktok_creative, "fetch_many", return_value=[]
            ):
                emitted = discover_public.discover_from_public(
                    platform="newsbreak",
                    keywords=["tinnitus"],
                )
        self.assertEqual(emitted, [])


if __name__ == "__main__":
    unittest.main()

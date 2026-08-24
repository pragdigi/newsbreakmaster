"""Unit tests for the MediaGo native integration.

No live API calls — HTTP and adapters are stubbed.
"""
from __future__ import annotations

import io
import os
import sys
import unittest
from unittest.mock import patch

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _png_bytes(w: int = 800, h: int = 800, color=(40, 120, 80)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


class _UploadStub:
    def __init__(self, data: bytes, filename: str):
        self._data = data
        self.filename = filename

    def read(self):
        return self._data


class MediaGoApiHelpersTest(unittest.TestCase):
    def test_basic_authorization(self):
        from mediago_api import basic_authorization

        self.assertEqual(basic_authorization("YWJj"), "Basic YWJj")
        self.assertEqual(basic_authorization("Basic YWJj"), "Basic YWJj")
        self.assertEqual(basic_authorization("basic YWJj"), "basic YWJj")

    def test_unwrap_list_nested_and_flat(self):
        from mediago_api import unwrap_list

        self.assertEqual(unwrap_list([{"id": "1"}]), [{"id": "1"}])
        self.assertEqual(unwrap_list({"results": [{"id": "2"}]}), [{"id": "2"}])
        self.assertEqual(
            unwrap_list({"data": {"accounts": [{"account_id": "9"}]}}),
            [{"account_id": "9"}],
        )
        self.assertEqual(unwrap_list({"nope": 1}), [])

    def test_authenticate_caches_token(self):
        from mediago_api import MediaGoClient

        c = MediaGoClient("tok")
        calls = {"n": 0}

        class _Resp:
            status_code = 200
            text = '{"access_token":"abc","expires_in":3600,"client_id":"146"}'

            def json(self):
                return {"access_token": "abc", "expires_in": 3600, "client_id": "146"}

        def fake_post(*_a, **_k):
            calls["n"] += 1
            return _Resp()

        c._session.post = fake_post  # type: ignore
        t1 = c.authenticate()
        t2 = c.authenticate()
        self.assertEqual(t1, "abc")
        self.assertEqual(t2, "abc")
        self.assertEqual(calls["n"], 1)
        self.assertEqual(c.resolved_auth_level, "client")

    def test_get_accounts_client_level(self):
        from mediago_api import MediaGoClient

        c = MediaGoClient("tok")
        c._resolved_level = "client"
        c._access_token = "abc"
        c._token_expires_at = 1e18
        c.get = lambda path, params=None, account_id=None: {  # type: ignore
            "data": {"accounts": [{"account_id": "1", "account_name": "A"}]}
        }
        rows = c.get_accounts()
        self.assertEqual(rows[0]["account_id"], "1")


class MediaGoScoringTest(unittest.TestCase):
    def test_bottom_quartile_and_no_conv(self):
        from platforms.mediago import score_source_rows

        rows = [
            {"site_id": "1", "site_name": "good.com", "spend": 100, "click": 50, "conversion": 10, "impression": 1000},
            {"site_id": "2", "site_name": "ok.com", "spend": 80, "click": 40, "conversion": 4, "impression": 800},
            {"site_id": "3", "site_name": "mid.com", "spend": 60, "click": 30, "conversion": 2, "impression": 600},
            {"site_id": "4", "site_name": "bad.com", "spend": 40, "click": 20, "conversion": 1, "impression": 400},
            {"site_id": "5", "site_name": "dead.com", "spend": 30, "click": 15, "conversion": 0, "impression": 300},
        ]
        scored = score_source_rows(rows, min_spend=1)
        by_id = {r["site_id"]: r for r in scored}
        self.assertEqual(by_id["5"]["weight"], 0.0)
        self.assertIn("no_conv", by_id["5"]["flag"])
        self.assertGreater(by_id["1"]["weight"], by_id["4"]["weight"])
        flagged_bottom = [r for r in scored if "bottom_quartile" in (r["flag"] or "")]
        self.assertGreaterEqual(len(flagged_bottom), 1)

    def test_aggregates_same_site_across_days(self):
        from platforms.mediago import score_source_rows

        rows = [
            {"site_id": "9", "site_name": "msn.com", "spend": 10, "click": 5, "conversion": 1, "impression": 100},
            {"site_id": "9", "site_name": "msn.com", "spend": 10, "click": 5, "conversion": 1, "impression": 100},
        ]
        scored = score_source_rows(rows)
        self.assertEqual(len(scored), 1)
        self.assertEqual(scored[0]["spend"], 20)
        self.assertEqual(scored[0]["conversions"], 2)


class MediaGoAdapterTest(unittest.TestCase):
    def test_flags_and_no_ad_sets(self):
        from platforms.mediago import MediaGoAdapter

        class _C:
            pass

        a = MediaGoAdapter(_C())
        self.assertEqual(a.platform, "mediago")
        self.assertFalse(a.supports_ad_set_scope)
        self.assertEqual(a.get_ad_groups("1", "2"), [])

    def test_normalize_campaign(self):
        from platforms.mediago import MediaGoAdapter

        n = MediaGoAdapter._normalize_campaign(
            {"campaign_id": "c1", "campaign_name": "Native", "status": 1, "daily_cap": 50},
            "acc",
        )
        self.assertEqual(n["id"], "c1")
        self.assertEqual(n["status"], "on")
        self.assertEqual(n["daily_budget_cents"], 5000)

    def test_canonicalize_report(self):
        from platforms.mediago import MediaGoAdapter

        a = MediaGoAdapter(object())  # type: ignore
        row = a._canonicalize_report_row(
            {"id": "c1", "name": "Camp", "spend": 20, "click": 10, "impression": 100, "conversion": 2, "cv_purchase": 2, "status": 1},
            "campaign",
        )
        self.assertEqual(row["scope"], "campaign")
        self.assertEqual(row["clicks"], 10)
        self.assertEqual(row["conversions"], 2)
        self.assertAlmostEqual(row["cpa"], 10.0)
        self.assertEqual(row["status"], "on")

    def test_block_sites_chunks_and_skips_zero(self):
        from platforms.mediago import MediaGoAdapter

        class _C:
            def __init__(self):
                self.calls = []

            def block_account_sites(self, account_id, sites, *, block=True):
                self.calls.append((account_id, list(sites), block))
                return {"ok": True}

        c = _C()
        a = MediaGoAdapter(c)
        sites = [{"site_id": 0, "domain_name": "skip.me"}] + [
            {"site_id": i, "domain_name": f"s{i}.com"} for i in range(1, 105)
        ]
        a.block_sites("acc", sites)
        self.assertEqual(len(c.calls), 2)
        self.assertEqual(len(c.calls[0][1]), 100)
        self.assertEqual(len(c.calls[1][1]), 4)


class MediaGoLauncherTest(unittest.TestCase):
    def test_build_payload_native_only(self):
        from bulk_launcher_mediago import build_campaign_payload

        payload = build_campaign_payload(
            {
                "campaign_name": "Test",
                "brand_name": "BrandX",
                "landing_page": "https://example.com",
                "daily_cap_usd": "40",
                "charge_type": "smart_bid",
                "objective": "conversions",
                "cpc_usd": "0.40",
            },
            [{"asset_name": "a1", "img": "https://img/1.jpg", "headline": "Hello world"}],
        )
        self.assertEqual(payload["creative_type"], "native")
        self.assertEqual(payload["status"], 0)
        self.assertEqual(payload["daily_cap"], 40.0)
        self.assertEqual(payload["ad"][0]["headline"], "Hello world")
        self.assertEqual(len(payload["day_parting"]), 7)
        self.assertEqual(len(payload["day_parting"][0]), 24)

    def test_build_payload_rejects_low_daily(self):
        from bulk_launcher_mediago import build_campaign_payload

        with self.assertRaises(ValueError):
            build_campaign_payload(
                {
                    "campaign_name": "X",
                    "brand_name": "B",
                    "landing_page": "https://x.com",
                    "daily_cap_usd": "5",
                },
                [],
            )

    def test_launch_creates_then_applies_exclusions(self):
        from bulk_launcher_mediago import mediago_bulk_launch

        class _Adapter:
            platform = "mediago"

            def __init__(self):
                self.created = []
                self.blocked = []

            def create_campaign(self, account_id, payload):
                self.created.append((account_id, payload))
                return {"campaign_id": "999"}

            def block_sites(self, account_id, sites, *, campaign_id=None, block=True):
                self.blocked.append((account_id, campaign_id, list(sites), block))
                return [{"ok": True}]

        adapter = _Adapter()
        form = {
            "account_id": "acc1",
            "campaign_name": "Native test",
            "brand_name": "Brand",
            "landing_page": "https://offer.example",
            "daily_cap_usd": "25",
            "cpc_usd": "0.50",
            "headline_0": "A native headline that converts",
            "apply_site_exclusions": "1",
        }
        files = {"creative_0": _UploadStub(_png_bytes(), "ad.png")}

        def host(img, name):
            return f"https://host.test/{name}"

        def builder(f, *, fmt="1.91:1"):
            return b"jpeg", "ad_1200x628.jpg"

        res = mediago_bulk_launch(
            adapter,
            form=form,
            files=files,
            host_image=host,
            creative_builder=builder,
            exclusions=[{"site_id": 2007904, "domain_name": "msn.com"}],
        )
        self.assertTrue(res["ok"])
        self.assertEqual(res["campaign_id"], "999")
        self.assertEqual(adapter.created[0][1]["creative_type"], "native")
        self.assertEqual(adapter.blocked[0][1], "999")

    def test_prepare_native_square(self):
        from bulk_launcher_mediago import prepare_native_creative
        from PIL import Image

        out, name = prepare_native_creative(_UploadStub(_png_bytes(), "sq.png"), fmt="1:1")
        im = Image.open(io.BytesIO(out))
        self.assertEqual(im.size, (1200, 1200))
        self.assertTrue(name.endswith("1200x1200.jpg"))

    def test_prepare_native_landscape_fallback(self):
        from bulk_launcher_mediago import prepare_native_creative
        from PIL import Image

        with patch("bulk_launcher_mediago._gemini_api_key", return_value=None):
            out, name = prepare_native_creative(_UploadStub(_png_bytes(), "sq.png"), fmt="1.91:1")
        im = Image.open(io.BytesIO(out))
        self.assertEqual(im.size, (1200, 628))
        self.assertTrue(name.endswith("1200x628.jpg"))


class MediaGoRegistryTest(unittest.TestCase):
    def test_get_adapter(self):
        from platforms import PLATFORMS, get_adapter, normalize_platform

        self.assertIn("mediago", PLATFORMS)
        self.assertEqual(normalize_platform("mediago"), "mediago")
        adapter = get_adapter("mediago", api_token="tok")
        self.assertEqual(adapter.platform, "mediago")

    def test_storage_exclusions(self):
        from tests.test_ai_studio import _TempStorage

        with _TempStorage():
            import storage

            saved = storage.save_site_exclusions(
                "acc1",
                [
                    {"site_id": "2007904", "domain_name": "msn.com", "reason": "bottom_quartile"},
                    {"site_id": "0", "domain_name": "skip"},
                ],
                platform="mediago",
            )
            self.assertEqual(len(saved), 1)
            loaded = storage.load_site_exclusions("acc1", platform="mediago")
            self.assertEqual(loaded[0]["site_id"], "2007904")
            self.assertEqual(storage.load_site_exclusions("other", platform="mediago"), [])


if __name__ == "__main__":
    unittest.main()

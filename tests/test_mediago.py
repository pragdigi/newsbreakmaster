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
                "target_cpa_usd": "25",
                "optimization_type": "8",
            },
            [{"asset_name": "a1", "img": "https://img/1.jpg", "headline": "Hello world"}],
        )
        self.assertEqual(payload["creative_type"], "native")
        self.assertEqual(payload["status"], 0)
        self.assertEqual(payload["daily_cap"], 40.0)
        self.assertEqual(payload["target_cpa"], 25.0)
        self.assertEqual(payload["optimization_type"], "8")
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
            "target_cpa_usd": "20",
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
        self.assertEqual(adapter.created[0][0], "acc1")
        self.assertEqual(adapter.created[0][1]["creative_type"], "native")
        self.assertEqual(adapter.created[0][1]["landing_page"], "https://offer.example")
        self.assertEqual(adapter.created[0][1]["target_cpa"], 20.0)
        self.assertEqual(adapter.created[0][1]["spend_mode"], 0)
        self.assertEqual(adapter.created[0][1]["status"], 0)
        self.assertEqual(res["status"], 0)
        self.assertEqual(adapter.blocked[0][1], "999")

    def test_launch_active_uses_selected_account(self):
        from bulk_launcher_mediago import mediago_bulk_launch

        class _Adapter:
            def create_campaign(self, account_id, payload):
                self.seen = (account_id, payload)
                return {"campaign_id": "42"}

        adapter = _Adapter()
        res = mediago_bulk_launch(
            adapter,
            form={
                "account_id": "xeviola-99",
                "campaign_name": "Live",
                "brand_name": "Brand",
                "copy_variant_url": "https://offer.example/v2",
                "daily_cap_usd": "100",
                "target_cpa_usd": "40",
                "campaign_status": "1",
                "spend_mode": "0",
                "headline_0": "A native headline that converts",
            },
            files={"creative_0": _UploadStub(_png_bytes(), "ad.png")},
            host_image=lambda img, name: f"https://host.test/{name}",
            creative_builder=lambda f, *, fmt="1.91:1": (b"jpeg", "ad.jpg"),
        )
        self.assertTrue(res["ok"])
        self.assertEqual(res["status"], 1)
        self.assertIn("ACTIVE", res["note"])
        self.assertEqual(adapter.seen[0], "xeviola-99")
        self.assertEqual(adapter.seen[1]["status"], 1)
        self.assertEqual(adapter.seen[1]["landing_page"], "https://offer.example/v2")
        self.assertEqual(adapter.seen[1]["spend_mode"], 0)

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


class MediaGoPixelAndCpaTest(unittest.TestCase):
    def test_optimization_type_mapping(self):
        from platforms.mediago import optimization_type_for_conversion

        self.assertEqual(optimization_type_for_conversion("purchase"), "8")
        self.assertEqual(optimization_type_for_conversion("Lead"), "10")
        self.assertEqual(optimization_type_for_conversion("add to cart"), "4")
        self.assertEqual(optimization_type_for_conversion("8"), "8")
        self.assertEqual(optimization_type_for_conversion(""), "-1")
        self.assertEqual(optimization_type_for_conversion("unknown"), "-1")

    def test_normalize_account_pixel(self):
        from platforms.mediago import normalize_account_pixel

        n = normalize_account_pixel(
            {
                "conversion_name": "purchase",
                "category": "Purchase",
                "status": 1,
                "include_in_total_conversion": 1,
            },
            "acc1",
        )
        self.assertEqual(n["pixel_id"], "purchase")
        self.assertEqual(n["optimization_type"], "8")
        self.assertEqual(n["name"], "Purchase")
        self.assertEqual(n["ad_account_id"], "acc1")

    def test_list_pixels_uses_manage_api_not_client_accounts(self):
        from platforms.mediago import MediaGoAdapter

        class _C:
            def list_account_pixels(self, account_id):
                self.called = account_id
                return [{"conversion_name": "lead", "category": "Lead", "status": 1}]

        c = _C()
        a = MediaGoAdapter(c)
        rows = a.list_pixels("99")
        self.assertEqual(c.called, "99")
        self.assertEqual(rows[0]["pixel_id"], "lead")
        self.assertEqual(rows[0]["optimization_type"], "10")

    def test_list_events_sets_pixel_id_from_conversion_name(self):
        from platforms.mediago import MediaGoAdapter

        class _C:
            def list_account_pixels(self, account_id):
                return [{"conversion_name": "purchase", "category": "Purchase", "status": 1}]

        events = MediaGoAdapter(_C()).list_events("acc")
        self.assertEqual(events[0]["pixel_id"], "purchase")
        self.assertEqual(events[0]["optimization_type"], "8")

    def test_client_list_account_pixels_hits_manage_account(self):
        from mediago_api import MediaGoClient

        c = MediaGoClient("tok")
        c._resolved_level = "client"
        c._access_token = "abc"
        c._token_expires_at = 1e18
        seen = {}

        def fake_get(path, params=None, account_id=None):
            seen["path"] = path
            seen["account_id"] = account_id
            return [
                {
                    "account_id": "7",
                    "account_name": "A",
                    "pixels": [{"conversion_name": "purchase", "category": "Purchase"}],
                }
            ]

        c.get = fake_get  # type: ignore
        rows = c.list_account_pixels("7")
        self.assertEqual(seen["path"], "/manage/v1/account")
        self.assertEqual(seen["account_id"], "7")
        self.assertEqual(rows[0]["conversion_name"], "purchase")

    def test_payload_defaults_target_cpa_and_pacing_and_lander_alias(self):
        from bulk_launcher_mediago import build_campaign_payload

        payload = build_campaign_payload(
            {
                "campaign_name": "X",
                "brand_name": "B",
                "landing_url": "https://offer.example/lander?x=1&",
                "objective": "conversions",
            },
            [{"asset_name": "a", "img": "https://i", "headline": "H"}],
        )
        self.assertEqual(payload["landing_page"], "https://offer.example/lander?x=1")
        self.assertEqual(payload["target_cpa"], 40.0)
        self.assertEqual(payload["daily_cap"], 100.0)
        self.assertEqual(payload["spend_mode"], 0)
        self.assertEqual(payload["charge_type"], "max_cv")
        self.assertEqual(payload["product_type"], "Health & Fitness")
        self.assertEqual(payload["status"], 0)
        self.assertEqual(payload["dp_timezone"], "EST")
        self.assertEqual(payload["location"][0]["region"], "US")
        self.assertEqual(payload["location"][0]["type"], "ALL")
        self.assertEqual(payload["platform_targeting"]["type"], "INCLUDE")
        self.assertEqual(payload["platform_targeting"]["value"], ["Mobile", "Desktop", "Tablet"])
        self.assertEqual(payload["audience"]["type"], "ALL")
        self.assertEqual(payload["os_targeting"]["type"], "ALL")
        self.assertEqual(payload["browser_targeting"]["type"], "ALL")
        self.assertEqual(payload["optimization_type"], "-1")

    def test_payload_pacing_standard_and_active_status(self):
        from bulk_launcher_mediago import build_campaign_payload

        payload = build_campaign_payload(
            {
                "campaign_name": "X",
                "brand_name": "B",
                "landing_page": "https://x.com",
                "daily_cap_usd": "100",
                "target_cpa_usd": "40",
                "pacing": "standard",
                "campaign_status": "1",
                "platform_mobile": "1",
                "platform_desktop": "1",
                "platform_tablet": "1",
                "platform_xbox": "1",
            },
            [{"asset_name": "a", "img": "https://i", "headline": "H"}],
        )
        self.assertEqual(payload["spend_mode"], 1)
        self.assertEqual(payload["status"], 1)
        self.assertEqual(payload["platform_targeting"], {"type": "ALL", "value": []})

    def test_payload_omits_target_cpa_for_awareness(self):
        from bulk_launcher_mediago import build_campaign_payload

        payload = build_campaign_payload(
            {
                "campaign_name": "X",
                "brand_name": "B",
                "landing_page": "https://x.com",
                "daily_cap_usd": "40",
                "objective": "awareness",
                "cpc_usd": "0.40",
            },
            [{"asset_name": "a", "img": "https://i", "headline": "H"}],
        )
        self.assertNotIn("target_cpa", payload)
        self.assertNotIn("optimization_type", payload)

    def test_payload_maps_pixel_id_to_optimization_type(self):
        from bulk_launcher_mediago import build_campaign_payload

        payload = build_campaign_payload(
            {
                "campaign_name": "X",
                "brand_name": "B",
                "landing_page": "https://x.com",
                "daily_cap_usd": "40",
                "objective": "lead",
                "target_cpa_usd": "15",
                "cpc_usd": "0.40",
                "pixel_id": "lead",
            },
            [{"asset_name": "a", "img": "https://i", "headline": "H"}],
        )
        self.assertEqual(payload["target_cpa"], 15.0)
        self.assertEqual(payload["optimization_type"], "10")

    def test_offer_pixels_map_keeps_other_platforms(self):
        from storage import merge_offer_platform_pixels, offer_pixel_ref

        existing = {"pixel_id": "nb1", "pixels": {"newsbreak": "nb1"}}
        merged = merge_offer_platform_pixels(existing, "mediago", "purchase")
        self.assertEqual(merged["newsbreak"], "nb1")
        self.assertEqual(merged["mediago"], "purchase")
        offer = {"pixel_id": "purchase", "pixels": merged}
        self.assertEqual(offer_pixel_ref(offer, "mediago"), "purchase")
        self.assertEqual(offer_pixel_ref(offer, "newsbreak"), "nb1")

    def test_offer_accounts_map_keeps_other_platforms(self):
        from storage import merge_offer_platform_accounts, offer_account_ids, offer_landing_url

        existing = {"ad_account_ids": ["nb1"], "accounts": {"newsbreak": "nb1"}}
        merged = merge_offer_platform_accounts(existing, "mediago", "146")
        self.assertEqual(merged["newsbreak"], "nb1")
        self.assertEqual(merged["mediago"], "146")
        offer = {
            "accounts": merged,
            "ad_account_ids": ["146"],
            "landing_page": "https://xeviola.com/pages/tinnito-lander",
        }
        self.assertEqual(offer_account_ids(offer, "mediago"), ["146"])
        self.assertEqual(offer_account_ids(offer, "newsbreak"), ["nb1", "146"])
        self.assertEqual(offer_landing_url(offer), "https://xeviola.com/pages/tinnito-lander")

    def test_create_campaign_sends_account_id_header(self):
        from mediago_api import MediaGoClient

        c = MediaGoClient("tok")
        c._resolved_level = "account"
        c._access_token = "abc"
        c._token_expires_at = 1e18
        seen = {}

        class _Resp:
            status_code = 200
            text = '{"campaign_id":"9"}'

            def json(self):
                return {"campaign_id": "9"}

        def fake_request(method, url, params=None, json=None, headers=None, timeout=None):
            seen["headers"] = headers
            seen["json"] = json
            return _Resp()

        c._session.request = fake_request  # type: ignore
        c.create_campaign("xeviola-99", {"campaign_name": "T", "status": 1})
        self.assertEqual(seen["headers"]["Account-Id"], "xeviola-99")
        self.assertEqual(seen["json"]["account_id"], "xeviola-99")
        self.assertEqual(seen["json"]["status"], 1)


class MediaGoAppPixelRoutesTest(unittest.TestCase):
    def test_sync_list_and_offer_pixels_map(self):
        from tests.test_ai_studio import _TempStorage

        with _TempStorage():
            import app as appmod

            class FakeAdapter:
                platform = "mediago"

                def get_accounts(self):
                    return [{"id": "acc1", "name": "Acme"}]

                def list_pixels(self, account_id):
                    return [
                        {
                            "pixel_id": "purchase",
                            "name": "Purchase",
                            "conversion_name": "purchase",
                            "optimization_type": "8",
                        }
                    ]

            with (
                patch.object(appmod, "_adapter", return_value=FakeAdapter()),
                patch.object(appmod, "_effective_token", return_value={"api_token": "x"}),
                patch.object(appmod, "_active_platform", return_value="mediago"),
            ):
                client = appmod.app.test_client()
                with client.session_transaction() as s:
                    s["platform"] = "mediago"
                    s["uid"] = "u1"
                resp = client.post("/api/mediago/sync-events")
                self.assertEqual(resp.status_code, 200)
                data = resp.get_json()
                self.assertEqual(data["pixels_added"], 1)
                self.assertEqual(data["added"], 1)

                listed = client.get("/api/mediago/pixels/acc1")
                self.assertEqual(listed.status_code, 200)
                self.assertEqual(listed.get_json()["pixels"][0]["pixel_id"], "purchase")

                pix = client.get("/api/pixels").get_json()["pixels"][0]
                offer_resp = client.post(
                    "/api/offers",
                    json={
                        "name": "Mag",
                        "pixel_id": pix["id"],
                        "target_cpa": 22,
                        "ad_account_ids": ["acc1"],
                        "landing_url": "https://xeviola.com/pages/tinnito-lander",
                    },
                )
                self.assertEqual(offer_resp.status_code, 200)
                offer = offer_resp.get_json()["offer"]
                self.assertEqual(offer["pixels"]["mediago"], pix["id"])
                self.assertEqual(offer["pixel_id"], pix["id"])
                self.assertEqual(offer["target_cpa"], 22)
                self.assertEqual(offer["accounts"]["mediago"], "acc1")
                self.assertEqual(offer["ad_account_ids"], ["acc1"])
                self.assertEqual(offer["landing_url"], "https://xeviola.com/pages/tinnito-lander")

    def test_pixels_route_rejects_wrong_platform(self):
        import app as appmod

        with (
            patch.object(appmod, "_effective_token", return_value={"api_token": "x"}),
            patch.object(appmod, "_active_platform", return_value="newsbreak"),
        ):
            client = appmod.app.test_client()
            resp = client.get("/api/mediago/pixels/acc1")
            self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main()

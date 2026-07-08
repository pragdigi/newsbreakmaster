"""Unit tests for the Outbrain/Teads integration.

Covers:
  - outbrain_api money helpers + envelope unwrap + pagination
  - OutbrainAdapter normalization + budget/status writes
  - bulk_launcher_outbrain form parsing, payload building, creative pipeline,
    preflight validation, and the budget -> campaign -> promotedLink flow
    (with a stubbed adapter + host_image, no network).
"""
from __future__ import annotations

import io
import os
import sys
import unittest
from datetime import date

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _png_bytes(w: int = 800, h: int = 800, color=(120, 60, 200)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


class _UploadStub:
    """Mimics a werkzeug FileStorage just enough for the launcher."""

    def __init__(self, data: bytes, filename: str):
        self._data = data
        self.filename = filename

    def read(self):
        return self._data


class _MultiDictForm(dict):
    """Tiny MultiDict shim supporting getlist for platforms/locations."""

    def __init__(self, simple: dict, multi: dict | None = None):
        super().__init__(simple)
        self._multi = multi or {}

    def getlist(self, key):
        if key in self._multi:
            return list(self._multi[key])
        v = self.get(key)
        return [v] if v not in (None, "") else []


# ----------------------------------------------------------------------
# outbrain_api helpers
# ----------------------------------------------------------------------
class OutbrainApiHelpersTest(unittest.TestCase):
    def test_money_round_trip(self):
        from outbrain_api import amount_to_cents, cents_to_amount

        self.assertEqual(cents_to_amount(5000), 50.0)
        self.assertEqual(amount_to_cents(50.0), 5000)
        self.assertEqual(amount_to_cents("0.55"), 55)
        self.assertIsNone(cents_to_amount(None))
        self.assertIsNone(amount_to_cents(""))

    def test_unwrap_list_named_and_generic(self):
        from outbrain_api import unwrap_list

        self.assertEqual(unwrap_list({"campaigns": [{"id": "x"}]}), [{"id": "x"}])
        self.assertEqual(unwrap_list([{"id": "1"}]), [{"id": "1"}])
        self.assertEqual(unwrap_list({"nope": 1}), [])

    def test_paginate_stops_on_total(self):
        from outbrain_api import OutbrainClient

        c = OutbrainClient(token="tok")
        pages = [
            {"campaigns": [{"id": "1"}, {"id": "2"}], "totalCount": 3},
            {"campaigns": [{"id": "3"}], "totalCount": 3},
        ]
        calls = {"n": 0}

        def fake_get(path, params=None):
            body = pages[calls["n"]]
            calls["n"] += 1
            return body

        c.get = fake_get  # type: ignore
        rows = list(c.paginate("/x", collection="campaigns", limit=2))
        self.assertEqual([r["id"] for r in rows], ["1", "2", "3"])
        self.assertEqual(calls["n"], 2)


# ----------------------------------------------------------------------
# Adapter
# ----------------------------------------------------------------------
class _FakeClient:
    def __init__(self):
        self.budget_updates = []
        self.campaign_updates = []
        self.pl_updates = []

    def get_campaign(self, cid, **kw):
        return {"id": cid, "budgetId": "bud-9"}

    def update_budget(self, bid, body):
        self.budget_updates.append((bid, body))
        return {"id": bid, **body}

    def update_campaign(self, cid, body):
        self.campaign_updates.append((cid, body))
        return {"id": cid, **body}

    def update_promoted_link(self, pid, body):
        self.pl_updates.append((pid, body))
        return {"id": pid, **body}


class OutbrainAdapterTest(unittest.TestCase):
    def _adapter(self):
        from platforms.outbrain import OutbrainAdapter

        return OutbrainAdapter(_FakeClient()), 

    def test_flags(self):
        from platforms.outbrain import OutbrainAdapter

        a = OutbrainAdapter(_FakeClient())
        self.assertEqual(a.platform, "outbrain")
        self.assertFalse(a.supports_ad_set_scope)
        self.assertEqual(a.get_ad_groups("m", "c"), [])

    def test_normalize_campaign(self):
        from platforms.outbrain import OutbrainAdapter

        a = OutbrainAdapter(_FakeClient())
        row = {
            "id": "c1", "name": "Camp", "enabled": True, "cpc": 0.5,
            "budgetId": "b1", "budget": {"id": "b1", "amount": 500, "dailyTarget": 25},
        }
        n = a._normalize_campaign(row, "mk1")
        self.assertEqual(n["id"], "c1")
        self.assertEqual(n["status"], "on")
        self.assertEqual(n["daily_budget_cents"], 2500)
        self.assertEqual(n["budget_id"], "b1")

    def test_update_status_routes(self):
        from platforms.outbrain import OutbrainAdapter

        fc = _FakeClient()
        a = OutbrainAdapter(fc)
        a.update_status("campaign", "c1", False)
        self.assertEqual(fc.campaign_updates, [("c1", {"enabled": False})])
        a.update_status("ad", "p1", True)
        self.assertEqual(fc.pl_updates, [("p1", {"enabled": True})])

    def test_update_budget_resolves_campaign_budget(self):
        from platforms.outbrain import OutbrainAdapter

        fc = _FakeClient()
        a = OutbrainAdapter(fc)
        a.update_budget("campaign", "c1", budget_cents=3000, budget_type="DAILY")
        self.assertEqual(fc.budget_updates, [("bud-9", {"dailyTarget": 30.0})])
        a.update_budget("budget", "b7", budget_cents=10000, budget_type="TOTAL")
        self.assertEqual(fc.budget_updates[-1], ("b7", {"amount": 100.0}))

    def test_report_row_canonicalization(self):
        from platforms.outbrain import OutbrainAdapter

        a = OutbrainAdapter(_FakeClient())
        row = {
            "metadata": {"id": "c1", "name": "Camp"},
            "metrics": {"impressions": 1000, "clicks": 50, "spend": 25.0,
                         "conversions": 5, "sumValue": 100.0},
        }
        n = a._canonicalize_report_row(row, "campaign", "mk1", None)
        self.assertEqual(n["scope"], "campaign")
        self.assertEqual(n["impressions"], 1000)
        self.assertAlmostEqual(n["ctr"], 5.0)
        self.assertAlmostEqual(n["cpa"], 5.0)
        self.assertAlmostEqual(n["roas"], 4.0)


# ----------------------------------------------------------------------
# Launcher
# ----------------------------------------------------------------------
class _LaunchAdapter:
    """Stub adapter capturing budget/campaign/promoted-link writes."""

    def __init__(self):
        self.budgets = []
        self.campaigns = []
        self.client = self  # launcher calls adapter.client.create_promoted_link
        self.promoted_links = []

    def create_budget(self, account_id, payload):
        self.budgets.append((account_id, payload))
        return {"id": f"bud-{len(self.budgets)}"}

    def create_campaign(self, account_id, payload):
        self.campaigns.append((account_id, payload))
        return {"id": f"camp-{len(self.campaigns)}"}

    # client surface
    def create_promoted_link(self, campaign_id, payload):
        self.promoted_links.append((campaign_id, payload))
        return {"id": f"pl-{len(self.promoted_links)}", "cachedImageUrl": "http://img/x.jpg"}


class OutbrainLauncherTest(unittest.TestCase):
    def _host(self):
        hosted = []

        def host_image(b: bytes, fname: str) -> str:
            hosted.append((len(b), fname))
            return f"https://public.example/creative/{fname}"

        return host_image, hosted

    def test_new_campaign_flow_1to1(self):
        from bulk_launcher_outbrain import outbrain_bulk_launch

        adapter = _LaunchAdapter()
        host_image, hosted = self._host()
        form = _MultiDictForm(
            {
                "account_id": "mk1",
                "campaign_mode": "new",
                "campaign_name": "Test Camp",
                "objective": "Traffic",
                "budget_amount_usd": "100",
                "cpc_usd": "0.40",
                "creative_format": "1:1",
                "language": "en",
                "landing_page_url": "https://example.com/lp",
                "cta_label": "LEARN_MORE",
                "headline_0": "Great offer headline",
            },
            multi={"platforms": ["DESKTOP", "MOBILE"]},
        )
        files = {"creative_0": _UploadStub(_png_bytes(), "ad0.png")}
        res = outbrain_bulk_launch(adapter, form=form, files=files, host_image=host_image)
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["creative_format"], "1:1")
        # budget payload
        _, bud = adapter.budgets[0]
        self.assertEqual(bud["amount"], 100.0)
        self.assertEqual(bud["type"], "MONTHLY")  # no end date -> evergreen
        self.assertTrue(bud["runForever"])
        # campaign payload
        _, camp = adapter.campaigns[0]
        self.assertEqual(camp["budgetId"], "bud-1")
        self.assertEqual(camp["cpc"], 0.40)
        self.assertFalse(camp["enabled"])
        self.assertEqual(camp["targeting"]["platform"], ["DESKTOP", "MOBILE"])
        self.assertEqual(camp["targeting"]["locations"], ["fc4deb5112fb4415a9edacdf4aafb0d8"])
        # promoted link
        cid, pl = adapter.promoted_links[0]
        self.assertEqual(cid, "camp-1")
        self.assertEqual(pl["text"], "Great offer headline")
        self.assertEqual(pl["url"], "https://example.com/lp")
        self.assertEqual(pl["callToAction"], "LEARN_MORE")
        self.assertTrue(pl["imageUrl"].startswith("https://public.example/"))
        self.assertTrue(hosted[0][1].endswith("_1200x1200.jpg"))

    def test_preflight_blocks_missing_headline(self):
        from bulk_launcher_outbrain import outbrain_bulk_launch

        adapter = _LaunchAdapter()
        host_image, _ = self._host()
        form = _MultiDictForm({
            "account_id": "mk1",
            "campaign_mode": "new",
            "campaign_name": "C",
            "budget_amount_usd": "100",
            "landing_page_url": "https://e.com",
            # headline_0 missing
        })
        files = {"creative_0": _UploadStub(_png_bytes(), "a.png")}
        res = outbrain_bulk_launch(adapter, form=form, files=files, host_image=host_image)
        self.assertFalse(res["ok"])
        self.assertEqual(adapter.budgets, [])  # nothing created
        self.assertTrue(res["errors"])

    def test_existing_campaign_skips_budget(self):
        from bulk_launcher_outbrain import outbrain_bulk_launch

        adapter = _LaunchAdapter()
        host_image, _ = self._host()
        form = _MultiDictForm({
            "account_id": "mk1",
            "campaign_mode": "existing",
            "campaign_id": "camp-existing",
            "creative_format": "1:1",
            "landing_page_url": "https://e.com",
            "headline_0": "Reuse this campaign",
        })
        files = {"creative_0": _UploadStub(_png_bytes(), "a.png")}
        res = outbrain_bulk_launch(adapter, form=form, files=files, host_image=host_image)
        self.assertTrue(res["ok"], res)
        self.assertTrue(res["campaign_reused"])
        self.assertEqual(adapter.budgets, [])
        self.assertEqual(adapter.campaigns, [])
        self.assertEqual(adapter.promoted_links[0][0], "camp-existing")

    def test_campaign_budget_with_end_date(self):
        from bulk_launcher_outbrain import outbrain_bulk_launch

        adapter = _LaunchAdapter()
        host_image, _ = self._host()
        form = _MultiDictForm({
            "account_id": "mk1",
            "campaign_mode": "new",
            "campaign_name": "C",
            "budget_amount_usd": "800",
            "end_time": "2026-12-31T00:00",
            "creative_format": "1:1",
            "landing_page_url": "https://e.com",
            "headline_0": "Bounded budget campaign",
        })
        files = {"creative_0": _UploadStub(_png_bytes(), "a.png")}
        res = outbrain_bulk_launch(adapter, form=form, files=files, host_image=host_image)
        self.assertTrue(res["ok"], res)
        _, bud = adapter.budgets[0]
        self.assertEqual(bud["type"], "CAMPAIGN")
        self.assertEqual(bud["endDate"], "2026-12-31")
        self.assertFalse(bud["runForever"])

    def test_prepare_creative_1to1_dims(self):
        from bulk_launcher_outbrain import prepare_creative
        from PIL import Image

        b, name = prepare_creative(_UploadStub(_png_bytes(1000, 1000), "x.png"), fmt="1:1")
        im = Image.open(io.BytesIO(b))
        self.assertEqual(im.size, (1200, 1200))
        self.assertTrue(name.endswith("_1200x1200.jpg"))


if __name__ == "__main__":
    unittest.main()

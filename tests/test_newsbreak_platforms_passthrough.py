"""End-to-end test: launch.html ``platforms_mode=custom`` form post lands
the validated ``platforms`` list inside the ``/ad-set/create`` JSON body.

We don't want a regression where ``bulk_launcher.bulk_launch`` strips a
new ad-set field — that already bit us once with ``placements`` /
``trafficPlatforms`` (silently dropped, no-op). This guards the data
path from form -> NewsBreakClient.create_ad_set.
"""
from __future__ import annotations

import io
import os
import sys
import unittest
from unittest import mock

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class _StubClient:
    """Records the ad-set create payload so the test can assert on it."""

    def __init__(self):
        self.created_campaigns: list[dict] = []
        self.created_ad_sets: list[dict] = []
        self.created_ads: list[dict] = []

    def create_campaign(self, payload):
        self.created_campaigns.append(dict(payload))
        return {"data": {"id": "cmp-1"}}

    def create_ad_set(self, payload):
        self.created_ad_sets.append(dict(payload))
        return {"data": {"id": f"as-{len(self.created_ad_sets)}"}}

    def create_ad(self, payload):
        self.created_ads.append(dict(payload))
        return {"data": {"id": f"ad-{len(self.created_ads)}"}}

    def upload_asset(self, file_bytes, filename, ad_account_id, **kw):
        return {"data": {"assetUrl": f"https://cdn/{filename}"}}


class PlatformsPassThroughTest(unittest.TestCase):
    def _run(self, ad_set_base_extras: dict) -> dict:
        from bulk_launcher import bulk_launch

        client = _StubClient()
        creatives = [{
            "file_bytes": b"fake-png-bytes",
            "filename": "creative-01.png",
            "headline": "Hello",
            "body": "World",
            "landing_url": "https://example.com",
            "media_name": "creative-01.png",
        }]
        ad_set_base = {
            "name_prefix": "Smoke set",
            "budgetType": "DAILY",
            "budget": 5000,
            "bidType": "CPC",
            "bidRate": 100,
            "startTime": 1_700_000_000,
            "endTime": 1_700_999_999,
            "targeting": {"location": {"positive": ["all"]}},
            "_ad_account_id_for_upload": "act-1",
            "_brand_name": "Acme",
            "_cta": "Learn More",
            **ad_set_base_extras,
        }
        result = bulk_launch(
            client,
            ad_account_id="act-1",
            campaign_mode="new",
            campaign_id=None,
            campaign_payload={"name": "smoke camp"},
            ad_set_base=ad_set_base,
            creatives=creatives,
            grouping="all_in_one",
            group_size=1,
        )
        # Sanity-check: the launch itself must succeed for the assertions
        # below to be meaningful.
        self.assertTrue(result["success"], result.get("errors"))
        self.assertEqual(len(client.created_ad_sets), 1)
        return client.created_ad_sets[0]

    def test_platforms_field_lands_in_ad_set_payload(self):
        payload = self._run({"platforms": ["NEWSBREAK", "PREMIUM_PARTNERS_GAMING"]})
        self.assertEqual(payload.get("platforms"), ["NEWSBREAK", "PREMIUM_PARTNERS_GAMING"])

    def test_no_platforms_when_omitted(self):
        # Default "All inventory" path: the field is absent so NewsBreak
        # uses its own default of ["APP_AND_WEB_UNLIMITED"]. Critically,
        # this matches the pre-Apr-2026 behavior the rest of our code
        # assumed.
        payload = self._run({})
        self.assertNotIn("platforms", payload)


if __name__ == "__main__":
    unittest.main()

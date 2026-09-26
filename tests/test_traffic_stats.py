"""Unit tests for the read-only traffic stats rollup. No network."""
from __future__ import annotations

import os
import unittest
from datetime import date
from unittest.mock import patch

from traffic_stats import missing_env, rollup_rows


class RollupTests(unittest.TestCase):
    def test_sums_spend_purchases_and_revenue(self):
        totals = rollup_rows(
            [
                {"spend": 10, "conversions": 2, "value": 40, "events": {"purchase": 2}},
                {"spend": 5.5, "conversions": 1, "value": 0, "events": {"purchase": 1}},
            ],
            prefer_purchase=True,
            revenue_supported=True,
        )
        self.assertEqual(totals["spend"], 15.5)
        self.assertEqual(totals["purchases"], 3)
        self.assertEqual(totals["revenue"], 40.0)
        self.assertEqual(totals["cpa"], 5.17)
        self.assertTrue(totals["revenue_available"])

    def test_revenue_omitted_when_unsupported(self):
        totals = rollup_rows(
            [{"spend": 20, "conversions": 2, "events": {"purchase": 2}, "value": 99}],
            prefer_purchase=True,
            revenue_supported=False,
        )
        self.assertFalse(totals["revenue_available"])
        self.assertIsNone(totals["revenue"])
        self.assertEqual(totals["cpa"], 10.0)

    def test_mediago_falls_back_to_conversion_without_cv_purchase(self):
        totals = rollup_rows(
            [{"spend": 8, "conversions": 2}],
            prefer_purchase=True,
            revenue_supported=True,
        )
        self.assertEqual(totals["purchases"], 2)
        self.assertEqual(totals["purchases_metric"], "conversion")
        self.assertFalse(totals["revenue_available"])

    def test_zero_cv_purchase_uses_conversion_total(self):
        """MediaGo sends cv_purchase=0 on every row. That must not hide conversion."""
        totals = rollup_rows(
            [
                {
                    "spend": 994.65,
                    "conversion": 18,
                    "conversions": 18,
                    "cv_purchase": 0,
                    "cv_start_checkout": 18,
                    "cv_lead": 0,
                    "roas": 0,
                },
                {
                    "spend": 897.88,
                    "conversion": 20,
                    "conversions": 20,
                    "cv_purchase": 0,
                    "cv_start_checkout": 20,
                    "roas": 0,
                },
            ],
            prefer_purchase=True,
            revenue_supported=True,
        )
        self.assertEqual(totals["purchases"], 38)
        self.assertEqual(totals["purchases_metric"], "conversion")
        self.assertFalse(totals["revenue_available"])
        self.assertAlmostEqual(totals["cpa"], 49.8)

    def test_nonzero_cv_purchase_beats_conversion_total(self):
        totals = rollup_rows(
            [{"spend": 40, "conversion": 10, "conversions": 10, "cv_purchase": 4, "cv_lead": 6}],
            prefer_purchase=True,
            revenue_supported=True,
        )
        self.assertEqual(totals["purchases"], 4)
        self.assertEqual(totals["purchases_metric"], "cv_purchase")

    def test_deal_completed_used_when_purchase_is_zero(self):
        totals = rollup_rows(
            [{"spend": 30, "conversion": 9, "cv_purchase": 0, "cv_deal_completed": 3, "cv_lead": 6}],
            prefer_purchase=True,
            revenue_supported=True,
        )
        self.assertEqual(totals["purchases"], 3)
        self.assertEqual(totals["purchases_metric"], "cv_deal_completed")


class MissingEnvTests(unittest.TestCase):
    def test_names_absent_vars_without_reading_values(self):
        with patch.dict(os.environ, {}, clear=False):
            for key in (
                "NEWSBREAK_ACCESS_TOKEN",
                "NEWSBREAK_DEFAULT_ORG_IDS",
                "MEDIAGO_API_TOKEN",
                "SMARTNEWS_CLIENT_ID",
                "SMARTNEWS_CLIENT_SECRET",
                "SMARTNEWS_API_KEY",
            ):
                os.environ.pop(key, None)
            self.assertEqual(
                missing_env("newsbreak"),
                ["NEWSBREAK_ACCESS_TOKEN", "NEWSBREAK_DEFAULT_ORG_IDS"],
            )
            self.assertEqual(missing_env("mediago"), ["MEDIAGO_API_TOKEN"])
            self.assertEqual(
                missing_env("smartnews"),
                ["SMARTNEWS_CLIENT_ID", "SMARTNEWS_CLIENT_SECRET"],
            )

    def test_smartnews_api_key_counts_as_secret(self):
        with patch.dict(
            os.environ,
            {"SMARTNEWS_CLIENT_ID": "1", "SMARTNEWS_API_KEY": "legacy", "SMARTNEWS_CLIENT_SECRET": ""},
            clear=False,
        ):
            self.assertEqual(missing_env("smartnews"), [])

    def test_skipped_shape_uses_dates(self):
        self.assertEqual(date(2026, 9, 25).isoformat(), "2026-09-25")


if __name__ == "__main__":
    unittest.main()

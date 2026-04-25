"""Tests for the rotation behaviour added to :mod:`ai_studio.research.bandit`.

The allocator must:

  * Within a single batch, fill every slot with a distinct style_id while
    the pool of available styles is large enough.
  * Bias picks toward styles that have NOT appeared in recent batches when
    a recency penalty is configured.
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ai_studio.research import bandit  # noqa: E402


class TestWithinBatchDistinct(unittest.TestCase):
    def test_distinct_when_pool_large_enough(self):
        catalog = [f"cat_{i}" for i in range(15)]
        candidates = [
            {"style_id": f"cand_{i}", "trials": 0, "wins": 0, "status": "candidate"}
            for i in range(10)
        ]
        # No history → no recency penalty in play; we're testing distinctness.
        with mock.patch.object(bandit, "_recent_allocation_counts", return_value={}):
            picks = bandit.allocate(
                catalog_styles=catalog,
                candidates=candidates,
                n=10,
                seed=42,
            )
        self.assertEqual(len(picks), 10)
        ids = [p["style_id"] for p in picks]
        self.assertEqual(len(set(ids)), 10, f"expected 10 distinct picks, got {ids}")

    def test_repeats_only_when_pool_smaller_than_n(self):
        # 3 catalog styles, 0 candidates, 8 slots → must repeat.
        catalog = ["a", "b", "c"]
        with mock.patch.object(bandit, "_recent_allocation_counts", return_value={}):
            picks = bandit.allocate(
                catalog_styles=catalog,
                candidates=[],
                n=8,
                seed=1,
            )
        self.assertEqual(len(picks), 8)
        # Every catalog style appears at least once.
        ids = {p["style_id"] for p in picks}
        self.assertEqual(ids, {"a", "b", "c"})


class TestRecencyPenalty(unittest.TestCase):
    def test_recently_used_styles_are_deprioritised(self):
        # Three identical-prior catalog styles: 'a' was used 10x recently,
        # 'b' and 'c' were each used once. We pre-fill all three with at
        # least one recent appearance so the catalog-floor pass doesn't
        # fire and skew the outcome (it picks 0-count styles first).
        catalog = ["a", "b", "c"]
        recent = {"a": 10, "b": 1, "c": 1}
        with mock.patch.object(bandit, "_recent_allocation_counts", return_value=recent):
            counts = {"a": 0, "b": 0, "c": 0}
            for s in range(2000):
                picks = bandit.allocate(
                    catalog_styles=catalog,
                    candidates=[],
                    n=1,
                    seed=s,
                    platform="newsbreak",
                )
                counts[picks[0]["style_id"]] += 1
        # Penalty: a → (1 - 0.05*10)=0.5, b/c → (1 - 0.05*1)=0.95.
        # Expect 'a' picked materially less than either 'b' or 'c'.
        self.assertLess(counts["a"], counts["b"], f"counts={counts}")
        self.assertLess(counts["a"], counts["c"], f"counts={counts}")

    def test_zero_recency_means_no_penalty(self):
        # If every style has identical recent history, no penalty should
        # dominate and all three should win a healthy share over many trials.
        # (We pre-fill recent counts to skip the floor pass.)
        catalog = ["a", "b", "c"]
        recent = {"a": 1, "b": 1, "c": 1}
        with mock.patch.object(bandit, "_recent_allocation_counts", return_value=recent):
            counts = {"a": 0, "b": 0, "c": 0}
            for s in range(900):
                picks = bandit.allocate(
                    catalog_styles=catalog,
                    candidates=[],
                    n=1,
                    seed=s,
                    platform="newsbreak",
                )
                counts[picks[0]["style_id"]] += 1
        for sid, c in counts.items():
            self.assertGreater(c, 100, f"{sid} starved: counts={counts}")


class TestCandidateRotation(unittest.TestCase):
    def test_distinct_candidates_within_batch_when_enough_exist(self):
        # 10 candidates available, batch wants 4 candidate slots
        # (research_ratio=0.4 of 10) → all 4 must be distinct.
        catalog = [f"cat_{i}" for i in range(10)]
        candidates = [
            {"style_id": f"cand_{i}", "trials": 0, "wins": 0, "status": "candidate"}
            for i in range(10)
        ]
        with mock.patch.object(bandit, "_recent_allocation_counts", return_value={}):
            picks = bandit.allocate(
                catalog_styles=catalog,
                candidates=candidates,
                n=10,
                research_ratio=0.4,
                seed=7,
            )
        cand_picks = [p for p in picks if p.get("is_candidate")]
        cand_ids = [p["style_id"] for p in cand_picks]
        self.assertEqual(
            len(set(cand_ids)), len(cand_ids),
            f"candidate picks repeated within one batch: {cand_ids}",
        )


if __name__ == "__main__":
    unittest.main()

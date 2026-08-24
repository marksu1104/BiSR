"""Hand-computable toy tests for the bespoke (non-shared) full-entity rank
helpers used by AnyBURL and LoGRe/StruProKGR, verifying they agree with the
shared definitions in ranking_metrics.py for equivalent scenarios.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "baselines" / "AnyBURL"))
sys.path.insert(0, str(REPO_ROOT / "baselines" / "StruProKGR"))

from ranking_metrics import average_tie_rank, optimistic_tie_rank  # noqa: E402
from score_anyburl import tie_aware_rank as anyburl_rank  # noqa: E402
from score_struprokgr import tie_aware_rank as strupro_rank  # noqa: E402


class AnyBURLRankTests(unittest.TestCase):
    def test_target_returned_and_tied_with_another_candidate(self):
        # cands: target(C)=3.0 tied with D=3.0, B=5.0 beats both. filtered={}.
        cands = [("B", 5.0), ("C", 3.0), ("D", 3.0)]
        rank = anyburl_rank(cands, "C", filtered_others=set(), num_entities=5)
        # greater=1 (B), equal=2 (C,D) -> average_tie_rank(1, 2) = 2.5
        self.assertEqual(rank, average_tie_rank(1, 2))

    def test_optimistic_mode_matches_shared_definition(self):
        cands = [("B", 5.0), ("C", 3.0), ("D", 3.0)]
        rank = anyburl_rank(cands, "C", filtered_others=set(), num_entities=5, tie_mode="optimistic")
        self.assertEqual(rank, optimistic_tie_rank(1))

    def test_target_not_predicted_falls_back_to_zero_score(self):
        # Only B is predicted (positive); target C and 3 other entities all
        # default to 0.0 among num_entities=5 (no filtering).
        cands = [("B", 2.0)]
        rank = anyburl_rank(cands, "C", filtered_others=set(), num_entities=5)
        # num_nonmasked=5; pos=1 (B); greater=1; equal = 5-1 = 4 (C + 3 unseen zero entities)
        self.assertEqual(rank, average_tie_rank(1, 4))

    def test_filtered_others_shrink_the_universe(self):
        # A is another known-true answer and must be excluded from candidates
        # AND from the entity count.
        cands = [("A", 9.0), ("C", 1.0)]
        rank = anyburl_rank(cands, "C", filtered_others={"A"}, num_entities=5)
        # num_nonmasked = 5-1 = 4; A excluded from scoring entirely.
        # explicit (after filtering A) = {C: 1.0}; greater=0, equal = 4-0=...
        # target_score=1.0>0 -> greater=sum(s>1.0 among {C:1.0})=0; equal=sum(s==1.0)=1 (C itself)
        self.assertEqual(rank, average_tie_rank(0, 1))


class StruProKGRRankTests(unittest.TestCase):
    def test_gold_predicted_with_positive_score(self):
        # gold_score=2.0, 1 higher-scored predicted entity, 2 tied (incl. gold)
        rank = strupro_rank(gold_score=2.0, n_higher=1, n_tied_in_pred=2,
                             filter_size=0, total_entities=10)
        self.assertEqual(rank, average_tie_rank(1, 2))

    def test_optimistic_mode(self):
        rank = strupro_rank(gold_score=2.0, n_higher=1, n_tied_in_pred=2,
                             filter_size=0, total_entities=10, tie_mode="optimistic")
        self.assertEqual(rank, optimistic_tie_rank(1))

    def test_gold_unpredicted_zero_score_ties_with_all_unpredicted_entities(self):
        # gold_score=0.0 (unpredicted); 3 predicted non-zero entities beat it;
        # total_entities=10, filter_size=1 -> num_nonmasked=9.
        rank = strupro_rank(gold_score=0.0, n_higher=3, n_tied_in_pred=0,
                             filter_size=1, total_entities=10)
        # n_zero = num_nonmasked(9) - n_higher(3) = 6 (includes gold itself)
        self.assertEqual(rank, average_tie_rank(3, 6))


if __name__ == "__main__":
    unittest.main()

"""Hand-computable toy tests for the shared Main/SOTA ranking evaluator.

Every expected value in this file is derived by hand from the definitions in
Section 5.3 of the thesis:

    rank_avg(q) = k_q + (m_q + 1) / 2     (Main Protocol: average-tie)
    rank_opt(q) = k_q + 1                 (SOTA Protocol: optimistic-tie)

where k_q = number of filtered candidates with a strictly higher score than
the target, and m_q = number of candidates tied with the target (including
the target itself). Both protocols share the same filtering and full-entity
universe; the functions tested here decide only the tie-handling axis (given
a fixed query direction). The Main and SOTA protocols overall also differ in
query direction -- Main is bidirectional, SOTA is tail-only -- which is
decided by the caller, not by these functions.
"""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ranking_metrics import (
    average_tie_rank,
    optimistic_tie_rank,
    rank_from_counts,
    sparse_filtered_rank,
)


class AverageAndOptimisticTieRankTests(unittest.TestCase):
    def test_average_tie_rank_matches_hand_formula(self):
        # k_q=1, m_q=2 -> 1 + (2+1)/2 = 2.5
        self.assertEqual(average_tie_rank(greater=1, equal=2), 2.5)

    def test_optimistic_tie_rank_matches_hand_formula(self):
        # k_q=1 -> 1 + 1 = 2, regardless of how many are tied
        self.assertEqual(optimistic_tie_rank(greater=1), 2.0)

    def test_optimistic_never_worse_than_average(self):
        for greater, equal in [(0, 1), (2, 5), (10, 1)]:
            self.assertLessEqual(
                optimistic_tie_rank(greater),
                average_tie_rank(greater, equal),
            )

    def test_rank_from_counts_dispatches_correctly(self):
        self.assertEqual(rank_from_counts(1, 2, "average"), 2.5)
        self.assertEqual(rank_from_counts(1, 2, "optimistic"), 2.0)

    def test_invalid_tie_mode_raises(self):
        with self.assertRaises(ValueError):
            rank_from_counts(0, 1, "best")

    def test_invalid_equal_count_raises(self):
        with self.assertRaises(ValueError):
            average_tie_rank(greater=0, equal=0)  # target must tie with itself


class SparseFilteredRankToyTests(unittest.TestCase):
    """universe = {A, B, C, D, E}; target = C throughout."""

    UNIVERSE = {"A", "B", "C", "D", "E"}

    def test_filtered_known_answers_are_excluded_from_ranking(self):
        # A is another known-true answer for this query and must be removed
        # from the ranking universe entirely (not merely outscored).
        scores = {"A": 9.0, "B": 5.0, "C": 3.0, "D": 3.0, "E": 1.0}
        rank = sparse_filtered_rank(scores, "C", self.UNIVERSE, true_answers={"A"})
        # eligible = {B, C, D, E}; greater={B}; equal={C,D} -> 1 + (2+1)/2
        self.assertEqual(rank, 2.5)

    def test_average_tie_counts_target_itself_in_the_tied_block(self):
        scores = {"B": 5.0, "C": 3.0, "D": 3.0}
        rank = sparse_filtered_rank(scores, "C", self.UNIVERSE, true_answers=set())
        # E is unreturned -> default 0.0 < target_score 3.0, doesn't affect C.
        # greater={B}=1, equal={C,D}=2 -> 1 + 3/2 = 2.5
        self.assertEqual(rank, 2.5)

    def test_optimistic_tie_ignores_tie_size(self):
        scores = {"B": 5.0, "C": 3.0, "D": 3.0}
        rank = sparse_filtered_rank(
            scores, "C", self.UNIVERSE, true_answers=set(), tie_mode="optimistic"
        )
        # Only B (5.0 > 3.0) counts; D's tie is ignored under optimistic rank.
        self.assertEqual(rank, 2.0)

    def test_target_tied_with_multiple_explicit_entities(self):
        scores = {"A": 3.0, "B": 3.0, "C": 3.0, "D": 3.0, "E": 3.0}
        rank_avg = sparse_filtered_rank(scores, "C", self.UNIVERSE, true_answers=set())
        rank_opt = sparse_filtered_rank(
            scores, "C", self.UNIVERSE, true_answers=set(), tie_mode="optimistic"
        )
        # Everyone ties -> greater=0, equal=5 (includes C) -> avg = 0+(5+1)/2=3.0
        self.assertEqual(rank_avg, 3.0)
        # optimistic best-case -> greater=0 -> rank=1 (C is first among the tie)
        self.assertEqual(rank_opt, 1.0)

    def test_target_not_returned_by_sparse_method_uses_default_score(self):
        # C (the target) is absent from `scores`; only B and D were returned.
        scores = {"B": 2.0, "D": -1.0}
        rank = sparse_filtered_rank(scores, "C", self.UNIVERSE, true_answers=set())
        # target_score = default = 0.0. Explicit: B(2.0>0 -> greater), D(-1.0<0).
        # Implicit (A, E) also default to 0.0 == target_score -> tie.
        # greater=1 (B), equal = C(self) + A + E = 3 -> 1 + 4/2 = 3.0
        self.assertEqual(rank, 3.0)

    def test_other_unreturned_candidates_are_folded_into_the_tie_at_default_score(self):
        # Only the target is returned; A, B, D, E are all unreturned (implicit).
        scores = {"C": 0.0}
        rank = sparse_filtered_rank(scores, "C", self.UNIVERSE, true_answers=set())
        # target_score = 0.0 (explicit, but equals default). All 4 implicit
        # entities also default to 0.0 -> equal = 1(self) + 4 = 5, greater=0.
        self.assertEqual(rank, 0 + (5 + 1) / 2.0)

    def test_empty_candidate_set_falls_back_entirely_to_default_score(self):
        scores: dict[str, float] = {}
        rank = sparse_filtered_rank(scores, "C", self.UNIVERSE, true_answers=set())
        # Every entity (including the target) defaults to 0.0 -> one giant tie
        # of size 5 -> 0 + (5+1)/2 = 3.0. This is the worst tie-averaged rank
        # the 5-entity universe can produce, which is the expected behaviour
        # when a method returns literally nothing for a query.
        self.assertEqual(rank, 3.0)

    def test_full_entity_universe_counts_unreturned_entities_beyond_the_dump(self):
        # 10-entity universe; only 2 entities (including target) are scored.
        universe = {f"e{i}" for i in range(10)}
        scores = {"e0": 1.0, "e1": -2.0}
        rank = sparse_filtered_rank(scores, "e0", universe, true_answers=set())
        # target_score=1.0. Only e1 is explicit and below it. The other 8
        # entities (e2..e9) are implicit at default 0.0 < 1.0 -> don't count.
        self.assertEqual(rank, 0 + (1 + 1) / 2.0)

    def test_filter_size_reduces_the_denominator_not_just_the_scores(self):
        # Two other true answers (A, E) must shrink the eligible universe,
        # not merely be scored out -- this is what "filtered ranking" means.
        scores = {"A": 100.0, "E": 100.0}  # both would tie for 1st if not filtered
        rank = sparse_filtered_rank(scores, "C", self.UNIVERSE, true_answers={"A", "E"})
        # eligible = {B, C, D} (A, E removed from the universe entirely).
        # target_score defaults to 0.0; B, D also default to 0.0 -> all tie.
        self.assertEqual(rank, 0 + (3 + 1) / 2.0)

    def test_target_missing_from_universe_raises(self):
        with self.assertRaises(ValueError):
            sparse_filtered_rank({}, "Z", self.UNIVERSE, true_answers=set())

    def test_unsupported_tie_mode_raises(self):
        with self.assertRaises(ValueError):
            sparse_filtered_rank(
                {"C": 1.0}, "C", self.UNIVERSE, true_answers=set(), tie_mode="best"
            )


class ReverseHeadPredictionToyTest(unittest.TestCase):
    """(?, r, t) must be scored as the reverse-relation query (t, r_rev, ?)."""

    def test_head_query_rewritten_as_reverse_tail_query(self):
        # Toy KG: (Alan, bornIn, Oxford) is the only fact for relation `bornIn`.
        # Head-prediction query (?, bornIn, Oxford) should recover Alan.
        universe = {"Alan", "Oxford", "Other1", "Other2"}

        # After rewriting, this becomes a tail query from head=Oxford under
        # the reverse relation bornIn_rev, i.e. (Oxford, bornIn_rev, ?).
        # A model scoring this reverse query returns these candidate scores:
        reverse_scores = {"Alan": 5.0, "Other1": 1.0}
        gold_head = "Alan"
        # No other known head answers exist for (bornIn, Oxford) here.
        rank = sparse_filtered_rank(reverse_scores, gold_head, universe, true_answers=set())
        # greater=0 (nobody beats 5.0); equal={Alan} only (Other2 defaults to
        # 0.0 < 5.0, doesn't tie) -> 0 + (1+1)/2 = 1.0 (correct top rank).
        self.assertEqual(rank, 1.0)

    def test_reverse_query_still_filters_other_true_heads(self):
        # Two entities are true heads for (bornIn, Oxford): Alan and Priya.
        # Ranking Alan must filter out Priya even though Priya scores higher.
        universe = {"Alan", "Priya", "Other1", "Other2"}
        reverse_scores = {"Priya": 9.0, "Alan": 2.0}
        rank = sparse_filtered_rank(
            reverse_scores, "Alan", universe, true_answers={"Priya"}
        )
        # eligible = {Alan, Other1, Other2}; Priya removed entirely.
        # greater=0, equal = Alan(self) + Other1 + Other2 (default 0<2 -> no)
        # Other1/Other2 default to 0.0 < 2.0 -> not tied -> equal=1 -> rank=1.0
        self.assertEqual(rank, 1.0)


if __name__ == "__main__":
    unittest.main()

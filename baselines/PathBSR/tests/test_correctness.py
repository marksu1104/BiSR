from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from pathbsr.config import DEFAULT_CONFIG
from pathbsr.data import augment_with_reverse_edges
from pathbsr.evaluation import filtered_rank
from pathbsr.graph import GraphStore, build_entity_features
from pathbsr.model import PathBSR
from pathbsr.paths import RuleMiner
from pathbsr.retrieval import BM25Index
from pathbsr.ranking import CandidateRanker


def build_graph(train, valid=(), test=(), config=DEFAULT_CONFIG):
    train_aug = augment_with_reverse_edges(train, config.reverse_suffix)
    _, train_entities = build_entity_features(train_aug, config)
    return GraphStore.build(train, valid, test, train_entities, config)


class CorrectnessTests(unittest.TestCase):
    def test_paper_defaults_match_validated_mainline(self):
        self.assertTrue(DEFAULT_CONFIG.enable_length1_rules)
        self.assertEqual(DEFAULT_CONFIG.path_semantics_mode, "walk")
        self.assertEqual(DEFAULT_CONFIG.path_budget_selection_mode, "deterministic_first")
        self.assertEqual(DEFAULT_CONFIG.rule_statistics_mode, "case_empirical")
        self.assertEqual(DEFAULT_CONFIG.rule_ordering_mode, "reliability")
        self.assertTrue(DEFAULT_CONFIG.bm25_normalize_proxy_scores)
        self.assertEqual(DEFAULT_CONFIG.verification_selection_mode, "candidate_score")
        self.assertEqual(DEFAULT_CONFIG.verification_top_m, 100)
        self.assertEqual(DEFAULT_CONFIG.verification_max_hops, 3)
        self.assertEqual(DEFAULT_CONFIG.bridge_edge_cap, 0)

    def test_model_graph_uses_train_only(self):
        train = [("a", "r", "b")]
        valid = [("a", "r", "c")]
        test = [("d", "r", "a")]
        graph = build_graph(train, valid, test)

        self.assertEqual(graph.out_adj["a"]["r"], {"b"})
        self.assertNotIn("c", graph.out_adj["a"]["r"])
        self.assertNotIn("d", graph.out_adj)
        self.assertEqual(graph.undirected_neighbors["a"], {"b"})
        self.assertEqual(graph.all_true_tails[("a", "r")], {"b", "c"})

    def test_path_budget_caps_all_lengths_combined(self):
        config = replace(
            DEFAULT_CONFIG, max_paths_per_case=1, path_search_cap_per_length=8
        )
        train = [
            ("h", "a", "x"),
            ("x", "b", "t"),
            ("h", "c", "y"),
            ("y", "d", "z"),
            ("z", "e", "t"),
        ]
        miner = RuleMiner.__new__(RuleMiner)
        miner.graph = build_graph(train, config=config)
        miner.config = config
        miner._paths_between_cache = {}
        miner.path_discovery_calls = 0
        miner.path_discovery_overflow_cases = 0
        miner.path_discovery_cap_hit_cases = 0
        miner.path_discovery_max_candidates = 0
        self.assertEqual(len(miner.find_paths_between("h", "t")), 1)
        self.assertEqual(miner.path_discovery_calls, 1)
        self.assertEqual(miner.path_discovery_overflow_cases, 1)
        self.assertEqual(miner.path_discovery_cap_hit_cases, 1)

    def test_strict_rule_statistics_exclude_target_and_reciprocal_edges(self):
        config = replace(
            DEFAULT_CONFIG,
            min_path_support=1,
            max_path_len=3,
            max_paths_per_case=16,
            path_search_cap_per_length=16,
        )
        # The direct fact creates the cyclic body r -> r_reverse -> r. It must
        # not count as an alternative explanation of itself. The a -> b body
        # is an independent path and must remain available.
        train = [
            ("h", "r", "t"),
            ("h", "a", "x"),
            ("x", "b", "t"),
        ]
        miner = RuleMiner(build_graph(train, config=config), config)
        rules = {path for path, _ in miner.get_relation_rules("r", 100)}

        self.assertIn(("a", "b"), rules)
        self.assertNotIn(("r", "r__reverse", "r"), rules)

    def test_length1_rules_are_flagged_and_exclude_the_target_edge(self):
        config = replace(
            DEFAULT_CONFIG,
            min_path_support=1,
            max_path_len=3,
            enable_length1_rules=True,
            max_paths_per_case=16,
            path_search_cap_per_length=16,
        )
        train = [
            ("h", "r", "t"),
            ("h", "alt", "t"),
            ("h", "a", "x"),
            ("x", "b", "t"),
        ]
        miner = RuleMiner(build_graph(train, config=config), config)
        rules = {path for path, _ in miner.get_relation_rules("r", 100)}

        self.assertIn(("alt",), rules)
        self.assertIn(("a", "b"), rules)
        self.assertNotIn(("r",), rules)

    def test_reference_execute_path_enforces_simple_object_identity_semantics(self):
        config = replace(DEFAULT_CONFIG, path_semantics_mode="reference")
        train = [
            ("h", "a", "x"),
            ("x", "b", "h"),
            ("x", "b", "t"),
        ]
        miner = RuleMiner(build_graph(train, config=config), config)
        self.assertEqual(miner.execute_path("h", ("a", "b")), ("t",))

    def test_budgeted_oi_keeps_legacy_discovery_but_uses_oi_execution(self):
        legacy_config = replace(DEFAULT_CONFIG, case_mid_cap=1, path_semantics_mode="legacy")
        oi_config = replace(legacy_config, path_semantics_mode="budgeted_oi")
        train = [
            ("h", "a", "m1"),
            ("h", "a", "m2"),
            ("m2", "b", "t"),
            ("m1", "b", "h"),
        ]
        legacy = RuleMiner(build_graph(train, config=legacy_config), legacy_config)
        budgeted_oi = RuleMiner(build_graph(train, config=oi_config), oi_config)

        self.assertEqual(
            legacy.find_paths_between("h", "t"),
            budgeted_oi.find_paths_between("h", "t"),
        )
        self.assertEqual(legacy.execute_path("h", ("a", "b")), ("h", "t"))
        self.assertEqual(budgeted_oi.execute_path("h", ("a", "b")), ("t",))

    def test_legacy_oi_keeps_bounded_discovery_with_exact_execution(self):
        legacy_config = replace(
            DEFAULT_CONFIG,
            min_path_support=1,
            case_mid_cap=1,
            path_semantics_mode="legacy",
            execution_branch_cap=1,
        )
        oi_config = replace(legacy_config, path_semantics_mode="legacy_oi")
        train = [
            ("h", "a", "x1"),
            ("h", "a", "x2"),
            ("x1", "b", "h"),
            ("x1", "b", "t1"),
            ("x2", "b", "t2"),
        ]
        legacy = RuleMiner(build_graph(train, config=legacy_config), legacy_config)
        exact_oi = RuleMiner(build_graph(train, config=oi_config), oi_config)

        self.assertEqual(
            legacy.find_paths_between("h", "t1"),
            exact_oi.find_paths_between("h", "t1"),
        )
        self.assertEqual(exact_oi.execute_path("h", ("a", "b")), ("t1", "t2"))
        self.assertEqual(exact_oi.path_execution_overflows, 0)

    def test_reference_oi_execution_is_exact_beyond_legacy_branch_cap(self):
        reference_config = replace(
            DEFAULT_CONFIG,
            min_path_support=1,
            path_semantics_mode="reference",
            execution_branch_cap=1,
        )
        budgeted_config = replace(
            reference_config,
            path_semantics_mode="budgeted_oi",
        )
        train = [
            ("h", "a", "x1"),
            ("h", "a", "x2"),
            ("x1", "b", "t1"),
            ("x2", "b", "t2"),
        ]
        reference = RuleMiner(build_graph(train, config=reference_config), reference_config)
        budgeted = RuleMiner(build_graph(train, config=budgeted_config), budgeted_config)

        self.assertEqual(reference.execute_path("h", ("a", "b")), ("t1", "t2"))
        self.assertEqual(budgeted.execute_path("h", ("a", "b")), tuple())
        self.assertEqual(reference.path_execution_overflows, 0)
        self.assertGreater(budgeted.path_execution_overflows, 0)

    def test_specialized_short_oi_executor_matches_generic_executor(self):
        reference_config = replace(
            DEFAULT_CONFIG,
            min_path_support=1,
            path_semantics_mode="reference",
            execution_branch_cap=100,
        )
        generic_config = replace(
            reference_config,
            path_semantics_mode="budgeted_oi",
        )
        train = [
            ("h", "a", "x1"),
            ("h", "a", "x2"),
            ("x1", "b", "h"),
            ("x1", "b", "y1"),
            ("x2", "b", "y2"),
            ("y1", "c", "t1"),
            ("y2", "c", "t2"),
        ]
        reference = RuleMiner(build_graph(train, config=reference_config), reference_config)
        generic = RuleMiner(build_graph(train, config=generic_config), generic_config)

        for path in (("a",), ("a", "b"), ("a", "b", "c")):
            self.assertEqual(
                reference.execute_path("h", path),
                generic.execute_path("h", path),
            )

    def test_reference_zero_case_cap_keeps_all_relation_paths(self):
        config = replace(
            DEFAULT_CONFIG,
            min_path_support=1,
            max_path_len=2,
            max_paths_per_case=0,
            path_semantics_mode="reference",
        )
        train = [
            ("h", "r", "t"),
            ("h", "a", "x"),
            ("x", "b", "t"),
            ("h", "c", "y"),
            ("y", "d", "t"),
        ]
        miner = RuleMiner(build_graph(train, config=config), config)
        paths = miner.find_paths_between("h", "t")

        self.assertIn(("a", "b"), paths)
        self.assertIn(("c", "d"), paths)
        self.assertEqual(miner.path_discovery_overflow_cases, 0)

    def test_reference_case_cap_reports_overflow(self):
        config = replace(
            DEFAULT_CONFIG,
            min_path_support=1,
            max_path_len=2,
            max_paths_per_case=1,
            path_semantics_mode="reference",
        )
        train = [
            ("h", "r", "t"),
            ("h", "a", "x"),
            ("x", "b", "t"),
            ("h", "c", "y"),
            ("y", "d", "t"),
        ]
        miner = RuleMiner(build_graph(train, config=config), config)
        miner.find_paths_between("h", "t")

        self.assertGreater(miner.path_discovery_overflow_cases, 0)
        self.assertGreaterEqual(miner.path_discovery_max_candidates, 2)

    def test_reference_discovery_avoids_lexical_first_found_truncation(self):
        legacy_config = replace(
            DEFAULT_CONFIG,
            min_path_support=1,
            max_path_len=3,
            max_paths_per_case=16,
            path_search_cap_per_length=8,
            case_mid_cap=1,
            path_semantics_mode="legacy",
        )
        reference_config = replace(
            legacy_config,
            path_semantics_mode="reference",
        )
        train = [
            ("h", "a", "m1"),
            ("h", "a", "m2"),
            ("m2", "b", "t"),
            ("h", "r", "t"),
        ]

        legacy_miner = RuleMiner(build_graph(train, config=legacy_config), legacy_config)
        reference_miner = RuleMiner(build_graph(train, config=reference_config), reference_config)

        self.assertNotIn(("a", "b"), legacy_miner.find_paths_between("h", "t"))
        self.assertIn(("a", "b"), reference_miner.find_paths_between("h", "t"))

    def test_seeded_legacy_budget_is_reproducible_and_not_lexical_first(self):
        config = replace(
            DEFAULT_CONFIG,
            min_path_support=1,
            max_path_len=2,
            max_paths_per_case=16,
            path_search_cap_per_length=8,
            case_mid_cap=1,
            path_semantics_mode="legacy",
            path_budget_selection_mode="seeded",
            path_sampling_seed=42,
        )
        train = [
            ("h", "a", "m1"),
            ("h", "a", "m2"),
            ("m2", "b", "t"),
            ("h", "r", "t"),
        ]
        first = RuleMiner(build_graph(train, config=config), config)
        second = RuleMiner(build_graph(train, config=config), config)

        self.assertEqual(
            first.find_paths_between("h", "t"),
            second.find_paths_between("h", "t"),
        )
        self.assertIn(("a", "b"), first.find_paths_between("h", "t"))

    def test_target_sampled_max_length_one_excludes_longer_paths(self):
        config = replace(
            DEFAULT_CONFIG,
            min_path_support=1,
            max_path_len=1,
            enable_length1_rules=True,
            path_semantics_mode="target_sampled",
        )
        train = [
            ("h", "r", "t"),
            ("h", "alt", "t"),
            ("h", "a", "x"),
            ("x", "b", "t"),
        ]
        miner = RuleMiner(build_graph(train, config=config), config)
        rules = {path for path, _ in miner.get_relation_rules("r", 100)}
        self.assertIn(("alt",), rules)
        self.assertNotIn(("a", "b"), rules)

    def test_legacy_max_length_one_excludes_longer_paths(self):
        config = replace(
            DEFAULT_CONFIG,
            min_path_support=1,
            max_path_len=1,
            enable_length1_rules=True,
            path_semantics_mode="legacy",
        )
        train = [
            ("h", "r", "t"),
            ("h", "alt", "t"),
            ("h", "a", "x"),
            ("x", "b", "t"),
        ]
        miner = RuleMiner(build_graph(train, config=config), config)
        rules = {path for path, _ in miner.get_relation_rules("r", 100)}
        self.assertIn(("alt",), rules)
        self.assertNotIn(("a", "b"), rules)

    def test_legacy_discovers_length_five_walk_rule(self):
        config = replace(
            DEFAULT_CONFIG,
            min_path_support=1,
            max_path_len=5,
            path_semantics_mode="legacy",
            path_search_cap_per_length=16,
            max_paths_per_case=32,
        )
        train = [
            ("h", "r", "t"),
            ("h", "a", "z1"),
            ("z1", "b", "z2"),
            ("z2", "c", "z3"),
            ("z3", "d", "z4"),
            ("z4", "e", "t"),
        ]
        miner = RuleMiner(build_graph(train, config=config), config)
        rules = {path for path, _ in miner.get_relation_rules("r", 100)}
        self.assertIn(("a", "b", "c", "d", "e"), rules)

    def test_target_sampled_discovers_length_five_simple_path(self):
        config = replace(
            DEFAULT_CONFIG,
            min_path_support=1,
            max_path_len=5,
            enable_length1_rules=True,
            path_semantics_mode="target_sampled",
            path_sampling_attempts_per_length=16,
            path_search_cap_per_length=8,
        )
        train = [
            ("h", "r", "t"),
            ("h", "a", "z1"),
            ("z1", "b", "z2"),
            ("z2", "c", "z3"),
            ("z3", "d", "z4"),
            ("z4", "e", "t"),
        ]
        miner = RuleMiner(build_graph(train, config=config), config)
        rules = {path for path, _ in miner.get_relation_rules("r", 100)}
        self.assertIn(("a", "b", "c", "d", "e"), rules)

    def test_uniform_sampled_uses_one_policy_through_length_five(self):
        config = replace(
            DEFAULT_CONFIG,
            min_path_support=1,
            max_path_len=5,
            enable_length1_rules=True,
            path_semantics_mode="uniform_sampled",
            path_sampling_attempts_per_length=16,
            path_search_cap_per_length=8,
        )
        train = [
            ("h", "r", "t"),
            ("h", "alt", "t"),
            ("h", "a", "z1"),
            ("z1", "b", "z2"),
            ("z2", "c", "z3"),
            ("z3", "d", "z4"),
            ("z4", "e", "t"),
        ]
        miner = RuleMiner(build_graph(train, config=config), config)
        rules = {path for path, _ in miner.get_relation_rules("r", 100)}
        self.assertIn(("alt",), rules)
        self.assertIn(("a", "b", "c", "d", "e"), rules)
        self.assertGreater(miner.path_sampling_attempts[2], 0)
        self.assertGreater(miner.path_sampling_attempts[3], 0)

    def test_global_rule_statistics_compute_confidence_and_head_support(self):
        config = replace(
            DEFAULT_CONFIG,
            min_path_support=1,
            rule_statistics_mode="global",
            rule_ordering_mode="confidence",
            path_semantics_mode="legacy",
        )
        train = [
            ("h1", "a", "x1"),
            ("x1", "b", "t1"),
            ("h1", "r", "t1"),
            ("h2", "a", "x2"),
            ("x2", "b", "t2"),
            ("h2", "r", "t2"),
            ("h3", "a", "x3"),
            ("x3", "b", "u"),
        ]
        miner = RuleMiner(build_graph(train, config=config), config)
        self.assertAlmostEqual(miner.path_confidence("r", ("a", "b")), 0.6)
        self.assertEqual(miner.path_head_support_value("r", ("a", "b")), 2)

    def test_legacy_reliability_uses_distinct_head_support(self):
        config = replace(
            DEFAULT_CONFIG,
            min_path_support=1,
            rule_statistics_mode="legacy",
            rule_ordering_mode="reliability",
            path_semantics_mode="legacy",
        )
        train = [
            ("h1", "a", "x1"),
            ("x1", "b", "t1"),
            ("x1", "b", "u1"),
            ("h1", "r", "t1"),
            ("h1", "r", "u1"),
            ("h2", "a", "x2"),
            ("x2", "b", "t2"),
            ("h2", "r", "t2"),
        ]
        miner = RuleMiner(build_graph(train, config=config), config)
        confidence = miner.path_confidence("r", ("a", "b"))
        self.assertEqual(miner.path_support[("r", ("a", "b"))], 3)
        self.assertEqual(miner.path_head_support_value("r", ("a", "b")), 2)
        self.assertAlmostEqual(
            miner.rule_quality("r", ("a", "b")),
            confidence * np.log1p(2.0),
        )

    def test_head_relation_observation_unit_counts_each_head_once(self):
        config = replace(
            DEFAULT_CONFIG,
            min_path_support=1,
            per_pair_confidence=True,
            rule_statistics_mode="legacy",
            path_semantics_mode="legacy",
        )
        train = [
            ("h1", "a", "x1"),
            ("x1", "b", "t1"),
            ("x1", "b", "u1"),
            ("h1", "r", "t1"),
            ("h1", "r", "u1"),
            ("h2", "a", "x2"),
            ("x2", "b", "t2"),
            ("h2", "r", "t2"),
        ]
        miner = RuleMiner(build_graph(train, config=config), config)

        self.assertEqual(miner.path_support[("r", ("a", "b"))], 2)
        self.assertEqual(miner.path_head_support_value("r", ("a", "b")), 2)

    def test_relation_max_rule_confidence_normalization(self):
        config = replace(
            DEFAULT_CONFIG,
            min_path_support=1,
            rule_confidence_normalization="relation_max",
        )
        train = [
            ("h1", "a", "x1"),
            ("x1", "b", "t1"),
            ("h1", "r", "t1"),
            ("h2", "c", "x2"),
            ("x2", "d", "t2"),
            ("x2", "d", "u2"),
            ("h2", "r", "t2"),
        ]
        miner = RuleMiner(build_graph(train, config=config), config)
        weights = [
            miner.path_answer_weight("r", path)
            for path, _ in miner.get_relation_rules("r", 100)
        ]

        self.assertTrue(weights)
        self.assertAlmostEqual(max(weights), 1.0)

    def test_relation_max_normalization_ignores_ineligible_head_support(self):
        config = replace(
            DEFAULT_CONFIG,
            rule_statistics_mode="global",
            rule_ordering_mode="reliability",
            rule_confidence_normalization="relation_max",
            min_rule_head_support=2,
        )
        miner = RuleMiner.__new__(RuleMiner)
        miner.config = config
        miner.path_confidence_scores = {
            ("r", ("a",)): 0.9,
            ("r", ("b",)): 0.5,
        }
        miner.path_head_support = {
            ("r", ("a",)): 1,
            ("r", ("b",)): 2,
        }
        miner._relation_max_confidence_cache = {}

        self.assertAlmostEqual(miner.path_answer_weight("r", ("b",)), 1.0)

    def test_bm25_binary_vs_log_query_tf(self):
        features = {
            "h": {"REL:r": 3, "ENT:x": 1},
            "p": {"REL:r": 1, "ENT:x": 1},
        }
        binary = BM25Index(features, query_tf_mode="binary")
        log = BM25Index(features, query_tf_mode="log")
        query = features["h"]
        self.assertGreater(log.score(query, "p"), binary.score(query, "p"))

    def test_no_answer_base_does_not_mutate_cached_case_scores(self):
        config = replace(DEFAULT_CONFIG, use_answer_base=False, verification_top_m=2)
        graph = build_graph([("h", "edge", "c")], config=config)
        ranker = CandidateRanker(graph, None, None, config)
        key = ("h", "r", config.topk_proxy)
        original = np.zeros(len(graph.all_entities), dtype=np.float32)
        original[graph.ent2idx["c"]] = 2.0
        ranker._case_path_score_cache[key] = original

        before = ranker.score_case_paths("h", "r").copy()
        final = ranker.score("h", "r")
        after = ranker.score_case_paths("h", "r")

        self.assertGreater(final[graph.ent2idx["c"]], before[graph.ent2idx["c"]])
        np.testing.assert_array_equal(after, before)

    def test_zero_verification_edge_cap_means_unlimited(self):
        config = replace(
            DEFAULT_CONFIG,
            verification_max_hops=3,
            bridge_edge_cap=0,
        )
        train = [
            ("h", "e", "x1"),
            ("h", "e", "x2"),
            ("x1", "e", "y1"),
            ("x2", "e", "y2"),
            ("y1", "e", "t"),
            ("y2", "e", "t"),
        ]
        graph = build_graph(train, config=config)
        ranker = CandidateRanker(graph, None, None, config)

        self.assertGreater(ranker.verification_value("h", "t"), 0.0)
        self.assertEqual(ranker.verification_edge_cap_hits, 0)
        self.assertEqual(ranker.verification_max_checked_edges, 2)

    def test_verification_hop_four_is_an_optional_simple_path_term(self):
        train = [
            ("h", "e", "w1"),
            ("w1", "e", "w2"),
            ("w2", "e", "w3"),
            ("w3", "e", "t"),
        ]
        config3 = replace(DEFAULT_CONFIG, verification_max_hops=3)
        config4 = replace(DEFAULT_CONFIG, verification_max_hops=4)
        score3 = CandidateRanker(build_graph(train, config=config3), None, None, config3).verification_value("h", "t")
        score4 = CandidateRanker(build_graph(train, config=config4), None, None, config4).verification_value("h", "t")

        self.assertEqual(score3, 0.0)
        self.assertGreater(score4, 0.0)

    def test_verification_hop_five_is_an_optional_simple_path_term(self):
        train = [
            ("h", "e", "w1"),
            ("w1", "e", "w2"),
            ("w2", "e", "w3"),
            ("w3", "e", "w4"),
            ("w4", "e", "t"),
        ]
        config4 = replace(DEFAULT_CONFIG, verification_max_hops=4)
        config5 = replace(DEFAULT_CONFIG, verification_max_hops=5)
        score4 = CandidateRanker(build_graph(train, config=config4), None, None, config4).verification_value("h", "t")
        score5 = CandidateRanker(build_graph(train, config=config5), None, None, config5).verification_value("h", "t")

        self.assertEqual(score4, 0.0)
        self.assertGreater(score5, 0.0)

    def test_verification_rejects_depth_above_five(self):
        config = replace(DEFAULT_CONFIG, verification_max_hops=6)
        with self.assertRaisesRegex(ValueError, "must be in"):
            CandidateRanker(
                build_graph([("h", "e", "t")], config=config), None, None, config
            )

    def test_walk_execution_reports_frontier_overflow(self):
        config = replace(DEFAULT_CONFIG, execution_branch_cap=1)
        train = [("h", "a", "x1"), ("h", "a", "x2")]
        miner = RuleMiner.__new__(RuleMiner)
        miner.graph = build_graph(train, config=config)
        miner.config = config
        miner._execute_path_cache = {}
        miner.path_execution_overflows = 0

        self.assertEqual(miner.execute_path("h", ("a",)), tuple())
        self.assertEqual(miner.path_execution_overflows, 1)

    def test_non_backtracking_rejects_immediate_return(self):
        config = replace(
            DEFAULT_CONFIG,
            min_path_support=1,
            path_semantics_mode="non_backtracking",
        )
        train = [
            ("h", "a", "x"),
            ("x", "b", "h"),
            ("x", "b", "t"),
        ]
        miner = RuleMiner(build_graph(train, config=config), config)

        self.assertEqual(miner.execute_path("h", ("a", "b")), ("t",))

    def test_reserved_reverse_suffix_collision_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "reserved reverse suffix"):
            PathBSR(
                [("h", "r__reverse", "t")],
                [],
                [],
                config=DEFAULT_CONFIG,
            )

    def test_non_positive_execution_branch_cap_fails_fast(self):
        config = replace(DEFAULT_CONFIG, execution_branch_cap=0)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            RuleMiner(build_graph([("h", "r", "t")], config=config), config)

    def test_top_m_boundary_has_stable_lexical_tie_break(self):
        graph = build_graph([("d", "edge", "a"), ("c", "edge", "b")])
        ranker = CandidateRanker(graph, None, None, DEFAULT_CONFIG)
        scores = np.ones(len(graph.all_entities), dtype=np.float32)
        selected = ranker._top_positive_candidate_indices(scores, 2)
        self.assertEqual(selected, [0, 1])

    def test_filtered_rank_filters_other_truths_and_handles_ties(self):
        entities = ["a", "b", "c", "d"]
        ent2idx = {entity: idx for idx, entity in enumerate(entities)}
        scores = np.array([0.1, 0.8, 0.8, 0.9], dtype=np.float32)
        truths = {("h", "r"): {"b", "d"}}

        self.assertEqual(filtered_rank(scores, ("h", "r"), "b", ent2idx, truths), 1.5)
        self.assertEqual(
            filtered_rank(scores, ("h", "r"), "b", ent2idx, truths, tie_mode="optimistic"),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()

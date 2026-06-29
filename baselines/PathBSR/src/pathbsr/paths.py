"""Rule mining and path execution."""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = lambda x, **kwargs: x

from .config import PathBSRConfig
from .graph import GraphStore

PathProgram = Tuple[str, ...]


class RuleMiner:
    """Mine relation-level path rules from training cases."""

    def __init__(self, graph: GraphStore, config: PathBSRConfig) -> None:
        self.graph = graph
        self.config = config
        self.path_support: Counter[Tuple[str, PathProgram]] = Counter()
        self.path_success: Counter[Tuple[str, PathProgram]] = Counter()
        self.path_total_outputs: Counter[Tuple[str, PathProgram]] = Counter()
        self.path_head_support: Counter[Tuple[str, PathProgram]] = Counter()
        self.path_confidence_scores: Dict[Tuple[str, PathProgram], float] = {}
        self.path_reliability_scores: Dict[Tuple[str, PathProgram], float] = {}
        self.path_sampling_attempts: Counter[int] = Counter()
        self.path_sampling_closures: Counter[int] = Counter()
        self.path_sampling_unique: Counter[int] = Counter()
        self.path_discovery_calls = 0
        self.path_discovery_overflow_cases = 0
        self.path_discovery_cap_hit_cases = 0
        self.path_discovery_max_candidates = 0
        self.path_execution_overflows = 0

        self._paths_between_cache: Dict[Tuple[str, str], Tuple[PathProgram, ...]] = {}
        self._relation_rule_cache: Dict[Tuple[str, int], Tuple[Tuple[PathProgram, float], ...]] = {}
        self._relation_max_confidence_cache: Dict[str, float] = {}
        self._execute_path_cache: Dict[Tuple[str, PathProgram], Tuple[str, ...]] = {}
        self._out_edges_cache: Dict[str, Tuple[Tuple[str, str], ...]] = {}

        if not 1 <= self.config.max_path_len <= 5:
            raise ValueError("max_path_len must be in [1, 5]")
        if self.config.path_semantics_mode not in {
            "walk",
            "non_backtracking",
            "legacy",
            "legacy_oi",
            "budgeted_oi",
            "reference",
            "target_sampled",
            "uniform_sampled",
        }:
            raise ValueError(f"Unknown path_semantics_mode: {self.config.path_semantics_mode}")
        if self.config.path_budget_selection_mode not in {
            "deterministic_first",
            "legacy_first",
            "seeded",
        }:
            raise ValueError(
                "path_budget_selection_mode must be deterministic_first or seeded"
            )
        if self.config.rule_confidence_normalization not in {"none", "relation_max"}:
            raise ValueError(
                "rule_confidence_normalization must be 'none' or 'relation_max'"
            )
        if self.config.path_semantics_mode in {"target_sampled", "uniform_sampled"}:
            if self.config.path_sampling_attempts_per_length < 0:
                raise ValueError("path_sampling_attempts_per_length must be non-negative")
        if self.config.execution_branch_cap <= 0:
            raise ValueError("execution_branch_cap must be positive")

        self._build_path_statistics()

    def _reverse_relation(self, relation: str) -> str:
        if relation.endswith(self.config.reverse_suffix):
            return relation[: -len(self.config.reverse_suffix)]
        return f"{relation}{self.config.reverse_suffix}"

    def _target_edge_pair(self, head: str, relation: str, tail: str) -> FrozenSet[Tuple[str, str, str]]:
        reverse_relation = self._reverse_relation(relation)
        return frozenset({
            (head, relation, tail),
            (tail, reverse_relation, head),
        })

    def _edge_allowed(self, head: str, relation: str, tail: str, forbidden_edges: FrozenSet[Tuple[str, str, str]]) -> bool:
        return not forbidden_edges or (head, relation, tail) not in forbidden_edges

    def _use_reference_path_semantics(self) -> bool:
        mode = self.config.path_semantics_mode
        if mode in {"walk", "non_backtracking", "legacy", "legacy_oi", "budgeted_oi"}:
            return False
        if mode in {"reference", "target_sampled", "uniform_sampled"}:
            return True
        raise ValueError(f"Unknown path_semantics_mode: {mode}")

    def _use_non_backtracking(self) -> bool:
        return self.config.path_semantics_mode == "non_backtracking"

    def _use_oi_execution(self) -> bool:
        mode = self.config.path_semantics_mode
        if mode in {"walk", "non_backtracking", "legacy"}:
            return False
        if mode in {
            "legacy_oi",
            "budgeted_oi",
            "reference",
            "target_sampled",
            "uniform_sampled",
        }:
            return True
        raise ValueError(f"Unknown path_semantics_mode: {mode}")

    def _use_target_sampled_discovery(self) -> bool:
        return self.config.path_semantics_mode == "target_sampled"

    def _use_uniform_sampled_discovery(self) -> bool:
        return self.config.path_semantics_mode == "uniform_sampled"

    def _build_path_statistics(self) -> None:
        if self.config.rule_statistics_mode in {"case_empirical", "legacy"}:
            if self.config.per_pair_confidence:
                head_hits = self._accumulate_per_pair()
            else:
                head_hits = self._accumulate_per_fact()
            for key, heads in head_hits.items():
                self.path_head_support[key] = len(heads)
            for key, support in self.path_support.items():
                if support < self.config.min_path_support:
                    continue
                total = self.path_total_outputs.get(key, 0)
                if total <= 0:
                    continue
                success = self.path_success.get(key, 0)
                confidence = (success + 1.0) / (total + 2.0)
                head_support = self.path_head_support.get(key, 0)
                self.path_confidence_scores[key] = confidence
                self.path_reliability_scores[key] = confidence * math.log1p(float(head_support))
            return

        if self.config.rule_statistics_mode == "global":
            candidate_rules = self._discover_candidate_rules()
            self._compute_global_rule_statistics(candidate_rules)
            return

        raise ValueError(f"Unknown rule_statistics_mode: {self.config.rule_statistics_mode}")

    def _accumulate_per_fact(self) -> Dict[Tuple[str, PathProgram], Set[str]]:
        """Original unit = one observation per training fact (head, relation, tail).
        A (head, relation) with K answers contributes a path up to K times, which
        over-weights high-cardinality relations."""
        head_hits: Dict[Tuple[str, PathProgram], Set[str]] = defaultdict(set)
        for train_head, relation, train_tail in tqdm(
            self.graph.train_aug, desc="Mine path rules", leave=True
        ):
            true_tails = self.graph.train_true_tails.get((train_head, relation), set())
            if not true_tails:
                continue
            if self.config.strict_path_statistics:
                forbidden_edges = self._target_edge_pair(train_head, relation, train_tail)
                paths = self.find_paths_between_excluding(train_head, train_tail, forbidden_edges)
            else:
                forbidden_edges = frozenset()
                paths = self.find_paths_between(train_head, train_tail)
            for path in paths:
                outputs = (
                    self.execute_path_excluding(train_head, path, forbidden_edges)
                    if forbidden_edges
                    else self.execute_path(train_head, path)
                )
                if not outputs:
                    continue
                key = (relation, path)
                output_set = set(outputs)
                self.path_support[key] += 1
                hits = output_set & true_tails
                self.path_success[key] += len(hits)
                self.path_total_outputs[key] += len(output_set)
                if hits:
                    head_hits[key].add(train_head)
        return head_hits

    def _all_target_edges(self, head: str, relation: str, true_tails: Set[str]) -> FrozenSet[Tuple[str, str, str]]:
        reverse_relation = self._reverse_relation(relation)
        edges: Set[Tuple[str, str, str]] = set()
        for tail in true_tails:
            edges.add((head, relation, tail))
            edges.add((tail, reverse_relation, head))
        return frozenset(edges)

    def _accumulate_per_pair(self) -> Dict[Tuple[str, PathProgram], Set[str]]:
        """Cleaner unit = one observation per (head, relation, path). Each path is
        counted once per head, with success/total over the head's full answer set,
        excluding ALL of the head's answer edges (the test scenario: the head has
        no known r-edges). Removes the per-fact double counting."""
        head_hits: Dict[Tuple[str, PathProgram], Set[str]] = defaultdict(set)
        for (head, relation), true_tails in tqdm(
            self.graph.train_true_tails.items(), desc="Mine path rules", leave=True
        ):
            if not true_tails:
                continue
            forbidden_edges = (
                self._all_target_edges(head, relation, true_tails)
                if self.config.strict_path_statistics
                else frozenset()
            )
            paths: Set[PathProgram] = set()
            tail_iter = sorted(true_tails)
            if not self._use_reference_path_semantics():
                tail_iter = tail_iter[: self.config.case_mid_cap]
            for tail in tail_iter:
                if forbidden_edges:
                    paths.update(self.find_paths_between_excluding(head, tail, forbidden_edges))
                else:
                    paths.update(self.find_paths_between(head, tail))
            for path in paths:
                outputs = (
                    self.execute_path_excluding(head, path, forbidden_edges)
                    if forbidden_edges
                    else self.execute_path(head, path)
                )
                if not outputs:
                    continue
                key = (relation, path)
                output_set = set(outputs)
                self.path_support[key] += 1
                hits = output_set & true_tails
                self.path_success[key] += len(hits)
                self.path_total_outputs[key] += len(output_set)
                if hits:
                    head_hits[key].add(head)
        return head_hits

    def _find_paths_between_legacy(
        self,
        src: str,
        dst: str,
        forbidden_edges: FrozenSet[Tuple[str, str, str]],
    ) -> Tuple[PathProgram, ...]:
        self.path_discovery_calls += 1
        cap_hit = False
        paths_by_len: Dict[int, Set[PathProgram]] = defaultdict(set)
        first_hop = self.graph.out_adj_list.get(src, {})
        path_cap = self.config.path_search_cap_per_length

        if path_cap <= 0 or self.config.max_paths_per_case <= 0:
            return tuple()

        if self.config.enable_length1_rules and self.config.max_path_len >= 1:
            rng = self._path_rng("legacy-search", src, dst, 1, forbidden_edges)
            for relation in self._legacy_traversal_order(
                self.graph.pair_relations.get((src, dst), set()), rng
            ):
                if self._edge_allowed(src, relation, dst, forbidden_edges):
                    paths_by_len[1].add((relation,))
                    if len(paths_by_len[1]) >= path_cap:
                        cap_hit = True
                        break

        if self.config.max_path_len >= 2:
            rng = self._path_rng("legacy-search", src, dst, 2, forbidden_edges)
            for r1 in self._legacy_traversal_order(first_hop, rng):
                if len(first_hop[r1]) > self.config.case_mid_cap:
                    cap_hit = True
                for mid1 in self._legacy_traversal_order(
                    first_hop[r1], rng, self.config.case_mid_cap
                ):
                    if not self._edge_allowed(src, r1, mid1, forbidden_edges):
                        continue
                    for r2 in self._legacy_traversal_order(
                        self.graph.pair_relations.get((mid1, dst), set()), rng
                    ):
                        if not self._edge_allowed(mid1, r2, dst, forbidden_edges):
                            continue
                        paths_by_len[2].add((r1, r2))
                        if len(paths_by_len[2]) >= path_cap:
                            cap_hit = True
                            break
                    if len(paths_by_len[2]) >= path_cap:
                        break
                if len(paths_by_len[2]) >= path_cap:
                    break

        for total_length in range(3, self.config.max_path_len + 1):
            rng = self._path_rng(
                "legacy-search", src, dst, total_length, forbidden_edges
            )
            for r1 in self._legacy_traversal_order(first_hop, rng):
                if len(paths_by_len[total_length]) >= path_cap:
                    break
                if len(first_hop[r1]) > self.config.case_mid_cap:
                    cap_hit = True
                for mid1 in self._legacy_traversal_order(
                    first_hop[r1], rng, self.config.case_mid_cap
                ):
                    if mid1 == src or not self._edge_allowed(src, r1, mid1, forbidden_edges):
                        continue
                    self._extend_legacy_paths(
                        current=mid1,
                        dst=dst,
                        forbidden_edges=forbidden_edges,
                        remaining_relations=total_length - 1,
                        prefix=(r1,),
                        seen={src, mid1},
                        out=paths_by_len[total_length],
                        path_cap=path_cap,
                        rng=rng,
                    )
                    if len(paths_by_len[total_length]) >= path_cap:
                        cap_hit = True
                        break

        paths: List[PathProgram] = []
        for length in sorted(paths_by_len):
            paths.extend(sorted(paths_by_len[length]))
        self.path_discovery_max_candidates = max(
            self.path_discovery_max_candidates, len(paths)
        )
        if len(paths) <= self.config.max_paths_per_case:
            if cap_hit:
                self.path_discovery_cap_hit_cases += 1
            return tuple(paths)
        self.path_discovery_overflow_cases += 1
        self.path_discovery_cap_hit_cases += 1
        if self.config.path_budget_selection_mode in {
            "deterministic_first",
            "legacy_first",
        }:
            return tuple(paths[: self.config.max_paths_per_case])
        rng = self._path_rng("legacy-final-budget", src, dst, 0, forbidden_edges)
        return tuple(sorted(rng.sample(paths, self.config.max_paths_per_case)))

    def _extend_legacy_paths(
        self,
        current: str,
        dst: str,
        forbidden_edges: FrozenSet[Tuple[str, str, str]],
        remaining_relations: int,
        prefix: PathProgram,
        seen: Set[str],
        out: Set[PathProgram],
        path_cap: int,
        rng: random.Random,
    ) -> None:
        if len(out) >= path_cap:
            return

        if remaining_relations == 1:
            for relation in self._legacy_traversal_order(
                self.graph.pair_relations.get((current, dst), set()), rng
            ):
                if not self._edge_allowed(current, relation, dst, forbidden_edges):
                    continue
                out.add(prefix + (relation,))
                if len(out) >= path_cap:
                    return
            return

        next_hop = self.graph.out_adj_list.get(current, {})
        for relation in self._legacy_traversal_order(next_hop, rng):
            if len(out) >= path_cap:
                return
            for tail in self._legacy_traversal_order(
                next_hop[relation], rng, self.config.case_mid_cap
            ):
                if tail in seen or not self._edge_allowed(current, relation, tail, forbidden_edges):
                    continue
                self._extend_legacy_paths(
                    current=tail,
                    dst=dst,
                    forbidden_edges=forbidden_edges,
                    remaining_relations=remaining_relations - 1,
                    prefix=prefix + (relation,),
                    seen=seen | {tail},
                    out=out,
                    path_cap=path_cap,
                    rng=rng,
                )
                if len(out) >= path_cap:
                    return

    def _legacy_traversal_order(
        self,
        values,
        rng: random.Random,
        cap: Optional[int] = None,
    ) -> List[str]:
        """Order a bounded legacy-search branch without changing its semantics."""
        ordered = sorted(values)
        if self.config.path_budget_selection_mode == "seeded":
            rng.shuffle(ordered)
        if cap is not None:
            return ordered[:cap]
        return ordered

    def _discover_candidate_rules(self) -> Dict[str, Set[PathProgram]]:
        candidate_rules: Dict[str, Set[PathProgram]] = defaultdict(set)
        if self.config.per_pair_confidence:
            for (head, relation), true_tails in tqdm(
                self.graph.train_true_tails.items(), desc="Discover candidate rules", leave=True
            ):
                if not true_tails:
                    continue
                forbidden_edges = (
                    self._all_target_edges(head, relation, true_tails)
                    if self.config.strict_path_statistics
                    else frozenset()
                )
                tail_iter = sorted(true_tails)
                if not self._use_reference_path_semantics():
                    tail_iter = tail_iter[: self.config.case_mid_cap]
                for tail in tail_iter:
                    if forbidden_edges:
                        paths = self.find_paths_between_excluding(head, tail, forbidden_edges)
                    else:
                        paths = self.find_paths_between(head, tail)
                    candidate_rules[relation].update(paths)
        else:
            for head, relation, tail in tqdm(
                self.graph.train_aug, desc="Discover candidate rules", leave=True
            ):
                forbidden_edges = (
                    self._target_edge_pair(head, relation, tail)
                    if self.config.strict_path_statistics
                    else frozenset()
                )
                if forbidden_edges:
                    paths = self.find_paths_between_excluding(head, tail, forbidden_edges)
                else:
                    paths = self.find_paths_between(head, tail)
                candidate_rules[relation].update(paths)
        return candidate_rules

    def _compute_global_rule_statistics(self, candidate_rules: Dict[str, Set[PathProgram]]) -> None:
        for relation, paths in tqdm(candidate_rules.items(), desc="Global rule statistics", leave=True):
            for path in paths:
                key = (relation, path)
                supp = 0
                pred_count = 0
                hit_heads: Set[str] = set()
                source_heads = sorted(self.graph.relation_heads.get(path[0], set())) if path else []
                for head in source_heads:
                    outputs = self.execute_path(head, path)
                    if not outputs:
                        continue
                    output_set = set(outputs)
                    pred_count += len(output_set)
                    true_tails = self.graph.train_true_tails.get((head, relation), set())
                    if not true_tails:
                        continue
                    hits = output_set & true_tails
                    if not hits:
                        continue
                    if self.config.strict_path_statistics:
                        valid_hits = 0
                        for tail in hits:
                            forbidden_edges = self._target_edge_pair(head, relation, tail)
                            excluded_outputs = self.execute_path_excluding(head, path, forbidden_edges)
                            if tail in excluded_outputs:
                                valid_hits += 1
                        if valid_hits > 0:
                            supp += valid_hits
                            hit_heads.add(head)
                    else:
                        supp += len(hits)
                        hit_heads.add(head)
                self.path_support[key] = supp
                self.path_total_outputs[key] = pred_count
                self.path_head_support[key] = len(hit_heads)
                if supp < self.config.min_path_support or pred_count <= 0:
                    continue
                confidence = (supp + 1.0) / (pred_count + 2.0)
                self.path_confidence_scores[key] = confidence
                self.path_reliability_scores[key] = confidence * math.log1p(float(len(hit_heads)))

    def find_paths_between(self, src: str, dst: str) -> Tuple[PathProgram, ...]:
        cache_key = (src, dst)
        if cache_key in self._paths_between_cache:
            return self._paths_between_cache[cache_key]

        if self._use_uniform_sampled_discovery():
            out = self._find_paths_between_uniform_sampled(src, dst, frozenset())
            self._paths_between_cache[cache_key] = out
            return out

        if self._use_target_sampled_discovery():
            out = self._find_paths_between_target_sampled(src, dst, frozenset())
            self._paths_between_cache[cache_key] = out
            return out

        if self._use_reference_path_semantics():
            out = self._find_paths_between_reference(src, dst, frozenset())
            self._paths_between_cache[cache_key] = out
            return out

        out = self._find_paths_between_legacy(src, dst, frozenset())
        self._paths_between_cache[cache_key] = out
        return out

    def find_paths_between_excluding(
        self,
        src: str,
        dst: str,
        forbidden_edges: FrozenSet[Tuple[str, str, str]],
    ) -> Tuple[PathProgram, ...]:
        if self._use_uniform_sampled_discovery():
            return self._find_paths_between_uniform_sampled(src, dst, forbidden_edges)

        if self._use_target_sampled_discovery():
            return self._find_paths_between_target_sampled(src, dst, forbidden_edges)

        if self._use_reference_path_semantics():
            return self._find_paths_between_reference(src, dst, forbidden_edges)

        return self._find_paths_between_legacy(src, dst, forbidden_edges)

    def execute_path(self, src: str, path: PathProgram) -> Tuple[str, ...]:
        cache_key = (src, path)
        if cache_key in self._execute_path_cache:
            return self._execute_path_cache[cache_key]

        if self._use_oi_execution():
            out = self._execute_path_reference(src, path, frozenset())
            self._execute_path_cache[cache_key] = out
            return out

        if self._use_non_backtracking():
            out = self._execute_path_non_backtracking(src, path, frozenset())
            self._execute_path_cache[cache_key] = out
            return out

        frontier: Set[str] = {src}
        for relation in path:
            next_frontier: Set[str] = set()
            for entity in frontier:
                for tail in self.graph.out_adj.get(entity, {}).get(relation, set()):
                    next_frontier.add(tail)
                    if len(next_frontier) > self.config.execution_branch_cap:
                        self.path_execution_overflows += 1
                        self._execute_path_cache[cache_key] = tuple()
                        return tuple()
            frontier = next_frontier
            if not frontier:
                break

        out = tuple(sorted(frontier))
        self._execute_path_cache[cache_key] = out
        return out

    def execute_path_excluding(
        self,
        src: str,
        path: PathProgram,
        forbidden_edges: FrozenSet[Tuple[str, str, str]],
    ) -> Tuple[str, ...]:
        if self._use_oi_execution():
            return self._execute_path_reference(src, path, forbidden_edges)

        if self._use_non_backtracking():
            return self._execute_path_non_backtracking(src, path, forbidden_edges)

        frontier: Set[str] = {src}
        for relation in path:
            next_frontier: Set[str] = set()
            for entity in frontier:
                for tail in self.graph.out_adj.get(entity, {}).get(relation, set()):
                    if not self._edge_allowed(entity, relation, tail, forbidden_edges):
                        continue
                    next_frontier.add(tail)
                    if len(next_frontier) > self.config.execution_branch_cap:
                        self.path_execution_overflows += 1
                        return tuple()
            frontier = next_frontier
            if not frontier:
                break

        return tuple(sorted(frontier))

    def _execute_path_non_backtracking(
        self,
        src: str,
        path: PathProgram,
        forbidden_edges: FrozenSet[Tuple[str, str, str]],
    ) -> Tuple[str, ...]:
        """Boolean relation composition that forbids immediate backtracking.

        The frontier carries (previous, current) entity pairs and a step to
        ``tail`` is rejected when ``tail == previous`` (i.e. v_{i+1} != v_{i-1}).
        Longer cycles are still permitted; this only removes the reciprocal
        backtracking that Reverse-Edge Augmentation makes trivially available.
        The branch cap bounds the number of live (prev, cur) pairs per hop.
        """
        frontier: Set[Tuple[Optional[str], str]] = {(None, src)}
        for relation in path:
            next_frontier: Set[Tuple[str, str]] = set()
            for prev, cur in frontier:
                for tail in self.graph.out_adj.get(cur, {}).get(relation, set()):
                    if tail == prev:
                        continue
                    if forbidden_edges and not self._edge_allowed(
                        cur, relation, tail, forbidden_edges
                    ):
                        continue
                    next_frontier.add((cur, tail))
                    if len(next_frontier) > self.config.execution_branch_cap:
                        self.path_execution_overflows += 1
                        return tuple()
            frontier = next_frontier
            if not frontier:
                break

        return tuple(sorted({cur for _, cur in frontier}))

    def _find_paths_between_reference(
        self,
        src: str,
        dst: str,
        forbidden_edges: FrozenSet[Tuple[str, str, str]],
    ) -> Tuple[PathProgram, ...]:
        if src == dst:
            return tuple()

        paths_by_len: Dict[int, Set[PathProgram]] = defaultdict(set)
        first_hop = self.graph.out_adj_list.get(src, {})

        if self.config.enable_length1_rules and self.config.max_path_len >= 1:
            for relation in sorted(self.graph.pair_relations.get((src, dst), set())):
                if self._edge_allowed(src, relation, dst, forbidden_edges):
                    paths_by_len[1].add((relation,))

        if self.config.max_path_len >= 2:
            for r1 in sorted(first_hop):
                for mid1 in first_hop[r1]:
                    if mid1 in {src, dst}:
                        continue
                    if not self._edge_allowed(src, r1, mid1, forbidden_edges):
                        continue
                    for r2 in sorted(self.graph.pair_relations.get((mid1, dst), set())):
                        if not self._edge_allowed(mid1, r2, dst, forbidden_edges):
                            continue
                        paths_by_len[2].add((r1, r2))

        if self.config.max_path_len >= 3:
            target_predecessors = {
                mid2
                for rel_map in self.graph.in_adj_list.get(dst, {}).values()
                for mid2 in rel_map
                if mid2 not in {src, dst}
            }
            for r1 in sorted(first_hop):
                for mid1 in first_hop[r1]:
                    if mid1 in {src, dst}:
                        continue
                    if not self._edge_allowed(src, r1, mid1, forbidden_edges):
                        continue
                    second_hop = self.graph.out_adj_list.get(mid1, {})
                    for r2 in sorted(second_hop):
                        for mid2 in second_hop[r2]:
                            if mid2 in {src, dst, mid1} or mid2 not in target_predecessors:
                                continue
                            if not self._edge_allowed(mid1, r2, mid2, forbidden_edges):
                                continue
                            for r3 in sorted(self.graph.pair_relations.get((mid2, dst), set())):
                                if not self._edge_allowed(mid2, r3, dst, forbidden_edges):
                                    continue
                                paths_by_len[3].add((r1, r2, r3))

        return self._finalize_reference_paths(
            paths_by_len, src, dst, forbidden_edges
        )

    def _find_paths_between_target_sampled(
        self,
        src: str,
        dst: str,
        forbidden_edges: FrozenSet[Tuple[str, str, str]],
    ) -> Tuple[PathProgram, ...]:
        """PathBSR-specific AnyBURL-style discovery.

        Lengths 1--3 use exact target-aware joins. Lengths 4--5 use bounded
        endpoint-closing samples from either endpoint. Every discovered path is
        simple/OI-valid; sampling limits recall, not path validity.
        """
        if src == dst:
            return tuple()

        paths_by_len: Dict[int, Set[PathProgram]] = defaultdict(set)
        max_len = self.config.max_path_len

        if self.config.enable_length1_rules and max_len >= 1:
            for relation in self.graph.pair_relations.get((src, dst), set()):
                if self._edge_allowed(src, relation, dst, forbidden_edges):
                    paths_by_len[1].add((relation,))

        target_predecessors = {
            head
            for rel_map in self.graph.in_adj.get(dst, {}).values()
            for head in rel_map
            if head not in {src, dst}
        }

        if max_len >= 2:
            for r1, tails in self.graph.out_adj.get(src, {}).items():
                for mid1 in tails & target_predecessors:
                    if not self._edge_allowed(src, r1, mid1, forbidden_edges):
                        continue
                    for r2 in self.graph.pair_relations.get((mid1, dst), set()):
                        if self._edge_allowed(mid1, r2, dst, forbidden_edges):
                            paths_by_len[2].add((r1, r2))

        if max_len >= 3:
            for r1, first_tails in self.graph.out_adj.get(src, {}).items():
                for mid1 in first_tails:
                    if mid1 in {src, dst}:
                        continue
                    if not self._edge_allowed(src, r1, mid1, forbidden_edges):
                        continue
                    for r2, second_tails in self.graph.out_adj.get(mid1, {}).items():
                        for mid2 in second_tails & target_predecessors:
                            if mid2 in {src, dst, mid1}:
                                continue
                            if not self._edge_allowed(mid1, r2, mid2, forbidden_edges):
                                continue
                            for r3 in self.graph.pair_relations.get((mid2, dst), set()):
                                if self._edge_allowed(mid2, r3, dst, forbidden_edges):
                                    paths_by_len[3].add((r1, r2, r3))

        for length in range(4, max_len + 1):
            paths_by_len[length].update(
                self._sample_endpoint_closing_paths(src, dst, length, forbidden_edges)
            )

        selected: List[PathProgram] = []
        for length in range(1, max_len + 1):
            paths = paths_by_len.get(length, set())
            if not paths:
                continue
            selected.extend(
                self._sample_path_budget(paths, src, dst, length, forbidden_edges)
            )
        return tuple(selected)

    def _find_paths_between_uniform_sampled(
        self,
        src: str,
        dst: str,
        forbidden_edges: FrozenSet[Tuple[str, str, str]],
    ) -> Tuple[PathProgram, ...]:
        """Use one endpoint-closing sampling policy for every body length.

        Length one is the zero-prefix special case: the closing relation is
        looked up directly. Lengths two through five sample ``length - 1``
        simple/OI-valid steps and close the final edge to the other endpoint.
        """
        if src == dst:
            return tuple()

        paths_by_len: Dict[int, Set[PathProgram]] = defaultdict(set)
        max_len = self.config.max_path_len
        if self.config.enable_length1_rules and max_len >= 1:
            for relation in self.graph.pair_relations.get((src, dst), set()):
                if self._edge_allowed(src, relation, dst, forbidden_edges):
                    paths_by_len[1].add((relation,))

        for length in range(2, max_len + 1):
            paths_by_len[length].update(
                self._sample_endpoint_closing_paths(src, dst, length, forbidden_edges)
            )

        selected: List[PathProgram] = []
        for length in range(1, max_len + 1):
            paths = paths_by_len.get(length, set())
            if paths:
                selected.extend(
                    self._sample_path_budget(paths, src, dst, length, forbidden_edges)
                )
        return tuple(selected)

    def _sample_endpoint_closing_paths(
        self,
        src: str,
        dst: str,
        length: int,
        forbidden_edges: FrozenSet[Tuple[str, str, str]],
    ) -> Set[PathProgram]:
        attempts = self.config.path_sampling_attempts_per_length
        cap = self.config.path_search_cap_per_length
        if attempts <= 0 or cap <= 0:
            return set()

        rng = self._path_rng("walk", src, dst, length, forbidden_edges)
        sampled: Set[PathProgram] = set()
        for _ in range(attempts):
            self.path_sampling_attempts[length] += 1
            reverse_walk = bool(rng.getrandbits(1))
            start, goal = (dst, src) if reverse_walk else (src, dst)
            current = start
            visited = {start}
            relations: List[str] = []
            failed = False

            for _step in range(length - 1):
                eligible = [
                    (relation, tail)
                    for relation, tail in self._out_edges(current)
                    if tail != goal
                    and tail not in visited
                    and self._edge_allowed(current, relation, tail, forbidden_edges)
                ]
                if not eligible:
                    failed = True
                    break
                relation, tail = rng.choice(eligible)
                relations.append(relation)
                visited.add(tail)
                current = tail

            if failed or goal in visited:
                continue
            closing_relations = sorted(
                relation
                for relation in self.graph.pair_relations.get((current, goal), set())
                if self._edge_allowed(current, relation, goal, forbidden_edges)
            )
            if not closing_relations:
                continue
            relations.append(rng.choice(closing_relations))
            path = tuple(relations)
            if reverse_walk:
                path = tuple(self._reverse_relation(relation) for relation in reversed(path))
            self.path_sampling_closures[length] += 1
            before = len(sampled)
            sampled.add(path)
            if len(sampled) > before:
                self.path_sampling_unique[length] += 1
            if len(sampled) >= cap:
                break
        return sampled

    def _sample_path_budget(
        self,
        paths: Set[PathProgram],
        src: str,
        dst: str,
        length: int,
        forbidden_edges: FrozenSet[Tuple[str, str, str]],
    ) -> Tuple[PathProgram, ...]:
        cap = self.config.path_search_cap_per_length
        ordered = sorted(paths)
        if cap <= 0 or len(ordered) <= cap:
            return tuple(ordered if cap != 0 else ())
        rng = self._path_rng("budget", src, dst, length, forbidden_edges)
        return tuple(sorted(rng.sample(ordered, cap)))

    def _path_rng(
        self,
        purpose: str,
        src: str,
        dst: str,
        length: int,
        forbidden_edges: FrozenSet[Tuple[str, str, str]],
    ) -> random.Random:
        forbidden = "\x1e".join("\x1f".join(edge) for edge in sorted(forbidden_edges))
        payload = "\x1d".join(
            (
                str(self.config.path_sampling_seed),
                purpose,
                src,
                dst,
                str(length),
                forbidden,
            )
        ).encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        return random.Random(seed)

    def _out_edges(self, entity: str) -> Tuple[Tuple[str, str], ...]:
        cached = self._out_edges_cache.get(entity)
        if cached is not None:
            return cached
        edges = tuple(
            sorted(
                (relation, tail)
                for relation, tails in self.graph.out_adj.get(entity, {}).items()
                for tail in tails
            )
        )
        self._out_edges_cache[entity] = edges
        return edges

    def _finalize_reference_paths(
        self,
        paths_by_len: Dict[int, Set[PathProgram]],
        src: str,
        dst: str,
        forbidden_edges: FrozenSet[Tuple[str, str, str]],
    ) -> Tuple[PathProgram, ...]:
        paths: List[PathProgram] = []
        for length in sorted(paths_by_len):
            paths.extend(sorted(paths_by_len[length]))
        self.path_discovery_calls += 1
        self.path_discovery_max_candidates = max(
            self.path_discovery_max_candidates, len(paths)
        )
        cap = self.config.max_paths_per_case
        if cap > 0 and len(paths) > cap:
            self.path_discovery_overflow_cases += 1
            if self.config.path_budget_selection_mode == "seeded":
                rng = self._path_rng(
                    "reference-final-budget", src, dst, 0, forbidden_edges
                )
                paths = sorted(rng.sample(paths, cap))
            else:
                paths = paths[:cap]
        return tuple(paths)

    def _execute_path_reference(
        self,
        src: str,
        path: PathProgram,
        forbidden_edges: FrozenSet[Tuple[str, str, str]],
    ) -> Tuple[str, ...]:
        if self.config.path_semantics_mode != "budgeted_oi" and len(path) <= 3:
            return self._execute_short_simple_path(src, path, forbidden_edges)

        frontier: Set[Tuple[str, Tuple[str, ...]]] = {(src, (src,))}
        for relation in path:
            next_frontier: Set[Tuple[str, Tuple[str, ...]]] = set()
            for entity, visited in frontier:
                for tail in self.graph.out_adj_list.get(entity, {}).get(relation, []):
                    if not self._edge_allowed(entity, relation, tail, forbidden_edges):
                        continue
                    if tail in visited:
                        continue
                    next_frontier.add((tail, visited + (tail,)))
                    if (
                        self.config.path_semantics_mode == "budgeted_oi"
                        and len(next_frontier) > self.config.execution_branch_cap
                    ):
                        self.path_execution_overflows += 1
                        return tuple()
            frontier = next_frontier
            if not frontier:
                break

        return tuple(sorted({entity for entity, _ in frontier}))

    def _execute_short_simple_path(
        self,
        src: str,
        path: PathProgram,
        forbidden_edges: FrozenSet[Tuple[str, str, str]],
    ) -> Tuple[str, ...]:
        """Exact Object Identity execution specialized for body lengths 1--3."""
        if not path:
            return (src,)

        r1 = path[0]
        first_hop = self._allowed_tails(src, r1, forbidden_edges).difference({src})
        if len(path) == 1:
            return tuple(sorted(first_hop))

        outputs: Set[str] = set()
        r2 = path[1]
        for mid1 in first_hop:
            second_hop = self._allowed_tails(
                mid1, r2, forbidden_edges
            ).difference({src, mid1})
            if len(path) == 2:
                outputs.update(second_hop)
                continue

            r3 = path[2]
            for mid2 in second_hop:
                outputs.update(
                    self._allowed_tails(mid2, r3, forbidden_edges).difference(
                        {src, mid1, mid2}
                    )
                )

        return tuple(sorted(outputs))

    def _allowed_tails(
        self,
        head: str,
        relation: str,
        forbidden_edges: FrozenSet[Tuple[str, str, str]],
    ) -> Set[str]:
        tails = self.graph.out_adj.get(head, {}).get(relation, set())
        if not forbidden_edges or not tails:
            return tails
        blocked = {
            tail
            for edge_head, edge_relation, tail in forbidden_edges
            if edge_head == head and edge_relation == relation
        }
        return tails.difference(blocked) if blocked else tails

    def path_confidence(self, relation: str, path: PathProgram) -> float:
        if (relation, path) not in self.path_confidence_scores:
            return 0.0
        return min(max(float(self.path_confidence_scores[(relation, path)]), 0.0), 1.0)

    def path_answer_weight(self, relation: str, path: PathProgram) -> float:
        confidence = self.path_confidence(relation, path)
        if confidence <= 0 or self.config.rule_confidence_normalization == "none":
            return confidence
        maximum = self._relation_max_confidence_cache.get(relation)
        if maximum is None:
            maximum = max(
                (
                    float(value)
                    for (rule_relation, rule_path), value in self.path_confidence_scores.items()
                    if rule_relation == relation
                    and self.path_head_support_value(rule_relation, rule_path)
                    >= self.config.min_rule_head_support
                ),
                default=0.0,
            )
            self._relation_max_confidence_cache[relation] = maximum
        if maximum <= 0:
            return 0.0
        return min(confidence / maximum, 1.0)

    def path_head_support_value(self, relation: str, path: PathProgram) -> int:
        return int(self.path_head_support.get((relation, path), 0))

    def rule_quality(self, relation: str, path: PathProgram) -> float:
        confidence = self.path_confidence_scores.get((relation, path), 0.0)
        if confidence <= 0:
            return 0.0
        head_support = self.path_head_support_value(relation, path)
        if head_support < self.config.min_rule_head_support:
            return 0.0
        if self.config.rule_statistics_mode == "global":
            if self.config.rule_ordering_mode == "confidence":
                return float(confidence)
            if self.config.rule_ordering_mode == "reliability":
                return float(self.path_reliability_scores.get((relation, path), 0.0))
            if self.config.rule_ordering_mode in {"support_weighted", "legacy_support"}:
                return float(confidence) * math.log1p(float(self.path_support.get((relation, path), 0)))
            raise ValueError(f"Unknown rule_ordering_mode: {self.config.rule_ordering_mode}")

        if self.config.rule_ordering_mode == "confidence":
            return float(confidence)
        if self.config.rule_ordering_mode == "reliability":
            return float(self.path_reliability_scores.get((relation, path), 0.0))
        if self.config.rule_ordering_mode in {"support_weighted", "legacy_support"}:
            support = self.path_support.get((relation, path), 0)
            return float(confidence) * math.log1p(support)
        raise ValueError(f"Unknown rule_ordering_mode: {self.config.rule_ordering_mode}")

    def get_relation_rules(
        self,
        relation: str,
        limit: Optional[int] = None,
    ) -> Tuple[Tuple[PathProgram, float], ...]:
        max_paths = self.config.rule_library_topk if limit is None else int(limit)
        cache_key = (relation, max_paths)
        if cache_key in self._relation_rule_cache:
            return self._relation_rule_cache[cache_key]

        ranked: List[Tuple[PathProgram, float]] = []
        for rule_relation, path in self.path_confidence_scores.keys():
            if rule_relation != relation:
                continue
            quality = self.rule_quality(relation, path)
            if quality > 0:
                ranked.append((path, quality))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        out = tuple(ranked[:max_paths])
        self._relation_rule_cache[cache_key] = out
        return out

    def summary(self) -> Dict[str, float]:
        values = list(self.path_confidence_scores.values())
        len_counts = Counter(len(path) for _, path in self.path_confidence_scores.keys())
        relation_counts = Counter(relation for relation, _ in self.path_confidence_scores.keys())
        return {
            "rules": int(len(values)),
            "rules_len1": int(len_counts.get(1, 0)),
            "rules_len2": int(len_counts.get(2, 0)),
            "rules_len3": int(len_counts.get(3, 0)),
            "rules_len4": int(len_counts.get(4, 0)),
            "rules_len5": int(len_counts.get(5, 0)),
            "sampling_attempts_len2": int(self.path_sampling_attempts.get(2, 0)),
            "sampling_closures_len2": int(self.path_sampling_closures.get(2, 0)),
            "sampling_unique_len2": int(self.path_sampling_unique.get(2, 0)),
            "sampling_closure_rate_len2": (
                float(self.path_sampling_closures.get(2, 0)) / float(self.path_sampling_attempts[2])
                if self.path_sampling_attempts.get(2, 0) > 0
                else 0.0
            ),
            "sampling_attempts_len3": int(self.path_sampling_attempts.get(3, 0)),
            "sampling_closures_len3": int(self.path_sampling_closures.get(3, 0)),
            "sampling_unique_len3": int(self.path_sampling_unique.get(3, 0)),
            "sampling_closure_rate_len3": (
                float(self.path_sampling_closures.get(3, 0)) / float(self.path_sampling_attempts[3])
                if self.path_sampling_attempts.get(3, 0) > 0
                else 0.0
            ),
            "sampling_attempts_len4": int(self.path_sampling_attempts.get(4, 0)),
            "sampling_closures_len4": int(self.path_sampling_closures.get(4, 0)),
            "sampling_unique_len4": int(self.path_sampling_unique.get(4, 0)),
            "sampling_closure_rate_len4": (
                float(self.path_sampling_closures.get(4, 0)) / float(self.path_sampling_attempts[4])
                if self.path_sampling_attempts.get(4, 0) > 0
                else 0.0
            ),
            "sampling_attempts_len5": int(self.path_sampling_attempts.get(5, 0)),
            "sampling_closures_len5": int(self.path_sampling_closures.get(5, 0)),
            "sampling_unique_len5": int(self.path_sampling_unique.get(5, 0)),
            "sampling_closure_rate_len5": (
                float(self.path_sampling_closures.get(5, 0)) / float(self.path_sampling_attempts[5])
                if self.path_sampling_attempts.get(5, 0) > 0
                else 0.0
            ),
            "relations_with_rules": int(len(relation_counts)),
            "mean_rules_per_relation": float(np.mean(list(relation_counts.values()))) if relation_counts else 0.0,
            "mean_confidence": float(np.mean(values)) if values else 0.0,
            "max_confidence": float(np.max(values)) if values else 0.0,
            "mean_head_support": float(np.mean(list(self.path_head_support.values()))) if self.path_head_support else 0.0,
            "path_discovery_calls": int(self.path_discovery_calls),
            "path_discovery_overflow_cases": int(self.path_discovery_overflow_cases),
            "path_discovery_overflow_rate": (
                float(self.path_discovery_overflow_cases) / float(self.path_discovery_calls)
                if self.path_discovery_calls > 0
                else 0.0
            ),
            "path_discovery_cap_hit_cases": int(self.path_discovery_cap_hit_cases),
            "path_discovery_cap_hit_rate": (
                float(self.path_discovery_cap_hit_cases) / float(self.path_discovery_calls)
                if self.path_discovery_calls > 0
                else 0.0
            ),
            "path_discovery_max_candidates": int(self.path_discovery_max_candidates),
            "path_execution_overflows": int(self.path_execution_overflows),
        }

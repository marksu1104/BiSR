"""Candidate ranking with case-path scoring and local verification."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse

from .config import PathBSRConfig
from .graph import GraphStore
from .paths import RuleMiner
from .retrieval import ProxyRetriever


class CandidateRanker:
    """Score candidates and apply local structural verification."""

    def __init__(
        self,
        graph: GraphStore,
        retriever: ProxyRetriever,
        rule_miner: RuleMiner,
        config: PathBSRConfig,
    ) -> None:
        self.graph = graph
        self.retriever = retriever
        self.rule_miner = rule_miner
        self.config = config
        self._case_path_score_cache: Dict[Tuple[str, str, int], np.ndarray] = {}
        self._verified_score_cache: Dict[Tuple[str, str, int], np.ndarray] = {}
        self._bridge_score_cache: Dict[Tuple[str, str], float] = {}
        self._local_rule_support_cache: Dict[Tuple[str, str, Tuple[str, ...]], int] = {}
        self._answer_prior_prob_cache: Dict[str, np.ndarray] = {}
        self._answer_base_bm25_cache: Dict[str, np.ndarray] = {}
        self._adjacency_cache: Dict[str, "sparse.csr_matrix"] = {}
        self._gate_heads_cache: Dict[str, Dict[Tuple[str, ...], frozenset]] = {}
        self.verification_pairs_evaluated = 0
        self.verification_edge_cap_hits = 0
        self.verification_max_checked_edges = 0
        if not 1 <= self.config.verification_max_hops <= 5:
            raise ValueError("verification_max_hops must be in [1, 5]")

    def _answer_prior_prob(self, relation: str) -> np.ndarray:
        """Per-relation answer distribution P(c | r): count of each entity as a
        tail of ``relation`` in the augmented train graph, normalized to sum 1
        (0 for entities never observed as an answer of r)."""
        cached = self._answer_prior_prob_cache.get(relation)
        if cached is not None:
            return cached
        counts = np.zeros(len(self.graph.all_entities), dtype=np.float32)
        for head in self.graph.relation_heads.get(relation, set()):
            for tail in self.graph.out_adj.get(head, {}).get(relation, set()):
                idx = self.graph.ent2idx.get(tail)
                if idx is not None:
                    counts[idx] += 1.0
        total = float(counts.sum())
        if total > 0:
            counts /= total
        self._answer_prior_prob_cache[relation] = counts
        return counts

    def _reverse_relation(self, relation: str) -> str:
        suffix = self.config.reverse_suffix
        if relation.endswith(suffix):
            return relation[: -len(suffix)]
        return f"{relation}{suffix}"

    def _answer_base_bm25(self, relation: str) -> np.ndarray:
        """Global answer base from the SAME BM25 machinery as proxy retrieval, but
        querying the relation instead of the head.

        An entity c answers ``relation`` iff it carries the feature
        ``REL:<inverse-relation>`` (the reverse edge it gains when it is a tail of
        r). So `BM25(query={that token}, doc=c)` scores how characteristically c is
        an answer of r. BM25's IDF + document-length normalization down-weight
        promiscuous entities, so this is "typical answer of r" rather than "globally
        popular entity" — the principled version of the raw-frequency base.
        Normalized to sum 1 so it plugs in as the same unit (m=1) prior.
        """
        cached = self._answer_base_bm25_cache.get(relation)
        if cached is not None:
            return cached
        token = f"REL:{self._reverse_relation(relation)}"
        query = {token: 1.0}
        vec = np.zeros(len(self.graph.all_entities), dtype=np.float32)
        for entity in self.retriever.index.overlap_entities(query):
            idx = self.graph.ent2idx.get(entity)
            if idx is None:
                continue
            score = self.retriever.index.score(query, entity)
            if score > 0:
                vec[idx] = score
        total = float(vec.sum())
        if total > 0:
            vec /= total
        self._answer_base_bm25_cache[relation] = vec
        return vec

    def _apply_answer_base(self, scores: np.ndarray, relation: str) -> np.ndarray:
        """Uniform additive base: score(c) = evidence(c) + m * base(c | r).

        Dirichlet-inspired m=1 pseudo-count regularization: every entity starts
        from a base proportional to how typical an answer it is for r, and real
        evidence accumulates on top. If the weighted evidence is viewed as
        fractional counts, this is the unnormalized posterior numerator; its
        query-constant denominator is omitted because it cannot change ranks.
        One formula for all entities (no zero-fill special case), no free weight.
        When evidence is
        large the base is negligible; when evidence is 0 the ranking is the base.

        `answer_base_mode` selects the base distribution: "prob" = raw P(c|r)
        frequency; "bm25" = IDF/length-normalized BM25 over the relation token
        (same retrieval mechanism as the proxies, popularity-discounted).
        """
        if not self.config.use_answer_base:
            # `score` applies bridge verification in place. Returning the
            # cached case-score array here would therefore corrupt
            # `_case_path_score_cache` in the no-answer-base ablation.
            return scores.copy()
        # m = 1 pseudo-count (one prior observation), matching the Laplace +1 in
        # rule confidence. Fixed, not tuned — keeps the score free of any weight.
        if self.config.answer_base_mode == "bm25":
            return scores + self._answer_base_bm25(relation)
        return scores + self._answer_prior_prob(relation)

    def score_case_paths(self, head: str, relation: str, topk: Optional[int] = None) -> np.ndarray:
        topk = self.config.topk_proxy if topk is None else int(topk)
        cache_key = (head, relation, topk)
        if cache_key in self._case_path_score_cache:
            return self._case_path_score_cache[cache_key]

        if self.config.path_mode != "rule_library":
            raise ValueError(f"Unknown path_mode: {self.config.path_mode}")
        scores = self._score_rule_library_paths(head, relation, topk)

        self._case_path_score_cache[cache_key] = scores
        return scores

    def _score_rule_library_paths(self, head: str, relation: str, topk: int) -> np.ndarray:
        scores = np.zeros(len(self.graph.all_entities), dtype=np.float32)
        relation_rules = self.rule_miner.get_relation_rules(relation, self.config.rule_library_topk)
        query_outputs_cache: Dict[Tuple[str, ...], Tuple[str, ...]] = {}
        gate_heads = (
            self._gate_heads(relation)
            if (
                self.config.use_case_gate
                and self.config.batched_gate
                and self.config.path_semantics_mode in {"walk", "legacy"}
            )
            else None
        )

        for proxy_head, proxy_similarity in self.retriever.retrieve(head, relation, topk):
            proxy_answers = self.graph.out_adj.get(proxy_head, {}).get(relation, set())
            if not proxy_answers:
                continue

            # Abductive best-explanation: a candidate's rule evidence is its
            # single most-confident firing rule, not the sum. Rules in a case
            # run through overlapping subgraphs and are correlated, so summing
            # would double-count; max takes the strongest explanation and is
            # invariant to how many redundant rules happen to fire.
            rule_evidence: Dict[str, float] = defaultdict(float)
            for path, _ in relation_rules:
                if self.config.use_case_gate:
                    if gate_heads is not None:
                        if proxy_head not in gate_heads.get(path, frozenset()):
                            continue
                    elif self._local_rule_support(proxy_head, relation, path) <= 0:
                        continue
                rule_weight = self.rule_miner.path_answer_weight(relation, path)
                if rule_weight <= 0:
                    continue
                outputs = query_outputs_cache.get(path)
                if outputs is None:
                    outputs = self.rule_miner.execute_path(head, path)
                    query_outputs_cache[path] = outputs
                if not outputs:
                    continue
                for candidate_tail in outputs:
                    if rule_weight > rule_evidence[candidate_tail]:
                        rule_evidence[candidate_tail] = rule_weight

            # Proxy-answer prior (the case answered c directly via r) plus its
            # best rule explanation; independent cases accumulate across proxies
            # in _add_proxy_votes (kernel-weighted by BM25 similarity).
            proxy_votes: Dict[str, float] = defaultdict(float)
            for proxy_answer in proxy_answers:
                proxy_votes[proxy_answer] += 1.0
            for candidate_tail, evidence in rule_evidence.items():
                proxy_votes[candidate_tail] += evidence

            self._add_proxy_votes(scores, proxy_votes, proxy_similarity)
        return scores

    def _local_rule_support(self, proxy_head: str, relation: str, path: Tuple[str, ...]) -> int:
        cache_key = (proxy_head, relation, path)
        if cache_key in self._local_rule_support_cache:
            return self._local_rule_support_cache[cache_key]

        proxy_answers = self.graph.out_adj.get(proxy_head, {}).get(relation, set())
        if not proxy_answers:
            self._local_rule_support_cache[cache_key] = 0
            return 0

        full_outputs = set(self.rule_miner.execute_path(proxy_head, path))
        explainable_answers = full_outputs & proxy_answers
        if not explainable_answers:
            self._local_rule_support_cache[cache_key] = 0
            return 0

        support = 0
        for proxy_answer in explainable_answers:
            forbidden_edges = self._target_edge_pair(proxy_head, relation, proxy_answer)
            outputs = self.rule_miner.execute_path_excluding(proxy_head, path, forbidden_edges)
            if proxy_answer in outputs:
                support += 1

        self._local_rule_support_cache[cache_key] = support
        return support

    def _adjacency(self, relation: str) -> "sparse.csr_matrix":
        """Boolean entity x entity adjacency for ``relation`` over the augmented
        train graph (same edges as execute_path), as a CSR matrix. Cached."""
        cached = self._adjacency_cache.get(relation)
        if cached is not None:
            return cached
        e2i = self.graph.ent2idx
        n = len(self.graph.all_entities)
        rows: List[int] = []
        cols: List[int] = []
        for head in self.graph.relation_heads.get(relation, set()):
            hi = e2i.get(head)
            if hi is None:
                continue
            for tail in self.graph.out_adj.get(head, {}).get(relation, set()):
                ti = e2i.get(tail)
                if ti is not None:
                    rows.append(hi)
                    cols.append(ti)
        mat = sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n)
        )
        self._adjacency_cache[relation] = mat
        return mat

    def _gate_heads(self, relation: str) -> Dict[Tuple[str, ...], frozenset]:
        """For each top-k rule of ``relation``, the set of heads that pass the
        case gate (local_support > 0), computed once via batched sparse matrix
        products. Exact for rules whose body does not traverse the head relation
        (or its reverse); exact per-head fallback otherwise."""
        cached = self._gate_heads_cache.get(relation)
        if cached is not None:
            return cached
        rules = self.rule_miner.get_relation_rules(relation, self.config.rule_library_topk)
        heads = sorted(self.graph.relation_heads.get(relation, set()))
        result: Dict[Tuple[str, ...], frozenset] = {}
        if heads:
            e2i = self.graph.ent2idx
            head_idx = np.fromiter((e2i[h] for h in heads), dtype=np.int64, count=len(heads))
            n = len(self.graph.all_entities)
            selection = sparse.csr_matrix(
                (np.ones(len(heads), dtype=np.float32), (np.arange(len(heads)), head_idx)),
                shape=(len(heads), n),
            )
            answers_of_head = self._adjacency(relation)[head_idx, :]  # |H| x N
            reverse_relation = self._reverse_relation(relation)
            for path, _ in rules:
                if relation in path or reverse_relation in path:
                    result[path] = frozenset(
                        h for h in heads if self._local_rule_support(h, relation, path) > 0
                    )
                    continue
                reach = selection
                cap = self.config.execution_branch_cap
                for rel in path:
                    reach = reach.dot(self._adjacency(rel))
                    reach.eliminate_zeros()
                    reach.data[:] = 1.0  # binarize reachability
                    # replicate execute_path's branch cap: a source whose frontier
                    # exceeds the cap at any hop yields an empty path (gate fails).
                    over = np.nonzero(np.diff(reach.indptr) > cap)[0]
                    if over.size:
                        keep = np.ones(reach.shape[0], dtype=np.float32)
                        keep[over] = 0.0
                        reach = sparse.diags(keep).dot(reach)
                        reach.eliminate_zeros()
                overlap = reach.multiply(answers_of_head)
                mask = np.asarray(overlap.sum(axis=1)).ravel() > 0
                result[path] = frozenset(heads[i] for i in np.nonzero(mask)[0])
        self._gate_heads_cache[relation] = result
        return result

    def _target_edge_pair(self, head: str, relation: str, tail: str) -> frozenset[Tuple[str, str, str]]:
        if relation.endswith(self.config.reverse_suffix):
            reverse_relation = relation[: -len(self.config.reverse_suffix)]
        else:
            reverse_relation = f"{relation}{self.config.reverse_suffix}"
        return frozenset({
            (head, relation, tail),
            (tail, reverse_relation, head),
        })

    def _add_proxy_votes(
        self,
        scores: np.ndarray,
        proxy_votes: Dict[str, float],
        proxy_similarity: float,
    ) -> None:
        if not proxy_votes:
            return
        if self.config.proxy_vote_normalization == "max":
            vote_norm = max(proxy_votes.values())
            if vote_norm <= 0:
                return
        elif self.config.proxy_vote_normalization == "none":
            vote_norm = 1.0
        else:
            raise ValueError(f"Unknown proxy_vote_normalization: {self.config.proxy_vote_normalization}")
        for candidate_tail, vote in proxy_votes.items():
            idx = self.graph.ent2idx.get(candidate_tail)
            if idx is not None:
                scores[idx] += float(proxy_similarity) * float(vote) / vote_norm

    def score(self, head: str, relation: str, topk: Optional[int] = None) -> np.ndarray:
        topk = self.config.topk_proxy if topk is None else int(topk)
        cache_key = (head, relation, topk)
        if cache_key in self._verified_score_cache:
            return self._verified_score_cache[cache_key]

        case_scores = self.score_case_paths(head, relation, topk)
        if self.config.verification_top_m <= 0:
            verified = self._apply_answer_base(case_scores, relation)
            self._verified_score_cache[cache_key] = verified
            return verified

        candidate_score = self._apply_answer_base(case_scores, relation)
        candidate_indices = self._top_positive_candidate_indices(
            candidate_score if self.config.verification_selection_mode == "candidate_score" else case_scores,
            self.config.verification_top_m,
        )
        if not candidate_indices:
            self._verified_score_cache[cache_key] = candidate_score
            return candidate_score

        bridge_scores: Dict[int, float] = {}
        max_bridge = 0.0
        for idx in candidate_indices:
            candidate_tail = self.graph.all_entities[idx]
            verification = self.verification_value(head, candidate_tail)
            if verification > 0:
                bridge_scores[idx] = verification
                max_bridge = max(max_bridge, verification)
        if max_bridge <= 0:
            self._verified_score_cache[cache_key] = candidate_score
            return candidate_score

        # The full estimator: score(c) = (pi(c|r) + case_evidence(c)) * (1 + rho(h,c)).
        # Product-of-experts verification — structural proximity confirms the
        # (prior + case) estimate multiplicatively; bridge=0 leaves it unchanged,
        # strong structure amplifies it. Parameter-free, no scale to tune.
        verified = candidate_score.copy()
        for idx, bridge in bridge_scores.items():
            if self.config.verification_norm_mode == "max":
                norm_bridge = bridge / max_bridge
            elif self.config.verification_norm_mode == "bounded":
                norm_bridge = bridge / (1.0 + bridge)
            else:
                raise ValueError(f"Unknown verification_norm_mode: {self.config.verification_norm_mode}")
            verified[idx] = float(verified[idx]) * (1.0 + norm_bridge)
        self._verified_score_cache[cache_key] = verified
        return verified

    def _top_positive_candidate_indices(self, score_vec: np.ndarray, top_m: int) -> List[int]:
        positive = np.flatnonzero(score_vec > 0)
        if len(positive) == 0 or top_m <= 0:
            return []
        if len(positive) > top_m:
            values = score_vec[positive]
            kth = len(values) - top_m
            threshold = np.partition(values, kth)[kth]
            above = positive[values > threshold]
            tied = positive[values == threshold]
            needed = top_m - len(above)
            # flatnonzero follows entity-index order; all_entities is lexical.
            # This gives the top-M boundary an explicit, stable tie-break.
            positive = np.concatenate((above, tied[:needed]))
        return sorted(positive.tolist(), key=lambda idx: (-float(score_vec[idx]), self.graph.all_entities[idx]))

    def verification_value(self, head: str, candidate_tail: str) -> float:
        """A relation-agnostic local structural verifier: a degree-normalized,
        hand-designed proximity between head and candidate over the undirected
        train graph, truncated at ``config.verification_max_hops``. Each hop adds
        one term:

        - hop 1: direct edge between head and candidate (binary).
        - hop 2: a common-neighbour term in the Adamic–Adar family,
          Σ 1/log2(deg(w)+2) over shared neighbours w (Adamic & Adar 2003).
        - hop 3: a degree-normalized length-3 simple-path count,
          Σ 1/sqrt(deg(w1)*deg(w2)) over head→w1→w2→candidate with (w1,w2)∈E.
        - optional hops 4--5: a strict continuation of the hop-3 degree penalty:
          every internal node contributes a deg(w)^(-1/2) factor.

        NB: this is a purpose-built verifier, NOT a standard Katz score
        (Σ β^k A^k) nor a spectral diffusion (D^-1/2 A D^-1/2)^k — the hop terms
        use different normalizations and it is applied multiplicatively only to
        the top case-scored candidates. Describe it honestly as a local
        structural verifier; do not claim a textbook diffusion identity.
        """
        cache_key = (head, candidate_tail)
        if cache_key in self._bridge_score_cache:
            return self._bridge_score_cache[cache_key]
        if head == candidate_tail:
            self._bridge_score_cache[cache_key] = 0.0
            return 0.0

        self.verification_pairs_evaluated += 1

        max_hops = self.config.verification_max_hops
        if max_hops < 1:
            self._bridge_score_cache[cache_key] = 0.0
            return 0.0
        edge_cap = self.config.bridge_edge_cap
        if edge_cap < 0:
            raise ValueError("verification edge cap must be non-negative")

        query_neighbors = self.graph.undirected_neighbors.get(head, set())
        candidate_neighbors = self.graph.undirected_neighbors.get(candidate_tail, set())
        if not query_neighbors or not candidate_neighbors:
            self._bridge_score_cache[cache_key] = 0.0
            return 0.0

        raw = 0.0
        # hop 1: direct edge
        if candidate_tail in query_neighbors:
            raw += 1.0

        query_side = query_neighbors - {candidate_tail}
        candidate_side = candidate_neighbors - {head}

        # hop 2: Adamic-Adar over shared neighbours
        if max_hops >= 2 and query_side and candidate_side:
            for bridge in query_side & candidate_side:
                degree = max(self.graph.undirected_degree.get(bridge, 1), 1)
                raw += 1.0 / math.log2(degree + 2.0)

        # hop 3: symmetric degree-normalized length-3 path count
        if max_hops >= 3 and query_side and candidate_side:
            checked_edges = 0
            query_bridges = sorted(query_side, key=lambda node: (self.graph.undirected_degree.get(node, 0), node))
            for query_bridge in query_bridges:
                query_degree = max(self.graph.undirected_degree.get(query_bridge, 1), 1)
                # The edge cap makes traversal order observable. Set iteration
                # depends on PYTHONHASHSEED, so sort to keep scores reproducible
                # across processes and machines.
                candidate_bridges = sorted(
                    self.graph.undirected_neighbors.get(query_bridge, set()) & candidate_side
                )
                for candidate_bridge in candidate_bridges:
                    if candidate_bridge == query_bridge:
                        continue
                    candidate_degree = max(self.graph.undirected_degree.get(candidate_bridge, 1), 1)
                    raw += 1.0 / math.sqrt(query_degree * candidate_degree)
                    checked_edges += 1
                    if edge_cap > 0 and checked_edges >= edge_cap:
                        self.verification_edge_cap_hits += 1
                        break
                if edge_cap > 0 and checked_edges >= edge_cap:
                    break
            self.verification_max_checked_edges = max(
                self.verification_max_checked_edges, checked_edges
            )

        # hop 4: h -> w1 -> w2 -> w3 -> t. Endpoint-side joins avoid a
        # breadth-first expansion over the whole graph. Object identities are
        # checked explicitly so only simple paths contribute.
        if max_hops >= 4 and query_side and candidate_side:
            checked_paths = 0
            stop = False
            for w1 in sorted(query_side):
                degree1 = max(self.graph.undirected_degree.get(w1, 1), 1)
                for w3 in sorted(candidate_side):
                    if w3 == w1:
                        continue
                    degree3 = max(self.graph.undirected_degree.get(w3, 1), 1)
                    middle_nodes = (
                        self.graph.undirected_neighbors.get(w1, set())
                        & self.graph.undirected_neighbors.get(w3, set())
                    )
                    for w2 in sorted(middle_nodes):
                        if w2 in {head, candidate_tail, w1, w3}:
                            continue
                        degree2 = max(self.graph.undirected_degree.get(w2, 1), 1)
                        raw += 1.0 / math.sqrt(degree1 * degree2 * degree3)
                        checked_paths += 1
                        if edge_cap > 0 and checked_paths >= edge_cap:
                            self.verification_edge_cap_hits += 1
                            stop = True
                            break
                    if stop:
                        break
                if stop:
                    break
            self.verification_max_checked_edges = max(
                self.verification_max_checked_edges, checked_paths
            )

        # hop 5: h -> w1 -> w2 -> w3 -> w4 -> t. This is a target-closing
        # join, not a global BFS. It is exact when verification_edge_cap=0.
        if max_hops >= 5 and query_side and candidate_side:
            checked_paths = 0
            stop = False
            for w1 in sorted(query_side):
                degree1 = max(self.graph.undirected_degree.get(w1, 1), 1)
                second_nodes = self.graph.undirected_neighbors.get(w1, set())
                for w2 in sorted(second_nodes):
                    if w2 in {head, candidate_tail, w1}:
                        continue
                    degree2 = max(self.graph.undirected_degree.get(w2, 1), 1)
                    for w3 in sorted(self.graph.undirected_neighbors.get(w2, set())):
                        if w3 in {head, candidate_tail, w1, w2}:
                            continue
                        closing_nodes = (
                            self.graph.undirected_neighbors.get(w3, set()) & candidate_side
                        )
                        for w4 in sorted(closing_nodes):
                            if w4 in {head, candidate_tail, w1, w2, w3}:
                                continue
                            degree3 = max(self.graph.undirected_degree.get(w3, 1), 1)
                            degree4 = max(self.graph.undirected_degree.get(w4, 1), 1)
                            raw += 1.0 / math.sqrt(
                                degree1 * degree2 * degree3 * degree4
                            )
                            checked_paths += 1
                            if edge_cap > 0 and checked_paths >= edge_cap:
                                self.verification_edge_cap_hits += 1
                                stop = True
                                break
                        if stop:
                            break
                    if stop:
                        break
                if stop:
                    break
            self.verification_max_checked_edges = max(
                self.verification_max_checked_edges, checked_paths
            )

        score = math.log1p(raw)
        self._bridge_score_cache[cache_key] = score
        return score

    def bridge_score(self, head: str, candidate_tail: str) -> float:
        """Backward-compatible alias for :meth:`verification_value`."""
        return self.verification_value(head, candidate_tail)

    def summary(self) -> Dict[str, float]:
        return {
            "verification_pairs_evaluated": int(self.verification_pairs_evaluated),
            "verification_edge_cap_hits": int(self.verification_edge_cap_hits),
            "verification_edge_cap_hit_rate": (
                float(self.verification_edge_cap_hits)
                / float(self.verification_pairs_evaluated)
                if self.verification_pairs_evaluated > 0
                else 0.0
            ),
            "verification_max_checked_edges": int(self.verification_max_checked_edges),
        }

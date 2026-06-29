"""Configuration for the paper-facing PathBSR model."""

from __future__ import annotations

from dataclasses import dataclass


REVERSE_SUFFIX = "__reverse"
SEED = 42
BM25_K1 = 1.2
BM25_B = 0.5


@dataclass(frozen=True)
class PathBSRConfig:
    """PathBSR hyperparameters and ablation switches.

    The defaults define the paper-facing PathBSR configuration. Non-default
    values are also part of the model API so validation ablations can be run
    without maintaining separate analysis-only retriever implementations.
    """

    topk_proxy: int = 100
    reverse_suffix: str = REVERSE_SUFFIX
    entity_feature_mode: str = "all_tf"
    # Proxy Retrieval similarity. "bm25" is the paper-facing default; "cosine"
    # and "overlap" are controlled ablation alternatives over the same Entity
    # Structural Features.
    proxy_similarity_mode: str = "bm25"
    bm25_query_tf_mode: str = "log"
    bm25_normalize_proxy_scores: bool = True
    bm25_idf_floor: float = 0.01

    min_path_support: int = 2
    # Maximum mined body length. The current implementation supports 1--5.
    max_path_len: int = 3
    # Included in the paper-facing model after the six-dataset validation
    # comparison. Keep the CLI disable switch for legacy reproduction/ablation.
    enable_length1_rules: bool = True
    # Fixed-length relation-sequence reachability (walk semantics). The exact
    # Object Identity executors remain available as validation/reference modes.
    path_semantics_mode: str = "walk"
    # Budget for optional sampled discovery branches used in ablations.
    path_sampling_attempts_per_length: int = 32
    path_sampling_seed: int = SEED
    # How bounded Path Mining chooses among overflowing branches.
    # deterministic_first preserves the reproducible bounded traversal;
    # seeded is the tested robustness alternative.
    path_budget_selection_mode: str = "deterministic_first"
    rule_statistics_mode: str = "case_empirical"
    # support_weighted = confidence * log(1 + per-fact support)
    # confidence = confidence only
    # reliability = confidence * log(1 + distinct successful heads)
    rule_ordering_mode: str = "reliability"
    # Path Answer weight. relation_max keeps the estimated within-relation
    # confidence ordering but calibrates the strongest rule to unit weight.
    rule_confidence_normalization: str = "none"
    min_rule_head_support: int = 0
    # rule-confidence statistical unit: False = per training fact (a
    # (head,relation) with K answers counts a path up to K times); True = per
    # (head,relation,path) — one observation per head, excluding all the head's
    # answer edges. The cleaner probability; removes high-cardinality over-weight.
    per_pair_confidence: bool = False
    # Search at most 8 programs at each path length, then retain at most 16
    # programs across enabled lengths. This makes both budgets explicit.
    max_paths_per_case: int = 16
    path_search_cap_per_length: int = 8
    case_mid_cap: int = 32
    execution_branch_cap: int = 200
    path_mode: str = "rule_library"
    # Rule budget per relation. 128 (up from 32) lets rule-rich relations use
    # their deeper rules. Validation improved on WD-singer and WN18RR and was
    # effectively flat on the checked FB settings; ranking metrics are not
    # claimed to be mathematically monotone in this budget.
    rule_library_topk: int = 128
    strict_path_statistics: bool = True

    verification_top_m: int = 100
    # Hops 4--5 are implemented as optional experimental extensions. The
    # validated paper-facing default remains three hops.
    verification_max_hops: int = 3
    # Zero means exact/unlimited. The previous cap of 2000 affected only 0.074%
    # of FB15K-237-50 verification pairs and did not improve validation MRR.
    bridge_edge_cap: int = 0

    # Frequency Answers: score(c) = local_score(c) + P(c|r). This is a
    # Dirichlet-inspired m=1 pseudo-count regularizer. If local contributions are
    # viewed as fractional counts, it is the unnormalized posterior numerator; the
    # query-constant denominator is irrelevant to ranking. No free weight: when
    # local support is present the unit prior is negligible; when a query has no
    # proxy/path answer the ranking falls back to the relation's answer prior. This is
    # a permanent part of the scoring model; the flag exists only for ablation.
    # per-case rule gate: a rule contributes through a retrieved proxy only if it
    # leak-free explains >=1 of that proxy's own answers (case-localized rule
    # relevance). False = all top-k rules apply to every proxy (global rules).
    # Kept as a flag to re-measure whether the gate still helps under the current
    # MAX aggregation + Frequency Answers (it was added before both).
    use_case_gate: bool = True
    # speed: compute the per-case rule gate by batched sparse matrix products,
    # once per relation, instead of per-(proxy,rule) path executions. Exact for
    # rules whose body does not traverse the head relation (the vast majority);
    # exact fallback otherwise. Off = the original per-call gate.
    batched_gate: bool = False

    # The field names below are retained for Python API backward
    # compatibility. Paper-facing terminology is Frequency Answers and Proxy
    # Answer normalization; the public CLI exposes only those names.
    use_answer_base: bool = True
    # "prob" = P(c|r) frequency; "bm25" is a legacy ablation branch.
    answer_base_mode: str = "prob"
    proxy_vote_normalization: str = "max"
    verification_selection_mode: str = "candidate_score"
    verification_norm_mode: str = "max"

    @property
    def use_frequency_answers(self) -> bool:
        """Canonical name for the retained backward-compatible field."""
        return self.use_answer_base

    @property
    def frequency_answer_mode(self) -> str:
        """Canonical name for the retained backward-compatible field."""
        return self.answer_base_mode

    @property
    def proxy_answer_normalization(self) -> str:
        """Canonical name for the retained backward-compatible field."""
        return self.proxy_vote_normalization

    @property
    def verification_edge_cap(self) -> int:
        """Canonical name for the retained backward-compatible field."""
        return self.bridge_edge_cap

DEFAULT_CONFIG = PathBSRConfig()


def config_to_dict(config: PathBSRConfig) -> dict[str, object]:
    values = dict(config.__dict__)
    values["use_frequency_answers"] = values.pop("use_answer_base")
    values["frequency_answer_mode"] = values.pop("answer_base_mode")
    values["proxy_answer_normalization"] = values.pop("proxy_vote_normalization")
    values["verification_edge_cap"] = values.pop("bridge_edge_cap")
    return values

#!/usr/bin/env python3
"""Run PathBSR on one or more datasets."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

def find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "datasets").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the PathBSR repository root")


ROOT = find_repo_root()
sys.path.insert(0, str(ROOT / "src"))

from pathbsr import DEFAULT_CONFIG, PathBSR, load_dataset_with_audit, remove_train_overlap  # noqa: E402


def log(message: str) -> None:
    print(f"[PathBSR] {message}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m pathbsr.cli", description="Evaluate PathBSR.")
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--deoverlap-eval",
        action="store_true",
        help="Sensitivity only: remove evaluation triples that occur exactly in train.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--progress-steps",
        type=int,
        default=10,
        help="Print evaluation progress this many times per dataset. Use 0 to disable.",
    )

    parser.add_argument("--topk-proxy", type=int, default=DEFAULT_CONFIG.topk_proxy)
    parser.add_argument(
        "--entity-feature-mode",
        choices=["rel", "rel_ent", "all_binary", "all_tf"],
        default=DEFAULT_CONFIG.entity_feature_mode,
    )
    parser.add_argument(
        "--bm25-query-tf-mode",
        choices=["binary", "log"],
        default=DEFAULT_CONFIG.bm25_query_tf_mode,
    )
    bm25_normalization_group = parser.add_mutually_exclusive_group()
    bm25_normalization_group.add_argument(
        "--bm25-normalize-proxy-scores",
        dest="bm25_normalize_proxy_scores",
        action="store_true",
    )
    bm25_normalization_group.add_argument(
        "--no-bm25-normalize-proxy-scores",
        dest="bm25_normalize_proxy_scores",
        action="store_false",
    )
    parser.set_defaults(
        bm25_normalize_proxy_scores=DEFAULT_CONFIG.bm25_normalize_proxy_scores
    )
    parser.add_argument("--bm25-idf-floor", type=float, default=DEFAULT_CONFIG.bm25_idf_floor)
    parser.add_argument("--min-path-support", type=int, default=DEFAULT_CONFIG.min_path_support)
    parser.add_argument("--max-path-len", type=int, default=DEFAULT_CONFIG.max_path_len)
    length1_group = parser.add_mutually_exclusive_group()
    length1_group.add_argument(
        "--enable-length1-rules", dest="enable_length1_rules", action="store_true"
    )
    length1_group.add_argument(
        "--disable-length1-rules", dest="enable_length1_rules", action="store_false"
    )
    parser.set_defaults(enable_length1_rules=DEFAULT_CONFIG.enable_length1_rules)
    parser.add_argument(
        "--path-semantics-mode",
        choices=[
            "walk",
            "non_backtracking",
            "legacy",
            "legacy_oi",
            "budgeted_oi",
            "reference",
            "target_sampled",
            "uniform_sampled",
        ],
        default=DEFAULT_CONFIG.path_semantics_mode,
    )
    parser.add_argument(
        "--path-sampling-attempts-per-length",
        type=int,
        default=DEFAULT_CONFIG.path_sampling_attempts_per_length,
    )
    parser.add_argument("--path-sampling-seed", type=int, default=DEFAULT_CONFIG.path_sampling_seed)
    parser.add_argument(
        "--path-budget-selection-mode",
        choices=["deterministic_first", "legacy_first", "seeded"],
        default=DEFAULT_CONFIG.path_budget_selection_mode,
    )
    parser.add_argument("--max-paths-per-case", type=int, default=DEFAULT_CONFIG.max_paths_per_case)
    parser.add_argument("--path-search-cap-per-length", type=int, default=DEFAULT_CONFIG.path_search_cap_per_length)
    parser.add_argument("--case-mid-cap", type=int, default=DEFAULT_CONFIG.case_mid_cap)
    parser.add_argument("--execution-branch-cap", type=int, default=DEFAULT_CONFIG.execution_branch_cap)
    parser.add_argument("--rule-library-topk", type=int, default=DEFAULT_CONFIG.rule_library_topk)
    parser.add_argument(
        "--rule-statistics-mode",
        choices=["case_empirical", "legacy", "global"],
        default=DEFAULT_CONFIG.rule_statistics_mode,
    )
    parser.add_argument(
        "--rule-ordering-mode",
        choices=["support_weighted", "legacy_support", "confidence", "reliability"],
        default=DEFAULT_CONFIG.rule_ordering_mode,
    )
    parser.add_argument("--min-rule-head-support", type=int, default=DEFAULT_CONFIG.min_rule_head_support)
    parser.add_argument(
        "--rule-confidence-normalization",
        choices=["none", "relation_max"],
        default=DEFAULT_CONFIG.rule_confidence_normalization,
    )
    parser.add_argument(
        "--rule-observation-unit",
        choices=["fact", "head_relation"],
        default="head_relation" if DEFAULT_CONFIG.per_pair_confidence else "fact",
        help=(
            "Statistical unit for case-empirical rule confidence: each training fact, "
            "or one observation per (head, relation, path)."
        ),
    )
    parser.add_argument(
        "--no-frequency-answers",
        dest="use_answer_base",
        action="store_false",
    )
    parser.add_argument(
        "--no-answer-base",
        dest="use_answer_base",
        action="store_false",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--no-case-gate", dest="use_case_gate", action="store_false")
    parser.add_argument("--batched-gate", dest="batched_gate", action="store_true")
    parser.set_defaults(use_answer_base=DEFAULT_CONFIG.use_answer_base, use_case_gate=DEFAULT_CONFIG.use_case_gate,
                        batched_gate=DEFAULT_CONFIG.batched_gate)
    parser.add_argument(
        "--frequency-answer-mode",
        dest="answer_base_mode",
        choices=["prob", "bm25"],
        default=DEFAULT_CONFIG.answer_base_mode,
    )
    parser.add_argument(
        "--answer-base-mode",
        dest="answer_base_mode",
        choices=["prob", "bm25"],
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--proxy-answer-normalization",
        dest="proxy_vote_normalization",
        choices=["max", "none"],
        default=DEFAULT_CONFIG.proxy_vote_normalization,
    )
    parser.add_argument(
        "--proxy-vote-normalization",
        dest="proxy_vote_normalization",
        choices=["max", "none"],
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--verification-top-m", type=int, default=DEFAULT_CONFIG.verification_top_m)
    parser.add_argument(
        "--verification-max-hops",
        type=int,
        choices=range(1, 6),
        default=DEFAULT_CONFIG.verification_max_hops,
    )
    parser.add_argument(
        "--verification-selection-mode",
        choices=["case_score", "candidate_score"],
        default=DEFAULT_CONFIG.verification_selection_mode,
    )
    parser.add_argument(
        "--verification-norm-mode",
        choices=["max", "bounded"],
        default=DEFAULT_CONFIG.verification_norm_mode,
    )
    parser.add_argument(
        "--verification-edge-cap",
        dest="bridge_edge_cap",
        type=int,
        default=DEFAULT_CONFIG.bridge_edge_cap,
        metavar="N",
    )
    parser.add_argument(
        "--bridge-edge-cap",
        dest="bridge_edge_cap",
        type=int,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def print_audit_summary(dataset_name: str, audit: dict[str, object]) -> None:
    splits = audit.get("splits", {})
    overlap = audit.get("overlap", {})
    duplicate_parts = []
    for split_name in ("train", "valid", "test"):
        split_audit = splits.get(split_name, {})
        duplicate_count = split_audit.get("duplicate_count", 0)
        if duplicate_count:
            duplicate_parts.append(f"{split_name} duplicates={duplicate_count}")
    overlap_parts = []
    for pair_name in ("train_valid", "train_test", "valid_test"):
        pair_audit = overlap.get(pair_name, {})
        overlap_count = pair_audit.get("count", 0)
        if overlap_count:
            overlap_parts.append(f"{pair_name} overlap={overlap_count}")
    if duplicate_parts or overlap_parts:
        joined = ", ".join(duplicate_parts + overlap_parts)
        print(f"[dataset-audit] {dataset_name}: {joined}", file=sys.stderr)


def write_result_rows(output: Path, rows: list[dict[str, object]]) -> None:
    """Atomically checkpoint completed dataset rows during long sweeps."""
    if not rows:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    fieldnames = list(rows[0].keys())
    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)


def main() -> None:
    args = parse_args()
    config = replace(
        DEFAULT_CONFIG,
        topk_proxy=args.topk_proxy,
        entity_feature_mode=args.entity_feature_mode,
        bm25_query_tf_mode=args.bm25_query_tf_mode,
        bm25_normalize_proxy_scores=args.bm25_normalize_proxy_scores,
        bm25_idf_floor=args.bm25_idf_floor,
        min_path_support=args.min_path_support,
        max_path_len=args.max_path_len,
        enable_length1_rules=args.enable_length1_rules,
        path_semantics_mode=args.path_semantics_mode,
        path_sampling_attempts_per_length=args.path_sampling_attempts_per_length,
        path_sampling_seed=args.path_sampling_seed,
        path_budget_selection_mode=args.path_budget_selection_mode,
        max_paths_per_case=args.max_paths_per_case,
        path_search_cap_per_length=args.path_search_cap_per_length,
        case_mid_cap=args.case_mid_cap,
        execution_branch_cap=args.execution_branch_cap,
        rule_library_topk=args.rule_library_topk,
        rule_statistics_mode=args.rule_statistics_mode,
        rule_ordering_mode=args.rule_ordering_mode,
        min_rule_head_support=args.min_rule_head_support,
        rule_confidence_normalization=args.rule_confidence_normalization,
        per_pair_confidence=args.rule_observation_unit == "head_relation",
        use_answer_base=args.use_answer_base,
        use_case_gate=args.use_case_gate,
        batched_gate=args.batched_gate,
        answer_base_mode=args.answer_base_mode,
        proxy_vote_normalization=args.proxy_vote_normalization,
        verification_top_m=args.verification_top_m,
        verification_max_hops=args.verification_max_hops,
        verification_selection_mode=args.verification_selection_mode,
        verification_norm_mode=args.verification_norm_mode,
        bridge_edge_cap=args.bridge_edge_cap,
    )

    rows: list[dict[str, object]] = []
    total_datasets = len(args.dataset)
    output_label = str(args.output) if args.output is not None else "stdout-only"
    log(
        f"starting run: datasets={total_datasets}, split={args.split}, "
        f"output={output_label}"
    )
    for dataset_index, dataset_name in enumerate(args.dataset, start=1):
        prefix = f"[{dataset_index}/{total_datasets}] {dataset_name}"
        dataset_start = time.time()
        log(f"{prefix}: loading dataset")
        train, valid, test, audit = load_dataset_with_audit(args.data_root, dataset_name)
        print_audit_summary(dataset_name, audit)
        log(
            f"{prefix}: loaded train={len(train):,}, valid={len(valid):,}, "
            f"test={len(test):,}"
        )
        log(f"{prefix}: building model")
        start = time.time()
        model = PathBSR(train, valid, test, config=config)
        build_sec = time.time() - start
        log(f"{prefix}: model built in {build_sec:.1f}s")
        official_eval_split = valid if args.split == "valid" else test
        if args.deoverlap_eval:
            eval_split, removed_overlap = remove_train_overlap(train, official_eval_split)
            evaluation_variant = "train_overlap_removed"
        else:
            eval_split = official_eval_split
            removed_overlap = 0
            evaluation_variant = "official"
        eval_examples = len(eval_split) if args.max_examples is None else min(args.max_examples, len(eval_split))
        eval_queries = 2 * eval_examples
        progress_every = None
        if args.progress_steps > 0 and eval_examples > 0:
            progress_every = max(1, eval_examples // args.progress_steps)
        log(
            f"{prefix}: evaluating {eval_examples:,} triples "
            f"({eval_queries:,} bidirectional queries), variant={evaluation_variant}"
        )
        eval_start = time.time()
        metrics = model.evaluate(
            eval_split,
            split_name=f"{dataset_name}-{args.split}",
            max_examples=args.max_examples,
            progress_every=progress_every,
        )
        eval_sec = time.time() - eval_start
        rule_summary = model.rule_summary()
        row = {
            "dataset": dataset_name,
            "split": args.split,
            "evaluation_variant": evaluation_variant,
            "removed_train_overlap": removed_overlap,
            "rule_library_topk": config.rule_library_topk,
            "build_sec": build_sec,
            "eval_sec": eval_sec,
            "queries_per_sec": (eval_queries / eval_sec) if eval_sec > 0 else 0.0,
            **rule_summary,
            **metrics,
        }
        rows.append(row)
        if args.output is not None:
            write_result_rows(args.output, rows)
            log(f"{prefix}: checkpointed {len(rows):,} row(s) to {args.output}")
        print(
            f"{dataset_name} {args.split}: "
            f"mrr={metrics['mrr']:.6f} "
            f"tail_mrr={metrics.get('tail_mrr', float('nan')):.6f} "
            f"tailopt_mrr={metrics.get('tailopt_mrr', float('nan')):.6f} "
            f"h@1={metrics['hits@1']:.6f} "
            f"h@3={metrics['hits@3']:.6f} "
            f"h@10={metrics['hits@10']:.6f} "
            f"rules={rule_summary['rules']:,} "
            f"build_sec={build_sec:.1f} "
            f"eval_sec={eval_sec:.1f} "
            f"qps={(eval_queries / eval_sec) if eval_sec > 0 else 0.0:.1f}"
        )
        log(f"{prefix}: finished in {time.time() - dataset_start:.1f}s")
    log("run finished")


if __name__ == "__main__":
    main()

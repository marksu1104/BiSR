#!/usr/bin/env python3
"""PathBSR validation experiments and thesis artifacts.

This is the thesis-facing experiment script. It extends the reporting helpers by
actually running the missing validation ablations that are safe to compute from
the current codebase:

* w/o Path Answers (rule library top-k = 0; proxy/frequency/verification stay on)
* w/o Path Verification (verification_top_m = 0)
* proxy retrieval with cosine similarity
* proxy retrieval with feature-overlap similarity

The script does not invent external baselines. AnyBURL is included in structural
figures only if prediction files or rows already exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing analysis dependency. Install the paper-analysis environment with:\n"
        "  python3 -m pip install -r requirements.txt"
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathbsr import DEFAULT_CONFIG, PathBSR, load_dataset  # noqa: E402
import pathbsr.model as pathbsr_model_module  # noqa: E402
import pathbsr.paths as pathbsr_paths_module  # noqa: E402
from pathbsr.retrieval import build_proxy_index  # noqa: E402
from pathbsr.exp import (  # noqa: E402
    DATASETS,
    MAIN_PROTOCOL,
    WD_CAVEAT,
    case_studies,
    dataset_statistics,
    fmt_float,
    write_case_markdown,
    write_csv,
    write_latex_table,
)


RESULTS = ROOT / "results"
ABLATION_DIR = RESULTS / "ablation"
DATASET_STATS_DIR = RESULTS / "dataset_statistics"
PATH_COUNT_DIR = RESULTS / "path_count"
CARDINALITY_DIR = RESULTS / "cardinality"
CASE_DIR = RESULTS / "case_studies"
RUNS = RESULTS / "runs"
EVALUATION_PROTOCOL_DIR = RESULTS / "evaluation_protocol"
RUN_METRICS = ABLATION_DIR / "pathbsr_validation_ablation_metrics.csv"
FINAL_TEST_METRICS = RUNS / "pathbsr_best_model_test.csv"
PROTOCOL_SCORE_TABLE = EVALUATION_PROTOCOL_DIR / "pathbsr_protocol_score_tables.csv"
PATHBSR_VALID_DETAIL = RUNS / "pathbsr_valid_detail.csv"
STRUCTURAL_FEATURES = RESULTS / "cache" / "structural_valid_features.csv"
STRUCTURAL_LONG = RESULTS / "cache" / "structural_valid_long.csv"
OBSOLETE_RUN_VARIANTS = {"proxy_jaccard", "proxy_tfidf_cosine"}

ABLATION_DATASETS = DATASETS
RUN_VARIANTS: dict[str, dict[str, Any]] = {
    "full_default": {
        "group": "current_model",
        "label": "Full PathBSR",
        "config": DEFAULT_CONFIG,
        "rebuild_model": True,
        "controlled_scope": "paper-facing PathBSR default configuration",
    },
    "feature_rel": {
        "group": "entity_feature",
        "label": "REL",
        "config": replace(DEFAULT_CONFIG, entity_feature_mode="rel"),
        "rebuild_model": True,
        "controlled_scope": "same PathBSR config except entity_feature_mode='rel'",
    },
    "feature_rel_ent": {
        "group": "entity_feature",
        "label": "REL+ENT",
        "config": replace(DEFAULT_CONFIG, entity_feature_mode="rel_ent"),
        "rebuild_model": True,
        "controlled_scope": "same PathBSR config except entity_feature_mode='rel_ent'",
    },
    "feature_all_binary": {
        "group": "entity_feature",
        "label": "REL+ENT+REL-ENT binary",
        "config": replace(DEFAULT_CONFIG, entity_feature_mode="all_binary"),
        "rebuild_model": True,
        "controlled_scope": "same PathBSR config except entity_feature_mode='all_binary'",
    },
    "feature_all_tf": {
        "group": "entity_feature",
        "label": "REL+ENT+REL-ENT TF",
        "config": DEFAULT_CONFIG,
        "rebuild_model": True,
        "controlled_scope": "same PathBSR config; default TF-weighted Entity Structural Features",
    },
    "core_no_path_answers": {
        "group": "core_component",
        "label": "w/o Path Answers",
        "config": replace(DEFAULT_CONFIG, rule_library_topk=0),
        "rebuild_model": False,
        "controlled_scope": "same PathBSR config except rule_library_topk=0",
    },
    "core_no_frequency_answers": {
        "group": "core_component",
        "label": "w/o Frequency Answers",
        "config": replace(DEFAULT_CONFIG, use_answer_base=False),
        "rebuild_model": False,
        "controlled_scope": "same PathBSR config except Frequency Answers disabled",
    },
    "core_no_verification": {
        "group": "core_component",
        "label": "w/o Path Verification",
        "config": replace(DEFAULT_CONFIG, verification_top_m=0),
        "rebuild_model": False,
        "controlled_scope": "same PathBSR config except verification_top_m=0",
    },
    "proxy_cosine": {
        "group": "proxy_similarity",
        "label": "cosine",
        "config": replace(DEFAULT_CONFIG, proxy_similarity_mode="cosine"),
        "rebuild_model": False,
        "controlled_scope": "same PathBSR config except proxy_similarity_mode='cosine'",
    },
    "proxy_overlap": {
        "group": "proxy_similarity",
        "label": "overlap",
        "config": replace(DEFAULT_CONFIG, proxy_similarity_mode="overlap"),
        "rebuild_model": False,
        "controlled_scope": "same PathBSR config except proxy_similarity_mode='overlap'",
    },
}
COMPUTED_VARIANTS = list(RUN_VARIANTS)
PATH_BUCKET_ORDER = ["0-10", "11+"]
FEW_PATH_BUCKET = "0-10"
CARDINALITY_ORDER = ["N-to-1", "N-to-N"]
MODEL_ORDER = ["TransE", "ConvE", "AnyBURL", "HoGRN", "PathBSR"]
MODEL_COLORS = {
    "TransE": "#1f77b4",
    "ConvE": "#ff7f0e",
    "AnyBURL": "#2ca02c",
    "HoGRN": "#d62728",
    "PathBSR": "#9467bd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PathBSR paper experiment artifacts.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete cached validation ablation metrics and rerun them.",
    )
    parser.add_argument(
        "--skip-expensive",
        action="store_true",
        help="Do not run missing validation ablations; only refresh tables from existing rows.",
    )
    parser.add_argument(
        "--skip-structural",
        action="store_true",
        help="Skip structural tables and figures that require scripts/structural_analysis.py outputs.",
    )
    return parser.parse_args()


def silence_internal_progress() -> None:
    """Disable tqdm progress bars from the model internals for batch artifacts."""
    quiet = lambda iterable, **_: iterable
    pathbsr_model_module.tqdm = quiet
    pathbsr_paths_module.tqdm = quiet


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing_rows = read_csv(path)
        fieldnames = list(existing_rows[0].keys()) if existing_rows else []
        changed = False
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
                changed = True
        if changed:
            write_csv(path, existing_rows, fieldnames=fieldnames)
    else:
        fieldnames = list(row.keys())
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)


def metric_row(dataset: str, variant: str, group: str, metrics: dict[str, float], build_sec: float, eval_sec: float) -> dict[str, Any]:
    return {
        "analysis_group": group,
        "variant": variant,
        "dataset": dataset,
        "split": "valid",
        "protocol": MAIN_PROTOCOL,
        "mrr": metrics["mrr"],
        "hits@1": metrics["hits@1"],
        "hits@3": metrics["hits@3"],
        "hits@10": metrics["hits@10"],
        "num_queries": metrics["num_queries"],
        "tail_mrr": metrics.get("tail_mrr", ""),
        "head_mrr": metrics.get("head_mrr", ""),
        "build_sec_shared_model": build_sec,
        "eval_sec": eval_sec,
        "variant_label": "",
        "controlled_scope": "",
        "wd_overlap_caveat": WD_CAVEAT if dataset == "WD-singer" else "",
    }


def set_model_config(model: PathBSR, config: Any, index: Any) -> None:
    """Mutate the model for analysis variants and clear dependent caches."""
    model.config = config
    model.retriever.config = config
    model.rule_miner.config = config
    model.ranker.config = config
    model.retriever.index = index
    model.retriever._proxy_cache.clear()
    model.retriever._direct_score_cache.clear()
    model.rule_miner._relation_rule_cache.clear()
    model.ranker._case_path_score_cache.clear()
    model.ranker._verified_score_cache.clear()
    model.ranker._bridge_score_cache.clear()
    model.ranker._local_rule_support_cache.clear()
    model.ranker._answer_prior_prob_cache.clear()
    model.ranker._answer_base_bm25_cache.clear()
    model.ranker._gate_heads_cache.clear()


def existing_computed_keys() -> set[tuple[str, str]]:
    return {(row["variant"], row["dataset"]) for row in read_csv(RUN_METRICS)}


def prune_obsolete_run_metrics() -> None:
    rows = read_csv(RUN_METRICS)
    if not rows:
        return
    kept = [row for row in rows if row.get("variant") not in OBSOLETE_RUN_VARIANTS]
    if len(kept) == len(rows):
        return
    write_csv(RUN_METRICS, kept, fieldnames=list(rows[0].keys()))


def run_missing_validation_ablations(force: bool = False) -> None:
    if force and RUN_METRICS.exists():
        RUN_METRICS.unlink()
    done = existing_computed_keys()

    for dataset in ABLATION_DATASETS:
        needed = [(variant, dataset) for variant in COMPUTED_VARIANTS if (variant, dataset) not in done]
        if not needed:
            print(f"[pathbsr-experiments] skip {dataset}: validation runs already exist", flush=True)
            continue

        train, valid, test = load_dataset(ROOT / "datasets", dataset)
        shared_model: PathBSR | None = None
        shared_build_sec = 0.0
        shared_indexes: dict[str, Any] = {}

        def get_shared_model() -> tuple[PathBSR, float, dict[str, Any]]:
            nonlocal shared_model, shared_build_sec, shared_indexes
            if shared_model is None:
                build_start = time.time()
                shared_model = PathBSR(train, valid, test, config=DEFAULT_CONFIG)
                shared_build_sec = time.time() - build_start
                shared_indexes["bm25"] = shared_model.retriever.index
                for mode in ("cosine", "overlap"):
                    cfg = replace(DEFAULT_CONFIG, proxy_similarity_mode=mode)
                    shared_indexes[mode] = build_proxy_index(shared_model.entity_features, cfg)
            return shared_model, shared_build_sec, shared_indexes

        for variant in COMPUTED_VARIANTS:
            if (variant, dataset) in done:
                continue
            spec = RUN_VARIANTS[variant]
            config = spec["config"]
            group = spec["group"]
            print(f"[pathbsr-experiments] evaluate {dataset} / {variant}", flush=True)
            if spec["rebuild_model"]:
                build_start = time.time()
                model = PathBSR(train, valid, test, config=config)
                build_sec = time.time() - build_start
            else:
                model, build_sec, indexes = get_shared_model()
                index = indexes.get(config.proxy_similarity_mode, indexes["bm25"])
                set_model_config(model, config, index)
            start = time.time()
            metrics = model.evaluate(valid, split_name=f"{dataset}-{variant}")
            eval_sec = time.time() - start
            row = metric_row(dataset, variant, group, metrics, build_sec, eval_sec)
            row["variant_label"] = spec["label"]
            row["controlled_scope"] = spec["controlled_scope"]
            append_csv(RUN_METRICS, row)
            done.add((variant, dataset))
            print(f"[pathbsr-experiments] {dataset} / {variant}: MRR={metrics['mrr']:.6f}", flush=True)


def computed_metrics() -> dict[tuple[str, str], dict[str, str]]:
    return {(row["variant"], row["dataset"]): row for row in read_csv(RUN_METRICS)}


def reverse_relation(relation: str) -> str:
    suffix = DEFAULT_CONFIG.reverse_suffix
    if relation.endswith(suffix):
        return relation[: -len(suffix)]
    return f"{relation}{suffix}"


def ensure_pathbsr_valid_detail(force: bool = False) -> None:
    """Generate query-level PathBSR validation ranks used by structural figures."""
    if PATHBSR_VALID_DETAIL.exists() and not force:
        print(f"[pathbsr-experiments] use existing {PATHBSR_VALID_DETAIL.relative_to(ROOT)}", flush=True)
        return

    rows: list[dict[str, Any]] = []
    for dataset in ("NELL23K", "WD-singer"):
        print(f"[pathbsr-experiments] generate PathBSR valid detail for {dataset}", flush=True)
        train, valid, test = load_dataset(ROOT / "datasets", dataset)
        model = PathBSR(train, valid, test, config=DEFAULT_CONFIG)
        total_queries = 2 * len(valid)
        completed = 0
        progress_every = max(1, len(valid) // 10)
        for triple_index, (head, relation, tail) in enumerate(valid):
            query_specs = (
                ("tail", triple_index * 2, head, relation, tail),
                ("head", triple_index * 2 + 1, tail, reverse_relation(relation), head),
            )
            for direction, query_index, query_head, query_relation, target in query_specs:
                scores = model.score(query_head, query_relation)
                rank = model.filtered_rank(scores, (query_head, query_relation), target)
                rows.append(
                    {
                        "dataset": dataset,
                        "split": "valid",
                        "query_index": query_index,
                        "direction": direction,
                        "original_h": head,
                        "original_r": relation,
                        "original_t": tail,
                        "query_h": query_head,
                        "query_r": query_relation,
                        "target": target,
                        "bsr_rank": rank,
                        "bsr_rr": 1.0 / rank,
                    }
                )
                completed += 1
            if (triple_index + 1) % progress_every == 0 or (triple_index + 1) == len(valid):
                print(
                    f"[pathbsr-experiments] {dataset}: valid detail "
                    f"{completed:,}/{total_queries:,} queries",
                    flush=True,
                )
    write_csv(PATHBSR_VALID_DETAIL, rows)
    print(f"[pathbsr-experiments] wrote {PATHBSR_VALID_DETAIL.relative_to(ROOT)}", flush=True)


def run_metric(computed: dict[tuple[str, str], dict[str, str]], variant: str, dataset: str) -> dict[str, str] | None:
    return computed.get((variant, dataset))


def run_source_note(row: dict[str, str] | None) -> str:
    return str(RUN_METRICS.relative_to(ROOT)) if row else ""


def write_complete_proxy_tables() -> None:
    computed = computed_metrics()
    rows: list[dict[str, Any]] = []
    for dataset in ABLATION_DATASETS:
        for variant, label, controlled in [
            ("full_default", "BM25", "reference"),
            ("proxy_cosine", "cosine", "yes"),
            ("proxy_overlap", "overlap", "yes"),
        ]:
            row = run_metric(computed, variant, dataset)
            spec = RUN_VARIANTS[variant]
            rows.append(
                {
                    "similarity": label,
                    "dataset": dataset,
                    "mrr": fmt_float(row["mrr"]) if row else "",
                    "hits@1": fmt_float(row["hits@1"]) if row else "",
                    "hits@3": fmt_float(row["hits@3"]) if row else "",
                    "hits@10": fmt_float(row["hits@10"]) if row else "",
                    "status": "available" if row else "missing",
                    "controlled_comparison": controlled if row else "no",
                    "controlled_scope": spec["controlled_scope"] if row else "not computed",
                    "source": run_source_note(row),
                    "wd_overlap_caveat": WD_CAVEAT if dataset == "WD-singer" else "",
                }
            )
    write_csv(ABLATION_DIR / "proxy_similarity_validation.csv", rows)
    write_latex_table(
        ABLATION_DIR / "proxy_similarity_validation.tex",
        rows,
        ["similarity", "dataset", "mrr", "hits@1", "hits@3", "hits@10", "status", "controlled_comparison"],
        "Proxy similarity ablation on the validation split.",
        "tab:pathbsr-proxy-similarity-ablation-detail",
    )
    pivot = []
    for label in ["BM25", "cosine", "overlap"]:
        item = {"similarity": label}
        vals = []
        for dataset in ABLATION_DATASETS:
            match = next(row for row in rows if row["similarity"] == label and row["dataset"] == dataset)
            item[dataset] = match["mrr"]
            if match["mrr"]:
                vals.append(float(match["mrr"]))
        item["Macro"] = fmt_float(sum(vals) / len(vals)) if vals else ""
        pivot.append(item)
    write_csv(ABLATION_DIR / "proxy_similarity_mrr_validation.csv", pivot)
    write_latex_table(
        ABLATION_DIR / "proxy_similarity_mrr_validation.tex",
        pivot,
        ["similarity", *ABLATION_DATASETS, "Macro"],
        "Proxy similarity ablation on the validation split.",
        "tab:pathbsr-proxy-similarity-ablation",
    )


def write_complete_core_tables() -> None:
    computed = computed_metrics()
    rows: list[dict[str, Any]] = []
    variants = [
        ("Full PathBSR", "full_default", "reference"),
        ("w/o Path Answers", "core_no_path_answers", "yes"),
        ("w/o Frequency Answers", "core_no_frequency_answers", "yes"),
        ("w/o Path Verification", "core_no_verification", "yes"),
    ]
    for label, key, controlled in variants:
        for dataset in ABLATION_DATASETS:
            source = run_metric(computed, key, dataset)
            full = run_metric(computed, "full_default", dataset)
            full_mrr = float(full["mrr"]) if full and full.get("mrr") else None
            mrr = float(source["mrr"]) if source and source.get("mrr") else None
            spec = RUN_VARIANTS[key]
            rows.append(
                {
                    "variant": label,
                    "dataset": dataset,
                    "mrr": fmt_float(mrr) if mrr is not None else "",
                    "hits@1": fmt_float(source["hits@1"]) if source and source.get("hits@1") else "",
                    "hits@3": fmt_float(source["hits@3"]) if source and source.get("hits@3") else "",
                    "hits@10": fmt_float(source["hits@10"]) if source and source.get("hits@10") else "",
                    "delta_mrr_vs_full": fmt_float(mrr - full_mrr) if mrr is not None and full_mrr is not None else "",
                    "status": "available" if source else "missing",
                    "controlled_comparison": controlled,
                    "controlled_scope": spec["controlled_scope"],
                    "protocol": MAIN_PROTOCOL,
                    "source": run_source_note(source),
                    "wd_overlap_caveat": WD_CAVEAT if dataset == "WD-singer" else "",
                }
            )
    write_csv(ABLATION_DIR / "core_components_validation.csv", rows)
    write_latex_table(
        ABLATION_DIR / "core_components_validation.tex",
        rows,
        ["variant", "dataset", "mrr", "hits@1", "hits@3", "hits@10", "delta_mrr_vs_full", "status"],
        "Core component ablation on the validation split.",
        "tab:pathbsr-core-component-ablation-detail",
    )
    pivot = []
    for label, key, controlled in variants:
        item = {"variant": label}
        vals = []
        for dataset in ABLATION_DATASETS:
            match = next(row for row in rows if row["variant"] == label and row["dataset"] == dataset)
            item[dataset] = match["mrr"]
            if match["mrr"]:
                vals.append(float(match["mrr"]))
        item["Macro"] = fmt_float(sum(vals) / len(vals)) if vals else ""
        item["controlled_comparison"] = controlled
        item["controlled_scope"] = RUN_VARIANTS[key]["controlled_scope"]
        pivot.append(item)
    write_csv(ABLATION_DIR / "core_components_mrr_validation.csv", pivot)
    write_latex_table(
        ABLATION_DIR / "core_components_mrr_validation.tex",
        pivot,
        ["variant", *ABLATION_DATASETS, "Macro"],
        "Core component ablation on the validation split.",
        "tab:pathbsr-core-component-ablation",
    )


def write_base_tables() -> None:
    stats = dataset_statistics()
    write_csv(DATASET_STATS_DIR / "dataset_statistics.csv", stats)
    write_latex_table(
        DATASET_STATS_DIR / "dataset_statistics.tex",
        stats,
        ["Dataset", "#Entities", "#Relations", "#Train", "#Valid", "#Test", "avg train degree", "duplicate note", "overlap note"],
        "Dataset statistics for PathBSR experiments.",
        "tab:pathbsr-dataset-statistics",
    )

    computed = computed_metrics()
    entity_variants = [
        ("feature_rel", "REL"),
        ("feature_rel_ent", "REL+ENT"),
        ("feature_all_binary", "REL+ENT+REL-ENT binary"),
        ("feature_all_tf", "REL+ENT+REL-ENT TF"),
    ]
    entity_rows: list[dict[str, Any]] = []
    for key, label in entity_variants:
        spec = RUN_VARIANTS[key]
        for dataset in ABLATION_DATASETS:
            row = run_metric(computed, key, dataset)
            entity_rows.append(
                {
                    "variant": label,
                    "dataset": dataset,
                    "mrr": fmt_float(row["mrr"]) if row else "",
                    "hits@1": fmt_float(row["hits@1"]) if row else "",
                    "hits@3": fmt_float(row["hits@3"]) if row else "",
                    "hits@10": fmt_float(row["hits@10"]) if row else "",
                    "status": "available" if row else "missing",
                    "controlled_comparison": "yes" if row else "no",
                    "controlled_scope": spec["controlled_scope"] if row else "not computed",
                    "protocol": MAIN_PROTOCOL,
                    "source": run_source_note(row),
                    "wd_overlap_caveat": WD_CAVEAT if dataset == "WD-singer" else "",
                }
            )
    write_csv(ABLATION_DIR / "entity_features_validation.csv", entity_rows)
    pivot = []
    entity_scope = {label: RUN_VARIANTS[key]["controlled_scope"] for key, label in entity_variants}
    for _, variant in entity_variants:
        item = {"variant": variant}
        vals = []
        for dataset in ABLATION_DATASETS:
            match = next(row for row in entity_rows if row.get("variant") == variant and row.get("dataset") == dataset)
            item[dataset] = match["mrr"]
            if match["mrr"]:
                vals.append(float(match["mrr"]))
        item["Macro"] = fmt_float(sum(vals) / len(vals)) if vals else ""
        item["controlled_scope"] = entity_scope[variant]
        pivot.append(item)
    write_csv(ABLATION_DIR / "entity_features_mrr_validation.csv", pivot)
    write_latex_table(
        ABLATION_DIR / "entity_features_mrr_validation.tex",
        pivot,
        ["variant", *ABLATION_DATASETS, "Macro"],
        "Entity Structural Features ablation on the validation split.",
        "tab:pathbsr-entity-feature-ablation",
    )


def combined_path_bucket(bucket: str) -> str | None:
    if bucket in {"0", "1", "2-10"}:
        return "0-10"
    if bucket in {"11-100", ">100"}:
        return "11+"
    return None


def require_structural_cache() -> None:
    missing = [path for path in (STRUCTURAL_FEATURES, STRUCTURAL_LONG) if not path.is_file()]
    if missing:
        missing_text = "\n".join(f"  - {path.relative_to(ROOT)}" for path in missing)
        raise FileNotFoundError(
            "Missing structural-analysis cache files:\n"
            f"{missing_text}\n\n"
            "Generate them first with:\n"
            "  PYTHONPATH=src python3 scripts/structural_analysis.py\n\n"
            "If you only want ablation tables and case studies for now, rerun with:\n"
            "  PYTHONPATH=src python3 scripts/pathbsr_experiments.py --skip-structural"
        )


def structural_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build NELL23K/WD-singer structural inputs with updated AnyBURL rows."""
    require_structural_cache()
    long_rows = read_csv(STRUCTURAL_LONG)
    feature_rows = read_csv(STRUCTURAL_FEATURES)
    target_datasets = {"NELL23K", "WD-singer"}

    feature_by_query: dict[tuple[str, str, str], dict[str, str]] = {}
    features_by_dataset: dict[str, list[dict[str, str]]] = {}
    for row in feature_rows:
        dataset = row["dataset"]
        if dataset not in target_datasets:
            continue
        path_bucket = combined_path_bucket(row["path_bucket"])
        if path_bucket is None:
            continue
        enriched = dict(row)
        enriched["combined_path_bucket"] = path_bucket
        features_by_dataset.setdefault(dataset, []).append(enriched)
        feature_by_query[(dataset, row["query_index"], row["direction"])] = enriched

    share_path_counts: dict[tuple[str, str], tuple[int, int]] = {}
    share_card_counts: dict[tuple[str, str], tuple[int, int]] = {}
    for dataset, rows in features_by_dataset.items():
        total = len(rows)
        for bucket in PATH_BUCKET_ORDER:
            count = sum(1 for row in rows if row["combined_path_bucket"] == bucket)
            share_path_counts[(dataset, bucket)] = (count, total)
        for bucket in CARDINALITY_ORDER:
            count = sum(1 for row in rows if row["rel_type"] == bucket)
            share_card_counts[(dataset, bucket)] = (count, total)
    missing_datasets = sorted(target_datasets.difference(features_by_dataset))
    if missing_datasets:
        raise ValueError(
            "Structural cache is incomplete. Missing target dataset rows for: "
            f"{missing_datasets}. Regenerate with scripts/structural_analysis.py."
        )

    rr_by_path: dict[tuple[str, str, str], list[float]] = {}
    rr_by_card: dict[tuple[str, str, str], list[float]] = {}

    def add_rr(dataset: str, model: str, feature: dict[str, str], rr: float) -> None:
        path_bucket = feature["combined_path_bucket"]
        rel_type = feature["rel_type"]
        rr_by_path.setdefault((dataset, path_bucket, model), []).append(rr)
        if rel_type in CARDINALITY_ORDER:
            rr_by_card.setdefault((dataset, rel_type, model), []).append(rr)

    for row in long_rows:
        dataset = row["dataset"]
        model = row["model"]
        if dataset not in target_datasets or model not in {"TransE", "ConvE", "HoGRN"}:
            continue
        path_bucket = combined_path_bucket(row["path_bucket"])
        if path_bucket is None:
            continue
        feature = {
            "combined_path_bucket": path_bucket,
            "rel_type": row["rel_type"],
        }
        add_rr(dataset, model, feature, float(row["rr"]))

    if PATHBSR_VALID_DETAIL.exists():
        for row in read_csv(PATHBSR_VALID_DETAIL):
            dataset = row["dataset"]
            if dataset not in target_datasets:
                continue
            feature = feature_by_query.get((dataset, row["query_index"], row["direction"]))
            if feature is None:
                continue
            add_rr(dataset, "PathBSR", feature, float(row["bsr_rr"]))

    for dataset in sorted(target_datasets):
        anyburl_path = ROOT / "external_predictions" / "valid_predictions" / "AnyBURL" / dataset / "valid_query_summary.csv"
        if not anyburl_path.exists():
            continue
        for row in read_csv(anyburl_path):
            feature = feature_by_query.get((dataset, row["query_index"], row["direction"]))
            if feature is None:
                continue
            rank = float(row["filtered_rank"])
            if rank <= 0:
                continue
            add_rr(dataset, "AnyBURL", feature, 1.0 / rank)

    def mean_or_blank(vals: list[float]) -> str:
        return fmt_float(sum(vals) / len(vals), 6) if vals else ""

    path_rows: list[dict[str, Any]] = []
    for dataset in sorted(target_datasets):
        for bucket in PATH_BUCKET_ORDER:
            count, total = share_path_counts.get((dataset, bucket), (0, len(features_by_dataset.get(dataset, []))))
            for model in MODEL_ORDER:
                vals = rr_by_path.get((dataset, bucket, model), [])
                path_rows.append(
                    {
                        "dataset": dataset,
                        "bucket": bucket,
                        "definition": "0-10 merges original buckets 0, 1, 2-10; 11+ merges 11-100 and >100; validation split",
                        "query_count": count,
                        "total_queries": total,
                        "query_share_pct": fmt_float(100.0 * count / total if total else 0.0, 3),
                        "model": model,
                        "mrr": mean_or_blank(vals),
                        "model_query_count": len(vals),
                        "split": "valid",
                        "protocol": "structural validation analysis; source definitions in structural-analysis notes",
                        "wd_overlap_caveat": WD_CAVEAT if dataset == "WD-singer" else "",
                    }
                )

    card_rows: list[dict[str, Any]] = []
    for dataset in sorted(target_datasets):
        for bucket in CARDINALITY_ORDER:
            count, total = share_card_counts.get((dataset, bucket), (0, len(features_by_dataset.get(dataset, []))))
            for model in MODEL_ORDER:
                vals = rr_by_card.get((dataset, bucket, model), [])
                card_rows.append(
                    {
                        "dataset": dataset,
                        "bucket": bucket,
                        "definition": "TransE mapping-property cardinality; head-query labels inverted; validation split",
                        "query_count": count,
                        "total_queries": total,
                        "query_share_pct": fmt_float(100.0 * count / total if total else 0.0, 3),
                        "model": model,
                        "mrr": mean_or_blank(vals),
                        "model_query_count": len(vals),
                        "split": "valid",
                        "protocol": "structural validation analysis; source definitions in structural-analysis notes",
                        "wd_overlap_caveat": WD_CAVEAT if dataset == "WD-singer" else "",
                    }
                )
    return path_rows, card_rows


def write_share_tables(path_rows: list[dict[str, Any]], card_rows: list[dict[str, Any]]) -> None:
    def shares(rows: list[dict[str, Any]], buckets: list[str], key_name: str) -> list[dict[str, Any]]:
        out = []
        for dataset in ["NELL23K", "WD-singer"]:
            item = {"Dataset": dataset}
            for bucket in buckets:
                match = next(row for row in rows if row["dataset"] == dataset and row["bucket"] == bucket)
                item[bucket] = match["query_share_pct"]
            item["note"] = WD_CAVEAT if dataset == "WD-singer" else ""
            out.append(item)
        return out

    path_share = shares(path_rows, PATH_BUCKET_ORDER, "path")
    card_share = shares(card_rows, CARDINALITY_ORDER, "cardinality")
    write_csv(PATH_COUNT_DIR / "nell_wd_pathcount_query_share_validation.csv", path_share)
    write_csv(CARDINALITY_DIR / "nell_wd_cardinality_query_share_validation.csv", card_share)
    write_latex_table(
        PATH_COUNT_DIR / "nell_wd_pathcount_query_share_validation.tex",
        path_share,
        ["Dataset", *PATH_BUCKET_ORDER],
        "Share of validation queries by merged path-count bucket.",
        "tab:path-count-nell-wd-share",
    )
    write_latex_table(
        CARDINALITY_DIR / "nell_wd_cardinality_query_share_validation.tex",
        card_share,
        ["Dataset", *CARDINALITY_ORDER],
        "Share of validation queries by relation cardinality.",
        "tab:cardinality-nell-wd-share",
    )


def plot_cardinality(card_rows: list[dict[str, Any]]) -> None:
    CARDINALITY_DIR.mkdir(parents=True, exist_ok=True)
    datasets = ["NELL23K", "WD-singer"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    available_models = [
        model
        for model in MODEL_ORDER
        if any(row["model"] == model and row.get("mrr") for row in card_rows)
    ]
    x = np.arange(len(CARDINALITY_ORDER))
    width = 0.8 / max(len(available_models), 1)
    for ax, dataset in zip(axes, datasets):
        for i, model in enumerate(available_models):
            vals = []
            for bucket in CARDINALITY_ORDER:
                match = next((row for row in card_rows if row["dataset"] == dataset and row["bucket"] == bucket and row["model"] == model), None)
                vals.append(float(match["mrr"]) if match and match.get("mrr") else np.nan)
            ax.bar(x + i * width, vals, width=width, label=model)
        ax.set_title(dataset)
        ax.set_xticks(x + width * (len(available_models) - 1) / 2)
        ax.set_xticklabels(CARDINALITY_ORDER)
        ax.set_xlabel("Relation cardinality")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    axes[0].set_ylabel("MRR")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=len(labels), frameon=False)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    for suffix in ("png", "pdf"):
        fig.savefig(CARDINALITY_DIR / f"nell_wd_cardinality_mrr.{suffix}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_few_path_summary(path_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Write the focused few-path comparison used by the thesis-facing figure."""
    out: list[dict[str, Any]] = []
    for dataset in ["NELL23K", "WD-singer"]:
        rows = [
            row
            for row in path_rows
            if row["dataset"] == dataset and row["bucket"] == FEW_PATH_BUCKET and row.get("mrr")
        ]
        by_model = {row["model"]: float(row["mrr"]) for row in rows}
        pathbsr_mrr = by_model.get("PathBSR")
        if pathbsr_mrr is None:
            continue
        baseline_items = [(model, score) for model, score in by_model.items() if model != "PathBSR"]
        best_baseline, best_baseline_mrr = max(baseline_items, key=lambda item: item[1])
        share = next(row["query_share_pct"] for row in rows if row["model"] == "PathBSR")
        out.append(
            {
                "Dataset": dataset,
                "few_path_bucket": FEW_PATH_BUCKET,
                "query_share_pct": share,
                "best_baseline": best_baseline,
                "best_baseline_mrr": fmt_float(best_baseline_mrr),
                "PathBSR_mrr": fmt_float(pathbsr_mrr),
                "delta_vs_best_baseline": fmt_float(pathbsr_mrr - best_baseline_mrr),
                "wd_overlap_caveat": WD_CAVEAT if dataset == "WD-singer" else "",
            }
        )
    write_csv(PATH_COUNT_DIR / "nell_wd_few_path_mrr_summary_validation.csv", out)
    write_latex_table(
        PATH_COUNT_DIR / "nell_wd_few_path_mrr_summary_validation.tex",
        out,
        ["Dataset", "few_path_bucket", "query_share_pct", "best_baseline", "best_baseline_mrr", "PathBSR_mrr", "delta_vs_best_baseline"],
        "Few-path validation MRR summary.",
        "tab:path-count-few-path-mrr-summary",
    )
    return out


def plot_few_path_focus(path_rows: list[dict[str, Any]]) -> None:
    """Focused grouped bar chart for the dominant few-path bucket."""
    PATH_COUNT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = ["NELL23K", "WD-singer"]
    few_rows = [
        row
        for row in path_rows
        if row["bucket"] == FEW_PATH_BUCKET and row.get("mrr")
    ]
    available_models = [
        model
        for model in MODEL_ORDER
        if any(row["model"] == model for row in few_rows)
    ]
    share_by_dataset = {
        dataset: next(row["query_share_pct"] for row in few_rows if row["dataset"] == dataset)
        for dataset in datasets
    }

    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    x = np.arange(len(datasets))
    width = 0.78 / max(len(available_models), 1)
    offsets = (np.arange(len(available_models)) - (len(available_models) - 1) / 2) * width
    for offset, model in zip(offsets, available_models):
        vals = []
        for dataset in datasets:
            match = next(
                (row for row in few_rows if row["dataset"] == dataset and row["model"] == model),
                None,
            )
            vals.append(float(match["mrr"]) if match else np.nan)
        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            label=model,
            color=MODEL_COLORS.get(model),
            edgecolor=None,
            linewidth=0,
            hatch=None,
        )
        if model == "PathBSR":
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        val + 0.008,
                        f"{val:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{dataset}\n({share_by_dataset[dataset]}% queries)" for dataset in datasets])
    ax.set_ylabel("MRR")
    ax.set_ylim(0, 0.42)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=len(available_models), frameon=False)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    for suffix in ("png", "pdf"):
        fig.savefig(PATH_COUNT_DIR / f"nell_wd_pathcount_mrr.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_few_path_margin(summary_rows: list[dict[str, Any]]) -> None:
    """Plot PathBSR's few-path margin over the strongest non-PathBSR baseline."""
    PATH_COUNT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = [row["Dataset"] for row in summary_rows]
    deltas = [float(row["delta_vs_best_baseline"]) for row in summary_rows]
    labels = [f"vs. {row['best_baseline']}" for row in summary_rows]

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    x = np.arange(len(datasets))
    bars = ax.bar(x, deltas, width=0.48, color=MODEL_COLORS["PathBSR"], edgecolor="black", hatch="//")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("MRR gain over best baseline")
    ax.set_xlabel("Few-path validation queries (0-10 training paths)")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    for bar, delta, label in zip(bars, deltas, labels):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            delta + (0.002 if delta >= 0 else -0.002),
            f"{delta:+.3f}\n{label}",
            ha="center",
            va="bottom" if delta >= 0 else "top",
            fontsize=8,
        )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(PATH_COUNT_DIR / f"nell_wd_few_path_margin.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_structural_artifacts() -> None:
    require_structural_cache()
    ensure_pathbsr_valid_detail()
    path_rows, card_rows = structural_inputs()
    write_csv(PATH_COUNT_DIR / "nell_wd_pathcount_inputs.csv", path_rows)
    write_csv(CARDINALITY_DIR / "nell_wd_cardinality_inputs.csv", card_rows)
    write_share_tables(path_rows, card_rows)
    plot_cardinality(card_rows)
    few_path_summary = write_few_path_summary(path_rows)
    plot_few_path_focus(path_rows)
    plot_few_path_margin(few_path_summary)


def write_cases() -> None:
    cases = case_studies()
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    (CASE_DIR / "case_studies.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    write_case_markdown(CASE_DIR / "case_studies.md", cases)


def write_evaluation_protocol_tables() -> None:
    """Export paper-facing protocol table from the completed final test run.

    This mirrors `notebooks/pathbsr_evaluation_protocol_tables.ipynb` so the
    cleaned script workflow can rebuild the protocol artifact without relying on
    notebook state.
    """
    if not FINAL_TEST_METRICS.exists():
        print(
            "[pathbsr-experiments] skip evaluation protocol table: "
            f"missing {FINAL_TEST_METRICS.relative_to(ROOT)}",
            flush=True,
        )
        return

    with FINAL_TEST_METRICS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    dataset_order = {dataset: idx for idx, dataset in enumerate([*DATASETS, "WN18RR"])}
    rows.sort(key=lambda row: dataset_order.get(row.get("dataset", ""), len(dataset_order)))

    export_rows: list[dict[str, str]] = []
    for row in rows:
        export_rows.append(
            {
                "dataset": row["dataset"],
                "split": row.get("split", "test"),
                "protocol_id": "primary_bidirectional_average_tie",
                "mrr": row["mrr"],
                "hits@1": row["hits@1"],
                "hits@3": row["hits@3"],
                "hits@10": row["hits@10"],
                "num_queries": row["num_queries"],
            }
        )
    for row in rows:
        export_rows.append(
            {
                "dataset": row["dataset"],
                "split": row.get("split", "test"),
                "protocol_id": "auxiliary_tail_only_optimistic",
                "mrr": row["tailopt_mrr"],
                "hits@1": row["tailopt_hits@1"],
                "hits@3": row["tailopt_hits@3"],
                "hits@10": row["tailopt_hits@10"],
                "num_queries": row["tailopt_num_queries"],
            }
        )

    write_csv(
        PROTOCOL_SCORE_TABLE,
        export_rows,
        fieldnames=["dataset", "split", "protocol_id", "mrr", "hits@1", "hits@3", "hits@10", "num_queries"],
    )


def write_results_readme() -> None:
    text = """# PathBSR experiment artifacts

Generated by `scripts/pathbsr_experiments.py`.

This directory contains thesis-facing PathBSR experiment artifacts organized by experiment type.

Rules followed:

- No result is invented.
- Validation ablations are separated from final test artifacts.
- PathBSR default means normalized-BM25/top-100 as defined in `src/pathbsr/config.py`.
- Validation tables are generated from `results/ablation/pathbsr_validation_ablation_metrics.csv`,
  which is filled by directly running the current codebase and configs in this script.
- Main Protocol means bidirectional filtered full-entity average-tie evaluation.
- WD-singer uses the official split only; every WD-singer result keeps the overlap caveat.
- AnyBURL rows are read from `external_predictions/valid_predictions` when available.

Main outputs:

- `dataset_statistics/dataset_statistics.{csv,tex}`
- `ablation/entity_features_mrr_validation.{csv,tex}`
- `ablation/proxy_similarity_mrr_validation.{csv,tex}`
- `ablation/core_components_mrr_validation.{csv,tex}`
- `ablation/pathbsr_validation_ablation_metrics.csv`
- `path_count/fb_pathbucket_query_share.{csv,tex}`
- `path_count/fb_pathcount_mrr_grouped_bar.{png,pdf}`
- `path_count/nell_wd_pathcount_inputs.csv`
- `path_count/nell_wd_pathcount_mrr.{png,pdf}`
- `cardinality/fb_cardinality_query_share.{csv,tex}`
- `cardinality/fb_cardinality_mrr_grouped_bar.{png,pdf}`
- `cardinality/all_datasets_cardinality_mrr_by_model.{png,pdf}` from `scripts/router_analysis.py`
- `cardinality/nell_wd_cardinality_inputs.csv`
- `cardinality/nell_wd_cardinality_mrr.{png,pdf}`
- `path_count/all_datasets_pathcount_mrr_by_model.{png,pdf}` from `scripts/router_analysis.py`
- `case_studies/case_studies.md`
- `evaluation_protocol/pathbsr_protocol_score_tables.csv`
- `router/per_relation_router_test_summary.{csv,png,pdf}` from `scripts/router_analysis.py`

Interpretation cautions:

- Missing ablation cells mean the corresponding current-code validation run has not been completed yet.
- Entity feature ablation variants are controlled by `PathBSRConfig.entity_feature_mode`.
- Proxy similarity ablation is computed through `PathBSRConfig.proxy_similarity_mode`; the rest of the PathBSR default pipeline is unchanged.
- `w/o Path Answers` is implemented as `rule_library_topk=0`; Proxy Answers, Frequency Answers, and Path Verification remain enabled.
- `w/o Path Verification` is implemented as `verification_top_m=0`; Candidate Scoring remains unchanged.
- WD-singer official split contains duplicate/overlap caveats, so claims involving WD-singer should mention the caveat.
"""
    (RESULTS / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    silence_internal_progress()
    prune_obsolete_run_metrics()
    write_base_tables()
    if not args.skip_expensive:
        run_missing_validation_ablations(force=args.force)
    else:
        print("[pathbsr-experiments] --skip-expensive: not running validation ablations", flush=True)
    write_complete_proxy_tables()
    write_complete_core_tables()
    write_evaluation_protocol_tables()
    if args.skip_structural:
        print("[pathbsr-experiments] --skip-structural: not generating structural figures/tables", flush=True)
    else:
        try:
            write_structural_artifacts()
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
    write_cases()
    write_results_readme()
    print(f"[pathbsr-experiments] done: {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

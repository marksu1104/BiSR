#!/usr/bin/env python3
"""Per-relation router and full structural comparison artifacts.

This script builds two kinds of paper-supporting artifacts:

1. full six-dataset structural figures for AnyBURL, ConvE, HoGRN, PathBSR, and
   TransE;
2. a per-relation router analysis where the router is selected on validation
   relations and evaluated on the test split.

The router is diagnostic, not the current PathBSR model. It is intended for
complementarity analysis and future-work discussion.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
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
from pathbsr.locations import data_root, workspace_root  # noqa: E402


DATASETS = ["FB15K-237-10", "FB15K-237-20", "FB15K-237-50", "NELL23K", "WD-singer", "WN18RR"]
STRUCTURAL_MODELS = ["AnyBURL", "ConvE", "HoGRN", "PathBSR", "TransE"]
ROUTER_MODELS = ["AnyBURL", "ConvE", "HoGRN", "PathBSR", "TransE", "TuckER"]
PATH_BUCKETS = ["0", "1", "2-10", "11-100", ">100"]
CARDINALITY_BUCKETS = ["1-to-1", "1-to-N", "N-to-1", "N-to-N"]
MODEL_COLORS = {
    "AnyBURL": "#2ca02c",
    "ConvE": "#ff7f0e",
    "HoGRN": "#d62728",
    "PathBSR": "#9467bd",
    "TransE": "#1f77b4",
    "TuckER": "#8c564b",
}

RESULTS = ROOT / "results"
RUNS = RESULTS / "runs"
CACHE = RESULTS / "cache"
PATH_COUNT_DIR = RESULTS / "path_count"
CARDINALITY_DIR = RESULTS / "cardinality"
ROUTER_DIR = RESULTS / "router"
DATA_ROOT = data_root()
EXPORTS = workspace_root() / "external_predictions"


def reverse_relation(relation: str) -> str:
    suffix = DEFAULT_CONFIG.reverse_suffix
    if relation.endswith(suffix):
        return relation[: -len(suffix)]
    return f"{relation}{suffix}"


def silence_internal_progress() -> None:
    """Keep this artifact builder to dataset-level progress messages."""
    pathbsr_model_module.tqdm = lambda iterable, **_: iterable
    pathbsr_paths_module.tqdm = lambda iterable, **_: iterable


def query_relation_map(dataset: str, split: str) -> pd.DataFrame:
    train, valid, test = load_dataset(DATA_ROOT, dataset)
    triples = {"valid": valid, "test": test}[split]
    rows: list[dict[str, Any]] = []
    for triple_index, (head, relation, tail) in enumerate(triples):
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "query_index": 2 * triple_index,
                "direction": "tail",
                "original_h": head,
                "original_r": relation,
                "original_t": tail,
            }
        )
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "query_index": 2 * triple_index + 1,
                "direction": "head",
                "original_h": head,
                "original_r": relation,
                "original_t": tail,
            }
        )
    return pd.DataFrame(rows)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ensure_pathbsr_detail(split: str, force: bool = False) -> Path:
    """Generate query-level PathBSR ranks for a split if missing or incomplete."""
    out = RUNS / f"pathbsr_{split}_detail.csv"
    existing = pd.DataFrame()
    if out.exists() and not force:
        existing = pd.read_csv(out)
        existing_keys = set(zip(existing["dataset"], existing["split"]))
    else:
        existing_keys = set()

    missing = [dataset for dataset in DATASETS if (dataset, split) not in existing_keys]
    if not missing:
        print(f"[router-analysis] use existing {out.relative_to(ROOT)}", flush=True)
        return out

    rows: list[dict[str, Any]] = [] if existing.empty or force else existing.to_dict("records")
    for dataset in missing:
        print(f"[router-analysis] generate PathBSR {split} detail for {dataset}", flush=True)
        train, valid, test = load_dataset(DATA_ROOT, dataset)
        triples = {"valid": valid, "test": test}[split]
        model = PathBSR(train, valid, test, config=DEFAULT_CONFIG)
        total_queries = 2 * len(triples)
        completed = 0
        progress_every = max(1, len(triples) // 10)
        for triple_index, (head, relation, tail) in enumerate(triples):
            specs = (
                ("tail", 2 * triple_index, head, relation, tail),
                ("head", 2 * triple_index + 1, tail, reverse_relation(relation), head),
            )
            for direction, query_index, query_head, query_relation, target in specs:
                scores = model.score(query_head, query_relation)
                rank = model.filtered_rank(scores, (query_head, query_relation), target)
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
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
            if (triple_index + 1) % progress_every == 0 or (triple_index + 1) == len(triples):
                print(
                    f"[router-analysis] {dataset} {split}: "
                    f"{completed:,}/{total_queries:,} queries",
                    flush=True,
                )
    write_csv(out, rows)
    print(f"[router-analysis] wrote {out.relative_to(ROOT)}", flush=True)
    return out


def load_pathbsr_rr(split: str, force: bool = False) -> pd.DataFrame:
    path = ensure_pathbsr_detail(split, force=force)
    df = pd.read_csv(path)
    df = df[df["dataset"].isin(DATASETS)].copy()
    df["model"] = "PathBSR"
    df["rr"] = df["bsr_rr"].astype(float)
    return df[["dataset", "model", "split", "query_index", "direction", "original_r", "rr"]]


def load_external_rr(split: str, model: str, dataset: str) -> pd.DataFrame | None:
    path = EXPORTS / f"{split}_predictions" / model / dataset / f"{split}_query_summary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, usecols=["dataset", "split", "query_index", "direction", "filtered_rank"])
    df["model"] = model
    df["rr"] = 1.0 / df["filtered_rank"].astype(float)
    mapping = query_relation_map(dataset, split)[["dataset", "split", "query_index", "original_r"]]
    df = df.merge(mapping, on=["dataset", "split", "query_index"], how="left")
    return df[["dataset", "model", "split", "query_index", "direction", "original_r", "rr"]]


def load_model_rr(split: str, models: list[str], force_pathbsr: bool = False) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model in models:
        if model == "PathBSR":
            frames.append(load_pathbsr_rr(split, force=force_pathbsr))
            continue
        for dataset in DATASETS:
            df = load_external_rr(split, model, dataset)
            if df is not None:
                frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No query-level rows found for split={split}")
    out = pd.concat(frames, ignore_index=True)
    return out[out["model"].isin(models)].copy()


def grouped_bar_facets(
    summary: pd.DataFrame,
    bucket_col: str,
    bucket_order: list[str],
    out_prefix: Path,
    ylabel: str = "MRR",
) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.2), sharey=True)
    axes = axes.ravel()
    x = np.arange(len(bucket_order))
    width = 0.16
    offsets = (np.arange(len(STRUCTURAL_MODELS)) - (len(STRUCTURAL_MODELS) - 1) / 2) * width
    for ax, dataset in zip(axes, DATASETS):
        part = summary[summary["dataset"] == dataset]
        for offset, model in zip(offsets, STRUCTURAL_MODELS):
            values = []
            for bucket in bucket_order:
                hit = part[(part[bucket_col] == bucket) & (part["model"] == model)]
                values.append(float(hit["mrr"].iloc[0]) if len(hit) else np.nan)
            ax.bar(
                x + offset,
                values,
                width=width,
                label=model,
                color=MODEL_COLORS[model],
                alpha=0.95,
            )
        ax.set_title(dataset)
        ax.set_xticks(x)
        ax.set_xticklabels(bucket_order, rotation=20 if len(bucket_order) > 4 else 0)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    axes[0].set_ylabel(ylabel)
    axes[3].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(STRUCTURAL_MODELS), frameon=False)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(out_prefix.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_full_structural_figures(force_pathbsr: bool = False) -> None:
    features_path = CACHE / "structural_valid_features.csv"
    long_path = CACHE / "structural_valid_long.csv"
    if not features_path.exists() or not long_path.exists():
        raise FileNotFoundError(
            "Missing structural cache. Run:\n"
            "  PYTHONPATH=src .venv/bin/python scripts/structural_analysis.py"
        )
    features = pd.read_csv(features_path)
    baseline_long = pd.read_csv(long_path)
    baseline_long = baseline_long[baseline_long["model"].isin([m for m in STRUCTURAL_MODELS if m != "PathBSR"])]

    pathbsr = load_pathbsr_rr("valid", force=force_pathbsr)
    pathbsr = pathbsr.merge(
        features[["dataset", "query_index", "path_bucket", "rel_type"]],
        on=["dataset", "query_index"],
        how="inner",
    )
    long_df = pd.concat(
        [
            baseline_long[["dataset", "model", "query_index", "rr", "path_bucket", "rel_type"]],
            pathbsr[["dataset", "model", "query_index", "rr", "path_bucket", "rel_type"]],
        ],
        ignore_index=True,
    )
    long_df = long_df[long_df["model"].isin(STRUCTURAL_MODELS)]

    path_summary = (
        long_df.groupby(["dataset", "path_bucket", "model"], as_index=False)
        .agg(mrr=("rr", "mean"), queries=("rr", "size"))
    )
    card_summary = (
        long_df[long_df["rel_type"].isin(CARDINALITY_BUCKETS)]
        .groupby(["dataset", "rel_type", "model"], as_index=False)
        .agg(mrr=("rr", "mean"), queries=("rr", "size"))
    )
    PATH_COUNT_DIR.mkdir(parents=True, exist_ok=True)
    CARDINALITY_DIR.mkdir(parents=True, exist_ok=True)
    path_summary.to_csv(PATH_COUNT_DIR / "all_datasets_pathcount_mrr_by_model.csv", index=False)
    card_summary.to_csv(CARDINALITY_DIR / "all_datasets_cardinality_mrr_by_model.csv", index=False)
    grouped_bar_facets(
        path_summary,
        "path_bucket",
        PATH_BUCKETS,
        PATH_COUNT_DIR / "all_datasets_pathcount_mrr_by_model",
    )
    grouped_bar_facets(
        card_summary,
        "rel_type",
        CARDINALITY_BUCKETS,
        CARDINALITY_DIR / "all_datasets_cardinality_mrr_by_model",
    )
    print("[router-analysis] wrote full structural comparison figures", flush=True)


def select_relation_router(valid_rr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    relation_scores = (
        valid_rr.groupby(["dataset", "original_r", "model"], as_index=False)
        .agg(valid_mrr=("rr", "mean"), valid_queries=("rr", "size"))
    )
    relation_scores = relation_scores.sort_values(
        ["dataset", "original_r", "valid_mrr", "model"],
        ascending=[True, True, False, True],
    )
    winners = relation_scores.groupby(["dataset", "original_r"], as_index=False).first()
    winners = winners.rename(columns={"model": "selected_model", "valid_mrr": "selected_valid_mrr"})

    overall = (
        valid_rr.groupby(["dataset", "model"], as_index=False)
        .agg(valid_mrr=("rr", "mean"), valid_queries=("rr", "size"))
        .sort_values(["dataset", "valid_mrr", "model"], ascending=[True, False, True])
    )
    best_single = overall.groupby("dataset", as_index=False).first().rename(
        columns={"model": "best_valid_single_model", "valid_mrr": "best_valid_single_mrr"}
    )
    return winners, best_single


def evaluate_router(valid_rr: pd.DataFrame, test_rr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    winners, best_single = select_relation_router(valid_rr)
    test_wide = test_rr.pivot_table(
        index=["dataset", "query_index", "original_r"],
        columns="model",
        values="rr",
        aggfunc="first",
    ).reset_index()

    winner_map = winners.set_index(["dataset", "original_r"])["selected_model"].to_dict()
    fallback_map = best_single.set_index("dataset")["best_valid_single_model"].to_dict()

    routed_rrs: list[dict[str, Any]] = []
    for _, row in test_wide.iterrows():
        dataset = str(row["dataset"])
        relation = str(row["original_r"])
        selected = winner_map.get((dataset, relation), fallback_map[dataset])
        rr = row.get(selected, np.nan)
        if pd.isna(rr):
            selected = fallback_map[dataset]
            rr = row.get(selected, np.nan)
        available = [row.get(model, np.nan) for model in ROUTER_MODELS if model in row.index]
        available = [float(value) for value in available if not pd.isna(value)]
        routed_rrs.append(
            {
                "dataset": dataset,
                "query_index": int(row["query_index"]),
                "original_r": relation,
                "selected_model": selected,
                "routed_rr": float(rr) if not pd.isna(rr) else np.nan,
                "oracle_rr": max(available) if available else np.nan,
            }
        )
    routed = pd.DataFrame(routed_rrs)

    test_by_model = (
        test_rr.groupby(["dataset", "model"], as_index=False)
        .agg(test_mrr=("rr", "mean"), test_queries=("rr", "size"))
    )
    best_test_single = (
        test_by_model.sort_values(["dataset", "test_mrr", "model"], ascending=[True, False, True])
        .groupby("dataset", as_index=False)
        .first()
        .rename(columns={"model": "best_test_single_model", "test_mrr": "best_test_single_mrr"})
    )
    pathbsr_test = test_by_model[test_by_model["model"] == "PathBSR"][
        ["dataset", "test_mrr"]
    ].rename(columns={"test_mrr": "pathbsr_test_mrr"})
    best_valid_test = best_single[["dataset", "best_valid_single_model"]].merge(
        test_by_model,
        left_on=["dataset", "best_valid_single_model"],
        right_on=["dataset", "model"],
        how="left",
    )[["dataset", "best_valid_single_model", "test_mrr"]].rename(
        columns={"test_mrr": "best_valid_single_test_mrr"}
    )
    routed_summary = routed.groupby("dataset", as_index=False).agg(
        routed_test_mrr=("routed_rr", "mean"),
        oracle_test_mrr=("oracle_rr", "mean"),
        test_queries=("routed_rr", "size"),
    )
    summary = (
        routed_summary.merge(pathbsr_test, on="dataset", how="left")
        .merge(best_valid_test, on="dataset", how="left")
        .merge(best_test_single[["dataset", "best_test_single_model", "best_test_single_mrr"]], on="dataset", how="left")
    )
    summary["delta_router_vs_pathbsr"] = summary["routed_test_mrr"] - summary["pathbsr_test_mrr"]
    summary["delta_router_vs_best_valid_single"] = (
        summary["routed_test_mrr"] - summary["best_valid_single_test_mrr"]
    )
    summary["gap_to_oracle"] = summary["oracle_test_mrr"] - summary["routed_test_mrr"]
    return routed, summary


def plot_router_summary(summary: pd.DataFrame) -> None:
    ROUTER_DIR.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("PathBSR", "pathbsr_test_mrr", MODEL_COLORS["PathBSR"]),
        ("Best valid single", "best_valid_single_test_mrr", "#7f7f7f"),
        ("Per-relation router", "routed_test_mrr", "#17becf"),
        ("Oracle", "oracle_test_mrr", "#bcbd22"),
    ]
    x = np.arange(len(DATASETS))
    width = 0.19
    fig, ax = plt.subplots(figsize=(12.5, 4.2))
    for i, (label, col, color) in enumerate(metrics):
        values = [float(summary.loc[summary["dataset"] == dataset, col].iloc[0]) for dataset in DATASETS]
        ax.bar(x + (i - 1.5) * width, values, width=width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, rotation=20)
    ax.set_ylabel("Test MRR")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(ROUTER_DIR / "per_relation_router_test_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_router_analysis(force_pathbsr: bool = False) -> None:
    valid_rr = load_model_rr("valid", ROUTER_MODELS, force_pathbsr=force_pathbsr)
    test_rr = load_model_rr("test", ROUTER_MODELS, force_pathbsr=force_pathbsr)
    winners, best_single = select_relation_router(valid_rr)
    routed, summary = evaluate_router(valid_rr, test_rr)

    ROUTER_DIR.mkdir(parents=True, exist_ok=True)
    valid_rr.to_csv(ROUTER_DIR / "router_valid_query_rr.csv", index=False)
    test_rr.to_csv(ROUTER_DIR / "router_test_query_rr.csv", index=False)
    winners.to_csv(ROUTER_DIR / "per_relation_router_selected_on_valid.csv", index=False)
    best_single.to_csv(ROUTER_DIR / "best_single_model_selected_on_valid.csv", index=False)
    routed.to_csv(ROUTER_DIR / "per_relation_router_test_detail.csv", index=False)
    summary.to_csv(ROUTER_DIR / "per_relation_router_test_summary.csv", index=False)
    assignment = winners.groupby(["dataset", "selected_model"], as_index=False).agg(
        relations=("original_r", "count"),
        mean_selected_valid_mrr=("selected_valid_mrr", "mean"),
    )
    assignment.to_csv(ROUTER_DIR / "per_relation_router_assignment_counts.csv", index=False)
    plot_router_summary(summary)
    print(f"[router-analysis] wrote router artifacts to {ROUTER_DIR.relative_to(ROOT)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PathBSR router and full structural comparison artifacts.")
    parser.add_argument("--skip-router", action="store_true", help="Only generate full structural comparison figures.")
    parser.add_argument("--skip-structural", action="store_true", help="Only generate router artifacts.")
    parser.add_argument("--force-pathbsr-detail", action="store_true", help="Regenerate PathBSR query-level detail CSVs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    silence_internal_progress()
    if not args.skip_structural:
        write_full_structural_figures(force_pathbsr=args.force_pathbsr_detail)
    if not args.skip_router:
        write_router_analysis(force_pathbsr=args.force_pathbsr_detail)


if __name__ == "__main__":
    main()

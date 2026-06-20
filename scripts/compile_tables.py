#!/usr/bin/env python3
"""
Compile all baseline CSV results into paper-ready tables.
Usage: python scripts/compile_tables.py
"""
import csv
from pathlib import Path

REPO = Path(__file__).parent.parent
OUTPUTS = REPO / "outputs"

DATASETS = ["WD-singer", "FB15K-237-10", "FB15K-237-20", "FB15K-237-50", "NELL23K", "WN18RR"]

# Model display order and labels
MAIN_ORDER = [
    ("TransE",       "TransE"),
    ("RotatE",       "RotatE"),
    ("DistMult",     "DistMult"),
    ("ComplEx",      "ComplEx"),
    ("ConvE",        "ConvE"),
    ("TuckER",       "TuckER"),
    ("conve",        "HoGRN"),
    ("point.rs.conve","DacKGR"),
    ("Prob-CBR",     "Prob-CBR"),
    ("AnyBURL",      "AnyBURL"),
    ("LoGRe",        "LoGRe"),
    ("StruProKGR",   "StruProKGR"),
    ("PathBSR",      "PathBSR"),
]

SOTA_ORDER = list(MAIN_ORDER)


def load_csv(path):
    """Returns dict: (dataset, model) -> row_dict"""
    data = {}
    if not Path(path).exists():
        return data
    with open(path) as f:
        for row in csv.DictReader(f):
            data[(row["Dataset"], row["Model"])] = row
    return data


def fmt(val, decimals=4):
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return "—"


def best_model_per_dataset(data, metric, dataset, models):
    """Return the model key with the highest value for a given dataset."""
    best_val, best_key = -1, None
    for model_key, _ in models:
        row = data.get((dataset, model_key))
        if row is None:
            continue
        try:
            v = float(row[metric])
            if v > best_val:
                best_val, best_key = v, model_key
        except (ValueError, TypeError):
            pass
    return best_key


def print_table(title, data, model_order, metric_avg, metric_tail, metric_h1, metric_h3, metric_h10):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

    col_w = 9
    ds_short = {
        "WD-singer": "WD-sing",
        "FB15K-237-10": "FB10",
        "FB15K-237-20": "FB20",
        "FB15K-237-50": "FB50",
        "NELL23K": "NELL",
        "WN18RR": "WN18RR",
    }

    # Header
    header = f"{'Model':<14}"
    for ds in DATASETS:
        header += f"  {ds_short[ds]:>{col_w}}"
    print(header)
    print("-" * (14 + len(DATASETS) * (col_w + 2)))

    # Find best per dataset for bold marking
    best = {}
    for ds in DATASETS:
        best[ds] = best_model_per_dataset(data, metric_avg, ds, model_order)

    for model_key, label in model_order:
        row_str = f"{label:<14}"
        for ds in DATASETS:
            row = data.get((ds, model_key))
            if row is None:
                row_str += f"  {'—':>{col_w}}"
                continue
            val = row.get(metric_avg, "—")
            mark = "*" if model_key == best[ds] else " "
            row_str += f"  {fmt(val):>{col_w-1}}{mark}"
        print(row_str)

    print(f"\n  Metric: {metric_avg}  (* = best per dataset)")


def main():
    # Load all data
    main_data = {}
    for f in OUTPUTS.glob("*_metrics.csv"):
        if "sota" in f.name:
            continue
        main_data.update(load_csv(f))

    sota_data = {}
    for f in OUTPUTS.glob("*_sota_metrics.csv"):
        sota_data.update(load_csv(f))

    # Older continuous-score runners predate separate SOTA CSVs. Their tail
    # columns use the same filtered tail protocol; prefer an explicit SOTA row
    # whenever present, otherwise retain that tail result as a labelled fallback.
    for key, row in main_data.items():
        sota_data.setdefault(key, row)

    # For PathBSR main: MRR_Avg is stored in MRR_Avg column
    # For PathBSR sota: MRR_Tail is stored in MRR_Tail column

    print_table(
        "TABLE 1 — Main Protocol (Bidirectional, Tie-aware) — MRR_Avg",
        main_data, MAIN_ORDER,
        metric_avg="MRR_Avg", metric_tail="MRR_Tail",
        metric_h1="Hits@1_Avg", metric_h3="Hits@3_Avg", metric_h10="Hits@10_Avg",
    )

    # Print full detail table (all metrics) per dataset
    print(f"\n{'='*80}")
    print("  TABLE 1 DETAIL — MRR / H@1 / H@3 / H@10 (Avg)")
    print(f"{'='*80}")
    for ds in DATASETS:
        print(f"\n  {ds}")
        print(f"  {'Model':<14}  {'MRR':>7}  {'H@1':>7}  {'H@3':>7}  {'H@10':>7}")
        print(f"  {'-'*46}")
        for model_key, label in MAIN_ORDER:
            row = main_data.get((ds, model_key))
            if row is None:
                continue
            mrr  = fmt(row.get("MRR_Avg",    "—"))
            h1   = fmt(row.get("Hits@1_Avg", "—"))
            h3   = fmt(row.get("Hits@3_Avg", "—"))
            h10  = fmt(row.get("Hits@10_Avg","—"))
            print(f"  {label:<14}  {mrr:>7}  {h1:>7}  {h3:>7}  {h10:>7}")

    print(f"\n{'='*80}")
    print("  TABLE 2 — SOTA Protocol (Tail-only, Optimistic) — MRR_Tail")
    print(f"{'='*80}")
    ds_short = {"WD-singer":"WD-sing","FB15K-237-10":"FB10","FB15K-237-20":"FB20",
                "FB15K-237-50":"FB50","NELL23K":"NELL","WN18RR":"WN18RR"}
    col_w = 9
    header = f"{'Model':<14}"
    for ds in DATASETS:
        header += f"  {ds_short[ds]:>{col_w}}"
    print(header)
    print("-" * (14 + len(DATASETS) * (col_w + 2)))
    for model_key, label in SOTA_ORDER:
        row_str = f"{label:<14}"
        for ds in DATASETS:
            row = sota_data.get((ds, model_key))
            if row is None:
                row_str += f"  {'—':>{col_w}}"
                continue
            val = row.get("MRR_Tail", "—")
            row_str += f"  {fmt(val):>{col_w}}"
        print(row_str)

    print("\nNotes:")
    print("  - Explicit *_sota_metrics.csv rows take precedence; otherwise MRR_Tail is a fallback.")
    print("  - AnyBURL is retained as measured: its gap from the paper is attributed to")
    print("    the native candidate-list/tie protocol difference, not parameter mismatch.")


if __name__ == "__main__":
    main()

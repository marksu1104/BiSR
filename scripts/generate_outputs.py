#!/usr/bin/env python3
"""Generate or verify every tracked research output from saved metrics."""

from __future__ import annotations

import argparse
import csv
import filecmp
import tempfile
from pathlib import Path

from build_result_tables import generate as generate_tables
from plot_efficiency import generate as generate_efficiency
from plot_mrr_lines import generate as generate_mrr_lines


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"

EXPECTED_METRICS = {
    "anyburl_metrics.csv": 6,
    "anyburl_sota_metrics.csv": 6,
    "dackgr_metrics.csv": 6,
    "dackgr_sota_metrics.csv": 6,
    "hogrn_metrics.csv": 6,
    "hogrn_sota_metrics.csv": 6,
    "logre_metrics.csv": 5,
    "logre_sota_metrics.csv": 5,
    "pathbsr_metrics.csv": 6,
    "pathbsr_sota_metrics.csv": 6,
    "probcbr_metrics.csv": 6,
    "probcbr_sota_metrics.csv": 6,
    "struprokgr_metrics.csv": 5,
    "struprokgr_sota_metrics.csv": 5,
    "traditional_metrics.csv": 36,
    "traditional_sota_metrics.csv": 36,
}
EXPECTED_TABLES = {
    "efficiency_table.csv",
    "main_table.csv",
    "sota_table.csv",
    "tables.md",
    "tables.tex",
}
EXPECTED_FIGURES = {
    "efficiency_scatter.png",
    "mrr_lines.png",
    "mrr_lines_main.png",
    "mrr_lines_sota.png",
}


def validate_metrics(metrics_dir: Path = METRICS_DIR) -> None:
    metrics_dir = Path(metrics_dir)
    actual = {path.name for path in metrics_dir.glob("*.csv")}
    expected = set(EXPECTED_METRICS)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unknown:
            details.append(f"unregistered: {', '.join(unknown)}")
        raise RuntimeError("Invalid metrics directory (" + "; ".join(details) + ")")

    for name, expected_rows in EXPECTED_METRICS.items():
        path = metrics_dir / name
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != expected_rows:
            raise RuntimeError(
                f"{name} contains {len(rows)} rows; expected {expected_rows}"
            )
        keys = []
        for row in rows:
            dataset = row.get("Dataset", row.get("dataset", ""))
            model = row.get("Model", row.get("model", ""))
            if not dataset or not model:
                raise RuntimeError(f"{name} contains a row without Dataset and Model")
            keys.append((dataset, model))
        if len(keys) != len(set(keys)):
            raise RuntimeError(f"{name} contains duplicate Dataset/Model rows")


def _actual_names(directory: Path) -> set[str]:
    if not directory.exists():
        return set()
    return {path.name for path in directory.iterdir() if path.is_file()}


def validate_output_inventory(root: Path = RESULTS_DIR) -> None:
    root = Path(root)
    inventories = (
        (root / "tables", EXPECTED_TABLES),
        (root / "figures", EXPECTED_FIGURES),
    )
    for directory, expected in inventories:
        actual = _actual_names(directory)
        unknown = sorted(actual - expected)
        if unknown:
            raise RuntimeError(
                f"Unregistered outputs in {directory}: {', '.join(unknown)}"
            )


def build_all(root: Path = RESULTS_DIR) -> list[Path]:
    root = Path(root)
    table_dir = root / "tables"
    figure_dir = root / "figures"
    generated = generate_tables(METRICS_DIR, table_dir)
    generated.extend(generate_mrr_lines(table_dir, figure_dir))
    generated.extend(generate_efficiency(table_dir, figure_dir))
    return generated


def build_tables_only(root: Path = RESULTS_DIR) -> list[Path]:
    return generate_tables(METRICS_DIR, Path(root) / "tables")


def build_figures_only(root: Path = RESULTS_DIR) -> list[Path]:
    table_dir = Path(root) / "tables"
    generated = generate_mrr_lines(table_dir, Path(root) / "figures")
    generated.extend(generate_efficiency(table_dir, Path(root) / "figures"))
    return generated


def check_outputs() -> None:
    validate_metrics()
    validate_output_inventory()
    with tempfile.TemporaryDirectory(prefix="sparsekgc-outputs-") as temp_dir:
        candidate_root = Path(temp_dir) / "results"
        generated = build_all(candidate_root)
        mismatches = []
        for candidate in generated:
            relative = candidate.relative_to(candidate_root)
            committed = RESULTS_DIR / relative
            if not committed.exists() or not filecmp.cmp(candidate, committed, shallow=False):
                mismatches.append(str(relative))
        if mismatches:
            raise RuntimeError(
                "Tracked outputs are missing or stale: " + ", ".join(sorted(mismatches))
            )
    print("All saved metrics are valid and every tracked output is reproducible.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--tables", action="store_true", help="Generate tables only")
    mode.add_argument("--figures", action="store_true", help="Generate figures only")
    mode.add_argument("--check", action="store_true", help="Verify outputs without changing them")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check:
        check_outputs()
        return

    validate_metrics()
    validate_output_inventory()
    if args.tables:
        generated = build_tables_only()
    elif args.figures:
        generated = build_figures_only()
    else:
        generated = build_all()
    print(f"Generated {len(generated)} output files.")


if __name__ == "__main__":
    main()

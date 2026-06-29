#!/usr/bin/env python3
"""SparseKGC adapter for PathBSR.

Called by run_baseline.py with:
    python run_pathbsr.py --datasets DS1 DS2 ... [--dry_run]

Runs the PathBSR CLI (scripts/run_pathbsr.py) for each dataset on both
valid and test splits, then writes two output CSVs in SparseKGC format:

  $SPARSEKGC_OUTPUT_DIR/pathbsr_metrics.csv      — Main protocol (MRR_Avg / Hits@3_Avg)
  $SPARSEKGC_OUTPUT_DIR/pathbsr_sota_metrics.csv — SOTA protocol (MRR_Tail / Hits@3_Tail)

Metric mapping from PathBSR CLI output:
  mrr           → MRR_Avg   (bidirectional average)
  hits@3        → Hits@3_Avg
  tailopt_mrr   → MRR_Tail  (tail-only, optimistic tie — SOTA protocol)
  tailopt_hits@3 → Hits@3_Tail
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_DATA_ROOT = REPO_ROOT / "datasets"
DEFAULT_OUTPUT_DIR = Path(os.environ.get("SPARSEKGC_OUTPUT_DIR") or (REPO_ROOT / "outputs"))

MAIN_FIELDS = ["Model", "Dataset", "MRR_Avg", "Hits@3_Avg", "MRR_Avg_Val", "Hits@3_Avg_Val", "seconds"]
SOTA_FIELDS = ["Model", "Dataset", "MRR_Tail", "Hits@3_Tail", "MRR_Tail_Val", "Hits@3_Tail_Val"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def run_split(datasets: list[str], split: str, data_root: Path, dry_run: bool) -> list[dict]:
    """Run PathBSR CLI for all datasets on one split, return parsed rows."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        out_path = Path(f.name)

    cli = ROOT / "scripts" / "run_pathbsr.py"
    cmd = [sys.executable, str(cli)]
    for ds in datasets:
        cmd += ["--dataset", ds]
    cmd += [
        "--split", split,
        "--data-root", str(data_root),
        "--output", str(out_path),
    ]

    if dry_run:
        print(f"[dry_run] {' '.join(str(c) for c in cmd)}", flush=True)
        return []

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    print(f"[pathbsr] Running {split} split for {len(datasets)} dataset(s)...", flush=True)
    start = time.perf_counter()
    subprocess.run(cmd, check=True, env=env)
    elapsed = time.perf_counter() - start
    print(f"[pathbsr] {split} split done in {elapsed:.1f}s", flush=True)

    rows = []
    if out_path.exists():
        with out_path.open() as f:
            rows = list(csv.DictReader(f))
        out_path.unlink()
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[pathbsr] Written {len(rows)} row(s) to {path}", flush=True)


def main() -> None:
    args = parse_args()
    output_dir = DEFAULT_OUTPUT_DIR

    # Key: dataset name → {"test": row, "valid": row}
    by_dataset: dict[str, dict[str, dict]] = {ds: {} for ds in args.datasets}

    for split in ("valid", "test"):
        rows = run_split(args.datasets, split, args.data_root, args.dry_run)
        for row in rows:
            ds = row.get("dataset", "")
            if ds in by_dataset:
                by_dataset[ds][split] = row

    if args.dry_run:
        return

    main_rows = []
    sota_rows = []

    for ds in args.datasets:
        splits = by_dataset[ds]
        test_row = splits.get("test", {})
        valid_row = splits.get("valid", {})

        def _f(d: dict, key: str) -> str:
            v = d.get(key, "")
            try:
                return f"{float(v):.6f}"
            except (TypeError, ValueError):
                return "—"

        # Wall-clock time: build + eval (test split)
        build_sec = float(test_row.get("build_sec", 0) or 0)
        eval_sec_v = float(valid_row.get("eval_sec", 0) or 0)
        eval_sec_t = float(test_row.get("eval_sec", 0) or 0)
        total_sec = build_sec + eval_sec_v + eval_sec_t

        main_rows.append({
            "Model":          "PathBSR",
            "Dataset":        ds,
            "MRR_Avg":        _f(test_row,  "mrr"),
            "Hits@3_Avg":     _f(test_row,  "hits@3"),
            "MRR_Avg_Val":    _f(valid_row, "mrr"),
            "Hits@3_Avg_Val": _f(valid_row, "hits@3"),
            "seconds":        f"{total_sec:.3f}",
        })
        sota_rows.append({
            "Model":           "PathBSR",
            "Dataset":         ds,
            "MRR_Tail":        _f(test_row,  "tailopt_mrr"),
            "Hits@3_Tail":     _f(test_row,  "tailopt_hits@3"),
            "MRR_Tail_Val":    _f(valid_row, "tailopt_mrr"),
            "Hits@3_Tail_Val": _f(valid_row, "tailopt_hits@3"),
        })

        print(
            f"[pathbsr] {ds}: "
            f"MRR_Avg={main_rows[-1]['MRR_Avg']} "
            f"MRR_Tail={sota_rows[-1]['MRR_Tail']} "
            f"seconds={total_sec:.1f}",
            flush=True,
        )

    write_csv(output_dir / "pathbsr_metrics.csv",      MAIN_FIELDS, main_rows)
    write_csv(output_dir / "pathbsr_sota_metrics.csv", SOTA_FIELDS, sota_rows)


if __name__ == "__main__":
    main()

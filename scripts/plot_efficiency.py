#!/usr/bin/env python3
"""Generate the thesis efficiency plot from generated result tables."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sparsekgc-matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TABLE_DIR = REPO_ROOT / "results" / "tables"
DEFAULT_FIGURE_DIR = REPO_ROOT / "results" / "figures"

PLOT_METHODS = [
    "TransE",
    "ConvE",
    "TuckER",
    "AnyBURL",
    "HoGRN",
    "LoGRe",
    "StruProKGR",
    "PathBSR",
]
EFF_DATASETS = ["FB15K-237-10", "NELL23K", "WD-singer"]
STYLE = {
    "TransE": ("#4c72b0", "o"),
    "ConvE": ("#55a868", "s"),
    "TuckER": ("#c4a000", "^"),
    "AnyBURL": ("#dd8452", "D"),
    "HoGRN": ("#8172b3", "v"),
    "LoGRe": ("#c44e52", "P"),
    "StruProKGR": ("#937860", "X"),
    "PathBSR": ("#cc0000", "*"),
}
LABEL_OVERRIDE = {
    ("ConvE", "NELL23K"): (-14, -14, "right"),
    ("TuckER", "NELL23K"): (14, -14, "left"),
    ("StruProKGR", "NELL23K"): (-10, 11, "right"),
    ("HoGRN", "NELL23K"): (10, 11, "left"),
    ("HoGRN", "WD-singer"): (16, 2, "left"),
    ("TuckER", "WD-singer"): (8, -16, "left"),
    ("ConvE", "WD-singer"): (-12, -8, "right"),
    ("StruProKGR", "WD-singer"): (-6, 12, "right"),
}
PNG_METADATA = {"Software": "SparseKGC"}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
    }
)


def load_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required table not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["Method"]: row for row in csv.DictReader(handle)}


def get_value(rows: dict[str, dict[str, str]], method: str, field: str) -> float:
    try:
        raw = rows[method][field]
        if raw in ("", "—"):
            raise ValueError
        return float(raw)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Missing {field} for {method}") from exc


def save_figure(fig, out: Path) -> list[Path]:
    path = out / "efficiency_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", metadata=PNG_METADATA)
    plt.close(fig)
    return [path]


def generate(
    table_dir: Path = DEFAULT_TABLE_DIR,
    out: Path = DEFAULT_FIGURE_DIR,
) -> list[Path]:
    table_dir = Path(table_dir).resolve()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    main_rows = load_rows(table_dir / "main_table.csv")
    efficiency_rows = load_rows(table_dir / "efficiency_table.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))

    for ax, dataset in zip(axes, EFF_DATASETS):
        points = []
        for method in PLOT_METHODS:
            mrr = get_value(main_rows, method, f"{dataset}_MRR")
            seconds = get_value(efficiency_rows, method, f"{dataset}_raw_s")
            points.append((method, mrr, seconds))

        mrr_values = [point[1] for point in points]
        runtime_values = [point[2] for point in points]
        y_min, y_max = min(mrr_values), max(mrr_values)
        y_pad = (y_max - y_min) * 0.18
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_xscale("log")
        ax.set_xlim(min(runtime_values) * 0.45, max(runtime_values) * 2.4)

        for method, mrr, seconds in points:
            color, marker = STYLE[method]
            is_pathbsr = method == "PathBSR"
            ax.scatter(
                seconds,
                mrr,
                color=color,
                marker=marker,
                s=460 if is_pathbsr else 95,
                zorder=6 if is_pathbsr else 4,
                edgecolors="white",
                linewidths=1.4 if is_pathbsr else 0.8,
            )
            dx, dy, align = LABEL_OVERRIDE.get(
                (method, dataset),
                (0, 15 if is_pathbsr else 11, "center"),
            )
            ax.annotate(
                method,
                xy=(seconds, mrr),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=align,
                va="bottom",
                fontsize=8.5 if is_pathbsr else 7.5,
                fontweight="bold" if is_pathbsr else "normal",
                color=color if is_pathbsr else "#333333",
            )

        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda value, _: f"{int(value)}" if value >= 1 else f"{value:.1f}")
        )
        ax.set_xlabel("Runtime (seconds, log scale)", fontsize=10)
        if dataset == EFF_DATASETS[0]:
            ax.set_ylabel("MRR", fontsize=10)
        ax.set_title(dataset, fontsize=11, fontweight="bold", pad=8)
        ax.grid(True, which="major", linestyle="--", alpha=0.30, zorder=0)
        ax.grid(True, which="minor", linestyle=":", alpha=0.15, zorder=0)
        ax.tick_params(labelsize=8.5)

    fig.tight_layout(pad=1.5)
    return save_figure(fig, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_FIGURE_DIR)
    args = parser.parse_args()
    generated = generate(args.tables, args.out)
    print(f"Generated {len(generated)} efficiency figure files in {args.out.resolve()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate MRR comparison plots from the generated result tables."""

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

DATASETS = [
    "FB15K-237-10",
    "FB15K-237-20",
    "FB15K-237-50",
    "NELL23K",
    "WD-singer",
]
MODEL_ORDER = [
    "TransE",
    "DistMult",
    "ComplEx",
    "ConvE",
    "RotatE",
    "TuckER",
    "Prob-CBR",
    "AnyBURL",
    "DacKGR",
    "HoGRN",
    "LoGRe",
    "StruProKGR",
    "PathBSR",
]
BASELINE_COLORS = [
    "#7c7c7c",
    "#1f77b4",
    "#6baed6",
    "#4c72b0",
    "#55a868",
    "#c4a000",
    "#8172b3",
    "#dd8452",
    "#b0b0b0",
    "#64b5cd",
    "#c44e52",
    "#937860",
]
PATHBSR_COLOR = "#cc0000"
PNG_METADATA = {"Software": "SparseKGC"}

PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": "#555555",
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def load_mrr(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        raise FileNotFoundError(f"Required table not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = {row["Method"]: row for row in csv.DictReader(handle)}

    data: dict[str, list[float]] = {}
    for method in MODEL_ORDER:
        if method not in rows:
            raise ValueError(f"Method {method!r} is missing from {path}")
        values = []
        for dataset in DATASETS:
            raw = rows[method].get(f"{dataset}_MRR", "")
            if raw in ("", "—"):
                raise ValueError(f"Missing MRR for {method} on {dataset} in {path}")
            values.append(float(raw) * 100.0)
        data[method] = values
    return data


def draw_panel(ax, data: dict[str, list[float]], title: str, show_legend: bool) -> None:
    x_values = list(range(len(DATASETS)))
    for index, method in enumerate(MODEL_ORDER[:-1]):
        ax.plot(
            x_values,
            data[method],
            color=BASELINE_COLORS[index],
            linewidth=1.4,
            marker="o",
            markersize=4.5,
            alpha=0.65,
            label=method,
            zorder=3,
        )

    pathbsr = data["PathBSR"]
    ax.plot(
        x_values,
        pathbsr,
        color=PATHBSR_COLOR,
        linewidth=3.2,
        marker="o",
        markersize=7,
        label="PathBSR",
        zorder=6,
    )
    ax.annotate(
        "PathBSR",
        xy=(x_values[-1], pathbsr[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=10,
        fontweight="bold",
        color=PATHBSR_COLOR,
    )

    ax.set_xticks(x_values)
    ax.set_xticklabels(DATASETS, rotation=15, ha="right", fontsize=9.5)
    ax.set_ylabel("MRR (×100)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(axis="y", which="major", linestyle="--", alpha=0.30, zorder=0)
    ax.grid(axis="y", which="minor", linestyle=":", alpha=0.15, zorder=0)
    ax.set_xlim(-0.3, len(DATASETS) - 0.3)
    if show_legend:
        ax.legend(
            loc="upper left",
            ncol=3,
            fontsize=8.0,
            framealpha=0.88,
            edgecolor="#cccccc",
            handlelength=1.8,
        )


def save_figure(fig, stem: Path) -> list[Path]:
    path = stem.with_suffix(".png")
    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
        metadata=PNG_METADATA,
    )
    plt.close(fig)
    return [path]


def generate(
    table_dir: Path = DEFAULT_TABLE_DIR,
    out: Path = DEFAULT_FIGURE_DIR,
) -> list[Path]:
    plt.rcdefaults()
    plt.rcParams.update(PLOT_STYLE)
    table_dir = Path(table_dir).resolve()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    main_data = load_mrr(table_dir / "main_table.csv")
    sota_data = load_mrr(table_dir / "sota_table.csv")
    generated: list[Path] = []

    for data, title, name in (
        (main_data, "Main Protocol — MRR", "mrr_lines_main"),
        (sota_data, "SOTA Protocol — MRR", "mrr_lines_sota"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        draw_panel(ax, data, title, show_legend=True)
        fig.tight_layout()
        generated.extend(save_figure(fig, out / name))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4))
    draw_panel(axes[0], main_data, "Main Protocol — MRR", show_legend=False)
    draw_panel(axes[1], sota_data, "SOTA Protocol — MRR", show_legend=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=7,
        fontsize=9,
        framealpha=0.9,
        edgecolor="#cccccc",
    )
    fig.subplots_adjust(bottom=0.20, left=0.06, right=0.97, top=0.90, wspace=0.12)
    generated.extend(save_figure(fig, out / "mrr_lines"))
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_FIGURE_DIR)
    args = parser.parse_args()
    generated = generate(args.tables, args.out)
    print(f"Generated {len(generated)} MRR figure files in {args.out.resolve()}")


if __name__ == "__main__":
    main()

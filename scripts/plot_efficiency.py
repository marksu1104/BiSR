#!/usr/bin/env python3
"""Generate efficiency scatter plot (MRR vs log-runtime) for the thesis."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS   = REPO_ROOT / "outputs"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
})

PLOT_METHODS = [
    "TransE", "ConvE", "TuckER",
    "AnyBURL", "HoGRN", "LoGRe", "StruProKGR", "PathBSR",
]

EFF_DATASETS = ["FB15K-237-10", "NELL23K", "WD-singer"]

MAIN_SRC = {
    "TransE":     (OUTPUTS / "traditional_metrics.csv",    "TransE"),
    "ConvE":      (OUTPUTS / "traditional_metrics.csv",    "ConvE"),
    "TuckER":     (OUTPUTS / "traditional_metrics.csv",    "TuckER"),
    "AnyBURL":    (OUTPUTS / "anyburl_metrics.csv",        "AnyBURL"),
    "HoGRN":      (OUTPUTS / "hogrn_metrics.csv",          "conve"),
    "LoGRe":      (OUTPUTS / "logre_metrics.csv",          "LoGRe"),
    "StruProKGR": (OUTPUTS / "struprokgr_metrics.csv",     "StruProKGR"),
    "PathBSR":    (OUTPUTS / "pathbsr_metrics.csv",        "PathBSR"),
}

# color, marker
STYLE = {
    "TransE":     ("#4c72b0", "o"),
    "ConvE":      ("#55a868", "s"),
    "TuckER":     ("#c4a000", "^"),
    "AnyBURL":    ("#dd8452", "D"),
    "HoGRN":      ("#8172b3", "v"),
    "LoGRe":      ("#c44e52", "P"),
    "StruProKGR": ("#937860", "X"),
    "PathBSR":    ("#cc0000", "*"),
}

# Per-(method, dataset) label offset overrides (dx_pts, dy_pts, ha)
# to separate overlapping labels. Default is (0, 11, "center").
LABEL_OVERRIDE = {
    ("ConvE",      "NELL23K"):   (-14, -14, "right"),
    ("TuckER",     "NELL23K"):   (14,  -14, "left"),
    ("StruProKGR", "NELL23K"):   (-10, 11,  "right"),
    ("HoGRN",      "NELL23K"):   (10,  11,  "left"),
    ("HoGRN",      "WD-singer"): (16,  2,   "left"),
    ("TuckER",     "WD-singer"): (8,   -16, "left"),
    ("ConvE",      "WD-singer"): (-12, -8,  "right"),
    ("StruProKGR", "WD-singer"): (-6,  12,  "right"),
}

_cache: dict = {}

def _load(path):
    key = str(path)
    if key not in _cache:
        _cache[key] = [] if not path.exists() else list(csv.DictReader(open(path)))
    return _cache[key]

def _get(path, mkey, ds, field):
    for row in _load(path):
        if row.get("Model", row.get("model","")) == mkey and \
           row.get("Dataset", row.get("dataset","")) == ds:
            try: return float(row.get(field, ""))
            except: return None
    return None

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=REPO_ROOT / "exp_results")
    return p.parse_args()

def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))

    for ax, ds in zip(axes, EFF_DATASETS):
        pts = []
        for m in PLOT_METHODS:
            path, mkey = MAIN_SRC[m]
            sec = _get(path, mkey, ds, "seconds")
            mrr = _get(path, mkey, ds, "MRR_Avg")
            if sec is None or mrr is None:
                continue
            pts.append((m, mrr, sec))

        mrrs = [p[1] for p in pts]
        secs = [p[2] for p in pts]
        y_min, y_max = min(mrrs), max(mrrs)
        y_pad = (y_max - y_min) * 0.18
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_xscale("log")
        ax.set_xlim(min(secs) * 0.45, max(secs) * 2.4)

        for m, mrr, sec in pts:
            color, marker = STYLE[m]
            is_pb = (m == "PathBSR")
            ax.scatter(
                sec, mrr,
                color=color, marker=marker,
                s=460 if is_pb else 95,
                zorder=6 if is_pb else 4,
                edgecolors="white",
                linewidths=1.4 if is_pb else 0.8,
            )
            dx, dy, ha = LABEL_OVERRIDE.get((m, ds), (0, 15 if is_pb else 11, "center"))
            ax.annotate(
                m,
                xy=(sec, mrr),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha, va="bottom",
                fontsize=8.5 if is_pb else 7.5,
                fontweight="bold" if is_pb else "normal",
                color=color if is_pb else "#333333",
            )

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, _: f"{int(v)}" if v >= 1 else f"{v:.1f}"
        ))
        ax.set_xlabel("log(Runtime)  [seconds]", fontsize=10)
        if ds == EFF_DATASETS[0]:
            ax.set_ylabel("MRR", fontsize=10)
        ax.set_title(ds, fontsize=11, fontweight="bold", pad=8)
        ax.grid(True, which="major", linestyle="--", alpha=0.30, zorder=0)
        ax.grid(True, which="minor", linestyle=":", alpha=0.15, zorder=0)
        ax.tick_params(labelsize=8.5)

    plt.tight_layout(pad=1.5)
    for ext in ("png", "pdf"):
        out = args.out / f"efficiency_scatter.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)

if __name__ == "__main__":
    main()

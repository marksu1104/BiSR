"""
Generate Chapter 5 result tables for the PathBSR thesis.

Usage:
    python scripts/build_result_tables.py [--out exp_results]

Outputs (all written to --out directory):
    main_table.csv          MRR_Avg / Hits@3_Avg for all 6 datasets × 13 models
    sota_table.csv          MRR_Tail / Hits@3_Tail for all 6 datasets × 13 models
    efficiency_table.csv    Wall-clock runtime for 3 representative datasets
    tables.md               All three tables as Markdown
    tables.tex              All three tables as LaTeX (booktabs)
"""

import argparse
import csv
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS   = REPO_ROOT / "outputs"

MODEL_ORDER = [
    "TransE", "DistMult", "ComplEx", "ConvE", "RotatE", "TuckER",
    "Prob-CBR", "AnyBURL", "DacKGR", "HoGRN", "LoGRe", "StruProKGR",
    "PathBSR",
]

DATASETS = [
    "FB15K-237-10", "FB15K-237-20", "FB15K-237-50",
    "NELL23K", "WD-singer", "WN18RR",
]

# (csv_path, model_key_in_csv)
MAIN_SRC = {
    "TransE":     (OUTPUTS / "traditional_metrics.csv",        "TransE"),
    "RotatE":     (OUTPUTS / "traditional_metrics.csv",        "RotatE"),
    "DistMult":   (OUTPUTS / "traditional_metrics.csv",        "DistMult"),
    "ComplEx":    (OUTPUTS / "traditional_metrics.csv",        "ComplEx"),
    "ConvE":      (OUTPUTS / "traditional_metrics.csv",        "ConvE"),
    "TuckER":     (OUTPUTS / "traditional_metrics.csv",        "TuckER"),
    "Prob-CBR":   (OUTPUTS / "probcbr_metrics.csv",            "Prob-CBR"),
    "AnyBURL":    (OUTPUTS / "anyburl_metrics.csv",            "AnyBURL"),
    "DacKGR":     (OUTPUTS / "dackgr_metrics.csv",             "point.rs.conve"),
    "HoGRN":      (OUTPUTS / "hogrn_metrics.csv",              "conve"),
    "LoGRe":      (OUTPUTS / "logre_metrics.csv",              "LoGRe"),
    "StruProKGR": (OUTPUTS / "struprokgr_metrics.csv",         "StruProKGR"),
    "PathBSR":    (OUTPUTS / "pathbsr_metrics.csv",            "PathBSR"),
}

SOTA_SRC = {
    "TransE":     (OUTPUTS / "traditional_sota_metrics.csv",   "TransE"),
    "RotatE":     (OUTPUTS / "traditional_sota_metrics.csv",   "RotatE"),
    "DistMult":   (OUTPUTS / "traditional_sota_metrics.csv",   "DistMult"),
    "ComplEx":    (OUTPUTS / "traditional_sota_metrics.csv",   "ComplEx"),
    "ConvE":      (OUTPUTS / "traditional_sota_metrics.csv",   "ConvE"),
    "TuckER":     (OUTPUTS / "traditional_sota_metrics.csv",   "TuckER"),
    "Prob-CBR":   (OUTPUTS / "probcbr_metrics.csv",            "Prob-CBR"),
    "AnyBURL":    (OUTPUTS / "anyburl_metrics.csv",            "AnyBURL"),   # MRR_Tail in main csv
    "DacKGR":     (OUTPUTS / "dackgr_metrics.csv",             "point.rs.conve"),
    "HoGRN":      (OUTPUTS / "hogrn_metrics.csv",              "conve"),
    "LoGRe":      (OUTPUTS / "logre_sota_metrics.csv",         "LoGRe"),
    "StruProKGR": (OUTPUTS / "struprokgr_sota_metrics.csv",    "StruProKGR"),
    "PathBSR":    (OUTPUTS / "pathbsr_sota_metrics.csv",       "PathBSR"),
}


# Efficiency table: only these three representative datasets
EFF_DATASETS = ["FB15K-237-10", "NELL23K", "WD-singer"]

# Efficiency table: only these methods (highlight PathBSR vs representative baselines)
EFF_METHODS = ["TransE", "ConvE", "TuckER", "AnyBURL", "HoGRN", "LoGRe", "StruProKGR", "PathBSR"]

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

_csv_cache: dict = {}

def _load(path: Path):
    key = str(path)
    if key not in _csv_cache:
        if not path.exists():
            _csv_cache[key] = []
        else:
            with open(path) as f:
                _csv_cache[key] = list(csv.DictReader(f))
    return _csv_cache[key]


def _get(path: Path, model_key: str, dataset: str, field: str) -> str:
    for row in _load(path):
        m = row.get("Model", row.get("model", ""))
        d = row.get("Dataset", row.get("dataset", ""))
        if m == model_key and d == dataset:
            v = row.get(field, "")
            return v if v not in ("", "—", "N/A") else ""
    return ""


def _seconds(path: Path, model_key: str, dataset: str) -> float | None:
    v = _get(path, model_key, dataset, "seconds")
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fmt_time(sec) -> str:
    if sec is None:
        return "—"
    sec = float(sec)
    if sec < 60:
        return f"{sec:.0f}s"
    if sec < 3600:
        return f"{sec / 60:.1f}m"
    return f"{sec / 3600:.1f}h"


# ---------------------------------------------------------------------------
# Build table data structures
# ---------------------------------------------------------------------------

def build_main_table():
    """Returns list of dicts: {Method, DS1_MRR, DS1_H3, DS2_MRR, ...}"""
    rows = []
    for m in MODEL_ORDER:
        path, mkey = MAIN_SRC[m]
        row = {"Method": m}
        for ds in DATASETS:
            row[f"{ds}_MRR"]  = _get(path, mkey, ds, "MRR_Avg")  or "—"
            row[f"{ds}_H3"]   = _get(path, mkey, ds, "Hits@3_Avg") or "—"
        rows.append(row)
    return rows


def build_sota_table():
    rows = []
    for m in MODEL_ORDER:
        path, mkey = SOTA_SRC[m]
        row = {"Method": m}
        for ds in DATASETS:
            row[f"{ds}_MRR"]  = _get(path, mkey, ds, "MRR_Tail")  or "—"
            row[f"{ds}_H3"]   = _get(path, mkey, ds, "Hits@3_Tail") or "—"
        rows.append(row)
    return rows


def build_efficiency_table():
    rows = []
    for m in MODEL_ORDER:
        path, mkey = MAIN_SRC[m]
        row = {"Method": m}
        for ds in EFF_DATASETS:
            sec = _seconds(path, mkey, ds)
            row[ds] = _fmt_time(sec)
            row[f"{ds}_raw_s"] = str(sec) if sec is not None else ""
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_main_csv(rows, out: Path):
    cols = ["Method"] + [f"{ds}_{m}" for ds in DATASETS for m in ("MRR", "H3")]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)


def write_sota_csv(rows, out: Path):
    write_main_csv(rows, out)  # same shape


def write_efficiency_csv(rows, out: Path):
    cols = ["Method"] + [f"{ds}_time" for ds in EFF_DATASETS] + \
           [f"{ds}_raw_s" for ds in EFF_DATASETS]
    renamed = []
    for r in rows:
        nr = {"Method": r["Method"]}
        for ds in EFF_DATASETS:
            nr[f"{ds}_time"]  = r[ds]
            nr[f"{ds}_raw_s"] = r[f"{ds}_raw_s"]
        renamed.append(nr)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(renamed)


# ---------------------------------------------------------------------------
# Ranking helpers
# ---------------------------------------------------------------------------

def _top_ranks(rows, col_key, n, lower_is_better=False):
    """Return {method: rank} for top-n methods in column col_key (1-indexed)."""
    vals = {}
    for r in rows:
        v = r.get(col_key, "—")
        if v not in ("—", "", None):
            try:
                vals[r["Method"]] = float(v)
            except ValueError:
                pass
    sign = 1 if lower_is_better else -1
    ranked = sorted(vals, key=lambda m: sign * vals[m])
    return {m: (i + 1) for i, m in enumerate(ranked[:n])}


def _md_mark(cell, rank):
    if rank == 1: return f"**{cell}**"
    if rank == 2: return f"<u>{cell}</u>"
    if rank == 3: return f"`{cell}`"
    return cell


def _tex_mark(cell, rank):
    if rank == 1: return r"\textbf{" + cell + "}"
    if rank == 2: return r"\underline{" + cell + "}"
    if rank == 3: return r"\textit{" + cell + "}"
    return cell


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

def _md_mrr_table(rows, title, datasets=DATASETS):
    ds_headers = []
    for ds in datasets:
        label = "WD†" if ds == "WD-singer" else ds.replace("FB15K-237-", "FB")
        ds_headers.append(label)

    # Precompute top-3 ranks per dataset (by MRR)
    col_ranks = {}
    for ds in datasets:
        col_ranks[ds] = _top_ranks(rows, f"{ds}_MRR", n=3)

    header = "| Method | " + " | ".join(f"{h} MRR / H@3" for h in ds_headers) + " |"
    sep    = "| --- |" + " --- |" * len(datasets)
    lines  = [f"### {title}", "", header, sep]

    for r in rows:
        m = r["Method"]
        cells = []
        for ds in datasets:
            mrr = r.get(f"{ds}_MRR", "—")
            h3  = r.get(f"{ds}_H3",  "—")
            if mrr != "—" and h3 != "—":
                cell = f"{float(mrr):.3f} / {float(h3):.3f}"
            else:
                cell = "—"
            rank = col_ranks[ds].get(m)
            cells.append(_md_mark(cell, rank) if cell != "—" else cell)
        method_str = f"**{m}**" if m == "PathBSR" else m
        row_str = f"| {method_str} | " + " | ".join(cells) + " |"
        lines.append(row_str)
    return "\n".join(lines)


def _md_eff_table(rows):
    ds_labels = [ds.replace("FB15K-237-", "FB") for ds in EFF_DATASETS]

    # Precompute top-2 ranks per dataset (lower seconds = better)
    col_ranks = {}
    for ds in EFF_DATASETS:
        col_ranks[ds] = _top_ranks(rows, f"{ds}_raw_s", n=2, lower_is_better=True)

    header = "| Method | " + " | ".join(f"{h} (s)" for h in ds_labels) + " |"
    sep    = "| --- |" + " --- |" * len(EFF_DATASETS)
    lines  = ["### Efficiency — Wall-clock Runtime (seconds)",
              "",
              "Top-2 per column (fastest): **bold** (1st) / <u>underline</u> (2nd).",
              "",
              header, sep]
    for r in rows:
        m = r["Method"]
        cells = []
        for ds in EFF_DATASETS:
            raw = r.get(f"{ds}_raw_s", "")
            try:
                sec = int(round(float(raw)))
                cell = str(sec)
            except (ValueError, TypeError):
                cell = "—"
            rank = col_ranks[ds].get(m)
            cells.append(_md_mark(cell, rank) if cell != "—" else cell)
        method_str = f"**{m}**" if m == "PathBSR" else m
        row_str = f"| {method_str} | " + " | ".join(cells) + " |"
        lines.append(row_str)
    return "\n".join(lines)


def write_markdown(main, sota, eff, out: Path):
    sections = [
        "# Chapter 5 Results — PathBSR\n",
        "Protocol — **Main**: bidirectional filtered full-entity ranking, average-tie (MRR_Avg / Hits@3_Avg).  "
        "**SOTA**: tail-only, optimistic tie-breaking (MRR_Tail / Hits@3_Tail).  "
        "WD† = WD-singer (official split, exact-triple overlap caveat).  "
        "LoGRe / StruProKGR do not support WN18RR.\n",
        "Top-3 per column (by MRR): **bold** (1st) / <u>underline</u> (2nd) / `code` (3rd).\n",
        _md_mrr_table(main, "Main Protocol (Bidirectional, Average-tie)"),
        "",
        _md_mrr_table(sota, "SOTA Protocol (Tail-only, Optimistic)"),
        "",
        _md_eff_table(eff),
        "",
        "> Efficiency note: times are wall-clock on aarch64 GH200 GPU nodes.  "
        "PathBSR = path-rule mining + valid/test inference (tqdm logs).",
    ]
    out.write_text("\n".join(sections))


# ---------------------------------------------------------------------------
# LaTeX renderer
# ---------------------------------------------------------------------------

def _latex_mrr_table(rows, caption, label, datasets=DATASETS):
    ds_labels = []
    for ds in datasets:
        ds_labels.append("WD\\textsuperscript{†}" if ds == "WD-singer"
                         else ds.replace("FB15K-237-", "FB15K-").replace("WN18RR", "WN18RR"))

    col_spec = "l" + "c" * len(datasets)

    # Top-3 per dataset by MRR
    col_ranks = {ds: _top_ranks(rows, f"{ds}_MRR", n=3) for ds in datasets}

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        r"\textbf{Method} & " + " & ".join(f"\\textbf{{{h}}}" for h in ds_labels) + r" \\",
        r"& " + " & ".join(r"\small MRR / H@3" for _ in ds_labels) + r" \\",
        r"\midrule",
    ]

    separators = {"Prob-CBR", "PathBSR"}
    for r in rows:
        m = r["Method"]
        if m in separators and m != "TransE":
            lines.append(r"\midrule")
        cells = []
        for ds in datasets:
            mrr = r.get(f"{ds}_MRR", "—")
            h3  = r.get(f"{ds}_H3",  "—")
            if mrr != "—" and h3 != "—":
                cell = f"{float(mrr):.3f} / {float(h3):.3f}"
                rank = col_ranks[ds].get(m)
                cells.append(_tex_mark(cell, rank))
            else:
                cells.append("—")
        m_tex = m.replace("-", r"\text{-}")
        method_cell = r"\textbf{" + m_tex + "}" if m == "PathBSR" else m_tex
        lines.append(method_cell + " & " + " & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _latex_eff_table(rows):
    ds_labels = [ds.replace("FB15K-237-", "FB15K-") for ds in EFF_DATASETS]

    # Top-2 per dataset by raw_s (lower = better)
    col_ranks = {ds: _top_ranks(rows, f"{ds}_raw_s", n=2, lower_is_better=True)
                 for ds in EFF_DATASETS}

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Wall-clock runtime (seconds) on three representative datasets (aarch64 GH200). "
        r"PathBSR time = path-rule mining + valid/test inference.}",
        r"\label{tab:efficiency}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Method} & " + " & ".join(f"\\textbf{{{h}}}" for h in ds_labels) + r" \\",
        r"\midrule",
    ]
    separators = {"Prob-CBR", "PathBSR"}
    for r in rows:
        m = r["Method"]
        if m in separators and m != "TransE":
            lines.append(r"\midrule")
        cells = []
        for ds in EFF_DATASETS:
            raw = r.get(f"{ds}_raw_s", "")
            try:
                cell = str(int(round(float(raw))))
            except (ValueError, TypeError):
                cell = "—"
            rank = col_ranks[ds].get(m)
            cells.append(_tex_mark(cell, rank) if cell != "—" else cell)
        m_tex = m.replace("-", r"\text{-}")
        method_cell = r"\textbf{" + m_tex + "}" if m == "PathBSR" else m_tex
        lines.append(method_cell + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def write_latex(main, sota, eff, out: Path):
    parts = [
        "% Auto-generated by scripts/build_result_tables.py\n",
        _latex_mrr_table(
            main,
            r"Main protocol results (MRR / Hits@3, bidirectional filtered, average-tie). "
            r"WD\textsuperscript{†} = WD-singer. — = not supported.",
            "tab:main",
        ),
        "",
        _latex_mrr_table(
            sota,
            r"SOTA-comparison results (MRR / Hits@3, tail-only, optimistic tie-breaking). "
            r"WD\textsuperscript{†} = WD-singer.",
            "tab:sota",
        ),
        "",
        _latex_eff_table(eff),
    ]
    out.write_text("\n".join(parts))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="exp_results",
                        help="Output directory (relative to repo root or absolute)")
    args = parser.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    main_rows = build_main_table()
    sota_rows = build_sota_table()
    eff_rows  = build_efficiency_table()

    write_main_csv(main_rows, out / "main_table.csv")
    write_sota_csv(sota_rows, out / "sota_table.csv")
    write_efficiency_csv(eff_rows, out / "efficiency_table.csv")
    write_markdown(main_rows, sota_rows, eff_rows, out / "tables.md")
    write_latex(main_rows, sota_rows, eff_rows, out / "tables.tex")

    print(f"Written to {out}/")
    for f in sorted(out.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()

# SparseKGC — AI Assistant Context

## Project Overview

**Researcher:** marksu (graduate student under Professor csliao)
**Method:** PathBSR — BM25-proxy CBR + multi-hop path mining, built on the Prob-CBR architecture
**Goal:** Sparse Knowledge Graph Completion (KGC) paper
**Datasets:** WD-singer, FB15K-237-10, FB15K-237-20, FB15K-237-50, WN18RR, NELL23K (6 total)

---

## Cluster Environment

**Login node:** x86\_64 (A09–A12) — code editing only, no experiments

**Experiment nodes (all experiments run here):**
- `gpu_long` partition: gpu1, aarch64 GH200, up to 3 days — default for all baselines
- `gpu_short` partition: gpu2/3/4, aarch64 GH200, up to 12 hours — quick re-runs

**Python environment:** `.venv/` inside this repo (aarch64, created by `sbatch exp_setup_venv.sh`)
- Has: torch (CUDA), numpy, tqdm, scipy, torch\_scatter, matplotlib, etc.
- All `exp_*.sh` scripts activate this venv: `source .venv/bin/activate`

**Java (AnyBURL):** `tools/jdk-21.0.11+10-aarch64/` — `run_anyburl.py` auto-detects arch

**CRITICAL: Never run `scancel`.** It causes nodes to drain. This has happened multiple times — wait for jobs to finish or time out naturally. Do not scancel under any circumstances.

---

## Repository Structure

```
SparseKGC/
├── run_baseline.py          # Unified CLI for all baselines
├── exp_*.sh                 # Slurm sbatch scripts (gpu_long, aarch64)
├── exp_setup_venv.sh        # One-time: create .venv/ on GPU node
├── baselines/
│   ├── entity_types/        # Shared entity→type mapping (5 datasets, no WN18RR)
│   ├── AnyBURL/             # Java rule mining; run_anyburl.py + score_anyburl.py
│   ├── StruProKGR/          # Path-based CBR: prepare_data.py, StruProKGR.py,
│   │                        #   run_struprokgr.py, score_struprokgr.py
│   ├── LoGRe/               # Logical graph reasoning: LoGRe.py, run_logre.py,
│   │                        #   prepare_data.py (reuses score_struprokgr.py)
│   ├── Prob-CBR/            # GPU, run via run_baseline.py probcbr
│   ├── DacKGR/              # GPU, run via run_baseline.py dackgr
│   ├── HoGRN/               # GPU (ConvE), run via run_baseline.py hogrn
│   └── tranditional/        # TransE/RotatE/DistMult/ComplEx/ConvE/TuckER
├── scripts/
│   ├── metrics_csv.py       # upsert_metrics_csv() helper
│   ├── log_format.py        # print_start() / print_result()
│   └── export_bsr_routing_predictions.py
├── datasets/                # gitignored — h\tt\tr tab-separated, no inverse triples
├── outputs/                 # gitignored — CSV results per baseline
├── logs/                    # gitignored — sbatch job logs
├── tools/                   # gitignored — JDK21 (x86 + aarch64 versions)
├── .venv/                   # gitignored — aarch64 Python env
├── CLAUDE.md                # This file
└── README.md                # Project README
```

---

## How to Run Experiments

```bash
# Submit one job per baseline (all on gpu_long, aarch64):
sbatch exp_traditional.sh       # TransE/RotatE/DistMult/ComplEx/ConvE/TuckER
sbatch exp_hogrn.sh             # HoGRN (conve)
sbatch exp_dackgr.sh            # DacKGR
sbatch exp_probcbr.sh           # Prob-CBR
sbatch exp_anyburl.sh           # AnyBURL (Java, CPU-only but same node)
sbatch exp_struprokgr.sh        # StruProKGR (numpy CPU, 3-day limit)
sbatch exp_logre.sh             # LoGRe (GPU ansim via torch)

# Override datasets:
sbatch --export=ALL,DATASETS="WD-singer NELL23K" exp_hogrn.sh

# Dry-run to verify commands without executing:
python run_baseline.py struprokgr --datasets WD-singer --dry_run
```

---

## Evaluation Protocols

**Main table** (paper results — primary):
- Filtered + full-entity ranking + **bidirectional** (tail queries + head queries via inverse relation) + **tie-aware** average rank
- Output: `outputs/<name>_metrics.csv`

**SOTA comparison table** (literature baseline):
- Tail-only + optimistic tie-breaking (first-match)
- Output: `outputs/<name>_sota_metrics.csv`

**Score dump format** (used by StruProKGR and LoGRe scorers):
- Per query line: `h\tr\tgold_entity\tgold_score\tn_higher\tn_tied`
- Tie-aware rank: if `gold_score > 0`: `n_higher + (n_tied+1)/2`; else: `n_higher + (n_zero+1)/2`

**Bidirectional eval pipeline** (StruProKGR and LoGRe):
1. Forward run on `test.triples` → `dump_forward.tsv`
2. Inverse run on `test_inv.triples` (each `(h,t,r)` → `(t,h,r_inv)`) → `dump_inverse.tsv`
3. `score_struprokgr.evaluate(fwd, inv, data_root, dataset)` → MRR/H@k per tail/head/avg

---

## Data Formats

- **SparseKGC raw** (`datasets/*/`): `h\tt\tr` — NO inverse triples
- **StruProKGR/LoGRe prepared** (`baselines/*/work/*/`): `h\tt\tr` WITH `r_inv` inverse triples
- **AnyBURL** (`baselines/AnyBURL/work/*/`): `h\tr\tt` (relation middle)
- Inverse relation naming: append/strip `_inv` suffix (e.g., `P31` ↔ `P31_inv`)

---

## Baseline-Specific Notes

### StruProKGR
- Supported: WD-singer, FB15K-237-10/20/50, NELL23K (no WN18RR — no entity types)
- Pure numpy/CPU; GPU is allocated but unused (for hardware consistency)
- Slow on large datasets: FB15K-237-50 can take 15–20h → use 3-day `gpu_long` limit
- Bug fix in `data_utils.get_inv_relation`: returns `None` for relations unseen in training

### LoGRe
- Supported: same 5 datasets as StruProKGR
- Uses `torch.matmul` for ansim matrix when CUDA available (much faster than numpy)
- Shares `score_struprokgr.py` scorer (same dump format)
- `use_wandb=0` default; no MongoDB dependency

### AnyBURL
- Runs on same aarch64 node; uses `tools/jdk-21.0.11+10-aarch64/`
- Java learn process exits non-zero when time limit hits — this is **normal** (check rules file)

---

## Current Experiment Status

| Baseline | Status | Notes |
|----------|--------|-------|
| TransE, RotatE, DistMult, ComplEx, ConvE, TuckER | Done | `outputs/traditional_metrics.csv` |
| HoGRN | Done | `outputs/hogrn_metrics.csv` |
| DacKGR | Done | `outputs/dackgr_metrics.csv` |
| Prob-CBR | Done | `outputs/probcbr_metrics.csv` |
| AnyBURL | Done | `outputs/anyburl_metrics.csv` |
| LoGRe | Done (5 datasets) | `outputs/logre_metrics.csv` |
| StruProKGR | In progress | Job 176606 running, WD-singer + 4 datasets |
| PathBSR | Done (tail-only exports) | WD-singer MRR=0.449, WN18RR=0.470, etc. |

---

## Key Results (tail-only MRR, PathBSR vs LoGRe paper)

| Dataset | PathBSR | LoGRe (reported) | Δ |
|---------|---------|-----------------|---|
| FB15K-237-10 | 0.242 | 0.228 | +0.014 ✓ |
| FB15K-237-20 | 0.272 | 0.261 | +0.011 ✓ |
| FB15K-237-50 | 0.318 | 0.297 | +0.021 ✓ |
| NELL23K | 0.282 | 0.259 | +0.023 ✓ |
| WD-singer | 0.449 | 0.459 | −0.010 ✗ |

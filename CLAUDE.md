# SparseKGC — AI Assistant Context

## Project Overview

**Researcher:** marksu (graduate student under Professor csliao)
**Method:** PathBSR — BM25 proxy CBR + multi-hop path mining, built on top of the Prob-CBR architecture
**Goal:** Sparse Knowledge Graph Completion (KGC) paper
**Datasets:** FB15K-237-10, FB15K-237-20, FB15K-237-50, NELL23K, WD-singer, WN18RR (6 total)

---

## Slurm Cluster Environment

**Login nodes:** x86_64 (A09–A12)

**GPU nodes (aarch64 / GH200):**
- `gpu_long` partition: gpu1, up to 3 days
- `gpu_short` partition: gpu2/3/4, up to 12 hours

**CPU nodes (x86_64):**
- `short` partition: A09–A12, up to 8 hours

**CRITICAL: Never run `scancel`.** It causes nodes to drain. This has happened multiple times — do not scancel jobs under any circumstances.

**Python environments:**
- `.venv` (aarch64 only): `/storage/professor/csliao/marksu/.venv` — use on GPU nodes
- x86 CPU nodes: use system `python3` (has numpy, tqdm; no torch)
- `PYTHON_BIN` env var controls which Python `run_baseline.py` uses

---

## Repository Structure

```
SparseKGC/
├── run_baseline.py          # Unified baseline execution entry point
├── exp_*.sh                 # sbatch scripts (one per baseline)
├── RESULTS_SUMMARY.md       # Current experiment results
├── RUN_EXPERIMENTS.md       # How to run experiments
├── CLAUDE.md                # This file
├── datasets/                # 6 datasets (not tracked by git)
├── outputs/                 # All experiment result CSVs (not tracked)
├── logs/                    # sbatch job logs (not tracked)
├── checkpoints/             # Model checkpoints (not tracked)
├── baselines/
│   ├── entity_types/        # Shared entity2type files (5 datasets; WN18RR excluded)
│   ├── AnyBURL/             # Rule mining (Java, CPU-only, x86)
│   ├── StruProKGR/          # Structure-aware CBR (numpy, CPU, x86)
│   ├── LoGRe/               # Logical Graph Reasoning (numpy/torch, CPU/GPU)
│   ├── Prob-CBR/            # Prob-CBR (GPU, aarch64)
│   ├── DacKGR/              # DacKGR (GPU, aarch64)
│   ├── HoGRN/               # HoGRN (GPU, aarch64)
│   └── tranditional/        # TransE/DistMult/ComplEx/ConvE/RotatE/TuckER (GPU)
└── scripts/
    ├── metrics_csv.py       # Upsert CSV helper
    ├── log_format.py        # Unified log format
    └── export_bsr_routing_predictions.py
```

---

## Evaluation Protocols (Important!)

**Main table protocol** (paper primary results):
- Filtered + full-entity ranking + **bidirectional** (head queries via inverse relation) + **tie-aware** average rank
- CSV: `outputs/*_metrics.csv`

**SOTA comparison protocol** (literature protocol):
- Tail-only + optimistic tie-breaking (used in LoGRe / StruProKGR papers)
- CSV: `outputs/*_sota_metrics.csv`

**Note:** LoGRe paper's AnyBURL numbers are bidirectional; other reported methods are tail-only (mixed protocols in the paper!).

---

## How to Run Experiments

### CPU/x86 baselines (short partition)
```bash
sbatch exp_anyburl.sh
sbatch exp_struprokgr.sh
sbatch exp_logre.sh
```

### GPU/aarch64 baselines (gpu_long or gpu_short)
```bash
sbatch exp_traditional.sh       # TransE/RotatE/DistMult/ComplEx/ConvE/TuckER
sbatch exp_hogrn.sh             # HoGRN (conve)
sbatch exp_dackgr.sh            # DacKGR
sbatch exp_probcbr.sh           # Prob-CBR
```

### Unified entry point
```bash
python run_baseline.py [baseline] --datasets [datasets...]
# baseline choices: traditional, hogrn, dackgr, probcbr, anyburl, struprokgr, logre
```

### Override datasets for a single submit
```bash
sbatch --export=ALL,DATASETS="WD-singer FB15K-237-10" exp_struprokgr.sh
```

---

## Data Formats

| Format | Description |
|--------|-------------|
| SparseKGC raw | `h\tt\tr` (tab-separated, no inverse triples) |
| StruProKGR / LoGRe | `h\tt\tr` WITH explicit `r_inv` inverse triples |
| AnyBURL | `h\tr\tt` (relation in the middle) |

**Bidirectional evaluation:** feed `test_inv.triples` (each test triple `(h, t, r)` inverted to `(t, h, r_inv)`) for head queries.

---

## AnyBURL Notes

- Java JDK21 path: `/storage/professor/csliao/marksu/tools/jdk-21.0.11+10/bin/java`
- Learn process exits with non-zero code when time limit is hit — this is **normal** (time-based stop). Check that the rules file exists to confirm success.

---

## Baseline-specific Notes

### StruProKGR
- Supported datasets: FB15K-237-10/20/50, NELL23K, WD-singer (no WN18RR — no entity types)
- Pure numpy/CPU; runs on x86 short partition

### LoGRe
- Supported datasets: same as StruProKGR (no WN18RR)
- numpy/CPU by default; uses torch for GPU-accelerated ansim matrix if CUDA available
- Imports torch with try/except — works without it

---

## Current Experiment Status

| Baseline | Status |
|----------|--------|
| TransE, DistMult, ComplEx, RotatE, ConvE, TuckER | Done |
| AnyBURL | Done (WD-singer MRR=0.392 bidirectional) |
| Prob-CBR | Done |
| DacKGR | Done |
| HoGRN | Done |
| StruProKGR | Integrated; pending full dataset run |
| LoGRe | Integrated; pending run |
| PathBSR | Done (tail-only: WD-singer=0.449, WN18RR=0.470, etc.) |

---

## Key Numbers (tail-only MRR, vs. LoGRe paper)

PathBSR vs LoGRe (reported):
- FB15K-237-10: 0.242 vs 0.228 (+0.014)
- FB15K-237-20: 0.272 vs 0.261 (+0.011)
- FB15K-237-50: 0.318 vs 0.297 (+0.021)
- NELL23K: 0.282 vs 0.259 (+0.023)
- WD-singer: 0.449 vs 0.459 (-0.010) — PathBSR slightly behind here

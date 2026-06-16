# SparseKGC

Research project for **PathBSR** — a sparse Knowledge Graph Completion method combining BM25-proxy Case-Based Reasoning with multi-hop path mining, built on the Prob-CBR framework.

**Author:** marksu (Graduate student, Professor csliao's lab)

---

## Setup

### 1. Clone and enter the project

```bash
git clone <repo-url> SparseKGC
cd SparseKGC
```

### 2. Create the Python environment (aarch64 GPU node only)

All experiments run on `gpu_long` (aarch64 GH200). Submit the one-time setup job from the **login node**:

```bash
sbatch exp_setup_venv.sh
```

This creates `.venv/` inside the repo. After it finishes, all other `exp_*.sh` scripts are ready to use.

### 3. Place datasets

Datasets are not tracked by git. Put them under `datasets/`:

```
datasets/
├── WD-singer/
├── FB15K-237-10/
├── FB15K-237-20/
├── FB15K-237-50/
├── WN18RR/
└── NELL23K/
```

Each dataset folder must contain `train.txt`, `valid.txt`, `test.txt` in `head\ttail\trelation` tab-separated format.

---

## Running Experiments

All baselines are submitted via a single sbatch script each. Results are written to `outputs/` as CSV files.

```bash
sbatch exp_traditional.sh    # TransE / RotatE / DistMult / ComplEx / ConvE / TuckER
sbatch exp_hogrn.sh          # HoGRN (ConvE score function)
sbatch exp_dackgr.sh         # DacKGR
sbatch exp_probcbr.sh        # Prob-CBR
sbatch exp_anyburl.sh        # AnyBURL (rule mining, Java)
sbatch exp_struprokgr.sh     # StruProKGR (path-based CBR, numpy)
sbatch exp_logre.sh          # LoGRe (logical graph reasoning, torch GPU ansim)
```

To run a subset of datasets:

```bash
sbatch --export=ALL,DATASETS="WD-singer NELL23K" exp_hogrn.sh
```

All scripts use the unified `.venv/` on `gpu_long` (aarch64). The underlying entry point is `run_baseline.py`:

```bash
python run_baseline.py <baseline> --datasets <dataset...> [--dry_run]
```

---

## Repository Structure

```
SparseKGC/
│
├── run_baseline.py          # Unified CLI entry point for all baselines
├── exp_*.sh                 # Slurm sbatch scripts (one per baseline)
├── exp_setup_venv.sh        # One-time: create .venv/ on GPU node
│
├── baselines/
│   ├── entity_types/        # Shared entity→type files (5 datasets)
│   ├── AnyBURL/             # Rule mining (Java, aarch64 JDK in tools/)
│   ├── StruProKGR/          # Structure-aware CBR (numpy, CPU)
│   ├── LoGRe/               # Logical Graph Reasoning (torch GPU ansim)
│   ├── Prob-CBR/            # Prob-CBR (GPU)
│   ├── DacKGR/              # DacKGR (GPU)
│   ├── HoGRN/               # HoGRN (GPU)
│   └── tranditional/        # KGE models: TransE/RotatE/DistMult/ComplEx/ConvE/TuckER
│
├── scripts/
│   ├── metrics_csv.py       # Upsert-by-key CSV helper
│   ├── log_format.py        # Structured stdout logging
│   └── export_bsr_routing_predictions.py
│
├── datasets/                # Not tracked — place dataset folders here
├── outputs/                 # Not tracked — CSV results written here
├── logs/                    # Not tracked — sbatch job logs
├── tools/                   # Not tracked — JDK21 (x86 + aarch64)
├── .venv/                   # Not tracked — aarch64 Python venv (created by exp_setup_venv.sh)
│
├── CLAUDE.md                # Context file for AI coding assistants
└── README.md                # This file
```

---

## Evaluation Protocols

Two protocols are used to fill two separate result tables:

| Protocol | Table | Queries | Tie-breaking | Filter |
|----------|-------|---------|--------------|--------|
| **Main** | Paper primary results | Bidirectional (tail + head via inverse) | Tie-aware average rank | Full-entity filtered |
| **SOTA comparison** | Literature baseline comparison | Tail-only | Optimistic first-match | Full-entity filtered |

Each baseline runner outputs two CSVs:
- `outputs/<name>_metrics.csv` — main protocol
- `outputs/<name>_sota_metrics.csv` — SOTA comparison protocol

---

## Cluster Notes

| Partition | Nodes | Architecture | Time limit | Use |
|-----------|-------|--------------|------------|-----|
| `gpu_long` | gpu1 | aarch64 GH200 | 3 days | All experiments (default) |
| `gpu_short` | gpu2/3/4 | aarch64 GH200 | 12 hours | Quick runs |
| `short` | A09–A12 | x86\_64 | 24 hours | Login-node tasks only |

**Never `scancel` a job** — it causes node drain. Wait for jobs to finish or time out naturally.

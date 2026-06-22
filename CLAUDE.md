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

**Python environment:** `.venv/` inside this repo (aarch64, created by `sbatch exp_setup_venv.sh` from pinned `requirements.txt`)
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
| TransE, RotatE, DistMult, ComplEx, ConvE | Done | `outputs/traditional_metrics.csv` |
| **TuckER** | **FIXED + rerun DONE** | emb_drop=0.3 fix reproduced in full seed-20 rerun: tail FB10 .255 / FB20 .270 / FB50 .316 / NELL .275 / WD .422 (all ≈ paper, Δ≤0.003). WN18RR uses emb_drop=0 (dense set) → .480, in line with peers. Whole traditional table regenerated at seed 20. |
| HoGRN | Done; rerun PENDING (186027) | Validated; reproducible rerun waits on gpu_long |
| DacKGR | Done; rerun PENDING (186028) | Validated (0.216≈paper 0.218); rerun waits on gpu_long |
| Prob-CBR | Done + reproduced | `outputs/probcbr_metrics.csv` (rerun 186025) |
| AnyBURL | Done + reproduced | `outputs/anyburl_metrics.csv` (rerun 186030, now via run_baseline.py) |
| LoGRe | **Done + ARCHIVED** ✅ | 5/5 match paper → `outputs/archive/` |
| StruProKGR | **Done + ARCHIVED** ✅ | 5/5 match paper → `outputs/archive/` |
| PathBSR | Done | `outputs/pathbsr_metrics.csv` + `outputs/pathbsr_sota_metrics.csv` |

---

## Key Results

### Main protocol (Bidirectional, Tie-aware, MRR_Avg)

| Dataset | PathBSR | LoGRe |
|---------|---------|-------|
| FB15K-237-10 | 0.1636 | 0.1053 |
| FB15K-237-20 | 0.1855 | 0.1520 |
| FB15K-237-50 | 0.2260 | 0.2012 |
| NELL23K | 0.2425 | 0.2023 |
| WD-singer | 0.3830 | 0.4059 |
| WN18RR | 0.4561 | — |

### SOTA protocol (Tail-only, Optimistic, MRR_Tail)

| Dataset | PathBSR | LoGRe (our run) | Δ |
|---------|---------|-----------------|---|
| FB15K-237-10 | 0.2547 | 0.1655 | +0.089 ✓ |
| FB15K-237-20 | 0.2759 | 0.2291 | +0.047 ✓ |
| FB15K-237-50 | 0.3192 | 0.2911 | +0.028 ✓ |
| NELL23K | 0.3017 | 0.2408 | +0.061 ✓ |
| WD-singer | 0.4670 | 0.4135 | +0.054 ✓ |
| WN18RR | 0.4884 | — | — |

---

## Two paper protocols (CRITICAL — explains why baselines have two target numbers)

The embedding baselines have **two different published numbers** for the same
model+dataset, because two source papers use two eval protocols:

- **HoGRN paper** = bidirectional avg `(tail+head)/2`, tie-aware → equals our
  **Main table** (`MRR_Avg`). e.g. FB10 ConvE = 0.165.
- **DacKGR / StruProKGR paper** = tail-only, DacKGR-framework topk-128 → equals
  our **SOTA table** (`MRR_Tail`). e.g. FB10 ConvE = 0.245.

Our runs reproduce BOTH: ConvE Avg 0.157≈HoGRN-0.165, Tail 0.243≈DacKGR-0.245.
HoGRN-as-a-method also appears tail-only (0.257) in the StruProKGR SOTA table.
TransE/DistMult run **higher** than the SOTA paper (user accepts "high is OK");
the only genuine "too low" problem was **TuckER** (see below).
DacKGR-paper baseline targets (tail MRR): see `outputs/archive/ARCHIVE_MANIFEST.md`
and memory `baseline-two-paper-protocols`.

---

## TuckER investigation — ROOT CAUSE FOUND (embedding dropout)

**Problem:** traditional TuckER capped at ~0.19 (FB10 tail) vs DacKGR's identical
model reaching 0.253. Exhaustively ruled out (all identical): model architecture,
data, lr, batch, epochs, optimizer, eval mode, tie-breaking, label-smoothing,
steps/epoch, seed. d_r-shrink/batch/epochs/seed all failed; weight-decay (1e-6)
and DacKGR label-smoothing (eps/N) each gave only +0.03 → ~0.22 ceiling; strong
wd (≥1e-5) collapsed to 0.05; 4000-epoch "grok" did NOT happen.

**Root cause:** DacKGR wraps **every embedding lookup** in `Dropout(0.3)` —
`EDropout` on entity input + on the **output projection matrix**
(`get_all_entity_embeddings`), `RDropout` on relations
(`src/knowledge_graph.py:339-349`, `emb_dropout_rate=0.3`). The traditional
reimplementation **omitted this** (it only had TuckER's internal
input/hidden dropouts 0.3/0.4/0.5). That extra regularization is what keeps
TuckER on the plateau long enough to generalize (~0.25) instead of overfitting
(~0.22).

**Fix (implemented):** added `E_dropout`/`R_dropout` to traditional TuckER on
entity-input, relation, and output-projection (`models/kge_models.py` TuckER),
controlled by `--tucker_emb_drop` (default 0.0 = legacy; **0.3 = DacKGR**).

**VALIDATED (FB10, jobs 186122/186123, 1500 epochs):** the grok happened.
- emb_drop=0.3 **only** (186123): valid peak **0.2593**, stable plateau ~0.258.
- emb_drop=0.3 **+ ls_dackgr=1** (186122): valid peak 0.2586, declines to ~0.241.
- Final test (FB10): **tail 0.2549 / avg 0.1682**, both **≥** targets 0.252/0.163.

**Winner = emb_drop 0.3 WITHOUT label smoothing.** This overturns the earlier
hypothesis that DacKGR-style label smoothing was needed — it is not, and slightly
hurts. ⚠️ The two jobs reported *identical* test numbers because the checkpoint
path `best_model_{model}_{dataset}.pth` had no run-id, so concurrent same-config
runs overwrote each other (now fixed via `--run_tag`, see below). The valid
trajectories are per-process and reliable; they are what distinguishes the two.

**FINAL `run_all.py` config (validated across all 6 datasets):**
- Unified budget (DEFAULT_ARGS, ALL models): max_epochs 2000, **patience 100**,
  eval_freq 1, seed 20, batch 256. patience chosen by simulating early stopping
  on the A/B valid curves: 100 → worst-case valid loss 0.0006; 80 → 0.0109 (WD
  has a meaningful late gain ~ep1729 needing >80). max_epochs 2000 because slow
  datasets peak ~ep1750 (FB50/WD); fast ones early-stop sooner.
- TuckER model-specific: lr 5e-4, **tucker_emb_drop 0.3**, selection sota.
  **batch 256 not DacKGR's 512** — A/B gave identical FB10 tail (0.2549) but grok
  ~40% faster, so slow datasets aren't cut off; 256 also = the table default.
- **`PER_DATASET_OVERRIDES[("TuckER","WN18RR")] = {"tucker_emb_drop": 0.0}`** —
  WN18RR is dense/standard, not sparse; emb_drop hurts it (0.34 vs 0.48) and it
  is in no sparse-KGC paper's TuckER comparison, so it uses standard TuckER.
- no ls_dackgr. Full seed-20 rerun (jobs 186576/577/578 + WN18RR ConvE/TuckER
  reflow 186674) merged into `outputs/traditional_*.csv`.

---

## Code changes this session (traditional KGE — `baselines/tranditional/`)

New CLI args on `main.py` (all default to legacy behavior so other models are
unaffected):
- `--tucker_emb_drop FLOAT` (0.0): TuckER embedding dropout — **THE TuckER fix**, set 0.3.
- `--ls_dackgr {0,1}` (0): label-smoothing additive `eps/N` (DacKGR) vs `1/N` (ConvE-std).
- `--tucker_rel_dim INT` (0): asymmetric TuckER d_r (investigated, did NOT help — leave 0).
- `--run_tag STR` (""): suffix for the checkpoint filename. Default keeps the
  legacy name (so `--eval_only` still finds it); pass distinct tags when running
  the **same model+dataset concurrently** to avoid overwriting each other's best
  model (this caused the identical-test-number artifact in 186122/186123).

DacKGR reference pipeline (`baselines/DacKGR/`): `experiment-emb.sh` guard widened
to allow tucker/distmult/transe; `configs/fb15k-237-10-{transe,distmult,tucker}.sh`
added; `src/emb/emb.py` `'!TransE'` left as-is (TransE→BCE, runs high=OK).

---

## Reproducibility / archive workflow

Validated baselines are sealed in `outputs/archive/` (CSVs + `ARCHIVE_MANIFEST.md`
documenting each vs its paper). Done: **LoGRe, StruProKGR**. The reproducible
sbatch entry points are the root `exp_*.sh` (all now route through
`run_baseline.py`, incl. AnyBURL).

## Next steps (resume here)

1. ✅ DONE: TuckER fixed (emb_drop 0.3) and the **full traditional table
   regenerated at seed 20** (all 6 models × 6 datasets, unified budget). Results
   merged into `outputs/traditional_*.csv`; verified vs paper (TuckER Δ≤0.003,
   ConvE within ~0.017; TransE/DistMult run high = accepted protocol artifact).
   TuckER diagnostics + the per-job isolated dirs have been cleaned up.
2. **Archive traditional** (seal `outputs/traditional_*.csv` into
   `outputs/archive/` + manifest). Optional: a second seed-20 pass to double-
   confirm reproducibility before sealing.
3. HoGRN (186027) + DacKGR (186028) reproducibility reruns are STILL REQUIRED
   (every baseline gets a rerun → confirm reproducible → archive). Waiting on
   gpu_long. Prob-CBR (186025) + AnyBURL (186030) already reproduced → can be
   archived now.
4. Compile final tables (`scripts/compile_tables.py`) once all archived.

NOTE: AnyBURL's row in the StruProKGR SOTA table is avg(tail+head), not tail —
our higher AnyBURL tail is correct, not a repro failure (see local memory).

# SparseKGC

SparseKGC is the research repository for **PathBSR**, a sparse knowledge graph
completion method that combines BM25-based proxy retrieval with multi-hop path
reasoning. The repository also contains adapted baseline implementations used in
the thesis experiments.

The primary repository goal is to regenerate every reported table and figure
from saved numerical results without rerunning expensive experiments.

## Quick Start

Python 3.12 is the canonical output environment. On standard Linux:

```bash
./setup_env.sh outputs
source ".venv-$(uname -m)/bin/activate"
python scripts/generate_outputs.py --check
python scripts/prepare_datasets.py --check
```

These checks validate all canonical datasets and rebuild every registered output
in a temporary directory for byte comparison. They do not modify files or run
experiments.

## Saved Outputs

Generate tables and figures from the saved metrics with:

```bash
python scripts/generate_outputs.py
python scripts/generate_outputs.py --tables
python scripts/generate_outputs.py --figures
```

The output layout is:

```text
results/
├── metrics/    # Saved final metrics produced by experiment code
├── tables/     # Generated CSV, Markdown, and LaTeX tables
└── figures/    # Generated PNG figures
```

Do not edit generated tables or figures by hand. Dataset conversions and caches
are created on demand under the ignored `outputs/preprocessed/` directory; the
tracked `datasets/` directory is always the canonical source.

## Run Experiments Locally

Experiment execution is optional and separate from saved-output generation:

```bash
./setup_env.sh experiments
source ".venv-$(uname -m)/bin/activate"
python run_baseline.py pathbsr --datasets NELL23K --dry_run
python run_baseline.py pathbsr --datasets NELL23K
```

The entry point supports:

```text
traditional  hogrn  dackgr  probcbr  anyburl  struprokgr  logre  pathbsr
```

Omit `--datasets` to use the six thesis datasets; request full `FB15K-237`
explicitly when needed. Raw runs, logs, checkpoints, and temporary predictions
remain under ignored working directories. Thesis metrics are saved separately
under `results/metrics/`.

PyTorch and CUDA wheels are platform dependent. `requirements.txt` records the
direct experiment dependencies. `requirements-lock.txt` preserves the complete
package snapshot used by the finished experiment environment and can be selected
with `./setup_env.sh experiments-lock`. A CUDA build compatible with the local
driver may need to be installed from the official PyTorch index.
AnyBURL additionally requires Java; its BSD-licensed release JAR is included.

## Run with Slurm

Slurm is an optional scheduling layer around the same Python entry point:

```bash
sbatch scripts/slurm/exp_pathbsr.sh
sbatch --export=ALL,DATASETS="WD-singer NELL23K" scripts/slurm/exp_hogrn.sh
```

Supply site-specific options such as `--account` or `--partition` through
`sbatch`. Set `SPARSEKGC_VENV` for a non-default environment path.

## Evaluation Protocols

| Protocol | Queries | Tie handling | Filtering |
| --- | --- | --- | --- |
| Main | Bidirectional tail and inverse-head queries | Average rank for ties | Full-entity filtered |
| SOTA comparison | Tail queries only | Optimistic first match | Full-entity filtered |

Both protocols rank over the full entity set and share the same filtering and
scoring; they differ only in tie handling. Sparse candidate methods (Prob-CBR,
AnyBURL, LoGRe, StruProKGR) assign score zero to entities they did not return,
which is each method's own definition of "no evidence" (their candidate scores
are non-negative sums), not an arbitrary placeholder. SOTA metrics are computed
independently under optimistic-tie handling from the same underlying
predictions as Main, not copied or reused from a method's native/original
tail-only output. See `REPRODUCIBILITY.md` for the per-baseline evaluator
mapping.

## Repository Structure

```text
SparseKGC/
├── baselines/       # PathBSR and adapted comparison methods
├── datasets/        # Canonical splits and input metadata
├── outputs/         # Ignored experiment working files
├── results/         # Saved metrics and generated paper outputs
├── scripts/         # Output, dataset, and Slurm utilities
├── run_baseline.py  # Platform-neutral experiment entry point
└── setup_env.sh     # Output or experiment environment setup
```

## Documentation

- [REPRODUCIBILITY.md](REPRODUCIBILITY.md): output contract and thesis artifact map.
- [THIRD_PARTY.md](THIRD_PARTY.md): upstream revisions, citations, licenses, and modifications.
- [datasets/README.md](datasets/README.md): dataset format and compatibility notes.

Main, SOTA, and efficiency results are integrated. Remaining structural,
ablation, and case-study artifacts will be added with their numerical sources
and generators. The thesis draft is not tracked.

The root MIT license covers project-authored work only. See `THIRD_PARTY.md` for
the status of vendored code and datasets.

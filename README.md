# SparseKGC

SparseKGC is the research repository for **PathBSR**, a sparse knowledge graph
completion method that combines BM25-based proxy retrieval with multi-hop path
reasoning. The repository also contains adapted baseline implementations used in
the thesis experiments.

The primary repository goal is to regenerate every reported table and figure
from saved numerical results without rerunning expensive experiments.

## Generate the Saved Outputs

Python 3.12 is the canonical output-generation environment. On a standard Linux
machine:

```bash
./setup_env.sh outputs
source ".venv-$(uname -m)/bin/activate"
python scripts/generate_outputs.py --check
```

`--check` rebuilds every registered table and figure in a temporary directory
and compares it byte-for-byte with the committed outputs. It does not change
files or run experiments.

To update the generated files from the saved metrics:

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

Files in `results/tables/` and `results/figures/` must never be edited by hand.
The generator rejects unregistered files in those directories.

## Prepare the Datasets

The seven canonical datasets are tracked only under `datasets/` in the shared
`head`, `tail`, `relation` format. Validate them without writing files:

```bash
python scripts/prepare_datasets.py --check
```

Baselines that require different names, column orders, inverse edges, or caches
create them under the ignored `outputs/preprocessed/` directory. DacKGR's
working layout can also be prepared and verified directly:

```bash
python scripts/prepare_datasets.py --baseline dackgr --datasets WN18RR
python scripts/prepare_datasets.py --baseline dackgr --datasets WN18RR --check
```

Prepared files are never treated as independent dataset sources. Existing
prepared files are preserved when they differ; use `--force` only after
inspecting the reported mismatch. Working data is generated only for datasets
named in the command that is being run. It consists of ordinary files rather
than links, is ignored by Git, and can be deleted and regenerated from
`datasets/`.

## Run Experiments Locally

Experiment execution is optional and separate from output generation:

```bash
./setup_env.sh experiments
source ".venv-$(uname -m)/bin/activate"
python run_baseline.py pathbsr --datasets NELL23K --dry_run
python run_baseline.py pathbsr --datasets NELL23K
```

Omit `--datasets` to run a baseline over the six datasets reported in the
thesis. The full `FB15K-237` dataset is retained for structural analysis and
can be requested explicitly when a baseline supports it:

```bash
python run_baseline.py hogrn --dry_run
python run_baseline.py dackgr --datasets FB15K-237
```

The same entry point supports:

```text
traditional  hogrn  dackgr  probcbr  anyburl  struprokgr  logre  pathbsr
```

Converted datasets, raw runs, logs, checkpoints, and temporary predictions are
written under ignored working directories such as `outputs/`. Completed metrics
selected for the thesis are saved separately under `results/metrics/`.

PyTorch and CUDA wheels are platform dependent. `requirements.txt` records the
experiment environment used by this project, but a CUDA build compatible with
the local driver may need to be installed from the official PyTorch index.
AnyBURL additionally requires Java and the external `AnyBURL-23-1x.jar`; set
`ANYBURL_JAVA` and `ANYBURL_JAR` when they are not available at the default
locations.

## Run with Slurm

Slurm is an optional scheduling layer around the same Python entry point:

```bash
sbatch scripts/slurm/exp_pathbsr.sh
sbatch --export=ALL,DATASETS="WD-singer NELL23K" scripts/slurm/exp_hogrn.sh
```

The wrappers contain no personal account, node, or repository path. Site-specific
options should be supplied to `sbatch`, for example `--account`, `--partition`,
or `--nodelist`. Set `SPARSEKGC_VENV` when the environment is not located at the
default architecture-specific path.

## Evaluation Protocols

| Protocol | Queries | Tie handling | Filtering |
| --- | --- | --- | --- |
| Main | Bidirectional tail and inverse-head queries | Average rank for ties | Full-entity filtered |
| SOTA comparison | Tail queries only | Optimistic first match | Full-entity filtered |

The protocols are intentionally kept separate. A baseline result is never copied
between protocols when its native evaluator uses different query or tie semantics.

## Repository Structure

```text
SparseKGC/
├── baselines/                 # Original and adapted baseline implementations
├── datasets/                  # Tracked canonical splits and input metadata
├── outputs/
│   ├── preprocessed/          # Generated baseline-specific dataset formats
│   └── ...                    # Raw runs, logs, caches, and checkpoints
├── results/                   # Saved metrics and generated research outputs
├── scripts/
│   ├── generate_outputs.py    # Generate or verify every registered output
│   ├── prepare_datasets.py    # Prepare and verify shared dataset splits
│   └── slurm/                 # Optional cluster wrappers
├── run_baseline.py            # Platform-neutral experiment entry point
├── setup_env.sh               # Output or experiment environment setup
├── REPRODUCIBILITY.md
└── THIRD_PARTY.md
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the thesis artifact map and
[THIRD_PARTY.md](THIRD_PARTY.md) for upstream sources, citations, licenses, and
known local modifications.

Project-authored code is MIT licensed. HoGRN and StruProKGR did not publish
software licenses. Their vendored code is retained with exact provenance to
make the experiments self-contained, but is explicitly excluded from the root
MIT grant; see `THIRD_PARTY.md` for the revisions and status.

## Current Scope

The saved main, SOTA, and efficiency results are integrated. Structural,
ablation, and case-study artifacts generated on the separate PathBSR machine
will be added only with their numerical source files and generating code. The
thesis draft itself is intentionally excluded from Git.

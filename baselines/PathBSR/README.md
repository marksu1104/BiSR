# PathBSR

PathBSR is a non-neural, gradient-free, and interpretable method for sparse
knowledge graph completion. It retrieves structurally similar proxy entities,
applies mined relation-path rules, and reranks candidates with a local structural
verification signal. It learns no embeddings or gradient-trained weights;
relation-path statistics are estimated offline from the training KG.

## Current method

1. **Reverse-Edge Augmentation** adds `(t, r__reverse, h)` for every training
   fact `(h, r, t)`.
2. **Entity Structural Features** use `REL + ENT + REL-ENT` TF profiles.
3. **BM25-based Proxy Retrieval** uses log-qTF, query-wise normalized weights,
   and a fixed top-100 proxy budget.
4. **Path Mining** uses masked deterministic target-closing discovery for body
   lengths 1--3, with explicit per-length/per-case budgets.
5. **Path execution** uses bounded Boolean walk/reachability semantics. Repeated
   entities are allowed; the frontier cap is fixed at 200.
6. **Rule statistics** use masked case-empirical per-fact confidence and
   distinct-head Reliability ordering; each relation's Global Path List retains
   at most 128 paths.
7. **Candidate Scoring** combines Proxy Answers, Path Answers, and Frequency
   Answers with MAX rule pooling and a per-case rule gate.
8. **Path Verification** exactly evaluates the top-100 Candidate-Score entities
   through hop 3 over the undirected training graph.
9. **Evaluation** is bidirectional, filtered, full-entity, and average-tie.

The defaults in this repository define the current PathBSR configuration.
Generated experiment outputs are written under `results/`, which is intentionally
ignored by Git and should be regenerated on each machine.

## Reproduce on a new machine

```bash
cd PathBSR
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .
PYTHONPATH=src .venv/bin/python -m pytest -q
```

If the system Python is externally managed, always call `.venv/bin/python`
directly instead of relying on `python3` after activation.

Dataset files are not versioned in this repository. Place the required
train/valid/test splits under `datasets/<DatasetName>/` before running model or
analysis scripts.

## Run PathBSR

```bash
PYTHONPATH=src .venv/bin/python scripts/run_pathbsr.py \
  --dataset NELL23K \
  --split valid \
  --output results/runs/example_valid.csv
```

Use `--deoverlap-eval` only for the separately labelled WD-singer sensitivity.
Run `PYTHONPATH=src .venv/bin/python scripts/run_pathbsr.py --help` for
experimental flags. The same entry point is also available as
`PYTHONPATH=src .venv/bin/python -m pathbsr.cli`.

To reproduce the current default test table:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_pathbsr.py \
  --dataset FB15K-237-10 \
  --dataset FB15K-237-20 \
  --dataset FB15K-237-50 \
  --dataset NELL23K \
  --dataset WD-singer \
  --dataset WN18RR \
  --output results/runs/pathbsr_best_model_test.csv
```

To regenerate thesis-facing validation ablations, structural analyses, and case
studies from the current codebase, run:

```bash
PYTHONPATH=src .venv/bin/python scripts/structural_analysis.py
PYTHONPATH=src .venv/bin/python scripts/router_analysis.py
PYTHONPATH=src .venv/bin/python scripts/pathbsr_experiments.py
```

Use `--skip-expensive` only when you want to refresh tables from already
computed `results/ablation/pathbsr_validation_ablation_metrics.csv` rows.

`results/` is intentionally ignored by Git and should be regenerated on each
machine. `external_predictions/` is also ignored because it can be very large;
copy it separately only when reproducing structural figures that include
external baseline prediction exports such as AnyBURL, TransE, ConvE, or HoGRN.

## Repository layout

```text
src/pathbsr/            model implementation
scripts/                PathBSR runner and thesis-facing experiment builders
tests/                  focused correctness tests
datasets/               local train/valid/test splits, not versioned
results/                runnable outputs, thesis-ready tables/figures, and cache
```

## Status

Architecture and defaults are selected from validation. Test artifacts are
reported for evaluation only and must not be used for further design changes.
Remaining publication work is external to model selection: shared-evaluator
baseline reruns, statistical reporting, qualitative examples, and paper writing.

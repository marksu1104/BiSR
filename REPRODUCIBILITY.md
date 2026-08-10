# Reproducibility

This document maps saved numerical results to every generated research output.
It follows thesis draft v1 and will be updated when the thesis structure changes.
The thesis PDF is a local reference and is intentionally not part of the
repository.

## Output Contract

Every project-owned non-Markdown result must have:

1. a declared numerical or configuration input;
2. a tracked generating script;
3. a documented command; and
4. deterministic verification where the file format permits it.

Markdown is the only manually maintained document format. Generated Markdown,
such as `results/tables/tables.md`, is still produced by code.

Run the complete verification without changing files:

```bash
python scripts/generate_outputs.py --check
```

The command validates the metric schemas and Dataset/Model keys, rejects
unregistered table or figure files, rebuilds all outputs in a temporary
directory, and performs a byte comparison with the committed files.

## Saved Metrics

The files in `results/metrics/` are the final numerical records selected for the
thesis. They are inputs to table and figure generation and outputs of the
experiment runners listed below.

| Saved metrics | Producer |
| --- | --- |
| `traditional_metrics.csv`, `traditional_sota_metrics.csv` | `run_baseline.py traditional` and `baselines/traditional/run_all.py` |
| `hogrn_metrics.csv` | `run_baseline.py hogrn` and `baselines/HoGRN/run_hogrn_all.py` |
| `dackgr_metrics.csv` | `run_baseline.py dackgr` and the DacKGR experiment scripts |
| `probcbr_metrics.csv` | `run_baseline.py probcbr` and `baselines/Prob-CBR/run_all.py` |
| `anyburl_metrics.csv` | `run_baseline.py anyburl` and `baselines/AnyBURL/run_anyburl.py` |
| `logre_metrics.csv`, `logre_sota_metrics.csv` | `run_baseline.py logre` and `baselines/LoGRe/run_logre.py` |
| `struprokgr_metrics.csv`, `struprokgr_sota_metrics.csv` | `run_baseline.py struprokgr` and `baselines/StruProKGR/run_struprokgr.py` |
| `pathbsr_metrics.csv`, `pathbsr_sota_metrics.csv` | `run_baseline.py pathbsr` and `baselines/PathBSR/run_pathbsr.py` |

The saved files preserve the completed experiment values. Generating the paper
outputs does not invoke any producer in this table.

The saved DacKGR WN18RR metric is a project adaptation. Its input uses the
preserved legacy uniform pruning scores in
`datasets/WN18RR/metadata/dackgr_pagerank.txt`; the original DacKGR paper did
not report WN18RR. Replacing that input would define a new experiment and must
not overwrite the saved metric.

## Dataset Contract

All canonical train, validation, and test inputs are tracked under `datasets/`.
Baseline-local dataset directories are not canonical sources. Required column
reordering, inverse triples, indexes, and caches are generated under
`outputs/preprocessed/<baseline>/`.

```bash
python scripts/prepare_datasets.py --check
python scripts/prepare_datasets.py --baseline dackgr --datasets WN18RR --check
```

The first command validates the canonical triple and metadata schemas. The
second also requires every generated DacKGR input to match its canonical source
byte-for-byte.

## Generated Outputs

| Output | Inputs | Generator |
| --- | --- | --- |
| `results/tables/main_table.csv` | Main-protocol saved metrics | `scripts/build_result_tables.py` |
| `results/tables/sota_table.csv` | SOTA-protocol saved metrics | `scripts/build_result_tables.py` |
| `results/tables/efficiency_table.csv` | Runtime columns in saved metrics | `scripts/build_result_tables.py` |
| `results/tables/tables.md` | Generated result tables | `scripts/build_result_tables.py` |
| `results/tables/tables.tex` | Generated result tables | `scripts/build_result_tables.py` |
| `results/figures/mrr_lines*.png` | Main and SOTA result tables | `scripts/plot_mrr_lines.py` |
| `results/figures/efficiency_scatter.png` | Main and efficiency result tables | `scripts/plot_efficiency.py` |

`scripts/generate_outputs.py` is the public entry point and calls these focused
generators in dependency order.

## Thesis Artifact Map

Status values are intentionally strict:

- **Integrated** means the numerical inputs, generator, and generated output are
  present and checked.
- **Waiting for source results** means relevant code exists, but the final CSV
  files from the separate PathBSR machine are not yet in this repository.
- **Pending generator** means the draft contains the artifact but a deterministic
  project generator has not been completed.

| Thesis item | Description | Code or intended source | Status |
| --- | --- | --- | --- |
| Table 1 | Representative sparse KGC methods | Structured method metadata | Pending generator |
| Table 2 | FB15K-237 path-count distribution | `baselines/PathBSR/scripts/structural_analysis.py` | Waiting for source results |
| Table 3 | FB15K-237 relation-cardinality distribution | `baselines/PathBSR/scripts/structural_analysis.py` | Waiting for source results |
| Table 4 | Dataset statistics | `baselines/PathBSR/scripts/pathbsr_experiments.py` | Waiting for source results |
| Table 5 | Evaluation protocols | Evaluation configuration metadata | Pending generator |
| Table 6 | PathBSR settings | `pathbsr.DEFAULT_CONFIG` | Pending generator |
| Table 7 | Main results | `results/tables/main_table.csv` | Integrated |
| Table 8 | SOTA results | `results/tables/sota_table.csv` | Integrated |
| Table 9 | Core ablation | `baselines/PathBSR/scripts/pathbsr_experiments.py` | Waiting for source results |
| Table 10 | Proxy-similarity ablation | `baselines/PathBSR/scripts/pathbsr_experiments.py` | Waiting for source results |
| Table 11 | Feature variants | `baselines/PathBSR/scripts/pathbsr_experiments.py` | Waiting for source results |
| Table 12 | Case-study summary | `baselines/PathBSR/scripts/pathbsr_experiments.py` | Waiting for source results |
| Figure 1 | FB15K-237 performance by path-count bucket | `baselines/PathBSR/scripts/structural_analysis.py` | Waiting for source results |
| Figure 2 | FB15K-237 performance by relation cardinality | `baselines/PathBSR/scripts/structural_analysis.py` | Waiting for source results |
| Figure 3 | PathBSR framework | Deterministic PNG generator | Pending generator |
| Figure 4 | Few-path performance on NELL23K and WD-singer | `baselines/PathBSR/scripts/pathbsr_experiments.py` | Waiting for source results |
| Figure 5 | Relation-cardinality performance on NELL23K and WD-singer | `baselines/PathBSR/scripts/pathbsr_experiments.py` | Waiting for source results |
| Figure 6 | Runtime and MRR comparison | `scripts/plot_efficiency.py` | Integrated |

The missing PathBSR artifacts will not be reconstructed by reading values from
the thesis PDF. When they are transferred from the other machine, the numerical
source files and their generation path must be added together.

The AI-generated draft of Figure 3 is a visual reference only. The committed
version will be recreated as a deterministic vector diagram generated by code.

## Platforms

Output generation is CPU-only and supports standard Linux on x86_64 and
aarch64. Experiment execution uses the platform-neutral `run_baseline.py` entry
point. Files under `scripts/slurm/` add scheduling only and do not contain a
second implementation of an experiment.

Wall-clock times in the saved efficiency metrics were measured on aarch64 GH200
nodes. Rebuilding the figure preserves those values; running the experiments on
other hardware is expected to produce different timing measurements.

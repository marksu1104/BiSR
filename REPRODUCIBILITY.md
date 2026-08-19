# Reproducibility

This document maps saved numerical results to generated research outputs and the
current thesis draft. The thesis PDF is a local reference and is not tracked.

## Output Contract

Every project-owned non-Markdown result must have:

1. a declared numerical or configuration input;
2. a tracked generating script;
3. a documented command; and
4. deterministic verification where the file format permits it.

Markdown is the only manually maintained document format. Generated Markdown,
such as `results/tables/tables.md`, is still produced by code.

Verify the complete output contract without changing files:

```bash
python scripts/generate_outputs.py --check
```

This validates metric schemas and keys, rejects unregistered outputs, and
rebuilds every output in a temporary directory for byte comparison.

## Saved Metrics

Files in `results/metrics/` are the final numerical records selected for the
thesis and the inputs to table and figure generation.

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

Output generation never invokes these producers. The saved DacKGR WN18RR metric
is a project adaptation using the preserved legacy uniform pruning scores in
`datasets/WN18RR/metadata/dackgr_pagerank.txt`; changing that input defines a
new experiment and must not overwrite the saved value.

## Dataset Contract

Canonical splits are tracked under `datasets/`. Baseline-specific formats and
caches are generated under `outputs/preprocessed/<baseline>/`.

```bash
python scripts/prepare_datasets.py --check
python scripts/prepare_datasets.py --baseline dackgr --datasets WN18RR --check
```

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

**Integrated** marks generated experimental artifacts covered by the output
contract. Descriptive tables and diagrams are documented separately because
they are not numerical experiment outputs.

| Thesis item | Description | Code or intended source | Status |
| --- | --- | --- | --- |
| Table 1 | Representative sparse KGC methods | Thesis literature review and cited sources | Documented |
| Table 2 | FB15K-237 path-count distribution | `baselines/PathBSR/scripts/structural_analysis.py` | Waiting for source results |
| Table 3 | FB15K-237 relation-cardinality distribution | `baselines/PathBSR/scripts/structural_analysis.py` | Waiting for source results |
| Table 4 | Dataset statistics | `baselines/PathBSR/scripts/pathbsr_experiments.py` | Waiting for source results |
| Table 5 | Evaluation protocols | Evaluation modes implemented by the baseline runners | Implemented |
| Table 6 | PathBSR settings | `pathbsr.DEFAULT_CONFIG` | Implemented |
| Table 7 | Main results | `results/tables/main_table.csv` | Integrated |
| Table 8 | SOTA results | `results/tables/sota_table.csv` | Integrated |
| Table 9 | Core ablation | `baselines/PathBSR/scripts/pathbsr_experiments.py` | Waiting for source results |
| Table 10 | Proxy-similarity ablation | `baselines/PathBSR/scripts/pathbsr_experiments.py` | Waiting for source results |
| Table 11 | Feature variants | `baselines/PathBSR/scripts/pathbsr_experiments.py` | Waiting for source results |
| Table 12 | Case-study summary | `baselines/PathBSR/scripts/pathbsr_experiments.py` | Waiting for source results |
| Figure 1 | FB15K-237 performance by path-count bucket | `baselines/PathBSR/scripts/structural_analysis.py` | Waiting for source results |
| Figure 2 | FB15K-237 performance by relation cardinality | `baselines/PathBSR/scripts/structural_analysis.py` | Waiting for source results |
| Figure 3 | PathBSR framework | Thesis conceptual diagram | Thesis-only |
| Figure 4 | Few-path performance on NELL23K and WD-singer | `baselines/PathBSR/scripts/pathbsr_experiments.py` | Waiting for source results |
| Figure 5 | Relation-cardinality performance on NELL23K and WD-singer | `baselines/PathBSR/scripts/pathbsr_experiments.py` | Waiting for source results |
| Figure 6 | Runtime and MRR comparison | `scripts/plot_efficiency.py` | Integrated |

Missing experimental artifacts must be transferred with their numerical
sources and generators, not reconstructed from the thesis PDF. Figure 3 is a
conceptual thesis diagram and is not a project-owned experimental output.

## Platforms

Output generation is CPU-only on Linux x86_64 and aarch64. Experiments use
`run_baseline.py`; Slurm files add scheduling only. GitHub Actions validates the
datasets and outputs and dry-runs each baseline on Ubuntu x86_64 without
launching experiments.

Wall-clock times in the saved efficiency metrics were measured on aarch64 GH200
nodes. Rebuilding the figure preserves those values; running the experiments on
other hardware is expected to produce different timing measurements.

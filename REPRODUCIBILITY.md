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
| `hogrn_metrics.csv`, `hogrn_sota_metrics.csv` | `run_baseline.py hogrn [--hogrn_restore]` and `baselines/HoGRN/run_hogrn_all.py` |
| `dackgr_metrics.csv`, `dackgr_sota_metrics.csv` | `run_baseline.py dackgr [--dackgr_stages ... inference]` and the DacKGR experiment scripts |
| `probcbr_metrics.csv`, `probcbr_sota_metrics.csv` | `run_baseline.py probcbr` and `baselines/Prob-CBR/run_all.py` |
| `anyburl_metrics.csv`, `anyburl_sota_metrics.csv` | `run_baseline.py anyburl` and `baselines/AnyBURL/run_anyburl.py` |
| `logre_metrics.csv`, `logre_sota_metrics.csv` | `run_baseline.py logre` and `baselines/LoGRe/run_logre.py` (scoring shared with StruProKGR) |
| `struprokgr_metrics.csv`, `struprokgr_sota_metrics.csv` | `run_baseline.py struprokgr` and `baselines/StruProKGR/run_struprokgr.py` |
| `pathbsr_metrics.csv`, `pathbsr_sota_metrics.csv` | `run_baseline.py pathbsr` and `baselines/PathBSR/run_pathbsr.py` |

Output generation never invokes these producers. The saved DacKGR WN18RR metric
is a project adaptation using the preserved legacy uniform pruning scores in
`datasets/WN18RR/metadata/dackgr_pagerank.txt`; changing that input defines a
new experiment and must not overwrite the saved value.

### Main/SOTA evaluation semantics

Both protocols are filtered and rank over the **full entity set**. Main is
bidirectional (tail queries plus inverse-relation head queries) with
average-tie ranking; SOTA is tail-only with optimistic-tie ranking -- they
differ in both prediction direction and tie handling, not tie handling alone.
`scripts/ranking_metrics.py` is the shared, unit-tested (`scripts/tests/`)
definition of both tie rules (`average_tie_rank`, `optimistic_tie_rank`,
`rank_from_counts`) and of the sparse-candidate case (`sparse_filtered_rank`),
which ranks a target against the complete entity universe even when a method
only returns scores for a subset of it. Every baseline's Main and SOTA numbers
now come from this shared logic, or from an equivalent full-entity formula
verified against it:

- **Embedding baselines** (TransE/DistMult/ComplEx/ConvE/RotatE/TuckER),
  **HoGRN**: score every entity densely each query, so full-entity ranking is
  immediate; for a fixed query direction, Main and SOTA are computed from the
  same filtered score tensor in one pass (`_run_eval_pass` / `predict()`),
  differing only in tie rule. Overall, Main (bidirectional) also averages
  this over the tail and inverse-relation head directions, while SOTA
  (tail-only) does not -- so the two protocols differ in direction as well.
- **DacKGR**: `hits_and_ranks_full_entity` (`average`/`optimistic`) ranks over
  the complete dense score tensor returned by the model. The legacy
  `hits_and_ranks` helper truncates ranking to `args.beam_size` (128, on every
  dataset in this repository, all far larger than 128 entities) and is no
  longer used to produce reported Main or SOTA metrics.
- **Prob-CBR, AnyBURL, LoGRe, StruProKGR** return sparse per-query candidate
  scores. An entity absent from a method's output is ranked at a
  `default_score` of exactly 0 -- not as a convenience placeholder, but
  because each method's own candidate score is a sum/product of non-negative
  terms (a precision/hit-rate in `[0, 1]` times a non-negative prior or decay
  weight), so "not returned" and "returned with score 0" are the same true
  value under that method's own scoring semantics. This replaces two prior
  defects: Prob-CBR's, LoGRe's, and StruProKGR's earlier SOTA numbers ranked
  the gold answer only within the method's own returned candidate list
  (`get_hits`/`get_rank_in_list`), never against the full entity set; LoGRe and
  StruProKGR are re-scored for both protocols directly from the saved
  `dump_forward.tsv` candidate dump (`score_struprokgr.py`, shared by both),
  so no baseline needs to be re-run to fix this.
- **PathBSR** already scores the full entity set as a dense, zero-initialized
  vector (`np.zeros(len(all_entities))`) per query, so an unscored candidate's
  score is genuinely 0 by construction; its Main/SOTA ranks
  (`pathbsr/evaluation.py:filtered_rank`) required no change.

Standalone DacKGR checkpoint inference (`--dackgr_stages ... inference`, no
`--train`) uses its fixed default seed (`543`) so the stochastic path-selection
strategy does not inherit random state from a preceding training process. This
means a standalone inference run and the inference at the end of a full
training run are each individually deterministic, but are not expected to
reproduce each other bit-for-bit, since the latter's random state has been
advanced by the training steps that preceded it; the resulting metric
differences between the two are small (observed within roughly 1% relative on
MRR) and are a seed/RNG-state effect, not an evaluator difference -- confirmed
by comparing both protocols computed from the *same* forward-pass score tensor
within a single run. HoGRN's `--hogrn_restore` flag is the analogous
restore-and-re-evaluate-only path for HoGRN (loads
`checkpoints/{dataset}_{score_func}_best`, skips training).

### Prob-CBR / LoGRe preprocessing determinism

Prob-CBR and LoGRe each build a per-dataset cache (subgraph paths, a
hierarchical-clustering assignment, a path-prior map, and a precision map)
under `linkage=<value>/` the first time they run, then reuse it on later
runs. Both `execute_one_program`'s branch-truncation sampling and the
subgraph random walk previously drew from the global `np.random` stream
seeded once at process start; whether the cache had to be rebuilt (which
itself calls `execute_one_program` while computing precision) or was already
present changed how many draws happened before real inference, so the same
`--seed` could produce different predictions depending on cache state alone.
Fixed by giving cache construction and real inference each their own
`np.random.default_rng(seed)` instance, threaded explicitly through
`execute_one_program`/`get_paths` in both `prob_cbr/pr_cbr.py` and
`LoGRe.py`, so neither touches global state or the other's stream. Verified
by rebuilding a deleted cache from scratch and confirming the resulting
predictions match a run that reused the pre-existing cache, bit-for-bit,
under the same seed.

`get_unique_entities` returns a Python `set`, whose iteration order for
string elements depends on the interpreter's per-process hash seed
(`PYTHONHASHSEED`, unset and therefore randomized here); this let two
*separate* cold rebuilds with the same `--seed` visit entities in a
different order and thus sample different subgraphs. Fixed by sorting the
entity list before the subgraph-sampling loop in both baselines. A smaller
residual gap remains between independent cold rebuilds specifically (not
between cold and warm use of one cache, which is exact): the sampled paths
around each entity are also stored in a `set`, so a rare score tie at the
`max_num_programs` ranking cutoff can still resolve differently across
separate rebuilds. This does not affect the metrics reported here, since
those are computed by reusing one already-built, committed cache rather than
rebuilding it repeatedly; closing it fully would mean changing the tie-break
rule inside path ranking itself, which is out of scope for a preprocessing
determinism fix.

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

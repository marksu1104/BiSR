# Canonical Datasets

This directory is the only source of dataset inputs in the repository. Each
dataset uses three tab-separated files with the column order
`head`, `tail`, `relation`:

```text
datasets/<dataset>/
├── train.txt
├── valid.txt
├── test.txt
└── metadata/
    ├── dackgr_pagerank.txt
    └── entity_types.txt       # only where required
```

Baseline-specific file names, column orders, inverse edges, indexes, and caches
are generated as ordinary files under the ignored `outputs/preprocessed/`
directory. They are created only for datasets requested by a baseline run.
They are disposable working data, not independent dataset sources, and must
not be edited as canonical inputs.

Validate every canonical file without writing anything:

```bash
python scripts/prepare_datasets.py --check
```

Prepare and verify DacKGR's required working layout:

```bash
python scripts/prepare_datasets.py --baseline dackgr --datasets WN18RR
python scripts/prepare_datasets.py --baseline dackgr --datasets WN18RR --check
```

## Preserved DacKGR Pruning Scores

The sparse FB15K-237 variants, NELL23K, and WD-singer PageRank files are exact
copies of the DacKGR release inputs. The WN18RR and full FB15K-237 files are
legacy project inputs in which every entity has the same score. They are kept
byte-for-byte because replacing them would change action pruning and could
change completed experiment results.

The saved WN18RR DacKGR result is therefore a local adaptation, not a result
reported by the original DacKGR paper. Re-evaluating it with a different
PageRank input is a new experiment and must not overwrite the saved metric.

Dataset origins, licenses, and citations are documented in
[`THIRD_PARTY.md`](../THIRD_PARTY.md).

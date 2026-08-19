# Third-Party Code, Data, and Citations

The root MIT license covers project-authored work only. Third-party code, data,
and artifacts retain their own terms; citation does not itself grant permission
to redistribute source code.

## Baseline Implementations

| Component | Recorded upstream revision | Primary reference | License status |
| --- | --- | --- | --- |
| DacKGR | [`2a0fe7c8140a31bd2467aa442128df480aae9bb9`](https://github.com/THU-KEG/DacKGR/commit/2a0fe7c8140a31bd2467aa442128df480aae9bb9) | [Lv et al., EMNLP 2020](https://aclanthology.org/2020.emnlp-main.459/) | MIT; retained in `baselines/DacKGR/LICENSE` |
| Prob-CBR | [`8bc8678c33baeb22a0ba56c59cfe5becc0a0cc81`](https://github.com/ameyagodbole/Prob-CBR/commit/8bc8678c33baeb22a0ba56c59cfe5becc0a0cc81) | [Godbole et al., Findings of EMNLP 2020](https://aclanthology.org/2020.findings-emnlp.427/) | MIT; retained in `baselines/Prob-CBR/LICENSE` |
| AnyBURL | Release 23-1x from the [official distribution](https://web.informatik.uni-mannheim.de/AnyBURL/) | [Meilicke et al., 2020](https://arxiv.org/abs/2004.04412) | BSD-3-Clause; JAR and license retained in `baselines/AnyBURL/` |
| HoGRN | [`7c85e84302a5e2bdf571bf12d738abae258e0e08`](https://github.com/TachiChan/HoGRN/commit/7c85e84302a5e2bdf571bf12d738abae258e0e08) | [Chen et al., TKDE](https://arxiv.org/abs/2207.07503) | No license was published; vendored for experiment reproducibility and excluded from the root license |
| LoGRe | [`6721817181a6f368e1eb792f9080f918bdd31a52`](https://github.com/gsp2014/LoGRe/commit/6721817181a6f368e1eb792f9080f918bdd31a52) | [Cheng et al., CIKM 2024](https://doi.org/10.1145/3627673.3679845) | Apache-2.0; retained in `baselines/LoGRe/LICENSE` |
| StruProKGR | [`76b9dbd8ff76fd5baa4b3c6badebaca42149bd9c`](https://github.com/YucanGuo/StruProKGR/commit/76b9dbd8ff76fd5baa4b3c6badebaca42149bd9c) | [Guo et al., 2025](https://arxiv.org/abs/2512.12613) | No license was published; vendored for experiment reproducibility and excluded from the root license |
| Traditional KGE models | Project-authored integration | Model papers listed below | Covered by the root MIT license |
| PathBSR | Original project implementation based methodologically on Prob-CBR | Thesis reference pending | Covered by the root MIT license |

As of 2026-08-11, neither HoGRN nor StruProKGR published a license in repository
metadata, history, documentation, or source headers. Their vendored files are
retained for experiment reproducibility, excluded from the root MIT license,
and do not make the complete repository license-clean.

## Nested DacKGR Attribution

DacKGR includes Salesforce MultiHopKG source recorded at
[`bed0fdd9de3365a6bc04645ca5b7e09f8a98d480`](https://github.com/salesforce/MultiHopKG/commit/bed0fdd9de3365a6bc04645ca5b7e09f8a98d480).
Its headers and BSD-3-Clause license are retained alongside DacKGR's MIT license.

## Local Adaptations

Changes from the recorded revisions include portable paths, shared local and
Slurm entry points, dataset conversion, deterministic seeds, metric exports,
protocol support, failure isolation, sparse-relation fixes, and aarch64 support.
LoGRe changes retain the notices required by Apache-2.0. Unrelated notebooks
with machine-specific paths and duplicate documentation PDFs are omitted.

`baselines/traditional/` is a project implementation of TransE, DistMult,
ComplEx, ConvE, RotatE, and TuckER rather than a copy of one upstream project.

## Canonical Dataset Provenance

Canonical inputs live under `datasets/`; generated baseline formats and caches
live under the ignored `outputs/preprocessed/` directory.

| Dataset | Provenance and terms |
| --- | --- |
| `FB15K-237-10`, `FB15K-237-20`, `FB15K-237-50` | Exact split files from the DacKGR release, constructed from FB15K-237 as described by [Lv et al.](https://aclanthology.org/2020.emnlp-main.459/); the underlying Freebase dumps are distributed under CC-BY |
| `NELL23K` | Exact split files from the DacKGR release, constructed from [NELL](https://rtw.ml.cmu.edu/rtw/resources); the NELL download page does not state an explicit license for the complete knowledge base |
| `WD-singer` | Exact split files from the DacKGR release, constructed from Wikidata; Wikidata structured data is CC0 |
| `FB15K-237` | Exact ConvE release split content after conversion from `head, relation, tail` to the repository's `head, tail, relation` order; underlying Freebase terms apply |
| `WN18RR` | Exact [ConvE release](https://github.com/TimDettmers/ConvE) split content after the same column-order conversion; the Princeton WordNet license must be retained for redistribution |

Official source terms:

- [Freebase data dump license](https://developers.google.com/freebase)
- [Wikidata licensing](https://www.wikidata.org/wiki/Wikidata:Licensing)
- [Princeton WordNet license](https://wordnet.princeton.edu/license-and-commercial-use)

The final `metadata/entity_types.txt` mappings come from LoGRe type data,
derivable NELL names, or Wikidata facts. DacKGR PageRank metadata matches its
release for the five original sparse datasets; WN18RR and full FB15K-237 retain
legacy uniform scores. See `datasets/README.md` for compatibility details.

## Traditional Model Citations

- TransE: Bordes et al., [Translating Embeddings for Modeling Multi-relational Data](https://proceedings.neurips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html).
- DistMult: Yang et al., [Embedding Entities and Relations for Learning and Inference in Knowledge Bases](https://arxiv.org/abs/1412.6575).
- ComplEx: Trouillon et al., [Complex Embeddings for Simple Link Prediction](https://proceedings.mlr.press/v48/trouillon16.html).
- ConvE: Dettmers et al., [Convolutional 2D Knowledge Graph Embeddings](https://arxiv.org/abs/1707.01476).
- RotatE: Sun et al., [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](https://arxiv.org/abs/1902.10197).
- TuckER: Balažević et al., [TuckER: Tensor Factorization for Knowledge Graph Completion](https://arxiv.org/abs/1901.09590).

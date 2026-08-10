# Third-Party Code, Data, and Citations

The root MIT license applies only to project-authored code and documentation.
It does not relicense third-party code, datasets, or model artifacts. Each
component below remains subject to its own terms.

Citation and permission are separate requirements. A paper citation gives
academic credit; it does not by itself grant permission to modify or
redistribute source code.

## Baseline Implementations

| Component | Recorded upstream revision | Primary reference | License status |
| --- | --- | --- | --- |
| DacKGR | [`2a0fe7c8140a31bd2467aa442128df480aae9bb9`](https://github.com/THU-KEG/DacKGR/commit/2a0fe7c8140a31bd2467aa442128df480aae9bb9) | [Lv et al., EMNLP 2020](https://aclanthology.org/2020.emnlp-main.459/) | MIT; retained in `baselines/DacKGR/LICENSE` |
| Prob-CBR | [`8bc8678c33baeb22a0ba56c59cfe5becc0a0cc81`](https://github.com/ameyagodbole/Prob-CBR/commit/8bc8678c33baeb22a0ba56c59cfe5becc0a0cc81) | [Godbole et al., Findings of EMNLP 2020](https://aclanthology.org/2020.findings-emnlp.427/) | MIT; retained in `baselines/Prob-CBR/LICENSE` |
| AnyBURL | Release 23-1x from the [official distribution](https://web.informatik.uni-mannheim.de/AnyBURL/) | [Meilicke et al., 2020](https://arxiv.org/abs/2004.04412) | BSD-3-Clause; the external JAR is not tracked |
| HoGRN | [`7c85e84302a5e2bdf571bf12d738abae258e0e08`](https://github.com/TachiChan/HoGRN/commit/7c85e84302a5e2bdf571bf12d738abae258e0e08) | [Chen et al., TKDE](https://arxiv.org/abs/2207.07503) | No license was published; vendored for experiment reproducibility and excluded from the root license |
| LoGRe | [`6721817181a6f368e1eb792f9080f918bdd31a52`](https://github.com/gsp2014/LoGRe/commit/6721817181a6f368e1eb792f9080f918bdd31a52) | [Cheng et al., CIKM 2024](https://doi.org/10.1145/3627673.3679845) | Apache-2.0; retained in `baselines/LoGRe/LICENSE` |
| StruProKGR | [`76b9dbd8ff76fd5baa4b3c6badebaca42149bd9c`](https://github.com/YucanGuo/StruProKGR/commit/76b9dbd8ff76fd5baa4b3c6badebaca42149bd9c) | [Guo et al., 2025](https://arxiv.org/abs/2512.12613) | No license was published; vendored for experiment reproducibility and excluded from the root license |
| Traditional KGE models | Project-authored integration | Model papers listed below | Covered by the root MIT license |
| PathBSR | Original project implementation based methodologically on Prob-CBR | Thesis reference pending | Covered by the root MIT license |

The HoGRN and StruProKGR checks covered GitHub's repository license metadata,
the only published branch, all commits in their current histories, README
files, and source headers. As of 2026-08-11, neither repository contained a
license statement. Their vendored files are retained so the recorded
experiments can be rerun without reconstructing modified upstream trees, and
are explicitly excluded from the root MIT license. This provenance record does
not grant additional rights to reuse either implementation, and the complete
repository is not presented as license-clean.

## Nested DacKGR Attribution

DacKGR contains source derived from Salesforce MultiHopKG, recorded at
[`bed0fdd9de3365a6bc04645ca5b7e09f8a98d480`](https://github.com/salesforce/MultiHopKG/commit/bed0fdd9de3365a6bc04645ca5b7e09f8a98d480).
Those files retain their Salesforce copyright and SPDX headers. The complete
BSD-3-Clause text is included as
`baselines/DacKGR/LICENSE.salesforce-BSD-3-Clause` in addition to DacKGR's MIT
license.

## Local Adaptations

The vendored implementations were compared against the revisions above.
Project changes include portable paths, unified local and Slurm entry points,
dataset conversion, deterministic seeds, metric exports, evaluation protocol
support, failure isolation, sparse-relation fixes, and aarch64 compatibility.

LoGRe's modified upstream files carry prominent change notices as required by
Apache-2.0. Project-authored wrappers and preparation scripts are separate from
the upstream algorithm files.

The Traditional directory is a project-owned implementation of TransE,
DistMult, ComplEx, ConvE, RotatE, and TuckER. It follows evaluation and training
patterns used in this repository, but it is not presented as a verbatim copy of
one upstream implementation.

## Canonical Dataset Provenance

Canonical inputs are stored only under `datasets/`. Baseline-specific formats
and caches are generated on demand as ordinary files under the ignored
`outputs/preprocessed/` directory.

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

The `metadata/entity_types.txt` files are the final auxiliary mappings used by
LoGRe and StruProKGR. The FB15K mappings are based on LoGRe's Apache-2.0 type
data, the NELL mapping is derivable from entity names, and the WD-singer mapping
contains Wikidata-derived type facts. These mappings are preserved as inputs;
they are not presented as newly measured outputs.

The DacKGR PageRank metadata for the five original sparse datasets is an exact
copy of the DacKGR release. WN18RR and full FB15K-237 use preserved legacy
uniform scores. See `datasets/README.md` for the compatibility rationale.

## Traditional Model Citations

- TransE: Bordes et al., [Translating Embeddings for Modeling Multi-relational Data](https://proceedings.neurips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html).
- DistMult: Yang et al., [Embedding Entities and Relations for Learning and Inference in Knowledge Bases](https://arxiv.org/abs/1412.6575).
- ComplEx: Trouillon et al., [Complex Embeddings for Simple Link Prediction](https://proceedings.mlr.press/v48/trouillon16.html).
- ConvE: Dettmers et al., [Convolutional 2D Knowledge Graph Embeddings](https://arxiv.org/abs/1707.01476).
- RotatE: Sun et al., [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](https://arxiv.org/abs/1902.10197).
- TuckER: Balažević et al., [TuckER: Tensor Factorization for Knowledge Graph Completion](https://arxiv.org/abs/1901.09590).

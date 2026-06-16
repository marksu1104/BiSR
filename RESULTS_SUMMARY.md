# SparseKGC Experiment Results Summary

## Setup
- **Datasets (6)**: FB15K-237-10, FB15K-237-20, FB15K-237-50, NELL23K, WD-singer, WN18RR (excluding full FB15K-237)
- **Evaluation protocol**: filtered + full-entity ranking + bidirectional evaluation (head prediction via inverse relations) + tie-aware average rank
- **Main metrics**: MRR and Hits@3; all reported values are rounded to three decimal places
- **Methods (10 baselines + PathBSR)**: traditional seed 42 / HoGRN seed 41504 / DacKGR seed 543 / Prob-CBR seed 42

| Family | Methods |
|---|---|
| Classic embedding | TransE, DistMult, ComplEx, ConvE |
| Advanced embedding | RotatE, TuckER |
| Rule mining | AnyBURL |
| Case-based reasoning | Prob-CBR |
| Multi-hop / GNN | DacKGR, HoGRN |
| **Ours** | **PathBSR** (BM25 proxy CBR + multi-hop path mining) |

---

## Table 1: MRR (head+tail average, main protocol)

| Method | FB15K-237-10 | FB15K-237-20 | FB15K-237-50 | NELL23K | WD-singer | WN18RR |
|---|---|---|---|---|---|---|
| TransE | 0.153 | 0.174 | 0.220 | 0.181 | 0.302 | 0.206 |
| DistMult | 0.123 | 0.157 | 0.213 | 0.199 | 0.354 | 0.442 |
| ComplEx | 0.154 | 0.165 | 0.192 | 0.186 | 0.315 | 0.437 |
| RotatE | 0.121 | 0.169 | 0.224 | 0.188 | 0.362 | 0.461 |
| ConvE | 0.157 | 0.169 | 0.203 | 0.207 | 0.360 | 0.423 |
| TuckER | 0.128 | 0.171 | **0.228** | 0.194 | 0.358 | 0.446 |
| AnyBURL | 0.144 | 0.172 | 0.220 | 0.213 | **0.392** | **0.483** |
| Prob-CBR | 0.087 | 0.145 | 0.195 | 0.179 | 0.188 | 0.458 |
| DacKGR | 0.120 | 0.139 | 0.166 | 0.142 | 0.274 | 0.374 |
| HoGRN | **0.170** | 0.179 | 0.221 | **0.242** | 0.387 | 0.480 |
| **PathBSR** | 0.159 | **0.184** | 0.226 | 0.234 | 0.376 | 0.446 |

PathBSR achieves the best MRR on FB15K-237-20. It is close to the strongest baseline on FB15K-237-50 and NELL23K, while AnyBURL remains strongest on WD-singer and WN18RR.

---

## Table 2: Hits@3 (head+tail average, main protocol)

| Method | FB15K-237-10 | FB15K-237-20 | FB15K-237-50 | NELL23K | WD-singer | WN18RR |
|---|---|---|---|---|---|---|
| TransE | 0.163 | 0.185 | 0.240 | 0.207 | 0.397 | 0.334 |
| DistMult | 0.132 | 0.165 | 0.229 | 0.223 | 0.386 | 0.454 |
| ComplEx | 0.162 | 0.175 | 0.206 | 0.203 | 0.346 | 0.446 |
| RotatE | 0.128 | 0.178 | 0.241 | 0.200 | 0.397 | 0.471 |
| ConvE | 0.163 | 0.177 | 0.218 | 0.222 | 0.386 | 0.433 |
| TuckER | 0.133 | 0.180 | 0.244 | 0.210 | 0.390 | 0.460 |
| AnyBURL | 0.153 | 0.183 | 0.236 | 0.228 | 0.423 | **0.496** |
| Prob-CBR | 0.092 | 0.152 | 0.208 | 0.193 | 0.209 | 0.475 |
| DacKGR | 0.130 | 0.151 | 0.180 | 0.153 | 0.302 | 0.392 |
| HoGRN | **0.180** | 0.188 | 0.237 | **0.265** | **0.424** | **0.496** |
| **PathBSR** | 0.170 | **0.198** | **0.247** | 0.256 | 0.404 | 0.469 |

PathBSR achieves the best Hits@3 on FB15K-237-20 and FB15K-237-50. On NELL23K, it is only 0.009 behind HoGRN.

---

## Table 3: Tail-only MRR (comparison with LoGRe / sparse KGC literature)

> LoGRe and related sparse KGC papers report **tail-only** metrics. For an apples-to-apples comparison, PathBSR should be compared under the same tail-only protocol.

| Method | FB15K-237-10 | FB15K-237-20 | FB15K-237-50 | NELL23K | WD-singer | WN18RR |
|---|---|---|---|---|---|---|
| TransE | 0.235 | 0.259 | 0.313 | 0.224 | 0.357 | 0.231 |
| DistMult | 0.192 | 0.239 | 0.305 | 0.233 | 0.410 | 0.464 |
| ComplEx | 0.239 | 0.251 | 0.281 | 0.224 | 0.378 | 0.452 |
| RotatE | 0.186 | 0.251 | 0.317 | 0.223 | 0.418 | 0.481 |
| ConvE | 0.243 | 0.255 | 0.296 | 0.248 | 0.428 | 0.440 |
| TuckER | 0.197 | 0.256 | 0.324 | 0.230 | 0.414 | 0.473 |
| AnyBURL | 0.223 | 0.257 | 0.311 | 0.257 | 0.468 | 0.511 |
| Prob-CBR | 0.135 | 0.219 | 0.284 | 0.212 | 0.195 | 0.482 |
| DacKGR | 0.216 | 0.245 | 0.287 | 0.208 | 0.348 | 0.406 |
| HoGRN | 0.256 | 0.267 | 0.316 | 0.290 | 0.464 | 0.507 |
| **PathBSR** | 0.242 | 0.272 | 0.318 | 0.282 | 0.449 | 0.470 |
| LoGRe (reported) | 0.228 | 0.261 | 0.297 | 0.259 | 0.459 | — |
| **PathBSR - LoGRe** | **+0.014** | **+0.011** | **+0.021** | **+0.023** | -0.010 | — |

**PathBSR wins over LoGRe on 4 out of 5 comparable datasets under tail-only evaluation, and is only 0.010 behind on WD-singer.**

---

## Table 4: Tail-only Hits@3

| Method | FB15K-237-10 | FB15K-237-20 | FB15K-237-50 | NELL23K | WD-singer | WN18RR |
|---|---|---|---|---|---|---|
| TransE | 0.255 | 0.279 | 0.344 | 0.257 | 0.473 | 0.365 |
| DistMult | 0.209 | 0.255 | 0.333 | 0.263 | 0.453 | 0.477 |
| ComplEx | 0.254 | 0.270 | 0.306 | 0.241 | 0.416 | 0.464 |
| RotatE | 0.202 | 0.271 | 0.346 | 0.236 | 0.457 | 0.494 |
| ConvE | 0.257 | 0.271 | 0.320 | 0.266 | 0.458 | 0.455 |
| TuckER | 0.207 | 0.274 | 0.350 | 0.249 | 0.450 | 0.490 |
| AnyBURL | 0.241 | 0.277 | 0.336 | 0.271 | 0.505 | **0.529** |
| Prob-CBR | 0.145 | 0.235 | 0.305 | 0.229 | 0.217 | 0.505 |
| DacKGR | 0.237 | 0.269 | 0.316 | 0.222 | 0.388 | 0.434 |
| HoGRN | **0.275** | 0.286 | 0.343 | **0.318** | **0.508** | 0.526 |
| **PathBSR** | 0.262 | **0.296** | 0.349 | 0.310 | 0.484 | 0.499 |

---

## Table 5: Time Efficiency (seconds)

> Baseline times are taken from the `seconds` column in each metrics CSV. PathBSR time is computed from the provided log as `mine path rules + valid + test`, so it represents end-to-end path mining plus validation/test inference time.

| Method | FB15K-237-10 | FB15K-237-20 | FB15K-237-50 | NELL23K | WD-singer | WN18RR | Avg |
|---|---|---|---|---|---|---|---|
| TransE | 351 | 431 | 954 | 556 | 293 | 974 | 593 |
| DistMult | 373 | 406 | 789 | 700 | 514 | 1775 | 759 |
| ComplEx | 359 | 402 | 460 | 402 | 1021 | 5283 | 1321 |
| RotatE | 804 | 1289 | 3338 | 1676 | 586 | 3656 | 1892 |
| ConvE | 236 | 324 | 537 | 255 | 366 | 2119 | 640 |
| TuckER | 866 | 988 | 2930 | 1821 | 1045 | 5238 | 2148 |
| AnyBURL | 121 | 134 | 157 | 109 | 109 | 117 | **125** |
| Prob-CBR | 664 | 1774 | 7449 | 529 | 14 | 1281 | 1952 |
| DacKGR | 17367 | 33234 | 75408 | 9431 | 6859 | 42580 | 30813 |
| HoGRN | 5983 | 4655 | 25404 | 2367 | 2175 | 6929 | 7919 |
| **PathBSR** | **48** | 142 | 527 | **16** | **11** | 358 | 184 |

### PathBSR Time Breakdown

| Dataset | Mine rules | Valid | Test | Total |
|---|---:|---:|---:|---:|
| FB15K-237-10 | 5s | 27s | 16s | 48s |
| FB15K-237-20 | 29s | 69s | 44s | 142s |
| FB15K-237-50 | 167s | 222s | 138s | 527s |
| NELL23K | 2s | 10s | 4s | 16s |
| WD-singer | 1s | 6s | 4s | 11s |
| WN18RR | 10s | 182s | 166s | 358s |

### Time Efficiency Analysis

1. **AnyBURL is the fastest on average, and PathBSR ranks second**: AnyBURL averages 125s, while PathBSR averages 184s. Both are much faster than training-based embedding and GNN methods.
2. **PathBSR is especially fast on small and medium datasets**: WD-singer takes 11s, NELL23K takes 16s, and FB15K-237-10 takes 48s. These times already include path rule mining, validation, and test inference.
3. **The main PathBSR bottleneck is inference rather than rule mining**: On WN18RR, mining takes only 10s, while validation and test inference take 348s combined. On FB15K-237-50, mining takes 167s, while validation and test inference take 360s combined. Candidate ranking and path lookup therefore dominate runtime.
4. **PathBSR is much more efficient than training-based methods**: On average, PathBSR is about 43.1x faster than HoGRN and about 167.7x faster than DacKGR. It is also roughly an order of magnitude faster than RotatE, TuckER, and Prob-CBR.
5. **PathBSR has a strong performance-efficiency trade-off**: It achieves the best MRR and Hits@3 on FB15K-237-20, the best Hits@3 on FB15K-237-50, and near-best performance on NELL23K, while its average runtime is only about 59s slower than AnyBURL.

---

## Key Takeaways

1. **The evaluation protocol has been cross-validated**: trained baselines align with LoGRe under tail-only evaluation; AnyBURL aligns with its official evaluation (0.391 vs. 0.394); the overall evaluation harness is reliable.
2. **PathBSR is competitive with LoGRe**: under the tail-only apples-to-apples protocol, PathBSR outperforms LoGRe on 4 out of 5 comparable datasets. The earlier impression that PathBSR was consistently worse came from comparing PathBSR's head+tail average against LoGRe's tail-only numbers.
3. **LoGRe mixes evaluation protocols in its table**: its AnyBURL result is bidirectional, while its own method and the other baselines are tail-only. This is not an issue with our harness.
4. **AnyBURL remains the strongest training-free baseline**: however, PathBSR outperforms AnyBURL on the FB/NELL series under tail-only evaluation; AnyBURL mainly leads on WD-singer and WN18RR.
5. **PathBSR's core value is low training cost with strong competitiveness**: under the main protocol, it achieves the best MRR/Hits@3 on FB15K-237-20 and the best Hits@3 on FB15K-237-50, while keeping runtime close to AnyBURL and far below HoGRN/DacKGR.

## Notes for Paper Writing
+- Use **tail-only** numbers when comparing against LoGRe, and optionally include bidirectional results for completeness.
+- LoGRe uses optimistic tie handling, while this evaluation uses tie-aware ranking. The difference is small for methods with few ties, but a fully aligned comparison can be obtained by rerunning tail-only + optimistic ranking.
+- The current metrics CSV files come from different experiment versions. Before publication, rerun all methods once using the finalized codebase and a clean full pipeline.

"""Filtered ranking metrics for bidirectional KGC evaluation."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np


def summarize_rank_list(ranks: Sequence[float]) -> Dict[str, float]:
    ranks_arr = np.array(ranks, dtype=np.float64)
    if len(ranks_arr) == 0:
        return {"mrr": 0.0, "hits@1": 0.0, "hits@3": 0.0, "hits@10": 0.0, "num_queries": 0}
    return {
        "mrr": float(np.mean(1.0 / ranks_arr)),
        "hits@1": float(np.mean(ranks_arr <= 1)),
        "hits@3": float(np.mean(ranks_arr <= 3)),
        "hits@10": float(np.mean(ranks_arr <= 10)),
        "num_queries": int(len(ranks_arr)),
    }


def summarize_metrics(
    ranks: Sequence[float],
    tail_ranks: Optional[Sequence[float]] = None,
    head_ranks: Optional[Sequence[float]] = None,
    tail_opt_ranks: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    metrics = summarize_rank_list(ranks)
    if tail_ranks is not None:
        metrics.update({f"tail_{key}": value for key, value in summarize_rank_list(tail_ranks).items()})
    if head_ranks is not None:
        metrics.update({f"head_{key}": value for key, value in summarize_rank_list(head_ranks).items()})
    # tail-only with optimistic tie handling — the cross-paper comparison protocol
    if tail_opt_ranks is not None:
        metrics.update({f"tailopt_{key}": value for key, value in summarize_rank_list(tail_opt_ranks).items()})
    return metrics


def filtered_rank(
    score_vec: np.ndarray,
    query_key: Tuple[str, str],
    target: str,
    ent2idx: dict[str, int],
    all_true_tails: dict[Tuple[str, str], set[str]],
    tie_mode: str = "aware",
) -> float:
    """Filtered rank of ``target``. tie_mode: "aware" = tie-aware average rank
    (greater + (equal+1)/2, our main protocol); "optimistic" = best-case rank
    (greater + 1, the convention several KGC papers / AnyBURL report)."""
    if tie_mode not in {"aware", "optimistic"}:
        raise ValueError(f"Unsupported tie_mode={tie_mode!r}")
    if score_vec.ndim != 1:
        raise ValueError(f"Expected a 1D score vector, got shape {score_vec.shape}")
    if len(score_vec) != len(ent2idx):
        raise ValueError(
            f"Score vector length {len(score_vec)} does not match entity vocabulary size {len(ent2idx)}"
        )
    if not np.all(np.isfinite(score_vec)):
        raise ValueError(f"Non-finite score detected for query {query_key}")

    target_idx = ent2idx.get(target)
    if target_idx is None:
        raise ValueError(f"Target entity {target!r} for query {query_key} is missing from the ranking vocabulary")

    filtered = score_vec.copy()
    for other in all_true_tails.get(query_key, set()):
        if other == target:
            continue
        idx = ent2idx.get(other)
        if idx is not None:
            filtered[idx] = -np.inf

    target_score = filtered[target_idx]
    if not np.isfinite(target_score):
        raise ValueError(f"Target score became non-finite for query {query_key} and target {target!r}")
    greater = int(np.sum(filtered > target_score))
    if tie_mode == "optimistic":
        return float(greater + 1)
    equal = int(np.sum(filtered == target_score))
    return float(greater + (equal + 1) / 2.0)

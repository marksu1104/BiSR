"""High-level PathBSR model orchestrator."""

from __future__ import annotations

import multiprocessing as _mp
import os
import sys
from itertools import chain
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = lambda x, **kwargs: x

from .config import PathBSRConfig, DEFAULT_CONFIG
from .data import Triplet, augment_with_reverse_edges
from .evaluation import filtered_rank, summarize_metrics
from .graph import GraphStore, build_entity_features
from .paths import RuleMiner
from .ranking import CandidateRanker
from .retrieval import ProxyRetriever, build_proxy_index

# Module-level slot populated by the parent process before forking workers.
# Workers inherit a copy-on-write view of the model — no pickling needed.
_WORKER_MODEL: "PathBSR | None" = None


def _eval_chunk(args: tuple) -> tuple:
    """Evaluate a slice of triples; runs in a forked worker."""
    indices, triples, K = args
    model = _WORKER_MODEL
    tail_ranks_out: List[float] = []
    tail_opt_ranks_out: List[float] = []
    head_ranks_out: List[float] = []
    for head, relation, tail in triples:
        tail_scores = model.score(head, relation, topk=K)
        tail_rank = filtered_rank(
            tail_scores, (head, relation), tail,
            model.graph.ent2idx, model.graph.all_true_tails,
        )
        tail_opt_rank = filtered_rank(
            tail_scores, (head, relation), tail,
            model.graph.ent2idx, model.graph.all_true_tails,
            tie_mode="optimistic",
        )
        reverse_relation = model.reverse_relation(relation)
        head_scores = model.score(tail, reverse_relation, topk=K)
        head_rank = filtered_rank(
            head_scores, (tail, reverse_relation), head,
            model.graph.ent2idx, model.graph.all_true_tails,
        )
        tail_ranks_out.append(tail_rank)
        tail_opt_ranks_out.append(tail_opt_rank)
        head_ranks_out.append(head_rank)
    return indices, tail_ranks_out, tail_opt_ranks_out, head_ranks_out


class PathBSR:
    """Paper-facing PathBSR pipeline.

    The class wires together graph construction, proxy retrieval, rule mining,
    candidate ranking, and bidirectional filtered evaluation.
    """

    def __init__(
        self,
        train_triplets: Sequence[Triplet],
        valid_triplets: Sequence[Triplet],
        test_triplets: Sequence[Triplet],
        config: PathBSRConfig = DEFAULT_CONFIG,
    ) -> None:
        self.config = config
        collisions = sorted(
            {
                relation
                for _, relation, _ in chain(
                    train_triplets, valid_triplets, test_triplets
                )
                if relation.endswith(self.config.reverse_suffix)
            }
        )
        if collisions:
            raise ValueError(
                "Original relation names collide with the reserved reverse suffix "
                f"{self.config.reverse_suffix!r}: {collisions[:5]}"
            )
        train_aug = augment_with_reverse_edges(train_triplets, self.config.reverse_suffix)
        train_entities = {entity for head, _, tail in train_aug for entity in (head, tail)}
        self.graph = GraphStore.build(
            train_triplets,
            valid_triplets,
            test_triplets,
            train_entities=train_entities,
            config=self.config,
        )
        self.entity_features, _ = build_entity_features(train_aug, self.config)
        self.out_index = build_proxy_index(self.entity_features, self.config)
        self.retriever = ProxyRetriever(
            self.graph,
            self.entity_features,
            self.out_index,
            self.config,
        )
        self.rule_miner = RuleMiner(self.graph, self.config)
        self.ranker = CandidateRanker(self.graph, self.retriever, self.rule_miner, self.config)

    @property
    def all_entities(self) -> list[str]:
        return self.graph.all_entities

    @property
    def ent2idx(self) -> dict[str, int]:
        return self.graph.ent2idx

    def reverse_relation(self, relation: str) -> str:
        if relation.endswith(self.config.reverse_suffix):
            return relation[: -len(self.config.reverse_suffix)]
        return f"{relation}{self.config.reverse_suffix}"

    def retrieve_proxies(
        self,
        head: str,
        relation: str,
        topk: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        return self.retriever.retrieve(head, relation, topk)

    def score_direct_cases(self, head: str, relation: str, topk: Optional[int] = None) -> np.ndarray:
        return self.retriever.score_direct_cases(head, relation, topk)

    def score_case_paths(self, head: str, relation: str, topk: Optional[int] = None) -> np.ndarray:
        return self.ranker.score_case_paths(head, relation, topk)

    def score(self, head: str, relation: str, topk: Optional[int] = None) -> np.ndarray:
        return self.ranker.score(head, relation, topk)


    def filtered_rank(self, score_vec: np.ndarray, query_key: Tuple[str, str], target: str) -> float:
        return filtered_rank(score_vec, query_key, target, self.graph.ent2idx, self.graph.all_true_tails)

    def evaluate(
        self,
        eval_triplets: Sequence[Triplet],
        split_name: str = "eval",
        max_examples: Optional[int] = None,
        K: Optional[int] = None,
        progress_every: Optional[int] = None,
        num_workers: Optional[int] = None,
    ) -> Dict[str, float]:
        data = list(eval_triplets) if max_examples is None else list(eval_triplets)[:max_examples]
        if not data:
            raise ValueError(f"{split_name}: no evaluation triplets were provided")

        # Determine effective worker count: default to all available CPUs.
        n_workers = num_workers if num_workers is not None else int(os.environ.get("PATHBSR_NUM_WORKERS", os.cpu_count() or 1))
        n_workers = max(1, min(n_workers, len(data)))

        if n_workers > 1:
            return self._evaluate_parallel(data, split_name, K, progress_every, n_workers)

        ranks: List[float] = []
        tail_ranks: List[float] = []
        head_ranks: List[float] = []
        tail_opt_ranks: List[float] = []

        for index, (head, relation, tail) in enumerate(tqdm(data, desc=split_name, leave=True), start=1):
            tail_scores = self.score(head, relation, topk=K)
            tail_rank = filtered_rank(
                tail_scores,
                (head, relation),
                tail,
                self.graph.ent2idx,
                self.graph.all_true_tails,
            )
            ranks.append(tail_rank)
            tail_ranks.append(tail_rank)
            tail_opt_ranks.append(
                filtered_rank(
                    tail_scores,
                    (head, relation),
                    tail,
                    self.graph.ent2idx,
                    self.graph.all_true_tails,
                    tie_mode="optimistic",
                )
            )

            reverse_relation = self.reverse_relation(relation)
            head_scores = self.score(tail, reverse_relation, topk=K)
            head_rank = filtered_rank(
                head_scores,
                (tail, reverse_relation),
                head,
                self.graph.ent2idx,
                self.graph.all_true_tails,
            )
            ranks.append(head_rank)
            head_ranks.append(head_rank)
            if progress_every is not None and progress_every > 0:
                if index == len(data) or index % progress_every == 0:
                    print(
                        f"[PathBSR] {split_name}: evaluated {index:,}/{len(data):,} triples "
                        f"({2 * index:,}/{2 * len(data):,} queries)",
                        file=sys.stderr,
                        flush=True,
                    )

        expected_queries = 2 * len(data)
        if len(ranks) != expected_queries or len(tail_ranks) != len(data) or len(head_ranks) != len(data):
            raise ValueError(
                f"{split_name}: incomplete evaluation results "
                f"(all={len(ranks)}/{expected_queries}, tail={len(tail_ranks)}/{len(data)}, "
                f"head={len(head_ranks)}/{len(data)})"
            )
        if len(tail_opt_ranks) != len(data):
            raise ValueError(f"{split_name}: incomplete optimistic tail ranks {len(tail_opt_ranks)}/{len(data)}")

        return summarize_metrics(ranks, tail_ranks, head_ranks, tail_opt_ranks)

    def _evaluate_parallel(
        self,
        data: List[Triplet],
        split_name: str,
        K: Optional[int],
        progress_every: Optional[int],
        n_workers: int,
    ) -> Dict[str, float]:
        global _WORKER_MODEL
        _WORKER_MODEL = self

        chunk_size = max(1, len(data) // n_workers)
        chunks = []
        for start in range(0, len(data), chunk_size):
            chunk = data[start:start + chunk_size]
            chunks.append((list(range(start, start + len(chunk))), chunk, K))

        print(
            f"[PathBSR] {split_name}: parallel eval with {n_workers} workers, "
            f"{len(data):,} triples in {len(chunks)} chunks",
            file=sys.stderr, flush=True,
        )

        ctx = _mp.get_context("fork")
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.map(_eval_chunk, chunks)

        tail_ranks: List[float] = [0.0] * len(data)
        tail_opt_ranks: List[float] = [0.0] * len(data)
        head_ranks: List[float] = [0.0] * len(data)
        for indices, tr, tor, hr in results:
            for local_i, global_i in enumerate(indices):
                tail_ranks[global_i] = tr[local_i]
                tail_opt_ranks[global_i] = tor[local_i]
                head_ranks[global_i] = hr[local_i]

        _WORKER_MODEL = None

        ranks = [v for pair in zip(tail_ranks, head_ranks) for v in pair]
        if progress_every:
            print(
                f"[PathBSR] {split_name}: parallel eval complete, {len(data):,} triples",
                file=sys.stderr, flush=True,
            )

        expected_queries = 2 * len(data)
        if len(ranks) != expected_queries:
            raise ValueError(f"{split_name}: incomplete results ({len(ranks)}/{expected_queries})")

        return summarize_metrics(ranks, tail_ranks, head_ranks, tail_opt_ranks)

    def rule_summary(self) -> Dict[str, float]:
        return {**self.rule_miner.summary(), **self.ranker.summary()}


# Backward-compatible name retained for old scripts and archived experiments.
PathBSR = PathBSR

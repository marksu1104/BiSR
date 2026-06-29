"""Proxy retrieval indexes for PathBSR."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple

import numpy as np

from .config import BM25_B, BM25_K1, PathBSRConfig
from .graph import FeatureMap, GraphStore


class BM25Index:
    """Small BM25 index over sparse entity features."""

    def __init__(
        self,
        features_by_entity: FeatureMap,
        k1: float = BM25_K1,
        b: float = BM25_B,
        query_tf_mode: str = "log",
        idf_floor: float = 0.01,
    ) -> None:
        self.features = features_by_entity
        self.k1 = float(k1)
        self.b = float(b)
        self.query_tf_mode = query_tf_mode
        self.idf_floor = float(idf_floor)
        self.entities = list(features_by_entity.keys())
        self.doc_len = {entity: float(sum(tokens.values())) for entity, tokens in features_by_entity.items()}
        self.avg_dl = float(np.mean(list(self.doc_len.values()))) if self.doc_len else 1.0
        if self.avg_dl <= 0:
            self.avg_dl = 1.0

        self.inverted: dict[str, Set[str]] = defaultdict(set)
        for entity, token_counts in self.features.items():
            for token in token_counts:
                self.inverted[token].add(entity)

        n_docs = max(len(self.entities), 1)
        self.idf: dict[str, float] = {}
        for token, entities in self.inverted.items():
            df = len(entities)
            raw_idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            self.idf[token] = max(raw_idf, self.idf_floor)

    def overlap_entities(self, query_tokens: Mapping[str, float]) -> Set[str]:
        candidates: Set[str] = set()
        for token in query_tokens:
            candidates.update(self.inverted.get(token, set()))
        return candidates

    def score(self, query_tokens: Mapping[str, float], entity: str) -> float:
        doc_tokens = self.features.get(entity)
        if not query_tokens or not doc_tokens:
            return 0.0

        overlap = set(query_tokens).intersection(doc_tokens)
        if not overlap:
            return 0.0

        dl = self.doc_len.get(entity, 0)
        norm = self.k1 * (1.0 - self.b + self.b * (dl / self.avg_dl))
        score = 0.0
        for token in overlap:
            tf = float(doc_tokens.get(token, 0))
            qtf = float(query_tokens.get(token, 0))
            if tf <= 0 or qtf <= 0:
                continue
            if self.query_tf_mode == "binary":
                query_weight = 1.0
            elif self.query_tf_mode == "log":
                query_weight = 1.0 + math.log(qtf) if qtf > 1.0 else 1.0
            else:
                raise ValueError(f"Unknown bm25_query_tf_mode: {self.query_tf_mode}")
            score += self.idf.get(token, self.idf_floor) * (tf * (self.k1 + 1.0)) / (tf + norm) * query_weight
        return score

    def score_many(
        self,
        query_tokens: Mapping[str, float],
        entities: Iterable[str],
    ) -> List[Tuple[str, float]]:
        scored: List[Tuple[str, float]] = []
        for entity in entities:
            score = self.score(query_tokens, entity)
            if score > 0:
                scored.append((entity, score))
        return scored


class FeatureSimilarityIndex:
    """Sparse feature similarity index used by proxy-retrieval ablations.

    The index shares the same public methods as :class:`BM25Index` so it can be
    plugged into :class:`ProxyRetriever` without changing the downstream answer
    generation or candidate scoring code.
    """

    def __init__(self, features_by_entity: FeatureMap, mode: str) -> None:
        if mode not in {"cosine", "overlap"}:
            raise ValueError(f"Unknown proxy_similarity_mode: {mode}")
        self.features = features_by_entity
        self.mode = mode
        self.inverted: dict[str, Set[str]] = defaultdict(set)
        for entity, token_counts in self.features.items():
            for token in token_counts:
                self.inverted[token].add(entity)
        self._norm: dict[str, float] = {}
        if mode == "cosine":
            for entity, token_counts in self.features.items():
                self._norm[entity] = math.sqrt(sum(float(value) ** 2 for value in token_counts.values()))

    def overlap_entities(self, query_tokens: Mapping[str, float]) -> Set[str]:
        candidates: Set[str] = set()
        for token in query_tokens:
            candidates.update(self.inverted.get(token, set()))
        return candidates

    def score(self, query_tokens: Mapping[str, float], entity: str) -> float:
        doc_tokens = self.features.get(entity)
        if not query_tokens or not doc_tokens:
            return 0.0
        overlap = set(query_tokens).intersection(doc_tokens)
        if not overlap:
            return 0.0
        if self.mode == "overlap":
            return float(len(overlap))

        dot = sum(float(query_tokens[token]) * float(doc_tokens[token]) for token in overlap)
        qnorm = math.sqrt(sum(float(value) ** 2 for value in query_tokens.values()))
        dnorm = self._norm.get(entity, 0.0)
        if qnorm <= 0 or dnorm <= 0:
            return 0.0
        return dot / (qnorm * dnorm)

    def score_many(
        self,
        query_tokens: Mapping[str, float],
        entities: Iterable[str],
    ) -> List[Tuple[str, float]]:
        scored: List[Tuple[str, float]] = []
        for entity in entities:
            score = self.score(query_tokens, entity)
            if score > 0:
                scored.append((entity, score))
        return scored


def build_proxy_index(features_by_entity: FeatureMap, config: PathBSRConfig) -> BM25Index | FeatureSimilarityIndex:
    """Build the proxy-retrieval index requested by ``config``."""
    if config.proxy_similarity_mode == "bm25":
        return BM25Index(
            features_by_entity,
            query_tf_mode=config.bm25_query_tf_mode,
            idf_floor=config.bm25_idf_floor,
        )
    if config.proxy_similarity_mode in {"cosine", "overlap"}:
        return FeatureSimilarityIndex(features_by_entity, config.proxy_similarity_mode)
    raise ValueError(f"Unknown proxy_similarity_mode: {config.proxy_similarity_mode}")


class ProxyRetriever:
    """Retrieve structurally similar source entities as cases."""

    def __init__(
        self,
        graph: GraphStore,
        entity_features: FeatureMap,
        index: BM25Index | FeatureSimilarityIndex,
        config: PathBSRConfig,
    ) -> None:
        self.graph = graph
        self.entity_features = entity_features
        self.index = index
        self.config = config
        self._proxy_cache: Dict[Tuple[str, str, int], List[Tuple[str, float]]] = {}
        self._direct_score_cache: Dict[Tuple[str, str, int], np.ndarray] = {}

    def retrieve(
        self,
        head: str,
        relation: str,
        topk: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        topk = self.config.topk_proxy if topk is None else int(topk)
        cache_key = (head, relation, topk)
        if cache_key in self._proxy_cache:
            return self._proxy_cache[cache_key]

        head_tokens = self.entity_features.get(head)
        if not head_tokens:
            self._proxy_cache[cache_key] = []
            return []

        overlap = self.index.overlap_entities(head_tokens)
        candidates = overlap.intersection(self.graph.relation_heads.get(relation, set()))
        candidates = {entity for entity in candidates if entity != head}
        scored = self.index.score_many(head_tokens, candidates)
        scored.sort(key=lambda item: (-item[1], item[0]))
        top_scored = scored[:topk]
        if self.config.bm25_normalize_proxy_scores and top_scored:
            total = sum(score for _, score in top_scored)
            if total > 0:
                top_scored = [(entity, score / total) for entity, score in top_scored]
        self._proxy_cache[cache_key] = top_scored
        return self._proxy_cache[cache_key]

    def score_direct_cases(self, head: str, relation: str, topk: Optional[int] = None) -> np.ndarray:
        topk = self.config.topk_proxy if topk is None else int(topk)
        cache_key = (head, relation, topk)
        if cache_key in self._direct_score_cache:
            return self._direct_score_cache[cache_key]

        scores = np.zeros(len(self.graph.all_entities), dtype=np.float32)
        for proxy_head, proxy_similarity in self.retrieve(head, relation, topk):
            for proxy_answer in self.graph.out_adj.get(proxy_head, {}).get(relation, set()):
                idx = self.graph.ent2idx.get(proxy_answer)
                if idx is not None:
                    scores[idx] += float(proxy_similarity)
        self._direct_score_cache[cache_key] = scores
        return scores

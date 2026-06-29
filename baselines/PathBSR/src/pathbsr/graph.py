"""Graph construction and sparse entity features."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Sequence, Set, Tuple

from .config import PathBSRConfig
from .data import Triplet, augment_with_reverse_edges

FeatureMap = Dict[str, Counter[str]]


def build_entity_features(
    train_aug: Sequence[Triplet],
    config: PathBSRConfig | None = None,
) -> Tuple[FeatureMap, Set[str]]:
    """Build REL + ENT + REL-ENT features and TF counts for BM25 case retrieval."""
    entity_features: FeatureMap = defaultdict(Counter)
    entities: Set[str] = set()
    feature_mode = "all_tf" if config is None else config.entity_feature_mode

    for head, relation, tail in train_aug:
        entities.add(head)
        entities.add(tail)
        rel_feature = f"REL:{relation}"
        ent_feature = f"ENT:{tail}"
        rel_ent_feature = f"REL-ENT:{relation}|{tail}"
        if feature_mode == "rel":
            entity_features[head][rel_feature] += 1
        elif feature_mode == "rel_ent":
            entity_features[head][rel_feature] += 1
            entity_features[head][ent_feature] += 1
        elif feature_mode == "all_binary":
            entity_features[head][rel_feature] = 1
            entity_features[head][ent_feature] = 1
            entity_features[head][rel_ent_feature] = 1
        elif feature_mode == "all_tf":
            entity_features[head][rel_feature] += 1
            entity_features[head][ent_feature] += 1
            entity_features[head][rel_ent_feature] += 1
        else:
            raise ValueError(f"Unknown entity_feature_mode: {feature_mode}")

    for entity in entities:
        _ = entity_features[entity]

    return dict(entity_features), entities


@dataclass
class GraphStore:
    """Shared graph indexes used by retrieval, paths, ranking, and evaluation."""

    train_raw: list[Triplet]
    valid_raw: list[Triplet]
    test_raw: list[Triplet]
    train_aug: list[Triplet]
    all_entities: list[str]
    ent2idx: dict[str, int]
    relation_heads: Dict[str, Set[str]]
    out_adj: Dict[str, Dict[str, Set[str]]]
    out_adj_list: Dict[str, Dict[str, list[str]]]
    in_adj: Dict[str, Dict[str, Set[str]]]
    in_adj_list: Dict[str, Dict[str, list[str]]]
    pair_relations: Dict[Tuple[str, str], Set[str]]
    all_true_tails: Dict[Tuple[str, str], Set[str]]
    train_true_tails: Dict[Tuple[str, str], Set[str]]
    undirected_neighbors: Dict[str, Set[str]]
    undirected_degree: Dict[str, int]

    @classmethod
    def build(
        cls,
        train_triplets: Sequence[Triplet],
        valid_triplets: Sequence[Triplet],
        test_triplets: Sequence[Triplet],
        train_entities: Set[str],
        config: PathBSRConfig,
    ) -> "GraphStore":
        train_raw = list(train_triplets)
        valid_raw = list(valid_triplets)
        test_raw = list(test_triplets)
        train_aug = augment_with_reverse_edges(train_raw, config.reverse_suffix)

        all_raw = train_raw + valid_raw + test_raw
        all_entities = sorted({entity for h, _, t in all_raw for entity in (h, t)} | train_entities)
        ent2idx = {entity: idx for idx, entity in enumerate(all_entities)}

        relation_heads: Dict[str, Set[str]] = defaultdict(set)
        out_adj: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        in_adj: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        pair_relations: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        train_true_tails: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        for head, relation, tail in train_aug:
            relation_heads[relation].add(head)
            out_adj[head][relation].add(tail)
            in_adj[tail][relation].add(head)
            pair_relations[(head, tail)].add(relation)
            train_true_tails[(head, relation)].add(tail)

        out_adj_list = {
            head: {relation: sorted(tails) for relation, tails in rel_map.items()}
            for head, rel_map in out_adj.items()
        }
        in_adj_list = {
            tail: {relation: sorted(heads) for relation, heads in rel_map.items()}
            for tail, rel_map in in_adj.items()
        }

        all_true_tails: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        for head, relation, tail in all_raw:
            all_true_tails[(head, relation)].add(tail)
            all_true_tails[(tail, f"{relation}{config.reverse_suffix}")].add(head)

        undirected_neighbors: Dict[str, Set[str]] = defaultdict(set)
        for head, _, tail in train_raw:
            if head == tail:
                continue
            undirected_neighbors[head].add(tail)
            undirected_neighbors[tail].add(head)
        undirected_degree = {
            entity: len(neighbors) for entity, neighbors in undirected_neighbors.items()
        }

        return cls(
            train_raw=train_raw,
            valid_raw=valid_raw,
            test_raw=test_raw,
            train_aug=train_aug,
            all_entities=all_entities,
            ent2idx=ent2idx,
            relation_heads=relation_heads,
            out_adj=out_adj,
            out_adj_list=out_adj_list,
            in_adj=in_adj,
            in_adj_list=in_adj_list,
            pair_relations=pair_relations,
            all_true_tails=all_true_tails,
            train_true_tails=train_true_tails,
            undirected_neighbors=undirected_neighbors,
            undirected_degree=undirected_degree,
        )

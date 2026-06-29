"""Dataset utilities."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

Triplet = Tuple[str, str, str]


def remove_train_overlap(
    train_triplets: Sequence[Triplet],
    evaluation_triplets: Sequence[Triplet],
) -> Tuple[list[Triplet], int]:
    """Return an evaluation-only sensitivity split without exact train facts."""
    train_facts = set(train_triplets)
    cleaned = [triple for triple in evaluation_triplets if triple not in train_facts]
    return cleaned, len(evaluation_triplets) - len(cleaned)
MAX_AUDIT_EXAMPLES = 5


def read_triplets(path: Path) -> List[Triplet]:
    """Read head<TAB>tail<TAB>relation triples as (head, relation, tail)."""
    triplets: List[Triplet] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            triplets.append(_parse_triplet_line(line, path, lineno))
    return triplets


def dataset_split_paths(data_root: Path, dataset_name: str) -> Dict[str, Path]:
    """Return the canonical train/valid/test file paths for one dataset."""
    data_dir = data_root / dataset_name
    return {
        "train": data_dir / "train.txt",
        "valid": data_dir / "valid.txt",
        "test": data_dir / "test.txt",
    }


def load_dataset(data_root: Path, dataset_name: str) -> Tuple[List[Triplet], List[Triplet], List[Triplet]]:
    """Load train/valid/test splits from a dataset directory."""
    train, valid, test, _ = load_dataset_with_audit(data_root, dataset_name)
    return train, valid, test


def load_dataset_with_audit(
    data_root: Path,
    dataset_name: str,
) -> Tuple[List[Triplet], List[Triplet], List[Triplet], Dict[str, Any]]:
    """Load one dataset and return raw triplets plus a machine-readable audit."""
    paths = dataset_split_paths(data_root, dataset_name)
    splits = {split_name: read_triplets(path) for split_name, path in paths.items()}
    audit = {
        "dataset": dataset_name,
        "splits": {
            split_name: {
                "path": str(path.resolve()),
                **summarize_split_triplets(splits[split_name]),
            }
            for split_name, path in paths.items()
        },
        "overlap": summarize_split_overlap(splits),
    }
    return splits["train"], splits["valid"], splits["test"], audit


def augment_with_reverse_edges(
    triplets: Sequence[Triplet],
    reverse_suffix: str,
) -> List[Triplet]:
    """Add reciprocal edges for bidirectional link prediction."""
    return list(triplets) + [(tail, f"{relation}{reverse_suffix}", head) for head, relation, tail in triplets]


def summarize_split_triplets(triplets: Sequence[Triplet]) -> Dict[str, Any]:
    """Summarize duplicate statistics for one split without modifying it."""
    counts = Counter(triplets)
    duplicate_examples = [
        {"triplet": list(triplet), "count": count}
        for triplet, count in counts.items()
        if count > 1
    ]
    duplicate_examples.sort(key=lambda item: (-item["count"], item["triplet"]))
    return {
        "raw_count": len(triplets),
        "unique_count": len(counts),
        "duplicate_count": len(triplets) - len(counts),
        "duplicate_examples": duplicate_examples[:MAX_AUDIT_EXAMPLES],
    }


def summarize_split_overlap(splits: Dict[str, Sequence[Triplet]]) -> Dict[str, Dict[str, Any]]:
    """Summarize pairwise split overlap using unique triplets only."""
    unique = {name: set(triplets) for name, triplets in splits.items()}
    pairs = [("train", "valid"), ("train", "test"), ("valid", "test")]
    overlap: Dict[str, Dict[str, Any]] = {}
    for left, right in pairs:
        key = f"{left}_{right}"
        shared = sorted(unique[left] & unique[right])
        overlap[key] = {
            "count": len(shared),
            "examples": [list(triplet) for triplet in shared[:MAX_AUDIT_EXAMPLES]],
        }
    return overlap


def _parse_triplet_line(line: str, path: Path, lineno: int) -> Triplet:
    raw = line.rstrip("\n\r")
    parts = raw.split("\t")
    if len(parts) != 3:
        raise ValueError(f"{path}:{lineno}: expected 3 tab-separated fields, got {len(parts)}")
    head, tail, relation = parts
    if not head or not tail or not relation:
        raise ValueError(f"{path}:{lineno}: empty head/tail/relation field is not allowed")
    return (head, relation, tail)

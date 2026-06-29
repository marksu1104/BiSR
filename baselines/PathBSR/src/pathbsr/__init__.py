"""PathBSR: sparse case-based reasoning for KGC."""

from __future__ import annotations

import random
import numpy as np

from .config import DEFAULT_CONFIG, PathBSRConfig, REVERSE_SUFFIX, SEED, config_to_dict
from .data import Triplet, augment_with_reverse_edges, load_dataset, load_dataset_with_audit, read_triplets, remove_train_overlap
from .model import PathBSR


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


__all__ = [
    "PathBSR",
    "PathBSRConfig",
    "DEFAULT_CONFIG",
    "REVERSE_SUFFIX",
    "SEED",
    "Triplet",
    "augment_with_reverse_edges",
    "config_to_dict",
    "load_dataset",
    "load_dataset_with_audit",
    "read_triplets",
    "remove_train_overlap",
    "set_seed",
]

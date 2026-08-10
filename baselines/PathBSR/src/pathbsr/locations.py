"""Portable workspace locations for standalone and vendored PathBSR layouts."""

from __future__ import annotations

import os
from pathlib import Path


PATHBSR_ROOT = Path(__file__).resolve().parents[2]


def workspace_root() -> Path:
    if PATHBSR_ROOT.parent.name == "baselines":
        return PATHBSR_ROOT.parent.parent
    return PATHBSR_ROOT


def data_root() -> Path:
    configured = os.environ.get("SPARSEKGC_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return workspace_root() / "datasets"

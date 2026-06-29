#!/usr/bin/env python3
"""Run the full PathBSR pipeline from the repository scripts directory.

This file is intentionally a thin wrapper around ``pathbsr.cli`` so the
implementation has a single source of truth while users still have a simple
script entry point:

    PYTHONPATH=src python3 scripts/run_pathbsr.py --dataset NELL23K --split valid
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathbsr.cli import main  # noqa: E402


if __name__ == "__main__":
    main()

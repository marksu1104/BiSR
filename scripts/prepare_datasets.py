#!/usr/bin/env python3
"""Validate canonical datasets and prepare baseline-specific working data."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_ROOT = REPO_ROOT / "datasets"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "preprocessed"

DATASETS = [
    "WD-singer",
    "FB15K-237-10",
    "WN18RR",
    "FB15K-237-20",
    "FB15K-237-50",
    "NELL23K",
    "FB15K-237",
]
SPLITS = ("train", "valid", "test")
ENTITY_TYPE_DATASETS = {
    "WD-singer",
    "FB15K-237-10",
    "FB15K-237-20",
    "FB15K-237-50",
    "NELL23K",
}
DACKGR_FILES = {
    "train.triples": "train.txt",
    "dev.triples": "valid.txt",
    "test.triples": "test.txt",
    "raw.kb": "train.txt",
    "raw.pgrk": "metadata/dackgr_pagerank.txt",
}


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required canonical dataset file not found: {path}")


def validate_columns(path: Path, expected: int, separator: str = "\t") -> int:
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\r\n").split(separator)
            if len(fields) != expected or any(not field for field in fields):
                raise ValueError(
                    f"Invalid row in {path}:{line_number}; expected {expected} "
                    "non-empty tab-separated fields"
                )
            rows += 1
    if rows == 0:
        raise ValueError(f"Canonical dataset file is empty: {path}")
    return rows


def validate_pagerank(path: Path) -> int:
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                entity, score = line.rstrip("\r\n").rsplit(":", 1)
                if not entity.strip():
                    raise ValueError
                float(score)
            except ValueError as exc:
                raise ValueError(f"Invalid PageRank row in {path}:{line_number}") from exc
            rows += 1
    if rows == 0:
        raise ValueError(f"PageRank file is empty: {path}")
    return rows


def validate_dataset(dataset: str) -> None:
    dataset_dir = CANONICAL_ROOT / dataset
    for split in SPLITS:
        path = dataset_dir / f"{split}.txt"
        require_file(path)
        validate_columns(path, expected=3)

    pagerank = dataset_dir / "metadata" / "dackgr_pagerank.txt"
    require_file(pagerank)
    validate_pagerank(pagerank)

    entity_types = dataset_dir / "metadata" / "entity_types.txt"
    if dataset in ENTITY_TYPE_DATASETS:
        require_file(entity_types)
        validate_columns(entity_types, expected=2)
    elif entity_types.exists():
        validate_columns(entity_types, expected=2)


def copy_exact(source: Path, target: Path, check_only: bool, force: bool) -> str:
    require_file(source)
    if target.exists():
        if target.is_file() and digest(source) == digest(target):
            return "verified"
        if check_only or not force:
            raise RuntimeError(
                f"Refusing to replace different prepared data: {target}. "
                "Inspect it first, then use --force if replacement is intended."
            )
    elif check_only:
        raise FileNotFoundError(f"Prepared dataset file not found: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)
    return "written"


def prepare_dackgr(dataset: str, output_root: Path, check_only: bool, force: bool) -> dict[str, int]:
    source_dir = CANONICAL_ROOT / dataset
    target_dir = output_root / "dackgr" / dataset
    counts = {"verified": 0, "written": 0}
    for target_name, source_name in DACKGR_FILES.items():
        status = copy_exact(
            source_dir / source_name,
            target_dir / target_name,
            check_only=check_only,
            force=force,
        )
        counts[status] += 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS)
    parser.add_argument(
        "--baseline",
        choices=("canonical", "dackgr"),
        default="canonical",
        help="Validate canonical inputs or prepare one baseline's working format",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--check", action="store_true", help="Verify without writing files")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace different files only in the ignored prepared-data directory",
    )
    args = parser.parse_args()
    if args.baseline != "canonical" and not args.datasets:
        parser.error("--datasets is required when preparing baseline working data")
    return args


def main() -> None:
    args = parse_args()
    datasets = args.datasets or DATASETS
    for dataset in datasets:
        validate_dataset(dataset)

    if args.baseline == "canonical":
        print(f"Validated {len(datasets)} canonical dataset(s).")
        return

    totals = {"verified": 0, "written": 0}
    for dataset in datasets:
        counts = prepare_dackgr(dataset, args.output_root, args.check, args.force)
        for key, value in counts.items():
            totals[key] += value
    action = "Verified" if args.check else "Prepared"
    print(
        f"{action} {sum(totals.values())} DacKGR input file(s) "
        f"({totals['written']} written, {totals['verified']} unchanged)."
    )


if __name__ == "__main__":
    main()

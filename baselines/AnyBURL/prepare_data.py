#!/usr/bin/env python3
"""
Convert SparseKGC datasets to AnyBURL's expected triple format.

SparseKGC stores triples as `head <TAB> tail <TAB> relation` (h t r).
AnyBURL expects `head <TAB> relation <TAB> tail` (h r t), tab-separated,
one triple per line, with separate train / valid / test files.

AnyBURL natively predicts both heads and tails (it learns and applies rules in
both directions), so we do NOT add inverse edges here -- the bidirectional
evaluation is handled by AnyBURL's own head/tail prediction, matching the
filtered bidirectional protocol used by the other SparseKGC baselines.

Output: <out_root>/<dataset>/{train,valid,test}.txt
"""
import argparse
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

DATASETS = ["WD-singer", "FB15K-237-10", "WN18RR",
            "FB15K-237-20", "FB15K-237-50", "NELL23K"]


def convert_file(src: Path, dst: Path):
    n = 0
    with src.open("r") as fin, dst.open("w") as fout:
        for line in fin:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            h, t, r = parts[0], parts[1], parts[2]
            # SparseKGC (h t r) -> AnyBURL (h r t)
            fout.write(f"{h}\t{r}\t{t}\n")
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description="Convert SparseKGC datasets to AnyBURL format")
    parser.add_argument("--data-root", type=str, default=str(REPO_ROOT / "datasets"))
    parser.add_argument("--out-root", type=str, default=str(SCRIPT_DIR / "work"))
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)

    for ds in args.datasets:
        src_dir = data_root / ds
        if not src_dir.exists():
            print(f"SKIP {ds}: {src_dir} not found")
            continue
        dst_dir = out_root / ds
        dst_dir.mkdir(parents=True, exist_ok=True)
        counts = {}
        for split in ("train", "valid", "test"):
            src = src_dir / f"{split}.txt"
            if not src.exists():
                print(f"  WARN {ds}/{split}.txt missing")
                continue
            counts[split] = convert_file(src, dst_dir / f"{split}.txt")
        print(f"OK {ds}: " + " ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Convert SparseKGC datasets (h t r, tab, no inverses) to LoGRe format
(h t r, tab, WITH explicit inverse triples r_inv) and generate entity2type.txt.

Outputs (written to work_dir/dataset/):
  train.txt        : train triples + inverses
  dev.txt          : valid triples + inverses
  dev.triples      : valid triples only (no inv)
  test.txt         : test triples + inverses (for all_kg_map filtering)
  test.triples     : test triples only (for tail-only eval)
  test_inv.triples : inverted test triples (t h r_inv) for head-query eval
  entity2type.txt  : entity -> type mapping
"""
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ENTITY_TYPES_DIR = SCRIPT_DIR.parent / "entity_types"


def _inv(r: str) -> str:
    return r[:-4] if r.endswith("_inv") else r + "_inv"


def _read_triples(path: Path):
    triples = []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                triples.append((parts[0], parts[1], parts[2]))
    return triples


def _write_triples(path: Path, triples):
    with open(path, "w") as f:
        for h, t, r in triples:
            f.write(f"{h}\t{t}\t{r}\n")


def _with_inverses(triples):
    out = []
    for h, t, r in triples:
        out.append((h, t, r))
        out.append((t, h, _inv(r)))
    return out


def prepare(src_dir: Path, work_dir: Path, dataset: str) -> Path:
    """Prepare LoGRe-format data from SparseKGC dataset.

    Returns the work sub-directory for this dataset.
    """
    src = src_dir / dataset
    out = work_dir / dataset
    out.mkdir(parents=True, exist_ok=True)

    train = _read_triples(src / "train.txt")
    valid = _read_triples(src / "valid.txt")
    test  = _read_triples(src / "test.txt")

    _write_triples(out / "train.txt",        _with_inverses(train))
    _write_triples(out / "dev.txt",          _with_inverses(valid))
    _write_triples(out / "dev.triples",      valid)
    _write_triples(out / "test.txt",         _with_inverses(test))
    _write_triples(out / "test.triples",     test)
    # inverse test file: (t, h, r_inv) for head-query evaluation
    _write_triples(out / "test_inv.triples", [(t, h, _inv(r)) for h, t, r in test])

    _build_entity2type(out, dataset, train + valid + test)
    return out


def _build_entity2type(out: Path, dataset: str, triples):
    bundled = ENTITY_TYPES_DIR / f"{dataset}.txt"
    type_map = {}
    if bundled.exists():
        with open(bundled) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 2:
                    type_map[parts[0]] = parts[1]

    entities = set()
    for h, t, _ in triples:
        entities.add(h)
        entities.add(t)

    with open(out / "entity2type.txt", "w") as f:
        for e in sorted(entities):
            f.write(f"{e}\t{type_map.get(e, 'unknown')}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    args = parser.parse_args()
    for ds in args.datasets:
        out = prepare(Path(args.data_root), Path(args.work_root), ds)
        print(f"Prepared {ds} -> {out}")

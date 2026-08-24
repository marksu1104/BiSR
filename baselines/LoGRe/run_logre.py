#!/usr/bin/env python3
"""
Run the LoGRe baseline end-to-end for the SparseKGC datasets.

Pipeline per dataset:
  1. prepare_data  – convert SparseKGC format -> LoGRe format (+ entity2type)
  2. forward run   – LoGRe on test.triples (tail queries) + score dump
  3. inverse run   – LoGRe on test_inv.triples (head queries) + score dump
  4. score         – bidirectional tie-aware metrics via score_struprokgr.evaluate
  5. upsert CSV    – main protocol metrics -> outputs/logre_metrics.csv
                     tail-only optimistic  -> outputs/logre_sota_metrics.csv

LoGRe is numpy/CPU (torch used for GPU ansim if available); no GPU required.
Runs on x86 (short partition).
WN18RR is not supported by LoGRe (no entity types for it).
"""
import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from log_format import print_start, print_result  # noqa: E402
from metrics_csv import upsert_metrics_csv        # noqa: E402
sys.path.insert(0, str(SCRIPT_DIR))
from prepare_data import prepare          # noqa: E402
sys.path.insert(0, str(SCRIPT_DIR.parent / "StruProKGR"))
from score_struprokgr import evaluate    # noqa: E402

SUPPORTED_DATASETS = [
    "WD-singer", "FB15K-237-10", "FB15K-237-20", "FB15K-237-50", "NELL23K",
]

LOGRE_DEFAULTS = dict(
    max_num_programs=1000,
    num_paths_to_collect=1000,
    max_path_len=3,
    decay_factor=0.95,
    max_branch=1000,
)

# Per-dataset hyperparameters from the official LoGRe repo README
# (https://github.com/gsp2014/LoGRe). These reproduce the paper's reported
# numbers. max_branch is not set in the README commands, so it uses the
# LoGRe.py argparse default (1000).
LOGRE_PER_DATASET = {
    "FB15K-237-10": dict(max_num_programs=1000, num_paths_to_collect=20000, max_path_len=6, decay_factor=0.95, max_branch=1000),
    "FB15K-237-20": dict(max_num_programs=500,  num_paths_to_collect=5000,  max_path_len=5, decay_factor=0.6,  max_branch=1000),
    "FB15K-237-50": dict(max_num_programs=100,  num_paths_to_collect=1000,  max_path_len=4, decay_factor=0.8,  max_branch=1000),
    "NELL23K":      dict(max_num_programs=100,  num_paths_to_collect=10000, max_path_len=6, decay_factor=0.5,  max_branch=1000),
    "WD-singer":    dict(max_num_programs=100,  num_paths_to_collect=20000, max_path_len=6, decay_factor=0.2,  max_branch=1000),
}


def apply_per_dataset(dataset: str, args):
    """Override args with the paper's per-dataset hyperparameters (if defined)."""
    cfg = LOGRE_PER_DATASET.get(dataset)
    if not cfg:
        return
    for k, v in cfg.items():
        setattr(args, k, v)


def timestamp():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S") + f",{now.microsecond // 1000:03d}"


def metrics_csv_path(suffix=""):
    root = os.environ.get("SPARSEKGC_OUTPUT_DIR")
    base = Path(root) if root else (REPO_ROOT / "outputs")
    base.mkdir(parents=True, exist_ok=True)
    return base / f"logre{suffix}_metrics.csv"


def baseline_log_dir(create=True):
    root = os.environ.get("SPARSEKGC_OUTPUT_DIR")
    d = (Path(root) / "logre") if root else (REPO_ROOT / "outputs" / "logre")
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def run_logre_once(work_dir: Path, dataset: str, test_file: str,
                   dump_file: Path, out_dir: Path, log_fh, args):
    """Run LoGRe for one direction (forward or inverse)."""
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "LoGRe.py"),
        "--data_dir", str(work_dir.parent),
        "--dataset", dataset,
        "--out_dir", str(out_dir),
        "--test",
        "--test_file_name", test_file,
        "--dump_scores_file", str(dump_file),
        "--name_of_run", f"{dataset}_{test_file.replace('.', '_')}",
        "--max_num_programs", str(args.max_num_programs),
        "--num_paths_to_collect", str(args.num_paths_to_collect),
        "--max_path_len",    str(args.max_path_len),
        "--decay_factor",    str(args.decay_factor),
        "--max_branch",      str(args.max_branch),
    ]
    if getattr(args, "dry_run", False):
        import shlex
        print(f"Dry-run | {shlex.join([str(x) for x in cmd])}", flush=True)
        return None
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
    sota_line = None
    for line in proc.stdout:
        log_fh.write(line)
        log_fh.flush()
        if line.startswith("LOGRE_METRICS"):
            sota_line = line.strip()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"LoGRe exited {proc.returncode} for {dataset} {test_file}")
    return sota_line


def parse_sota(line):
    """Extract {mrr, h1, h3, h10} from LOGRE_METRICS stdout line."""
    if line is None:
        return None
    m = {}
    for key in ("mrr", "h1", "h3", "h10"):
        pat = rf"{key}=([\d.]+)"
        match = re.search(pat, line)
        if match:
            m[key] = float(match.group(1))
    return m if m else None


def run_one(dataset: str, args):
    data_root = Path(args.data_root)
    work_root = Path(args.work_root)
    dry_run = bool(getattr(args, "dry_run", False))
    log_root = baseline_log_dir(create=not dry_run)
    out_dir = log_root / dataset
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    # 0. Apply paper's per-dataset hyperparameters
    apply_per_dataset(dataset, args)

    # 1. Prepare data
    work_dir = work_root / dataset if dry_run else prepare(data_root, work_root, dataset)

    params = (
        f"max_programs={args.max_num_programs} num_paths={args.num_paths_to_collect} "
        f"max_len={args.max_path_len} decay={args.decay_factor} max_branch={args.max_branch}"
    )
    print_start(timestamp(), "logre", "LoGRe", dataset, params)

    log_file = log_root / f"LoGRe_{dataset}.log"
    fwd_dump = out_dir / "dump_forward.tsv"
    inv_dump = out_dir / "dump_inverse.tsv"
    if dry_run:
        run_logre_once(work_dir, dataset, "test.triples", fwd_dump, out_dir, None, args)
        run_logre_once(work_dir, dataset, "test_inv.triples", inv_dump, out_dir, None, args)
        return

    start = time.perf_counter()
    with log_file.open("w", buffering=1) as log_fh:
        # 2. Forward run (tail queries: test.triples)
        sota_line = run_logre_once(work_dir, dataset, "test.triples", fwd_dump, out_dir, log_fh, args)

        # 3. Inverse run (head queries: test_inv.triples)
        run_logre_once(work_dir, dataset, "test_inv.triples", inv_dump, out_dir, log_fh, args)

    seconds = time.perf_counter() - start

    # 4. Score under Main Protocol (bidirectional, filtered, full-entity, average-tie)
    res = evaluate(fwd_dump, inv_dump, data_root, dataset, tie_mode="average")

    final_line = (
        "FINAL_EVAL_METRICS baseline=logre model=LoGRe dataset={} split=test "
        "mrr_tail={:.5f} mrr_head={:.5f} mrr_avg={:.5f} "
        "h1_tail={:.5f} h1_head={:.5f} h1_avg={:.5f} "
        "h3_tail={:.5f} h3_head={:.5f} h3_avg={:.5f} "
        "h10_tail={:.5f} h10_head={:.5f} h10_avg={:.5f}".format(
            dataset,
            res["tail"]["mrr"], res["head"]["mrr"], res["avg"]["mrr"],
            res["tail"]["h1"],  res["head"]["h1"],  res["avg"]["h1"],
            res["tail"]["h3"],  res["head"]["h3"],  res["avg"]["h3"],
            res["tail"]["h10"], res["head"]["h10"], res["avg"]["h10"],
        )
    )

    # 5a. Upsert main-protocol CSV
    upsert_metrics_csv(str(metrics_csv_path()), [
        dataset, "LoGRe",
        f"{res['tail']['mrr']:.5f}", f"{res['head']['mrr']:.5f}", f"{res['avg']['mrr']:.5f}",
        f"{res['tail']['h1']:.5f}",  f"{res['head']['h1']:.5f}",  f"{res['avg']['h1']:.5f}",
        f"{res['tail']['h3']:.5f}",  f"{res['head']['h3']:.5f}",  f"{res['avg']['h3']:.5f}",
        f"{res['tail']['h10']:.5f}", f"{res['head']['h10']:.5f}", f"{res['avg']['h10']:.5f}",
        f"{seconds:.3f}",
    ])

    # 5b. Score under SOTA Protocol (tail-only, filtered, full-entity,
    # optimistic-tie) directly from the same forward dump used in step 4 --
    # NOT the legacy LOGRE_METRICS stdout line, which only ranked the gold
    # answer within LoGRe's own returned candidate list rather than the full
    # entity set and is therefore not a valid Full-entity-set SOTA number
    # under thesis Table 5.
    sota_res = evaluate(fwd_dump, None, data_root, dataset, tie_mode="optimistic")
    upsert_metrics_csv(str(metrics_csv_path("_sota")), [
        dataset, "LoGRe",
        f"{sota_res['tail']['mrr']:.5f}", "—", "—",
        f"{sota_res['tail']['h1']:.5f}",  "—", "—",
        f"{sota_res['tail']['h3']:.5f}",  "—", "—",
        f"{sota_res['tail']['h10']:.5f}", "—", "—",
        f"{seconds:.3f}",
    ])

    print_result(timestamp(), "logre", "LoGRe", dataset, log_file, None, final_line, seconds, "ok")


def main():
    parser = argparse.ArgumentParser(description="Run LoGRe baseline over SparseKGC datasets")
    parser.add_argument("--datasets", nargs="+", default=SUPPORTED_DATASETS)
    parser.add_argument("--data-root",  default=str(REPO_ROOT / "datasets"))
    parser.add_argument(
        "--work-root",
        default=str(REPO_ROOT / "outputs" / "preprocessed" / "logre"),
    )
    parser.add_argument("--max-num-programs",    type=int, default=LOGRE_DEFAULTS["max_num_programs"],
                        dest="max_num_programs")
    parser.add_argument("--num-paths-to-collect", type=int, default=LOGRE_DEFAULTS["num_paths_to_collect"],
                        dest="num_paths_to_collect")
    parser.add_argument("--max-path-len",        type=int, default=LOGRE_DEFAULTS["max_path_len"],
                        dest="max_path_len")
    parser.add_argument("--decay-factor",        type=float, default=LOGRE_DEFAULTS["decay_factor"],
                        dest="decay_factor")
    parser.add_argument("--max-branch",          type=int, default=LOGRE_DEFAULTS["max_branch"],
                        dest="max_branch")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    unsupported = [d for d in args.datasets if d not in SUPPORTED_DATASETS]
    if unsupported:
        print(f"WARNING: LoGRe does not support {unsupported}; they will be skipped.", flush=True)
        args.datasets = [d for d in args.datasets if d in SUPPORTED_DATASETS]

    failed = []
    for dataset in args.datasets:
        try:
            run_one(dataset, args)
        except Exception as exc:
            print(f"FAILED | baseline=LoGRe | dataset={dataset} | {exc}", flush=True)
            import traceback; traceback.print_exc()
            failed.append(dataset)

    if failed:
        print(f"LoGRe finished with {len(failed)} failed dataset(s): {failed}", flush=True)


if __name__ == "__main__":
    main()

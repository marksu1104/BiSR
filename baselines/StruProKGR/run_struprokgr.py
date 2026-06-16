#!/usr/bin/env python3
"""
Run the StruProKGR baseline end-to-end for the SparseKGC datasets.

Pipeline per dataset:
  1. prepare_data  – convert SparseKGC format -> StruProKGR format (+ entity2type)
  2. forward run   – StruProKGR on test.triples (tail queries) + score dump
  3. inverse run   – StruProKGR on test_inv.triples (head queries) + score dump
  4. score         – bidirectional tie-aware metrics via score_struprokgr.py
  5. upsert CSV    – main protocol metrics -> outputs/struprokgr_metrics.csv
                     tail-only optimistic  -> outputs/struprokgr_sota_metrics.csv

StruProKGR is CPU-only / pure numpy; no GPU needed. Runs on x86 (short partition).
WN18RR is not supported (StruProKGR has no entity types for it).
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
from score_struprokgr import evaluate    # noqa: E402

SUPPORTED_DATASETS = [
    "WD-singer", "FB15K-237-10", "FB15K-237-20", "FB15K-237-50", "NELL23K",
]

STRUPRO_DEFAULTS = dict(
    max_num_programs=100,
    max_path_len=6,
    max_path_branch=30,
    diminishing_factor=0.5,
    decay_factor=0.2,
)


def timestamp():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S") + f",{now.microsecond // 1000:03d}"


def metrics_csv_path(suffix=""):
    root = os.environ.get("SPARSEKGC_OUTPUT_DIR")
    base = Path(root) if root else (REPO_ROOT / "outputs")
    base.mkdir(parents=True, exist_ok=True)
    return base / f"struprokgr{suffix}_metrics.csv"


def baseline_log_dir():
    root = os.environ.get("SPARSEKGC_OUTPUT_DIR")
    d = (Path(root) / "struprokgr") if root else (REPO_ROOT / "outputs" / "struprokgr")
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_strupro(work_dir: Path, dataset: str, test_file: str,
                dump_file: Path, out_dir: Path, log_fh, args):
    """Run StruProKGR for one direction (forward or inverse)."""
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "StruProKGR.py"),
        "--data_dir", str(work_dir.parent),
        "--dataset", dataset,
        "--out_dir", str(out_dir),
        "--test",
        "--test_file_name", test_file,
        "--dump_scores_file", str(dump_file),
        "--name_of_run", f"{dataset}_{test_file.replace('.', '_')}",
        "--max_num_programs", str(args.max_num_programs),
        "--max_path_len",     str(args.max_path_len),
        "--max_path_branch",  str(args.max_path_branch),
        "--diminishing_factor", str(args.diminishing_factor),
        "--decay_factor",     str(args.decay_factor),
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
        if line.startswith("STRUPROKGR_METRICS"):
            sota_line = line.strip()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"StruProKGR exited {proc.returncode} for {dataset} {test_file}")
    return sota_line


def parse_sota(line):
    """Extract {mrr, h1, h3, h10} from STRUPROKGR_METRICS stdout line."""
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
    out_dir   = baseline_log_dir() / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare data
    work_dir = prepare(data_root, work_root, dataset)

    params = (
        f"max_programs={args.max_num_programs} max_len={args.max_path_len} "
        f"max_branch={args.max_path_branch} diminish={args.diminishing_factor} "
        f"decay={args.decay_factor}"
    )
    print_start(timestamp(), "struprokgr", "StruProKGR", dataset, params)

    log_file = baseline_log_dir() / f"StruProKGR_{dataset}.log"
    start = time.perf_counter()
    with log_file.open("w", buffering=1) as log_fh:
        # 2. Forward run (tail queries: test.triples)
        fwd_dump = out_dir / "dump_forward.tsv"
        sota_line = run_strupro(work_dir, dataset, "test.triples", fwd_dump, out_dir, log_fh, args)

        # 3. Inverse run (head queries: test_inv.triples)
        inv_dump = out_dir / "dump_inverse.tsv"
        run_strupro(work_dir, dataset, "test_inv.triples", inv_dump, out_dir, log_fh, args)

    seconds = time.perf_counter() - start

    if getattr(args, "dry_run", False):
        return

    # 4. Score under main protocol (bidirectional + tie-aware)
    res = evaluate(fwd_dump, inv_dump, data_root, dataset)

    final_line = (
        "FINAL_EVAL_METRICS baseline=struprokgr model=StruProKGR dataset={} split=test "
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
        dataset, "StruProKGR",
        f"{res['tail']['mrr']:.5f}", f"{res['head']['mrr']:.5f}", f"{res['avg']['mrr']:.5f}",
        f"{res['tail']['h1']:.5f}",  f"{res['head']['h1']:.5f}",  f"{res['avg']['h1']:.5f}",
        f"{res['tail']['h3']:.5f}",  f"{res['head']['h3']:.5f}",  f"{res['avg']['h3']:.5f}",
        f"{res['tail']['h10']:.5f}", f"{res['head']['h10']:.5f}", f"{res['avg']['h10']:.5f}",
        f"{seconds:.3f}",
    ])

    # 5b. Upsert SOTA-protocol CSV (tail-only optimistic from forward run)
    sota = parse_sota(sota_line)
    if sota:
        upsert_metrics_csv(str(metrics_csv_path("_sota")), [
            dataset, "StruProKGR",
            f"{sota.get('mrr', 0):.5f}", "—", "—",
            f"{sota.get('h1', 0):.5f}",  "—", "—",
            f"{sota.get('h3', 0):.5f}",  "—", "—",
            f"{sota.get('h10', 0):.5f}", "—", "—",
            "—",
        ])

    print_result(timestamp(), "struprokgr", "StruProKGR", dataset, log_file, None, final_line, seconds, "ok")


def main():
    parser = argparse.ArgumentParser(description="Run StruProKGR baseline over SparseKGC datasets")
    parser.add_argument("--datasets", nargs="+", default=SUPPORTED_DATASETS)
    parser.add_argument("--data-root",  default=str(REPO_ROOT / "datasets"))
    parser.add_argument("--work-root",  default=str(SCRIPT_DIR / "work"))
    parser.add_argument("--max-num-programs",  type=int, default=STRUPRO_DEFAULTS["max_num_programs"],
                        dest="max_num_programs")
    parser.add_argument("--max-path-len",      type=int, default=STRUPRO_DEFAULTS["max_path_len"],
                        dest="max_path_len")
    parser.add_argument("--max-path-branch",   type=int, default=STRUPRO_DEFAULTS["max_path_branch"],
                        dest="max_path_branch")
    parser.add_argument("--diminishing-factor", type=float, default=STRUPRO_DEFAULTS["diminishing_factor"],
                        dest="diminishing_factor")
    parser.add_argument("--decay-factor",       type=float, default=STRUPRO_DEFAULTS["decay_factor"],
                        dest="decay_factor")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing them.")
    args = parser.parse_args()

    unsupported = [d for d in args.datasets if d not in SUPPORTED_DATASETS]
    if unsupported:
        print(f"WARNING: StruProKGR does not support {unsupported}; they will be skipped.", flush=True)
        args.datasets = [d for d in args.datasets if d in SUPPORTED_DATASETS]

    failed = []
    for dataset in args.datasets:
        try:
            run_one(dataset, args)
        except Exception as exc:
            print(f"FAILED | baseline=StruProKGR | dataset={dataset} | {exc}", flush=True)
            import traceback; traceback.print_exc()
            failed.append(dataset)

    if failed:
        print(f"StruProKGR finished with {len(failed)} failed dataset(s): {failed}", flush=True)


if __name__ == "__main__":
    main()

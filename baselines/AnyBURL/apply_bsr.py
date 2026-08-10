#!/usr/bin/env python3
"""
Generate AnyBURL top-K predictions for the BSR routing-export, on BOTH the valid
and test splits, reusing the rules already learned by run_anyburl.py.

The main pipeline (run_anyburl.py) only predicts the test split with TOP_K=100
(for the metrics CSV). The BSR export wants up to top-200 candidates per query on
valid AND test, so this script re-runs AnyBURL `Apply` (rules are unchanged) with
TOP_K_OUTPUT=200, writing to dedicated files so the metrics predictions-100 file
is left untouched:

    work/<dataset>/predictions-bsr-valid
    work/<dataset>/predictions-bsr-test

AnyBURL's candidate list for a query (h, r, ?) depends only on (h, r) and the
rules, so valid/test predictions are consistent for shared queries; we still emit
both files so the export has full coverage of each split's triples.
"""
import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

_JDK_ARCH = "aarch64" if platform.machine() == "aarch64" else "x86"
DEFAULT_JAVA = str(REPO_ROOT / "tools" / f"jdk-21.0.11+10-{_JDK_ARCH}" / "bin" / "java")
DEFAULT_JAR = str(SCRIPT_DIR / "AnyBURL-23-1x.jar")
DATASETS = ["WD-singer", "FB15K-237-10", "WN18RR",
            "FB15K-237-20", "FB15K-237-50", "NELL23K"]


def run_java(java, jar, mainclass, config, cwd, xmx):
    cmd = [java, f"-Xmx{xmx}", "-cp", jar, mainclass, config]
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    proc.wait()
    return proc.returncode


def apply_split(work_dir, rules_file, split, top_k, threads, java, jar, xmx):
    pred_file = work_dir / f"predictions-bsr-{split}"
    cfg = work_dir / f"config-apply-bsr-{split}.properties"
    cfg.write_text(
        f"PATH_TRAINING = {work_dir / 'train.txt'}\n"
        f"PATH_VALID    = {work_dir / 'valid.txt'}\n"
        f"PATH_TEST     = {work_dir / (split + '.txt')}\n"
        f"PATH_RULES    = {rules_file}\n"
        f"PATH_OUTPUT   = {pred_file}\n"
        f"TOP_K_OUTPUT  = {top_k}\n"
        f"WORKER_THREADS = {threads}\n"
    )
    rc = run_java(java, jar, "de.unima.ki.anyburl.Apply", cfg.name, work_dir, xmx)
    if not pred_file.exists() or pred_file.stat().st_size == 0:
        raise RuntimeError(f"AnyBURL Apply produced no predictions (exit {rc}): {pred_file}")
    return pred_file


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument(
        "--work-root",
        default=str(REPO_ROOT / "outputs" / "preprocessed" / "anyburl"),
    )
    ap.add_argument("--splits", nargs="+", default=["valid", "test"], choices=["valid", "test"])
    ap.add_argument("--learn-time", type=int, default=100, help="rules-<learn_time> to reuse")
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--java", default=os.environ.get("ANYBURL_JAVA", DEFAULT_JAVA))
    ap.add_argument("--jar", default=os.environ.get("ANYBURL_JAR", DEFAULT_JAR))
    ap.add_argument("--xmx", default="12G")
    args = ap.parse_args()

    failed = []
    for ds in args.datasets:
        work_dir = Path(args.work_root) / ds
        rules_file = work_dir / f"rules-{args.learn_time}"
        if not rules_file.exists():
            print(f"FAILED | {ds}: rules file missing ({rules_file}); run run_anyburl.py first", flush=True)
            failed.append(ds)
            continue
        for split in args.splits:
            if not (work_dir / f"{split}.txt").exists():
                print(f"FAILED | {ds}/{split}: {split}.txt missing", flush=True)
                failed.append(f"{ds}/{split}")
                continue
            print(f"==== AnyBURL Apply (BSR top-{args.top_k}): {ds} / {split} ====", flush=True)
            try:
                pf = apply_split(work_dir, rules_file, split, args.top_k,
                                 args.threads, args.java, args.jar, args.xmx)
                print(f"  -> {pf} ({pf.stat().st_size} bytes)", flush=True)
            except RuntimeError as exc:
                print(f"FAILED | {ds}/{split}: {exc}", flush=True)
                failed.append(f"{ds}/{split}")
    if failed:
        print(f"\napply_bsr finished with {len(failed)} failure(s): {failed}", flush=True)
        sys.exit(1)
    print("\napply_bsr: all done.", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run the AnyBURL baseline end-to-end for the SparseKGC datasets:
prepare data -> Learn rules -> Apply (predict) -> score under the SparseKGC
protocol -> upsert outputs/anyburl_metrics.csv.

AnyBURL is a CPU-only Java tool; this orchestrator shells out to `java`.
Per-dataset failures are isolated (logged + skipped) so one bad dataset does
not abort the rest, matching the other one-shot SparseKGC runners.
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from log_format import print_start, print_result  # noqa: E402
sys.path.insert(0, str(SCRIPT_DIR))
from prepare_data import convert_file  # noqa: E402
from score_anyburl import evaluate  # noqa: E402
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from metrics_csv import upsert_metrics_csv  # noqa: E402

DATASETS = ["WD-singer", "FB15K-237-10", "WN18RR",
            "FB15K-237-20", "FB15K-237-50", "NELL23K"]

_JDK_ARCH = "aarch64" if platform.machine() == "aarch64" else "x86"
_BUNDLED_JAVA = REPO_ROOT / "tools" / f"jdk-21.0.11+10-{_JDK_ARCH}" / "bin" / "java"
DEFAULT_JAVA = str(_BUNDLED_JAVA) if _BUNDLED_JAVA.exists() else (shutil.which("java") or "java")
DEFAULT_JAR = str(SCRIPT_DIR / "AnyBURL-23-1x.jar")


def timestamp():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S") + f",{now.microsecond // 1000:03d}"


def prepare_dataset(data_root, work_dir, dataset):
    src_dir = Path(data_root) / dataset
    work_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        convert_file(src_dir / f"{split}.txt", work_dir / f"{split}.txt")


def write_configs(work_dir, learn_time, top_k, threads, seed):
    rules_prefix = work_dir / "rules"
    rules_file = work_dir / f"rules-{learn_time}"
    pred_file = work_dir / f"predictions-{learn_time}"
    (work_dir / "config-learn.properties").write_text(
        f"PATH_TRAINING = {work_dir / 'train.txt'}\n"
        f"PATH_OUTPUT   = {rules_prefix}\n"
        f"SNAPSHOTS_AT  = {learn_time}\n"
        f"WORKER_THREADS = {threads}\n"
        f"SEED = {seed}\n"
    )
    (work_dir / "config-apply.properties").write_text(
        f"PATH_TRAINING = {work_dir / 'train.txt'}\n"
        f"PATH_VALID    = {work_dir / 'valid.txt'}\n"
        f"PATH_TEST     = {work_dir / 'test.txt'}\n"
        f"PATH_RULES    = {rules_file}\n"
        f"PATH_OUTPUT   = {pred_file}\n"
        f"TOP_K_OUTPUT  = {top_k}\n"
        f"WORKER_THREADS = {threads}\n"
    )
    return rules_file, pred_file


def run_java(java, jar, mainclass, config, cwd, xmx, log_fh):
    """Run an AnyBURL Java class; return its exit code (do NOT raise on it).

    AnyBURL's time-based Learn interrupts its worker threads when the snapshot
    time is reached, which makes the JVM return a non-zero exit code even though
    the rules were written successfully. So success is judged by the caller from
    the produced output file, not from this return code.
    """
    cmd = [java, f"-Xmx{xmx}", "-cp", jar, mainclass, config]
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        log_fh.write(line)
        log_fh.flush()
    proc.wait()
    return proc.returncode


def metrics_csv_path():
    root = os.environ.get("SPARSEKGC_OUTPUT_DIR")
    base = Path(root) if root else (REPO_ROOT / "outputs")
    base.mkdir(parents=True, exist_ok=True)
    return base / "anyburl_metrics.csv"


def baseline_log_dir():
    root = os.environ.get("SPARSEKGC_OUTPUT_DIR")
    d = (Path(root) / "anyburl") if root else (REPO_ROOT / "outputs" / "anyburl")
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_one(dataset, args):
    work_dir = Path(args.work_root) / dataset
    prepare_dataset(args.data_root, work_dir, dataset)
    rules_file, pred_file = write_configs(work_dir, args.learn_time, args.top_k,
                                          args.threads, args.seed)
    params = f"learn_time={args.learn_time} top_k={args.top_k} seed={args.seed} threads={args.threads}"
    print_start(timestamp(), "anyburl", "AnyBURL", dataset, params)

    log_file = baseline_log_dir() / f"AnyBURL_{dataset}.log"
    start = time.perf_counter()
    with log_file.open("w", buffering=1) as fh:
        # Learn: AnyBURL exits non-zero on its normal time-based stop, so judge
        # success by the rules file being written rather than the exit code.
        run_java(args.java, args.jar, "de.unima.ki.anyburl.Learn",
                 "config-learn.properties", work_dir, args.xmx, fh)
        if not rules_file.exists() or rules_file.stat().st_size == 0:
            raise RuntimeError(f"AnyBURL Learn produced no rules file: {rules_file}")
        rc = run_java(args.java, args.jar, "de.unima.ki.anyburl.Apply",
                      "config-apply.properties", work_dir, args.xmx, fh)
        if not pred_file.exists() or pred_file.stat().st_size == 0:
            raise RuntimeError(f"AnyBURL Apply produced no predictions (exit {rc}): {pred_file}")
    seconds = time.perf_counter() - start

    res = evaluate(str(pred_file), str(work_dir))
    final_line = (
        "FINAL_EVAL_METRICS baseline=anyburl model=AnyBURL dataset={} split=test "
        "mrr_tail={:.5f} mrr_head={:.5f} mrr_avg={:.5f} "
        "h1_tail={:.5f} h1_head={:.5f} h1_avg={:.5f} "
        "h3_tail={:.5f} h3_head={:.5f} h3_avg={:.5f} "
        "h10_tail={:.5f} h10_head={:.5f} h10_avg={:.5f}".format(
            dataset,
            res["tail"]["mrr"], res["head"]["mrr"], res["avg"]["mrr"],
            res["tail"]["h1"], res["head"]["h1"], res["avg"]["h1"],
            res["tail"]["h3"], res["head"]["h3"], res["avg"]["h3"],
            res["tail"]["h10"], res["head"]["h10"], res["avg"]["h10"],
        )
    )
    upsert_metrics_csv(str(metrics_csv_path()), [
        dataset, "AnyBURL",
        f"{res['tail']['mrr']:.5f}", f"{res['head']['mrr']:.5f}", f"{res['avg']['mrr']:.5f}",
        f"{res['tail']['h1']:.5f}", f"{res['head']['h1']:.5f}", f"{res['avg']['h1']:.5f}",
        f"{res['tail']['h3']:.5f}", f"{res['head']['h3']:.5f}", f"{res['avg']['h3']:.5f}",
        f"{res['tail']['h10']:.5f}", f"{res['head']['h10']:.5f}", f"{res['avg']['h10']:.5f}",
        f"{seconds:.3f}",
    ])
    print_result(timestamp(), "anyburl", "AnyBURL", dataset, log_file, None, final_line, seconds, "ok")


def main():
    parser = argparse.ArgumentParser(description="Run AnyBURL baseline over SparseKGC datasets")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--data-root", default=str(REPO_ROOT / "datasets"))
    parser.add_argument(
        "--work-root",
        default=str(REPO_ROOT / "outputs" / "preprocessed" / "anyburl"),
    )
    parser.add_argument("--java", default=os.environ.get("ANYBURL_JAVA", DEFAULT_JAVA))
    parser.add_argument("--jar", default=os.environ.get("ANYBURL_JAR", DEFAULT_JAR))
    parser.add_argument("--learn-time", type=int, default=100, help="AnyBURL SNAPSHOTS_AT seconds")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--xmx", default="12G")
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        for dataset in args.datasets:
            print(
                "[dry_run] "
                f"dataset={dataset} data={Path(args.data_root) / dataset} "
                f"work={Path(args.work_root) / dataset} java={args.java} jar={args.jar}",
                flush=True,
            )
        return

    failed = []
    for dataset in args.datasets:
        try:
            run_one(dataset, args)
        except (subprocess.CalledProcessError, FileNotFoundError, RuntimeError) as exc:
            print(f"FAILED | baseline=AnyBURL | dataset={dataset} | {exc}", flush=True)
            failed.append(dataset)
    if failed:
        print(f"AnyBURL finished with {len(failed)} failed dataset(s): {failed}", flush=True)


if __name__ == "__main__":
    main()

#!/bin/bash
#SBATCH --job-name=skgc_anyburl
#SBATCH --output=slurm-%x-%j.log
#SBATCH --partition=gpu_long
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#
# AnyBURL baseline (rule mining). CPU-only Java tool.
# Runs on gpu_long (aarch64) for consistency with all other baselines.
# Uses aarch64 JDK from tools/jdk-21.0.11+10-aarch64/.
# Results upserted into outputs/anyburl_metrics.csv.
#
# Customize per-submission, e.g.:
#   sbatch --export=ALL,DATASETS="WD-singer FB15K-237-10" scripts/slurm/exp_anyburl.sh

source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

DATASETS="${DATASETS:-WD-singer FB15K-237-10 WN18RR FB15K-237-20 FB15K-237-50 NELL23K}"
read -r -a DATASET_ARGS <<< "${DATASETS}"

python -u run_baseline.py anyburl \
  --datasets "${DATASET_ARGS[@]}" \
  --anyburl_threads "${SLURM_CPUS_PER_TASK:-8}"

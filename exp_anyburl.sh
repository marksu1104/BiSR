#!/bin/bash
#SBATCH --job-name=skgc_anyburl
#SBATCH --output=/storage/professor/csliao/marksu/SparseKGC/logs/anyburl_%j.log
#SBATCH --partition=gpu_long
#SBATCH -w gpu1
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --account=csliao
#
# AnyBURL baseline (rule mining). CPU-only Java tool.
# Runs on gpu_long (aarch64) for consistency with all other baselines.
# Uses aarch64 JDK from tools/jdk-21.0.11+10-aarch64/.
# Results upserted into outputs/anyburl_metrics.csv.
#
# Customize per-submission, e.g.:
#   sbatch --export=ALL,DATASETS="WD-singer FB15K-237-10" exp_anyburl.sh

export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

source /storage/professor/csliao/marksu/SparseKGC/.venv/bin/activate

cd /storage/professor/csliao/marksu/SparseKGC
mkdir -p logs outputs

DATASETS="${DATASETS:-WD-singer FB15K-237-10 WN18RR FB15K-237-20 FB15K-237-50 NELL23K}"

python -u run_baseline.py anyburl \
  --datasets $DATASETS \
  --anyburl_threads "${SLURM_CPUS_PER_TASK:-8}"

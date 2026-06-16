#!/bin/bash
#SBATCH --job-name=skgc_anyburl
#SBATCH --output=/storage/professor/csliao/marksu/SparseKGC/logs/anyburl_%j.log
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --account=csliao
#
# AnyBURL baseline (rule mining). CPU-only Java tool -- runs on the CPU `short`
# partition, no GPU. Learns rules then applies them, scoring under the SparseKGC
# filtered/tie-aware/bidirectional protocol. Results upserted into
# outputs/anyburl_metrics.csv keyed by (Dataset, Model).
#
# Customize per-submission, e.g.:
#   sbatch --export=ALL,DATASETS="WD-singer FB15K-237-10" exp_anyburl.sh

export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

# NOTE: the project .venv is aarch64 (GH200-only); the CPU `short` partition is
# x86_64. AnyBURL's orchestrator is pure-stdlib Python (no torch/numpy), so use
# the system python3 here rather than the architecture-incompatible venv.
cd /storage/professor/csliao/marksu/SparseKGC
mkdir -p logs outputs
export SPARSEKGC_OUTPUT_DIR="/storage/professor/csliao/marksu/SparseKGC/outputs"

DATASETS="${DATASETS:-WD-singer FB15K-237-10 WN18RR FB15K-237-20 FB15K-237-50 NELL23K}"

python3 -u baselines/AnyBURL/run_anyburl.py \
  --datasets $DATASETS \
  --threads "${SLURM_CPUS_PER_TASK:-8}"

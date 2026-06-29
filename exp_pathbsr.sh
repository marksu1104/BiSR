#!/bin/bash
#SBATCH --job-name=skgc_pathbsr
#SBATCH --output=/storage/professor/csliao/marksu/SparseKGC/logs/pathbsr_%j.log
#SBATCH --partition=gpu_short
#SBATCH -w gpu2
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=csliao
#
# PathBSR baseline (BM25-proxy CBR + multi-hop path mining).
# Results upserted into outputs/pathbsr_metrics.csv.
#
# Customize per-submission, e.g.:
#   sbatch --export=ALL,DATASETS="WD-singer FB15K-237-10" exp_pathbsr.sh

export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

source /storage/professor/csliao/marksu/SparseKGC/.venv/bin/activate

cd /storage/professor/csliao/marksu/SparseKGC
mkdir -p logs outputs

DATASETS="${DATASETS:-WD-singer FB15K-237-10 WN18RR FB15K-237-20 FB15K-237-50 NELL23K}"

# Use all allocated CPUs for parallel triple evaluation within each dataset
export PATHBSR_NUM_WORKERS=${SLURM_CPUS_PER_TASK:-8}

python -u run_baseline.py pathbsr \
  --datasets $DATASETS

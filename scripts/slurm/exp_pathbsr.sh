#!/bin/bash
#SBATCH --job-name=skgc_pathbsr
#SBATCH --output=slurm-%x-%j.log
#SBATCH --partition=gpu_short
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#
# PathBSR baseline (BM25-proxy CBR + multi-hop path mining).
# Results upserted into outputs/pathbsr_metrics.csv.
#
# Customize per-submission, e.g.:
#   sbatch --export=ALL,DATASETS="WD-singer FB15K-237-10" scripts/slurm/exp_pathbsr.sh

source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

DATASETS="${DATASETS:-WD-singer FB15K-237-10 WN18RR FB15K-237-20 FB15K-237-50 NELL23K}"
read -r -a DATASET_ARGS <<< "${DATASETS}"

# Use all allocated CPUs for parallel triple evaluation within each dataset
export PATHBSR_NUM_WORKERS="${SLURM_CPUS_PER_TASK:-8}"

python -u run_baseline.py pathbsr \
  --datasets "${DATASET_ARGS[@]}"

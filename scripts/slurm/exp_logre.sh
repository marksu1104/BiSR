#!/bin/bash
#SBATCH --job-name=skgc_logre
#SBATCH --output=slurm-%x-%j.log
#SBATCH --partition=gpu_long
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#
# LoGRe baseline. Runs on gpu_long (aarch64) — the GPU is used for the ansim
# matrix computation (torch.matmul), which is significantly faster than CPU numpy.
# Results into outputs/logre_metrics.csv (bidirectional, tie-aware)
# and outputs/logre_sota_metrics.csv (tail-only, optimistic).
#
# WN18RR is not supported by LoGRe (no entity types).
#   sbatch --export=ALL,DATASETS="WD-singer FB15K-237-10" scripts/slurm/exp_logre.sh

source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

DATASETS="${DATASETS:-WD-singer FB15K-237-10 FB15K-237-20 FB15K-237-50 NELL23K}"
read -r -a DATASET_ARGS <<< "${DATASETS}"

python -u run_baseline.py logre \
  --datasets "${DATASET_ARGS[@]}"

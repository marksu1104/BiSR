#!/bin/bash
#SBATCH --job-name=skgc_struprokgr
#SBATCH --output=slurm-%x-%j.log
#SBATCH --partition=gpu_long
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#
# StruProKGR baseline (pure numpy/CPU). Runs on gpu_long (aarch64) for
# consistency with all other baselines. GPU is allocated but not used by
# StruProKGR itself (only CPU cores are utilized).
# Results into outputs/struprokgr_metrics.csv (bidirectional, tie-aware)
# and outputs/struprokgr_sota_metrics.csv (tail-only, optimistic).
#
# WN18RR is not supported by StruProKGR (no entity types).
#   sbatch --export=ALL,DATASETS="WD-singer FB15K-237-10" scripts/slurm/exp_struprokgr.sh

source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

DATASETS="${DATASETS:-WD-singer FB15K-237-10 FB15K-237-20 FB15K-237-50 NELL23K}"
read -r -a DATASET_ARGS <<< "${DATASETS}"

python -u run_baseline.py struprokgr \
  --datasets "${DATASET_ARGS[@]}"

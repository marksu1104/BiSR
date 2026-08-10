#!/bin/bash
#SBATCH --job-name=skgc_probcbr
#SBATCH --output=slurm-%x-%j.log
#SBATCH --partition=gpu_short
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#
# Prob-CBR baseline.
# Results are upserted into outputs/probcbr_metrics.csv keyed by
# (Dataset, Model), so reruns always keep only the latest row per combo.
#
# Customize per-submission by setting DATASETS env var, e.g.:
#   sbatch --export=ALL,DATASETS="WN18RR NELL23K" scripts/slurm/exp_probcbr.sh

source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

DATASETS="${DATASETS:-WD-singer FB15K-237-10 WN18RR FB15K-237-20 FB15K-237-50 NELL23K}"
read -r -a DATASET_ARGS <<< "${DATASETS}"

python -u run_baseline.py probcbr --gpu 0 \
  --datasets "${DATASET_ARGS[@]}"

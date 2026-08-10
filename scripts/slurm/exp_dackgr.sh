#!/bin/bash
#SBATCH --job-name=skgc_dackgr
#SBATCH --output=slurm-%x-%j.log
#SBATCH --partition=gpu_long
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#
# DacKGR baseline.
# Results are upserted into outputs/dackgr_metrics.csv keyed by
# (Dataset, Model), so reruns always keep only the latest row per combo.
#
# Customize per-submission by setting DATASETS env var, e.g.:
#   sbatch --export=ALL,DATASETS="WD-singer FB15K-237-10" scripts/slurm/exp_dackgr.sh

source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

DATASETS="${DATASETS:-WD-singer FB15K-237-10 WN18RR FB15K-237-20 FB15K-237-50 NELL23K}"
read -r -a DATASET_ARGS <<< "${DATASETS}"

python -u run_baseline.py dackgr --gpu 0 \
  --datasets "${DATASET_ARGS[@]}"

#!/bin/bash
#SBATCH --job-name=skgc_traditional
#SBATCH --output=slurm-%x-%j.log
#SBATCH --partition=gpu_long
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#
# Traditional KGE baselines (TransE/RotatE/DistMult/ComplEx/ConvE/TuckER).
# Results are upserted into outputs/traditional_metrics.csv keyed by
# (Dataset, Model), so reruns always keep only the latest row per combo.
#
# A full 6-model x 6-dataset run takes ~13h, which exceeds the gpu_short 12h
# cap, so this defaults to gpu_long. For a quick subset you can override back
# to the faster short partition, e.g.:
#   sbatch --partition=gpu_short --export=ALL,DATASETS="WD-singer" scripts/slurm/exp_traditional.sh
#   sbatch --export=ALL,MODELS="ConvE TuckER" scripts/slurm/exp_traditional.sh

source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

DATASETS="${DATASETS:-WD-singer FB15K-237-10 WN18RR FB15K-237-20 FB15K-237-50 NELL23K}"
MODELS="${MODELS:-TransE RotatE DistMult ComplEx ConvE TuckER}"
read -r -a DATASET_ARGS <<< "${DATASETS}"
read -r -a MODEL_ARGS <<< "${MODELS}"

python -u run_baseline.py traditional --gpu 0 \
  --models "${MODEL_ARGS[@]}" \
  --datasets "${DATASET_ARGS[@]}"

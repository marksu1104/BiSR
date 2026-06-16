#!/bin/bash
#SBATCH --job-name=skgc_probcbr
#SBATCH --output=/storage/professor/csliao/marksu/SparseKGC/logs/probcbr_%j.log
#SBATCH --partition=gpu_short
#SBATCH -w gpu4
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --account=csliao
#
# Prob-CBR baseline.
# Results are upserted into outputs/probcbr_metrics.csv keyed by
# (Dataset, Model), so reruns always keep only the latest row per combo.
#
# Customize per-submission by setting DATASETS env var, e.g.:
#   sbatch --export=ALL,DATASETS="WN18RR NELL23K" exp_probcbr.sh

export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

source /storage/professor/csliao/marksu/SparseKGC/.venv/bin/activate
export PYTHON_BIN="$(which python)"

cd /storage/professor/csliao/marksu/SparseKGC
mkdir -p logs outputs

DATASETS="${DATASETS:-WD-singer FB15K-237-10 WN18RR FB15K-237-20 FB15K-237-50 NELL23K}"

python -u run_baseline.py probcbr --gpu 0 \
  --datasets $DATASETS

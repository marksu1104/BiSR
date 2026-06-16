#!/bin/bash
#SBATCH --job-name=skgc_hogrn
#SBATCH --output=/storage/professor/csliao/marksu/SparseKGC/logs/hogrn_%j.log
#SBATCH --partition=gpu_long
#SBATCH -w gpu1
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --account=csliao
#
# HoGRN baseline (conve score function, per run_baseline.py).
# Results are upserted into outputs/hogrn_metrics.csv keyed by
# (Dataset, Model), so reruns always keep only the latest row per combo.
#
# A full 6-dataset run takes ~13h, which exceeds the gpu_short 12h cap, so this
# defaults to gpu_long. For a quick subset you can override to the short
# partition, e.g.:
#   sbatch --partition=gpu_short -w gpu3 --export=ALL,DATASETS="FB15K-237-20 FB15K-237-50" exp_hogrn.sh

export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

source /storage/professor/csliao/marksu/.venv/bin/activate
export PYTHON_BIN="$(which python)"

cd /storage/professor/csliao/marksu/SparseKGC
mkdir -p logs outputs

DATASETS="${DATASETS:-WD-singer FB15K-237-10 WN18RR FB15K-237-20 FB15K-237-50 NELL23K}"

python -u run_baseline.py hogrn --gpu 0 \
  --datasets $DATASETS

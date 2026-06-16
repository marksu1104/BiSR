#!/bin/bash
#SBATCH --job-name=skgc_traditional
#SBATCH --output=/storage/professor/csliao/marksu/SparseKGC/logs/traditional_%j.log
#SBATCH --partition=gpu_long
#SBATCH -w gpu1
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --account=csliao
#
# Traditional KGE baselines (TransE/RotatE/DistMult/ComplEx/ConvE/TuckER).
# Results are upserted into outputs/traditional_metrics.csv keyed by
# (Dataset, Model), so reruns always keep only the latest row per combo.
#
# A full 6-model x 6-dataset run takes ~13h, which exceeds the gpu_short 12h
# cap, so this defaults to gpu_long. For a quick subset you can override back
# to the faster short partition, e.g.:
#   sbatch --partition=gpu_short -w gpu2 --export=ALL,DATASETS="WD-singer" exp_traditional.sh
#   sbatch --export=ALL,MODELS="ConvE TuckER" exp_traditional.sh

export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

source /storage/professor/csliao/marksu/SparseKGC/.venv/bin/activate
export PYTHON_BIN="$(which python)"

cd /storage/professor/csliao/marksu/SparseKGC
mkdir -p logs outputs

DATASETS="${DATASETS:-WD-singer FB15K-237-10 WN18RR FB15K-237-20 FB15K-237-50 NELL23K}"
MODELS="${MODELS:-TransE RotatE DistMult ComplEx ConvE TuckER}"

python -u run_baseline.py traditional --gpu 0 \
  --models $MODELS \
  --datasets $DATASETS

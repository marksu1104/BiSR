#!/bin/bash
#SBATCH --job-name=skgc_logre
#SBATCH --output=/storage/professor/csliao/marksu/SparseKGC/logs/logre_%j.log
#SBATCH --partition=gpu_long
#SBATCH -w gpu1
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=csliao
#
# LoGRe baseline. Runs on gpu_long (aarch64) — the GPU is used for the ansim
# matrix computation (torch.matmul), which is significantly faster than CPU numpy.
# Results into outputs/logre_metrics.csv (bidirectional, tie-aware)
# and outputs/logre_sota_metrics.csv (tail-only, optimistic).
#
# WN18RR is not supported by LoGRe (no entity types).
#   sbatch --export=ALL,DATASETS="WD-singer FB15K-237-10" exp_logre.sh

export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

source /storage/professor/csliao/marksu/SparseKGC/.venv/bin/activate

cd /storage/professor/csliao/marksu/SparseKGC
mkdir -p logs outputs

DATASETS="${DATASETS:-WD-singer FB15K-237-10 FB15K-237-20 FB15K-237-50 NELL23K}"

python -u run_baseline.py logre \
  --datasets $DATASETS

#!/bin/bash
#SBATCH --job-name=skgc_bsr
#SBATCH --output=/storage/professor/csliao/marksu/SparseKGC/logs/bsr_export_%j.log
#SBATCH --partition=gpu_short
#SBATCH -w gpu2
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --account=csliao
#
# Export BSR routing/hybrid-expert prediction CSVs from trained checkpoints
# (HoGRN, TransE, ConvE, TuckER) for the 6 sparse datasets (FB15K-237 full
# excluded). Inference only -- consumes existing checkpoints, writes CSVs to
# external_predictions/.

export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

source /storage/professor/csliao/marksu/SparseKGC/.venv/bin/activate
cd /storage/professor/csliao/marksu/SparseKGC
mkdir -p logs

python -u scripts/export_bsr_routing_predictions.py \
  --data-root datasets \
  --output-root external_predictions \
  --models HoGRN TransE ConvE TuckER AnyBURL \
  --datasets WD-singer FB15K-237-10 FB15K-237-20 FB15K-237-50 NELL23K WN18RR \
  --splits valid test \
  --top-k 200 \
  --seed 42 \
  --gpu 0

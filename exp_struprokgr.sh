#!/bin/bash
#SBATCH --job-name=skgc_struprokgr
#SBATCH --output=/storage/professor/csliao/marksu/SparseKGC/logs/struprokgr_%j.log
#SBATCH --partition=gpu_long
#SBATCH -w gpu1
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=csliao
#
# StruProKGR baseline (pure numpy/CPU). Runs on gpu_long (aarch64) for
# consistency with all other baselines. GPU is allocated but not used by
# StruProKGR itself (only CPU cores are utilized).
# Results into outputs/struprokgr_metrics.csv (bidirectional, tie-aware)
# and outputs/struprokgr_sota_metrics.csv (tail-only, optimistic).
#
# WN18RR is not supported by StruProKGR (no entity types).
#   sbatch --export=ALL,DATASETS="WD-singer FB15K-237-10" exp_struprokgr.sh

export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

source /storage/professor/csliao/marksu/SparseKGC/.venv/bin/activate

cd /storage/professor/csliao/marksu/SparseKGC
mkdir -p logs outputs

DATASETS="${DATASETS:-WD-singer FB15K-237-10 FB15K-237-20 FB15K-237-50 NELL23K}"

python -u run_baseline.py struprokgr \
  --datasets $DATASETS

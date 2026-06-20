#!/bin/bash
#SBATCH --job-name=tucker_grok
#SBATCH --output=/storage/professor/csliao/marksu/SparseKGC/logs/tucker_grok_%j.log
#SBATCH --partition=gpu_short
#SBATCH --time=11:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --account=csliao
#
# TuckER long-horizon grokking test on FB15K-237-10 (traditional codebase).
# wd=1e-6 groks early (~ep300) but caps at 0.22; stronger wd was still on the
# 0.05 plateau at ep1000. Grokking theory: stronger weight decay groks LATER
# but generalizes HIGHER -> give it 4000 epochs. WD passed via env.
# Target: tail 0.252 / avg 0.163.

export TMPDIR="${SLURM_TMPDIR:-/tmp}"; export TEMP="$TMPDIR"; export TMP="$TMPDIR"
source /storage/professor/csliao/marksu/SparseKGC/.venv/bin/activate
cd /storage/professor/csliao/marksu/SparseKGC/baselines/tranditional

DATA=/storage/professor/csliao/marksu/SparseKGC/datasets/FB15K-237-10
LOGDIR=/storage/professor/csliao/marksu/SparseKGC/logs
WD="${WD:-1e-5}"

echo "==================== ls + wd=${WD}, epochs=4000 ===================="
python -u main.py --model TuckER --dataset FB15K-237-10 --data_path "$DATA" \
    --emb_dim 200 --lr 0.0005 --l2 ${WD} --batch_size 512 --max_epochs 4000 --patience 4000 \
    --eval_freq 5 --seed 20 --selection_protocol sota --loss bce --ls_dackgr 1 --gpu 0 \
    2>&1 | tee "${LOGDIR}/tucker_grok_wd${WD}.log"

echo ""
echo "######## GROK RESULT wd=${WD} (target tail=0.252 avg=0.163) ########"
grep "FINAL_EVAL_METRICS" "${LOGDIR}/tucker_grok_wd${WD}.log" | tail -1
echo "--- peak valid ---"
grep -E "valid SOTA\]: MRR" "${LOGDIR}/tucker_grok_wd${WD}.log" | sed -E 's/.*MRR: ([0-9.]+).*/\1/' | sort -g | tail -1

#!/bin/bash
#SBATCH --job-name=tucker_combo
#SBATCH --output=/storage/professor/csliao/marksu/SparseKGC/logs/tucker_combo_%j.log
#SBATCH --partition=gpu_short
#SBATCH -w gpu4
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --account=csliao
#
# TuckER combined fix on FB15K-237-10 (traditional codebase).
# Independent gains found: label smoothing (eps/N) -> 0.215, weight decay 1e-6
# -> 0.219 (both +0.03 over the 0.189 baseline). Different mechanisms -> stack
# them. Sweep wd while holding ls_dackgr=1. lr 0.0005, batch 512, epochs 1000,
# no early stop, seed 20. Target: tail 0.252 / avg 0.163.

export TMPDIR="${SLURM_TMPDIR:-/tmp}"; export TEMP="$TMPDIR"; export TMP="$TMPDIR"
source /storage/professor/csliao/marksu/SparseKGC/.venv/bin/activate
cd /storage/professor/csliao/marksu/SparseKGC/baselines/tranditional

DATA=/storage/professor/csliao/marksu/SparseKGC/datasets/FB15K-237-10
LOGDIR=/storage/professor/csliao/marksu/SparseKGC/logs

for wd in 1e-6 1e-5 5e-5; do
    echo "==================== ls_dackgr=1 + wd=${wd} ===================="
    python -u main.py --model TuckER --dataset FB15K-237-10 --data_path "$DATA" \
        --emb_dim 200 --lr 0.0005 --l2 ${wd} --batch_size 512 --max_epochs 1000 --patience 1000 \
        --eval_freq 1 --seed 20 --selection_protocol sota --loss bce --ls_dackgr 1 --gpu 0 \
        2>&1 | tee "${LOGDIR}/tucker_combo_wd${wd}.log"
done

echo ""
echo "######## COMBO SUMMARY (target tail=0.252 avg=0.163) ########"
for wd in 1e-6 1e-5 5e-5; do
    fin=$(grep "FINAL_EVAL_METRICS" "${LOGDIR}/tucker_combo_wd${wd}.log" | tail -1 | sed -E 's/.*(mrr_tail=[0-9.]+ mrr_head=[0-9.]+ mrr_avg=[0-9.]+).*/\1/')
    peak=$(grep -E "valid SOTA\]: MRR" "${LOGDIR}/tucker_combo_wd${wd}.log" | sed -E 's/.*MRR: ([0-9.]+).*/\1/' | sort -g | tail -1)
    echo "ls+wd=${wd}: ${fin} | peak_valid=${peak}"
done

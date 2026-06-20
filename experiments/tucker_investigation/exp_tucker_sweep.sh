#!/bin/bash
#SBATCH --job-name=tucker_sweep
#SBATCH --output=/storage/professor/csliao/marksu/SparseKGC/logs/tucker_sweep_%j.log
#SBATCH --partition=gpu_short
#SBATCH -w gpu3
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --account=csliao
#
# TuckER relation-dim sweep on FB15K-237-10 (traditional codebase).
# Goal: find d_r that lifts TuckER out of the sparse-data overfit.
# Target (StruProKGR/DacKGR paper): tail MRR 0.252 / bidir-avg 0.163.
# Entity dim fixed at 200; all other settings match the existing TuckER
# config (lr 0.0005, batch 128, bce, sota selection). Only d_r varies.

export TMPDIR="${SLURM_TMPDIR:-/tmp}"; export TEMP="$TMPDIR"; export TMP="$TMPDIR"
source /storage/professor/csliao/marksu/SparseKGC/.venv/bin/activate
cd /storage/professor/csliao/marksu/SparseKGC/baselines/tranditional

DATA=/storage/professor/csliao/marksu/SparseKGC/datasets/FB15K-237-10
LOGDIR=/storage/professor/csliao/marksu/SparseKGC/logs

for rd in 30 50 100 200; do
    echo "==================== tucker_rel_dim=${rd} ===================="
    python -u main.py --model TuckER --dataset FB15K-237-10 --data_path "$DATA" \
        --emb_dim 200 --tucker_rel_dim ${rd} \
        --lr 0.0005 --batch_size 128 --max_epochs 500 --patience 25 \
        --eval_freq 1 --seed 20 --selection_protocol sota --loss bce --gpu 0 \
        2>&1 | tee "${LOGDIR}/tucker_sweep_rd${rd}.log"
done

echo ""
echo "######## SWEEP SUMMARY (FB10, target tail=0.252 avg=0.163) ########"
for rd in 30 50 100 200; do
    line=$(grep "FINAL_EVAL_METRICS" "${LOGDIR}/tucker_sweep_rd${rd}.log" | tail -1)
    echo "d_r=${rd}: ${line}"
done

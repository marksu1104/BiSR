#!/bin/bash
#SBATCH --job-name=tucker_sweep2
#SBATCH --output=/storage/professor/csliao/marksu/SparseKGC/logs/tucker_sweep2_%j.log
#SBATCH --partition=gpu_short
#SBATCH -w gpu3
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --account=csliao
#
# TuckER batch/epochs sweep on FB15K-237-10 (traditional codebase).
# The d_r sweep showed dimension is NOT the lever (d_r=200 best at 0.189).
# DacKGR's identical 200/200 TuckER reaches 0.253 with batch=512/epochs=1000,
# so the gap is training budget. Isolate batch vs epochs here. d_r stays 200.
# Target (paper): tail 0.252 / bidir-avg 0.163.

export TMPDIR="${SLURM_TMPDIR:-/tmp}"; export TEMP="$TMPDIR"; export TMP="$TMPDIR"
source /storage/professor/csliao/marksu/SparseKGC/.venv/bin/activate
cd /storage/professor/csliao/marksu/SparseKGC/baselines/tranditional

DATA=/storage/professor/csliao/marksu/SparseKGC/datasets/FB15K-237-10
LOGDIR=/storage/professor/csliao/marksu/SparseKGC/logs

# tag:batch  (epochs fixed 1000, patience 40 so early-stop doesn't cut it short)
for bs in 128 256 512; do
    echo "==================== batch=${bs} epochs=1000 ===================="
    python -u main.py --model TuckER --dataset FB15K-237-10 --data_path "$DATA" \
        --emb_dim 200 --lr 0.0005 --batch_size ${bs} --max_epochs 1000 --patience 40 \
        --eval_freq 1 --seed 20 --selection_protocol sota --loss bce --gpu 0 \
        2>&1 | tee "${LOGDIR}/tucker_sweep2_bs${bs}.log"
done

echo ""
echo "######## SWEEP2 SUMMARY (FB10, target tail=0.252 avg=0.163) ########"
for bs in 128 256 512; do
    line=$(grep "FINAL_EVAL_METRICS" "${LOGDIR}/tucker_sweep2_bs${bs}.log" | tail -1)
    echo "batch=${bs}: ${line}"
done

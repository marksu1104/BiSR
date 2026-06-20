#!/bin/bash
#SBATCH --job-name=tucker_wd
#SBATCH --output=/storage/professor/csliao/marksu/SparseKGC/logs/tucker_wd_%j.log
#SBATCH --partition=gpu_short
#SBATCH -w gpu2
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --account=csliao
#
# TuckER weight-decay sweep on FB15K-237-10 (traditional codebase).
# Diagnosis: identical recipe, but RNG-consumption order differs between
# codebases -> different W init -> traditional lands in an overfit basin
# (valid peaks 0.18 then declines) while DacKGR grokked to 0.253. Weight decay
# is the canonical driver of grokking / delayed generalization (Power et al.
# 2022): it should push ANY init from the memorization solution toward the
# generalizing one, independent of basin. lr 0.0005, batch 512, epochs 1000,
# no early stop, best-valid. Target: tail 0.252 / avg 0.163.

export TMPDIR="${SLURM_TMPDIR:-/tmp}"; export TEMP="$TMPDIR"; export TMP="$TMPDIR"
source /storage/professor/csliao/marksu/SparseKGC/.venv/bin/activate
cd /storage/professor/csliao/marksu/SparseKGC/baselines/tranditional

DATA=/storage/professor/csliao/marksu/SparseKGC/datasets/FB15K-237-10
LOGDIR=/storage/professor/csliao/marksu/SparseKGC/logs

for wd in 1e-6 1e-5 1e-4; do
    echo "==================== weight_decay=${wd} ===================="
    python -u main.py --model TuckER --dataset FB15K-237-10 --data_path "$DATA" \
        --emb_dim 200 --lr 0.0005 --l2 ${wd} --batch_size 512 --max_epochs 1000 --patience 1000 \
        --eval_freq 1 --seed 20 --selection_protocol sota --loss bce --gpu 0 \
        2>&1 | tee "${LOGDIR}/tucker_wd_${wd}.log"
done

echo ""
echo "######## WD SWEEP SUMMARY (target tail=0.252 avg=0.163) ########"
for wd in 1e-6 1e-5 1e-4; do
    fin=$(grep "FINAL_EVAL_METRICS" "${LOGDIR}/tucker_wd_${wd}.log" | tail -1 | sed -E 's/.*(mrr_tail=[0-9.]+ mrr_head=[0-9.]+ mrr_avg=[0-9.]+).*/\1/')
    peak=$(grep -E "valid SOTA\]: MRR" "${LOGDIR}/tucker_wd_${wd}.log" | sed -E 's/.*MRR: ([0-9.]+).*/\1/' | sort -g | tail -1)
    echo "wd=${wd}: ${fin} | peak_valid=${peak}"
done

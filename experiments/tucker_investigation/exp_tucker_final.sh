#!/bin/bash
#SBATCH --job-name=tucker_final
#SBATCH --output=/storage/professor/csliao/marksu/SparseKGC/logs/tucker_final_%j.log
#SBATCH --partition=gpu_short
#SBATCH -w gpu3
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --account=csliao
#
# Decisive TuckER test on FB15K-237-10 (traditional codebase).
# Diagnosis: TuckER groks (delayed generalization) on sparse data; our
# early-stopping cut it off mid-grok at ~0.18. DacKGR's identical model trained
# the full 1000 epochs and grokked to 0.253. Here: batch 512, epochs 1000, NO
# early stopping (patience=1000), best-valid checkpoint over the whole run.
# Target: tail 0.252 / avg 0.163.

export TMPDIR="${SLURM_TMPDIR:-/tmp}"; export TEMP="$TMPDIR"; export TMP="$TMPDIR"
source /storage/professor/csliao/marksu/SparseKGC/.venv/bin/activate
cd /storage/professor/csliao/marksu/SparseKGC/baselines/tranditional

DATA=/storage/professor/csliao/marksu/SparseKGC/datasets/FB15K-237-10
LOGDIR=/storage/professor/csliao/marksu/SparseKGC/logs

python -u main.py --model TuckER --dataset FB15K-237-10 --data_path "$DATA" \
    --emb_dim 200 --lr 0.0005 --batch_size 512 --max_epochs 1000 --patience 1000 \
    --eval_freq 1 --seed 20 --selection_protocol sota --loss bce --gpu 0 \
    2>&1 | tee "${LOGDIR}/tucker_final_run.log"

echo ""
echo "######## RESULT (target tail=0.252 avg=0.163) ########"
grep "FINAL_EVAL_METRICS" "${LOGDIR}/tucker_final_run.log" | tail -1
echo "--- best valid reached ---"
grep -E "valid SOTA\]: MRR" "${LOGDIR}/tucker_final_run.log" | sort -t: -k2 -g | tail -1

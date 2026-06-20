#!/bin/bash
#SBATCH --job-name=tucker_ed
#SBATCH --output=/storage/professor/csliao/marksu/SparseKGC/logs/tucker_ed_%j.log
#SBATCH --partition=gpu_short
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --account=csliao
#
# TuckER embedding-dropout fix on FB15K-237-10 (traditional codebase).
# ROOT CAUSE FOUND: DacKGR wraps every embedding lookup (entity input,
# relation, AND output-projection matrix) in Dropout(emb_dropout_rate=0.3) via
# EDropout/RDropout. The reimplementation omitted this. That extra regularization
# is what keeps TuckER on the plateau long enough to grok to ~0.25 instead of
# overfitting at ~0.22. Match DacKGR: emb_drop 0.3, lr 0.0005, batch 512, wd 0,
# long training. LS passed via env (1 = DacKGR eps/N smoothing). Target: 0.252.

export TMPDIR="${SLURM_TMPDIR:-/tmp}"; export TEMP="$TMPDIR"; export TMP="$TMPDIR"
source /storage/professor/csliao/marksu/SparseKGC/.venv/bin/activate
cd /storage/professor/csliao/marksu/SparseKGC/baselines/tranditional

DATA=/storage/professor/csliao/marksu/SparseKGC/datasets/FB15K-237-10
LOGDIR=/storage/professor/csliao/marksu/SparseKGC/logs
LS="${LS:-1}"

echo "==================== emb_drop=0.3 ls_dackgr=${LS} ===================="
python -u main.py --model TuckER --dataset FB15K-237-10 --data_path "$DATA" \
    --emb_dim 200 --tucker_emb_drop 0.3 --lr 0.0005 --batch_size 512 \
    --max_epochs 1500 --patience 1500 --eval_freq 5 --seed 20 \
    --selection_protocol sota --loss bce --ls_dackgr ${LS} --gpu 0 \
    2>&1 | tee "${LOGDIR}/tucker_ed_ls${LS}.log"

echo ""
echo "######## RESULT emb_drop=0.3 ls=${LS} (target tail=0.252 avg=0.163) ########"
grep "FINAL_EVAL_METRICS" "${LOGDIR}/tucker_ed_ls${LS}.log" | tail -1
echo "--- peak valid ---"
grep -E "valid SOTA\]: MRR" "${LOGDIR}/tucker_ed_ls${LS}.log" | sed -E 's/.*MRR: ([0-9.]+).*/\1/' | sort -g | tail -1

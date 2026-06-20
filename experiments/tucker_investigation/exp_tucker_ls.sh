#!/bin/bash
#SBATCH --job-name=tucker_ls
#SBATCH --output=/storage/professor/csliao/marksu/SparseKGC/logs/tucker_ls_%j.log
#SBATCH --partition=gpu_short
#SBATCH -w gpu4
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --account=csliao
#
# TuckER label-smoothing test on FB15K-237-10 (traditional codebase).
# The ONLY confirmed structural diff vs DacKGR: traditional adds 1/N to every
# negative (total negative mass ~1.0), DacKGR adds eps/N (~0.1) -> 10x weaker
# negative regularization. Test DacKGR's variant (ls_dackgr=1). All else equal:
# lr 0.0005, batch 512, 1000 epochs, no early stop, seed 20.
# Target: tail 0.252 / avg 0.163.

export TMPDIR="${SLURM_TMPDIR:-/tmp}"; export TEMP="$TMPDIR"; export TMP="$TMPDIR"
source /storage/professor/csliao/marksu/SparseKGC/.venv/bin/activate
cd /storage/professor/csliao/marksu/SparseKGC/baselines/tranditional

DATA=/storage/professor/csliao/marksu/SparseKGC/datasets/FB15K-237-10
LOGDIR=/storage/professor/csliao/marksu/SparseKGC/logs

echo "==================== ls_dackgr=1 (eps/N) ===================="
python -u main.py --model TuckER --dataset FB15K-237-10 --data_path "$DATA" \
    --emb_dim 200 --lr 0.0005 --batch_size 512 --max_epochs 1000 --patience 1000 \
    --eval_freq 1 --seed 20 --selection_protocol sota --loss bce --ls_dackgr 1 --gpu 0 \
    2>&1 | tee "${LOGDIR}/tucker_ls_dackgr.log"

echo ""
echo "######## LS TEST (target tail=0.252 avg=0.163) ########"
grep "FINAL_EVAL_METRICS" "${LOGDIR}/tucker_ls_dackgr.log" | tail -1
echo "--- peak valid ---"
grep -E "valid SOTA\]: MRR" "${LOGDIR}/tucker_ls_dackgr.log" | sed -E 's/.*MRR: ([0-9.]+).*/\1/' | sort -g | tail -1

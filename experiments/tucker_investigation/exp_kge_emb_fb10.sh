#!/bin/bash
#SBATCH --job-name=kge_emb_fb10
#SBATCH --output=/storage/professor/csliao/marksu/SparseKGC/logs/kge_emb_fb10_%j.log
#SBATCH --partition=gpu_long
#SBATCH -w gpu1
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --account=csliao
#
# Validation run: reproduce the paper's SOTA-table KGE baselines (TransE,
# DistMult, ConvE, TuckER) on FB15K-237-10 through the DacKGR embedding
# pipeline -- the framework the DacKGR paper used to produce those numbers.
# TransE uses classic margin-ranking (the '!TransE' guard in emb.py was the bug
# that forced BCE); ConvE/TuckER/DistMult use 1-N (KvsAll) BCE.
#
# Targets (DacKGR/StruProKGR paper, MRR_Tail on FB10):
#   TransE 0.105   DistMult 0.074   ConvE 0.245   TuckER 0.252

export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export TEMP="$TMPDIR"; export TMP="$TMPDIR"

source /storage/professor/csliao/marksu/SparseKGC/.venv/bin/activate
export PYTHON_BIN="$(which python)"
export DACKGR_WRITE_METRICS=0   # validation only; parse logs, don't touch CSVs

cd /storage/professor/csliao/marksu/SparseKGC/baselines/DacKGR

LOGDIR=/storage/professor/csliao/marksu/SparseKGC/logs
for cfg in transe distmult conve tucker; do
    echo "==================== $cfg ===================="
    bash experiment-emb.sh "configs/fb15k-237-10-${cfg}.sh" --train 0 \
        2>&1 | tee "${LOGDIR}/kge_emb_fb10_${cfg}.log"
done

echo ""
echo "######## SUMMARY (FB10 tail MRR) ########"
for cfg in transe distmult conve tucker; do
    line=$(grep "MRR: Tail" "${LOGDIR}/kge_emb_fb10_${cfg}.log" | tail -1)
    echo "${cfg}: ${line}"
done

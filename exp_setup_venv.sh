#!/bin/bash
#SBATCH --job-name=setup_venv
#SBATCH --output=/storage/professor/csliao/marksu/SparseKGC/logs/setup_venv_%j.log
#SBATCH --partition=gpu_short
#SBATCH -w gpu2
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=csliao
#
# One-time setup: (re)create SparseKGC/.venv/ on an aarch64 GPU node from the
# pinned requirements.txt. Run on a GPU node so torch can see CUDA.
#
# NOTE: requirements.txt pins exact versions but NOT the package index. On
# aarch64 + CUDA, torch / torch_scatter may need a matching wheel index, e.g.:
#   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
#   pip install torch_scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
# If the bulk install below fails on those two, install them manually first,
# then re-run. After (re)creating .venv, refresh the pins with:
#   .venv/bin/pip freeze > requirements.txt

set -e
REPO=/storage/professor/csliao/marksu/SparseKGC

python3 -m venv "$REPO/.venv"
"$REPO/.venv/bin/pip" install --quiet --upgrade pip
"$REPO/.venv/bin/pip" install --quiet -r "$REPO/requirements.txt"

echo ""
echo "=== Verification ==="
"$REPO/.venv/bin/python" -c "import torch, numpy, tqdm, scipy, torch_scatter; print('torch', torch.__version__, '/ cuda:', torch.cuda.is_available(), '/ numpy', numpy.__version__)"
echo "Setup complete: $REPO/.venv"

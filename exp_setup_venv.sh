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
# One-time setup: create SparseKGC/.venv/ on aarch64 GPU node.
# After this completes, GPU sbatch scripts will use SparseKGC/.venv instead of ~/marksu/.venv.

set -e
REPO=/storage/professor/csliao/marksu/SparseKGC
OLD_VENV=/storage/professor/csliao/marksu/.venv

# Create new venv
python3 -m venv "$REPO/.venv"
"$REPO/.venv/bin/pip" install --quiet --upgrade pip

# Install from old venv's pip freeze (handles all packages correctly)
"$OLD_VENV/bin/pip" freeze 2>/dev/null > /tmp/old_venv_freeze.txt
echo "Old venv packages:"
cat /tmp/old_venv_freeze.txt

"$REPO/.venv/bin/pip" install --quiet -r /tmp/old_venv_freeze.txt

echo ""
echo "=== Verification ==="
"$REPO/.venv/bin/python" -c "import torch, numpy, tqdm; print('torch', torch.__version__, '/ cuda:', torch.cuda.is_available(), '/ numpy', numpy.__version__)"
echo "Setup complete: $REPO/.venv"

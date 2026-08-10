#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-outputs}"
PYTHON="${PYTHON_BIN:-python3}"
ARCH="$(uname -m)"
VENV_DIR="${SPARSEKGC_VENV:-${REPO_ROOT}/.venv-${ARCH}}"

case "${MODE}" in
  outputs)
    REQUIREMENTS="${REPO_ROOT}/requirements-outputs.txt"
    ;;
  experiments)
    REQUIREMENTS="${REPO_ROOT}/requirements.txt"
    ;;
  *)
    echo "Usage: ./setup_env.sh [outputs|experiments]" >&2
    exit 2
    ;;
esac

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON}" -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r "${REQUIREMENTS}"

if [[ "${MODE}" == "outputs" ]]; then
  "${VENV_DIR}/bin/python" -c \
    "import matplotlib, numpy; print(f'matplotlib={matplotlib.__version__} numpy={numpy.__version__}')"
else
  "${VENV_DIR}/bin/python" -c \
    "import numpy, scipy, torch; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} numpy={numpy.__version__}')"
fi

echo "Environment ready: ${VENV_DIR}"

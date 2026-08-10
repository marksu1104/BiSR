#!/usr/bin/env bash
set -euo pipefail

SLURM_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SLURM_SCRIPT_DIR}/../.." && pwd)"
ARCH_VENV="${REPO_ROOT}/.venv-$(uname -m)"

if [[ -n "${SPARSEKGC_VENV:-}" ]]; then
  VENV_DIR="${SPARSEKGC_VENV}"
elif [[ -x "${ARCH_VENV}/bin/python" ]]; then
  VENV_DIR="${ARCH_VENV}"
else
  VENV_DIR="${REPO_ROOT}/.venv"
fi

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "Python environment not found: ${VENV_DIR}" >&2
  echo "Run ./setup_env.sh experiments before submitting jobs." >&2
  exit 1
fi

export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"

source "${VENV_DIR}/bin/activate"
export PYTHON_BIN="$(command -v python)"
cd "${REPO_ROOT}"
mkdir -p outputs

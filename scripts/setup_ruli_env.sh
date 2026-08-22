#!/usr/bin/env bash

set -euo pipefail

RULI_ROOT="${RULI_ROOT:-/workspace/Ruli}"
VENV_DIR="${RULI_VENV_DIR:-${RULI_ROOT}/.venv}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${RULI_PYTHON:-python3}"
TORCH_INDEX_URL="${RULI_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[ERROR] Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

BASE_TORCH="$("${PYTHON_BIN}" -c \
    'import torch; print(torch.__version__, torch.version.cuda or "cpu", sep="|")' \
    2>/dev/null || true)"

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "[INFO] Creating ${VENV_DIR} with access to base-image packages"
    mkdir -p "${RULI_ROOT}"
    "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
else
    echo "[INFO] Reusing existing environment at ${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade "pip==24.2" "setuptools==75.1.0" wheel

if python -c 'import torch' >/dev/null 2>&1; then
    echo "[INFO] Reusing PyTorch $(python -c 'import torch; print(torch.__version__)')"
elif [[ -n "${BASE_TORCH}" ]]; then
    echo "[WARN] Base-image PyTorch ${BASE_TORCH} is not visible in this existing venv."
    echo "[INFO] Installing the working RunPod PyTorch build into the venv."
    python -m pip install "torch==2.8.0" --index-url "${TORCH_INDEX_URL}"
else
    echo "[INFO] No existing PyTorch found; installing torch 2.8.0 from ${TORCH_INDEX_URL}"
    python -m pip install "torch==2.8.0" --index-url "${TORCH_INDEX_URL}"
fi

# RULI compatibility pins plus reproducible Python-3.12-compatible runtime pins.
# NumPy 1.26.4 replaces upstream's Python-3.12-incompatible numpy==1.23.5 pin.
python -m pip install \
    "transformers==4.39.1" \
    "accelerate==0.28.0" \
    "huggingface-hub==0.25.0" \
    "datasets==2.21.0" \
    "numpy==1.26.4" \
    "scikit-learn==1.5.2" \
    "scipy==1.14.1" \
    "matplotlib==3.9.2" \
    "tqdm==4.66.5" \
    "regex==2024.9.11" \
    "safetensors==0.4.5" \
    "sentence-transformers==3.0.1" \
    "networkx==3.3" \
    "pandas==2.2.3" \
    "pyarrow==17.0.0"

python -m pip install --no-deps -e "${EXPERIMENT_ROOT}"

echo "[VERIFY] Checking installed dependency consistency"
python -m pip check

echo "[VERIFY] Importing RULI and Experiment 1 runtime dependencies"
python - <<'PY'
import sys

import accelerate
import datasets
import matplotlib
import networkx
import numpy
import pandas
import pyarrow
import regex
import safetensors
import scipy
import sentence_transformers
import sklearn
import torch
import tqdm
import transformers

print(f"Python version: {sys.version.split()[0]}")
print(f"torch version: {torch.__version__}")
print(f"CUDA availability: {torch.cuda.is_available()}")
print(f"torch CUDA build: {torch.version.cuda}")
print(f"transformers version: {transformers.__version__}")
print(f"accelerate version: {accelerate.__version__}")
print(f"datasets version: {datasets.__version__}")
print(f"sklearn version: {sklearn.__version__}")
print(f"numpy version: {numpy.__version__}")
print(f"scipy version: {scipy.__version__}")
print(f"matplotlib version: {matplotlib.__version__}")
print(f"networkx version: {networkx.__version__}")
print(f"sentence-transformers version: {sentence_transformers.__version__}")
print(f"pandas version: {pandas.__version__}")
print(f"pyarrow version: {pyarrow.__version__}")
print(f"tqdm version: {tqdm.__version__}")
print(f"regex version: {regex.__version__}")
print(f"safetensors version: {safetensors.__version__}")
PY

echo "[INFO] Environment ready. Activate it with: source ${VENV_DIR}/bin/activate"

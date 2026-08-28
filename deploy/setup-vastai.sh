#!/usr/bin/env bash
set -euo pipefail

app_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3.10}"

if [[ "${EUID}" -eq 0 ]]; then
    sudo_bin=""
else
    sudo_bin="sudo"
fi

if ! command -v git >/dev/null 2>&1; then
    "${sudo_bin}" apt-get update
    "${sudo_bin}" apt-get install -y git
fi

if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "Python executable not found: ${python_bin}" >&2
    echo "Set PYTHON_BIN to a Python 3.10-3.12 executable." >&2
    exit 1
fi

if [[ ! -x "${app_dir}/.venv/bin/python" ]]; then
    "${python_bin}" -m venv --system-site-packages "${app_dir}/.venv"
fi

"${app_dir}/.venv/bin/python" -m pip install --upgrade pip
"${app_dir}/.venv/bin/python" -m pip install "${app_dir}"

if ! "${app_dir}/.venv/bin/python" -c 'import torch' >/dev/null 2>&1; then
    echo "PyTorch is not available. Use a Vast.ai PyTorch image or" >&2
    echo "install a CUDA-compatible PyTorch build." >&2
    exit 1
fi

echo "Setup complete. Start with:"
echo "  bash ${app_dir}/scripts/start-simpleclick.sh"

#!/usr/bin/env bash
set -euo pipefail

app_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Pydantic reads .env for the API process. Load the same file here so the
# bootstrap step uses an overridden source/checkpoint path as well.
if [[ -f "${app_dir}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${app_dir}/.env"
    set +a
fi

python_bin="${app_dir}/.venv/bin/python"
gdown_bin="${app_dir}/.venv/bin/gdown"
if [[ ! -x "${python_bin}" || ! -x "${gdown_bin}" ]]; then
    echo "Missing .venv dependencies; run deploy/setup-vastai.sh first" >&2
    exit 1
fi

simpleclick_root="${SIMPLECLICK_ROOT:-${app_dir}/.cache/SimpleClick}"
export SIMPLECLICK_ROOT="${simpleclick_root}"
checkpoint_path="${SIMPLECLICK_CHECKPOINT_PATH:-}"
if [[ -z "${checkpoint_path}" ]]; then
    checkpoint_path="${simpleclick_root}/weights/simpleclick_models/cocolvis_vit_huge.pth"
fi
if [[ "${simpleclick_root}" != /* ]]; then
    simpleclick_root="${app_dir}/${simpleclick_root}"
    export SIMPLECLICK_ROOT="${simpleclick_root}"
fi
if [[ "${checkpoint_path}" != /* ]]; then
    checkpoint_path="${app_dir}/${checkpoint_path}"
fi
export SIMPLECLICK_CHECKPOINT_PATH="${checkpoint_path}"
checkpoint_id="${SIMPLECLICK_CHECKPOINT_ID:-1GXk6q5fwKo2twkY5ZZGjVKCgJv7XeLAW}"
repo_url="${SIMPLECLICK_REPO_URL:-https://github.com/uncbiag/SimpleClick.git}"
repo_ref="${SIMPLECLICK_REF:-v1.0}"

if [[ ! -d "${simpleclick_root}/isegm" ]]; then
    mkdir -p "$(dirname "${simpleclick_root}")"
    echo "Cloning SimpleClick ${repo_ref} into ${simpleclick_root}" >&2
    git clone --depth 1 --branch "${repo_ref}" "${repo_url}" "${simpleclick_root}"
fi

if [[ ! -s "${checkpoint_path}" ]]; then
    mkdir -p "$(dirname "${checkpoint_path}")"
    temporary_checkpoint="$(mktemp "${checkpoint_path}.download.XXXXXX")"
    echo "Downloading SimpleClick checkpoint into ${checkpoint_path}" >&2
    "${gdown_bin}" "${checkpoint_id}" -O "${temporary_checkpoint}"
    mv "${temporary_checkpoint}" "${checkpoint_path}"
fi

exec "${python_bin}" -m uvicorn app.main:app \
    --host "${SIMPLECLICK_HOST:-0.0.0.0}" \
    --port "${SIMPLECLICK_PORT:-8000}" \
    --workers "${SIMPLECLICK_WORKERS:-1}"

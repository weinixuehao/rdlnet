#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
PIP="${ROOT}/.venv/bin/pip"

CKPT="${1:-${ROOT}/output/rdlnet/rdlnet_20260423_141008_best.pt}"
OUT_DIR="${2:-${ROOT}/output/_export_rdlnet}"
IMG_SIZE="${3:-1024}"
EXPORT_MODE="${4:-points}"   # points|full
INPUT_RANGE="${5:-0_1}"      # 0_1|0_255

cd "${ROOT}"

if [[ ! -x "${PY}" ]]; then
  echo "ERROR: venv python not found: ${PY}" >&2
  echo "Create venv at ${ROOT}/.venv first." >&2
  exit 1
fi

exec "${PY}" "${ROOT}/scripts/export_rdlnet_tflite.py" \
  --ckpt "${CKPT}" \
  --out-dir "${OUT_DIR}" \
  --img-size "${IMG_SIZE}" \
  --use-sam-pixel-norm \
  --input-range "${INPUT_RANGE}" \
  --export "${EXPORT_MODE}"


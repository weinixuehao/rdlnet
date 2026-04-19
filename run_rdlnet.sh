#!/bin/sh
# Stage-2 RDLNet training wrapper (repo root). RWMD LabelMe tree by default.
#
# Usage:
#   ./run_rdlnet.sh
#   ./run_rdlnet.sh --epochs 20 --batch-size 2
#   ./run_rdlnet.sh --resume checkpoints/rdlnet.pt --epochs 10
#   RESUME=checkpoints/rdlnet.pt ./run_rdlnet.sh --epochs 10
#
# Do not set RESUME and also pass --resume (duplicate flags).

set -eu
cd "$(CDPATH= cd -- "$(dirname "$0")" && pwd)"

if [ -f .venv/bin/activate ]; then
  # shellcheck source=/dev/null
  . .venv/bin/activate
fi

if [ -n "${RESUME:-}" ]; then
  set -- --resume "$RESUME" "$@"
fi

exec python train_rdlnet.py \
  --rwmd-root dataset/RWMD_dataset/RWMD_dataset_v1 \
  --num-classes 2 \
  --rwmd-label-mode main_bg \
  --distill-checkpoint checkpoints/distill_stage1.pt \
  --output checkpoints/rdlnet.pt \
  --epochs 50 \
  --batch-size 2 \
  --num-workers 4 \
  "$@"

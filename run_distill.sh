#!/bin/sh
# Stage-1 distillation wrapper (repo root).
#
# Usage:
#   ./run_distill.sh
#   ./run_distill.sh --epochs 10
#   ./run_distill.sh --resume checkpoints/distill_stage1.pt --epochs 10
#   RESUME=checkpoints/distill_stage1.pt ./run_distill.sh --epochs 10
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

exec python train_distill.py \
  --image-dir dataset/train2017 \
  --teacher-checkpoint checkpoints/sam/sam_vit_h_4b8939.pth \
  --output checkpoints/distill_stage1.pt \
  "$@"

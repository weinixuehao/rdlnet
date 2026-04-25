#!/bin/sh
# Stage-1 distillation wrapper (repo root).
#
# Usage:
#   ./run_distill.sh
#   ./run_distill.sh --epochs 10
#   ./run_distill.sh --resume checkpoints/distill_stage1/<run_stamp>_lite40/checkpoint.pt --epochs 10
#   RESUME=checkpoints/distill_stage1/<run_stamp>_lite40/checkpoint.pt ./run_distill.sh --epochs 10
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
  --coco-train-dir dataset/coco/train2017 \
  --coco-instances-json dataset/coco/annotations/instances_train2017.json \
  --coco-val-dir dataset/coco/val2017 \
  --coco-val-instances-json dataset/coco/annotations/instances_val2017.json \
  --teacher-checkpoint checkpoints/sam/sam_vit_h_4b8939.pth \
  --output checkpoints/distill \
  --seed 42 \
  --epochs 1000 \
  --batch-size 2 \
  --grad-accum-steps 4 \
  --num-workers 4 \
  --lite 10 \
  --amp \
  "$@"

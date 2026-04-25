#!/usr/bin/env sh
# Stage-2 RDLNet training wrapper (repo root). Set RWMD_ROOT to preprocessed train_resize (see data_preprocessing_rwdm_1.py).
#
# Usage:
#   ./run_rdlnet.sh
#   ./run_rdlnet.sh --epochs 20 --batch-size 2
#   ./run_rdlnet.sh --grad-accum-steps 4   # effective batch_size * 4 per optimizer step
#   ./run_rdlnet.sh --resume output/rdlnet/20250424_153022 --epochs 10
#   RESUME=output/rdlnet/20250424_153022 ./run_rdlnet.sh --epochs 10
#
# Do not set RESUME and also pass --resume (duplicate flags).
#
# Requires LF line endings (Unix). On macOS and Ubuntu, prefer .venv/bin/python when present.

set -eu
cd "$(CDPATH= cd -- "$(dirname "$0")" && pwd)"

if [ -f .venv/bin/activate ]; then
  # shellcheck source=/dev/null
  . .venv/bin/activate
fi

if [ -n "${RESUME:-}" ]; then
  set -- --resume "$RESUME" "$@"
fi

  # --distill-checkpoint checkpoints/distill_stage1.pt \
exec python train_rdlnet.py \
  --rwmd-root "output/data/train_resize" \
  --val-rwmd-root "output/data/test_resize" \
  --resume output/rdlnet/20260424_111048 \
  --num-classes 2 \
  --output output/rdlnet \
  --lite 20 \
  --epochs 500 \
  --batch-size 2 \
  --grad-accum-steps 4 \
  --grad-clip-norm 4.0 \
  --num-workers 4 \
  --amp \
  "$@"

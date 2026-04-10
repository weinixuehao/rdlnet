#!/usr/bin/env python3
"""
Random subset of COCO train2017 images.

Default: copy files (suitable for zipping / Google Drive upload).
Use --symlink to save local disk when you only train in-place.

Example:
  python scripts/sample_coco_train_subset.py --num 30000 --seed 42
  python scripts/sample_coco_train_subset.py --num 30000 --symlink
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src",
        type=Path,
        default=Path("data/coco/train2017"),
        help="Directory of COCO train2017 *.jpg",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/coco/train2017_30k"),
        help="Output directory",
    )
    p.add_argument("--num", type=int, default=30_000, help="How many images to sample")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    p.add_argument(
        "--symlink",
        action="store_true",
        help="Create symlinks instead of copying (not for Drive upload)",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    src = (root / args.src).resolve() if not args.src.is_absolute() else args.src.resolve()
    out = (root / args.out).resolve() if not args.out.is_absolute() else args.out.resolve()

    if not src.is_dir():
        print(f"Missing source dir: {src}", file=sys.stderr)
        sys.exit(1)

    files = sorted(f for f in os.listdir(src) if f.lower().endswith(".jpg"))
    n = len(files)
    if n < args.num:
        print(f"Only {n} jpg files in {src}, cannot sample {args.num}", file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)
    chosen = random.sample(files, args.num)

    out.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(chosen):
        dst = out / name
        source = src / name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        if args.symlink:
            dst.symlink_to(source)
        else:
            shutil.copy2(source, dst)
        if (i + 1) % 5000 == 0:
            print(f"  ... {i + 1}/{len(chosen)}")

    mode = "symlinks" if args.symlink else "copies"
    print(f"Created {len(chosen)} {mode} under {out}")
    print(f"seed={args.seed} (re-run with same seed for identical subset)")


if __name__ == "__main__":
    main()

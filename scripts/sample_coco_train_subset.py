#!/usr/bin/env python3
"""
Random subset of COCO train2017 images.

By default reads ``dataset/train2017.zip`` (no need to unzip the full train set first).
Alternatively pass ``--src`` to use an already-extracted directory.

Default: copy sampled files out of the zip (or from ``--src``).
Use ``--symlink`` only with ``--src`` (not supported for zip mode).

Example:
  python scripts/sample_coco_train_subset.py --num 30000 --seed 42
  python scripts/sample_coco_train_subset.py --num 30000 --src dataset/train2017 --symlink
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import zipfile
from pathlib import Path


def _jpg_names_in_dir(src: Path) -> list[str]:
    return sorted(f for f in os.listdir(src) if f.lower().endswith(".jpg"))


def _jpg_members_in_zip(zf: zipfile.ZipFile) -> list[str]:
    """Member paths ending in .jpg (COCO zip uses train2017/000000xxxxxx.jpg)."""
    return sorted(n for n in zf.namelist() if n.lower().endswith(".jpg"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--zip",
        type=Path,
        default=Path("dataset/train2017.zip"),
        help="COCO train2017 zip under repo (used when --src is not set)",
    )
    p.add_argument(
        "--src",
        type=Path,
        default=None,
        help="Directory of extracted train2017 *.jpg; if set, --zip is ignored",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("dataset/train2017_30k"),
        help="Output directory",
    )
    p.add_argument("--num", type=int, default=30_000, help="How many images to sample")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    p.add_argument(
        "--symlink",
        action="store_true",
        help="Create symlinks instead of copying (requires --src, not for --zip)",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = (root / args.out).resolve() if not args.out.is_absolute() else args.out.resolve()

    use_dir = args.src is not None
    if use_dir:
        src = (root / args.src).resolve() if not args.src.is_absolute() else args.src.resolve()
        if args.symlink and not src.is_dir():
            print(f"Missing source dir: {src}", file=sys.stderr)
            sys.exit(1)
        if not src.is_dir():
            print(f"Missing source dir: {src}", file=sys.stderr)
            sys.exit(1)
        files = _jpg_names_in_dir(src)
        n = len(files)
        if n < args.num:
            print(f"Only {n} jpg files in {src}, cannot sample {args.num}", file=sys.stderr)
            sys.exit(1)
        random.seed(args.seed)
        chosen_names = random.sample(files, args.num)

        out.mkdir(parents=True, exist_ok=True)
        for i, name in enumerate(chosen_names):
            dst = out / name
            source = src / name
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            if args.symlink:
                dst.symlink_to(source)
            else:
                shutil.copy2(source, dst)
            if (i + 1) % 5000 == 0:
                print(f"  ... {i + 1}/{len(chosen_names)}")
        mode = "symlinks" if args.symlink else "copies"
        print(f"Created {len(chosen_names)} {mode} under {out}")
        print(f"seed={args.seed} (re-run with same seed for identical subset)")
        return

    if args.symlink:
        print("--symlink requires --src (extracted directory); zip mode always extracts files.", file=sys.stderr)
        sys.exit(1)

    zip_path = (root / args.zip).resolve() if not args.zip.is_absolute() else args.zip.resolve()
    if not zip_path.is_file():
        print(f"Missing zip: {zip_path}", file=sys.stderr)
        print("Place COCO train2017.zip there or pass --zip / --src.", file=sys.stderr)
        sys.exit(1)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = _jpg_members_in_zip(zf)
        n = len(members)
        if n < args.num:
            print(f"Only {n} jpg entries in {zip_path}, cannot sample {args.num}", file=sys.stderr)
            sys.exit(1)
        random.seed(args.seed)
        chosen = random.sample(members, args.num)

        out.mkdir(parents=True, exist_ok=True)
        for i, member in enumerate(chosen):
            name = Path(member).name
            dst = out / name
            if dst.exists():
                dst.unlink()
            with zf.open(member, "r") as src_f, open(dst, "wb") as dst_f:
                shutil.copyfileobj(src_f, dst_f)
            if (i + 1) % 5000 == 0:
                print(f"  ... {i + 1}/{len(chosen)}")

    print(f"Extracted {len(chosen)} images from {zip_path} -> {out}")
    print(f"seed={args.seed} (re-run with same seed for identical subset)")


if __name__ == "__main__":
    main()

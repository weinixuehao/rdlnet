#!/usr/bin/env python3
"""
Stage 1: Light-SAM multiplex distillation (paper Sec. 3.1).

Example::

    python train_distill.py \\
        --image-dir /path/to/unlabeled/images \\
        --teacher-checkpoint /path/to/sam_vit_h_4b8939.pth \\
        --output checkpoints/distill_stage1.pt

Requires: SAM ViT-H weights and a folder of RGB images (any domain; natural images OK).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rdlnet.data import DistillImageFolder
from rdlnet.distill import (
    DistillConfig,
    LightSAMMultiplexDistillation,
    create_distillation_setup,
    distill_trainable_state_dict,
    load_distill_trainable_state_dict,
)
from rdlnet.sam_backbone import RDLNetSAMEncoder


def pick_device() -> torch.device:
    """Prefer CUDA, then Apple MPS, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collate_distill(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    paths = [b[1] for b in batch]
    return imgs, paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1: distill Light-SAM student from SAM ViT-H")
    p.add_argument("--image-dir", type=str, required=True, help="Directory of RGB images (recursive)")
    p.add_argument("--teacher-checkpoint", type=str, required=True, help="sam_vit_h_4b8939.pth")
    p.add_argument("--output", type=str, default="checkpoints/distill_stage1.pt")
    p.add_argument("--resume", type=str, default=None, help="Resume from stage-1 checkpoint (weights + optimizer if present)")
    p.add_argument("--epochs", type=int, default=1, help="Number of additional epochs to run in this session")
    p.add_argument(
        "--save-every-steps",
        type=int,
        default=0,
        help="If >0, also save a copy of the checkpoint every N steps (for Colab disconnects mid-epoch)",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=1024, help="Must match stage 2 RDLNet img_size")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--weight-kl", type=float, default=1.0)
    p.add_argument("--weight-md", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device()
    print(f"device => {device}")

    ds = DistillImageFolder(args.image_dir, img_size=args.img_size)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_distill,
        pin_memory=device.type == "cuda",
    )

    student_wrap = RDLNetSAMEncoder(img_size=args.img_size)
    cfg = DistillConfig(
        temperature=args.temperature,
        weight_kl=args.weight_kl,
        weight_md=args.weight_md,
        teacher_checkpoint=args.teacher_checkpoint,
    )
    distill: LightSAMMultiplexDistillation = create_distillation_setup(
        student_wrap.encoder,
        teacher_checkpoint=args.teacher_checkpoint,
        cfg=cfg,
    ).to(device)

    opt = torch.optim.AdamW(distill.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    global_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        load_distill_trainable_state_dict(distill, ck)
        if isinstance(ck, dict) and "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        meta = ck.get("meta") or {}
        start_epoch = int(meta.get("epochs_done", 0))
        global_step = int(ck.get("global_step", 0))
        print(f"Resumed from {args.resume}: epochs_done={start_epoch}, global_step={global_step}")

    os.makedirs(Path(args.output).parent or ".", exist_ok=True)
    out_path = Path(args.output)
    step_ckpt = out_path.with_name(out_path.stem + "_latest.pt")

    end_epoch = start_epoch + args.epochs
    for epoch in range(start_epoch, end_epoch):
        distill.train()
        for imgs, _paths in loader:
            print(_paths)
            imgs = imgs.to(device)
            out = distill(imgs)
            loss = out["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            global_step += 1
            if global_step % 50 == 0:
                print(
                    f"epoch {epoch+1}/{end_epoch} step {global_step} "
                    f"loss={loss.item():.4f} kl={out['loss_kl'].item():.4f} md={out['loss_md'].item():.4f}"
                )
            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                meta = {
                    "img_size": args.img_size,
                    "backbone_dim": student_wrap.embed_dim,
                    "backbone_depth": student_wrap.depth,
                    "epochs_done": epoch,
                    "note": "mid-epoch snapshot",
                }
                ckpt = distill_trainable_state_dict(distill, meta=meta)
                ckpt["optimizer"] = opt.state_dict()
                ckpt["global_step"] = global_step
                torch.save(ckpt, step_ckpt)
                print(f"Saved step checkpoint -> {step_ckpt}")

        meta = {
            "img_size": args.img_size,
            "backbone_dim": student_wrap.embed_dim,
            "backbone_depth": student_wrap.depth,
            "epochs_done": epoch + 1,
        }
        ckpt = distill_trainable_state_dict(distill, meta=meta)
        ckpt["optimizer"] = opt.state_dict()
        ckpt["global_step"] = global_step
        torch.save(ckpt, args.output)
        print(f"Saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()

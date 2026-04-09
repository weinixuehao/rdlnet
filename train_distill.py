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


def collate_distill(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    paths = [b[1] for b in batch]
    return imgs, paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1: distill Light-SAM student from SAM ViT-H")
    p.add_argument("--image-dir", type=str, required=True, help="Directory of RGB images (recursive)")
    p.add_argument("--teacher-checkpoint", type=str, required=True, help="sam_vit_h_4b8939.pth")
    p.add_argument("--output", type=str, default="checkpoints/distill_stage1.pt")
    p.add_argument("--resume", type=str, default=None, help="Resume from stage-1 checkpoint")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=1024, help="Must match stage 2 RDLNet img_size")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--weight-kl", type=float, default=1.0)
    p.add_argument("--weight-md", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

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

    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        load_distill_trainable_state_dict(distill, ck)

    opt = torch.optim.AdamW(distill.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(Path(args.output).parent or ".", exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        distill.train()
        for imgs, _paths in loader:
            imgs = imgs.to(device)
            out = distill(imgs)
            loss = out["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            global_step += 1
            if global_step % 50 == 0:
                print(
                    f"epoch {epoch+1}/{args.epochs} step {global_step} "
                    f"loss={loss.item():.4f} kl={out['loss_kl'].item():.4f} md={out['loss_md'].item():.4f}"
                )

        meta = {
            "img_size": args.img_size,
            "backbone_dim": student_wrap.embed_dim,
            "backbone_depth": student_wrap.depth,
            "epochs_done": epoch + 1,
        }
        ckpt = distill_trainable_state_dict(distill, meta=meta)
        torch.save(ckpt, args.output)
        print(f"Saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()

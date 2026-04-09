#!/usr/bin/env python3
"""
Stage 2: Train full RDLNet on document localization annotations (paper losses, Sec. 3.4).

Initialize the SAM student backbone from stage-1 ``distill_stage1.pt`` (recommended).

Example::

    python train_rdlnet.py \\
        --annotations data/annotations.json \\
        --image-root data/images \\
        --distill-checkpoint checkpoints/distill_stage1.pt \\
        --output checkpoints/rdlnet.pt

See ``rdlnet.data.doc_json`` for the JSON format.
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

from rdlnet.data import DocLocalizationJsonDataset, collate_doc_batch
from rdlnet.distill import load_student_encoder_into_rdlnet_from_checkpoint
from rdlnet.losses import RDLNetLoss, build_matcher
from rdlnet.model import RDLNet, RDLNetConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2: train RDLNet on annotated documents")
    p.add_argument("--annotations", type=str, required=True, help="JSON list (see rdlnet.data.doc_json)")
    p.add_argument("--image-root", type=str, required=True, help="Root for file_name and mask paths")
    p.add_argument("--distill-checkpoint", type=str, default=None, help="Stage-1 student_encoder weights")
    p.add_argument("--output", type=str, default="checkpoints/rdlnet.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    cfg = RDLNetConfig(
        img_size=args.img_size,
        use_sam_pixel_norm=True,
    )
    if args.img_size % cfg.patch_size != 0:
        raise ValueError("img_size must be divisible by patch_size")

    ds = DocLocalizationJsonDataset(
        args.annotations,
        args.image_root,
        img_size=args.img_size,
        num_classes=cfg.num_classes,
        num_points=cfg.num_points,
        max_instances=cfg.num_queries,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_doc_batch,
        pin_memory=device.type == "cuda",
    )

    model = RDLNet(cfg).to(device)
    if args.distill_checkpoint:
        load_student_encoder_into_rdlnet_from_checkpoint(model, args.distill_checkpoint)
        print(f"Loaded student backbone from {args.distill_checkpoint}")

    matcher = build_matcher(cfg)
    criterion = RDLNetLoss(cfg, matcher)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(Path(args.output).parent or ".", exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            images = batch["images"].to(device)
            out = model(images)
            loss, logs = criterion(
                out["pred_logits"],
                out["pred_masks"],
                out["pred_points"],
                batch["tgt_labels"],
                batch["tgt_masks"],
                batch["tgt_points"],
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach())
            n_batches += 1
            global_step += 1
            if global_step % 20 == 0:
                print(
                    f"epoch {epoch+1}/{args.epochs} step {global_step} loss={loss.item():.4f} "
                    f"cls={logs['loss_cls'].item():.4f} mask={logs['loss_mask'].item():.4f}"
                )

        avg = epoch_loss / max(n_batches, 1)
        print(f"epoch {epoch+1} mean loss={avg:.4f}")

        torch.save(
            {
                "model": model.state_dict(),
                "config": cfg,
                "epoch": epoch + 1,
            },
            args.output,
        )
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

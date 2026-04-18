#!/usr/bin/env python3
"""
Stage 2: Train full RDLNet on document localization annotations (paper losses, Sec. 3.4).

Initialize the SAM student backbone from stage-1 ``distill_stage1.pt`` (recommended).

Example (manifest JSON)::

    python train_rdlnet.py \\
        --annotations data/annotations.json \\
        --image-root data/images \\
        --distill-checkpoint checkpoints/distill_stage1.pt \\
        --output checkpoints/rdlnet.pt

Example (RWMD LabelMe tree, e.g. ``RWMD_dataset_v1``)::

    python train_rdlnet.py \\
        --rwmd-root dataset/RWMD_dataset/RWMD_dataset_v1 \\
        --num-classes 9 \\
        --distill-checkpoint checkpoints/distill_stage1.pt \\
        --output checkpoints/rdlnet.pt

See ``rdlnet.data.doc_json`` for the manifest format and :class:`RWMDLabelMeDataset`.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rdlnet.data import DocLocalizationJsonDataset, RWMDLabelMeDataset, collate_doc_batch
from rdlnet.distill import load_student_encoder_into_rdlnet_from_checkpoint
from rdlnet.losses import RDLNetLoss, build_matcher
from rdlnet.model import RDLNet, RDLNetConfig


def pick_device() -> torch.device:
    """Prefer CUDA, then Apple MPS, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2: train RDLNet on annotated documents")
    p.add_argument("--annotations", type=str, default=None, help="JSON list (see rdlnet.data.doc_json)")
    p.add_argument("--image-root", type=str, default=None, help="Root for file_name and mask paths")
    p.add_argument(
        "--rwmd-root",
        type=str,
        default=None,
        help="RWMD LabelMe root (recursive *.json). If set, --annotations/--image-root are not used.",
    )
    p.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Override RDLNetConfig.num_classes (recommended 9 for RWMD folder mode; default 3 for manifest)",
    )
    p.add_argument(
        "--rwmd-label-mode",
        type=str,
        choices=["folder", "layer", "zero"],
        default="folder",
        help="RWMD: class label from scene folder, from layer id, or all 0",
    )
    p.add_argument(
        "--rwmd-instance-order",
        type=str,
        choices=["foreground_first", "numeric_then_foreground", "json_order"],
        default="foreground_first",
        help="RWMD: which polygons to keep when truncating to num_queries",
    )
    p.add_argument(
        "--distill-checkpoint",
        type=str,
        default=None,
        help="Stage-1 student_encoder weights (ignored if --resume loads a full model)",
    )
    p.add_argument("--output", type=str, default="checkpoints/rdlnet.pt")
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume full training state (model+optimizer+epoch) from a previous train_rdlnet checkpoint",
    )
    p.add_argument("--epochs", type=int, default=50, help="Additional epochs when resuming, or total epochs from scratch")
    p.add_argument(
        "--save-every-steps",
        type=int,
        default=0,
        help="If >0, save a snapshot every N steps to *_latest.pt (Colab disconnect recovery)",
    )
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--ignore-padded-points",
        action="store_true",
        help="Exclude padded (0,0) point pairs from point matching cost and point distance loss",
    )
    p.add_argument(
        "--padded-point-eps",
        type=float,
        default=0.0,
        help="Treat |x|<=eps and |y|<=eps as padded (default: 0.0)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device()
    print(f"device => {device}")

    num_classes = args.num_classes if args.num_classes is not None else 3
    cfg = RDLNetConfig(
        img_size=args.img_size,
        num_classes=num_classes,
        use_sam_pixel_norm=True,
        ignore_padded_points=args.ignore_padded_points,
        padded_point_eps=args.padded_point_eps,
    )
    if args.img_size % cfg.patch_size != 0:
        raise ValueError("img_size must be divisible by patch_size")

    if args.rwmd_root:
        ds = RWMDLabelMeDataset(
            args.rwmd_root,
            img_size=args.img_size,
            num_classes=cfg.num_classes,
            num_points=cfg.num_points,
            max_instances=cfg.num_queries,
            label_mode=args.rwmd_label_mode,
            instance_order=args.rwmd_instance_order,
        )
    else:
        if not args.annotations or not args.image_root:
            raise SystemExit("Provide either --rwmd-root, or both --annotations and --image-root")
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
    matcher = build_matcher(cfg)
    criterion = RDLNetLoss(cfg, matcher)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    global_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        start_epoch = int(ck.get("epoch", 0))
        global_step = int(ck.get("global_step", 0))
        print(f"Resumed from {args.resume}: epoch={start_epoch}, global_step={global_step}")
    elif args.distill_checkpoint:
        load_student_encoder_into_rdlnet_from_checkpoint(model, args.distill_checkpoint)
        print(f"Loaded student backbone from {args.distill_checkpoint}")

    os.makedirs(Path(args.output).parent or ".", exist_ok=True)
    out_path = Path(args.output)
    step_ckpt = out_path.with_name(out_path.stem + "_latest.pt")

    end_epoch = start_epoch + args.epochs
    for epoch in range(start_epoch, end_epoch):
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
                    f"epoch {epoch+1}/{end_epoch} step {global_step} loss={loss.item():.4f} "
                    f"cls={logs['loss_cls'].item():.4f} mask={logs['loss_mask'].item():.4f}"
                )
            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "config": asdict(cfg),
                        "epoch": epoch,
                        "global_step": global_step,
                        "note": "mid-epoch",
                    },
                    step_ckpt,
                )
                print(f"Saved step checkpoint -> {step_ckpt}")

        avg = epoch_loss / max(n_batches, 1)
        print(f"epoch {epoch+1} mean loss={avg:.4f}")

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "config": asdict(cfg),
                "epoch": epoch + 1,
                "global_step": global_step,
            },
            args.output,
        )
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

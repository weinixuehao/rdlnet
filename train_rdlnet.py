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

Example (RWMD preprocessed ``train_resize`` from ``data_preprocessing_rwdm_1``)::

    python train_rdlnet.py \\
        --rwmd-root path/to/out/train_resize \\
        --num-classes 2 \\
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
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rdlnet.data import DocLocalizationJsonDataset, RWMDLabelMeDataset, collate_doc_batch
from rdlnet.device import pick_device
from rdlnet.distill import load_student_encoder_into_rdlnet_from_checkpoint
from rdlnet.losses import RDLNetLoss, build_matcher
from rdlnet.model import RDLNet, RDLNetConfig
from rdlnet.viz_rdlnet import train_compare_grid_u8

LOSS_HISTORY_KEY = "train_rdlnet_loss_history"


def pack_loss_history(
    epochs: list[int],
    loss: list[float],
    loss_cls: list[float],
    loss_dist: list[float],
    loss_dice: list[float],
    loss_mask: list[float],
) -> dict[str, Any]:
    return {
        "epochs": list(epochs),
        "loss": list(loss),
        "loss_cls": list(loss_cls),
        "loss_dist": list(loss_dist),
        "loss_dice": list(loss_dice),
        "loss_mask": list(loss_mask),
    }


def load_loss_history_from_ck(ck: dict) -> tuple[list[int], list[float], list[float], list[float], list[float], list[float]]:
    h = ck.get(LOSS_HISTORY_KEY)
    if not isinstance(h, dict):
        return [], [], [], [], [], []
    try:
        ep = [int(x) for x in h["epochs"]]
        lo = [float(x) for x in h["loss"]]
        lc = [float(x) for x in h["loss_cls"]]
        ld = [float(x) for x in h["loss_dist"]]
        ldi = [float(x) for x in h["loss_dice"]]
        lm = [float(x) for x in h["loss_mask"]]
    except (KeyError, TypeError, ValueError):
        return [], [], [], [], [], []
    n = min(len(ep), len(lo), len(lc), len(ld), len(ldi), len(lm))
    if n == 0:
        return [], [], [], [], [], []
    return ep[:n], lo[:n], lc[:n], ld[:n], ldi[:n], lm[:n]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2: train RDLNet on annotated documents")
    p.add_argument("--annotations", type=str, default=None, help="JSON list (see rdlnet.data.doc_json)")
    p.add_argument("--image-root", type=str, default=None, help="Root for file_name and mask paths")
    p.add_argument(
        "--rwmd-root",
        type=str,
        default=None,
        help="RWMD preprocessed folder: img/, mask/, label_points_resize.json (see data_preprocessing_rwdm_1.py).",
    )
    p.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Override RDLNetConfig.num_classes (RWMD preprocessed typically uses 2: top sheet vs other)",
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
    p.add_argument(
        "--amp",
        action="store_true",
        help="Enable CUDA automatic mixed precision (AMP) training via autocast + GradScaler",
    )
    p.add_argument(
        "--amp-dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="AMP autocast dtype when --amp is set (bf16 is recommended when supported)",
    )
    p.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Clip gradient L2 norm before optimizer step (0 disables). Helps avoid mask BCE blow-ups.",
    )
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--max-batches-per-epoch",
        type=int,
        default=0,
        help="If >0, stop each epoch after this many train batches (debug/smoke runs)",
    )
    p.add_argument(
        "--viz-samples",
        type=int,
        default=4,
        help="Grid PNG: max rows (batch images) per save; 0 disables all viz",
    )
    p.add_argument(
        "--viz-every-steps",
        type=int,
        default=30,
        help="Save train viz grid every N global steps (current batch); 0 disables",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device()
    print(f"device => {device}")

    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"TensorBoard not available (install tensorboard). Import error: {e}") from e

    if args.num_classes is not None:
        num_classes = args.num_classes
    elif args.rwmd_root:
        num_classes = 2
    else:
        num_classes = 3
    cfg = RDLNetConfig(
        img_size=args.img_size,
        num_classes=num_classes,
        use_sam_pixel_norm=True,
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
    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if args.amp and not use_amp:
        print("WARN: --amp requested but CUDA is not available; AMP disabled.")

    hist_ep: list[int] = []
    hist_loss: list[float] = []
    hist_cls: list[float] = []
    hist_dist: list[float] = []
    hist_dice: list[float] = []
    hist_mask: list[float] = []

    start_epoch = 0
    global_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        start_epoch = int(ck.get("epoch", 0))
        global_step = int(ck.get("global_step", 0))
        hist_ep, hist_loss, hist_cls, hist_dist, hist_dice, hist_mask = load_loss_history_from_ck(ck)
        del ck
        print(f"Resumed from {args.resume}: epoch={start_epoch}, global_step={global_step}")
    elif args.distill_checkpoint:
        load_student_encoder_into_rdlnet_from_checkpoint(model, args.distill_checkpoint)
        print(f"Loaded student backbone from {args.distill_checkpoint}")

    os.makedirs(Path(args.output).parent or ".", exist_ok=True)
    out_path = Path(args.output)
    step_ckpt = out_path.with_name(out_path.stem + "_latest.pt")

    tb_logdir = out_path.parent / f"{out_path.stem}_tb"
    tb_logdir.mkdir(parents=True, exist_ok=True)
    print(f"tensorboard logdir => {tb_logdir}")

    end_epoch = start_epoch + args.epochs
    import numpy as np

    with SummaryWriter(log_dir=str(tb_logdir)) as tb:
        for epoch in range(start_epoch, end_epoch):
            model.train()
            epoch_loss = 0.0
            sum_cls = 0.0
            sum_dist = 0.0
            sum_dice = 0.0
            sum_mask = 0.0
            n_batches = 0
            max_batches = int(args.max_batches_per_epoch)
            for bi, batch in enumerate(tqdm(loader, desc=f"epoch {epoch + 1}/{end_epoch}", leave=True)):
                images = batch["images"].to(device)
                tgt_labels = [t.to(device) for t in batch["tgt_labels"]]
                tgt_masks = [t.to(device) for t in batch["tgt_masks"]]
                tgt_points = [t.to(device) for t in batch["tgt_points"]]
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
                    out = model(images)
                    loss, logs = criterion(
                        out["pred_logits"],
                        out["pred_masks"],
                        out["pred_points"],
                        tgt_labels,
                        tgt_masks,
                        tgt_points,
                    )
                if not torch.isfinite(loss).item():
                    tqdm.write(
                        f"WARN: non-finite loss at step={global_step} (loss={float(loss.detach().cpu())}); skipping update"
                    )
                    opt.zero_grad(set_to_none=True)
                    if use_amp:
                        scaler.update()
                    continue
                if use_amp:
                    scaler.scale(loss).backward()
                    if args.grad_clip_norm > 0:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    if args.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                    opt.step()
                epoch_loss += float(loss.detach())
                sum_cls += float(logs["loss_cls"].item())
                sum_dist += float(logs["loss_dist"].item())
                sum_dice += float(logs["loss_dice"].item())
                sum_mask += float(logs["loss_mask"].item())
                n_batches += 1
                global_step += 1

                if args.viz_samples > 0 and args.viz_every_steps > 0 and global_step % args.viz_every_steps == 0:
                    with torch.no_grad():
                        _mi = criterion.matcher(
                            out["pred_logits"],
                            out["pred_masks"],
                            out["pred_points"],
                            tgt_labels,
                            tgt_masks,
                            tgt_points,
                        )
                    matched_indices = [(a.cpu(), b.cpu()) for a, b in _mi]
                    grid_u8 = train_compare_grid_u8(
                        images.detach().cpu(),
                        {k: v.detach().cpu() for k, v in out.items()},
                        [t.detach().cpu() for t in tgt_labels],
                        [t.detach().cpu() for t in tgt_masks],
                        [t.detach().cpu() for t in tgt_points],
                        max_samples=args.viz_samples,
                        matched_indices=matched_indices,
                    )
                    chw = np.transpose(grid_u8, (2, 0, 1))
                    tb.add_image("train/compare_grid", chw, global_step=global_step, dataformats="CHW")

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
                    tqdm.write(f"Saved step checkpoint -> {step_ckpt}")
                if max_batches > 0 and (bi + 1) >= max_batches:
                    break

            nb = max(n_batches, 1)
            avg = epoch_loss / nb
            avg_cls = sum_cls / nb
            avg_dist = sum_dist / nb
            avg_dice = sum_dice / nb
            avg_mask = sum_mask / nb
            ep_no = epoch + 1
            hist_ep.append(ep_no)
            hist_loss.append(avg)
            hist_cls.append(avg_cls)
            hist_dist.append(avg_dist)
            hist_dice.append(avg_dice)
            hist_mask.append(avg_mask)
            tb.add_scalars(
                "train_epoch/losses",
                {
                    "total": avg,
                    "cls": avg_cls,
                    "dist": avg_dist,
                    "dice": avg_dice,
                    "mask": avg_mask,
                },
                global_step=ep_no,
            )

            ckpt: dict[str, Any] = {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "config": asdict(cfg),
                "epoch": ep_no,
                "global_step": global_step,
                LOSS_HISTORY_KEY: pack_loss_history(
                    hist_ep, hist_loss, hist_cls, hist_dist, hist_dice, hist_mask
                ),
            }
            torch.save(ckpt, args.output)
            print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

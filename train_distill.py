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
from tqdm.auto import tqdm

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


LOSS_PLOT_HISTORY_KEY = "loss_plot_history"


def load_loss_plot_history_from_ck(ck: dict) -> tuple[list[int], list[float], list[float], list[float]]:
    """Restore per-epoch loss lists saved in a stage-1 checkpoint (for PNG continuity on --resume)."""
    h = ck.get(LOSS_PLOT_HISTORY_KEY)
    if not isinstance(h, dict):
        return [], [], [], []
    try:
        ep = [int(x) for x in h["epochs"]]
        lo = [float(x) for x in h["loss"]]
        kl = [float(x) for x in h["loss_kl"]]
        md = [float(x) for x in h["loss_md"]]
    except (KeyError, TypeError, ValueError):
        return [], [], [], []
    n = min(len(ep), len(lo), len(kl), len(md))
    if n == 0:
        return [], [], [], []
    return ep[:n], lo[:n], kl[:n], md[:n]


def pack_loss_plot_history(
    epochs: list[int],
    losses: list[float],
    kls: list[float],
    mds: list[float],
) -> dict[str, object]:
    return {
        "epochs": list(epochs),
        "loss": list(losses),
        "loss_kl": list(kls),
        "loss_md": list(mds),
    }


def save_loss_plot_png(
    path: Path,
    epochs: list[int],
    losses: list[float],
    kls: list[float],
    mds: list[float],
) -> None:
    """Save loss curves as a PNG. Intended to run after every epoch (file is overwritten)."""
    if not epochs:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, losses, label="loss")
    ax.plot(epochs, kls, label="loss_kl")
    ax.plot(epochs, mds, label="loss_md")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean (per batch, train)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
    p.add_argument(
        "--loss-plot",
        type=str,
        default=None,
        help="PNG path for loss curves (default: next to --output, e.g. distill_stage1_loss.png)",
    )
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
    hist_ep: list[int] = []
    hist_loss: list[float] = []
    hist_kl: list[float] = []
    hist_md: list[float] = []
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        if not isinstance(ck, dict):
            raise TypeError("--resume checkpoint must be a dict")
        load_distill_trainable_state_dict(distill, ck)
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        meta = ck.get("meta") or {}
        start_epoch = int(meta.get("epochs_done", 0))
        global_step = int(ck.get("global_step", 0))
        hist_ep, hist_loss, hist_kl, hist_md = load_loss_plot_history_from_ck(ck)
        print(f"Resumed from {args.resume}: epochs_done={start_epoch}, global_step={global_step}")
        if hist_ep:
            print(f"loss plot history: {len(hist_ep)} points restored (full curve redrawn each epoch)")
            if len(hist_ep) != start_epoch:
                print(
                    f"Warning: loss_plot_history length ({len(hist_ep)}) != epochs_done ({start_epoch}); "
                    "curve may be inconsistent (old ckpt or hand-edited file)."
                )
            elif hist_ep[-1] != start_epoch:
                print(
                    f"Warning: last loss epoch id ({hist_ep[-1]}) != epochs_done ({start_epoch}); "
                    "plot x-axis may be wrong."
                )

    os.makedirs(Path(args.output).parent or ".", exist_ok=True)
    out_path = Path(args.output)
    step_ckpt = out_path.with_name(out_path.stem + "_latest.pt")
    loss_plot_path = Path(args.loss_plot) if args.loss_plot else out_path.with_name(out_path.stem + "_loss.png")
    print(f"loss plot (PNG) => {loss_plot_path}")

    end_epoch = start_epoch + args.epochs
    for epoch in range(start_epoch, end_epoch):
        distill.train()
        pbar = tqdm(
            loader,
            desc=f"epoch {epoch + 1}/{end_epoch}",
            dynamic_ncols=True,
        )
        epoch_loss_sum = 0.0
        epoch_kl_sum = 0.0
        epoch_md_sum = 0.0
        n_batches = 0
        for imgs, _paths in pbar:
            imgs = imgs.to(device)
            out = distill(imgs)
            loss = out["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            global_step += 1
            epoch_loss_sum += float(loss.detach())
            epoch_kl_sum += float(out["loss_kl"].detach())
            epoch_md_sum += float(out["loss_md"].detach())
            n_batches += 1
            pbar.set_postfix(
                step=global_step,
                loss=f"{loss.item():.4f}",
                kl=f"{out['loss_kl'].item():.4f}",
                md=f"{out['loss_md'].item():.4f}",
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
                ckpt[LOSS_PLOT_HISTORY_KEY] = pack_loss_plot_history(hist_ep, hist_loss, hist_kl, hist_md)
                torch.save(ckpt, step_ckpt)
                print(f"Saved step checkpoint -> {step_ckpt}")

        if n_batches > 0:
            m_loss = epoch_loss_sum / n_batches
            m_kl = epoch_kl_sum / n_batches
            m_md = epoch_md_sum / n_batches
            hist_ep.append(epoch + 1)
            hist_loss.append(m_loss)
            hist_kl.append(m_kl)
            hist_md.append(m_md)
            # One PNG update per finished epoch (not only when all epochs end).
            save_loss_plot_png(loss_plot_path, hist_ep, hist_loss, hist_kl, hist_md)
            print(f"Updated loss plot -> {loss_plot_path}")

        meta = {
            "img_size": args.img_size,
            "backbone_dim": student_wrap.embed_dim,
            "backbone_depth": student_wrap.depth,
            "epochs_done": epoch + 1,
        }
        ckpt = distill_trainable_state_dict(distill, meta=meta)
        ckpt["optimizer"] = opt.state_dict()
        ckpt["global_step"] = global_step
        ckpt[LOSS_PLOT_HISTORY_KEY] = pack_loss_plot_history(hist_ep, hist_loss, hist_kl, hist_md)
        torch.save(ckpt, args.output)
        print(f"Saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()

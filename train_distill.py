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
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
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


def print_runtime_context(
    device: torch.device,
    image_dir: str,
    ds_len: int,
    batch_size: int,
    num_workers: int,
    distill: nn.Module,
) -> None:
    """Where compute runs + loader settings (SSD/CPU decode vs GPU)."""
    pin = device.type == "cuda"
    print(f"image-dir: {image_dir}  |  dataset: {ds_len} images")
    print(f"DataLoader: num_workers={num_workers}, pin_memory={pin}")
    p0 = next(distill.parameters())
    print(f"model params: device={p0.device}, dtype={p0.dtype}")
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        print(f"CUDA: {torch.cuda.get_device_name(idx)}")
        cap = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
        print(f"CUDA VRAM: {cap:.1f} GiB total")
        print(f"CUDA mem allocated (now): {torch.cuda.memory_allocated(idx) / 1e6:.0f} MB")
    elif device.type == "mps":
        print("MPS: Apple GPU (no nvidia-smi)")
    else:
        print("CPU: training will be much slower; check torch CUDA build + drivers if you expect a GPU.")


def profile_one_batch(
    loader: DataLoader,
    distill: nn.Module,
    opt: torch.optim.Optimizer,
    device: torch.device,
    pin_memory: bool,
    use_amp: bool,
) -> None:
    """
    Time: dataloader vs forward+backward (no optimizer.step — avoids changing weights).
    Large 'dataloader' => disk decode / CPU / num_workers; large 'forward+backward' => GPU compute bound.
    """
    it = iter(loader)
    t0 = time.perf_counter()
    imgs, _paths = next(it)
    t1 = time.perf_counter()
    imgs = imgs.to(device, non_blocking=pin_memory)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()

    distill.train()
    opt.zero_grad(set_to_none=True)
    with autocast(enabled=use_amp):
        out = distill(imgs)
    loss = out["loss"]
    # One-step timing: no GradScaler (avoids mutating training scaler before the loop).
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    t3 = time.perf_counter()
    opt.zero_grad(set_to_none=True)

    dl_ms = (t1 - t0) * 1000
    h2d_ms = (t2 - t1) * 1000
    comp_ms = (t3 - t2) * 1000
    total_ms = (t3 - t0) * 1000
    print(
        "[profile-batch] times (ms):  dataloader_next={:.1f}  h2d+sync={:.1f}  forward+backward+sync={:.1f}  (sum={:.1f})".format(
            dl_ms, h2d_ms, comp_ms, total_ms
        )
    )
    print(f"[profile-batch] imgs: device={imgs.device}, is_cuda={imgs.is_cuda}")
    if dl_ms > comp_ms * 1.5:
        print("[profile-batch] hint: dataloader >> compute — try faster disk, more num_workers, or smaller img_size.")
    elif comp_ms > dl_ms * 2 and device.type == "cuda":
        print("[profile-batch] hint: GPU compute dominates — expected for ViT-H teacher + 1024; batch_size>1 if VRAM allows.")


def collate_distill(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    paths = [b[1] for b in batch]
    return imgs, paths


def make_distill_loader(
    ds: DistillImageFolder,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    *,
    seed: int,
    epoch_index: int,
) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed + epoch_index * 1_000_003)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_distill,
        pin_memory=device.type == "cuda",
        generator=g,
    )


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
    p.add_argument(
        "--samples-per-epoch",
        type=int,
        default=None,
        help="Each epoch: random sample this many images from the full pool (no replacement). "
        "Omits to use every image once per epoch (full pass).",
    )
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
    p.add_argument(
        "--grad-accum-steps",
        type=int,
        default=8,
        help="Micro-batches per optimizer step (effective batch = batch_size * this). Loss scaled as 1/steps per backward.",
    )
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
    p.add_argument(
        "--profile-batch",
        action="store_true",
        help="Print runtime (CPU/GPU/dataloader) and time one batch: disk+decode vs forward+backward",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed: torch/random, DataLoader shuffle, and (with --samples-per-epoch) subset indices per epoch",
    )
    p.add_argument(
        "--amp",
        action="store_true",
        help="CUDA mixed precision (fp16 forward under autocast, fp32 master weights + GradScaler). Ignored on non-CUDA.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.grad_accum_steps < 1:
        raise SystemExit("--grad-accum-steps must be >= 1")
    if args.samples_per_epoch is not None and args.samples_per_epoch < 1:
        raise SystemExit("--samples-per-epoch must be >= 1")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = pick_device()
    use_amp = bool(args.amp and device.type == "cuda")
    if args.amp and device.type != "cuda":
        print("Warning: --amp is only supported on CUDA; training in fp32.")
    print(f"device => {device}")
    if device.type == "cuda":
        print(f"         ({torch.cuda.get_device_name(0)})")
    print(f"AMP (fp16 autocast): {'on' if use_amp else 'off'}")

    ds = DistillImageFolder(
        args.image_dir,
        img_size=args.img_size,
        samples_per_epoch=args.samples_per_epoch,
        subset_seed=args.seed,
    )
    if args.samples_per_epoch is not None:
        print(
            f"samples_per_epoch={args.samples_per_epoch}  |  pool_size={ds.pool_size}  |  subset_seed={args.seed} "
            "(deterministic subset per global epoch index; not a full pass per epoch)"
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
    scaler: GradScaler | None = GradScaler() if use_amp else None

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
        ck_seed = (ck.get("meta") or {}).get("seed")
        if ck_seed is not None and int(ck_seed) != args.seed:
            print(
                f"Warning: --seed ({args.seed}) != checkpoint meta seed ({int(ck_seed)}); "
                "subset/DataLoader order may differ from the run that wrote this file."
            )
        ck_amp = (ck.get("meta") or {}).get("amp")
        if ck_amp is not None and bool(ck_amp) != use_amp:
            print(
                f"Warning: --amp ({use_amp}) != checkpoint meta amp ({bool(ck_amp)}); "
                "optimizer/scaler state may be mismatched."
            )
        if scaler is not None and isinstance(ck.get("scaler"), dict):
            scaler.load_state_dict(ck["scaler"])

    if args.samples_per_epoch is not None:
        ds.resample_epoch(start_epoch)
    loader = make_distill_loader(
        ds,
        args.batch_size,
        args.num_workers,
        device,
        seed=args.seed,
        epoch_index=start_epoch,
    )

    os.makedirs(Path(args.output).parent or ".", exist_ok=True)
    out_path = Path(args.output)
    step_ckpt = out_path.with_name(out_path.stem + "_latest.pt")
    loss_plot_path = Path(args.loss_plot) if args.loss_plot else out_path.with_name(out_path.stem + "_loss.png")
    print(f"loss plot (PNG) => {loss_plot_path}")
    eff_bs = args.batch_size * args.grad_accum_steps
    print(
        f"grad_accum_steps={args.grad_accum_steps}  |  effective batch size (for optimizer) ≈ {eff_bs}"
    )

    if args.profile_batch:
        print_runtime_context(device, args.image_dir, len(ds), args.batch_size, args.num_workers, distill)
        profile_one_batch(
            loader, distill, opt, device, pin_memory=device.type == "cuda", use_amp=use_amp
        )
        print("--- training starts ---")

    end_epoch = start_epoch + args.epochs
    for epoch in range(start_epoch, end_epoch):
        # First epoch of this session uses ds/loader from resample_epoch(start_epoch) before the loop.
        # Resampling here every epoch would discard that subset and break --profile-batch vs epoch 0 alignment.
        if args.samples_per_epoch is not None and epoch > start_epoch:
            ds.resample_epoch(epoch)
            loader = make_distill_loader(
                ds,
                args.batch_size,
                args.num_workers,
                device,
                seed=args.seed,
                epoch_index=epoch,
            )
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
        accum = args.grad_accum_steps
        accum_count = 0
        opt.zero_grad(set_to_none=True)
        for imgs, _paths in pbar:
            imgs = imgs.to(device)
            with autocast(enabled=use_amp):
                out = distill(imgs)
            loss = out["loss"]
            if scaler is not None:
                scaler.scale(loss / accum).backward()
            else:
                (loss / accum).backward()
            accum_count += 1
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
                accum=f"{accum_count}/{accum}",
            )
            if accum_count >= accum:
                if scaler is not None:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                accum_count = 0
            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                meta = {
                    "img_size": args.img_size,
                    "backbone_dim": student_wrap.embed_dim,
                    "backbone_depth": student_wrap.depth,
                    "epochs_done": epoch,
                    "grad_accum_steps": args.grad_accum_steps,
                    "samples_per_epoch": args.samples_per_epoch,
                    "seed": args.seed,
                    "amp": use_amp,
                    "note": "mid-epoch snapshot",
                }
                ckpt = distill_trainable_state_dict(distill, meta=meta)
                ckpt["optimizer"] = opt.state_dict()
                ckpt["global_step"] = global_step
                ckpt[LOSS_PLOT_HISTORY_KEY] = pack_loss_plot_history(hist_ep, hist_loss, hist_kl, hist_md)
                if scaler is not None:
                    ckpt["scaler"] = scaler.state_dict()
                torch.save(ckpt, step_ckpt)
                print(f"Saved step checkpoint -> {step_ckpt}")

        if accum_count > 0:
            scale = accum / accum_count
            if scaler is not None:
                scaler.unscale_(opt)
            for p in distill.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)

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
            "grad_accum_steps": args.grad_accum_steps,
            "samples_per_epoch": args.samples_per_epoch,
            "seed": args.seed,
            "amp": use_amp,
        }
        ckpt = distill_trainable_state_dict(distill, meta=meta)
        ckpt["optimizer"] = opt.state_dict()
        ckpt["global_step"] = global_step
        ckpt[LOSS_PLOT_HISTORY_KEY] = pack_loss_plot_history(hist_ep, hist_loss, hist_kl, hist_md)
        if scaler is not None:
            ckpt["scaler"] = scaler.state_dict()
        torch.save(ckpt, args.output)
        print(f"Saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()

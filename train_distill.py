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
from datetime import datetime

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rdlnet.data import CocoTrain2017BoxPrompts, DistillImageFolder
from rdlnet.device import pick_device
from rdlnet.distill import (
    DistillConfig,
    LightSAMMultiplexDistillation,
    LightSAMMultiplexDistillationDecoderKD,
    build_image_encoder_student_table2,
    build_sam_mask_decoder,
    build_sam_prompt_encoder,
    build_teacher_image_encoder_vit_h_with_neck,
    create_distillation_setup,
    distill_trainable_state_dict,
    load_distill_trainable_state_dict,
    load_sam_submodules_from_checkpoint,
)
from rdlnet.sam_backbone import RDLNetSAMEncoder


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
    batch = next(it)
    t1 = time.perf_counter()
    if isinstance(batch, (tuple, list)) and len(batch) >= 2 and torch.is_tensor(batch[1]):
        # COCO box prompts: (imgs, boxes, meta)
        imgs, boxes = batch[0], batch[1]
        imgs = imgs.to(device, non_blocking=pin_memory)
        boxes = boxes.to(device, non_blocking=pin_memory)
    else:
        imgs, _paths = batch
        imgs = imgs.to(device, non_blocking=pin_memory)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()

    distill.train()
    opt.zero_grad(set_to_none=True)
    with autocast("cuda", enabled=use_amp):
        out = distill(imgs, boxes) if "boxes" in locals() else distill(imgs)
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


def collate_distill_coco_box(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    boxes = torch.stack([b[1] for b in batch], dim=0)  # [B,4]
    metas = [b[2] for b in batch]
    return imgs, boxes, metas


@torch.no_grad()
def validate_one_epoch_coco(
    loader: DataLoader,
    distill: nn.Module,
    device: torch.device,
    *,
    use_amp: bool,
) -> tuple[float, float, float]:
    distill.eval()
    loss_sum = 0.0
    kl_sum = 0.0
    md_sum = 0.0
    n = 0
    for imgs, boxes, _meta in loader:
        imgs = imgs.to(device)
        boxes = boxes.to(device)
        with autocast("cuda", enabled=use_amp):
            out = distill(imgs, boxes)
        loss_sum += float(out["loss"].detach())
        kl_sum += float(out["loss_kl"].detach())
        md_sum += float(out["loss_md"].detach())
        n += 1
    if n == 0:
        return float("inf"), float("inf"), float("inf")
    return loss_sum / n, kl_sum / n, md_sum / n


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1: distill Light-SAM student from SAM ViT-H")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image-dir", type=str, default=None, help="Directory of RGB images (recursive) [encoder-only distill]")
    src.add_argument("--coco", action="store_true", help="Use extracted COCO train2017/ + instances json (box prompts + decoder KD)")
    p.add_argument("--coco-train-dir", type=str, default="dataset/coco/train2017", help="Extracted COCO train2017/ directory")
    p.add_argument("--coco-instances-json", type=str, default="dataset/coco/annotations/instances_train2017.json")
    p.add_argument("--coco-val-dir", type=str, default="dataset/coco/val2017", help="Extracted COCO val2017/ directory")
    p.add_argument("--coco-val-instances-json", type=str, default="dataset/coco/annotations/instances_val2017.json")
    p.add_argument(
        "--samples-per-epoch",
        type=int,
        default=None,
        help="Each epoch: random sample this many images from the full pool (no replacement). "
        "Omits to use every image once per epoch (full pass).",
    )
    p.add_argument("--teacher-checkpoint", type=str, required=True, help="sam_vit_h_4b8939.pth")
    p.add_argument(
        "--output",
        type=str,
        default="output/distill",
        help="Output root directory. Each run creates a new timestamped subfolder under this directory.",
    )
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


def _make_run_dir(output_root: str | Path) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Always create a unique folder per run (timestamp-based).
    return root / stamp


def _resolve_in_run_dir(run_dir: Path, maybe_path: str | None, default_name: str) -> Path:
    if not maybe_path:
        return run_dir / default_name
    p = Path(maybe_path)
    return p if p.is_absolute() else (run_dir / p)


def _make_tb_writer(run_dir: Path) -> SummaryWriter:
    tb_dir = run_dir / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(tb_dir))


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

    cfg = DistillConfig(
        temperature=args.temperature,
        weight_kl=args.weight_kl,
        weight_md=args.weight_md,
        teacher_checkpoint=args.teacher_checkpoint,
    )

    if args.coco:
        ds = CocoTrain2017BoxPrompts(
            images=args.coco_train_dir,
            instances_json=args.coco_instances_json,
            img_size=args.img_size,
            seed=args.seed,
            instances_per_image=1,
        )
        student_image_encoder = build_image_encoder_student_table2(img_size=args.img_size)
        teacher_image_encoder = build_teacher_image_encoder_vit_h_with_neck()
        teacher_prompt = build_sam_prompt_encoder(img_size=args.img_size)
        student_prompt = build_sam_prompt_encoder(img_size=args.img_size)
        teacher_mask = build_sam_mask_decoder()
        student_mask = build_sam_mask_decoder()
        load_sam_submodules_from_checkpoint(
            teacher_image_encoder=teacher_image_encoder,
            teacher_prompt_encoder=teacher_prompt,
            teacher_mask_decoder=teacher_mask,
            student_prompt_encoder=student_prompt,
            student_mask_decoder=student_mask,
            checkpoint_path=args.teacher_checkpoint,
            strict=False,
        )
        distill = LightSAMMultiplexDistillationDecoderKD(
            teacher_image_encoder=teacher_image_encoder,
            student_image_encoder=student_image_encoder,
            teacher_prompt_encoder=teacher_prompt,
            student_prompt_encoder=student_prompt,
            teacher_mask_decoder=teacher_mask,
            student_mask_decoder=student_mask,
            cfg=cfg,
        ).to(device)
        trainable_params = [p for p in distill.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        student_wrap = None
    else:
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
        distill = create_distillation_setup(
            student_wrap.encoder,
            teacher_checkpoint=args.teacher_checkpoint,
            cfg=cfg,
        ).to(device)
        opt = torch.optim.AdamW(distill.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler: GradScaler | None = GradScaler("cuda") if use_amp else None

    start_epoch = 0
    global_step = 0
    best_val_kd = float("inf")
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
        print(f"Resumed from {args.resume}: epochs_done={start_epoch}, global_step={global_step}")
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
        if isinstance(ck.get("best_val_kd"), (int, float)):
            best_val_kd = float(ck["best_val_kd"])

    if args.samples_per_epoch is not None and not args.coco:
        ds.resample_epoch(start_epoch)
    if args.coco:
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_distill_coco_box,
            pin_memory=device.type == "cuda",
        )
        val_ds = CocoTrain2017BoxPrompts(
            images=args.coco_val_dir,
            instances_json=args.coco_val_instances_json,
            img_size=args.img_size,
            seed=args.seed + 1337,
            instances_per_image=1,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_distill_coco_box,
            pin_memory=device.type == "cuda",
        )
    else:
        loader = make_distill_loader(
            ds,
            args.batch_size,
            args.num_workers,
            device,
            seed=args.seed,
            epoch_index=start_epoch,
        )

    run_dir = _make_run_dir(args.output)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "checkpoint.pt"
    step_ckpt = run_dir / "latest.pt"
    best_ckpt = run_dir / "best.pt"
    print(f"run_dir => {run_dir}")
    writer = _make_tb_writer(run_dir)
    print(f"tensorboard logdir => {run_dir / 'tb'}")
    eff_bs = args.batch_size * args.grad_accum_steps
    print(
        f"grad_accum_steps={args.grad_accum_steps}  |  effective batch size (for optimizer) ≈ {eff_bs}"
    )

    if args.profile_batch:
        print_runtime_context(
            device,
            args.image_dir or f"COCO(train2017={args.coco_train_dir})",
            len(ds),
            args.batch_size,
            args.num_workers,
            distill,
        )
        profile_one_batch(
            loader, distill, opt, device, pin_memory=device.type == "cuda", use_amp=use_amp
        )
        print("--- training starts ---")

    end_epoch = start_epoch + args.epochs
    try:
        for epoch in range(start_epoch, end_epoch):
            # First epoch of this session uses ds/loader from resample_epoch(start_epoch) before the loop.
            # Resampling here every epoch would discard that subset and break --profile-batch vs epoch 0 alignment.
            if args.samples_per_epoch is not None and (not args.coco) and epoch > start_epoch:
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
            for batch in pbar:
                if args.coco:
                    imgs, boxes, _meta = batch
                    imgs = imgs.to(device)
                    boxes = boxes.to(device)
                else:
                    imgs, _paths = batch
                    imgs = imgs.to(device)
                with autocast("cuda", enabled=use_amp):
                    out = distill(imgs, boxes) if args.coco else distill(imgs)
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
                writer.add_scalar("train/kd_step", float(loss.detach()), global_step)
                writer.add_scalar("train/kl_step", float(out["loss_kl"].detach()), global_step)
                writer.add_scalar("train/md_step", float(out["loss_md"].detach()), global_step)
                try:
                    lr = float(opt.param_groups[0]["lr"])
                    writer.add_scalar("train/lr", lr, global_step)
                except Exception:
                    pass
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
                        "backbone_dim": getattr(student_wrap, "embed_dim", None),
                        "backbone_depth": getattr(student_wrap, "depth", None),
                        "epochs_done": epoch,
                        "grad_accum_steps": args.grad_accum_steps,
                        "samples_per_epoch": args.samples_per_epoch,
                        "seed": args.seed,
                        "amp": use_amp,
                        "dataset": "coco" if args.coco else "folder",
                        "run_dir": str(run_dir),
                        "note": "mid-epoch snapshot",
                    }
                    ckpt = distill_trainable_state_dict(distill, meta=meta)
                    ckpt["optimizer"] = opt.state_dict()
                    ckpt["global_step"] = global_step
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
            writer.add_scalar("train/kd_epoch", m_loss, epoch + 1)
            writer.add_scalar("train/kl_epoch", m_kl, epoch + 1)
            writer.add_scalar("train/md_epoch", m_md, epoch + 1)

        if args.coco:
            val_kd, val_kl, val_md = validate_one_epoch_coco(val_loader, distill, device, use_amp=use_amp)
            print(f"[val] kd={val_kd:.4f}  kl={val_kl:.4f}  md={val_md:.4f}  (best_kd={best_val_kd:.4f})")
            writer.add_scalar("val/kd_epoch", val_kd, epoch + 1)
            writer.add_scalar("val/kl_epoch", val_kl, epoch + 1)
            writer.add_scalar("val/md_epoch", val_md, epoch + 1)
            if val_kd < best_val_kd:
                best_val_kd = val_kd
                meta_best = {
                    "img_size": args.img_size,
                    "epochs_done": epoch + 1,
                    "seed": args.seed,
                    "amp": use_amp,
                    "dataset": "coco",
                    "run_dir": str(run_dir),
                    "note": "best by val KD",
                    "val_kd": float(val_kd),
                    "val_kl": float(val_kl),
                    "val_md": float(val_md),
                }
                best_state = distill_trainable_state_dict(distill, meta=meta_best)
                best_state["best_val_kd"] = best_val_kd
                torch.save(best_state, best_ckpt)
                print(f"Saved best checkpoint -> {best_ckpt}")

        meta = {
            "img_size": args.img_size,
            "backbone_dim": getattr(student_wrap, "embed_dim", None),
            "backbone_depth": getattr(student_wrap, "depth", None),
            "epochs_done": epoch + 1,
            "grad_accum_steps": args.grad_accum_steps,
            "samples_per_epoch": args.samples_per_epoch,
            "seed": args.seed,
            "amp": use_amp,
            "dataset": "coco" if args.coco else "folder",
            "run_dir": str(run_dir),
        }
        ckpt = distill_trainable_state_dict(distill, meta=meta)
        ckpt["optimizer"] = opt.state_dict()
        ckpt["global_step"] = global_step
        ckpt["best_val_kd"] = best_val_kd
        if scaler is not None:
            ckpt["scaler"] = scaler.state_dict()
        torch.save(ckpt, out_path)
        print(f"Saved checkpoint to {out_path}")
    finally:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()

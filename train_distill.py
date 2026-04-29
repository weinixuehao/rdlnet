#!/usr/bin/env python3
"""
Stage 1: Light-SAM multiplex distillation (paper Sec. 3.1).

Example::

    python train_distill.py \\
        --teacher-checkpoint /path/to/sam_vit_h_4b8939.pth \\
        --output output/distill

Requires: Extracted COCO train/val images + instances json, plus SAM ViT-H weights.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.amp import autocast
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from tqdm.auto import tqdm

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rdlnet.data import CocoTrain2017BoxPrompts, RWMDPreprocessedPointPrompts, collate_distill_rwmd_points
from rdlnet.device import pick_device
from rdlnet.distill import (
    DistillConfig,
    LightSAMMultiplexDistillationDecoderKD,
    build_sam_vit_for_rdlnet_cfg,
    build_sam_mask_decoder,
    build_sam_prompt_encoder,
    build_teacher_image_encoder_vit_h_with_neck,
    distill_trainable_state_dict,
    load_distill_trainable_state_dict,
    load_sam_submodules_from_checkpoint,
)
from rdlnet.model import RDLNetConfig, apply_lite_preset


def _optimizer_step(
    opt: torch.optim.Optimizer,
) -> bool:
    """
    Returns True iff parameters were actually updated.
    BF16 autocast does not require GradScaler; optimizer.step() always runs.
    """
    opt.step()
    return True


def collate_distill_coco_box(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    boxes = torch.stack([b[1] for b in batch], dim=0)  # [B,4]
    metas = [b[2] for b in batch]
    return imgs, boxes, metas


def _draw_points_on_axes(ax, points_xy: torch.Tensor, point_labels: torch.Tensor):
    # points_xy: [P,2], labels: [P]
    try:
        pts = points_xy.detach().cpu()
        lbs = point_labels.detach().cpu()
        for (x, y), lb in zip(pts.tolist(), lbs.tolist()):
            if int(lb) < 0:
                continue
            if int(lb) == 1:
                ax.scatter([x], [y], s=40, c="lime", marker="x", linewidths=2)
            else:
                ax.scatter([x], [y], s=40, c="red", marker="x", linewidths=2)
    except Exception:
        return


def _chw_u8_hwc(images_chw: torch.Tensor):
    import numpy as np

    x = images_chw.detach().cpu()
    if x.dtype != torch.uint8:
        x = x.clamp(0, 255).to(torch.uint8)
    return np.transpose(x.numpy(), (1, 2, 0))


def _distill_vis_grid_u8(
    images: torch.Tensor,
    *,
    boxes_xyxy: torch.Tensor | None = None,
    points_xy: torch.Tensor | None = None,
    point_labels: torch.Tensor | None = None,
    low_res_logits: torch.Tensor,
    title: str,
    max_samples: int,
    mask_alpha: float = 0.45,
):
    """
    Build a visualization grid: [image+box | mask overlay].
    Returns RGB uint8 (H, W, 3) as numpy.
    """
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bsz = int(images.shape[0])
    b = min(bsz, int(max_samples))
    if b <= 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    # pick a single mask per sample (max over multimask heads)
    prob = low_res_logits.detach().float().sigmoid()
    if prob.ndim == 4:
        prob = prob.max(dim=1).values  # [B, h, w]

    # upscale prob to image resolution
    h_img = int(images.shape[-2])
    w_img = int(images.shape[-1])
    prob_up = F.interpolate(prob.unsqueeze(1), size=(h_img, w_img), mode="bilinear", align_corners=False).squeeze(1)

    if (boxes_xyxy is None) == (points_xy is None):
        raise ValueError("Provide exactly one of boxes_xyxy or points_xy for visualization")
    ncols = 2
    fig, axes = plt.subplots(b, ncols, figsize=(4.6 * ncols, 3.8 * b), squeeze=False)
    for i in range(b):
        rgb = _chw_u8_hwc(images[i])
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title("image + prompt")
        axes[i, 0].axis("off")

        if boxes_xyxy is not None:
            x1, y1, x2, y2 = [float(v) for v in boxes_xyxy[i].detach().cpu().tolist()]
            rect = plt.Rectangle((x1, y1), max(0.0, x2 - x1), max(0.0, y2 - y1), fill=False, linewidth=2.0, edgecolor="lime")
            axes[i, 0].add_patch(rect)
        else:
            _draw_points_on_axes(axes[i, 0], points_xy[i], point_labels[i])

        axes[i, 1].imshow(rgb)
        p = prob_up[i].detach().cpu().numpy()
        axes[i, 1].imshow(p, cmap="magma", alpha=float(mask_alpha), vmin=0.0, vmax=1.0)
        # Add a threshold contour to make the mask boundary obvious on document images.
        try:
            axes[i, 1].contour(p, levels=[0.5], colors=["cyan"], linewidths=1.5)
        except Exception:
            pass
        axes[i, 1].set_title("mask overlay")
        axes[i, 1].axis("off")

    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    fig.canvas.draw()
    # Matplotlib 3.8+ on Agg may not expose tostring_rgb(); use buffer_rgba() instead.
    rgba = np.asarray(fig.canvas.buffer_rgba())  # [H, W, 4] uint8
    buf = rgba[..., :3].copy()
    plt.close(fig)
    return buf


def _distill_vis_compare_grid_u8(
    images: torch.Tensor,
    *,
    boxes_xyxy: torch.Tensor | None = None,
    points_xy: torch.Tensor | None = None,
    point_labels: torch.Tensor | None = None,
    low_res_t: torch.Tensor,
    low_res_s: torch.Tensor,
    title: str,
    max_samples: int,
    mask_alpha: float = 0.45,
):
    """
    Build a comparison grid: [image+box | teacher overlay | student overlay].
    Returns RGB uint8 (H, W, 3) as numpy.
    """
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bsz = int(images.shape[0])
    b = min(bsz, int(max_samples))
    if b <= 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    if (boxes_xyxy is None) == (points_xy is None):
        raise ValueError("Provide exactly one of boxes_xyxy or points_xy for visualization")

    def _prob_up(low_res_logits: torch.Tensor) -> torch.Tensor:
        prob = low_res_logits.detach().float().sigmoid()
        if prob.ndim == 4:
            prob = prob.max(dim=1).values  # [B, h, w]
        h_img = int(images.shape[-2])
        w_img = int(images.shape[-1])
        return (
            F.interpolate(prob.unsqueeze(1), size=(h_img, w_img), mode="bilinear", align_corners=False)
            .squeeze(1)
            .clamp(0.0, 1.0)
        )

    t_up = _prob_up(low_res_t)
    s_up = _prob_up(low_res_s)

    ncols = 3
    fig, axes = plt.subplots(b, ncols, figsize=(4.6 * ncols, 3.8 * b), squeeze=False)
    for i in range(b):
        rgb = _chw_u8_hwc(images[i])

        axes[i, 0].imshow(rgb)
        if boxes_xyxy is not None:
            x1, y1, x2, y2 = [float(v) for v in boxes_xyxy[i].detach().cpu().tolist()]
            rect = plt.Rectangle((x1, y1), max(0.0, x2 - x1), max(0.0, y2 - y1), fill=False, linewidth=2.0, edgecolor="lime")
            axes[i, 0].add_patch(rect)
        else:
            _draw_points_on_axes(axes[i, 0], points_xy[i], point_labels[i])
        axes[i, 0].set_title("image + prompt")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(rgb)
        pt = t_up[i].detach().cpu().numpy()
        axes[i, 1].imshow(pt, cmap="magma", alpha=float(mask_alpha), vmin=0.0, vmax=1.0)
        try:
            axes[i, 1].contour(pt, levels=[0.5], colors=["cyan"], linewidths=1.5)
        except Exception:
            pass
        axes[i, 1].set_title("teacher mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(rgb)
        ps = s_up[i].detach().cpu().numpy()
        axes[i, 2].imshow(ps, cmap="magma", alpha=float(mask_alpha), vmin=0.0, vmax=1.0)
        try:
            axes[i, 2].contour(ps, levels=[0.5], colors=["cyan"], linewidths=1.5)
        except Exception:
            pass
        axes[i, 2].set_title("student mask")
        axes[i, 2].axis("off")

    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    fig.canvas.draw()
    # Matplotlib 3.8+ on Agg may not expose tostring_rgb(); use buffer_rgba() instead.
    rgba = np.asarray(fig.canvas.buffer_rgba())  # [H, W, 4] uint8
    buf = rgba[..., :3].copy()
    plt.close(fig)
    return buf


@torch.no_grad()
def validate_one_epoch_coco(
    loader: DataLoader,
    distill: nn.Module,
    device: torch.device,
    *,
    use_amp: bool,
    max_batches: int = 0,
) -> tuple[float, float, float]:
    distill.eval()
    loss_sum = 0.0
    kl_sum = 0.0
    md_sum = 0.0
    n = 0
    for imgs, boxes, _meta in loader:
        if max_batches and n >= int(max_batches):
            break
        imgs = imgs.to(device)
        boxes = boxes.to(device)
        with autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            out = distill(imgs, boxes_xyxy=boxes)
        loss_sum += float(out["loss"].detach())
        kl_sum += float(out["loss_kl"].detach())
        md_sum += float(out["loss_md"].detach())
        n += 1
    if n == 0:
        return float("inf"), float("inf"), float("inf")
    return loss_sum / n, kl_sum / n, md_sum / n


@torch.no_grad()
def validate_one_epoch_rwmd(
    loader: DataLoader,
    distill: nn.Module,
    device: torch.device,
    *,
    use_amp: bool,
    max_batches: int = 0,
) -> tuple[float, float, float]:
    distill.eval()
    loss_sum = 0.0
    kl_sum = 0.0
    md_sum = 0.0
    n = 0
    for imgs, points, point_labels, _meta in loader:
        if max_batches and n >= int(max_batches):
            break
        imgs = imgs.to(device)
        points = points.to(device)
        point_labels = point_labels.to(device)
        with autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            out = distill(imgs, points_xy=points, point_labels=point_labels)
        loss_sum += float(out["loss"].detach())
        kl_sum += float(out["loss_kl"].detach())
        md_sum += float(out["loss_md"].detach())
        n += 1
    if n == 0:
        return float("inf"), float("inf"), float("inf")
    return loss_sum / n, kl_sum / n, md_sum / n


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1: distill Light-SAM student from SAM ViT-H")
    p.add_argument(
        "--dataset",
        type=str,
        default="coco",
        choices=["coco", "rwmd"],
        help="Distillation dataset source. 'rwmd' expects preprocessed folders with img/ and mask/.",
    )
    p.add_argument("--coco-train-dir", type=str, default="dataset/coco/train2017", help="Extracted COCO train2017/ directory")
    p.add_argument("--coco-instances-json", type=str, default="dataset/coco/annotations/instances_train2017.json")
    p.add_argument("--coco-val-dir", type=str, default="dataset/coco/val2017", help="Extracted COCO val2017/ directory")
    p.add_argument("--coco-val-instances-json", type=str, default="dataset/coco/annotations/instances_val2017.json")
    p.add_argument("--rwmd-train-root", type=str, default="output/data/train_resize_aug", help="RWMD preprocessed train root (contains img/ and mask/)")
    p.add_argument("--rwmd-val-root", type=str, default="output/data/test_resize", help="RWMD preprocessed val/test root (contains img/ and mask/)")
    p.add_argument("--rwmd-max-points", type=int, default=8, help="Max positive points per sample (one per connected component, capped).")
    p.add_argument("--teacher-checkpoint", type=str, required=True, help="sam_vit_h_4b8939.pth")
    p.add_argument(
        "--output",
        type=str,
        default="output/distill",
        help="Output root directory. Each run creates a new timestamped subfolder under this directory.",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a stage-1 run directory (auto-loads checkpoint.pt) or a checkpoint.pt file. "
        "When resuming, training continues in the SAME run directory (no new folder). "
        "Note: best.pt is not resumable.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Total target epochs for this run. If resuming and checkpoint has epochs_done=E, "
        "this invocation trains remaining max(0, epochs - E) epochs.",
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
        "--max-grad-norm",
        type=float,
        default=4.0,
        help="Gradient clipping max norm. Set to 0 to disable.",
    )
    p.add_argument(
        "--tb-vis-interval",
        type=int,
        default=200,
        help="TensorBoard mask visualization interval in optimizer steps (0 disables).",
    )
    p.add_argument(
        "--tb-log-interval",
        type=int,
        default=20,
        help="TensorBoard scalar logging interval in optimizer steps (0 disables).",
    )
    p.add_argument(
        "--tb-vis-max-samples",
        type=int,
        default=4,
        help="Max samples (rows) per TensorBoard visualization grid.",
    )
    p.add_argument(
        "--train-max-batches",
        type=int,
        default=0,
        help="Cap train epoch length by number of DataLoader batches (0 = full epoch). Useful for quick curves/debug.",
    )
    p.add_argument(
        "--val-max-batches",
        type=int,
        default=0,
        help="Cap validation by number of DataLoader batches (0 = full validation set).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed: torch/random and DataLoader shuffle",
    )
    p.add_argument(
        "--amp",
        action="store_true",
        help="CUDA mixed precision (bf16 forward under autocast). Ignored on non-CUDA.",
    )
    p.add_argument(
        "--lite",
        type=int,
        default=40,
        choices=[10, 20, 40],
        help="Student ViT size (COCO and folder); must match train_rdlnet --lite for --distill-checkpoint; "
        "see apply_lite_preset in rdlnet/model.py.",
    )
    return p.parse_args()


def _make_run_dir(output_root: str | Path, *, lite: int) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Always create a unique folder per run (timestamp-based), but include key config in name.
    return root / f"{stamp}_lite{int(lite)}"


def _resolve_resume_path(resume: str | Path) -> tuple[Path, Path]:
    """
    Returns (run_dir, checkpoint_path).
    Accepts either:
      - <run_dir>/ (directory) -> <run_dir>/checkpoint.pt
      - <run_dir>/checkpoint.pt (file) -> run_dir = parent
    """
    p = Path(resume).expanduser().resolve()
    if p.is_dir():
        run_dir = p
        ckpt = run_dir / "checkpoint.pt"
    else:
        ckpt = p
        run_dir = ckpt.parent
    if not ckpt.is_file():
        raise SystemExit(f"--resume checkpoint not found: {ckpt}")
    return run_dir, ckpt


def _make_tb_writer(run_dir: Path) -> SummaryWriter:
    tb_dir = run_dir / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(tb_dir))


def main() -> None:
    args = parse_args()
    if args.grad_accum_steps < 1:
        raise SystemExit("--grad-accum-steps must be >= 1")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = pick_device()
    use_amp = bool(args.amp and device.type == "cuda")
    if args.amp and device.type != "cuda":
        print("Warning: --amp is only supported on CUDA; training in fp32.")
    if use_amp and not torch.cuda.is_bf16_supported():
        print("Warning: CUDA device does not support bf16 autocast; disabling --amp (falling back to fp32).")
        use_amp = False
    print(f"device => {device}")
    if device.type == "cuda":
        print(f"         ({torch.cuda.get_device_name(0)})")
    print(f"AMP (bf16 autocast): {'on' if use_amp else 'off'}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    cfg = DistillConfig(
        temperature=args.temperature,
        weight_kl=args.weight_kl,
        weight_md=args.weight_md,
        teacher_checkpoint=args.teacher_checkpoint,
    )

    rdl_cfg = RDLNetConfig(img_size=args.img_size, use_sam_pixel_norm=True)
    apply_lite_preset(rdl_cfg, int(args.lite))

    if args.dataset == "coco":
        ds = CocoTrain2017BoxPrompts(
            images=args.coco_train_dir,
            instances_json=args.coco_instances_json,
            img_size=args.img_size,
            seed=args.seed,
            instances_per_image=1,
        )
        val_ds = CocoTrain2017BoxPrompts(
            images=args.coco_val_dir,
            instances_json=args.coco_val_instances_json,
            img_size=args.img_size,
            seed=args.seed + 1337,
            instances_per_image=1,
        )
        collate_fn = collate_distill_coco_box
        val_collate_fn = collate_distill_coco_box
    else:
        ds = RWMDPreprocessedPointPrompts(
            root=args.rwmd_train_root,
            img_size=args.img_size,
            seed=args.seed,
            max_points=int(args.rwmd_max_points),
        )
        val_ds = RWMDPreprocessedPointPrompts(
            root=args.rwmd_val_root,
            img_size=args.img_size,
            seed=args.seed + 1337,
            max_points=int(args.rwmd_max_points),
        )
        collate_fn = collate_distill_rwmd_points
        val_collate_fn = collate_distill_rwmd_points
    student_image_encoder = build_sam_vit_for_rdlnet_cfg(rdl_cfg)
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
    print(
        f"distill student ({args.dataset}): lite={args.lite}  backbone embed_dim={rdl_cfg.backbone_dim}  "
        f"depth={rdl_cfg.backbone_depth}  (align: train_rdlnet.py --lite {args.lite})"
    )
    scheduler = MultiStepLR(opt, milestones=[40_000, 80_000, 120_000], gamma=0.1)

    start_epoch = 0
    global_step = 0
    best_val_kd = float("inf")
    run_dir: Path | None = None
    if args.resume:
        run_dir, resume_ckpt = _resolve_resume_path(args.resume)
        ck = torch.load(resume_ckpt, map_location=device)
        if not isinstance(ck, dict):
            raise TypeError("--resume checkpoint must be a dict")
        load_distill_trainable_state_dict(distill, ck)
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        if "scheduler" in ck and isinstance(ck["scheduler"], dict):
            scheduler.load_state_dict(ck["scheduler"])
        meta = ck.get("meta") or {}
        step_unit = meta.get("step_unit")
        if step_unit != "update":
            raise SystemExit(
                "This run uses update-step semantics (global_step == optimizer updates). "
                f"Refusing to resume checkpoint with step_unit={step_unit!r}."
            )
        start_epoch = int(meta.get("epochs_done", 0))
        global_step = int(ck.get("global_step", 0))
        print(f"Resumed from {resume_ckpt}: run_dir={run_dir}  epochs_done={start_epoch}, global_step={global_step}")
        ck_seed = (ck.get("meta") or {}).get("seed")
        if ck_seed is not None and int(ck_seed) != args.seed:
            print(
                f"Warning: --seed ({args.seed}) != checkpoint meta seed ({int(ck_seed)}); "
                "DataLoader order may differ from the run that wrote this file."
            )
        ck_amp = (ck.get("meta") or {}).get("amp")
        if ck_amp is not None and bool(ck_amp) != use_amp:
            print(
                f"Warning: --amp ({use_amp}) != checkpoint meta amp ({bool(ck_amp)}); "
                "optimizer state may be mismatched."
            )
        if isinstance(ck.get("best_val_kd"), (int, float)):
            best_val_kd = float(ck["best_val_kd"])

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=val_collate_fn,
        pin_memory=device.type == "cuda",
    )

    if run_dir is None:
        run_dir = _make_run_dir(args.output, lite=int(args.lite))
        run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "checkpoint.pt"
    best_ckpt = run_dir / "best.pt"
    print(f"run_dir => {run_dir}")
    writer = _make_tb_writer(run_dir)
    print(f"tensorboard logdir => {run_dir / 'tb'}")
    eff_bs = args.batch_size * args.grad_accum_steps
    print(
        f"grad_accum_steps={args.grad_accum_steps}  |  effective batch size (for optimizer) ≈ {eff_bs}"
    )

    end_epoch = int(args.epochs)
    if end_epoch < start_epoch:
        print(f"Warning: --epochs ({end_epoch}) < epochs_done ({start_epoch}); nothing to do.")
        return
    if end_epoch == start_epoch:
        print(f"Nothing to do: already at epochs_done == --epochs == {end_epoch}.")
        return
    try:
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
            accum = args.grad_accum_steps
            accum_count = 0
            opt.zero_grad(set_to_none=True)
            step_loss_sum = 0.0
            step_kl_sum = 0.0
            step_md_sum = 0.0
            step_n = 0
            t_last_update = time.perf_counter()
            for batch in pbar:
                if args.train_max_batches and n_batches >= int(args.train_max_batches):
                    break
                if args.dataset == "coco":
                    imgs, boxes, _meta = batch
                    imgs = imgs.to(device)
                    boxes = boxes.to(device)
                    prompt_kwargs = {"boxes_xyxy": boxes}
                else:
                    imgs, points, point_labels, _meta = batch
                    imgs = imgs.to(device)
                    points = points.to(device)
                    point_labels = point_labels.to(device)
                    prompt_kwargs = {"points_xy": points, "point_labels": point_labels}
                with autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                    out = distill(imgs, **prompt_kwargs)
                loss = out["loss"]
                (loss / accum).backward()
                accum_count += 1
                epoch_loss_sum += float(loss.detach())
                epoch_kl_sum += float(out["loss_kl"].detach())
                epoch_md_sum += float(out["loss_md"].detach())
                n_batches += 1
                step_loss_sum += float(loss.detach())
                step_kl_sum += float(out["loss_kl"].detach())
                step_md_sum += float(out["loss_md"].detach())
                step_n += 1
                pbar.set_postfix(
                    step=n_batches,
                    loss=f"{loss.item():.4f}",
                    kl=f"{out['loss_kl'].item():.4f}",
                    md=f"{out['loss_md'].item():.4f}",
                    accum=f"{accum_count}/{accum}",
                )
                if accum_count >= accum:
                    if args.max_grad_norm and args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(args.max_grad_norm))
                    did_step = _optimizer_step(opt)
                    if did_step:
                        scheduler.step()
                    opt.zero_grad(set_to_none=True)
                    accum_count = 0
                    if did_step:
                        dt = time.perf_counter() - t_last_update
                        t_last_update = time.perf_counter()
                        global_step += 1
                        try:
                            lr = float(opt.param_groups[0]["lr"])
                            writer.add_scalar("train/lr", lr, global_step)
                        except Exception:
                            pass
                        if args.tb_log_interval and args.tb_log_interval > 0 and (global_step % int(args.tb_log_interval) == 0):
                            denom = max(1, int(step_n))
                            writer.add_scalar("train/kd", step_loss_sum / denom, global_step)
                            writer.add_scalar("train/kl", step_kl_sum / denom, global_step)
                            writer.add_scalar("train/md", step_md_sum / denom, global_step)
                            if dt > 0:
                                writer.add_scalar("time/update_sec", float(dt), global_step)
                            step_loss_sum = 0.0
                            step_kl_sum = 0.0
                            step_md_sum = 0.0
                            step_n = 0
                        if args.tb_vis_interval and args.tb_vis_interval > 0 and (global_step % int(args.tb_vis_interval) == 0):
                            distill.eval()
                            with torch.no_grad():
                                vis = distill.predict_low_res_logits(imgs, **prompt_kwargs)
                            grid = _distill_vis_compare_grid_u8(
                                imgs.detach().cpu(),
                                boxes_xyxy=(boxes.detach().cpu() if args.dataset == "coco" else None),
                                points_xy=(points.detach().cpu() if args.dataset != "coco" else None),
                                point_labels=(point_labels.detach().cpu() if args.dataset != "coco" else None),
                                low_res_t=vis["low_res_t"].detach().cpu(),
                                low_res_s=vis["low_res_s"].detach().cpu(),
                                title=f"train masks (epoch {epoch + 1}, step {global_step})",
                                max_samples=int(args.tb_vis_max_samples),
                            )
                            writer.add_image(
                                f"train/masks/compare/step_{global_step:08d}",
                                grid.transpose(2, 0, 1),
                                global_step=global_step,
                                dataformats="CHW",
                            )
                            distill.train()

            if accum_count > 0:
                scale = accum / accum_count
                for p in distill.parameters():
                    if p.grad is not None:
                        p.grad.mul_(scale)
                if args.max_grad_norm and args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(args.max_grad_norm))
                did_step = _optimizer_step(opt)
                if did_step:
                    scheduler.step()
                opt.zero_grad(set_to_none=True)
                if did_step:
                    global_step += 1
                    try:
                        lr = float(opt.param_groups[0]["lr"])
                        writer.add_scalar("train/lr", lr, global_step)
                    except Exception:
                        pass

            if args.dataset == "coco":
                val_kd, val_kl, val_md = validate_one_epoch_coco(
                    val_loader,
                    distill,
                    device,
                    use_amp=use_amp,
                    max_batches=int(args.val_max_batches),
                )
            else:
                val_kd, val_kl, val_md = validate_one_epoch_rwmd(
                    val_loader,
                    distill,
                    device,
                    use_amp=use_amp,
                    max_batches=int(args.val_max_batches),
                )
            print(f"[val] kd={val_kd:.4f}  kl={val_kl:.4f}  md={val_md:.4f}  (best_kd={best_val_kd:.4f})")
            writer.add_scalar("val/kd_epoch", val_kd, epoch + 1)
            writer.add_scalar("val/kl_epoch", val_kl, epoch + 1)
            writer.add_scalar("val/md_epoch", val_md, epoch + 1)
            if args.tb_vis_max_samples and int(args.tb_vis_max_samples) > 0:
                distill.eval()
                try:
                    vb = next(iter(val_loader))
                    if args.dataset == "coco":
                        vimgs, vboxes, _vmeta = vb
                        vimgs = vimgs.to(device)
                        vboxes = vboxes.to(device)
                        v_prompt_kwargs = {"boxes_xyxy": vboxes}
                        v_points = None
                        v_point_labels = None
                    else:
                        vimgs, v_points, v_point_labels, _vmeta = vb
                        vimgs = vimgs.to(device)
                        v_points = v_points.to(device)
                        v_point_labels = v_point_labels.to(device)
                        v_prompt_kwargs = {"points_xy": v_points, "point_labels": v_point_labels}
                        vboxes = None
                    with torch.no_grad():
                        vis = distill.predict_low_res_logits(vimgs, **v_prompt_kwargs)
                    grid = _distill_vis_compare_grid_u8(
                        vimgs.detach().cpu(),
                        boxes_xyxy=(vboxes.detach().cpu() if args.dataset == "coco" else None),
                        points_xy=(v_points.detach().cpu() if args.dataset != "coco" else None),
                        point_labels=(v_point_labels.detach().cpu() if args.dataset != "coco" else None),
                        low_res_t=vis["low_res_t"].detach().cpu(),
                        low_res_s=vis["low_res_s"].detach().cpu(),
                        title=f"val masks (epoch {epoch + 1})",
                        max_samples=int(args.tb_vis_max_samples),
                    )
                    writer.add_image(
                        f"val/masks/compare/epoch_{epoch + 1:04d}",
                        grid.transpose(2, 0, 1),
                        global_step=epoch + 1,
                        dataformats="CHW",
                    )
                except Exception as e:
                    print(f"Warning: failed to log val mask visualization: {e}")
                finally:
                    distill.train()
            if val_kd < best_val_kd:
                best_val_kd = val_kd
                meta_best = {
                    "img_size": args.img_size,
                    "backbone_dim": rdl_cfg.backbone_dim,
                    "backbone_depth": rdl_cfg.backbone_depth,
                    "lite": int(args.lite),
                    "epochs_done": epoch + 1,
                    "epochs_target": int(args.epochs),
                    "seed": args.seed,
                    "amp": use_amp,
                    "dataset": str(args.dataset),
                    "run_dir": str(run_dir),
                    "note": "best by val KD",
                    "step_unit": "update",
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
                "backbone_dim": rdl_cfg.backbone_dim,
                "backbone_depth": rdl_cfg.backbone_depth,
                "lite": int(args.lite),
                "epochs_done": epoch + 1,
                "epochs_target": int(args.epochs),
                "grad_accum_steps": args.grad_accum_steps,
                "seed": args.seed,
                "amp": use_amp,
                "dataset": str(args.dataset),
                "run_dir": str(run_dir),
                "step_unit": "update",
            }
            ckpt = distill_trainable_state_dict(distill, meta=meta)
            ckpt["optimizer"] = opt.state_dict()
            ckpt["scheduler"] = scheduler.state_dict()
            ckpt["global_step"] = global_step
            ckpt["best_val_kd"] = best_val_kd
            torch.save(ckpt, out_path)
            print(f"Saved checkpoint to {out_path}")
    finally:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()

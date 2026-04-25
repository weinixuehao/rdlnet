#!/usr/bin/env python3
"""
Stage 2: Train full RDLNet on document localization annotations (paper losses, Sec. 3.4).

Initialize the SAM student backbone from stage-1 ``distill_stage1.pt`` (recommended).

Example (manifest JSON)::

    python train_rdlnet.py \\
        --annotations data/annotations.json \\
        --image-root data/images \\
        --distill-checkpoint checkpoints/distill_stage1.pt \\
        --output output/rdlnet

Example (RWMD preprocessed ``train_resize`` from ``data_preprocessing_rwdm_1``)::

    python train_rdlnet.py \\
        --rwmd-root path/to/out/train_resize \\
        --num-classes 2 \\
        --distill-checkpoint checkpoints/distill_stage1.pt \\
        --output output/rdlnet

Each fresh run creates ``<output>/<YYYYMMDD_HHMMSS>/`` with ``rdlnet.pt`` (each epoch
end), ``rdlnet_best.pt``, and ``tensorboard/``. Resume with ``--resume <that folder>``.

See ``rdlnet.data.doc_json`` for the manifest format and :class:`RWMDLabelMeDataset`.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
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

# Fixed names inside each run directory (--output/<timestamp>/).
CKPT_MAIN = "rdlnet.pt"
CKPT_BEST = "rdlnet_best.pt"
TB_SUBDIR = "tensorboard"


@dataclass
class LossSums:
    total: float = 0.0
    cls: float = 0.0
    dist: float = 0.0
    dice: float = 0.0
    mask: float = 0.0
    n: int = 0

    def update(self, loss: torch.Tensor, logs: dict[str, torch.Tensor]) -> None:
        self.total += float(loss.detach())
        self.cls += float(logs["loss_cls"].item())
        self.dist += float(logs["loss_dist"].item())
        self.dice += float(logs["loss_dice"].item())
        self.mask += float(logs["loss_mask"].item())
        self.n += 1

    def averages(self) -> dict[str, float]:
        denom = max(self.n, 1)
        return {
            "total": self.total / denom,
            "cls": self.cls / denom,
            "dist": self.dist / denom,
            "dice": self.dice / denom,
            "mask": self.mask / denom,
        }


def _should_stop(i: int, max_batches: int) -> bool:
    return max_batches > 0 and (i + 1) >= max_batches


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


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    images = batch["images"].to(device)
    tgt_labels = [t.to(device) for t in batch["tgt_labels"]]
    tgt_masks = [t.to(device) for t in batch["tgt_masks"]]
    tgt_points = [t.to(device) for t in batch["tgt_points"]]
    return images, tgt_labels, tgt_masks, tgt_points


def _forward_and_loss(
    model: torch.nn.Module,
    criterion: RDLNetLoss,
    images: torch.Tensor,
    tgt_labels: list[torch.Tensor],
    tgt_masks: list[torch.Tensor],
    tgt_points: list[torch.Tensor],
    *,
    device_type: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
    with torch.amp.autocast(device_type=device_type, enabled=use_amp, dtype=amp_dtype):
        out = model(images)
        loss, logs = criterion(
            out["pred_logits"],
            out["pred_masks"],
            out["pred_points"],
            tgt_labels,
            tgt_masks,
            tgt_points,
        )
    return out, loss, logs


def _log_compare_grid_to_tb(
    tb: Any,
    images: torch.Tensor,
    out: dict[str, torch.Tensor],
    tgt_labels: list[torch.Tensor],
    tgt_masks: list[torch.Tensor],
    tgt_points: list[torch.Tensor],
    *,
    tag: str,
    global_step: int,
    max_samples: int,
) -> None:
    if max_samples <= 0:
        return
    grid_u8 = train_compare_grid_u8(
        images.detach().cpu(),
        {k: v.detach().cpu() for k, v in out.items()},
        [t.detach().cpu() for t in tgt_labels],
        [t.detach().cpu() for t in tgt_masks],
        [t.detach().cpu() for t in tgt_points],
        max_samples=max_samples,
    )
    chw = np.transpose(grid_u8, (2, 0, 1))
    tb.add_image(tag, chw, global_step=global_step, dataformats="CHW")


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
        "--val-rwmd-root",
        type=str,
        default=None,
        help="Optional RWMD preprocessed folder for validation (img/, mask/, label_points_resize.json).",
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
    p.add_argument(
        "--output",
        type=str,
        default="output/rdlnet",
        help="Run root directory: new runs create <output>/<YYYYMMDD_HHMMSS>/ with ckpts + tensorboard/",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a previous run folder (same timestamp dir under --output), not a .pt path",
    )
    p.add_argument("--epochs", type=int, default=50, help="Additional epochs when resuming, or total epochs from scratch")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation: optimizer step every N micro-batches (effective batch ≈ batch_size × N). 1 = off.",
    )
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
    p.add_argument(
        "--lite",
        type=int,
        default=40,
        choices=[40, 20],
        help="Lightweight preset: 40 = default (paper-ish), 20 = smaller backbone/decoder dims.",
    )
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
        help="Compare grid: max batch rows per image; 0 disables train+val TensorBoard grids",
    )
    p.add_argument(
        "--viz-every-steps",
        type=int,
        default=30,
        help="Save train viz grid every N optimizer steps (after grad accumulation); 0 disables",
    )
    return p.parse_args()


def _quad_iou_from_points_norm(
    pred_pts_flat: torch.Tensor,
    gt_pts_flat: torch.Tensor,
    *,
    h: int,
    w: int,
    n_corners: int = 4,
) -> float:
    """Compute polygon IoU (JI) from normalized corner points in [0,1].

    Points are expected to be ordered (TL/TR/BR/BL). Uses rasterization for robustness.
    """
    import cv2

    def _to_poly(pts_flat: torch.Tensor) -> np.ndarray | None:
        pts = pts_flat.detach().float().view(-1, 2)[:n_corners].cpu().numpy()
        if pts.size == 0:
            return None
        if np.any(pts < 0.0):
            raise ValueError("Invalid corner points: negative values found (expected normalized 0..1).")
        xs = np.clip(pts[:, 0] * float(max(w - 1, 1)), 0.0, float(max(w - 1, 1)))
        ys = np.clip(pts[:, 1] * float(max(h - 1, 1)), 0.0, float(max(h - 1, 1)))
        poly = np.stack([xs, ys], axis=1)
        poly = np.round(poly).astype(np.int32)
        return poly.reshape(-1, 1, 2)

    p_poly = _to_poly(pred_pts_flat)
    g_poly = _to_poly(gt_pts_flat)
    if p_poly is None or g_poly is None:
        raise ValueError("Empty corner points (expected at least 4 points).")

    pm = np.zeros((h, w), dtype=np.uint8)
    gm = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(pm, [p_poly], 1)
    cv2.fillPoly(gm, [g_poly], 1)
    inter = int((pm & gm).sum())
    union = int((pm | gm).sum())
    if union <= 0:
        return 1.0
    return float(inter) / float(union)


def main() -> None:
    args = parse_args()
    accum = int(args.grad_accum_steps)
    if accum < 1:
        raise SystemExit("--grad-accum-steps must be >= 1")
    device = pick_device()
    print(f"device => {device}")
    if accum > 1:
        print(f"gradient accumulation: {accum} micro-batches per optimizer step (effective batch ≈ {args.batch_size * accum})")

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
    if int(args.lite) == 20:
        # Keep heads unchanged (8) so dims remain divisible.
        cfg.backbone_dim = 256
        cfg.hidden_dim = 192
        cfg.ffn_dim = 1024
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

    if args.val_rwmd_root:
        train_ds, val_ds = ds, RWMDLabelMeDataset(
            args.val_rwmd_root,
            img_size=args.img_size,
            num_classes=cfg.num_classes,
            num_points=cfg.num_points,
            max_instances=cfg.num_queries,
        )
        print(f"val dataset => RWMDLabelMeDataset({args.val_rwmd_root})")
    else:
        train_ds, val_ds = ds, None

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_doc_batch,
        pin_memory=device.type == "cuda",
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_doc_batch,
            pin_memory=device.type == "cuda",
        )

    model = RDLNet(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mb_fp32 = n_params * 4 / (1024 * 1024)
    mb_fp16 = n_params * 2 / (1024 * 1024)
    print(
        "model params => "
        f"total={n_params:,} ({n_params/1e6:.2f}M), "
        f"trainable={n_trainable:,} ({n_trainable/1e6:.2f}M), "
        f"weights≈{mb_fp32:.1f}MB(fp32) / {mb_fp16:.1f}MB(fp16)"
    )
    matcher = build_matcher(cfg)
    criterion = RDLNetLoss(cfg, matcher)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Paper (Sec. 3.5): AdamW with LR drop by 0.1 every 40000 iterations, 160000 total iters.
    # In this codebase, an "iteration" corresponds to a real optimizer update (i.e., after grad accumulation).
    lr_milestones = [40_000, 80_000, 120_000]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=lr_milestones, gamma=0.1)
    optimizer_step = 0
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
    best_val_ji = float("-inf")
    if args.resume:
        run_dir = Path(args.resume).expanduser().resolve()
        if run_dir.is_file():
            raise SystemExit(
                "--resume must be a run directory (e.g. output/rdlnet/20250424_153022), not a .pt file."
            )
        if not run_dir.is_dir():
            raise SystemExit(f"--resume run directory does not exist: {run_dir}")
        resume_ckpt = run_dir / CKPT_MAIN
        if not resume_ckpt.is_file():
            raise SystemExit(
                f"No {CKPT_MAIN} in {run_dir}. Pass the timestamp run folder under --output."
            )
        ck = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        optimizer_step = int(ck.get("optimizer_step", 0))
        if isinstance(ck.get("scheduler"), dict):
            scheduler.load_state_dict(ck["scheduler"])
        else:
            # Backward-compatible resume for older checkpoints (no scheduler saved).
            # Align internal scheduler counter to the restored optimizer_step.
            scheduler.last_epoch = optimizer_step - 1
        start_epoch = int(ck.get("epoch", 0))
        hist_ep, hist_loss, hist_cls, hist_dist, hist_dice, hist_mask = load_loss_history_from_ck(ck)
        if isinstance(ck.get("best_val_ji"), (int, float)):
            best_val_ji = float(ck["best_val_ji"])
        del ck
        print(
            f"Resumed from {resume_ckpt} (run {run_dir}): epoch={start_epoch}, optimizer_step={optimizer_step}"
        )
    elif args.distill_checkpoint:
        load_student_encoder_into_rdlnet_from_checkpoint(model, args.distill_checkpoint)
        print(f"Loaded student backbone from {args.distill_checkpoint}")

    if not args.resume:
        out_root = Path(args.output).expanduser().resolve()
        if out_root.is_file():
            raise SystemExit(
                f"--output must be a directory, not a file ({out_root}). "
                "Example: --output output/rdlnet"
            )
        run_ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        run_dir = out_root / run_ts
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run directory => {run_dir}")

    out_path = run_dir / CKPT_MAIN
    best_ckpt_path = run_dir / CKPT_BEST
    print(f"output checkpoint => {out_path}")

    tb_logdir = run_dir / TB_SUBDIR
    tb_logdir.mkdir(parents=True, exist_ok=True)
    print(f"tensorboard logdir => {tb_logdir}")

    end_epoch = start_epoch + args.epochs

    with SummaryWriter(log_dir=str(tb_logdir)) as tb:
        for epoch in range(start_epoch, end_epoch):
            model.train()
            sums = LossSums()
            max_batches = int(args.max_batches_per_epoch)
            n_plan = len(loader) if max_batches <= 0 else min(len(loader), max_batches)
            micro = 0
            for bi, batch in enumerate(tqdm(loader, desc=f"epoch {epoch + 1}/{end_epoch}", leave=True)):
                images, tgt_labels, tgt_masks, tgt_points = _batch_to_device(batch, device)
                is_last_in_epoch = (bi + 1) == n_plan
                if micro == 0:
                    opt.zero_grad(set_to_none=True)
                out, loss, logs = _forward_and_loss(
                    model,
                    criterion,
                    images,
                    tgt_labels,
                    tgt_masks,
                    tgt_points,
                    device_type=device.type,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                )
                if not torch.isfinite(loss).item():
                    tqdm.write(
                        f"WARN: non-finite loss at step={optimizer_step} (loss={float(loss.detach().cpu())}); skipping update"
                    )
                    opt.zero_grad(set_to_none=True)
                    micro = 0
                    if use_amp:
                        scaler.update()
                    continue
                loss_scaled = loss / accum
                if use_amp:
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()
                micro += 1
                do_step = (micro == accum) or (is_last_in_epoch and 0 < micro < accum)
                if do_step:
                    if is_last_in_epoch and 0 < micro < accum:
                        for param in model.parameters():
                            g = param.grad
                            if g is not None:
                                g.mul_(accum / micro)
                    if args.grad_clip_norm > 0:
                        if use_amp:
                            scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                    if use_amp:
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    optimizer_step += 1
                    scheduler.step()
                    tb.add_scalar("train/lr", float(opt.param_groups[0]["lr"]), global_step=optimizer_step)

                    if (
                        args.viz_samples > 0
                        and args.viz_every_steps > 0
                        and optimizer_step % args.viz_every_steps == 0
                    ):
                        _log_compare_grid_to_tb(
                            tb,
                            images,
                            out,
                            tgt_labels,
                            tgt_masks,
                            tgt_points,
                            tag=f"train/compare_grid/epoch_{epoch + 1:04d}/step_{optimizer_step:08d}",
                            global_step=optimizer_step,
                            max_samples=args.viz_samples,
                        )

                    micro = 0
                sums.update(loss, logs)
                if _should_stop(bi, max_batches):
                    break

            av = sums.averages()
            ep_no = epoch + 1
            hist_ep.append(ep_no)
            hist_loss.append(av["total"])
            hist_cls.append(av["cls"])
            hist_dist.append(av["dist"])
            hist_dice.append(av["dice"])
            hist_mask.append(av["mask"])
            tb.add_scalars(
                "train_epoch/losses",
                av,
                global_step=ep_no,
            )

            val_loss: float | None = None
            val_ji: float | None = None
            if val_loader is not None:
                model.eval()
                v_sums = LossSums()
                ji_sum = 0.0
                ji_n = 0
                val_viz_logged = False
                with torch.no_grad():
                    for vbi, batch in enumerate(tqdm(val_loader, desc=f"val {epoch + 1}/{end_epoch}", leave=False)):
                        paths = batch.get("paths") or []
                        images, tgt_labels, tgt_masks, tgt_points = _batch_to_device(batch, device)
                        out, loss, logs = _forward_and_loss(
                            model,
                            criterion,
                            images,
                            tgt_labels,
                            tgt_masks,
                            tgt_points,
                            device_type=device.type,
                            use_amp=use_amp,
                            amp_dtype=amp_dtype,
                        )
                        if not torch.isfinite(loss).item():
                            continue
                        v_sums.update(loss, logs)
                        if args.viz_samples > 0 and not val_viz_logged:
                            val_viz_logged = True
                            _log_compare_grid_to_tb(
                                tb,
                                images,
                                out,
                                tgt_labels,
                                tgt_masks,
                                tgt_points,
                                tag=f"val/compare_grid/epoch_{ep_no:04d}",
                                global_step=ep_no,
                                max_samples=args.viz_samples,
                            )
                        # JI (IoU) from main document quad corners (class_id=0).
                        probs0 = torch.softmax(out["pred_logits"].detach(), dim=-1)[..., 0]  # [B, Nq]
                        q_best = torch.argmax(probs0, dim=1)  # [B]
                        bsz = images.shape[0]
                        h, w = int(images.shape[2]), int(images.shape[3])
                        if not paths:
                            paths = ["?"] * int(bsz)
                        for i in range(bsz):
                            tl = tgt_labels[i].long()
                            fg = (tl == 0).nonzero(as_tuple=False).view(-1)
                            if fg.numel() != 1:
                                raise ValueError(
                                    f"Expected exactly 1 main document (label==0) per image, got {int(fg.numel())}. "
                                    f"path={paths[i]}"
                                )
                            gi = int(fg[0].item())
                            pi = int(q_best[i].item())
                            gt_pts_flat = tgt_points[i][gi]
                            # If label==0 exists, its first 4 corners must be valid normalized coords.
                            # Otherwise RWMD `label_points_resize.json` likely missed this image.
                            if torch.any(gt_pts_flat[: 4 * 2] < 0).item():
                                raise ValueError(
                                    "Invalid GT corners for label==0 (expected 4 valid normalized points in [0,1]). "
                                    f"path={paths[i]} gt_first8={gt_pts_flat[:8].detach().cpu().tolist()}"
                                )
                            ji = _quad_iou_from_points_norm(
                                out["pred_points"][i, pi],
                                gt_pts_flat,
                                h=h,
                                w=w,
                                n_corners=4,
                            )
                            ji_sum += float(ji)
                            ji_n += 1
                        if _should_stop(vbi, max_batches):
                            break
                v_av = v_sums.averages()
                val_loss = float(v_av["total"])
                if ji_n > 0:
                    val_ji = float(ji_sum / ji_n)
                tb.add_scalars(
                    "val_epoch/losses",
                    v_av,
                    global_step=ep_no,
                )
                if val_ji is not None:
                    tb.add_scalar("val_epoch/ji", val_ji, global_step=ep_no)
                model.train()

            ckpt: dict[str, Any] = {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": asdict(cfg),
                "epoch": ep_no,
                "optimizer_step": optimizer_step,
                "val_loss": val_loss,
                "val_ji": val_ji,
                "best_val_ji": best_val_ji,
                LOSS_HISTORY_KEY: pack_loss_history(
                    hist_ep, hist_loss, hist_cls, hist_dist, hist_dice, hist_mask
                ),
            }
            torch.save(ckpt, out_path)

            if val_ji is not None and val_ji > best_val_ji:
                best_val_ji = val_ji
                best_ckpt = dict(ckpt)
                best_ckpt["best_val_ji"] = best_val_ji
                best_ckpt["is_best"] = True
                torch.save(best_ckpt, best_ckpt_path)
                print(f"Saved best checkpoint -> {best_ckpt_path} (val_ji={best_val_ji:.6g})")


if __name__ == "__main__":
    main()

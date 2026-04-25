"""Training-time grids: input vs GT vs prediction (masks + corner points).

Preprocessed RWMD GT check: ``python -m rdlnet.viz_rdlnet --rwmd-root path/to/train_resize --output out.png``
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor


def _chw_to_uint8_hwc(t: Tensor) -> np.ndarray:
    x = t.detach().float().cpu().numpy()
    if x.ndim != 3:
        raise ValueError("expected CHW image tensor")
    x = np.transpose(x, (1, 2, 0))
    if x.max() <= 1.0 + 1e-3:
        x = x * 255.0
    return np.clip(x, 0, 255).astype(np.uint8)


def _resize_mask_to_hw(m: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize a single-channel mask to (h, w); values in ~[0, 1] float."""
    m = np.asarray(m, dtype=np.float32)
    if m.ndim != 2:
        raise ValueError("mask must be 2D")
    if m.shape[0] == h and m.shape[1] == w:
        return m
    from PIL import Image

    u8 = np.clip(m * 255.0, 0, 255).astype(np.uint8)
    im = Image.fromarray(u8, mode="L")
    im = im.resize((w, h), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def _palette(n: int) -> np.ndarray:
    base = np.array(
        [
            [255, 64, 64],
            [64, 255, 64],
            [64, 128, 255],
            [255, 200, 64],
            [200, 64, 255],
            [64, 255, 255],
            [255, 128, 192],
            [180, 255, 128],
        ],
        dtype=np.float32,
    )
    return np.stack([base[i % len(base)] for i in range(n)], axis=0)


def _blend_instances(
    rgb: np.ndarray,
    masks_hw: List[np.ndarray],
    alpha: float = 0.42,
    mask_thresh: float = 0.5,
) -> np.ndarray:
    """Overlay binary-ish masks (H,W), same size as ``rgb``, in different colors."""
    if not masks_hw:
        return rgb
    colors = _palette(len(masks_hw))
    out = rgb.astype(np.float32)
    for i, m in enumerate(masks_hw):
        mu = (m > mask_thresh).astype(np.float32)
        col = colors[i]
        for c in range(3):
            out[..., c] = out[..., c] * (1.0 - alpha * mu) + col[c] * alpha * mu
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_quad_corners(ax, pts_flat: np.ndarray, h: int, w: int, color, n_corners: int = 4) -> None:
    n = pts_flat.size // 2
    pts = pts_flat.reshape(n, 2)[:n_corners]
    # Skip padded / invalid points (negative coords are used as padding).
    pts = pts[(pts[:, 0] >= 0.0) & (pts[:, 1] >= 0.0)]
    if pts.size == 0:
        return
    xs = pts[:, 0] * float(max(w - 1, 1))
    ys = pts[:, 1] * float(max(h - 1, 1))
    ax.scatter(xs, ys, c=color, s=14, linewidths=0.4, edgecolors="white", zorder=5)
    if len(pts) >= 2:
        loop = np.vstack([pts, pts[:1]])
        ax.plot(
            loop[:, 0] * float(max(w - 1, 1)),
            loop[:, 1] * float(max(h - 1, 1)),
            color=color,
            linewidth=1.1,
            alpha=0.95,
            zorder=4,
        )


def save_annotations_viz_grid(
    path: Path | str,
    images: Tensor,
    tgt_masks: List[Tensor],
    tgt_points: List[Tensor],
    *,
    max_samples: int = 8,
    mask_thresh: float = 0.5,
    n_corner_vis: int = 4,
    suptitle: str = "Annotations (GT only)",
) -> None:
    """
    Save a 2-column grid: ``[RGB | GT masks + corner points]`` — no predictions.

    Intended to sanity-check dataloader outputs (e.g. ``collate_doc_batch``) without running the model.
    ``images``: ``B×C×H×W`` float tensor on CPU or GPU; ``tgt_masks`` / ``tgt_points`` match training.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    bsz = images.shape[0]
    b = min(bsz, max_samples)
    if b <= 0:
        return

    ncols = 2
    fig, axes = plt.subplots(b, ncols, figsize=(4.2 * ncols, 3.5 * b), squeeze=False)

    for i in range(b):
        rgb = _chw_to_uint8_hwc(images[i])
        h, w = int(rgb.shape[0]), int(rgb.shape[1])

        tm = tgt_masks[i].float().cpu().numpy()
        gt_layers = (
            [_resize_mask_to_hw(tm[j], h, w) for j in range(tm.shape[0])] if tm.size > 0 else []
        )
        gt_vis = _blend_instances(rgb.copy(), gt_layers, mask_thresh=mask_thresh)

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title("image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt_vis)
        axes[i, 1].set_title(f"GT ({len(gt_layers)} inst.)")
        axes[i, 1].axis("off")
        tpts = tgt_points[i].float().cpu().numpy()
        for j in range(tpts.shape[0]):
            _draw_quad_corners(axes[i, 1], tpts[j], h, w, f"C{j % 10}", n_corner_vis)

    fig.suptitle(suptitle, fontsize=11, y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=125, bbox_inches="tight")
    plt.close(fig)


def _fig_to_rgb_u8(fig) -> np.ndarray:
    """Render a matplotlib figure to RGB uint8 (H, W, 3)."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    # Matplotlib 3.8+ prefers buffer_rgba(); older versions may not have it.
    if hasattr(fig.canvas, "buffer_rgba"):
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        return rgba[..., :3]
    # Fallback: tostring_argb -> convert to rgb
    if hasattr(fig.canvas, "tostring_argb"):
        argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
        return argb[..., 1:4]
    raise AttributeError("matplotlib canvas has no RGB buffer method (expected buffer_rgba or tostring_argb)")


def annotations_viz_grid_u8(
    images: Tensor,
    tgt_masks: List[Tensor],
    tgt_points: List[Tensor],
    *,
    max_samples: int = 8,
    mask_thresh: float = 0.5,
    n_corner_vis: int = 4,
    suptitle: str = "Annotations (GT only)",
) -> np.ndarray:
    """Return the same GT grid as :func:`save_annotations_viz_grid`, as RGB uint8."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bsz = images.shape[0]
    b = min(bsz, max_samples)
    if b <= 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    ncols = 2
    fig, axes = plt.subplots(b, ncols, figsize=(4.2 * ncols, 3.5 * b), squeeze=False)

    for i in range(b):
        rgb = _chw_to_uint8_hwc(images[i])
        h, w = int(rgb.shape[0]), int(rgb.shape[1])

        tm = tgt_masks[i].float().cpu().numpy()
        gt_layers = (
            [_resize_mask_to_hw(tm[j], h, w) for j in range(tm.shape[0])] if tm.size > 0 else []
        )
        gt_vis = _blend_instances(rgb.copy(), gt_layers, mask_thresh=mask_thresh)

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title("image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt_vis)
        axes[i, 1].set_title(f"GT ({len(gt_layers)} inst.)")
        axes[i, 1].axis("off")
        tpts = tgt_points[i].float().cpu().numpy()
        for j in range(tpts.shape[0]):
            _draw_quad_corners(axes[i, 1], tpts[j], h, w, f"C{j % 10}", n_corner_vis)

    fig.suptitle(suptitle, fontsize=11, y=1.02)
    fig.tight_layout()
    out = _fig_to_rgb_u8(fig)
    plt.close(fig)
    return out


def train_compare_grid_u8(
    images: Tensor,
    out: Dict[str, Tensor],
    tgt_labels: List[Tensor],
    tgt_masks: List[Tensor],
    tgt_points: List[Tensor],
    *,
    max_samples: int = 4,
    mask_thresh: float = 0.5,
    n_corner_vis: int = 4,
) -> np.ndarray:
    """Return a grid: columns = [RGB | GT masks+points | pred masks+points], as RGB uint8 (H, W, 3).

    Requires ``out[\"pred_logits\"]`` with shape ``[B, Nq, num_classes+1]`` (last class = no-object).

    - **Corners (pred):** only the query with largest class-0 probability (same as val JI); other
      queries' ``pred_points`` are not drawn.
    - **Pred masks:** every query whose argmax class is **not** no-object (so foreground + background
      instance masks); no-object queries are skipped. Main-doc query mask is drawn last so it stays on top.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bsz = images.shape[0]
    b = min(bsz, max_samples)
    if b <= 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    pred_masks = out["pred_masks"].float().cpu()
    pred_points = out["pred_points"].float().cpu()
    pred_prob = pred_masks.sigmoid()
    if "pred_logits" not in out:
        raise ValueError("train_compare_grid_u8 requires out['pred_logits']")
    pred_logits = out["pred_logits"].float().cpu()

    ncols = 3
    fig, axes = plt.subplots(b, ncols, figsize=(4.2 * ncols, 3.5 * b), squeeze=False)

    for i in range(b):
        rgb = _chw_to_uint8_hwc(images[i])
        h, w = int(rgb.shape[0]), int(rgb.shape[1])

        tm = tgt_masks[i].float().cpu().numpy()
        gt_layers = (
            [_resize_mask_to_hw(tm[j], h, w) for j in range(tm.shape[0])] if tm.size > 0 else []
        )
        gt_vis = _blend_instances(rgb.copy(), gt_layers, mask_thresh=mask_thresh)

        pm = pred_prob[i].numpy()
        nq = int(pm.shape[0])
        ppts = pred_points[i].numpy()

        lg = pred_logits[i]
        if lg.ndim != 2 or int(lg.shape[0]) != nq or int(lg.shape[1]) < 2:
            raise ValueError(f"bad pred_logits[{i}]: shape={tuple(lg.shape)} nq={nq}")
        probs = torch.softmax(lg, dim=-1)
        no_idx = int(lg.shape[1]) - 1  # DETR-style: last logit = no-object
        main_q = int(torch.argmax(probs[:, 0]).item())
        pred_cls = probs.argmax(dim=-1)
        others = [
            q for q in range(nq) if int(pred_cls[q].item()) != no_idx and q != main_q
        ]
        main_is_obj = int(pred_cls[main_q].item()) != no_idx
        mask_order = others + ([main_q] if main_is_obj else [])
        pred_layers = [_resize_mask_to_hw(pm[q], h, w) for q in mask_order]
        pred_vis = _blend_instances(rgb.copy(), pred_layers, mask_thresh=mask_thresh)
        pred_title = f"pred ({len(pred_layers)} masks, corners q={main_q})"

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title("image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt_vis)
        axes[i, 1].set_title(f"GT ({len(gt_layers)} inst.)")
        axes[i, 1].axis("off")
        tl = tgt_labels[i].long().cpu()
        fg = (tl == 0).nonzero(as_tuple=False).view(-1)
        tpts = tgt_points[i].float().cpu().numpy()
        for j in fg.tolist():
            if 0 <= j < tpts.shape[0]:
                _draw_quad_corners(axes[i, 1], tpts[j], h, w, f"C{j % 10}", n_corner_vis)

        axes[i, 2].imshow(pred_vis)
        axes[i, 2].set_title(pred_title)
        axes[i, 2].axis("off")
        _draw_quad_corners(axes[i, 2], ppts[main_q], h, w, "C0", n_corner_vis)

    fig.suptitle("RDLNet: image | ground truth | prediction", fontsize=11, y=1.02)
    fig.tight_layout()
    out_u8 = _fig_to_rgb_u8(fig)
    plt.close(fig)
    return out_u8


def main() -> None:
    """Visualize preprocessed RWMD samples (GT only, no model)."""
    import argparse
    import sys

    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from rdlnet.data import RWMDLabelMeDataset, collate_doc_batch
    from rdlnet.model import RDLNetConfig

    p = argparse.ArgumentParser(
        description=(
            "Visualize preprocessed RWMD samples (GT only, no model). "
            "Requires img/, mask/, label_points_resize.json (see data_preprocessing_rwdm_1.py)."
        ),
    )
    p.add_argument(
        "--rwmd-root",
        type=Path,
        default=Path("output/data"),
        help=(
            "RWMD dataset directory OR a preprocessed split directory. "
            "If a dataset directory is passed, all '*_resize' splits under it are processed "
            "(e.g. train_resize/, test_resize/)."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("output/rwmd_gt_sample.png"),
        help="Output PNG path",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="If set, write one PNG per sample into this directory (overrides --output)",
    )
    p.add_argument(
        "--tb-logdir",
        type=Path,
        default=None,
        help="If set, log one GT grid image per sample to TensorBoard in this logdir (overrides --output/--output-dir)",
    )
    p.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index (for --output-dir/--tb-logdir)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of samples (0 = all; for --output-dir/--tb-logdir)",
    )
    p.add_argument("--img-size", type=int, default=1024)
    p.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Override class count (default: 2 for RWMD main/background)",
    )
    args = p.parse_args()

    rwmd_root = args.rwmd_root.resolve()
    if not rwmd_root.is_dir():
        raise SystemExit(f"Not a directory: {rwmd_root}")

    def _is_preprocessed_split_dir(p: Path) -> bool:
        return (p / "img").is_dir() and (p / "mask").is_dir() and (p / "label_points_resize.json").is_file()

    # Accept either a preprocessed split directory (train_resize/) or a dataset directory containing splits.
    if _is_preprocessed_split_dir(rwmd_root):
        split_dirs = [rwmd_root]
    else:
        split_dirs = sorted([d for d in rwmd_root.glob("*_resize") if d.is_dir() and _is_preprocessed_split_dir(d)])
        if not split_dirs:
            raise SystemExit(
                "No preprocessed '*_resize' splits found under --rwmd-root. "
                "Expected e.g. train_resize/ with img/, mask/, label_points_resize.json."
            )

    if args.num_classes is not None:
        num_classes = args.num_classes
    else:
        num_classes = 2

    cfg = RDLNetConfig(
        img_size=args.img_size,
        num_classes=num_classes,
        use_sam_pixel_norm=True,
    )
    if args.img_size % cfg.patch_size != 0:
        raise SystemExit("img_size must be divisible by patch_size")

    start = max(int(args.start), 0)
    limit = int(args.limit) if args.limit else 0

    if args.tb_logdir is not None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as e:  # pragma: no cover
            raise SystemExit(f"TensorBoard not available (install tensorboard). Import error: {e}") from e

        logdir = args.tb_logdir
        if not logdir.is_absolute():
            logdir = (repo / logdir).resolve()
        logdir.mkdir(parents=True, exist_ok=True)

        with SummaryWriter(log_dir=str(logdir)) as w:
            global_step = 0
            for split_root in split_dirs:
                ds = RWMDLabelMeDataset(
                    split_root,
                    img_size=cfg.img_size,
                    num_classes=cfg.num_classes,
                    num_points=cfg.num_points,
                    max_instances=cfg.num_queries,
                )
                end = len(ds)
                if limit > 0:
                    end = min(end, start + limit)
                if start >= end:
                    continue
                for i in range(start, end):
                    sample = ds[i]
                    batch = collate_doc_batch([sample])
                    stem = Path(sample["path"]).stem
                    grid = annotations_viz_grid_u8(
                        batch["images"],
                        batch["tgt_masks"],
                        batch["tgt_points"],
                        max_samples=1,
                        suptitle=f"RWMD GT — {split_root.name} [{i}]",
                    )
                    chw = np.transpose(grid, (2, 0, 1))
                    w.add_image(f"rwmd_gt/{split_root.name}/{stem}", chw, global_step=global_step, dataformats="CHW")
                    global_step += 1
        print(logdir)
        return

    if args.output_dir is not None:
        out_dir = args.output_dir
        if not out_dir.is_absolute():
            out_dir = (repo / out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        for split_root in split_dirs:
            split_out = out_dir / split_root.name
            split_out.mkdir(parents=True, exist_ok=True)
            ds = RWMDLabelMeDataset(
                split_root,
                img_size=cfg.img_size,
                num_classes=cfg.num_classes,
                num_points=cfg.num_points,
                max_instances=cfg.num_queries,
            )
            end = len(ds)
            if limit > 0:
                end = min(end, start + limit)
            if start >= end:
                continue
            for i in range(start, end):
                sample = ds[i]
                batch = collate_doc_batch([sample])
                stem = Path(sample["path"]).stem
                out_path = split_out / f"{stem}_gt.png"
                save_annotations_viz_grid(
                    out_path,
                    batch["images"],
                    batch["tgt_masks"],
                    batch["tgt_points"],
                    max_samples=1,
                    suptitle=f"RWMD GT — {split_root.name} [{i}]",
                )
        print(out_dir)
        return

    out_path = args.output
    if not out_path.is_absolute():
        out_path = (repo / out_path).resolve()
    # Default: write the first sample of the first split (respecting --start).
    first_split = split_dirs[0]
    ds = RWMDLabelMeDataset(
        first_split,
        img_size=cfg.img_size,
        num_classes=cfg.num_classes,
        num_points=cfg.num_points,
        max_instances=cfg.num_queries,
    )
    if start < 0 or start >= len(ds):
        raise SystemExit(f"--start {start} out of range [0, {len(ds)}) for split {first_split}")
    batch = collate_doc_batch([ds[start]])
    save_annotations_viz_grid(
        out_path,
        batch["images"],
        batch["tgt_masks"],
        batch["tgt_points"],
        max_samples=1,
        suptitle=f"RWMD GT — {first_split.name} [{start}]",
    )
    print(out_path)


if __name__ == "__main__":
    main()

"""Training-time grids: input vs GT vs prediction (masks + corner points).

Preprocessed RWMD GT check: ``python -m rdlnet.viz_rdlnet --rwmd-root path/to/train_resize --output out.png``
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

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


def save_train_compare_grid(
    path: Path | str,
    images: Tensor,
    out: Dict[str, Tensor],
    tgt_masks: List[Tensor],
    tgt_points: List[Tensor],
    *,
    max_samples: int = 4,
    mask_thresh: float = 0.5,
    n_corner_vis: int = 4,
) -> None:
    """
    Save a grid: columns = [RGB | GT masks+points | pred masks+points], rows = batch samples.

    ``out`` must contain ``pred_masks`` (logits B×Nq×H×W) and ``pred_points`` (sigmoid B×Nq×(P*2)).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    bsz = images.shape[0]
    b = min(bsz, max_samples)
    if b <= 0:
        return

    pred_masks = out["pred_masks"].float().cpu()
    pred_points = out["pred_points"].float().cpu()
    pred_prob = pred_masks.sigmoid()

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
        pred_layers = [_resize_mask_to_hw(pm[q], h, w) for q in range(nq)]
        pred_vis = _blend_instances(rgb.copy(), pred_layers, mask_thresh=mask_thresh)

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title("image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt_vis)
        axes[i, 1].set_title(f"GT ({len(gt_layers)} inst.)")
        axes[i, 1].axis("off")
        tpts = tgt_points[i].float().cpu().numpy()
        for j in range(tpts.shape[0]):
            _draw_quad_corners(axes[i, 1], tpts[j], h, w, f"C{j % 10}", n_corner_vis)

        axes[i, 2].imshow(pred_vis)
        axes[i, 2].set_title(f"pred ({nq} q.)")
        axes[i, 2].axis("off")
        ppts = pred_points[i].numpy()
        for q in range(nq):
            _draw_quad_corners(axes[i, 2], ppts[q], h, w, f"C{q % 10}", n_corner_vis)

    fig.suptitle("RDLNet: image | ground truth | prediction", fontsize=11, y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=125, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Load one sample from a preprocessed RWMD folder and write a GT-only PNG via :func:`save_annotations_viz_grid`."""
    import argparse
    import sys

    from rdlnet.data import RWMDLabelMeDataset, collate_doc_batch
    from rdlnet.model import RDLNetConfig

    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    p = argparse.ArgumentParser(
        description=(
            "Visualize one preprocessed RWMD sample (GT only, no model). "
            "Requires img/, mask/, label_points_resize.json (see data_preprocessing_rwdm_1.py)."
        ),
    )
    p.add_argument(
        "--rwmd-root",
        type=Path,
        default=Path("output/data/train_resize"),
        help="Preprocessed RWMD root (e.g. path/to/train_resize)",
    )
    p.add_argument(
        "--index",
        type=int,
        default=5,
        help="Index into sorted img/*.png (default 0)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("output/rwmd_gt_sample.png"),
        help="Output PNG path",
    )
    p.add_argument("--img-size", type=int, default=1024)
    p.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Override class count (default: 2 for RWMD main/background)",
    )
    p.add_argument(
        "--ignore-padded-points",
        action="store_true",
    )
    p.add_argument("--padded-point-eps", type=float, default=0.0)
    args = p.parse_args()

    rwmd_root = args.rwmd_root
    if rwmd_root is None:
        raise SystemExit("Pass --rwmd-root to a preprocessed folder (train_resize with img/, mask/, label_points_resize.json)")
    rwmd_root = rwmd_root.resolve()
    if not rwmd_root.is_dir():
        raise SystemExit(f"Not a directory: {rwmd_root}")

    if args.num_classes is not None:
        num_classes = args.num_classes
    else:
        num_classes = 2

    cfg = RDLNetConfig(
        img_size=args.img_size,
        num_classes=num_classes,
        use_sam_pixel_norm=True,
        ignore_padded_points=args.ignore_padded_points,
        padded_point_eps=args.padded_point_eps,
    )
    if args.img_size % cfg.patch_size != 0:
        raise SystemExit("img_size must be divisible by patch_size")

    ds = RWMDLabelMeDataset(
        rwmd_root,
        img_size=cfg.img_size,
        num_classes=cfg.num_classes,
        num_points=cfg.num_points,
        max_instances=cfg.num_queries,
    )
    if args.index < 0 or args.index >= len(ds):
        raise SystemExit(f"--index {args.index} out of range [0, {len(ds)})")
    batch = collate_doc_batch([ds[args.index]])

    out_path = args.output
    if not out_path.is_absolute():
        out_path = (repo / out_path).resolve()
    save_annotations_viz_grid(
        out_path,
        batch["images"],
        batch["tgt_masks"],
        batch["tgt_points"],
        max_samples=1,
        suptitle=f"RWMD GT — {rwmd_root.name} [{args.index}]",
    )
    print(out_path)


if __name__ == "__main__":
    main()

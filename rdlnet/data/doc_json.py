"""
Stage-2 RDLNet training: RWMD preprocessed dataset loader.

RWMD offline data: :class:`RWMDLabelMeDataset` expects ``img/*.png``, ``mask/*.png`` (instance ids),
and ``label_points_resize.json`` from ``data_preprocessing_rwdm_1.run_rwmd_preprocess``.

Semantic **class** (use ``num_classes=2``): largest instance id in the mask → 0 (top sheet), others → 1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset


def collate_doc_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    return {
        "images": images,
        "tgt_labels": [b["labels"] for b in batch],
        "tgt_masks": [b["masks"] for b in batch],
        "tgt_points": [b["points"] for b in batch],
        "paths": [b["path"] for b in batch],
        "valid_masks": torch.stack([b["valid_mask"] for b in batch], dim=0),
        "geoms": [b.get("geom") for b in batch],
    }


def _rwmd_strip_closing_vertex(points_xy: Sequence[Sequence[float]]) -> np.ndarray:
    """
    LabelMe often stores a closed polygon with the first vertex repeated at the end.
    That yields len==5 for a quad, which would otherwise miss the n==4 branch and fall
    back to an axis-aligned box (wrong corners on rotated docs).
    """
    arr = np.asarray(points_xy, dtype=np.float64)
    if arr.shape[0] >= 2 and np.allclose(arr[0], arr[-1], rtol=0.0, atol=1e-3):
        arr = arr[:-1]
    return arr


def _rwmd_quad_corners_xy(points_xy: Sequence[Sequence[float]]) -> np.ndarray:
    """
    Four document corners in pixel space (4, 2) float32.
    If the polygon already has 4 vertices, use them as-is; otherwise use an axis-aligned
    bounding box (min/max), which is robust without extra dependencies.
    """
    arr = _rwmd_strip_closing_vertex(points_xy)
    if arr.shape[0] == 4:
        return arr.astype(np.float32)
    if arr.shape[0] < 3:
        return np.zeros((4, 2), dtype=np.float32)
    x0, y0 = float(arr[:, 0].min()), float(arr[:, 1].min())
    x1, y1 = float(arr[:, 0].max()), float(arr[:, 1].max())
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def _rwmd_flatten_points_norm(quad_xy: np.ndarray, w0: int, h0: int, num_points: int) -> Tensor:
    """First 4 rows are normalized quad corners; remaining points are padded with -1."""
    wh = np.array([float(w0), float(h0)], dtype=np.float32)
    q = (quad_xy / wh).astype(np.float32)
    pts = np.full((num_points, 2), -1.0, dtype=np.float32)
    n = min(4, num_points)
    pts[:n] = q[:n]
    return torch.from_numpy(pts.reshape(-1))


def _rwmd_quad_from_instance_mask(binary_hw: np.ndarray) -> np.ndarray:
    """Axis-aligned quad (4,2) float32 from a boolean/float mask [H,W]."""
    ys, xs = np.where(binary_hw > 0.5)
    if xs.size == 0:
        return np.zeros((4, 2), dtype=np.float32)
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def _rwmd_main_bg_class(inst_id: int, fore_idx: int, num_classes: int) -> int:
    """Top sheet (``inst_id == fore_idx``) → 0; other instances → 1."""
    if num_classes < 2:
        return 0
    return 0 if inst_id == fore_idx else 1


class RWMDLabelMeDataset(Dataset):
    """
    Preprocessed RWMD layout from ``run_rwmd_preprocess`` (see ``data_preprocessing_rwdm_1.py``):

    - ``img/*.png`` — RGB images (BGR on disk from OpenCV; we convert when loading).
    - ``mask/*.png`` — uint8 instance ids (largest id = top sheet).
    - ``label_points_resize.json`` — ``foreground_doc`` polygon per basename for quad targets.

    LabelMe digits: **max digit** = foreground; each smaller digit is one **background** instance, with
    all polygons of the same digit unioned (see ``genarate_label_from_ori``). **Classes:** top sheet
    (largest instance id in mask) → 0; others → 1 (use ``num_classes=2``).
    """

    def __init__(
        self,
        rwmd_root: Union[str, Path],
        img_size: int,
        num_classes: int,
        num_points: int,
        max_instances: int,
    ) -> None:
        super().__init__()
        self.rwmd_root = Path(rwmd_root).resolve()
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_points = num_points
        if max_instances <= 0:
            raise ValueError(f"max_instances must be > 0, got {max_instances}")
        self.max_instances = max_instances
        self.point_dim = num_points * 2

        img_dir = self.rwmd_root / "img"
        lp = self.rwmd_root / "label_points_resize.json"
        geom_p = self.rwmd_root / "geom_resize.json"
        if not img_dir.is_dir():
            raise ValueError(f"RWMDLabelMeDataset expects directory {img_dir}/")
        if not lp.is_file():
            raise ValueError(f"RWMDLabelMeDataset expects {lp}")
        self._img_paths = sorted(img_dir.glob("*.png"))
        if not self._img_paths:
            raise ValueError(f"No PNG files in {img_dir}")
        with open(lp, "r", encoding="utf-8") as f:
            self._label_points_resize: Dict[str, Any] = json.load(f)
        self._geom_by_name: Dict[str, Any] = {}
        if geom_p.is_file():
            with open(geom_p, "r", encoding="utf-8") as f:
                g = json.load(f)
            if isinstance(g, dict):
                self._geom_by_name = g

    def __len__(self) -> int:
        return len(self._img_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path_img = self._img_paths[idx]
        path_mask = self.rwmd_root / "mask" / path_img.name
        if not path_mask.is_file():
            raise FileNotFoundError(path_mask)

        img_bgr = cv2.imread(str(path_img), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(path_img)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h0, w0 = int(img_rgb.shape[0]), int(img_rgb.shape[1])
        if (h0, w0) != (self.img_size, self.img_size):
            raise ValueError(
                f"RWMD preprocessed images must be {self.img_size}x{self.img_size} (got {w0}x{h0}). "
                "Re-run dataset/RWMD_dataset/data_preprocessing_rwdm_1.py with the new letterbox pipeline."
            )

        m_u8 = cv2.imread(str(path_mask), cv2.IMREAD_GRAYSCALE)
        if m_u8 is None:
            raise FileNotFoundError(path_mask)
        if (int(m_u8.shape[0]), int(m_u8.shape[1])) != (self.img_size, self.img_size):
            raise ValueError(
                f"RWMD preprocessed masks must be {self.img_size}x{self.img_size} (got {m_u8.shape[1]}x{m_u8.shape[0]})."
            )

        ids = sorted(int(x) for x in np.unique(m_u8) if int(x) > 0)
        if not ids:
            labels_t = torch.zeros(0, dtype=torch.long)
            masks_t = torch.zeros(0, self.img_size, self.img_size)
            points_t = torch.zeros(0, self.point_dim)
            image = torch.from_numpy(np.asarray(img_rgb).copy()).float().permute(2, 0, 1)
            valid_mask = torch.zeros(self.img_size, self.img_size, dtype=torch.bool)
            return {
                "image": image,
                "labels": labels_t,
                "masks": masks_t,
                "points": points_t,
                "valid_mask": valid_mask,
                "geom": None,
                "path": str(path_img),
            }

        fore_idx = max(ids)

        fg_raw = self._label_points_resize.get(path_img.name) or self._label_points_resize.get(path_img.stem, [])
        geom = self._geom_by_name.get(path_img.name) or self._geom_by_name.get(path_img.stem) or None
        # valid region from geom (letterbox); True = valid, False = padding.
        valid_mask = torch.zeros(self.img_size, self.img_size, dtype=torch.bool)
        if isinstance(geom, dict):
            pad_x = int(geom.get("pad_x", 0))
            pad_y = int(geom.get("pad_y", 0))
            new_w = int(geom.get("new_w", 0))
            new_h = int(geom.get("new_h", 0))
            if new_w > 0 and new_h > 0:
                x0 = max(0, pad_x)
                y0 = max(0, pad_y)
                x1 = min(self.img_size, pad_x + new_w)
                y1 = min(self.img_size, pad_y + new_h)
                if x1 > x0 and y1 > y0:
                    valid_mask[y0:y1, x0:x1] = True

        labels: List[int] = []
        masks: List[Tensor] = []
        points: List[Tensor] = []

        ids_kept = ids[-self.max_instances :]

        for inst_id in ids_kept:
            bin_np = (m_u8 == inst_id).astype(np.float32)
            m = torch.from_numpy(bin_np).unsqueeze(0)
            # Preprocessed masks are already img_size×img_size; keep this op for safety but it is a no-op.
            m = F.interpolate(m.unsqueeze(0), size=(self.img_size, self.img_size), mode="nearest").squeeze(0).squeeze(0)
            masks.append(m)

            if inst_id == fore_idx and isinstance(fg_raw, list) and len(fg_raw) >= 3:
                quad = _rwmd_quad_corners_xy(fg_raw)
                # `label_points_resize.json` stores pixel coords in the **padded** img_size×img_size frame.
                pt = _rwmd_flatten_points_norm(quad, self.img_size, self.img_size, self.num_points)
            else:
                # Background instances have masks/labels but no corner-point supervision.
                pt = torch.full((self.point_dim,), -1.0, dtype=torch.float32)
            points.append(pt)

            lab = _rwmd_main_bg_class(inst_id, fore_idx, self.num_classes)
            if not (0 <= lab < self.num_classes):
                raise ValueError(f"label {lab} out of range [0, {self.num_classes})")
            labels.append(lab)

        image = torch.from_numpy(np.asarray(img_rgb).copy()).float().permute(2, 0, 1)

        labels_t = torch.tensor(labels, dtype=torch.long)
        masks_t = torch.stack(masks, dim=0)
        points_t = torch.stack(points, dim=0)

        return {
            "image": image,
            "labels": labels_t,
            "masks": masks_t,
            "points": points_t,
            "valid_mask": valid_mask,
            "geom": geom,
            "path": str(path_img),
        }

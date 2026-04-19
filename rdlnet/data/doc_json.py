"""
Stage-2 RDLNet training: JSON manifest for document instances per image.

Each image can list up to ``num_queries`` instances (extra are truncated).

Example ``annotations.json`` (list of records)::

    [
      {
        "file_name": "img001.jpg",
        "instances": [
          {
            "label": 0,
            "mask": "masks/img001_0.png",
            "points": [0.1, 0.05, 0.9, 0.05, 0.9, 0.95, 0.1, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
          }
        ]
      }
    ]

- ``label``: int in ``[0, num_classes-1]`` (see RDLNetConfig.num_classes).
- ``mask``: path to a binary or grayscale PNG (same coordinate frame as image); resized with nearest neighbor.
- ``points``: length ``num_points * 2``, normalized coordinates ``x,y`` in ``[0, 1]`` in **original** image space;
  unused pairs can be zeros (e.g. 4 corners + padding for 9 points).

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
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .rwmd_exif_points import load_rgb_exif_aligned


def _load_pil_rgb_exif(path: Union[str, Path]) -> Image.Image:
    """
    Decode image and apply EXIF Orientation so width/height match viewers and LabelMe
    (JPEGs are often stored landscape with Orientation=rotate; raw ``Image.open`` is wrong).
    """
    _, im = load_rgb_exif_aligned(path)
    return im


def collate_doc_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    return {
        "images": images,
        "tgt_labels": [b["labels"] for b in batch],
        "tgt_masks": [b["masks"] for b in batch],
        "tgt_points": [b["points"] for b in batch],
        "paths": [b["path"] for b in batch],
    }


class DocLocalizationJsonDataset(Dataset):
    def __init__(
        self,
        annotation_file: Union[str, Path],
        image_root: Union[str, Path],
        img_size: int,
        num_classes: int,
        num_points: int,
        max_instances: int,
    ) -> None:
        super().__init__()
        self.image_root = Path(image_root)
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_points = num_points
        self.max_instances = max_instances
        self.point_dim = num_points * 2

        with open(annotation_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raise ValueError("annotations JSON must be a list of records")
        self.records = raw

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        rel = rec["file_name"]
        path = self.image_root / rel
        if not path.is_file():
            raise FileNotFoundError(path)

        im = _load_pil_rgb_exif(path)
        w0, h0 = im.size
        im = im.resize((self.img_size, self.img_size), Image.BICUBIC)
        image = torch.from_numpy(np.asarray(im).copy()).float().permute(2, 0, 1)

        instances: List[Dict[str, Any]] = rec.get("instances") or []
        labels: List[int] = []
        masks: List[Tensor] = []
        points: List[Tensor] = []

        for inst in instances[: self.max_instances]:
            lab = int(inst["label"])
            if not (0 <= lab < self.num_classes):
                raise ValueError(f"label {lab} out of range [0, {self.num_classes})")
            labels.append(lab)

            m_path = Path(inst["mask"])
            if not m_path.is_absolute():
                m_path = self.image_root / m_path
            m = Image.open(m_path).convert("L")
            if m.size != (w0, h0):
                m = m.resize((w0, h0), Image.NEAREST)
            m = torch.from_numpy(np.asarray(m)).float() / 255.0
            m = (m > 0.5).float()
            m = m.unsqueeze(0)
            m = F.interpolate(m.unsqueeze(0), size=(self.img_size, self.img_size), mode="nearest").squeeze(0).squeeze(0)
            masks.append(m)

            pts = inst["points"]
            if len(pts) != self.point_dim:
                raise ValueError(f"points length {len(pts)} != {self.point_dim}")
            pt = torch.tensor(pts, dtype=torch.float32).view(self.num_points, 2)
            points.append(pt.reshape(-1))

        if not labels:
            labels_t = torch.zeros(0, dtype=torch.long)
            masks_t = torch.zeros(0, self.img_size, self.img_size)
            points_t = torch.zeros(0, self.point_dim)
        else:
            labels_t = torch.tensor(labels, dtype=torch.long)
            masks_t = torch.stack(masks, dim=0)
            points_t = torch.stack(points, dim=0)

        return {
            "image": image,
            "labels": labels_t,
            "masks": masks_t,
            "points": points_t,
            "path": str(path),
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
    """First 4 rows are normalized quad corners; remaining points are zero-padded."""
    wh = np.array([float(w0), float(h0)], dtype=np.float32)
    q = (quad_xy / wh).astype(np.float32)
    pts = np.zeros((num_points, 2), dtype=np.float32)
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

    **Classes:** top sheet (max instance id) → 0; others → 1 (use ``num_classes=2``).
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
        self.max_instances = max_instances
        self.point_dim = num_points * 2

        img_dir = self.rwmd_root / "img"
        lp = self.rwmd_root / "label_points_resize.json"
        if not img_dir.is_dir():
            raise ValueError(f"RWMDLabelMeDataset expects directory {img_dir}/")
        if not lp.is_file():
            raise ValueError(f"RWMDLabelMeDataset expects {lp}")
        self._img_paths = sorted(img_dir.glob("*.png"))
        if not self._img_paths:
            raise ValueError(f"No PNG files in {img_dir}")
        with open(lp, "r", encoding="utf-8") as f:
            self._label_points_resize: Dict[str, Any] = json.load(f)

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
        h0, w0 = img_rgb.shape[0], img_rgb.shape[1]

        m_u8 = cv2.imread(str(path_mask), cv2.IMREAD_GRAYSCALE)
        if m_u8 is None:
            raise FileNotFoundError(path_mask)

        ids = sorted(int(x) for x in np.unique(m_u8) if int(x) > 0)
        if not ids:
            labels_t = torch.zeros(0, dtype=torch.long)
            masks_t = torch.zeros(0, self.img_size, self.img_size)
            points_t = torch.zeros(0, self.point_dim)
            im = Image.fromarray(img_rgb).resize((self.img_size, self.img_size), Image.BICUBIC)
            image = torch.from_numpy(np.asarray(im).copy()).float().permute(2, 0, 1)
            return {"image": image, "labels": labels_t, "masks": masks_t, "points": points_t, "path": str(path_img)}

        fore_idx = max(ids)

        fg_raw = self._label_points_resize.get(path_img.name) or self._label_points_resize.get(path_img.stem, [])

        labels: List[int] = []
        masks: List[Tensor] = []
        points: List[Tensor] = []

        for inst_id in ids[: self.max_instances]:
            bin_np = (m_u8 == inst_id).astype(np.float32)
            m = torch.from_numpy(bin_np).unsqueeze(0)
            m = F.interpolate(m.unsqueeze(0), size=(self.img_size, self.img_size), mode="nearest").squeeze(0).squeeze(0)
            masks.append(m)

            if inst_id == fore_idx and isinstance(fg_raw, list) and len(fg_raw) >= 3:
                quad = _rwmd_quad_corners_xy(fg_raw)
            else:
                quad = _rwmd_quad_from_instance_mask(bin_np)
            pt = _rwmd_flatten_points_norm(quad, w0, h0, self.num_points)
            points.append(pt)

            lab = _rwmd_main_bg_class(inst_id, fore_idx, self.num_classes)
            if not (0 <= lab < self.num_classes):
                raise ValueError(f"label {lab} out of range [0, {self.num_classes})")
            labels.append(lab)

        im = Image.fromarray(img_rgb).resize((self.img_size, self.img_size), Image.BICUBIC)
        image = torch.from_numpy(np.asarray(im).copy()).float().permute(2, 0, 1)

        labels_t = torch.tensor(labels, dtype=torch.long)
        masks_t = torch.stack(masks, dim=0)
        points_t = torch.stack(points, dim=0)

        return {
            "image": image,
            "labels": labels_t,
            "masks": masks_t,
            "points": points_t,
            "path": str(path_img),
        }

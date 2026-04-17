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

RWMD (LabelMe-style) folders such as ``RWMD_dataset_v1`` can be loaded directly with
:class:`RWMDLabelMeDataset` — polygons are rasterized on the fly and quadrilateral
corners are derived from the annotation (4-point polygon or axis-aligned bounding box).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import Tensor
from torch.utils.data import Dataset

from .rwmd_exif_points import align_labelme_points_xy, load_rgb_exif_aligned


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


def _rwmd_rasterize_polygon(w: int, h: int, points_xy: Sequence[Sequence[float]]) -> Tensor:
    """Binary mask [H,W] in {0,1} float32 from a closed polygon in pixel coordinates."""
    if len(points_xy) < 3:
        return torch.zeros(h, w, dtype=torch.float32)
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    flat = [(float(p[0]), float(p[1])) for p in points_xy]
    draw.polygon(flat, outline=255, fill=255)
    m = torch.from_numpy(np.asarray(mask, dtype=np.float32)) / 255.0
    return (m > 0.5).float()


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


def _rwmd_resolve_image_path(json_path: Path, labelme: Dict[str, Any]) -> Path:
    d = json_path.parent
    ip = labelme.get("imagePath")
    if isinstance(ip, str) and ip:
        p = d / ip
        if p.is_file():
            return p
    stem = json_path.stem
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        p = d / (stem + ext)
        if p.is_file():
            return p
    raise FileNotFoundError(f"No image file next to {json_path}")


def _rwmd_order_shapes(
    shapes: List[Dict[str, Any]],
    order: Literal["foreground_first", "numeric_then_foreground", "json_order"],
) -> List[Dict[str, Any]]:
    fg: List[Dict[str, Any]] = []
    num: List[Dict[str, Any]] = []
    for s in shapes:
        lab = s.get("label")
        if lab == "foreground_doc":
            fg.append(s)
        elif isinstance(lab, str) and lab.isdigit():
            num.append(s)
    num.sort(key=lambda s: int(s["label"]))

    if order == "foreground_first":
        ordered = fg + num
    elif order == "numeric_then_foreground":
        ordered = num + fg
    else:
        ordered = [s for s in shapes if s.get("label") == "foreground_doc" or (isinstance(s.get("label"), str) and s["label"].isdigit())]
    return ordered


def _rwmd_instance_class(
    shape: Dict[str, Any],
    *,
    label_mode: Literal["folder", "layer", "zero"],
    folder_class: int,
    num_classes: int,
) -> int:
    if label_mode == "zero":
        return 0
    lab = shape.get("label")
    if label_mode == "folder":
        return int(folder_class) % max(num_classes, 1)
    # layer
    if lab == "foreground_doc":
        return 0
    if isinstance(lab, str) and lab.isdigit():
        k = int(lab) - 1
        if num_classes <= 0:
            return 0
        return int(k % num_classes)
    return 0


class RWMDLabelMeDataset(Dataset):
    """
    Load RWMD-style LabelMe JSON (``shapes`` with ``foreground_doc`` and numeric labels).

    Each polygon becomes one instance: binary mask from the polygon fill, and
    ``num_points`` keypoints with the first four corners normalized to ``[0,1]`` (quad from
    the annotation if it has 4 vertices, else axis-aligned bounding box of the polygon).

    **Truncation:** ``max_instances`` (typically ``num_queries``) keeps the first shapes
    after ordering — use ``foreground_first`` to prioritize the foreground quadrilateral.
    """

    def __init__(
        self,
        rwmd_root: Union[str, Path],
        img_size: int,
        num_classes: int,
        num_points: int,
        max_instances: int,
        label_mode: Literal["folder", "layer", "zero"] = "folder",
        instance_order: Literal["foreground_first", "numeric_then_foreground", "json_order"] = "foreground_first",
        json_paths: Optional[List[Path]] = None,
    ) -> None:
        super().__init__()
        self.rwmd_root = Path(rwmd_root).resolve()
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_points = num_points
        self.max_instances = max_instances
        self.point_dim = num_points * 2
        self.label_mode = label_mode
        self.instance_order = instance_order

        if json_paths is not None:
            self.json_paths = [Path(p).resolve() for p in json_paths]
        else:
            self.json_paths = sorted(self.rwmd_root.rglob("*.json"))
        if not self.json_paths:
            raise ValueError(f"No JSON files under {self.rwmd_root}")

        folder_names = sorted({p.parent.name for p in self.json_paths})
        self._folder_to_id = {name: i for i, name in enumerate(folder_names)}

    def __len__(self) -> int:
        return len(self.json_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        json_path = self.json_paths[idx]
        with open(json_path, "r", encoding="utf-8") as f:
            labelme = json.load(f)
        shapes = labelme.get("shapes") or []
        path = _rwmd_resolve_image_path(json_path, labelme)

        im_raw, im_aligned = load_rgb_exif_aligned(path)
        w0, h0 = im_aligned.size
        im = im_aligned.resize((self.img_size, self.img_size), Image.BICUBIC)
        image = torch.from_numpy(np.asarray(im).copy()).float().permute(2, 0, 1)

        folder_class = self._folder_to_id.get(json_path.parent.name, 0)
        ordered = _rwmd_order_shapes(shapes, self.instance_order)

        labels: List[int] = []
        masks: List[Tensor] = []
        points: List[Tensor] = []

        for shape in ordered[: self.max_instances]:
            pts = shape.get("points")
            if not isinstance(pts, list) or len(pts) < 3:
                continue
            pts = align_labelme_points_xy(pts, im_raw, im_aligned, labelme)
            m = _rwmd_rasterize_polygon(w0, h0, pts)
            m = m.unsqueeze(0)
            m = F.interpolate(m.unsqueeze(0), size=(self.img_size, self.img_size), mode="nearest").squeeze(0).squeeze(0)
            masks.append(m)

            quad = _rwmd_quad_corners_xy(pts)
            pt = _rwmd_flatten_points_norm(quad, w0, h0, self.num_points)
            points.append(pt)

            lab = _rwmd_instance_class(
                shape,
                label_mode=self.label_mode,
                folder_class=folder_class,
                num_classes=self.num_classes,
            )
            if not (0 <= lab < self.num_classes):
                raise ValueError(f"label {lab} out of range [0, {self.num_classes})")
            labels.append(lab)

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

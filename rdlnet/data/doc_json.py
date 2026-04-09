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
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
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

        im = Image.open(path).convert("RGB")
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

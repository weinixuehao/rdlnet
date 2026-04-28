"""
RWMD preprocessed dataset loader for stage-1 distillation with **point prompts**.

Expected directory layout (as produced by RWMD preprocessing / augmentation scripts):
  - img/*.png          (square, typically 1024×1024)
  - mask/*.png         (uint8 instance-id mask; 0 = background)

Returns:
  image: float tensor [3, S, S] in 0–255 (SAM-style convention)
  points_xy: float tensor [P, 2] in pixel coords (x, y) on the resized image frame
  point_labels: int tensor [P] with 1 for positive points
  meta: dict

Why points (not boxes):
  A single document/background instance may be split into multiple disconnected regions due to
  occlusions/overlap. Sampling one positive point per connected component lets SAM-style prompting
  cover such split instances more reliably than a tight box.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class RWMDPointPromptSample:
    file_name: str
    instance_id: int
    num_components: int


def _stable_rng(seed: int, idx: int) -> random.Random:
    return random.Random((int(seed) << 32) + int(idx))


def _choose_instance_id(mask_u8: np.ndarray, rng: random.Random) -> int:
    ids = np.unique(mask_u8)
    ids = ids[(ids != 0)]
    if ids.size == 0:
        return 0
    return int(ids[rng.randrange(int(ids.size))])


def _sample_points_one_per_component(bin_mask: np.ndarray, *, rng: random.Random, max_points: int) -> Tuple[np.ndarray, int]:
    """
    bin_mask: H×W uint8/bool mask for a single instance id.
    Returns (points_xy float32 [P,2], num_components)
    """
    m = (bin_mask.astype(np.uint8) > 0).astype(np.uint8)
    n_cc, labels = cv2.connectedComponents(m, connectivity=8)
    # labels in [0..n_cc-1], where 0 is background
    comp_ids = [c for c in range(1, int(n_cc)) if (labels == c).any()]
    rng.shuffle(comp_ids)
    if max_points > 0:
        comp_ids = comp_ids[: int(max_points)]

    pts: List[Tuple[float, float]] = []
    for c in comp_ids:
        ys, xs = np.where(labels == c)
        if xs.size == 0:
            continue
        j = rng.randrange(int(xs.size))
        pts.append((float(xs[j]), float(ys[j])))

    # fallback: if something went wrong, sample any foreground pixel
    if not pts:
        ys, xs = np.where(m > 0)
        if xs.size > 0:
            j = rng.randrange(int(xs.size))
            pts.append((float(xs[j]), float(ys[j])))

    out = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    return out, int(max(0, n_cc - 1))


class RWMDPreprocessedPointPrompts(Dataset):
    """
    RWMD preprocessed (square) images + instance-id masks -> point prompts.
    """

    def __init__(
        self,
        *,
        root: str | Path,
        img_size: int = 1024,
        seed: int = 42,
        max_points: int = 8,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.img_dir = self.root / "img"
        self.mask_dir = self.root / "mask"
        self.img_size = int(img_size)
        self.seed = int(seed)
        self.max_points = int(max_points)
        if self.max_points < 1:
            raise ValueError("--max-points must be >= 1")

        if not self.img_dir.is_dir() or not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Expected {self.root} to contain img/ and mask/")

        self._imgs = sorted(self.img_dir.glob("*.png"))
        if not self._imgs:
            raise RuntimeError(f"No .png images found under {self.img_dir}")

    def __len__(self) -> int:
        return len(self._imgs)

    def __getitem__(self, idx: int):
        ip = self._imgs[int(idx)]
        mp = self.mask_dir / ip.name
        if not mp.is_file():
            raise FileNotFoundError(f"Missing mask for {ip.name}: {mp}")

        bgr = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        m_u8 = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if bgr is None or m_u8 is None:
            raise RuntimeError(f"Failed to read RWMD sample: {ip} / {mp}")

        h, w = bgr.shape[:2]
        if h != self.img_size or w != self.img_size or m_u8.shape[0] != h or m_u8.shape[1] != w:
            raise ValueError(
                f"RWMD sample must be square {self.img_size}: got image={w}x{h}, mask={m_u8.shape[1]}x{m_u8.shape[0]} for {ip.name}"
            )

        rng = _stable_rng(self.seed, int(idx))
        inst_id = _choose_instance_id(m_u8, rng)
        if inst_id == 0:
            # Degenerate: no instances; return a single center point (still valid shape-wise).
            points = np.asarray([[0.5 * (w - 1), 0.5 * (h - 1)]], dtype=np.float32)
            num_cc = 0
        else:
            bin_mask = (m_u8 == int(inst_id)).astype(np.uint8)
            points, num_cc = _sample_points_one_per_component(bin_mask, rng=rng, max_points=self.max_points)
            if points.size == 0:
                points = np.asarray([[0.5 * (w - 1), 0.5 * (h - 1)]], dtype=np.float32)

        # BGR -> RGB, HWC -> CHW in 0-255 float
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(rgb).float().permute(2, 0, 1).contiguous()

        pts = torch.from_numpy(points.astype(np.float32))  # [P,2]
        labels = torch.ones((pts.shape[0],), dtype=torch.int64)
        meta: Dict[str, object] = {
            "file_name": ip.name,
            "instance_id": int(inst_id),
            "num_components": int(num_cc),
        }
        return img, pts, labels, meta


def collate_distill_rwmd_points(batch):
    """
    Collate for RWMD point prompts.
    Pads points to max P in batch with label=-1 (ignored by SAM prompt encoder).
    """
    imgs = torch.stack([b[0] for b in batch], dim=0)
    metas = [b[3] for b in batch]

    p_max = max(int(b[1].shape[0]) for b in batch)
    bsz = len(batch)
    points = torch.zeros((bsz, p_max, 2), dtype=torch.float32)
    point_labels = torch.full((bsz, p_max), fill_value=-1, dtype=torch.int64)
    for i, (_img, pts, lbs, _m) in enumerate(batch):
        n = int(pts.shape[0])
        points[i, :n] = pts
        point_labels[i, :n] = lbs
    return imgs, points, point_labels, metas


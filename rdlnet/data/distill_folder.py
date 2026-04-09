"""Unlabeled RGB images for stage-1 Light-SAM distillation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(root: Union[str, Path]) -> List[Path]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")
    out: List[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in _IMG_EXT:
            out.append(p)
    return out


class DistillImageFolder(Dataset):
    """
    Loads images from a directory (recursive). Returns float tensor [3, H, W] in **0–255**
    (same convention as `distill.sam_normalize_images` after stacking).
    """

    def __init__(
        self,
        root: Union[str, Path],
        img_size: int = 1024,
    ) -> None:
        super().__init__()
        self.paths = list_images(root)
        if not self.paths:
            raise RuntimeError(f"No images found under {root}")
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, str]:
        path = self.paths[idx]
        im = Image.open(path).convert("RGB")
        im = im.resize((self.img_size, self.img_size), Image.BICUBIC)
        t = torch.from_numpy(np.asarray(im).copy()).float()  # H,W,3
        t = t.permute(2, 0, 1).contiguous()
        return t, str(path)

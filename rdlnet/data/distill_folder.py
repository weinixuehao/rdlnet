"""Unlabeled RGB images for stage-1 Light-SAM distillation."""

from __future__ import annotations

import hashlib
import random
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


def subset_rng(seed: int, epoch_index: int) -> random.Random:
    """Stable per-(seed, epoch) RNG for reproducible subsets (same across runs / Python versions)."""
    h = hashlib.sha256(f"rdlnet_distill_subset:{seed}:{epoch_index}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "little"))


class DistillImageFolder(Dataset):
    """
    Loads images from a directory (recursive). Returns float tensor [3, H, W] in **0–255**
    (same convention as `distill.sam_normalize_images` after stacking).

    If ``samples_per_epoch`` is set, :meth:`resample_epoch` draws a new random subset (without
    replacement) of that size from the full path list. The caller must invoke ``resample_epoch``
    with the **global** epoch index (0-based, including after ``--resume``) before the first
    ``DataLoader`` iteration (see ``train_distill.py``). Subsets are deterministic given
    ``subset_seed`` and ``epoch_index``.
    """

    def __init__(
        self,
        root: Union[str, Path],
        img_size: int = 1024,
        samples_per_epoch: Optional[int] = None,
        subset_seed: int = 42,
    ) -> None:
        super().__init__()
        self.paths = list_images(root)
        if not self.paths:
            raise RuntimeError(f"No images found under {root}")
        self.img_size = img_size
        self.samples_per_epoch = samples_per_epoch
        self._subset_seed = subset_seed
        if samples_per_epoch is not None:
            if samples_per_epoch < 1:
                raise ValueError("samples_per_epoch must be >= 1")
            self._epoch_indices: List[int] = []
        else:
            self._epoch_indices = list(range(len(self.paths)))

    @property
    def pool_size(self) -> int:
        return len(self.paths)

    def resample_epoch(self, epoch_index: int) -> None:
        """Redraw subset for global epoch ``epoch_index`` (0-based). No-op if ``samples_per_epoch`` is None."""
        if self.samples_per_epoch is None:
            return
        k = min(self.samples_per_epoch, len(self.paths))
        rng = subset_rng(self._subset_seed, epoch_index)
        self._epoch_indices = rng.sample(range(len(self.paths)), k)

    def __len__(self) -> int:
        return len(self._epoch_indices)

    def __getitem__(self, idx: int) -> Tuple[Tensor, str]:
        path = self.paths[self._epoch_indices[idx]]
        im = Image.open(path).convert("RGB")
        # Letterbox to square (keep aspect, then pad).
        w0, h0 = im.size
        s = float(self.img_size) / float(max(int(w0), int(h0), 1))
        new_w = max(1, int(round(float(w0) * s)))
        new_h = max(1, int(round(float(h0) * s)))
        pad_x = int((self.img_size - new_w) // 2)
        pad_y = int((self.img_size - new_h) // 2)
        im_r = im.resize((new_w, new_h), Image.BICUBIC)
        canvas = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))
        canvas.paste(im_r, (pad_x, pad_y))
        t = torch.from_numpy(np.asarray(canvas).copy()).float()  # H,W,3
        t = t.permute(2, 0, 1).contiguous()
        return t, str(path)

"""
COCO instance dataset loader for stage-1 distillation with box prompts.

Reads images and annotations from extracted files (no zip support).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class CocoBoxSample:
    image_id: int
    file_name: str
    bbox_xywh: Tuple[float, float, float, float]


def _load_instances_json(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file() or p.suffix != ".json":
        raise FileNotFoundError(f"Expected extracted COCO instances json file, got: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _bbox_xywh_to_xyxy(b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x, y, w, h = b
    return (x, y, x + w, y + h)


class CocoTrain2017BoxPrompts(Dataset):
    """
    Returns:
      image: float tensor [3, S, S] in 0–255 (SAM-style), resized to square.
      box_xyxy: float tensor [4] in resized-image pixel coords.
      meta: dict with image_id, file_name (optional debug)
    """

    def __init__(
        self,
        *,
        images: Union[str, Path],
        instances_json: str | Path,
        img_size: int = 1024,
        seed: int = 42,
        instances_per_image: int = 1,
        max_images: int | None = None,
    ) -> None:
        super().__init__()
        self.images = Path(images)
        self.instances_json = Path(instances_json)
        self.img_size = int(img_size)
        self.seed = int(seed)
        self.instances_per_image = int(instances_per_image)
        if self.instances_per_image < 1:
            raise ValueError("instances_per_image must be >= 1")

        if not self.images.is_dir():
            raise FileNotFoundError(f"Expected extracted COCO images directory, got: {self.images}")

        data = _load_instances_json(self.instances_json)
        images = data.get("images", [])
        anns = data.get("annotations", [])

        id_to_file: Dict[int, str] = {int(im["id"]): str(im["file_name"]) for im in images}
        id_to_size: Dict[int, Tuple[int, int]] = {int(im["id"]): (int(im["width"]), int(im["height"])) for im in images}

        per_img: Dict[int, List[Tuple[float, float, float, float]]] = {}
        for a in anns:
            if a.get("iscrowd", 0) == 1:
                continue
            img_id = int(a["image_id"])
            bbox = a.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x, y, w, h = map(float, bbox)
            if w <= 1 or h <= 1:
                continue
            per_img.setdefault(img_id, []).append((x, y, w, h))

        samples: List[CocoBoxSample] = []
        for img_id, bxs in per_img.items():
            fn = id_to_file.get(img_id)
            if not fn:
                continue
            # store all bboxes; we will sample per __getitem__ deterministically
            for b in bxs:
                samples.append(CocoBoxSample(image_id=img_id, file_name=fn, bbox_xywh=b))

        # We want one entry per image for stable batching, but still allow selecting among instances.
        self._image_ids = sorted(per_img.keys())
        if max_images is not None:
            self._image_ids = self._image_ids[: int(max_images)]

        self._id_to_file = id_to_file
        self._id_to_size = id_to_size
        self._per_img_boxes = per_img

    def __len__(self) -> int:
        return len(self._image_ids)

    def _rng_for_index(self, idx: int) -> random.Random:
        # stable per-index RNG
        return random.Random((self.seed << 32) + idx)

    def __getitem__(self, idx: int):
        image_id = self._image_ids[idx]
        file_name = self._id_to_file[image_id]
        w0, h0 = self._id_to_size[image_id]

        # open image
        p = self.images / file_name
        im = Image.open(p).convert("RGB")

        # resize to square (simple, deterministic; keeps SAM normalization convention)
        im_resized = im.resize((self.img_size, self.img_size), Image.BICUBIC)
        arr = np.asarray(im_resized).copy()  # H,W,3 uint8
        t = torch.from_numpy(arr).float().permute(2, 0, 1).contiguous()  # 3,H,W in 0–255

        # sample box(es) from this image
        boxes = self._per_img_boxes[image_id]
        rng = self._rng_for_index(idx)
        chosen = [boxes[rng.randrange(len(boxes))] for _ in range(self.instances_per_image)]

        # scale bbox from original image coords to resized coords
        sx = self.img_size / max(1.0, float(w0))
        sy = self.img_size / max(1.0, float(h0))
        out_boxes: List[Tensor] = []
        for b in chosen:
            x1, y1, x2, y2 = _bbox_xywh_to_xyxy(b)
            x1 *= sx
            x2 *= sx
            y1 *= sy
            y2 *= sy
            out_boxes.append(torch.tensor([x1, y1, x2, y2], dtype=torch.float32))

        box = out_boxes[0] if len(out_boxes) == 1 else torch.stack(out_boxes, dim=0)  # [4] or [K,4]
        meta = {"image_id": image_id, "file_name": file_name}
        return t, box, meta


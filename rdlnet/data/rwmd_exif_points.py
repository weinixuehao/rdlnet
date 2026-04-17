"""
Map LabelMe polygon coordinates into the same pixel frame as ``ImageOps.exif_transpose``.

Some JPEGs store landscape pixels + EXIF Orientation; LabelMe may save ``imageWidth``/``imageHeight``
matching that raw bitmap while polygons are in raw coordinates. Others save upright dimensions
and points already upright. We branch on ``(imageWidth, imageHeight)`` vs raw vs transposed sizes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

from PIL import ExifTags, Image, ImageOps
from PIL.Image import Transpose

# Same mapping as ``PIL.ImageOps.exif_transpose`` (orientation -> single Transpose op).
_ORIENT_TO_METHOD = {
    2: Transpose.FLIP_LEFT_RIGHT,
    3: Transpose.ROTATE_180,
    4: Transpose.FLIP_TOP_BOTTOM,
    5: Transpose.TRANSPOSE,
    6: Transpose.ROTATE_270,
    7: Transpose.TRANSVERSE,
    8: Transpose.ROTATE_90,
}


def _transpose_point_xy_forward(x: float, y: float, W: float, H: float, method: Transpose) -> Tuple[float, float]:
    """
    Where a pixel at (x, y) in the *pre-transpose* bitmap appears after ``Image.transpose(method)``.

    Formulas match Pillow's transpose tests (Tests/test_image_transpose.py).
    """
    if method is Transpose.FLIP_LEFT_RIGHT:
        return (W - 1.0 - x, y)
    if method is Transpose.FLIP_TOP_BOTTOM:
        return (x, H - 1.0 - y)
    if method is Transpose.ROTATE_180:
        return (W - 1.0 - x, H - 1.0 - y)
    if method is Transpose.ROTATE_90:
        return (y, W - 1.0 - x)
    if method is Transpose.ROTATE_270:
        return (H - 1.0 - y, x)
    if method is Transpose.TRANSPOSE:
        return (y, x)
    if method is Transpose.TRANSVERSE:
        return (H - 1.0 - y, W - 1.0 - x)
    return (x, y)


def map_points_raw_to_exif_aligned(points_xy: Sequence[Sequence[float]], im_raw: Image.Image) -> List[List[float]]:
    """Apply the same geometric op as ``ImageOps.exif_transpose`` to polygon coordinates."""
    exif = im_raw.getexif()
    ori = exif.get(ExifTags.Base.Orientation, 1)
    try:
        orientation = int(ori)
    except (TypeError, ValueError):
        orientation = 1
    method = _ORIENT_TO_METHOD.get(orientation)
    if method is None:
        return [[float(p[0]), float(p[1])] for p in points_xy]
    W, H = float(im_raw.size[0]), float(im_raw.size[1])
    out: List[List[float]] = []
    for p in points_xy:
        x, y = float(p[0]), float(p[1])
        nx, ny = _transpose_point_xy_forward(x, y, W, H, method)
        out.append([nx, ny])
    return out


def _scale_points_xy(
    points_xy: Sequence[Sequence[float]], jw: int, jh: int, aw: int, ah: int
) -> List[List[float]]:
    sx = aw / float(jw)
    sy = ah / float(jh)
    if abs(sx - sy) < 1e-6:
        s = (sx + sy) * 0.5
        sx = sy = s
    return [[float(p[0]) * sx, float(p[1]) * sy] for p in points_xy]


def align_labelme_points_xy(
    points_xy: Sequence[Sequence[float]],
    im_raw: Image.Image,
    im_aligned: Image.Image,
    labelme: Dict[str, Any],
) -> List[List[float]]:
    """
    Return polygon points in the same coordinate system as ``im_aligned`` (after ``exif_transpose``).

    - If JSON ``(imageWidth, imageHeight)`` matches ``im_aligned.size``, points are already upright.
    - If they match ``im_raw.size`` and EXIF transposition changed size, points are in raw storage space.
    - If metadata is missing but raw vs aligned sizes differ, assume raw coordinates (common RWMD case).
    """
    rw, rh = im_raw.size
    aw, ah = im_aligned.size
    base = [[float(p[0]), float(p[1])] for p in points_xy]

    jw = labelme.get("imageWidth")
    jh = labelme.get("imageHeight")
    if jw is not None and jh is not None:
        jw_i, jh_i = int(jw), int(jh)
    else:
        jw_i, jh_i = None, None

    if (rw, rh) == (aw, ah):
        if jw_i is not None and jh_i is not None and (jw_i, jh_i) != (aw, ah):
            return _scale_points_xy(base, jw_i, jh_i, aw, ah)
        return base

    # Pixel array was transposed by exif_transpose.
    if jw_i is not None and jh_i is not None:
        if (jw_i, jh_i) == (aw, ah):
            return base
        if (jw_i, jh_i) == (rw, rh):
            return map_points_raw_to_exif_aligned(base, im_raw)
        return _scale_points_xy(map_points_raw_to_exif_aligned(base, im_raw), jw_i, jh_i, aw, ah)

    return map_points_raw_to_exif_aligned(base, im_raw)


def load_rgb_exif_aligned(path: Union[str, Path]) -> tuple[Image.Image, Image.Image]:
    """Return ``(im_raw, im_aligned_rgb)`` for resolving LabelMe coordinate spaces."""
    im_raw = Image.open(path)
    im_raw.load()
    im_aligned = ImageOps.exif_transpose(im_raw.convert("RGB"))
    return im_raw, im_aligned

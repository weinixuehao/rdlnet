#!/usr/bin/env python3
"""
Visualize RWMD LabelMe annotations: polygons, foreground_doc highlight, and quad corners
(same corner rule as ``RWMDLabelMeDataset._rwmd_quad_corners_xy``).

Usage (from repo root)::

    python scripts/visualize_rwmd.py --rwmd-root dataset/RWMD_dataset/RWMD_dataset_v1 \\
        --out-dir vis_rwmd --num-samples 20 --seed 0

Requires: Pillow only for deps; loads ``rdlnet/data/rwmd_exif_points.py`` via importlib (no torch).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal

from PIL import Image, ImageDraw, ImageFont

_REPO = Path(__file__).resolve().parent.parent
_EXIF_POINTS = _REPO / "rdlnet" / "data" / "rwmd_exif_points.py"
_spec = importlib.util.spec_from_file_location("rwmd_exif_points", _EXIF_POINTS)
assert _spec and _spec.loader
_rwmd_exif = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rwmd_exif)
align_labelme_points_xy: Callable[..., List[List[float]]] = _rwmd_exif.align_labelme_points_xy
load_rgb_exif_aligned = _rwmd_exif.load_rgb_exif_aligned

# --- mirror rdlnet.data.doc_json helpers (no torch) -----------------------------

ShapeOrder = Literal["foreground_first", "numeric_then_foreground", "json_order"]


def _resolve_image_path(json_path: Path, labelme: Dict[str, Any]) -> Path:
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


def _order_shapes(shapes: List[Dict[str, Any]], order: ShapeOrder) -> List[Dict[str, Any]]:
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
        return fg + num
    if order == "numeric_then_foreground":
        return num + fg
    return [
        s
        for s in shapes
        if s.get("label") == "foreground_doc" or (isinstance(s.get("label"), str) and s["label"].isdigit())
    ]


def _strip_closing_vertex(points_xy: List[List[float]]) -> List[List[float]]:
    """Match ``_rwmd_strip_closing_vertex`` in ``rdlnet.data.doc_json`` (no numpy)."""
    if len(points_xy) < 2:
        return points_xy
    p0, p1 = points_xy[0], points_xy[-1]
    if abs(float(p0[0]) - float(p1[0])) <= 1e-3 and abs(float(p0[1]) - float(p1[1])) <= 1e-3:
        return points_xy[:-1]
    return points_xy


def _quad_corners_xy(points_xy: List[List[float]]) -> list[tuple[float, float]]:
    points_xy = _strip_closing_vertex(points_xy)
    n = len(points_xy)
    if n == 4:
        return [(float(p[0]), float(p[1])) for p in points_xy]
    if n < 3:
        return [(0.0, 0.0)] * 4
    xs = [float(p[0]) for p in points_xy]
    ys = [float(p[1]) for p in points_xy]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


# Foreground / background overlay: ~50% fill so regions read clearly (not washed out).
_FG_RGB = (45, 175, 85)
_BG_RGB = (255, 200, 45)
_FILL_ALPHA = 128  # 0.5 * 255


def _try_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in (
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        p = Path(name)
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def draw_sample(
    json_path: Path,
    out_path: Path,
    *,
    instance_order: ShapeOrder,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        labelme = json.load(f)
    shapes = labelme.get("shapes") or []
    img_path = _resolve_image_path(json_path, labelme)
    im_raw, im_aligned = load_rgb_exif_aligned(img_path)
    im = im_aligned.convert("RGBA")
    w, h = im.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw_base = ImageDraw.Draw(im)

    ordered = _order_shapes(shapes, instance_order)
    # Draw background (yellow) first, foreground (green) last — same instances as training
    # order, but compositing must put foreground on top or the page interior reads as yellow.
    bg_shapes = [s for s in ordered if s.get("label") != "foreground_doc"]
    fg_shapes = [s for s in ordered if s.get("label") == "foreground_doc"]
    draw_order = bg_shapes + fg_shapes

    for shape in draw_order:
        lab = str(shape.get("label", "?"))
        pts = shape.get("points")
        if not isinstance(pts, list) or len(pts) < 3:
            continue
        pts = align_labelme_points_xy(pts, im_raw, im_aligned, labelme)
        flat = [(float(p[0]), float(p[1])) for p in pts]
        is_fg = lab == "foreground_doc"
        rgb = _FG_RGB if is_fg else _BG_RGB
        fill = (*rgb, _FILL_ALPHA)
        outline = (*rgb, 255)
        width = 4 if is_fg else 2
        draw.polygon(flat, outline=outline, fill=fill, width=width)

        quad = _quad_corners_xy(pts)
        for j, (qx, qy) in enumerate(quad):
            r = 10 if is_fg else 7
            draw_base.ellipse(
                [qx - r, qy - r, qx + r, qy + r],
                outline=(255, 255, 255, 255),
                width=2,
                fill=(*rgb, 235),
            )
            draw.text((qx + 12, qy - 8), f"{lab}:{j}", font=font, fill=(255, 255, 255, 255))

    composed = Image.alpha_composite(im, overlay)
    composed.convert("RGB").save(out_path, quality=95)


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize RWMD LabelMe JSON + quad corners")
    p.add_argument("--rwmd-root", type=str, required=True, help="e.g. dataset/RWMD_dataset/RWMD_dataset_v1")
    p.add_argument("--out-dir", type=str, required=True, help="Directory to write PNGs")
    p.add_argument("--num-samples", type=int, default=30, help="How many random JSON files to render")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--instance-order",
        type=str,
        choices=["foreground_first", "numeric_then_foreground", "json_order"],
        default="foreground_first",
        help="Same ordering as RWMDLabelMeDataset / train_rdlnet",
    )
    args = p.parse_args()

    root = Path(args.rwmd_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_json = sorted(root.rglob("*.json"))
    if not all_json:
        raise SystemExit(f"No JSON under {root}")

    rng = random.Random(args.seed)
    pick = all_json if args.num_samples >= len(all_json) else rng.sample(all_json, args.num_samples)

    font = _try_font(16)
    for jp in pick:
        rel = jp.relative_to(root)
        safe = str(rel).replace("/", "_")
        out_path = out_dir / f"{safe}.png"
        draw_sample(jp, out_path, instance_order=args.instance_order, font=font)
        print(out_path)

    print(f"Done: {len(pick)} images -> {out_dir}")


if __name__ == "__main__":
    main()

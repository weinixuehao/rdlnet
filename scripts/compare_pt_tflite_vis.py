#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare RDLNet PyTorch (.pt) vs TFLite outputs visually (points polygon).")
    p.add_argument("--ckpt", type=str, required=True, help="Path to rdlnet_best.pt (train_rdlnet checkpoint).")
    p.add_argument("--tflite", type=str, required=True, help="Path to exported rdlnet_points.tflite.")
    p.add_argument("--image", type=str, required=True, help="Path to input image.")
    p.add_argument("--out", type=str, required=True, help="Output visualization image path (png/jpg).")
    p.add_argument("--img-size", type=int, default=1024, help="Resize image to this size (H=W). Must match export.")
    p.add_argument("--input-range", type=str, default="0_1", choices=["0_1", "0_255"], help="Input range fed into TFLite.")
    p.add_argument("--doc-class-id", type=int, default=0, help="Class id used to pick best query (softmax prob).")
    p.add_argument("--device", type=str, default="cpu", help="Torch device for .pt inference (cpu/cuda/mps).")
    return p.parse_args()


def _load_rgb_resized_u8(path: Path, img_size: int) -> np.ndarray:
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"Failed to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h0, w0 = int(rgb.shape[0]), int(rgb.shape[1])
    s = float(img_size) / float(max(h0, w0))
    new_w = max(1, int(round(float(w0) * s)))
    new_h = max(1, int(round(float(h0) * s)))
    pad_x = int((img_size - new_w) // 2)
    pad_y = int((img_size - new_h) // 2)
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w, :] = resized
    return canvas


def _load_rgb_u8(path: Path) -> np.ndarray:
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _prep_input(rgb_u8: np.ndarray, *, input_range: str) -> np.ndarray:
    x = rgb_u8.astype(np.float32)
    if input_range == "0_1":
        x *= 1.0 / 255.0
    # NCHW float32
    x = np.transpose(x, (2, 0, 1))[None, ...]
    return x


def _sam_norm_torch(x: "torch.Tensor", *, input_range: str) -> "torch.Tensor":
    import torch

    y = x
    if input_range == "0_1":
        y = y * 255.0
    mean = torch.tensor([123.675, 116.28, 103.53], device=y.device, dtype=y.dtype).view(1, 3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375], device=y.device, dtype=y.dtype).view(1, 3, 1, 1)
    return (y - mean) / std


def _pick_best_q(pred_logits: np.ndarray, *, doc_class_id: int) -> tuple[int, float]:
    # pred_logits: [Nq, C]
    x = pred_logits.astype(np.float32)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    p = e / np.clip(np.sum(e, axis=-1, keepdims=True), 1e-12, None)
    s = p[:, int(doc_class_id)]
    q = int(np.argmax(s))
    return q, float(s[q])


def _first4_poly_xy(points_flat: np.ndarray) -> np.ndarray:
    pts = points_flat.reshape(-1, 2)[:4]
    pts = pts[(pts[:, 0] >= 0.0) & (pts[:, 1] >= 0.0)]
    return pts


def _inverse_letterbox_points01_to_orig_px(
    pts01: np.ndarray, *, orig_w: int, orig_h: int, out_size: int
) -> np.ndarray:
    """
    pts01: (N,2) normalized in [0,1] w.r.t the letterbox canvas out_size×out_size.
    returns: (N,2) float32 in original image pixel coords.
    """
    s = float(out_size) / float(max(int(orig_w), int(orig_h), 1))
    new_w = max(1, int(round(float(orig_w) * s)))
    new_h = max(1, int(round(float(orig_h) * s)))
    pad_x = int((out_size - new_w) // 2)
    pad_y = int((out_size - new_h) // 2)
    pts = pts01.astype(np.float32).copy()
    # normalized -> canvas pixels
    pts[:, 0] = pts[:, 0] * float(out_size)
    pts[:, 1] = pts[:, 1] * float(out_size)
    # inverse letterbox
    pts[:, 0] = (pts[:, 0] - float(pad_x)) / float(s)
    pts[:, 1] = (pts[:, 1] - float(pad_y)) / float(s)
    # clamp to image bounds (keep floats for nicer overlay)
    pts[:, 0] = np.clip(pts[:, 0], 0.0, float(max(orig_w - 1, 1)))
    pts[:, 1] = np.clip(pts[:, 1], 0.0, float(max(orig_h - 1, 1)))
    return pts


def _draw_overlay(rgb_u8: np.ndarray, *, pts01: np.ndarray, title: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    h, w = int(rgb_u8.shape[0]), int(rgb_u8.shape[1])
    im = Image.fromarray(rgb_u8, mode="RGB")
    draw = ImageDraw.Draw(im)

    poly = []
    for (xn, yn) in pts01.tolist():
        px = int(round(xn * float(max(w - 1, 1))))
        py = int(round(yn * float(max(h - 1, 1))))
        px = max(0, min(w - 1, px))
        py = max(0, min(h - 1, py))
        poly.append((px, py))
        r = 5
        draw.ellipse((px - r, py - r, px + r, py + r), fill=(0, 255, 0), outline=(255, 255, 255), width=1)
    if len(poly) >= 3:
        draw.line(poly + [poly[0]], fill=(0, 128, 255), width=3, joint="curve")

    draw.rectangle((0, 0, w, 26), fill=(0, 0, 0))
    draw.text((6, 5), title, fill=(255, 255, 255))
    return np.asarray(im, dtype=np.uint8)


def infer_pt(
    *,
    ckpt_path: Path,
    inp_nchw: np.ndarray,
    input_range: str,
    device: str,
    doc_class_id: int,
) -> tuple[int, float, np.ndarray]:
    import torch

    from rdlnet.model import RDLNet, RDLNetConfig

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise SystemExit(f"Invalid checkpoint: expected dict with key 'model': {ckpt_path}")

    cfg_dict = ckpt.get("config") if isinstance(ckpt.get("config"), dict) else {}
    field_names = {f.name for f in fields(RDLNetConfig)}
    init_kwargs = {k: v for (k, v) in cfg_dict.items() if k in field_names}
    init_kwargs["use_sam_pixel_norm"] = False  # keep preprocessing explicit here
    cfg = RDLNetConfig(**init_kwargs)

    model = RDLNet(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    x = torch.from_numpy(inp_nchw).to(device=device, dtype=torch.float32)
    x = _sam_norm_torch(x, input_range=input_range)
    with torch.no_grad():
        out = model(x)

    pred_logits = out["pred_logits"][0].float().detach().cpu().numpy()  # [Nq,C]
    pred_points = out["pred_points"][0].float().detach().cpu().numpy()  # [Nq,2P]
    q, score = _pick_best_q(pred_logits, doc_class_id=doc_class_id)
    return q, score, pred_points[q]


def infer_tflite(
    *,
    tflite_path: Path,
    inp_nchw: np.ndarray,
    doc_class_id: int,
) -> tuple[int, float, np.ndarray]:
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    in0 = interp.get_input_details()[0]
    interp.set_tensor(in0["index"], inp_nchw.astype(in0["dtype"]))
    interp.invoke()

    outs = interp.get_output_details()
    # Identify logits vs points by last dim parity (points is 2P, even).
    logits = None
    points = None
    for d in outs:
        shape = tuple(int(x) for x in d["shape"])
        y = interp.get_tensor(d["index"])
        if len(shape) == 3 and (shape[2] % 2 == 0) and shape[2] >= 4:
            points = y[0]  # [Nq,2P]
        elif len(shape) == 3:
            logits = y[0]  # [Nq,C]
    if logits is None or points is None:
        raise SystemExit(f"Failed to identify outputs. Got: {[tuple(map(int, d['shape'])) for d in outs]}")

    q, score = _pick_best_q(logits, doc_class_id=doc_class_id)
    return q, score, points[q]


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    tflite_path = Path(args.tflite)
    img_path = Path(args.image)
    out_path = Path(args.out)

    rgb_canvas = _load_rgb_resized_u8(img_path, int(args.img_size))
    inp = _prep_input(rgb_canvas, input_range=str(args.input_range))

    q_pt, s_pt, pts_pt = infer_pt(
        ckpt_path=ckpt_path,
        inp_nchw=inp,
        input_range=str(args.input_range),
        device=str(args.device),
        doc_class_id=int(args.doc_class_id),
    )
    q_tf, s_tf, pts_tf = infer_tflite(tflite_path=tflite_path, inp_nchw=inp, doc_class_id=int(args.doc_class_id))

    poly_pt = _first4_poly_xy(pts_pt)
    poly_tf = _first4_poly_xy(pts_tf)

    # Visualize on ORIGINAL image (more realistic for downstream use).
    rgb_orig = _load_rgb_u8(img_path)
    orig_h, orig_w = int(rgb_orig.shape[0]), int(rgb_orig.shape[1])
    poly_pt_px = _inverse_letterbox_points01_to_orig_px(
        poly_pt, orig_w=orig_w, orig_h=orig_h, out_size=int(args.img_size)
    )
    poly_tf_px = _inverse_letterbox_points01_to_orig_px(
        poly_tf, orig_w=orig_w, orig_h=orig_h, out_size=int(args.img_size)
    )

    # Reuse drawing by converting px -> normalized in original frame.
    poly_pt01_orig = poly_pt_px.copy()
    poly_tf01_orig = poly_tf_px.copy()
    poly_pt01_orig[:, 0] /= float(max(orig_w - 1, 1))
    poly_pt01_orig[:, 1] /= float(max(orig_h - 1, 1))
    poly_tf01_orig[:, 0] /= float(max(orig_w - 1, 1))
    poly_tf01_orig[:, 1] /= float(max(orig_h - 1, 1))

    left = _draw_overlay(rgb_orig, pts01=poly_pt01_orig, title=f"PT q={q_pt} score={s_pt:.4f}")
    right = _draw_overlay(rgb_orig, pts01=poly_tf01_orig, title=f"TFLite q={q_tf} score={s_tf:.4f}")

    vis = np.concatenate([left, right], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.fromarray(vis, mode="RGB").save(out_path)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()


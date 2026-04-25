#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from dataclasses import fields
from rdlnet.model import RDLNetConfig, apply_lite_preset

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export RDLNet .pt checkpoint to TFLite via LiteRT Torch")
    p.add_argument("--ckpt", type=str, required=True, help="Path to train_rdlnet checkpoint (.pt)")
    p.add_argument("--out-dir", type=str, default="export", help="Output directory")
    p.add_argument("--img-size", type=int, default=1024, help="Model input size (H=W)")
    p.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Number of foreground classes (not counting background). If omitted, try ckpt['config']['num_classes'], else 2.",
    )
    p.add_argument(
        "--use-sam-pixel-norm",
        action="store_true",
        help="Apply SAM pixel mean/std normalization (fixed, export-friendly preprocessing; no data-dependent branches).",
    )
    p.add_argument(
        "--input-range",
        type=str,
        default="0_1",
        choices=["0_1", "0_255"],
        help="Expected input image value range for preprocessing. Use 0_1 if you feed float32 in [0,1]; use 0_255 if you feed [0,255].",
    )
    p.add_argument(
        "--export",
        type=str,
        default="points",
        choices=["points", "full"],
        help="Export mode: 'points' exports pred_logits + pred_points (recommended). 'full' adds pred_masks too.",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Export batch size (fixed).",
    )
    p.add_argument(
        "--fp16",
        action="store_true",
        help="Enable FP16 weight quantization in TFLite converter (smaller model; I/O stays float32).",
    )
    return p.parse_args()


def _load_cfg(ckpt: dict, *, img_size: int, num_classes: int | None):
    cfg_dict = ckpt.get("config") if isinstance(ckpt.get("config"), dict) else {}

    # train_rdlnet saves `config = asdict(cfg)` (see train_rdlnet.py), so this should contain the
    # *fully materialized* architecture (already after apply_lite_preset).
    # For robustness, also support older checkpoints that store only `lite`.
    field_names = {f.name for f in fields(RDLNetConfig)}
    init_kwargs: dict = {}
    for k, v in cfg_dict.items():
        if k in field_names:
            init_kwargs[k] = v

    # Always let export-time args override these.
    init_kwargs["img_size"] = int(img_size)
    if num_classes is not None:
        init_kwargs["num_classes"] = int(num_classes)
    else:
        # Default fallback only when ckpt doesn't carry num_classes.
        if "num_classes" not in init_kwargs:
            init_kwargs["num_classes"] = 2
    # Export uses a wrapper for fixed preprocessing (export-friendly). Keep the core model free of
    # data-dependent preprocessing branches.
    init_kwargs["use_sam_pixel_norm"] = False

    cfg = RDLNetConfig(**init_kwargs)

    return cfg


def export_tflite(
    *,
    ckpt_path: Path,
    out_dir: Path,
    img_size: int,
    num_classes: int | None,
    use_sam_pixel_norm: bool,
    input_range: str,
    batch: int,
    export: str,
    fp16: bool,
) -> Path:
    import torch
    import torch.nn as nn

    from rdlnet.model import RDLNet

    try:
        import litert_torch
    except Exception as e:
        raise SystemExit(
            "litert-torch is required for direct TFLite export.\n"
            "Install it in your environment, e.g.\n"
            "  pip install -r requirements-export.txt\n"
            f"Import error: {e}"
        ) from e

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise SystemExit(f"Invalid checkpoint: expected dict with key 'model': {ckpt_path}")

    # IMPORTANT: RDLNet._preprocess_pixels contains a data-dependent branch (x.max()) that can break export.
    # Keep the core model free of such branches and apply fixed preprocessing in the wrapper.
    cfg = _load_cfg(ckpt, img_size=img_size, num_classes=num_classes)
    model = RDLNet(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    class Wrapper(nn.Module):
        def __init__(self, m: nn.Module, *, sam_norm: bool, in_range: str) -> None:
            super().__init__()
            self.m = m
            self.sam_norm = bool(sam_norm)
            self.in_range = str(in_range)

        def forward(self, images):  # type: ignore[override]
            x = images
            if self.sam_norm:
                if self.in_range == "0_1":
                    x = x * 255.0
                mean = torch.tensor([123.675, 116.28, 103.53], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
                std = torch.tensor([58.395, 57.12, 57.375], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
                x = (x - mean) / std
            out = self.m(x)
            if export == "full":
                return out["pred_logits"], out["pred_masks"], out["pred_points"]
            return out["pred_logits"], out["pred_points"]

    wrapped = Wrapper(model, sam_norm=use_sam_pixel_norm, in_range=input_range)
    wrapped.eval()
    dummy = torch.randn(batch, 3, img_size, img_size)

    out_dir.mkdir(parents=True, exist_ok=True)
    tflite_path = out_dir / ("rdlnet_points.tflite" if export == "points" else "rdlnet_full.tflite")
    ai_edge_flags = None
    if fp16:
        # LiteRT Torch goes through TF Lite converter; this sets "float16 weight quantization".
        # https://www.tensorflow.org/lite/performance/post_training_quantization#float16_quantization
        import tensorflow as tf

        ai_edge_flags = {
            "optimizations": [tf.lite.Optimize.DEFAULT],
            "target_spec": {"supported_types": [tf.float16]},
        }

    edge_model = litert_torch.convert(wrapped, (dummy,), _ai_edge_converter_flags=ai_edge_flags)
    edge_model.export(str(tflite_path))
    return tflite_path


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    out_dir = Path(args.out_dir)
    tflite_path = export_tflite(
        ckpt_path=ckpt_path,
        out_dir=out_dir,
        img_size=int(args.img_size),
        num_classes=args.num_classes,
        use_sam_pixel_norm=bool(args.use_sam_pixel_norm),
        input_range=str(args.input_range),
        batch=int(args.batch),
        export=str(args.export),
        fp16=bool(args.fp16),
    )
    print(f"TFLite exported -> {tflite_path}")

    # Minimal "main document points" postprocess note:
    print("\nPostprocess (main document points):")
    print("- doc_class_id = 0 (top sheet)")
    print("- q* = argmax_q softmax(pred_logits[q])[doc_class_id]")
    print("- main_points = pred_points[q*]  # normalized 0..1, multiply by (W,H) for pixels")


if __name__ == "__main__":
    # Avoid OpenMP oversubscription during export by default.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()


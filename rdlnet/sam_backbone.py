"""
Light-SAM encoder using Meta SAM `ImageEncoderViT` (segment-anything).

Aligned with supplementary Table 2 (student): embed_dim=384, depth=12, heads=8,
patch=16, global_attn_indexes=(2, 8), window_size=14, relative positional encodings.

Does not apply the SAM neck (1×1 + 3×3 convs): RDLNet uses raw patch tokens I' as in the paper.

Loads `modeling/{common,image_encoder}.py` directly so we do not import `segment_anything/__init__.py`
(which requires torchvision).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from functools import partial
from pathlib import Path
from typing import List, Tuple, Type

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAM_ROOT = _REPO_ROOT / "segment-anything"
_IMPORT_ERR: Exception | None = None
ImageEncoderViT: Type[nn.Module] | None = None
PromptEncoder: Type[nn.Module] | None = None
MaskDecoder: Type[nn.Module] | None = None
TwoWayTransformer: Type[nn.Module] | None = None

if _SAM_ROOT.is_dir():
    try:

        def _load_module(name: str, path: Path) -> object:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load {path}")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return mod

        if "segment_anything" not in sys.modules:
            p = types.ModuleType("segment_anything")
            p.__path__ = [str(_SAM_ROOT / "segment_anything")]  # type: ignore[attr-defined]
            sys.modules["segment_anything"] = p
        if "segment_anything.modeling" not in sys.modules:
            p = types.ModuleType("segment_anything.modeling")
            p.__path__ = [str(_SAM_ROOT / "segment_anything" / "modeling")]  # type: ignore[attr-defined]
            sys.modules["segment_anything.modeling"] = p

        _modeling = _SAM_ROOT / "segment_anything" / "modeling"
        _common_path = _modeling / "common.py"
        _enc_path = _modeling / "image_encoder.py"
        _prompt_path = _modeling / "prompt_encoder.py"
        _mask_path = _modeling / "mask_decoder.py"
        _tx_path = _modeling / "transformer.py"

        if _common_path.is_file() and _enc_path.is_file() and _prompt_path.is_file() and _mask_path.is_file() and _tx_path.is_file():
            _load_module("segment_anything.modeling.common", _common_path)
            _tx_mod = _load_module("segment_anything.modeling.transformer", _tx_path)
            _ie_mod = _load_module("segment_anything.modeling.image_encoder", _enc_path)
            _pe_mod = _load_module("segment_anything.modeling.prompt_encoder", _prompt_path)
            _md_mod = _load_module("segment_anything.modeling.mask_decoder", _mask_path)

            TwoWayTransformer = _tx_mod.TwoWayTransformer
            ImageEncoderViT = _ie_mod.ImageEncoderViT
            PromptEncoder = _pe_mod.PromptEncoder
            MaskDecoder = _md_mod.MaskDecoder
        else:
            _IMPORT_ERR = FileNotFoundError("segment-anything modeling files missing")
    except Exception as e:  # pragma: no cover
        _IMPORT_ERR = e
        ImageEncoderViT = None
        PromptEncoder = None
        MaskDecoder = None
        TwoWayTransformer = None
else:
    _IMPORT_ERR = FileNotFoundError("segment-anything directory not found")


class RDLNetSAMEncoder(nn.Module):
    """
    SAM ViT image encoder (windowed + global blocks) without neck.
    Returns last tokens and three cross-level sequences (layers 1, l/2, l in 1-based indexing).
    """

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        window_size: int = 14,
        global_attn_indexes: Tuple[int, ...] = (2, 8),
    ) -> None:
        if ImageEncoderViT is None:
            raise ImportError(
                "segment_anything is required for RDLNetSAMEncoder. "
                "Install with: pip install -e ./segment-anything "
                f"(original error: {_IMPORT_ERR})"
            )
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.patch_size = patch_size
        # Paper Table 2: first, middle, last block (1-based 1, l/2, l)
        self.cross_level_indices = (0, depth // 2 - 1, depth - 1)

        self.encoder = ImageEncoderViT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            out_chans=256,  # unused (neck removed)
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            use_abs_pos=True,
            use_rel_pos=True,
            rel_pos_zero_init=True,
            window_size=window_size,
            global_attn_indexes=global_attn_indexes,
        )
        # RDLNet reads patch embeddings, not SAM mask-decoder features
        del self.encoder.neck

    def _pos_embed(self, x: Tensor) -> Tensor:
        """Interpolate absolute positional embedding if H,W differ from constructor img_size."""
        if self.encoder.pos_embed is None:
            return x
        _, h, w, _ = x.shape
        ph, pw = self.encoder.pos_embed.shape[1], self.encoder.pos_embed.shape[2]
        if h == ph and w == pw:
            return x + self.encoder.pos_embed
        pe = self.encoder.pos_embed.permute(0, 3, 1, 2)
        pe = F.interpolate(pe, size=(h, w), mode="bicubic", align_corners=False)
        pe = pe.permute(0, 2, 3, 1)
        return x + pe

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """
        Args:
            x: [B, 3, H, W] in same convention as training (optionally SAM-normalized upstream).
        Returns:
            last: [B, H*W, C] final patch tokens (after all blocks, no SAM neck).
            intermediates: three tensors at layers (0, l/2-1, l-1), each [B, H*W, C].
        """
        x = self.encoder.patch_embed(x)
        x = self._pos_embed(x)

        inter: List[Tensor] = []
        for i, blk in enumerate(self.encoder.blocks):
            x = blk(x)
            if i in self.cross_level_indices:
                b, h, w, c = x.shape
                inter.append(x.reshape(b, h * w, c))

        b, h, w, c = x.shape
        last = x.reshape(b, h * w, c)
        return last, inter


def build_backbone(cfg) -> nn.Module:
    """Factory: SAM Table-2 encoder when possible, else fallback ViT from `backbone.py`."""
    use_sam = getattr(cfg, "use_sam_image_encoder", True)
    if use_sam:
        try:
            return RDLNetSAMEncoder(
                img_size=cfg.img_size,
                patch_size=cfg.patch_size,
                embed_dim=cfg.backbone_dim,
                depth=cfg.backbone_depth,
                num_heads=cfg.backbone_heads,
                window_size=getattr(cfg, "sam_window_size", 14),
                global_attn_indexes=tuple(getattr(cfg, "sam_global_attn_indexes", (2, 8))),
            )
        except ImportError:
            pass
    from .backbone import LightSAMViT

    return LightSAMViT(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        embed_dim=cfg.backbone_dim,
        depth=cfg.backbone_depth,
        num_heads=cfg.backbone_heads,
    )

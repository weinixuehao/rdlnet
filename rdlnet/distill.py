"""
Light-SAM multiplex distillation (paper Sec. 3.1, Eq. 2–3).

Teacher: large SAM image encoder (supplementary Table 1: ViT-H style).
Student: small encoder (Table 2), **random init**, trained only via this loss until convergence,
then used as RDLNet backbone for downstream localization (supplementary: prompt/mask decoders
frozen during backbone distillation; here we distill **image encoder only**).

L_kl = τ² · KL( p_teacher || p_student ) on per-patch distributions over channel dim (after linear
projection of student features to teacher dim so softmax spaces match).

L_md = || F_S F_S^T − F_T F_T^T ||_F²  for aligned intermediate blocks (multiplex relation map).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .sam_backbone import ImageEncoderViT

# -----------------------------------------------------------------------------
# Forward helpers (ImageEncoderViT with or without neck — neck unused here)
# -----------------------------------------------------------------------------


def sam_encoder_block_outputs(encoder: nn.Module, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
    """
    Run SAM-style image encoder and collect [B,H,W,C] after each block.

    `encoder` must be `ImageEncoderViT` (neck may be absent).
    """
    x = encoder.patch_embed(x)
    if encoder.pos_embed is not None:
        _, h, w, _ = x.shape
        ph, pw = encoder.pos_embed.shape[1], encoder.pos_embed.shape[2]
        if h == ph and w == pw:
            x = x + encoder.pos_embed
        else:
            pe = encoder.pos_embed.permute(0, 3, 1, 2)
            pe = F.interpolate(pe, size=(h, w), mode="bicubic", align_corners=False)
            x = x + pe.permute(0, 2, 3, 1)

    block_outs: List[Tensor] = []
    for blk in encoder.blocks:
        x = blk(x)
        block_outs.append(x)
    return x, block_outs


def align_student_to_teacher_layers(num_student: int, num_teacher: int) -> List[int]:
    """Map each student block index to a teacher block index (endpoints aligned)."""
    if num_student <= 0 or num_teacher <= 0:
        return []
    if num_student == 1:
        return [num_teacher - 1]
    return [min(num_teacher - 1, i * (num_teacher - 1) // (num_student - 1)) for i in range(num_student)]


def multiplex_relation_loss(f_student: Tensor, f_teacher: Tensor) -> Tensor:
    """
    Eq. (2) second line: || F_S F_S^T − F_T F_T^T ||_F^2
    f_* : [B, N, D_*]
    """
    g_s = torch.bmm(f_student, f_student.transpose(1, 2))
    g_t = torch.bmm(f_teacher, f_teacher.transpose(1, 2))
    return (g_s - g_t).pow(2).mean()


def kl_logits_student_to_teacher(z_s: Tensor, z_t: Tensor, tau: float) -> Tensor:
    """
    Eq. (2) first line: τ² KL( p_s || p_t ) with p = softmax(z/τ) over last dim.
    Paper notation uses p_s as teacher and p_t as student; PyTorch kl_div matches KL(target || input)
    with input = log q_student, target = p_teacher.
    z_* : [B, N, D] with same D after projection.
    """
    log_p_s = F.log_softmax(z_s / tau, dim=-1)
    p_t = F.softmax(z_t / tau, dim=-1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (tau**2)


# -----------------------------------------------------------------------------
# Teacher loading (ViT-H / Table 1)
# -----------------------------------------------------------------------------


def build_teacher_image_encoder_vit_h() -> nn.Module:
    """SAM ViT-H image encoder (Table 1). Load SA-1B weights with `load_teacher_weights_from_sam_checkpoint`."""
    if ImageEncoderViT is None:
        raise ImportError("segment_anything ImageEncoderViT unavailable; check sam_backbone.")
    from functools import partial

    enc = ImageEncoderViT(
        depth=32,
        embed_dim=1280,
        img_size=1024,
        patch_size=16,
        mlp_ratio=4.0,
        out_chans=256,
        qkv_bias=True,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
        use_abs_pos=True,
        use_rel_pos=True,
        rel_pos_zero_init=True,
        window_size=14,
        global_attn_indexes=(7, 15, 23, 31),
        num_heads=16,
    )
    del enc.neck
    return enc


def load_teacher_weights_from_sam_checkpoint(encoder: nn.Module, checkpoint_path: str | Path, strict: bool = False) -> None:
    """
    Load `image_encoder.*` tensors from a full SAM checkpoint (e.g. sam_vit_h_4b8939.pth).
    If `encoder` has no neck, set strict=False to ignore neck keys.
    """
    path = Path(checkpoint_path)
    w = torch.load(path, map_location="cpu")
    if isinstance(w, dict) and "model" in w:
        w = w["model"]
    prefix = "image_encoder."
    sub = {k[len(prefix) :]: v for k, v in w.items() if k.startswith(prefix)}
    missing, unexpected = encoder.load_state_dict(sub, strict=strict)
    if missing and strict:
        raise RuntimeError(f"Missing keys: {missing}")


# -----------------------------------------------------------------------------
# Pixel normalization (same as SAM)
# -----------------------------------------------------------------------------


def sam_normalize_images(images: Tensor) -> Tensor:
    """images in [0,255] float or uint8; if max<=1, scale to 0–255."""
    x = images.float()
    if x.max() <= 1.0 + 1e-3:
        x = x * 255.0
    mean = images.new_tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
    std = images.new_tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
    return (x - mean) / std


# -----------------------------------------------------------------------------
# Main loss module
# -----------------------------------------------------------------------------


@dataclass
class DistillConfig:
    temperature: float = 4.0
    weight_kl: float = 1.0
    weight_md: float = 1.0
    teacher_checkpoint: Optional[str] = None


class LightSAMMultiplexDistillation(nn.Module):
    """
    Multiplex distillation between a frozen teacher encoder and trainable student encoder.

    Student should match Table 2 (e.g. RDLNetSAMEncoder.encoder). Initialize student randomly;
    optionally load teacher from `sam_vit_h_4b8939.pth` via `load_teacher_weights_from_sam_checkpoint`.
    """

    def __init__(
        self,
        teacher_encoder: nn.Module,
        student_encoder: nn.Module,
        d_teacher: int = 1280,
        d_student: int = 384,
        cfg: Optional[DistillConfig] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or DistillConfig()
        self.teacher = teacher_encoder
        self.student = student_encoder
        self.d_teacher = d_teacher
        self.d_student = d_student

        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

        # Map student channels to teacher channels for KL on shared softmax space (paper does not fix D mismatch).
        self.kl_proj = nn.Linear(d_student, d_teacher)

        n_t = len(self.teacher.blocks)
        n_s = len(self.student.blocks)
        self._teacher_idx_per_student = align_student_to_teacher_layers(n_s, n_t)

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            images: [B, 3, H, W], H/W multiples of 16 (typically 1024). SAM-normalized inside.
        Returns:
            dict with 'loss', 'loss_kl', 'loss_md'
        """
        x = sam_normalize_images(images)

        with torch.no_grad():
            final_t, blocks_t = sam_encoder_block_outputs(self.teacher, x)

        final_s, blocks_s = sam_encoder_block_outputs(self.student, x)

        b, h, w, _ = final_t.shape
        z_t = final_t.reshape(b, h * w, self.d_teacher)
        z_s = self.kl_proj(final_s.reshape(b, h * w, self.d_student))
        loss_kl = kl_logits_student_to_teacher(z_s, z_t, self.cfg.temperature)

        loss_md = final_s.new_tensor(0.0)
        for ls, lt in enumerate(self._teacher_idx_per_student):
            fs = blocks_s[ls].reshape(b, h * w, self.d_student)
            ft = blocks_t[lt].reshape(b, h * w, self.d_teacher)
            # Relation map in respective spaces (paper Eq. 2); compare geometry at aligned depths.
            loss_md = loss_md + multiplex_relation_loss(fs, ft)

        loss_md = loss_md / len(self._teacher_idx_per_student)

        loss = self.cfg.weight_kl * loss_kl + self.cfg.weight_md * loss_md
        return {
            "loss": loss,
            "loss_kl": loss_kl.detach(),
            "loss_md": loss_md.detach(),
        }


def distill_trainable_state_dict(distill_mod: LightSAMMultiplexDistillation, meta: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Checkpoint for stage 1: student encoder + KL head (teacher not saved)."""
    out: Dict[str, object] = {
        "student_encoder": distill_mod.student.state_dict(),
        "kl_proj": distill_mod.kl_proj.state_dict(),
    }
    if meta is not None:
        out["meta"] = meta
    return out


def load_distill_trainable_state_dict(distill_mod: LightSAMMultiplexDistillation, state: Dict[str, object]) -> None:
    distill_mod.student.load_state_dict(state["student_encoder"])
    distill_mod.kl_proj.load_state_dict(state["kl_proj"])


def load_distilled_student_into_rdlnet(rdlnet: nn.Module, student_encoder: nn.Module) -> None:
    """
    After distillation, copy weights into `RDLNet.backbone` when it is `RDLNetSAMEncoder`
    (`backbone.encoder` matches `student_encoder`).
    """
    if not hasattr(rdlnet, "backbone") or not hasattr(rdlnet.backbone, "encoder"):
        raise TypeError("Expected RDLNet with sam_backbone.RDLNetSAMEncoder")
    rdlnet.backbone.encoder.load_state_dict(student_encoder.state_dict())


def load_student_encoder_into_rdlnet_from_checkpoint(rdlnet: nn.Module, checkpoint_path: str | Path) -> None:
    """Load only ``student_encoder`` from a stage-1 checkpoint file into the RDLNet SAM backbone."""
    ck = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ck, dict) and "student_encoder" in ck:
        sd = ck["student_encoder"]
    elif isinstance(ck, dict) and not any(k in ck for k in ("kl_proj", "meta", "model")):
        sd = ck
    else:
        raise KeyError("Expected a stage-1 checkpoint with key 'student_encoder', or a raw encoder state_dict.")
    if not hasattr(rdlnet, "backbone") or not hasattr(rdlnet.backbone, "encoder"):
        raise TypeError("Expected RDLNet with sam_backbone.RDLNetSAMEncoder")
    rdlnet.backbone.encoder.load_state_dict(sd)


def create_distillation_setup(
    student_encoder: nn.Module,
    teacher_checkpoint: Optional[str] = None,
    cfg: Optional[DistillConfig] = None,
) -> LightSAMMultiplexDistillation:
    """
    Build teacher (ViT-H), load pretrained if path given, wrap student + KL projection.
    `student_encoder` is typically `RDLNetSAMEncoder(...).encoder`.
    """
    teacher = build_teacher_image_encoder_vit_h()
    if teacher_checkpoint:
        load_teacher_weights_from_sam_checkpoint(teacher, teacher_checkpoint, strict=False)
    d_t = teacher.blocks[0].attn.qkv.in_features
    d_s = student_encoder.blocks[0].attn.qkv.in_features
    return LightSAMMultiplexDistillation(teacher, student_encoder, d_teacher=d_t, d_student=d_s, cfg=cfg)

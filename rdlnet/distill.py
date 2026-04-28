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

from .sam_backbone import ImageEncoderViT, MaskDecoder, PromptEncoder, TwoWayTransformer

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


# -----------------------------------------------------------------------------
# Decoder-output KD variant (box prompts, frozen prompt/mask decoders)
# -----------------------------------------------------------------------------


def build_sam_prompt_encoder(*, img_size: int = 1024, patch_size: int = 16, prompt_embed_dim: int = 256) -> nn.Module:
    if PromptEncoder is None:
        raise ImportError("segment_anything PromptEncoder unavailable; check sam_backbone.")
    image_embedding_size = img_size // patch_size
    return PromptEncoder(
        embed_dim=prompt_embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(img_size, img_size),
        mask_in_chans=16,
    )


def build_sam_mask_decoder(*, prompt_embed_dim: int = 256) -> nn.Module:
    if MaskDecoder is None or TwoWayTransformer is None:
        raise ImportError("segment_anything MaskDecoder/TwoWayTransformer unavailable; check sam_backbone.")
    return MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=prompt_embed_dim,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=prompt_embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )


def build_sam_vit_for_rdlnet_cfg(cfg) -> nn.Module:
    """
    Student ``ImageEncoderViT`` **with** SAM neck (needed for COCO box-prompt + mask-decoder KD).

    Width/depth/window come from :class:`rdlnet.model.RDLNetConfig` after
    :func:`rdlnet.model.apply_lite_preset` so the checkpoint matches stage-2 ``RDLNet.backbone``.
    """
    if ImageEncoderViT is None:
        raise ImportError("segment_anything ImageEncoderViT unavailable; check sam_backbone.")
    from functools import partial

    return ImageEncoderViT(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        in_chans=3,
        embed_dim=cfg.backbone_dim,
        depth=cfg.backbone_depth,
        num_heads=cfg.backbone_heads,
        mlp_ratio=4.0,
        out_chans=256,
        qkv_bias=True,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
        use_abs_pos=True,
        use_rel_pos=True,
        rel_pos_zero_init=True,
        window_size=cfg.sam_window_size,
        global_attn_indexes=tuple(cfg.sam_global_attn_indexes),
    )


def build_image_encoder_student_table2(*, img_size: int = 1024) -> nn.Module:
    """
    Student image encoder (supplementary Table 2) **with SAM neck** — same as
    ``build_sam_vit_for_rdlnet_cfg(apply_lite_preset(RDLNetConfig, 40))``.
    """
    from .model import RDLNetConfig, apply_lite_preset

    cfg = RDLNetConfig(img_size=img_size, use_sam_pixel_norm=True)
    apply_lite_preset(cfg, 40)
    return build_sam_vit_for_rdlnet_cfg(cfg)


def build_teacher_image_encoder_vit_h_with_neck() -> nn.Module:
    """Teacher image encoder ViT-H with SAM neck kept (required for mask decoder KD)."""
    if ImageEncoderViT is None:
        raise ImportError("segment_anything ImageEncoderViT unavailable; check sam_backbone.")
    from functools import partial

    return ImageEncoderViT(
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


def load_sam_submodules_from_checkpoint(
    *,
    teacher_image_encoder: nn.Module,
    teacher_prompt_encoder: nn.Module,
    teacher_mask_decoder: nn.Module,
    student_prompt_encoder: nn.Module,
    student_mask_decoder: nn.Module,
    checkpoint_path: str | Path,
    strict: bool = False,
) -> None:
    """
    Load `image_encoder.*`, `prompt_encoder.*`, `mask_decoder.*` from a SAM checkpoint.

    Student prompt/mask modules are typically initialized from the same weights and frozen.
    Student image encoder is intentionally NOT loaded (random init) for stage-1 distillation.
    """
    path = Path(checkpoint_path)
    w = torch.load(path, map_location="cpu")
    if isinstance(w, dict) and "model" in w:
        w = w["model"]
    if not isinstance(w, dict):
        raise TypeError("Expected a SAM checkpoint state_dict (dict)")

    def _sub(prefix: str) -> Dict[str, Tensor]:
        p = prefix + "."
        return {k[len(p) :]: v for k, v in w.items() if k.startswith(p)}

    teacher_image_encoder.load_state_dict(_sub("image_encoder"), strict=strict)
    teacher_prompt_encoder.load_state_dict(_sub("prompt_encoder"), strict=strict)
    teacher_mask_decoder.load_state_dict(_sub("mask_decoder"), strict=strict)
    # student prompt/mask decoders share the frozen teacher weights
    student_prompt_encoder.load_state_dict(_sub("prompt_encoder"), strict=strict)
    student_mask_decoder.load_state_dict(_sub("mask_decoder"), strict=strict)


def kl_softmax_2class_from_binary_logits(z_s: Tensor, z_t: Tensor, tau: float) -> Tensor:
    """
    Paper-style KD on logits with temperature + softmax + τ² scaling, adapted to a single binary logit.

    Given per-pixel scalar logits z (foreground), construct 2-class logits [0, z] (background, foreground).
    This is equivalent to sigmoid-based Bernoulli distillation, but matches the softmax-KL form.
    """
    zs = (z_s / tau).reshape(-1)
    zt = (z_t / tau).reshape(-1)
    logits_s = torch.stack([torch.zeros_like(zs), zs], dim=-1)
    logits_t = torch.stack([torch.zeros_like(zt), zt], dim=-1)
    log_p_s = F.log_softmax(logits_s, dim=-1)
    p_t = F.softmax(logits_t, dim=-1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (tau**2)


class LightSAMMultiplexDistillationDecoderKD(nn.Module):
    """
    Stage-1 distillation matching the paper's "KD = KM + KL" layout:
    - KM (multiplex relation map): encoder block geometry alignment (same as `loss_md`).
    - KL (output KD): KL on mask-decoder `low_res_masks` logits using box prompts.

    Teacher: full SAM parts are frozen.
    Student: image encoder trainable; prompt encoder + mask decoder frozen.
    """

    def __init__(
        self,
        *,
        teacher_image_encoder: nn.Module,
        student_image_encoder: nn.Module,
        teacher_prompt_encoder: nn.Module,
        student_prompt_encoder: nn.Module,
        teacher_mask_decoder: nn.Module,
        student_mask_decoder: nn.Module,
        cfg: Optional[DistillConfig] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or DistillConfig()

        self.teacher_image_encoder = teacher_image_encoder
        self.teacher_prompt_encoder = teacher_prompt_encoder
        self.teacher_mask_decoder = teacher_mask_decoder

        self.student_image_encoder = student_image_encoder
        self.student_prompt_encoder = student_prompt_encoder
        self.student_mask_decoder = student_mask_decoder

        for m in (self.teacher_image_encoder, self.teacher_prompt_encoder, self.teacher_mask_decoder):
            for p in m.parameters():
                p.requires_grad_(False)
            m.eval()

        for m in (self.student_prompt_encoder, self.student_mask_decoder):
            for p in m.parameters():
                p.requires_grad_(False)
            m.eval()

        n_t = len(self.teacher_image_encoder.blocks)
        n_s = len(self.student_image_encoder.blocks)
        self._teacher_idx_per_student = align_student_to_teacher_layers(n_s, n_t)
        self.d_teacher = self.teacher_image_encoder.blocks[0].attn.qkv.in_features
        self.d_student = self.student_image_encoder.blocks[0].attn.qkv.in_features

    def forward(
        self,
        images: Tensor,
        *,
        boxes_xyxy: Optional[Tensor] = None,
        points_xy: Optional[Tensor] = None,
        point_labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Args:
            images: [B, 3, H, W] in 0–255 or 0–1; normalized to SAM stats inside.
            boxes_xyxy: optional [B, 4] in resized-image pixel coords (same frame as images).
            points_xy: optional [B, P, 2] points (x,y) in pixel coords (same frame as images).
            point_labels: optional [B, P] int labels (1=pos, 0=neg, -1=pad).
        """
        if (boxes_xyxy is None) == (points_xy is None):
            raise ValueError("Exactly one of boxes_xyxy or points_xy must be provided")
        if points_xy is not None and point_labels is None:
            raise ValueError("point_labels must be provided when using points_xy")

        x = sam_normalize_images(images)

        with torch.no_grad():
            # teacher: (1) token blocks for KM, (2) neck output for mask decoder KL
            _final_t_tok, blocks_t = sam_encoder_block_outputs(self.teacher_image_encoder, x)
            emb_t = self.teacher_image_encoder(x)
            if boxes_xyxy is not None:
                sparse_t, dense_t = self.teacher_prompt_encoder(points=None, boxes=boxes_xyxy, masks=None)
            else:
                sparse_t, dense_t = self.teacher_prompt_encoder(points=(points_xy, point_labels), boxes=None, masks=None)
            low_res_t, _iou_t = self.teacher_mask_decoder(
                image_embeddings=emb_t,
                image_pe=self.teacher_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_t,
                dense_prompt_embeddings=dense_t,
                multimask_output=True,
            )

        # student forward: prompt/mask are frozen but must remain in graph so grads reach image encoder
        _final_s_tok, blocks_s = sam_encoder_block_outputs(self.student_image_encoder, x)
        emb_s = self.student_image_encoder(x)
        if boxes_xyxy is not None:
            sparse_s, dense_s = self.student_prompt_encoder(points=None, boxes=boxes_xyxy, masks=None)
        else:
            sparse_s, dense_s = self.student_prompt_encoder(points=(points_xy, point_labels), boxes=None, masks=None)
        low_res_s, _iou_s = self.student_mask_decoder(
            image_embeddings=emb_s,
            image_pe=self.student_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_s,
            dense_prompt_embeddings=dense_s,
            multimask_output=True,
        )

        # KL on mask decoder low-res logits (paper-style τ² KL on softened distributions)
        loss_kl = kl_softmax_2class_from_binary_logits(low_res_s, low_res_t, self.cfg.temperature)

        # Multiplex relation map loss on aligned blocks (pre-neck token space)
        # Need consistent token grid size; use teacher token shape to determine N.
        b, h, w, _ = blocks_t[-1].shape
        n = h * w
        loss_md = low_res_s.new_tensor(0.0)
        for ls, lt in enumerate(self._teacher_idx_per_student):
            fs = blocks_s[ls].reshape(b, n, self.d_student)
            ft = blocks_t[lt].reshape(b, n, self.d_teacher)
            loss_md = loss_md + multiplex_relation_loss(fs, ft)
        loss_md = loss_md / len(self._teacher_idx_per_student)

        loss = self.cfg.weight_kl * loss_kl + self.cfg.weight_md * loss_md
        return {"loss": loss, "loss_kl": loss_kl.detach(), "loss_md": loss_md.detach()}

    @torch.no_grad()
    def predict_low_res_logits(
        self,
        images: Tensor,
        *,
        boxes_xyxy: Optional[Tensor] = None,
        points_xy: Optional[Tensor] = None,
        point_labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Return low-res mask logits for visualization/debug.

        Returns:
          - low_res_t: teacher low-res logits, shape [B, M, h, w]
          - low_res_s: student low-res logits, shape [B, M, h, w]
        """
        if (boxes_xyxy is None) == (points_xy is None):
            raise ValueError("Exactly one of boxes_xyxy or points_xy must be provided")
        if points_xy is not None and point_labels is None:
            raise ValueError("point_labels must be provided when using points_xy")

        x = sam_normalize_images(images)

        emb_t = self.teacher_image_encoder(x)
        if boxes_xyxy is not None:
            sparse_t, dense_t = self.teacher_prompt_encoder(points=None, boxes=boxes_xyxy, masks=None)
        else:
            sparse_t, dense_t = self.teacher_prompt_encoder(points=(points_xy, point_labels), boxes=None, masks=None)
        low_res_t, _iou_t = self.teacher_mask_decoder(
            image_embeddings=emb_t,
            image_pe=self.teacher_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_t,
            dense_prompt_embeddings=dense_t,
            multimask_output=True,
        )

        emb_s = self.student_image_encoder(x)
        if boxes_xyxy is not None:
            sparse_s, dense_s = self.student_prompt_encoder(points=None, boxes=boxes_xyxy, masks=None)
        else:
            sparse_s, dense_s = self.student_prompt_encoder(points=(points_xy, point_labels), boxes=None, masks=None)
        low_res_s, _iou_s = self.student_mask_decoder(
            image_embeddings=emb_s,
            image_pe=self.student_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_s,
            dense_prompt_embeddings=dense_s,
            multimask_output=True,
        )

        return {"low_res_t": low_res_t, "low_res_s": low_res_s}


def distill_trainable_state_dict(distill_mod: nn.Module, meta: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """
    Checkpoint for stage 1: trainable student parts only (teacher not saved).

    Supports both:
    - `LightSAMMultiplexDistillation` (encoder-only KL head): saves `student_encoder` + `kl_proj`
    - `LightSAMMultiplexDistillationDecoderKD` (decoder-output KD): saves `student_encoder` only
    """
    if hasattr(distill_mod, "student") and hasattr(distill_mod, "kl_proj"):
        out: Dict[str, object] = {
            "student_encoder": getattr(distill_mod, "student").state_dict(),
            "kl_proj": getattr(distill_mod, "kl_proj").state_dict(),
        }
    elif hasattr(distill_mod, "student_image_encoder"):
        out = {"student_encoder": getattr(distill_mod, "student_image_encoder").state_dict()}
    else:
        raise TypeError("Unsupported distillation module type for checkpointing")
    if meta is not None:
        out["meta"] = meta
    return out


def load_distill_trainable_state_dict(distill_mod: nn.Module, state: Dict[str, object]) -> None:
    if hasattr(distill_mod, "student") and hasattr(distill_mod, "kl_proj"):
        getattr(distill_mod, "student").load_state_dict(state["student_encoder"])
        getattr(distill_mod, "kl_proj").load_state_dict(state["kl_proj"])
        return
    if hasattr(distill_mod, "student_image_encoder"):
        getattr(distill_mod, "student_image_encoder").load_state_dict(state["student_encoder"])
        return
    raise TypeError("Unsupported distillation module type for loading")


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
    # Downstream `RDLNetSAMEncoder` deletes SAM neck; accept checkpoints that include neck weights.
    if isinstance(sd, dict) and any(k.startswith("neck.") for k in sd.keys()):
        sd = {k: v for k, v in sd.items() if not k.startswith("neck.")}
    rdlnet.backbone.encoder.load_state_dict(sd, strict=False)


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

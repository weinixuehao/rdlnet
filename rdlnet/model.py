"""RDLNet full model (ACM MM 2024). Hyper-parameters from supplementary Table 3."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .ms_deform_attn import MultiScaleDeformableAttention
from .sam_backbone import build_backbone


@dataclass
class RDLNetConfig:
    img_size: int = 1024
    patch_size: int = 16
    backbone_dim: int = 384
    backbone_depth: int = 12
    backbone_heads: int = 8
    # SAM ImageEncoderViT (supplementary Table 2); ignored if use_sam_image_encoder=False
    use_sam_image_encoder: bool = True
    sam_window_size: int = 14
    sam_global_attn_indexes: Tuple[int, ...] = (2, 8)
    # SAM-style ImageNet normalization (same as build_sam). Use True with real photos in 0–255 or 0–1.
    use_sam_pixel_norm: bool = False
    hidden_dim: int = 256
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    num_feature_levels: int = 4
    num_queries: int = 5
    num_classes: int = 3
    num_points: int = 9
    num_heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.1
    encoder_n_points: int = 4


def _init_deformable_attn(m: nn.Module) -> None:
    if isinstance(m, MultiScaleDeformableAttention):
        nn.init.constant_(m.sampling_offsets.weight, 0.0)
        thetas = torch.arange(m.n_heads, dtype=torch.float32) * (2.0 * math.pi / m.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(m.n_heads, 1, 1, 2)
        grid_init = grid_init.repeat(1, m.n_levels, m.n_points, 1)
        for i in range(m.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            m.sampling_offsets.bias.copy_(grid_init.reshape(-1))
        nn.init.constant_(m.attention_weights.weight, 0.0)
        nn.init.constant_(m.attention_weights.bias, 0.0)


class DeformableEncoderLayer(nn.Module):
    def __init__(self, cfg: RDLNetConfig) -> None:
        super().__init__()
        self.self_attn = MultiScaleDeformableAttention(
            cfg.hidden_dim, cfg.num_heads, cfg.num_feature_levels, cfg.encoder_n_points
        )
        self.norm1 = nn.LayerNorm(cfg.hidden_dim)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.linear1 = nn.Linear(cfg.hidden_dim, cfg.ffn_dim)
        self.linear2 = nn.Linear(cfg.ffn_dim, cfg.hidden_dim)
        self.norm2 = nn.LayerNorm(cfg.hidden_dim)
        self.dropout2 = nn.Dropout(cfg.dropout)
        self.act = nn.GELU()

    def forward(
        self,
        x: Tensor,
        ref_points: Tensor,
        spatial_shapes: List[Tuple[int, int]],
        value_mask: Optional[Tensor],
        pos: Tensor,
    ) -> Tensor:
        z = self.norm1(x)
        q = z + pos
        v = x + pos
        attn = self.self_attn(q, v, spatial_shapes, ref_points, value_mask)
        x = x + self.dropout1(attn)
        y = self.norm2(x)
        y = self.linear2(self.dropout2(self.act(self.linear1(y))))
        x = x + self.dropout2(y)
        return x


class MaskedDecoderLayer(nn.Module):
    """Cross-attention (with prior mask bias) before self-attention, per paper Fig.3."""

    def __init__(self, cfg: RDLNetConfig) -> None:
        super().__init__()
        d, h = cfg.hidden_dim, cfg.num_heads
        self.num_heads = h
        self.scale = (d // h) ** -0.5

        self.norm_cross = nn.LayerNorm(d)
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.dropout = nn.Dropout(cfg.dropout)

        self.norm_self = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, h, dropout=cfg.dropout, batch_first=True)

        self.norm_ffn = nn.LayerNorm(d)
        self.linear1 = nn.Linear(d, cfg.ffn_dim)
        self.linear2 = nn.Linear(cfg.ffn_dim, d)
        self.act = nn.GELU()

        self.mask_embed = nn.Linear(d, d)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        memory_hr: Tensor,
        prior_attn_bias: Optional[Tensor],
        query_pos: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        prior_attn_bias: [B, Nq, Nmem] added to attention logits (Eq. 7).
        memory_hr: first-scale slice of memory for mask branch logits [B, H*W, C].
        """
        b, nq, d = tgt.shape
        nm = memory.shape[1]

        x = self.norm_cross(tgt)
        q = self.q_proj(x + query_pos)
        k = self.k_proj(memory)
        v = self.v_proj(memory)

        qh = q.view(b, nq, self.num_heads, d // self.num_heads).transpose(1, 2)
        kh = k.view(b, nm, self.num_heads, d // self.num_heads).transpose(1, 2)
        vh = v.view(b, nm, self.num_heads, d // self.num_heads).transpose(1, 2)

        attn = (qh @ kh.transpose(-2, -1)) * self.scale
        if prior_attn_bias is not None:
            bias = prior_attn_bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn = attn + bias
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        o = (attn @ vh).transpose(1, 2).reshape(b, nq, d)
        tgt = tgt + self.dropout(self.out_proj(o))

        z = self.norm_self(tgt)
        sa, _ = self.self_attn(z + query_pos, z + query_pos, z)
        tgt = tgt + self.dropout(sa)

        y = self.norm_ffn(tgt)
        y = self.linear2(self.dropout(self.act(self.linear1(y))))
        tgt = tgt + self.dropout(y)

        mask_logits = torch.bmm(self.mask_embed(tgt), memory_hr.transpose(1, 2))
        return tgt, mask_logits


class RDLNet(nn.Module):
    def __init__(self, cfg: RDLNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_point_coords = cfg.num_points * 2

        self.backbone = build_backbone(cfg)

        c = cfg.backbone_dim
        h = cfg.hidden_dim
        self.cross_proj = nn.ModuleList([nn.Conv2d(c, h, kernel_size=1) for _ in range(3)])
        self.cross_fuse = nn.Conv2d(h * 3, h, kernel_size=1)

        self.prior_conv = nn.Conv2d(c, h, kernel_size=1)
        self.prior_phi = nn.Conv2d(c, h, kernel_size=3, padding=1)
        self.prior_psi = nn.Linear(h, cfg.num_queries)

        self.level_embed = nn.Parameter(torch.Tensor(cfg.num_feature_levels, h))
        nn.init.normal_(self.level_embed, std=0.02)

        self.enc_pos_head = nn.Linear(h, h)
        self.encoder_layers = nn.ModuleList([DeformableEncoderLayer(cfg) for _ in range(cfg.num_encoder_layers)])
        for layer in self.encoder_layers:
            _init_deformable_attn(layer.self_attn)

        self.query_embed = nn.Embedding(cfg.num_queries, h)
        self.dec_pos = nn.Embedding(cfg.num_queries, h)

        self.decoder_layers = nn.ModuleList([MaskedDecoderLayer(cfg) for _ in range(cfg.num_decoder_layers)])

        nc = cfg.num_classes + 1
        self.class_embed = nn.Linear(h, nc)
        # Sec. 3.3: hybrid semantic (tgt) + mask pooled context + queries (in tgt)
        self.point_head = nn.Linear(2 * h, self.num_point_coords)

        self.mask_pixel_proj = nn.Linear(h, h)

        self.input_proj = nn.ModuleList([nn.Conv2d(h, h, kernel_size=3, stride=2, padding=1) for _ in range(3)])

        # segment_anything/build_sam.py pixel stats (for optional use_sam_pixel_norm)
        self.register_buffer(
            "_pixel_mean", torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "_pixel_std", torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1), persistent=False
        )

    def _preprocess_pixels(self, images: Tensor) -> Tensor:
        if not self.cfg.use_sam_pixel_norm:
            return images
        x = images
        if x.max() <= 1.0 + 1e-3:
            x = x * 255.0
        return (x - self._pixel_mean) / self._pixel_std

    def _build_multiscale(self, feat: Tensor) -> Tuple[Tensor, List[Tuple[int, int]], Tensor]:
        """feat [B,h,H,W] -> concat tokens, spatial_shapes, level_start_index."""
        b, c, h0, w0 = feat.shape
        levels = [feat]
        x = feat
        for proj in self.input_proj:
            x = proj(x)
            levels.append(x)

        if len(levels) < self.cfg.num_feature_levels:
            raise RuntimeError("num_feature_levels exceeds built pyramid depth")
        levels = levels[: self.cfg.num_feature_levels]

        spatial_shapes = [(t.shape[2], t.shape[3]) for t in levels]
        out = []
        for i, t in enumerate(levels):
            _, _, hi, wi = t.shape
            flat = t.flatten(2).transpose(1, 2)
            out.append(flat + self.level_embed[i].view(1, 1, -1))
        src = torch.cat(out, dim=1)
        start = torch.cat([torch.tensor([0], device=feat.device), torch.tensor([s[0] * s[1] for s in spatial_shapes]).cumsum(0)[:-1]])
        return src, spatial_shapes, start

    def _encoder_reference_points(
        self, spatial_shapes: List[Tuple[int, int]], batch: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        """[B, N, n_levels, 2] image-normalized reference coords (same physical point for all levels)."""
        refs = []
        for hi, wi in spatial_shapes:
            y = torch.linspace(0.5 / hi, 1.0 - 0.5 / hi, hi, device=device, dtype=dtype)
            x = torch.linspace(0.5 / wi, 1.0 - 0.5 / wi, wi, device=device, dtype=dtype)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            ref = torch.stack([xx, yy], dim=-1).reshape(hi * wi, 2)
            refs.append(ref)
        ref_cat = torch.cat(refs, dim=0)
        ref_stacked = ref_cat.unsqueeze(1).expand(-1, self.cfg.num_feature_levels, -1)
        return ref_stacked.unsqueeze(0).expand(batch, -1, -1, -1)

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            images: [B, 3, H, W], typically 1024×1024.
        Returns:
            dict with pred_logits, pred_masks, pred_points, prior_mask_logits, aux_decoder_masks
        """
        cfg = self.cfg
        images = self._preprocess_pixels(images)
        b, _, gh, gw = images.shape
        gph, gpw = gh // cfg.patch_size, gw // cfg.patch_size

        _, inter = self.backbone(images)
        f1, fm, fl = inter

        def to_bchw(t: Tensor) -> Tensor:
            return t.transpose(1, 2).reshape(b, cfg.backbone_dim, gph, gpw)

        t1, tm, tl = to_bchw(f1), to_bchw(fm), to_bchw(fl)

        fused = self.cross_fuse(torch.cat([self.cross_proj[0](t1), self.cross_proj[1](tm), self.cross_proj[2](tl)], dim=1))

        prior_feat = F.relu(self.prior_conv(tl) + self.prior_phi(tl))
        prior_logits = torch.einsum("bcxy,qc->bqxy", prior_feat, self.prior_psi.weight)

        src, spatial_shapes, _ = self._build_multiscale(fused)
        ref_pts = self._encoder_reference_points(spatial_shapes, b, images.device, images.dtype)

        pos2d = self.enc_pos_head(src)
        value_mask = None

        x = src
        for layer in self.encoder_layers:
            x = layer(x, ref_pts, spatial_shapes, value_mask, pos2d)

        memory = x
        mem_hr_len = spatial_shapes[0][0] * spatial_shapes[0][1]
        mem_hr = memory[:, :mem_hr_len]
        pixel_emb = self.mask_pixel_proj(mem_hr).transpose(1, 2).reshape(
            b, cfg.hidden_dim, spatial_shapes[0][0], spatial_shapes[0][1]
        )

        q_emb = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        q_pos = self.dec_pos.weight.unsqueeze(0).expand(b, -1, -1)
        tgt = torch.zeros_like(q_emb)

        nm_full = memory.shape[1]
        scale = 1.0 / math.sqrt(cfg.hidden_dim)

        def bias_from_hr(logits_hw: Tensor) -> Tensor:
            pad = logits_hw.new_zeros(b, cfg.num_queries, nm_full)
            pad[:, :, :mem_hr_len] = logits_hw * scale
            return pad

        attn_bias = bias_from_hr(prior_logits.flatten(2))

        aux_masks: List[Tensor] = []
        for li, dec in enumerate(self.decoder_layers):
            if li > 0 and aux_masks:
                attn_bias = bias_from_hr(aux_masks[-1])
            tgt, mlog = dec(tgt, memory, mem_hr, attn_bias, q_pos)
            aux_masks.append(mlog)

        mask_logits = torch.einsum("bqc,bcxy->bqxy", tgt, pixel_emb)

        last_mlog = aux_masks[-1] if aux_masks else prior_logits.flatten(2)
        w = F.softmax(last_mlog, dim=-1)
        mask_ctx = torch.bmm(w, mem_hr)
        point_in = torch.cat([tgt, mask_ctx], dim=-1)

        return {
            "pred_logits": self.class_embed(tgt),
            "pred_masks": mask_logits,
            "pred_points": self.point_head(point_in).sigmoid(),
            "prior_mask_logits": prior_logits,
            "aux_decoder_mask_logits": torch.stack(aux_masks, dim=1) if aux_masks else prior_logits.flatten(2).unsqueeze(1),
        }

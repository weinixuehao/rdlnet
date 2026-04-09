# Multi-scale deformable attention (pure PyTorch, grid_sample path).
# Structurally aligned with Deformable DETR / Zhu et al. ICCV 2020.

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiScaleDeformableAttention(nn.Module):
    """Samples value features at learned offsets on multiple feature levels."""

    def __init__(self, d_model: int, n_heads: int, n_levels: int, n_points: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.dim_per_head = d_model // n_heads

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: Tensor,
        value_levels: Tensor,
        spatial_shapes: List[Tuple[int, int]],
        reference_points: Tensor,
        value_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            query: [B, Nq, C]
            value_levels: concatenated flat tokens [B, sum(HW_l), C]
            spatial_shapes: list of (H, W) per level
            reference_points: [B, Nq, n_levels, 2] in [0,1], normalized (x, y)
            value_mask: optional [B, sum(HW)] bool True=valid
        """
        b, nq, c = query.shape
        assert c == self.d_model
        if len(spatial_shapes) != self.n_levels:
            raise ValueError(f"spatial_shapes ({len(spatial_shapes)}) must match n_levels ({self.n_levels})")

        value = self.value_proj(value_levels)
        if value_mask is not None:
            value = value.masked_fill(~value_mask[..., None], 0.0)
        value = value.view(b, -1, self.n_heads, self.dim_per_head)

        sampling_offsets = self.sampling_offsets(query).view(
            b, nq, self.n_heads, self.n_levels, self.n_points, 2
        )
        attn_w = self.attention_weights(query).view(b, nq, self.n_heads, self.n_levels * self.n_points)
        attn_w = F.softmax(attn_w, dim=-1).view(b, nq, self.n_heads, self.n_levels, self.n_points)

        # Per level (height, width): normalize offsets like Deformable DETR — (W, H) order
        offset_normalizer = torch.tensor(
            [[float(w), float(h)] for h, w in spatial_shapes],
            device=query.device,
            dtype=query.dtype,
        ).view(1, 1, 1, self.n_levels, 1, 2)

        sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer

        out = self._aggregate(value, spatial_shapes, sampling_locations, attn_w)
        return self.output_proj(out)

    def _aggregate(
        self,
        value: Tensor,
        spatial_shapes: List[Tuple[int, int]],
        sampling_locations: Tensor,
        attention_weights: Tensor,
    ) -> Tensor:
        b, _, num_heads, hidden_dim = value.shape
        _, num_queries, num_heads_, num_levels, num_points, _ = sampling_locations.shape
        assert num_heads == num_heads_

        value_list = value.split([h * w for h, w in spatial_shapes], dim=1)
        sampling_grids = 2 * sampling_locations - 1
        sampling_value_list = []
        for level_id, (height, width) in enumerate(spatial_shapes):
            value_l = (
                value_list[level_id]
                .flatten(2)
                .transpose(1, 2)
                .reshape(b * num_heads, hidden_dim, height, width)
            )
            grid_l = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)
            sampled = F.grid_sample(
                value_l,
                grid_l,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampling_value_list.append(sampled)

        stacked = torch.stack(sampling_value_list, dim=-2).flatten(-2)
        aw = attention_weights.transpose(1, 2).reshape(
            b * num_heads, 1, num_queries, num_levels * num_points
        )
        out = (stacked * aw).sum(-1).view(b, num_heads * hidden_dim, num_queries)
        return out.transpose(1, 2).contiguous()



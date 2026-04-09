"""Light-SAM style ViT image encoder (Table 2: d_model=384, depth=12, heads=8, patch=16)."""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class LightSAMViT(nn.Module):
    """
    ViT patch encoder aligned with supplementary Table 2 (student).
    Returns hidden states after blocks 0, depth//2-1, depth-1 (1-based: 1, l/2, l).
    """

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.grid_size = img_size // patch_size
        num_patches = self.grid_size * self.grid_size

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # layers to read for cross-level fusion (1-based indices 1, l/2, l)
        self.cross_level_indices = (0, depth // 2 - 1, depth - 1)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def interpolate_pos_embed(self, h: int, w: int) -> torch.Tensor:
        if h == self.grid_size and w == self.grid_size:
            return self.pos_embed
        pe = self.pos_embed.reshape(1, self.grid_size, self.grid_size, -1).permute(0, 3, 1, 2)
        pe = torch.nn.functional.interpolate(pe, size=(h, w), mode="bicubic", align_corners=False)
        return pe.permute(0, 2, 3, 1).reshape(1, h * w, -1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: [B, 3, H, W] with H,W multiples of patch_size (default 1024).
        Returns:
            last_normed: [B, N, C] final tokens
            intermediates: three tensors from blocks at first, middle, last (each [B, N, C]).
        """
        b, _, h, w = x.shape
        gh, gw = h // self.patch_size, w // self.patch_size
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.interpolate_pos_embed(gh, gw)
        x = self.pos_drop(x)

        inter: List[torch.Tensor] = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.cross_level_indices:
                inter.append(x)

        x = self.norm(x)
        return x, inter

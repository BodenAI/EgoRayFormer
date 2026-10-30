"""Generic attention / FFN / positional-encoding building blocks."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


class MultiheadSelfAttention(nn.Module):
    """Self-attention with an optional additive bias [1|B, H, N, N] and/or boolean mask [N, N]."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(
        self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, N, d]
        mask = None
        if attn_bias is not None:
            mask = attn_bias.to(q.dtype)
            if attn_mask is not None:
                mask = mask.masked_fill(~attn_mask, float("-inf"))
        elif attn_mask is not None:
            mask = attn_mask
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)
        return self.proj(out.transpose(1, 2).reshape(B, N, C))


class MultiheadCrossAttention(nn.Module):
    """Standard cross-attention; ``with_out_proj=False`` returns concatenated heads (EgoPoseFormer v2 eq. 7)."""

    def __init__(self, dim: int, num_heads: int, kv_dim: Optional[int] = None, with_out_proj: bool = True, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        kv_dim = kv_dim or dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(kv_dim, dim)
        self.v_proj = nn.Linear(kv_dim, dim)
        self.out_proj = nn.Linear(dim, dim) if with_out_proj else nn.Identity()
        self.dropout = dropout

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, Nq, C = q.shape
        Nk = k.shape[1]
        qh = self.q_proj(q).reshape(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        kh = self.k_proj(k).reshape(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        vh = self.v_proj(v).reshape(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0)
        return self.out_proj(out.transpose(1, 2).reshape(B, Nq, C))


def sine_pos_embed_2d(h: int, w: int, dim: int, temperature: float = 10000.0, device=None) -> torch.Tensor:
    """DETR-style 2D sine positional encoding -> [h*w, dim]."""
    assert dim % 4 == 0
    d = dim // 4
    ys = torch.arange(h, device=device, dtype=torch.float32)[:, None].expand(h, w)
    xs = torch.arange(w, device=device, dtype=torch.float32)[None, :].expand(h, w)
    omega = temperature ** (torch.arange(d, device=device, dtype=torch.float32) / d)
    px = xs[..., None] / omega
    py = ys[..., None] / omega
    pe = torch.cat([px.sin(), px.cos(), py.sin(), py.cos()], dim=-1)  # [h, w, dim]
    return pe.reshape(h * w, dim)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

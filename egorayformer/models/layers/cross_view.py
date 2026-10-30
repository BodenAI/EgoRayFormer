"""Multi-view attention blocks.

* ``ConditionedMultiViewCrossAttention``: query tokens attend to each view's feature map with a
  learnable camera embedding added to the queries; views are fused by the output linear layer
  (EgoPoseFormer v2, eq. 7). Used by the Stage-1 proposal decoder.
* ``CrossViewBlock``: encoder-side stereo cross-attention (CroCo / DUSt3R style): tokens of one view
  attend to the tokens of the other view(s). Zero-initialised layer-scale keeps a pretrained
  backbone's features unchanged at initialisation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import FFN, MultiheadCrossAttention


class ConditionedMultiViewCrossAttention(nn.Module):
    def __init__(self, embed_dims: int, num_heads: int, num_views: int, feat_dims: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.num_views = num_views
        self.cam_embed = nn.Embedding(num_views, embed_dims)
        nn.init.normal_(self.cam_embed.weight, std=0.02)
        self.attn = MultiheadCrossAttention(embed_dims, num_heads, kv_dim=feat_dims or embed_dims, with_out_proj=False, dropout=dropout)
        self.fuse = nn.Linear(num_views * embed_dims, embed_dims)

    def forward(self, q: torch.Tensor, feats: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """q [B, N, C]; feats [B, V, L, Cf]; pos [B or 1, V or 1, L, Cf] -> [B, N, C]."""
        outs = []
        for v in range(self.num_views):
            qv = q + self.cam_embed.weight[v]
            kv = feats[:, v]
            pv = pos[:, v] if pos.shape[1] > 1 else pos[:, 0]
            outs.append(self.attn(qv, kv + pv, kv))
        return self.fuse(torch.cat(outs, dim=-1))


class CrossViewBlock(nn.Module):
    """``ray_bias`` (GTA / Geometry-Biased-Transformer style): the cross-view
    attention logits get a learned per-head bias ``-softplus(a_h) * angle(r_i, r_j)`` between the device-frame centre
    rays of the two tokens - a *relative* camera encoding (tokens looking in similar directions attend to each other)
    that needs no ray input channels; the ablation counterpart of the ``ray_pe`` input channels."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0, init_scale: float = 0.0,
                 ray_bias: bool = False):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = MultiheadCrossAttention(dim, num_heads, dropout=dropout)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = FFN(dim, int(dim * mlp_ratio), dropout)
        self.gamma_attn = nn.Parameter(torch.full((dim,), init_scale))
        self.gamma_ffn = nn.Parameter(torch.full((dim,), init_scale))
        self.ray_bias = ray_bias
        if ray_bias:
            self.ray_scale = nn.Parameter(torch.full((num_heads,), math.log(math.e - 1.0)))  # softplus -> 1 per radian

    def _bias(self, rays_q: torch.Tensor, rays_kv: torch.Tensor) -> torch.Tensor:
        """rays_q [B, L, 3], rays_kv [B, Lk, 3] unit rays -> additive logits [B, H, L, Lk]."""
        cos = torch.einsum("bld,bkd->blk", rays_q.float(), rays_kv.float()).clamp(-1.0, 1.0)
        ang = torch.acos(cos.clamp(-1.0 + 1e-6, 1.0 - 1e-6))                          # [B, L, Lk] rad
        return -(F.softplus(self.ray_scale)[None, :, None, None] * ang[:, None])

    def forward(self, x: torch.Tensor, rays: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x [B, V, L, C]: each view attends to the concatenation of the other views; rays [B, V, L, 3] (optional,
        device-frame unit centre rays of the tokens) enable the geometric attention bias."""
        B, V, L, C = x.shape
        outs = []
        for v in range(V):
            others = torch.cat([x[:, u] for u in range(V) if u != v], dim=1)
            kv = self.norm_kv(others)
            mask = None
            if self.ray_bias and rays is not None:
                r_others = torch.cat([rays[:, u] for u in range(V) if u != v], dim=1)
                mask = self._bias(rays[:, v], r_others).to(x.dtype)
            y = x[:, v] + self.gamma_attn * self.attn(self.norm_q(x[:, v]), kv, kv, attn_mask=mask)
            y = y + self.gamma_ffn * self.ffn(self.norm_ffn(y))
            outs.append(y)
        return torch.stack(outs, dim=1)

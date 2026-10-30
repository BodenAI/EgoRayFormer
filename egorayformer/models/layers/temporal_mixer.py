"""Causal Temporal Feature Mixer (TFM) - re-implementation of the TSR-Ego baseline module (Azam et al., arXiv
2607.09169, Sec. 3.2) used as a *pixel-aligned* temporal-fusion control for our trajectory-aligned sampling.

For every (view, pixel) the C-dim feature sequence over time goes through ``blocks`` residual blocks:
    F <- F + PWConv(GELU(DWConv_causal(LN_C(F))))
with a left-padded depthwise 1D convolution over time (kernel ``kernel``, dilation ``dilations[m]``), per-frame
channel LayerNorm (no statistics across time, so nothing leaks from the future) and a zero-initialised pointwise
convolution so that the whole module starts as the identity. Never mixes neighbouring pixels.

``forward_step`` streams one frame at a time with a per-block cache of the last (kernel-1)*dilation LayerNormed
inputs, and is numerically identical to the parallel ``forward``.
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CausalBlock(nn.Module):
    def __init__(self, dim: int, kernel: int, dilation: int):
        super().__init__()
        self.kernel, self.dilation = kernel, dilation
        self.norm = nn.LayerNorm(dim)
        self.dw = nn.Conv1d(dim, dim, kernel, groups=dim, bias=True)
        self.pw = nn.Conv1d(dim, dim, 1)
        nn.init.zeros_(self.pw.weight)
        nn.init.zeros_(self.pw.bias)

    @property
    def history(self) -> int:
        return (self.kernel - 1) * self.dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [Nb, T, C] -> [Nb, T, C]."""
        u = self.norm(x).transpose(1, 2)                                  # [Nb, C, T]
        u = F.pad(u, (self.history, 0))                                    # causal left padding (zeros after LN)
        y = F.conv1d(u, self.dw.weight, self.dw.bias, groups=u.shape[1], dilation=self.dilation)
        return x + self.pw(F.gelu(y)).transpose(1, 2)

    def forward_step(self, x_t: torch.Tensor, cache: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
        """x_t [Nb, C]; cache[key] holds the last ``history`` LayerNormed inputs [Nb, history, C] (zeros initially)."""
        u_t = self.norm(x_t)
        hist = cache.get(key)
        if hist is None:
            hist = x_t.new_zeros(x_t.shape[0], self.history, x_t.shape[1])
        seq = torch.cat([hist, u_t[:, None]], dim=1)                       # [Nb, history+1, C]
        taps = seq[:, self.history - self.dilation * torch.arange(self.kernel - 1, -1, -1, device=x_t.device)]  # [Nb, k, C] oldest->newest
        y = (taps * self.dw.weight[:, 0, :].t()[None]).sum(1) + self.dw.bias  # depthwise conv at the last position
        out = x_t + F.conv1d(F.gelu(y)[..., None], self.pw.weight, self.pw.bias)[..., 0]
        cache[key] = seq[:, 1:] if self.history > 0 else seq[:, :0]
        return out


class CausalTemporalFeatureMixer(nn.Module):
    def __init__(self, dim: int, blocks: int = 2, kernel: int = 3, dilations: Sequence[int] = (1, 2)):
        super().__init__()
        assert len(dilations) >= blocks
        self.blocks = nn.ModuleList([_CausalBlock(dim, kernel, int(dilations[m])) for m in range(blocks)])

    @property
    def receptive_field(self) -> int:
        return 1 + sum(b.history for b in self.blocks)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats [B, T, V, C, h, w] -> same shape; temporal-only mixing per (b, v, y, x)."""
        B, T, V, C, h, w = feats.shape
        x = feats.permute(0, 2, 4, 5, 1, 3).reshape(B * V * h * w, T, C)
        for blk in self.blocks:
            x = blk(x)
        return x.reshape(B, V, h, w, T, C).permute(0, 4, 1, 5, 2, 3).contiguous()

    def forward_step(self, feats_t: torch.Tensor, cache: Dict[str, torch.Tensor]) -> torch.Tensor:
        """feats_t [B, V, C, h, w] at one time step; ``cache`` persists across steps."""
        B, V, C, h, w = feats_t.shape
        x = feats_t.permute(0, 1, 3, 4, 2).reshape(B * V * h * w, C)
        for m, blk in enumerate(self.blocks):
            x = blk.forward_step(x, cache, f"tfm{m}")
        return x.reshape(B, V, h, w, C).permute(0, 1, 4, 2, 3).contiguous()

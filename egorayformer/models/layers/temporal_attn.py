"""Causal temporal attention with a KV cache (EgoPoseFormer v2 style).

Every token (joint / body / track) attends only to *its own* history within a window ``w``:
training runs all T frames in parallel with a causal band mask; inference appends to a per-layer
KV cache. ``window <= 1`` disables the module (single-frame model).

Positional encoding (``pos_mode``):
  * ``rope``      rotary embedding on q / k by absolute frame index (EgoPoseFormer v2),
  * ``age``       learned additive bias per head indexed by the token age t - tau in [0, w) (the "age embedding"
                  of TSR-Ego; relative, hence suited to sliding-window inference),
  * ``rope+age``  both.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotary_cos_sin(positions: torch.Tensor, dim: int, base: float = 10000.0):
    """positions [T] -> cos, sin [T, dim] (pairs interleaved as (even, odd))."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=positions.device, dtype=torch.float32) / dim))
    freqs = positions.float()[:, None] * inv_freq[None, :]  # [T, dim/2]
    emb = torch.repeat_interleave(freqs, 2, dim=-1)  # [T, dim]
    return emb.cos(), emb.sin()


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [B, H, T, d]; cos/sin [T, d]."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return x * cos.to(x.dtype) + rot * sin.to(x.dtype)


class CausalTemporalAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, window: int, dropout: float = 0.0, pos_mode: str = "rope"):
        super().__init__()
        assert dim % num_heads == 0
        assert pos_mode in ("rope", "age", "rope+age")
        self.dim, self.num_heads, self.head_dim = dim, num_heads, dim // num_heads
        self.window = int(window)
        self.pos_mode = pos_mode
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout
        self.age_bias = nn.Embedding(max(self.window, 1), num_heads) if "age" in pos_mode else None
        if self.age_bias is not None:
            nn.init.zeros_(self.age_bias.weight)

    @property
    def enabled(self) -> bool:
        return self.window > 1

    @property
    def use_rope(self) -> bool:
        return "rope" in self.pos_mode

    def _band_mask(self, T: int, device) -> torch.Tensor:
        i = torch.arange(T, device=device)[:, None]
        j = torch.arange(T, device=device)[None, :]
        return (j <= i) & (i - j < self.window)  # True = attend

    def _mask_with_bias(self, q_pos: torch.Tensor, k_pos: torch.Tensor, allowed: torch.Tensor, dtype) -> torch.Tensor:
        """Additive float mask [H, Tq, Tk]: -inf where not allowed, + age bias otherwise."""
        age = (q_pos[:, None] - k_pos[None, :]).clamp(0, self.window - 1)      # [Tq, Tk]
        bias = self.age_bias(age).permute(2, 0, 1).to(dtype)                    # [H, Tq, Tk]
        return bias.masked_fill(~allowed[None], float("-inf"))

    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x [B, T, N, C] -> [B, T, N, C] (parallel, causal)."""
        if not self.enabled:
            return x
        B, T, N, C = x.shape
        xb = x.permute(0, 2, 1, 3).reshape(B * N, T, C)
        qkv = self.qkv(xb).reshape(B * N, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [BN, H, T, d]
        pos = torch.arange(T, device=x.device) if positions is None else positions
        if self.use_rope:
            cos, sin = rotary_cos_sin(pos, self.head_dim)
            q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        allowed = self._band_mask(T, x.device)
        mask = self._mask_with_bias(pos, pos, allowed, q.dtype)[None] if self.age_bias is not None else allowed
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)
        out = self.proj(out.transpose(1, 2).reshape(B * N, T, C))
        return out.reshape(B, N, T, C).permute(0, 2, 1, 3)

    def forward_step(self, x_t: torch.Tensor, cache: Dict[str, torch.Tensor], step: int) -> torch.Tensor:
        """Streaming: x_t [B, N, C] at absolute time ``step``; cache holds rotated k / v (and their steps) of the window."""
        if not self.enabled:
            return x_t
        B, N, C = x_t.shape
        xb = x_t.reshape(B * N, 1, C)
        qkv = self.qkv(xb).reshape(B * N, 1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [BN, H, 1, d]
        pos_t = torch.tensor([step], device=x_t.device)
        if self.use_rope:
            cos, sin = rotary_cos_sin(pos_t, self.head_dim)
            q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        if "k" in cache:
            k = torch.cat([cache["k"], k], dim=2)[:, :, -self.window:]
            v = torch.cat([cache["v"], v], dim=2)[:, :, -self.window:]
            k_pos = torch.cat([cache["pos"], pos_t])[-self.window:]
        else:
            k_pos = pos_t
        cache["k"], cache["v"], cache["pos"] = k, v, k_pos
        mask = None
        if self.age_bias is not None:
            allowed = torch.ones(1, k.shape[2], dtype=torch.bool, device=x_t.device)
            mask = self._mask_with_bias(pos_t, k_pos, allowed, q.dtype)[None]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(out.transpose(1, 2).reshape(B * N, 1, C)).reshape(B, N, C)

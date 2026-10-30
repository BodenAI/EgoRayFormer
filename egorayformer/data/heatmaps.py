"""Gaussian joint heatmaps rendered on the fly from normalised 2D keypoints."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def render_gaussian_heatmaps(
    uv_norm: torch.Tensor,
    valid: torch.Tensor,
    size: Tuple[int, int] = (64, 64),
    sigma: float = 1.5,
) -> torch.Tensor:
    """uv_norm [..., J, 2] in [0, 1], valid [..., J] -> heatmaps [..., J, h, w] with peak value 1."""
    h, w = size
    dev = uv_norm.device
    xs = torch.arange(w, device=dev, dtype=torch.float32) + 0.5  # pixel centres
    ys = torch.arange(h, device=dev, dtype=torch.float32) + 0.5
    uv_norm = torch.nan_to_num(uv_norm.float(), nan=-10.0)     # missing joints (NaN) render as empty maps, not NaN
    cx = (uv_norm[..., 0] * w)[..., None, None]  # [..., J, 1, 1]
    cy = (uv_norm[..., 1] * h)[..., None, None]
    d2 = (xs[None, :] - cx) ** 2 + (ys[:, None] - cy) ** 2
    hm = torch.exp(-d2 / (2.0 * sigma * sigma))
    hm = hm * valid[..., None, None].to(hm.dtype)
    return hm


def local_softargmax(
    heatmaps: torch.Tensor, uv_norm: torch.Tensor, window: int = 7, temperature: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sub-pixel 2D estimate from a ``window`` x ``window`` (heatmap pixels) soft-argmax around ``uv_norm`` (Bulat et al.,
    BMVC 2021 "Subpixel Heatmap Regression").

    heatmaps [B, V, J, h, w] (each joint its own channel), uv_norm [B, V, J, 2] in [0, 1] ->
    uv_hat [B, V, J, 2] (normalised, clamped to [0, 1]) and confidence [B, V, J] = peak value of the window in [0, 1].
    The window is bilinearly sampled with ``grid_sample`` so the anchor need not lie on the pixel grid; samples outside
    the image read 0 (padding). Softmax temperature 0.1 on heatmaps with peak 1 keeps the estimate sharp.
    """
    B, V, J, h, w = heatmaps.shape
    dev, dt = heatmaps.device, heatmaps.dtype
    r = window // 2
    off = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
    oy, ox = torch.meshgrid(off, off, indexing="ij")                                   # [win, win] heatmap pixels
    offsets = torch.stack([ox, oy], -1).reshape(1, -1, 2)                              # [1, win*win, 2]
    scale = torch.tensor([1.0 / w, 1.0 / h], device=dev)
    uv = uv_norm.float().reshape(B * V * J, 1, 2)
    pts = uv + offsets * scale                                                         # [BVJ, win*win, 2] normalised
    hm = heatmaps.reshape(B * V * J, 1, h, w).float()
    grid = (pts * 2.0 - 1.0).reshape(B * V * J, 1, -1, 2)
    vals = F.grid_sample(hm, grid, mode="bilinear", padding_mode="zeros", align_corners=False).reshape(B * V * J, -1)
    inside = ((pts >= 0.0) & (pts <= 1.0)).all(-1)                                     # [BVJ, win*win]
    logits = (vals / temperature).masked_fill(~inside, -1e4)
    wts = torch.softmax(logits, dim=-1)
    uv_hat = (wts[..., None] * pts).sum(1).clamp(0.0, 1.0)                             # [BVJ, 2]
    conf = vals.amax(dim=-1).clamp(0.0, 1.0)
    return uv_hat.reshape(B, V, J, 2).to(dt), conf.reshape(B, V, J).to(dt)


def heatmap_softargmax(heatmaps: torch.Tensor, temperature: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
    """Global soft-argmax of every heatmap: [..., h, w] -> normalised uv [..., 2] in [0, 1] and peak value [...]
    (geometric proposal initialisation: heatmap peaks -> two-ray intersection -> P0)."""
    h, w = heatmaps.shape[-2:]
    flat = heatmaps.float().flatten(-2)                                     # [..., h*w]
    wts = torch.softmax(flat / temperature, dim=-1)
    ys = (torch.arange(h, device=heatmaps.device, dtype=torch.float32) + 0.5) / h
    xs = (torch.arange(w, device=heatmaps.device, dtype=torch.float32) + 0.5) / w
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([gx, gy], -1).reshape(-1, 2)                         # [h*w, 2]
    uv = wts @ grid                                                         # [..., 2]
    return uv.clamp(0.0, 1.0), flat.amax(dim=-1).clamp(0.0, 1.0)


def sample_heatmap_at(heatmaps: torch.Tensor, uv_norm: torch.Tensor) -> torch.Tensor:
    """Bilinearly read each joint's own heatmap at its 2D location.

    heatmaps [B, V, J, h, w], uv_norm [B, V, J, 2] in [0, 1] -> values [B, V, J]
    """
    B, V, J, h, w = heatmaps.shape
    hm = heatmaps.reshape(B * V * J, 1, h, w)
    grid = (uv_norm.reshape(B * V * J, 1, 1, 2) * 2.0 - 1.0).to(hm.dtype)
    vals = F.grid_sample(hm, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return vals.reshape(B, V, J)

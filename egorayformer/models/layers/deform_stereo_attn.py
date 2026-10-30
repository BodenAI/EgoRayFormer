"""Deformable Stereo Attention with a pure-PyTorch (grid_sample) sampling core.

Module structure and initialisation follow EgoPoseFormer v1 ``deform_attn.py`` / ``egoposeformer.py``
(Apache-2.0, Chenhongyi Yang; in turn from ViT-Adapter / Deformable-DETR). The sampling core
re-implements mmcv's ``multi_scale_deformable_attn_pytorch`` (Apache-2.0, OpenMMLab) so that no
compiled CUDA operator is needed and the model exports to ONNX (GridSample, opset >= 16).
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_


def ms_deform_attn_pytorch(
    value: torch.Tensor,
    spatial_shapes: Sequence[Tuple[int, int]],
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """value [B, L_in, H, C]; sampling_locations [B, Lq, H, Lv, P, 2] in [0, 1]; weights [B, Lq, H, Lv, P]."""
    bs, _, n_heads, c = value.shape
    _, n_q, _, n_lv, n_p, _ = sampling_locations.shape
    value_list = value.split([h * w for h, w in spatial_shapes], dim=1)
    grids = 2.0 * sampling_locations - 1.0
    sampled = []
    for lv, (h, w) in enumerate(spatial_shapes):
        v_l = value_list[lv].flatten(2).transpose(1, 2).reshape(bs * n_heads, c, h, w)
        g_l = grids[:, :, :, lv].transpose(1, 2).flatten(0, 1)  # [bs*H, Lq, P, 2]
        sampled.append(F.grid_sample(v_l, g_l, mode="bilinear", padding_mode="zeros", align_corners=False))
    aw = attention_weights.transpose(1, 2).reshape(bs * n_heads, 1, n_q, n_lv * n_p)
    out = (torch.stack(sampled, dim=-2).flatten(-2) * aw).sum(-1).view(bs, n_heads * c, n_q)
    return out.transpose(1, 2).contiguous()


class MSDeformAttn(nn.Module):
    def __init__(self, d_model: int = 128, n_levels: int = 1, n_heads: int = 4, n_points: int = 16):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_levels, self.n_heads, self.n_points = d_model, n_levels, n_heads, n_points
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2)
        grid_init = grid_init.repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        input_flatten: torch.Tensor,
        spatial_shapes: Sequence[Tuple[int, int]],
    ) -> torch.Tensor:
        """query [B, Lq, C]; reference_points [B, Lq, Lv, 2] in [0,1]; input_flatten [B, L_in, C]."""
        B, Lq, _ = query.shape
        L_in = input_flatten.shape[1]
        assert sum(h * w for h, w in spatial_shapes) == L_in
        value = self.value_proj(input_flatten).view(B, L_in, self.n_heads, self.d_model // self.n_heads)
        offsets = self.sampling_offsets(query).view(B, Lq, self.n_heads, self.n_levels, self.n_points, 2)
        weights = self.attention_weights(query).view(B, Lq, self.n_heads, self.n_levels * self.n_points)
        weights = F.softmax(weights, -1).view(B, Lq, self.n_heads, self.n_levels, self.n_points)
        normalizer = torch.tensor([[w, h] for h, w in spatial_shapes], dtype=query.dtype, device=query.device)  # [Lv, 2]
        locations = reference_points[:, :, None, :, None, :] + offsets / normalizer[None, None, None, :, None, :]
        out = ms_deform_attn_pytorch(value.float(), spatial_shapes, locations.float(), weights.float())
        return self.output_proj(out.to(query.dtype))


class DeformStereoAttn(nn.Module):
    """Per-view deformable attention around projected 2D anchors (EgoPoseFormer v1 ``DeformStereoAttn``)."""

    def __init__(self, embed_dims: int, num_heads: int, num_points: int = 16):
        super().__init__()
        self.attn = MSDeformAttn(d_model=embed_dims, n_levels=1, n_heads=num_heads, n_points=num_points)

    def forward(self, query: torch.Tensor, feat: torch.Tensor, anchors_2d: torch.Tensor) -> torch.Tensor:
        """query [B, J, C]; feat [B, C, h, w]; anchors_2d [B, J, 2] normalised -> [B, J, C]."""
        B, C, h, w = feat.shape
        kv = feat.flatten(2).transpose(1, 2)  # [B, hw, C]
        ref = anchors_2d.detach().reshape(B, -1, 1, 2)
        return self.attn(query, ref, kv, [(h, w)])

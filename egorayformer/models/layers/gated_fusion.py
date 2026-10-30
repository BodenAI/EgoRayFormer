"""Confidence-gated multi-view fusion of per-view sampled joint features.

EgoPoseFormer v1 fuses the two views with ``Linear(concat_v(mask_fov(f_v)))``. Following EgoRear's
uncertainty-aware masking we replace the hard FOV mask by a learned gate driven by the heatmap
confidence at the projected 2D location and the in-FOV flag. With ``use_conf=False`` and the
default initialisation the module reduces to the v1 behaviour.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ConfidenceGatedViewFusion(nn.Module):
    def __init__(self, embed_dims: int, num_views: int, use_conf: bool = True, return_gates: bool = False):
        super().__init__()
        self.num_views = num_views
        self.use_conf = use_conf
        self.return_gates = return_gates
        self.fuse = nn.Linear(num_views * embed_dims, embed_dims)
        # gate = sigmoid(w_conf * conf + w_fov * in_fov + b)
        self.gate_w = nn.Parameter(torch.tensor([4.0, 4.0]))
        self.gate_b = nn.Parameter(torch.tensor(-4.0))
        self.gate_embed = nn.Linear(num_views, embed_dims, bias=False)  # lets the fused token know the gates
        nn.init.zeros_(self.gate_embed.weight)

    def forward(self, view_feats: torch.Tensor, in_fov: torch.Tensor, conf: Optional[torch.Tensor] = None):
        """view_feats [B, V, J, C]; in_fov [B, V, J] bool; conf [B, V, J] in [0,1] or None -> [B, J, C]."""
        fov = in_fov.to(view_feats.dtype)
        if self.use_conf and conf is not None:
            c = conf.detach().clamp(0.0, 1.0).to(view_feats.dtype)
            gate = torch.sigmoid(self.gate_w[0] * c + self.gate_w[1] * fov + self.gate_b) * fov
        else:
            gate = fov
        gated = view_feats * gate[..., None]
        B, V, J, C = gated.shape
        fused = self.fuse(gated.permute(0, 2, 1, 3).reshape(B, J, V * C))
        if self.use_conf and conf is not None:
            fused = fused + self.gate_embed(gate.permute(0, 2, 1))
        return (fused, gate) if self.return_gates else fused

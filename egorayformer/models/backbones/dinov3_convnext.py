"""Tier B (efficient): DINOv3-distilled ConvNeXt (timm) + EfficientFPN -> stride-4 features."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from .base import BaseEncoder, create_timm_model
from .fpn import EfficientFPN

_FALLBACKS = {
    "tiny": "convnext_tiny.fb_in22k_ft_in1k",
    "small": "convnext_small.fb_in22k_ft_in1k",
    "base": "convnext_base.fb_in22k_ft_in1k",
    "large": "convnext_large.fb_in22k_ft_in1k",
}


class DINOv3ConvNeXtFPN(BaseEncoder):
    def __init__(
        self,
        size: str = "tiny",
        pretrained: bool = True,
        out_channels: int = 128,
        drop_path: float = 0.1,
        num_views: int = 2,
        model_name: Optional[str] = None,
    ):
        super().__init__()
        name = model_name or f"convnext_{size}.dinov3_lvd1689m"
        self.requested_name, self.requested_pretrained = name, pretrained
        self.body, self.loaded_name, self.loaded_pretrained = create_timm_model(
            name, pretrained, fallback=_FALLBACKS.get(size), features_only=True, out_indices=(0, 1, 2, 3),
            drop_path_rate=drop_path,
        )
        chans: List[int] = self.body.feature_info.channels()
        self.fpn = EfficientFPN(chans, out_channels)
        self.out_channels = out_channels
        self.out_stride = 4
        self.num_views = num_views

    def forward(self, img: torch.Tensor, geom: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        return self.fpn(self.body(img))

    def param_groups(self, lr, weight_decay, layer_decay: float = 1.0, lr_scale: float = 1.0):
        # stage-wise decay: stem/stage0 lowest, fpn full lr
        groups: Dict[str, Dict] = {}
        n_stages = 4

        def stage_of(name: str) -> int:
            if name.startswith("fpn"):
                return n_stages + 1
            if "stages_" in name or "stages." in name:
                tok = name.split("stages")[1].lstrip("_.").split(".")[0]
                return int(tok) + 1 if tok.isdigit() else 1
            return 0  # stem

        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            s = stage_of(n)
            scale = lr_scale * (layer_decay ** (n_stages + 1 - s))
            nd = p.ndim <= 1 or n.endswith(".bias")
            key = f"enc_s{s}_{'nd' if nd else 'd'}"
            g = groups.setdefault(key, {"params": [], "lr": lr * scale, "weight_decay": 0.0 if nd else weight_decay, "name": key})
            g["params"].append(p)
        return list(groups.values())

"""Tier A (fair baseline): torchvision ResNet-18 + EfficientFPN, identical to EgoPoseFormer v1.

Adapted from EgoPoseFormer v1 ``pose_estimation/models/backbones/resnet.py`` (Apache-2.0, Chenhongyi Yang).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torchvision

from .base import BaseEncoder
from .fpn import EfficientFPN

_RESNETS = {
    "resnet18": (torchvision.models.resnet18, [64, 128, 256, 512]),
    "resnet34": (torchvision.models.resnet34, [64, 128, 256, 512]),
    "resnet50": (torchvision.models.resnet50, [256, 512, 1024, 2048]),
}


class ResNet18FPN(BaseEncoder):
    def __init__(self, model_name: str = "resnet18", pretrained: bool = True, out_channels: int = 128, num_views: int = 2):
        super().__init__()
        ctor, chans = _RESNETS[model_name]
        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = ctor(weights=weights)
        layers = list(backbone.children())
        self.stem = nn.Sequential(*layers[:3])          # /2
        self.layer1 = nn.Sequential(*layers[3:5])       # /4
        self.layer2, self.layer3, self.layer4 = layers[5], layers[6], layers[7]  # /8 /16 /32
        self.fpn = EfficientFPN(chans, out_channels)
        self.out_channels = out_channels
        self.out_stride = 4
        self.num_views = num_views

    def forward(self, img: torch.Tensor, geom: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        x = self.stem(img)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return self.fpn([c2, c3, c4, c5])

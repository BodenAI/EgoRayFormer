"""EfficientFPN: top-down fusion of a 4-level pyramid into one stride-4 map.

Adapted from EgoPoseFormer v1 ``pose_estimation/models/backbones/resnet.py`` (Apache-2.0, Chenhongyi Yang).
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientFPN(nn.Module):
    def __init__(self, in_channels: Sequence[int], out_channels: int, with_relu: bool = True):
        super().__init__()
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        act = (lambda: nn.ReLU(inplace=False)) if with_relu else (lambda: nn.Identity())
        self.lateral_convs = nn.ModuleList([nn.Sequential(nn.Conv2d(c, out_channels, 1), act()) for c in self.in_channels])
        self.fuse_convs = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(out_channels * 2, out_channels, 1), act()) for _ in self.in_channels[1:]]
        )
        self.fpn_convs = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1), act()) for _ in self.in_channels[1:]]
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        assert len(inputs) == len(self.in_channels)
        laterals = [conv(x) for conv, x in zip(self.lateral_convs, inputs)]
        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode="bilinear", align_corners=True)
            laterals[i - 1] = self.fpn_convs[i - 1](self.fuse_convs[i - 1](torch.cat((laterals[i - 1], up), dim=1)))
        return laterals[0]

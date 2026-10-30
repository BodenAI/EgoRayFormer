"""Encoder registry: every encoder maps images [N, 3, H, W] -> features [N, C, H/4, W/4]."""

from __future__ import annotations

from typing import Dict

import torch.nn as nn

from .base import BaseEncoder
from .dinov3_convnext import DINOv3ConvNeXtFPN
from .r18_fpn import ResNet18FPN
from .stereo_fisheye_vit import StereoFisheyeViT

ENCODERS: Dict[str, type] = {
    "r18_fpn": ResNet18FPN,
    "dinov3_convnext": DINOv3ConvNeXtFPN,
    "stereo_fisheye_vit": StereoFisheyeViT,
}


def build_encoder(cfg: Dict) -> BaseEncoder:
    cfg = dict(cfg)
    kind = cfg.pop("type")
    if kind not in ENCODERS:
        raise KeyError(f"unknown encoder type {kind!r}; available: {list(ENCODERS)}")
    return ENCODERS[kind](**cfg)


__all__ = ["BaseEncoder", "ResNet18FPN", "DINOv3ConvNeXtFPN", "StereoFisheyeViT", "build_encoder", "ENCODERS"]

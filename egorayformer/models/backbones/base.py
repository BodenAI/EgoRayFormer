from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class BaseEncoder(nn.Module):
    """Common interface. Subclasses set ``out_channels`` / ``out_stride`` and implement ``forward``.

    ``needs_geometry`` tells the model which per-view geometry tensors to provide in ``geom``:
    'ray_map' -> [N, 3, H, W] device-frame unit rays per input pixel,
    'gnomonic' -> 'grid' [N, np, np, 2] and 'rays' [N, 3, np, np] for the virtual undistorted image.
    """

    out_channels: int = 128
    out_stride: int = 4
    needs_geometry: Tuple[str, ...] = ()
    num_views: int = 2

    def forward(self, img: torch.Tensor, geom: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        raise NotImplementedError

    def param_groups(self, lr: float, weight_decay: float, layer_decay: float = 1.0, lr_scale: float = 1.0) -> List[Dict]:
        """Default: a single group (no layer-wise decay)."""
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 or n.endswith(".bias") else decay).append(p)
        return [
            {"params": decay, "lr": lr * lr_scale, "weight_decay": weight_decay, "name": "encoder"},
            {"params": no_decay, "lr": lr * lr_scale, "weight_decay": 0.0, "name": "encoder_no_decay"},
        ]


def local_weight_file(name: str) -> Optional[str]:
    """Look for ``<EGORAYFORMER_TIMM_WEIGHTS_DIR>/<model_name>.safetensors`` (default ../weights/timm next to the package).

    Useful where the HuggingFace hub is not reachable: download the weights elsewhere and copy them here.
    """
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    default = os.path.normpath(os.path.join(here, "..", "..", "..", "..", "weights", "timm"))
    for d in [os.environ.get("EGORAYFORMER_TIMM_WEIGHTS_DIR"), default]:
        if not d:
            continue
        for ext in (".safetensors", ".pth", ".bin"):
            p = os.path.join(d, name + ext)
            if os.path.isfile(p):
                return p
    return None


def create_timm_model(name: str, pretrained: bool, fallback: Optional[str] = None, **kwargs):
    """timm.create_model that prefers a local weight file, then the hub, then a fallback model, then random init."""
    import timm

    local = local_weight_file(name) if pretrained else None
    try:
        if local:
            log.info("loading %s from local file %s", name, local)
            return timm.create_model(name, pretrained=True, pretrained_cfg_overlay=dict(file=local), **kwargs), name, True
        return timm.create_model(name, pretrained=pretrained, **kwargs), name, pretrained
    except Exception as e:  # noqa: BLE001
        if not pretrained:
            raise
        msg = f"could not load pretrained weights for {name!r} ({type(e).__name__}: {str(e)[:200]})."
        if fallback:
            warnings.warn(msg + f" Falling back to {fallback!r}. Accept the DINOv3 license on HF and set HF_TOKEN for the real weights.")
            try:
                return timm.create_model(fallback, pretrained=True, **kwargs), fallback, True
            except Exception as e2:  # noqa: BLE001
                warnings.warn(f"fallback {fallback!r} also failed ({type(e2).__name__}); using random init of {name!r}.")
        else:
            warnings.warn(msg + " Using random initialisation.")
        return timm.create_model(name, pretrained=False, **kwargs), name, False

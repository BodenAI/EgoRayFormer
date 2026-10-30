"""Tier C (main): StereoFisheyeViT.

DINOv3 ViT (timm, RoPE) with
  * gnomonic fisheye tokenization: the input is resampled into a "virtual" image whose 16x16 blocks
    are undistorted tangent-plane patches (FisheyeViT, Wang et al. CVPR 2024); the sampling grid is
    precomputed from the calibration by ``ScaramuzzaCamera.gnomonic_grid``,
  * ray-direction positional encoding: 3 extra input channels holding the device-frame unit ray
    of every (virtual) pixel, consumed by a zero-initialised extension of the patch-embedding conv,
  * stereo cross-view attention blocks interleaved with the last ``xview_layers`` ViT blocks
    (CroCo / DUSt3R style, zero-initialised layer scale),
  * a ViTDet-style simple feature pyramid: stride 16 -> 8 -> 4.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.cross_view import CrossViewBlock
from .base import BaseEncoder, create_timm_model

_FALLBACKS = {
    "vit_small_patch16_dinov3.lvd1689m": "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "vit_small_plus_patch16_dinov3.lvd1689m": "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "vit_base_patch16_dinov3.lvd1689m": "vit_base_patch16_224.augreg_in21k_ft_in1k",
    "vit_large_patch16_dinov3.lvd1689m": "vit_large_patch16_224.augreg_in21k_ft_in1k",
}


class StereoFisheyeViT(BaseEncoder):
    def __init__(
        self,
        model_name: str = "vit_base_patch16_dinov3.lvd1689m",
        pretrained: bool = True,
        img_size: int = 256,
        patch_size: int = 16,
        fisheye_patch: bool = True,
        ray_pe: bool = True,
        xview_layers: int = 2,
        out_channels: int = 128,
        drop_path: float = 0.1,
        num_views: int = 2,
        fallback_model: Optional[str] = None,
        feature_lut: bool = False,
        xview_ray_bias: bool = False,
    ):
        super().__init__()
        self.img_size, self.patch_size = img_size, patch_size
        self.fisheye_patch, self.ray_pe = fisheye_patch, ray_pe
        # Relative camera encoding in the cross-view attention (angle between token centre rays) - the
        # ablation counterpart of the absolute ray-direction input channels (ray_pe)
        self.xview_ray_bias = xview_ray_bias
        self.num_views = num_views
        self.out_channels, self.out_stride = out_channels, 4
        # With gnomonic tokens the feature map lives in the *virtual* (tangent-patch)
        # layout while the decoder samples it at real fisheye pixels; the intra-block misalignment reaches 1.1 (p90) /
        # 4.2 (max) stride-4 pixels on the UnrealEgo rig. ``feature_lut``
        # resamples the output into the real layout with the exact inverse map (``real_to_virtual_lut``).
        self.feature_lut = feature_lut and fisheye_patch
        needs = ["gnomonic"] if fisheye_patch else (["ray_map"] if (ray_pe or xview_ray_bias) else [])
        if self.feature_lut:
            needs.append("lut")
        self.needs_geometry = tuple(needs)

        self.requested_name, self.requested_pretrained = model_name, pretrained
        self.vit, self.loaded_name, self.loaded_pretrained = create_timm_model(
            model_name, pretrained, fallback=fallback_model or _FALLBACKS.get(model_name),
            img_size=img_size, num_classes=0, drop_path_rate=drop_path,
        )
        self.embed_dim: int = self.vit.embed_dim
        self.num_prefix = int(getattr(self.vit, "num_prefix_tokens", 1))
        self.grid_n = img_size // patch_size

        if ray_pe:
            self._extend_patch_embed(extra_channels=3)

        n_blocks = len(self.vit.blocks)
        self.xview_start = n_blocks - min(xview_layers, n_blocks) if xview_layers > 0 else n_blocks
        heads = getattr(self.vit.blocks[0].attn, "num_heads", 8)
        self.xview_blocks = nn.ModuleList([CrossViewBlock(self.embed_dim, heads, ray_bias=xview_ray_bias) for _ in range(n_blocks - self.xview_start)])

        # simple feature pyramid: stride 16 -> 8 -> 4
        mid = max(out_channels, 256)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(self.embed_dim, mid, 2, stride=2), nn.GroupNorm(32, mid), nn.GELU())
        self.up2 = nn.Sequential(nn.ConvTranspose2d(mid, out_channels, 2, stride=2), nn.GroupNorm(32, out_channels), nn.GELU())
        self.out_conv = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.GroupNorm(32, out_channels))

    # ------------------------------------------------------------------ helpers
    def _extend_patch_embed(self, extra_channels: int) -> None:
        self.vit.patch_embed.proj = RayExtendedPatchProj(self.vit.patch_embed.proj, extra_channels)

    def _embed(self, x: torch.Tensor):
        vit = self.vit
        x = vit.patch_embed(x)
        rope = None
        out = vit._pos_embed(x)
        if isinstance(out, tuple):  # timm Eva (RoPE models such as DINOv3)
            x, rope = out
        else:  # timm VisionTransformer
            x = out
            if hasattr(vit, "patch_drop"):
                x = vit.patch_drop(x)
            if hasattr(vit, "norm_pre"):
                x = vit.norm_pre(x)
        return x, rope

    @staticmethod
    def _run_block(blk: nn.Module, x: torch.Tensor, rope) -> torch.Tensor:
        return blk(x, rope=rope) if rope is not None else blk(x)

    # ------------------------------------------------------------------ forward
    def forward(self, img: torch.Tensor, geom: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """img [N, 3, H, W] with N = B*V (views contiguous per sample); returns [N, C, H/4, W/4]."""
        N, _, H, W = img.shape
        V = self.num_views
        assert N % V == 0, "batch must be a multiple of num_views"
        x = img
        if self.fisheye_patch:
            assert geom is not None and "grid" in geom, "StereoFisheyeViT(fisheye_patch=True) needs geom['grid']"
            x = F.grid_sample(x, geom["grid"].to(x.dtype), mode="bilinear", padding_mode="zeros", align_corners=False)
        rays = None
        if self.ray_pe or self.xview_ray_bias:
            rays = geom["rays"] if (geom is not None and "rays" in geom) else geom["ray_map"]
            if rays.shape[-2:] != x.shape[-2:]:
                rays = F.interpolate(rays, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if self.ray_pe:
            x = torch.cat([x, rays.to(x.dtype)], dim=1)
        tok_rays = None
        if self.xview_ray_bias and rays is not None:  # unit centre ray per token (mean ray of the block), [B, V, L, 3]
            n = x.shape[-1] // self.patch_size
            tr = F.normalize(F.adaptive_avg_pool2d(rays.float(), n), dim=1)
            tok_rays = tr.flatten(2).transpose(1, 2).reshape(N // V, V, n * n, 3)

        tokens, rope = self._embed(x)
        for i, blk in enumerate(self.vit.blocks):
            tokens = self._run_block(blk, tokens, rope)
            if i >= self.xview_start and len(self.xview_blocks) > 0:
                xv = tokens.reshape(N // V, V, tokens.shape[1], tokens.shape[2])
                if tok_rays is not None and tokens.shape[1] != tok_rays.shape[2]:  # prefix (cls / register) tokens
                    pad = tokens.shape[1] - tok_rays.shape[2]
                    tr_full = torch.cat([tok_rays.new_zeros(N // V, V, pad, 3), tok_rays], dim=2)
                else:
                    tr_full = tok_rays
                tokens = self.xview_blocks[i - self.xview_start](xv, rays=tr_full).reshape(N, tokens.shape[1], tokens.shape[2])
        tokens = self.vit.norm(tokens)
        patches = tokens[:, self.num_prefix:]
        n = int(math.sqrt(patches.shape[1]))
        fmap = patches.transpose(1, 2).reshape(N, self.embed_dim, n, n)
        out = self.out_conv(self.up2(self.up1(fmap)))
        if self.feature_lut:  # virtual layout -> real image layout (same stride-4 size), so real uv sample correctly
            assert geom is not None and "lut" in geom, "StereoFisheyeViT(feature_lut=True) needs geom['lut']"
            # the sampling coordinates stay float32: a bf16 LUT moves samples by up to 0.09 feature px (0.4 input px at
            # 384^2, stride 4); only the resampled features return to the encoder's dtype
            with torch.autocast(device_type=out.device.type, enabled=False):
                lut = geom["lut"].float()
                if lut.shape[1:3] != out.shape[-2:]:
                    lut = F.interpolate(lut.permute(0, 3, 1, 2), size=out.shape[-2:], mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
                out = F.grid_sample(out.float(), lut, mode="bilinear", padding_mode="zeros", align_corners=False).to(out.dtype)
        return out

    # ------------------------------------------------------------------ optimisation
    def param_groups(self, lr: float, weight_decay: float, layer_decay: float = 0.8, lr_scale: float = 1.0) -> List[Dict]:
        """Layer-wise decay (times ``lr_scale``) on the pretrained ViT only. The randomly initialised modules (cross-view
        blocks, pyramid) train at the full decoder lr, the zero-initialised ray-direction input channels at ``lr_scale``
        (inside layer 0 they would get lr_scale * layer_decay ** (n_blocks + 1), i.e. ~1e-5 of the decoder lr)."""
        n_blocks = len(self.vit.blocks)
        groups: Dict[str, Dict] = {}

        def group(name: str) -> Tuple[str, float]:
            if name.startswith("vit.patch_embed.proj.ray."):
                return "enc_rayin", lr_scale
            if name.startswith("vit.blocks."):
                lid = int(name.split(".")[2]) + 1
            elif name.startswith("vit.norm"):
                lid = n_blocks + 1
            elif name.startswith("vit."):
                lid = 0  # patch_embed, cls/reg tokens, pos embed
            else:
                return "enc_new", 1.0  # xview blocks, pyramid
            return f"enc_l{lid}", lr_scale * (layer_decay ** (n_blocks + 1 - lid))

        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            name, scale = group(n)
            nd = p.ndim <= 1 or n.endswith(".bias") or "token" in n or "pos_embed" in n or "gamma" in n
            key = f"{name}_{'nd' if nd else 'd'}"
            g = groups.setdefault(key, {"params": [], "lr": lr * scale, "weight_decay": 0.0 if nd else weight_decay, "name": key})
            g["params"].append(p)
        return list(groups.values())


class RayExtendedPatchProj(nn.Module):
    """Pretrained RGB patch projection plus a zero-initialised projection of the extra ray-direction input channels:
    identical to one conv over the concatenated input, but a separate parameter for the optimiser."""

    def __init__(self, proj: nn.Conv2d, extra_channels: int):
        super().__init__()
        self.rgb = proj
        self.ray = nn.Conv2d(extra_channels, proj.out_channels, proj.kernel_size, proj.stride, proj.padding, bias=False)
        nn.init.zeros_(self.ray.weight)
        self.in_channels = proj.in_channels + extra_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.rgb.in_channels
        return self.rgb(x[:, :c]) + self.ray(x[:, c:])

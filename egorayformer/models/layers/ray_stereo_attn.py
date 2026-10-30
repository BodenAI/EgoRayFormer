"""Ray-consistent 3D stereo sampling (P2 of the redesign).

Instead of predicting independent 2D sampling offsets per view (EgoPoseFormer v1 / TSR-Ego deformable stereo
attention), every joint query predicts K *3D* offsets expressed in the ray-local frame of its anchor (one
component along the viewing ray = depth hypotheses, two lateral components). The same 3D hypotheses
X_k = P + Delta_k are projected with the calibrated fisheye model into *both* views, so the left and right
samples of a point describe the same 3D location and the per-point cross-view fusion can directly measure
stereo agreement. The fused point features are aggregated with query-predicted attention weights.

The module also returns per-view aggregated features from which ``ResidualConfHead`` predicts a 2D residual and a
confidence for the differentiable two-ray intersection (see ``ray_geometry``).

Module structure / initialisation follow the deformable-attention conventions of EgoPoseFormer v1 (Apache-2.0,
Chenhongyi Yang; in turn Deformable-DETR): zero-initialised offset weights with a structured bias, zero-initialised
attention weights (uniform at start), pure-PyTorch ``grid_sample`` sampling core (no compiled operator, ONNX-friendly).
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from .ray_geometry import ray_local_frame

ProjectFn = Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]  # points [N, M, 3] -> uv [N, V, M, 2], fov [N, V, M]
# positive along-ray levels of the "jacobian" hypotheses in disparity steps; "layered" uses depth_frac x ref x
# (0.5, 1, 2) = the same pattern in units of half a depth_frac step
JACOBIAN_LEVELS = (1.0, 2.0, 4.0)


class RayStereoAttn(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        num_points: int = 16,
        num_views: int = 2,
        lateral_cm: Sequence[float] = (3.0, 6.0),
        depth_cm: float = 3.0,
        depth_scaled: bool = True,
        ref_distance_cm: float = 50.0,
        depth_hyp: str = "pm",
        depth_frac: float = 0.15,
        content_attn: bool = False,
        corr_channel: bool = False,
        mask_invalid_views: bool = False,
        along_scaled: bool = True,
    ):
        super().__init__()
        assert embed_dims % num_heads == 0
        assert depth_hyp in ("pm", "layered", "jacobian")
        # ``depth_hyp="jacobian"`` expresses the along-ray component of every hypothesis in
        # disparity *steps* instead of centimetres scaled linearly with the distance: the caller passes the length of one
        # step at each anchor (``depth_unit`` cm = target disparity step / disparity rate of the real fisheye projection,
        # clamped), so the hypotheses are evenly spaced in image evidence; the lateral components keep the distance scaling.
        # A hypothesis outside a view's image would be sampled at the clamped border pixel and
        # fused like real evidence (the FOV flag was only an extra input). With ``mask_invalid_views`` the samples of
        # such a view are replaced by a learnt missing-view vector before the fusion / aggregation and its left-right
        # correlation is zeroed, so the pixels of an invalid view cannot act as matching evidence.
        self.mask_invalid_views = mask_invalid_views
        self.missing_view = nn.Parameter(torch.zeros(embed_dims // num_heads)) if mask_invalid_views else None
        self.dim, self.H, self.K, self.V = embed_dims, num_heads, num_points, num_views
        self.hd = embed_dims // num_heads
        # Offsets are specified for a joint ``ref_distance_cm`` from the device and scale with the anchor distance:
        # the same *angular* footprint in the image (a 3 cm lateral offset is ~0.6 feature pixels at 1.5 m and 256^2
        # input) and a depth spread proportional to the stereo depth uncertainty (which grows ~ distance^2 / baseline).
        self.depth_scaled, self.ref_distance = depth_scaled, ref_distance_cm
        # ``along_scaled=False`` keeps the along-ray component in cm (the "fixed" step rule) while the lateral components keep the distance scaling of the linear / jacobian variants
        self.along_scaled = along_scaled
        # ``depth_hyp="layered"`` initialises the K hypotheses as depth *levels*
        # {0, +/-delta/2, +/-delta, +/-2 delta} x {centre, lateral ring} with delta = depth_frac * ref_distance (a real
        # search along the ray, including the anchor itself; see ``_layered_bias``) instead of the alternating +/-3 cm;
        # ``content_attn`` adds a query-point dot product to the query-predicted point logits (stereo matching by
        # content, ODTFormer / StereoDETR style); ``corr_channel`` appends the left-right cosine similarity of every
        # hypothesis to the per-point fusion input (a 1D correlation volume along the ray).
        self.depth_hyp, self.depth_frac = depth_hyp, depth_frac
        self.content_attn, self.corr_channel = content_attn, corr_channel
        self.offsets = nn.Linear(embed_dims, num_heads * num_points * 3)
        self.attn_w = nn.Linear(embed_dims, num_heads * num_points)
        self.attn_q = nn.Linear(embed_dims, embed_dims) if content_attn else None
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        n_pairs = num_views * (num_views - 1) // 2
        self.view_fuse = nn.Sequential(nn.Linear(num_views * (self.hd + 1) + (n_pairs if corr_channel else 0), self.hd), nn.GELU())
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        self._reset_parameters(lateral_cm, depth_cm)

    def _layered_bias(self, lateral_cm: Sequence[float], levels: Optional[Sequence[float]] = None) -> torch.Tensor:
        """[H, K, 3] ray-local offsets: the symmetric depth levels {0, +-l} (cm at the reference distance; disparity
        steps for "jacobian") x {centre, lateral ring}, ring rotated per head.

        The K points of every head are split over *all* levels - each +-l pair gets the same
        count, the anchor level the remainder - and the rings of -l mirror those of +l. The former layout cut a
        centre + 3-ring list per level at K points, which kept {-1, 0, +1, +2} of {0, +-1, +-2, +-4} (mean +0.5 step).
        Levels that do not fit (K < 2 x levels + 1) are dropped from the outside in."""
        H, K = self.H, self.K
        if levels is None:
            delta = self.depth_frac * self.ref_distance
            levels = [0.5 * delta, delta, 2.0 * delta]
        pos = sorted({abs(float(v)) for v in levels if abs(float(v)) > 0})[: max(0, (K - 1) // 2)]
        base, extra = divmod(K, 1 + 2 * len(pos))
        plan = [(0.0, base + extra, 0)] + [(s * p, base, i + 1) for i, p in enumerate(pos) for s in (1.0, -1.0)]
        radius = float(lateral_cm[0])
        bias = torch.zeros(H, K, 3)
        for h in range(H):
            pts = []
            for depth, count, li in plan:
                pts.append((depth, 0.0, 0.0))
                n_ring = count - 1
                for i in range(n_ring):
                    theta = (2.0 * math.pi * i / n_ring + math.pi * h / (n_ring * H)
                             + math.pi * li / (n_ring * max(1, len(pos))) + (math.pi if depth < 0 else 0.0))
                    pts.append((depth, radius * math.cos(theta), radius * math.sin(theta)))
            bias[h] = torch.tensor(pts)
        return bias

    def depth_levels(self) -> torch.Tensor:
        """Initial along-ray component of every hypothesis [H, K] (cm at the reference distance)."""
        return self.offsets.bias.detach().view(self.H, self.K, 3)[..., 0]

    def _reset_parameters(self, lateral_cm: Sequence[float], depth_cm: float) -> None:
        constant_(self.offsets.weight, 0.0)
        H, K = self.H, self.K
        if self.depth_hyp == "layered":
            bias = self._layered_bias(lateral_cm)
        elif self.depth_hyp == "jacobian":  # along-ray levels in disparity steps (0, +-1, +-2, +-4 steps)
            bias = self._layered_bias(lateral_cm, levels=JACOBIAN_LEVELS)
        else:
            bias = torch.zeros(H, K, 3)
            n_dir = max(1, K // len(lateral_cm))
            for h in range(H):
                for k in range(K):
                    ring, idx = divmod(k, n_dir)
                    theta = 2.0 * math.pi * idx / n_dir + math.pi * h / (n_dir * H)  # rotate the pattern per head
                    radius = float(lateral_cm[min(ring, len(lateral_cm) - 1)])
                    bias[h, k, 0] = depth_cm * (1.0 if k % 2 == 0 else -1.0)          # along the ray: +/- depth
                    bias[h, k, 1] = radius * math.cos(theta)                          # lateral e1
                    bias[h, k, 2] = radius * math.sin(theta)                          # lateral e2
        with torch.no_grad():
            self.offsets.bias.copy_(bias.reshape(-1))
        constant_(self.attn_w.weight, 0.0)
        constant_(self.attn_w.bias, 0.0)
        if self.attn_q is not None:
            xavier_uniform_(self.attn_q.weight)
            constant_(self.attn_q.bias, 0.0)
        xavier_uniform_(self.value_proj.weight)
        constant_(self.value_proj.bias, 0.0)
        xavier_uniform_(self.out_proj.weight)
        constant_(self.out_proj.bias, 0.0)

    # ------------------------------------------------------------------ pieces (also used by TrajectoryAttention)
    def offsets_local(self, query: torch.Tensor) -> torch.Tensor:
        """query [N, J, C] -> ray-local 3D offsets [N, J, H, K, 3] in cm."""
        N, J, _ = query.shape
        return self.offsets(query).view(N, J, self.H, self.K, 3)

    def scaled_offsets(self, delta_local: torch.Tensor, anchor: torch.Tensor, center: torch.Tensor,
                       depth_unit: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Ray-local offsets in cm at the anchor: delta_local [N, J, H, K, 3] -> [N, J, H, K, 3]. ``depth_unit`` [N, J]
        (cm per disparity step, ``depth_hyp="jacobian"``) replaces the distance scaling of the along-ray component."""
        d = delta_local.float()
        if self.depth_scaled:
            dist = (anchor.float() - center.float()).norm(dim=-1, keepdim=True)       # [N, J, 1]
            scale = (dist / self.ref_distance).clamp(0.5, 4.0)[:, :, None, None]     # [N, J, 1, 1, 1]
            d = d * scale if self.along_scaled else torch.cat([d[..., :1], d[..., 1:] * scale], dim=-1)
        if self.depth_hyp == "jacobian" and depth_unit is not None:
            d = torch.cat([delta_local[..., :1].float() * depth_unit.float()[:, :, None, None, None], d[..., 1:]], dim=-1)
        return d

    def to_world(self, delta_local: torch.Tensor, anchor: torch.Tensor, center: torch.Tensor,
                 depth_unit: Optional[torch.Tensor] = None) -> torch.Tensor:
        """delta_local [N, J, H, K, 3], anchor [N, J, 3], center [N, 1, 3] -> 3D sampling points [N, J, H, K, 3]."""
        anchor, center = anchor.float(), center.float()
        r, e1, e2 = ray_local_frame(anchor, center)
        d = self.scaled_offsets(delta_local, anchor, center, depth_unit)
        delta = d[..., 0:1] * r[:, :, None, None] + d[..., 1:2] * e1[:, :, None, None] + d[..., 2:3] * e2[:, :, None, None]
        return anchor[:, :, None, None] + delta

    def project_values(self, feats: torch.Tensor) -> torch.Tensor:
        """feats [N, V, C, h, w] -> value [N, V, H, hd, h, w] (float32)."""
        N, V, C, h, w = feats.shape
        v = self.value_proj(feats.flatten(3).transpose(2, 3))                  # [N, V, hw, C]
        return v.transpose(2, 3).reshape(N, V, self.H, self.hd, h, w).float()

    def sample(self, value: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        """value [N, V, H, hd, h, w]; uv [N, V, J, H, K, 2] in [0,1] -> sampled [N, V, J, K, H, hd]."""
        N, V, H, hd, h, w = value.shape
        J, K = uv.shape[2], uv.shape[4]
        grid = (2.0 * uv.float() - 1.0).permute(0, 1, 3, 2, 4, 5).reshape(N * V * H, J, K, 2)
        out = F.grid_sample(value.reshape(N * V * H, hd, h, w), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        return out.reshape(N, V, H, hd, J, K).permute(0, 1, 4, 5, 2, 3)

    def mask_samples(self, sampled: torch.Tensor, fov: torch.Tensor) -> torch.Tensor:
        """Samples of views where the hypothesis is outside the image -> the learnt missing-view vector
        (identity when ``mask_invalid_views`` is off). sampled [N, V, J, K, H, hd], fov [N, V, J, H, K] bool."""
        if self.missing_view is None:
            return sampled
        valid = fov.permute(0, 1, 2, 4, 3)[..., None]                            # [N, V, J, K, H, 1]
        return torch.where(valid, sampled, self.missing_view.to(sampled.dtype))

    def fuse_views(self, sampled: torch.Tensor, fov: torch.Tensor) -> torch.Tensor:
        """sampled [N, V, J, K, H, hd], fov [N, V, J, H, K] bool -> per-point fused feature [N, J, K, H, hd]."""
        valid = fov.permute(0, 1, 2, 4, 3)                                       # [N, V, J, K, H]
        sampled = self.mask_samples(sampled, fov)
        flags = valid.to(sampled.dtype)[..., None]                               # [N, V, J, K, H, 1]
        x = torch.cat([sampled, flags], dim=-1).permute(0, 2, 3, 4, 1, 5)       # [N, J, K, H, V, hd+1]
        x = x.flatten(-2)
        if self.corr_channel:  # cosine similarity of the views' samples of the *same* 3D hypothesis
            s = F.normalize(sampled.float(), dim=-1)
            corr = []
            for i in range(self.V):
                for j in range(i + 1, self.V):
                    c = (s[:, i] * s[:, j]).sum(-1)                              # [N, J, K, H]
                    if self.mask_invalid_views:                                  # no correlation with a missing view
                        c = c * (valid[:, i] & valid[:, j]).to(c.dtype)
                    corr.append(c)
            x = torch.cat([x, torch.stack(corr, dim=-1).to(x.dtype)], dim=-1)
        return self.view_fuse(x)

    def sample_points(self, value: torch.Tensor, points: torch.Tensor, project_fn: ProjectFn):
        """points [N, J, H, K, 3] -> fused [N, J, K, H, hd], sampled [N, V, J, K, H, hd] (invalid views masked), fov [N, V, J, H, K]."""
        N, J, H, K, _ = points.shape
        uv, fov = project_fn(points.reshape(N, J * H * K, 3))
        uv = uv.reshape(N, self.V, J, H, K, 2)
        fov = fov.reshape(N, self.V, J, H, K)
        sampled = self.mask_samples(self.sample(value, uv), fov)
        return self.fuse_views(sampled, fov), sampled, fov

    # ------------------------------------------------------------------ forward
    def forward(self, query: torch.Tensor, feats: torch.Tensor, anchor: torch.Tensor, center: torch.Tensor, project_fn: ProjectFn,
                depth_unit: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """query [N, J, C]; feats [N, V, C, h, w]; anchor [N, J, 3] (target frame, cm); center [N, 1, 3]; ``depth_unit``
        [N, J] cm per disparity step (``depth_hyp="jacobian"`` only).

        Returns ``out`` [N, J, C] (residual to add to the joint tokens), ``view_feat`` [N, V, J, C] (per-view
        weighted samples for the residual / confidence head), ``delta_local`` [N, J, H, K, 3] (shared with the
        trajectory sampler), ``value`` [N, V, H, hd, h, w], ``weights`` [N, J, H, K] and ``points`` [N, J, H, K, 3].
        """
        N, J, C = query.shape
        delta_local = self.offsets_local(query)
        points = self.to_world(delta_local, anchor.detach(), center, depth_unit)
        value = self.project_values(feats)
        fused, sampled, fov = self.sample_points(value, points, project_fn)     # [N, J, K, H, hd]
        logits = self.attn_w(query).view(N, J, self.H, self.K).float()           # [N, J, H, K]
        if self.attn_q is not None:  # content-based term: how well does the query match the fused point feature
            q = self.attn_q(query).view(N, J, self.H, 1, self.hd).float()
            logits = logits + (q * fused.float().permute(0, 1, 3, 2, 4)).sum(-1) / math.sqrt(self.hd)
        weights = torch.softmax(logits, dim=-1)                                  # [N, J, H, K]
        wk = weights.permute(0, 1, 3, 2)[..., None]                               # [N, J, K, H, 1]
        agg = (fused * wk).sum(2)                                                 # [N, J, H, hd]
        out = self.out_proj(agg.reshape(N, J, C).to(query.dtype))
        view_feat = (sampled * wk[:, None]).sum(3).reshape(N, self.V, J, C).to(query.dtype)  # [N, V, J, C]
        res = {"out": out, "view_feat": view_feat, "delta_local": delta_local, "value": value, "weights": weights, "points": points,
               "depth_unit": depth_unit}
        # expected along-ray offset of the attended hypotheses (cm, at the anchor's scale) and its spread: a soft
        # depth estimate from the matching weights, exposed for the gate / analysis
        depth = self.scaled_offsets(delta_local, anchor, center, depth_unit)[..., 0]   # [N, J, H, K]
        mean = (weights * depth).sum(-1)                                          # [N, J, H]
        var = (weights * (depth - mean[..., None]) ** 2).sum(-1)
        res["depth_expect"], res["depth_var"] = mean.mean(-1), var.mean(-1)       # [N, J]
        flat = depth.detach().flatten(2)                                          # range of the along-ray hypotheses
        res["hyp_min"], res["hyp_max"] = flat.amin(-1), flat.amax(-1)             # [N, J] cm
        return res

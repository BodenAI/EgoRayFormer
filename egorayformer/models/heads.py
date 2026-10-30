"""Prediction heads."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.skeleton import MAX_JOINTS, SKELETON_IDS, Skeleton, get_skeleton, mapping_matrix
from .layers.attention import MLP


class HeatmapHead(nn.Module):
    """stride-4 features -> per-joint 2D heatmaps at the same resolution."""

    def __init__(self, in_channels: int, num_joints: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1), nn.GroupNorm(32, hidden), nn.GELU(),
            nn.Conv2d(hidden, num_joints, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class OffsetHead(nn.Module):
    def __init__(self, embed_dims: int, num_layers: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dims)
        self.mlp = MLP(embed_dims, embed_dims, 3, num_layers=num_layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm(tokens))


class UncertaintyHead(nn.Module):
    """Per-joint 3D covariance via its Cholesky factor L (6 params: 3 log-diagonal + 3 off-diagonal).

    ``ray_aligned``: L is expressed in the ray-local frame (r, e1, e2) of the
    joint's anchor - first axis along the viewing ray, i.e. the stereo depth direction - so that the along-ray sigma
    is a single parameter (fed to the gate / P3 weights) and calibration can be analysed along vs. across the ray.
    The world-frame covariance is R L L^T R^T with R = [r e1 e2] (see ``world_cholesky``)."""

    def __init__(self, embed_dims: int, min_std_cm: float = 0.3, max_std_cm: float = 60.0, ray_aligned: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dims)
        self.mlp = MLP(embed_dims, embed_dims, 6, num_layers=2)
        self.min_std, self.max_std = min_std_cm, max_std_cm
        self.ray_aligned = ray_aligned
        nn.init.zeros_(self.mlp.net[-1].weight)
        with torch.no_grad():
            self.mlp.net[-1].bias.zero_()
            self.mlp.net[-1].bias[:3] = math.log(3.0)  # init std ~ 3 cm

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm(tokens))  # raw params [..., 6]

    def log_std(self, params: torch.Tensor) -> torch.Tensor:
        """raw [..., 6] -> mean log-std over the three axes [...] (cm), clamped to the head's range."""
        return params[..., :3].float().clamp(math.log(self.min_std), math.log(self.max_std)).mean(-1)

    def cholesky(self, params: torch.Tensor) -> torch.Tensor:
        """raw [..., 6] -> lower-triangular L [..., 3, 3] with positive diagonal (cm); always float32."""
        params = params.float()
        log_diag = params[..., :3]
        diag = torch.exp(log_diag).clamp(self.min_std, self.max_std)
        off = params[..., 3:]
        L = torch.zeros(*params.shape[:-1], 3, 3, dtype=params.dtype, device=params.device)
        L[..., 0, 0], L[..., 1, 1], L[..., 2, 2] = diag[..., 0], diag[..., 1], diag[..., 2]
        L[..., 1, 0], L[..., 2, 0], L[..., 2, 1] = off[..., 0], off[..., 1], off[..., 2]
        return L

    def local_std(self, params: torch.Tensor) -> torch.Tensor:
        """raw [..., 6] -> the three diagonal stds [..., 3] (cm); with ``ray_aligned`` index 0 is along the ray."""
        return torch.exp(params[..., :3].float()).clamp(self.min_std, self.max_std)

    @staticmethod
    def world_cholesky(L_local: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
        """Lower-triangular Cholesky factor of R L L^T R^T in the world frame ([..., 3, 3], float32; autocast off:
        the matmuls would otherwise run in bf16 under Lightning's mixed precision)."""
        with torch.autocast(device_type=L_local.device.type, enabled=False):
            Lf, Rf = L_local.float(), rot.float()
            Sigma = Rf @ Lf @ Lf.transpose(-1, -2) @ Rf.transpose(-1, -2)
            eye = torch.eye(3, device=Sigma.device, dtype=Sigma.dtype)
            return torch.linalg.cholesky(Sigma + 1e-6 * eye)


class ResidualConfHead(nn.Module):
    """Per-view 2D residual (feature pixels) + confidence logit from the view-aggregated sampled feature and the
    joint query (MVGFormer-style appearance module, adapted to fisheye rays). Zero-initialised: no residual, c=0.5."""

    def __init__(self, embed_dims: int, hidden: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dims * 2)
        self.mlp = MLP(embed_dims * 2, hidden, 3, num_layers=2)
        nn.init.zeros_(self.mlp.net[-1].weight)
        nn.init.zeros_(self.mlp.net[-1].bias)

    def forward(self, view_feat: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """view_feat [N, V, J, C], query [N, J, C] -> [N, V, J, 3] = (du, dv, conf_logit)."""
        q = query[:, None].expand_as(view_feat)
        return self.mlp(self.norm(torch.cat([view_feat, q], dim=-1)))


class GeoGate(nn.Module):
    """Uncertainty gate g in [0,1] deciding between the triangulated update and the learned offset.

    Inputs per joint (view-count agnostic, see ``RefineLayer.forward``): mean / min view confidence, fraction of valid
    views, both-valid flag, cos(ray angle), log trace Sigma_geo / 10, mean log-std of the learned head (or the
    constant reference when ``gate_uses_logstd`` is off), mean / min heatmap confidence (0.5 when unavailable) = 9
    inputs; ``gate_extra_inputs`` appends geo/learned consistency, along-ray log-std, informativeness flag, depth
    expectation and log depth spread (14). ``init_bias`` < 0 starts with the learned branch dominating; the gate is
    forced to 0 unless both rays are valid and the intersection lies in front of both cameras.
    """

    def __init__(self, hidden: int = 32, init_bias: float = -2.0, in_dim: int = 9):
        super().__init__()
        self.in_dim = in_dim  # 9 (default) or 14 with the extra inputs
        self.mlp = MLP(in_dim, hidden, 1, num_layers=2)
        nn.init.zeros_(self.mlp.net[-1].weight)
        with torch.no_grad():
            self.mlp.net[-1].bias.fill_(init_bias)

    def forward(self, feats: torch.Tensor, both_valid: torch.Tensor) -> torch.Tensor:
        """feats [N, J, in_dim], both_valid [N, J] bool -> g [N, J]."""
        g = torch.sigmoid(self.mlp(feats.float())[..., 0])
        return g * both_valid.to(g.dtype)


class DirectionalGeoGate(nn.Module):
    """Two gates per joint - one for the component of the update along the anchor's viewing
    ray (the stereo depth direction) and one for the two lateral components - instead of one scalar gate.

    Inputs: the scalar gate's features (9 / 14) plus the observability of the intersection - log of the three eigenvalues
    of its normal matrix sum_v w_v (I - d d^T) (depth direction ~ the smallest), log disparity rate (feature px / cm),
    anchor distance / 100 - and the geometric / learned disagreement along and across the ray (log cm / 5): 7 more.
    Zero-initialised with ``init_bias`` on both outputs, so it starts as the scalar gate's initialisation; forced to 0
    without two valid rays in front of the cameras, exactly like ``GeoGate``. It is a learned weighting, not a posterior:
    the geometric and learned candidates come from the same images and their errors are correlated."""

    EXTRA = 7

    def __init__(self, hidden: int = 32, init_bias: float = -2.0, in_dim: int = 9 + 7):
        super().__init__()
        self.in_dim = in_dim
        self.mlp = MLP(in_dim, hidden, 2, num_layers=2)
        nn.init.zeros_(self.mlp.net[-1].weight)
        with torch.no_grad():
            self.mlp.net[-1].bias.fill_(init_bias)

    def forward(self, feats: torch.Tensor, both_valid: torch.Tensor) -> torch.Tensor:
        """feats [N, J, in_dim], both_valid [N, J] bool -> g [N, J, 2] = (along the ray, lateral)."""
        g = torch.sigmoid(self.mlp(feats.float()))
        return g * both_valid.to(g.dtype)[..., None]


class BoneLengthHead(nn.Module):
    """P5: per-person bone lengths (cm) from the body token, as log-lengths around a per-bone
    prior. Zero-initialised so every sequence starts at the prior; the streaming path accumulates a causal running
    mean of the per-frame predictions (bone lengths are constant over a sequence) that feeds the ray-sphere prior."""

    def __init__(self, embed_dims: int, num_bones: int, prior_cm: Optional[Sequence[float]] = None, hidden: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dims)
        self.mlp = MLP(embed_dims, hidden, num_bones, num_layers=2)
        nn.init.zeros_(self.mlp.net[-1].weight)
        nn.init.zeros_(self.mlp.net[-1].bias)
        prior = torch.full((num_bones,), 30.0) if prior_cm is None else torch.tensor(list(prior_cm), dtype=torch.float32)
        self.register_buffer("log_prior", prior.clamp_min(1.0).log(), persistent=False)

    def forward(self, body_token: torch.Tensor) -> torch.Tensor:
        """[N, C] -> bone lengths [N, num_bones] in cm (positive)."""
        return torch.exp(self.log_prior + self.mlp(self.norm(body_token)).float().clamp(-1.5, 1.5))


class BoneGate(nn.Module):
    """P5: weight of the ray-sphere candidate P_bone against the (geo / learned) estimate of a joint.

    Inputs per joint: confidence of the chosen view, in-FOV flag, hit flag, gap / 10 (cm), consistency
    log|P_bone - P| / 5, bone length / 100, geometry gate of the joint, learned log-std, parent's bone gate."""

    def __init__(self, hidden: int = 32, init_bias: float = -2.0):
        super().__init__()
        self.mlp = MLP(9, hidden, 1, num_layers=2)
        nn.init.zeros_(self.mlp.net[-1].weight)
        with torch.no_grad():
            self.mlp.net[-1].bias.fill_(init_bias)

    def forward(self, feats: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """feats [N, M, 9], valid [N, M] bool -> g_b [N, M]."""
        return torch.sigmoid(self.mlp(feats.float())[..., 0]) * valid.to(feats.dtype)


class VisibilityHead(nn.Module):
    """Per-view joint visibility logits (EvaPose-style): is the joint inside the view's image *and* visible
    (not self-occluded / truncated)? Predicted from the refined joint token and, when available, the per-view
    sampled feature of the ray-consistent attention; the in-FOV flag is given as an input so the head only has to
    learn the occlusion part. Supervised with BCE against the projected-visibility labels of the dataset (FOV +
    in-front; self-occlusion labels can replace them when a dataset provides them)."""

    def __init__(self, embed_dims: int, num_views: int, hidden: int = 64):
        super().__init__()
        self.num_views = num_views
        self.norm_q = nn.LayerNorm(embed_dims)
        self.norm_v = nn.LayerNorm(embed_dims)
        self.mlp = MLP(2 * embed_dims + 1, hidden, 1, num_layers=2)
        nn.init.zeros_(self.mlp.net[-1].weight)
        with torch.no_grad():
            self.mlp.net[-1].bias.fill_(2.0)  # start "visible when in FOV"

    def forward(self, query: torch.Tensor, view_feat: Optional[torch.Tensor], in_fov: torch.Tensor) -> torch.Tensor:
        """query [N, J, C]; view_feat [N, V, J, C] or None; in_fov [N, V, J] bool -> logits [N, V, J]."""
        N, J, C = query.shape
        q = self.norm_q(query)[:, None].expand(N, self.num_views, J, C)
        if view_feat is None:
            view_feat = query.new_zeros(N, self.num_views, J, C)
        v = self.norm_v(view_feat.to(query.dtype))
        x = torch.cat([q, v, in_fov.to(query.dtype)[..., None]], dim=-1)
        logit = self.mlp(x)[..., 0]
        # outside the field of view a joint is never visible: force a strongly negative logit
        return torch.where(in_fov, logit, torch.full_like(logit, -10.0))


class JointMapper(nn.Module):
    """Linear map from the canonical query skeleton to the output skeleton of each dataset.

    A table over the *global* skeleton ids (``SKELETON_IDS``) holds one [MAX_JOINTS, J_canon] matrix per skeleton,
    NaN-padded beyond the skeleton's joint count; the canonical row is the (frozen) identity, the requested output
    skeletons start from ``mapping_matrix`` and are fine-tuned (``learnable``). ``one_to_one`` [S, MAX_JOINTS]
    gives the canonical joint copied by a target joint (-1 otherwise) to transfer 2D supervision.
    """

    def __init__(self, canonical: Skeleton, output_skeletons: Sequence[str], learnable: bool = True):
        super().__init__()
        self.canonical = canonical
        S, Jc = len(SKELETON_IDS), canonical.num_joints
        W0 = np.zeros((S, MAX_JOINTS, Jc), dtype=np.float32)
        valid = np.zeros((S, MAX_JOINTS), dtype=bool)
        learn = np.zeros((S, 1, 1), dtype=np.float32)
        one = np.full((S, MAX_JOINTS), -1, dtype=np.int64)
        num_joints = np.zeros(S, dtype=np.int64)
        for name, sid in SKELETON_IDS.items():
            if name == canonical.name or name in output_skeletons:
                W, o = mapping_matrix(canonical.name, name)
                J = W.shape[0]
                W0[sid, :J], valid[sid, :J], one[sid, :J], num_joints[sid] = W, True, o, J
                learn[sid] = float(learnable and name != canonical.name)
        self.register_buffer("W0", torch.from_numpy(W0))
        self.register_buffer("valid", torch.from_numpy(valid))
        self.register_buffer("learn_mask", torch.from_numpy(learn))
        self.register_buffer("one_to_one", torch.from_numpy(one))
        self.register_buffer("num_joints", torch.from_numpy(num_joints))
        self.delta = nn.Parameter(torch.zeros(S, MAX_JOINTS, Jc))
        self.names = {sid: name for name, sid in SKELETON_IDS.items()}

    @property
    def weight(self) -> torch.Tensor:
        return self.W0 + self.delta * self.learn_mask

    def forward(self, pose: torch.Tensor, skeleton_ids: torch.Tensor) -> torch.Tensor:
        """pose [B, ..., J_canon, 3]; skeleton_ids [B] -> [B, ..., MAX_JOINTS, 3] with NaN beyond each skeleton's joints."""
        W = self.weight.index_select(0, skeleton_ids)                            # [B, Jmax, Jc]
        shape = pose.shape
        p = pose.float().reshape(shape[0], -1, shape[-2], 3)                      # [B, T', Jc, 3]
        with torch.autocast(device_type=p.device.type, enabled=False):            # poses stay float32 (cm)
            out = torch.einsum("bkj,btjc->btkc", W.float(), p)                    # [B, T', Jmax, 3]
        mask = self.valid.index_select(0, skeleton_ids)[:, None, :, None]         # [B, 1, Jmax, 1]
        out = torch.where(mask, out, torch.full_like(out, float("nan")))
        return out.reshape(*shape[:-2], MAX_JOINTS, 3)

    def canonical_ids(self, skeleton_ids: torch.Tensor) -> torch.Tensor:
        """[B] -> [B, MAX_JOINTS] canonical joint copied by each target joint (-1 when not 1:1)."""
        return self.one_to_one.index_select(0, skeleton_ids)

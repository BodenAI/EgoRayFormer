"""EgoRayFormer: causal stereo-fisheye egocentric 3D pose estimation - "everything in ray space".

Pipeline per frame (T frames processed in parallel during training, streamed at inference):

  encoder (per view)  -> stride-4 features F_v, auxiliary 2D heatmaps H_v
  [TFM]               -> optional pixel-aligned causal temporal mixing (TSR-Ego baseline control)
  tokens              -> J joint tokens + 1 body token (+ optional track-prior token)
  Stage-1 proposal    -> conditioned multi-view cross-attention, kinematic self-attention, causal temporal
                         attention, FFN -> coarse 3D pose P0 (or a learned P0 when ``proposal_layers == 0``)
  Stage-2 refinement  -> L x [ project anchors into both views,
                             P2: ray-consistent 3D stereo sampling (shared 3D hypotheses for both views) or the
                                 v1 per-view Deformable Stereo Attention,
                             P3: trajectory-aligned sampling of the past w frames around the joint's own past anchors,
                             kinematic self-attention, causal temporal attention, FFN, offset head,
                             P4: per-layer Cholesky uncertainty,
                             P2: per-view 2D residual + confidence -> differentiable two-ray intersection P_geo,
                                 uncertainty gate g:  P_l = base + g (P_geo - base) + (1 - g) dP_learn ]
  heads               -> refined 3D pose per layer, per-joint Cholesky uncertainty

Ablation switches are constructor arguments (see configs/).
"""

from __future__ import annotations

import math
import warnings
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.camera import MODEL_SCARAMUZZA, RigBank, StereoRig, get_rig
from ..data.heatmaps import heatmap_softargmax, local_softargmax, sample_heatmap_at
from ..data.skeleton import HEATMAP_JOINT_IDS, NUM_JOINTS, get_skeleton
from .backbones import build_encoder
from .heads import (BoneGate, BoneLengthHead, DirectionalGeoGate, GeoGate, HeatmapHead, JointMapper, OffsetHead,
                    ResidualConfHead, UncertaintyHead, VisibilityHead)
from .layers.attention import FFN, MLP, MultiheadSelfAttention, sine_pos_embed_2d
from .layers.cross_view import ConditionedMultiViewCrossAttention
from .layers.deform_stereo_attn import DeformStereoAttn
from .layers.gated_fusion import ConfidenceGatedViewFusion
from .layers.kin_bias import KinematicAttentionBias
from .layers.ray_geometry import (disparity_rate, intersect_ray_sphere, intersect_rays, log_trace, normal_matrix_eigvals,
                                  pairwise_ray_cosine, ray_local_frame, so3_exp)
from .layers.ray_stereo_attn import RayStereoAttn
from .layers.temporal_attn import CausalTemporalAttention
from .layers.temporal_mixer import CausalTemporalFeatureMixer
from .layers.trajectory_sampling import LOG_REF_STD, PastFrames, TrajectoryAttention, build_past_parallel

TemporalFn = Callable[[nn.Module, torch.Tensor], torch.Tensor]
FOCAL_PX = 516.8  # first W2C coefficient of the UnrealEgo calibration: px per rad near the image centre
AUX_COND_DIM = 10  # headset / gravity conditioning vector: gravity dir (3) + device height (1) + 6D rotation (6)


@dataclass
class RefineInputs:
    feats: torch.Tensor                       # [N, V, C, h, w]
    ref: torch.Tensor                         # [N, J, 3] anchors used for projection (detached)
    base: torch.Tensor                        # [N, J, 3] pose the layer's update is added to
    uv_ref: torch.Tensor                      # [N, V, J, 2]
    fov_ref: torch.Tensor                     # [N, V, J] bool
    conf_hm: Optional[torch.Tensor]           # [N, V, J] heatmap confidence at uv_ref or None
    center: torch.Tensor                      # [N, 1, 3] device centre in the target frame
    project_fn: Callable                      # points [N, M, 3] -> uv [N, V, M, 2], fov [N, V, M]
    rays_fn: Callable                         # uv [N, V, M, 2] -> o [N, V, 1, 3], d [N, V, M, 3]
    kin_bias: Optional[torch.Tensor]
    temporal_fn: TemporalFn
    project_with_origin: Optional[Callable] = None   # (points, origin, rig_ids, rot) -> uv, fov  (P3 past frames)
    past_builder: Optional[Callable[[torch.Tensor], Optional[PastFrames]]] = None  # value -> past frames (P3)
    hm: Optional[torch.Tensor] = None         # [N, V, K, h, w] full heatmaps (local soft-argmax residual)
    bone_len: Optional[torch.Tensor] = None   # [N, num_bones] memorised bone lengths in cm (P5 ray-sphere prior)
    gt: Optional[torch.Tensor] = None         # [N, J, 3] GT joints (training only; RayDN-style along-ray hard negatives)
    angular_sigma: Optional[torch.Tensor] = None  # [N, V, 1] rad: half a feature pixel per (sample, view) from the rig calibration
    diag: bool = False                        # read-only probe of this layer's P2 samples (EgoRayFormer.diag_probe)


class ProposalLayer(nn.Module):
    def __init__(self, dim, heads, ffn_dims, num_views, temporal_window, dropout=0.0, temporal_pos="rope"):
        super().__init__()
        self.norm_cross = nn.LayerNorm(dim)
        self.cross = ConditionedMultiViewCrossAttention(dim, heads, num_views, dropout=dropout)
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = MultiheadSelfAttention(dim, heads, dropout)
        self.norm_temp = nn.LayerNorm(dim)
        self.temporal = CausalTemporalAttention(dim, heads, temporal_window, dropout, pos_mode=temporal_pos)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = FFN(dim, ffn_dims, dropout)

    def forward(self, x, feats, pos, kin_bias, temporal_fn: TemporalFn):
        x = x + self.cross(self.norm_cross(x), feats, pos)
        x = x + self.self_attn(self.norm_self(x), attn_bias=kin_bias)
        if self.temporal.enabled:
            x = x + temporal_fn(self.temporal, self.norm_temp(x))
        return x + self.ffn(self.norm_ffn(x))


class RefineLayer(nn.Module):
    def __init__(
        self, dim: int, heads: int, ffn_dims: int, num_views: int, num_joints: int, deform_points: int,
        conf_gate: bool, temporal_window: int, dropout: float = 0.0, sampling: str = "dsa", geo_branch: bool = False,
        geo_gate_init: float = -2.0, traj_window: int = 0, temporal_pos: str = "rope", uncertainty: bool = False,
        visibility_head: bool = False, geo_info_min_angle_deg: float = 0.0, geo_info_max_dist_cm: float = 0.0,
        gate_extra_inputs: bool = False, p2_kwargs: Optional[Dict] = None, hm_softargmax: bool = False,
        hm_softargmax_window: int = 7, heatmap_joint_ids: Optional[Sequence[int]] = None,
        kin_prior: bool = False, kin_levels: Optional[Sequence[Tuple[List[int], List[int], List[int]]]] = None,
        raydn_negatives: int = 0, raydn_depth_frac: float = 0.15, unc_ray_aligned: bool = False,
        unc_detach_tokens: bool = False, gate_uses_logstd: bool = True, traj_unc_scale_fixed: bool = False,
        geo_fusion: str = "scalar", p2_disp_step_px: float = 0.5, p2_depth_step_min_cm: float = 0.5,
        p2_depth_step_max_cm: float = 15.0, geo_gate_eval_override: float = -1.0, keep_unmodified: bool = False,
        layer_order: str = "cross_first",
    ):
        super().__init__()
        assert sampling in ("dsa", "ray_consistent")
        assert geo_fusion in ("scalar", "directional"), geo_fusion
        # "temporal_first": TSR-Ego's layer order (causal temporal self-attention -> joint self-attention -> stereo
        # cross-attention -> FFN); "cross_first" keeps stereo cross-attention -> joint -> temporal -> FFN
        assert layer_order in ("cross_first", "temporal_first"), layer_order
        self.layer_order = layer_order
        # "directional" = one gate along the anchor's viewing ray and one for the lateral axes (DirectionalGeoGate)
        self.geo_fusion = geo_fusion
        # with p2_depth_hyp "jacobian" one along-ray step = p2_disp_step_px of disparity at the anchor, clamped
        self.disp_step_px, self.depth_step_range = p2_disp_step_px, (p2_depth_step_min_cm, p2_depth_step_max_cm)
        # a constant gate at inference (>= 0) where the intersection is valid - the post-hoc
        # re-calibration the directional diagnostics compare against (training is unaffected)
        self.geo_gate_eval_override = geo_gate_eval_override
        # with the override active: also return the learned-gate pose (aux "pose_unmodified") for the streamed state
        self.keep_unmodified = keep_unmodified
        self.num_joints, self.sampling, self.num_views = num_joints, sampling, num_views
        self.unc_ray_aligned = unc_ray_aligned  # Cholesky factor expressed in the anchor's ray-local frame
        # P4 as a pure side output: the uncertainty head reads detached tokens (no NLL gradient into
        # the trunk) and its log-std is not consumed by the geometry gate (``gate_uses_logstd=False``) nor by the P3
        # weights (``traj_unc_scale_fixed``); the track token has its own switch (``track_prior_uses_uncertainty``).
        self.unc_detach_tokens, self.gate_uses_logstd = unc_detach_tokens, gate_uses_logstd
        # RayDN-style along-ray hard negatives (training only) - the P2 matching features must tell the GT
        # joint from points displaced along its viewing ray (depth offsets up to raydn_depth_frac x distance)
        self.raydn_negatives, self.raydn_depth_frac = (raydn_negatives if sampling == "ray_consistent" else 0), raydn_depth_frac
        if self.raydn_negatives > 0:
            self.raydn_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        # P5: kinematic ray geometry - joints solved level by level down the tree as the
        # intersection of their best view's ray with the sphere of the memorised bone length around the parent
        self.kin_prior = kin_prior and geo_branch and sampling == "ray_consistent"
        self.kin_levels = [(list(j), list(p), list(b)) for j, p, b in (kin_levels or [])] if self.kin_prior else []
        if self.kin_prior:
            self.bone_gate = BoneGate(init_bias=-2.0)
        p2_kwargs = dict(p2_kwargs or {})  # depth_hyp / depth_frac / content_attn / corr_channel of RayStereoAttn
        # Second 2D residual source - local soft-argmax of the auxiliary heatmaps around the projected anchor,
        # blended into the regressed residual with a per-joint learnable weight (init 0.5) scaled by the peak value
        self.hm_softargmax, self.hm_window = hm_softargmax, hm_softargmax_window
        self.hm_joint_ids = list(heatmap_joint_ids) if heatmap_joint_ids is not None else list(range(1, num_joints))
        if hm_softargmax:
            self.hm_mix_logit = nn.Parameter(torch.zeros(num_joints))
        # "Geometrically informative" joints = both rays valid, ray angle >= min_angle and anchor distance
        # <= max_dist (0 = no threshold). Exposed per layer as aux["geo_info"] for masking the geometric losses.
        self.geo_info_min_cos = math.cos(math.radians(geo_info_min_angle_deg)) if geo_info_min_angle_deg > 0 else None
        self.geo_info_max_dist = geo_info_max_dist_cm if geo_info_max_dist_cm > 0 else None
        self.gate_extra_inputs = gate_extra_inputs
        self.norm_dsa = nn.LayerNorm(dim)
        if sampling == "dsa":
            self.dsa = DeformStereoAttn(dim, heads, deform_points)  # shared across views (v1)
            self.fusion = ConfidenceGatedViewFusion(dim, num_views, use_conf=conf_gate)
            self.ray_attn = None
        else:
            self.ray_attn = RayStereoAttn(dim, heads, deform_points, num_views, **p2_kwargs)
        self.traj = (TrajectoryAttention(dim, heads, deform_points, traj_window, unc_scale_fixed=traj_unc_scale_fixed)
                     if (sampling == "ray_consistent" and traj_window > 0) else None)
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = MultiheadSelfAttention(dim, heads, dropout)
        self.norm_temp = nn.LayerNorm(dim)
        self.temporal = CausalTemporalAttention(dim, heads, temporal_window, dropout, pos_mode=temporal_pos)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = FFN(dim, ffn_dims, dropout)
        self.offset_head = OffsetHead(dim)
        self.unc_head = UncertaintyHead(dim, ray_aligned=unc_ray_aligned) if uncertainty else None
        self.vis_head = VisibilityHead(dim, num_views) if visibility_head else None
        self.geo = geo_branch and sampling == "ray_consistent"
        if self.geo:
            self.norm_geo = nn.LayerNorm(dim)
            self.res_head = ResidualConfHead(dim)
            # + geometry/learned consistency, along-ray learned sigma, informativeness flag, soft depth
            # expectation / spread of the P2 hypotheses (9 -> 14 inputs)
            base_in = 14 if gate_extra_inputs else 9
            if geo_fusion == "directional":
                self.gate = DirectionalGeoGate(in_dim=base_in + DirectionalGeoGate.EXTRA, init_bias=geo_gate_init)
            else:
                self.gate = GeoGate(in_dim=base_in, init_bias=geo_gate_init)

    def forward(self, x: torch.Tensor, inp: RefineInputs) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """x [N, J+E, C] -> updated tokens, pose [N, J, 3] and auxiliary outputs."""
        J = self.num_joints
        if self.layer_order == "temporal_first":
            if self.temporal.enabled:
                x = x + inp.temporal_fn(self.temporal, self.norm_temp(x))
            x = x + self.self_attn(self.norm_self(x), attn_bias=inp.kin_bias)
        q = self.norm_dsa(x[:, :J])
        aux: Dict[str, torch.Tensor] = {}
        # disparity rate of the anchor under the real camera model (feature px / cm along its viewing ray): step size of the
        # "jacobian" hypotheses, input of the directional gate and a diagnostic; no gradient
        aux["disp_rate"], _ = disparity_rate(inp.ref.detach(), inp.center, inp.project_fn, tuple(inp.feats.shape[-2:]))
        if self.sampling == "dsa":
            view_feats = torch.stack([self.dsa(q, inp.feats[:, v], inp.uv_ref[:, v]) for v in range(inp.feats.shape[1])], dim=1)
            fused = self.fusion(view_feats, inp.fov_ref, inp.conf_hm)
            res = None
        else:
            depth_unit = None
            if self.ray_attn.depth_hyp == "jacobian":   # one along-ray step = disp_step_px of disparity (clamped)
                lo, hi = self.depth_step_range
                depth_unit = (self.disp_step_px / aux["disp_rate"].clamp_min(1e-6)).clamp(lo, hi)
                aux["depth_unit"] = depth_unit
            res = self.ray_attn(q, inp.feats, inp.ref, inp.center, inp.project_fn, depth_unit=depth_unit)
            fused = res["out"]
            aux["hyp_min"], aux["hyp_max"] = res["hyp_min"], res["hyp_max"]
            if inp.diag:  # the sampled 3D hypotheses of this forward, their projections and the value maps they read
                N, J_, H, K, _ = res["points"].shape
                pts = res["points"].detach().reshape(N, J_ * H * K, 3)
                uv_c, fov_c = inp.project_fn(pts)
                V = uv_c.shape[1]
                aux.update(p2_value=res["value"].detach(), p2_points=pts.reshape(N, J_, H * K, 3),
                           p2_uv=uv_c.reshape(N, V, J_, H * K, 2), p2_fov=fov_c.reshape(N, V, J_, H * K))
            if self.traj is not None and inp.past_builder is not None:
                fused = fused + self.traj(q, res["delta_local"], self.ray_attn, inp.past_builder(res["value"]), inp.project_with_origin,
                                          depth_unit=depth_unit)
            if self.raydn_negatives > 0 and self.training and inp.gt is not None:
                aux["raydn_logits"], aux["raydn_valid"] = self._raydn_logits(res["value"], inp)
        x = torch.cat([x[:, :J] + fused, x[:, J:]], dim=1)
        if self.layer_order == "cross_first":
            x = x + self.self_attn(self.norm_self(x), attn_bias=inp.kin_bias)
            if self.temporal.enabled:
                x = x + inp.temporal_fn(self.temporal, self.norm_temp(x))
        x = x + self.ffn(self.norm_ffn(x))
        offset = self.offset_head(x[:, :J])
        if self.unc_head is not None:
            aux["unc_raw"] = self.unc_head(x[:, :J].detach() if self.unc_detach_tokens else x[:, :J])
            logstd = self.unc_head.log_std(aux["unc_raw"])
            if self.unc_ray_aligned:  # rotation ray-local -> target frame of the anchor's ray (columns r, e1, e2)
                r_ax, e1_ax, e2_ax = ray_local_frame(inp.ref.float(), inp.center.float())
                aux["unc_rot"] = torch.stack([r_ax, e1_ax, e2_ax], dim=-1)                # [N, J, 3, 3]
                aux["logstd_ray"] = torch.log(self.unc_head.local_std(aux["unc_raw"])[..., 0])  # along the ray
        else:
            logstd = offset.new_full(offset.shape[:2], LOG_REF_STD, dtype=torch.float32)
        aux["logstd"] = logstd
        if self.vis_head is not None:
            aux["vis_logit"] = self.vis_head(x[:, :J], res["view_feat"] if res is not None else None, inp.fov_ref)  # [N, V, J]
        base = inp.base.float()
        pose = base + offset.float()
        # diagnostics of the last layer: the learned candidate, the anchor and the device
        # centre (the ray-local basis of the directional analysis)
        aux["learn_pred"], aux["anchor"], aux["center"] = pose.detach(), inp.ref.detach(), inp.center.detach()
        if self.geo and res is not None:
            N, V = inp.uv_ref.shape[:2]
            h, w = inp.feats.shape[-2:]
            rc = self.res_head(res["view_feat"], self.norm_geo(x[:, :J])).float()          # [N, V, J, 3]
            du = rc[..., :2] / rc.new_tensor([float(w), float(h)])                         # feature px -> normalised
            uv_res = (inp.uv_ref.float() + du).clamp(0.0, 1.0)
            conf = torch.sigmoid(rc[..., 2])                                               # [N, V, J]
            if self.hm_softargmax and inp.hm is not None:  # blend in the heatmap soft-argmax estimate
                ids = self.hm_joint_ids
                uv_hm, c_hm = local_softargmax(inp.hm.detach().float(), inp.uv_ref[:, :, ids].float(), self.hm_window)
                mix = torch.sigmoid(self.hm_mix_logit[ids]).float()[None, None, :, None] * c_hm[..., None]  # [N, V, K, 1]
                uv_res = uv_res.clone()
                uv_res[:, :, ids] = uv_res[:, :, ids] + mix * (uv_hm - uv_res[:, :, ids])
                conf = conf.clone()
                conf[:, :, ids] = conf[:, :, ids] * (0.5 + 0.5 * c_hm)
                aux["uv_hm"], aux["conf_hm_peak"] = uv_hm, c_hm
            fov = inp.fov_ref.to(conf.dtype)
            if "vis_logit" in aux:  # predicted visibility down-weights views that see the joint but occluded
                conf = conf * torch.sigmoid(aux["vis_logit"].float().detach())
            o, d = inp.rays_fn(uv_res)
            # angular noise of half a feature pixel: per (sample, view) from the rig calibration when available,
            # else the UnrealEgo constant (1024 px calibration, FOCAL_PX px/rad near the centre)
            sigma = inp.angular_sigma if inp.angular_sigma is not None else 0.5 * 1024.0 / float(w) / FOCAL_PX
            p_geo, rho, cov = intersect_rays(o, d, conf * fov, prior=inp.ref, angular_sigma=sigma)
            # view-count agnostic gate features (any V): confidences / heatmaps summarised over the valid views
            n_valid = fov.sum(1)                                                            # [N, J]
            # the intersection is only trusted when >= 2 views see the joint *and* it lies in front of every valid
            # camera (rho > 0; diverging rays put the least-squares point behind the cameras)
            enough = (n_valid >= 2) & ((rho > 0) | (fov <= 0)).all(1)
            denom = n_valid.clamp_min(1.0)
            conf_mean = (conf * fov).sum(1) / denom
            conf_min = torch.where(fov > 0, conf, torch.ones_like(conf)).amin(1)
            hm = inp.conf_hm.float() if inp.conf_hm is not None else torch.full_like(conf, 0.5)
            hm_mean = (hm * fov).sum(1) / denom
            hm_min = torch.where(fov > 0, hm, torch.ones_like(hm)).amin(1)
            ray_cos = pairwise_ray_cosine(d, inp.fov_ref)                                  # [N, J] most parallel pair
            anchor_dist = (inp.ref.float() - inp.center.float()).norm(dim=-1)              # [N, J] cm from the device
            geo_info = enough
            if self.geo_info_min_cos is not None:
                geo_info = geo_info & (ray_cos <= self.geo_info_min_cos)
            if self.geo_info_max_dist is not None:
                geo_info = geo_info & (anchor_dist <= self.geo_info_max_dist)
            gate_logstd = logstd.float() if self.gate_uses_logstd else torch.full_like(logstd.float(), LOG_REF_STD)
            feats = [conf_mean, conf_min, n_valid / float(V), enough.to(conf.dtype), ray_cos,
                     log_trace(cov) / 10.0, gate_logstd, hm_mean, hm_min]
            if self.gate_extra_inputs:
                # consistency of the two candidates, learned sigma along the viewing ray, informativeness flag
                p_learn = base + offset.float()
                consist = torch.log((p_geo - p_learn).norm(dim=-1).clamp_min(1e-2)) / 5.0
                r_dir = F.normalize(inp.ref.float() - inp.center.float(), dim=-1)          # [N, J, 3]
                if not self.gate_uses_logstd:
                    logstd_ray = gate_logstd
                elif "logstd_ray" in aux:                                                   # ray-aligned head: direct
                    logstd_ray = aux["logstd_ray"].float()
                elif "unc_raw" in aux and self.unc_head is not None:
                    Lc = self.unc_head.cholesky(aux["unc_raw"])                             # [N, J, 3, 3]
                    with torch.autocast(device_type=Lc.device.type, enabled=False):        # r^T L L^T r in float32
                        var_ray = ((Lc.transpose(-1, -2) @ r_dir[..., None])[..., 0] ** 2).sum(-1)
                    logstd_ray = 0.5 * torch.log(var_ray.clamp_min(1e-4))
                else:
                    logstd_ray = logstd.float()
                feats += [consist, logstd_ray, geo_info.to(conf.dtype),
                          res["depth_expect"].float() / 10.0, 0.5 * torch.log(res["depth_var"].float().clamp_min(1e-2))]
            eig = normal_matrix_eigvals(d.detach(), (conf * fov).detach())                 # [N, J, 3] ascending
            d_geo = (p_geo - base).clamp(-60.0, 60.0)
            off = offset.float()
            if self.geo_fusion == "directional":
                # along the anchor's viewing ray and across it, every component with its own gate
                r_dir = F.normalize(inp.ref.float() - inp.center.float(), dim=-1)          # [N, J, 3] predicted basis
                delta = p_geo - (base + off)
                along = (delta * r_dir).sum(-1, keepdim=True)
                feats += [torch.log(eig.clamp_min(1e-6))[..., 0], torch.log(eig.clamp_min(1e-6))[..., 1],
                          torch.log(eig.clamp_min(1e-6))[..., 2], torch.log(aux["disp_rate"].clamp_min(1e-4)),
                          anchor_dist / 100.0, torch.log(along.abs()[..., 0].clamp_min(1e-2)) / 5.0,
                          torch.log((delta - along * r_dir).norm(dim=-1).clamp_min(1e-2)) / 5.0]
                g2 = self.gate(torch.stack(feats, dim=-1), enough)                         # [N, J, 2]
                dg_par = (d_geo * r_dir).sum(-1, keepdim=True)
                do_par = (off * r_dir).sum(-1, keepdim=True)

                def fuse2(gg):
                    gp, gl = gg[..., :1], gg[..., 1:]
                    return (base + (gp * dg_par + (1.0 - gp) * do_par) * r_dir
                            + gl * (d_geo - dg_par * r_dir) + (1.0 - gl) * (off - do_par * r_dir))

                if self.geo_gate_eval_override >= 0 and not self.training:
                    if self.keep_unmodified:
                        aux["pose_unmodified"] = fuse2(g2)
                    g2 = torch.full_like(g2, float(self.geo_gate_eval_override)) * enough.to(g2.dtype)[..., None]
                g_par, g_perp = g2[..., :1], g2[..., 1:]
                pose = fuse2(g2)
                g = (g_par[..., 0] + 2.0 * g_perp[..., 0]) / 3.0                          # summary for the statistics / P5
                aux.update(gate_par=g_par[..., 0], gate_perp=g_perp[..., 0])
            else:
                g = self.gate(torch.stack(feats, dim=-1), enough)                          # [N, J]
                if self.geo_gate_eval_override >= 0 and not self.training:
                    if self.keep_unmodified:
                        aux["pose_unmodified"] = base + g[..., None] * d_geo + (1.0 - g[..., None]) * off
                    g = torch.full_like(g, float(self.geo_gate_eval_override)) * enough.to(g.dtype)
                pose = base + g[..., None] * d_geo + (1.0 - g[..., None]) * off
            aux.update(geo_pred=p_geo, gate=g, uv_res=uv_res, view_conf=conf, geo_info=geo_info, ray_cos=ray_cos,
                       anchor_dist=anchor_dist)
            # the candidate the gate actually blends (clamped) and the observability of the intersection
            aux.update(geo_cand=(base + d_geo).detach(), n_valid=n_valid.detach(), conf_min=conf_min.detach(), geo_eig=eig)
            if self.kin_prior and inp.bone_len is not None and self.kin_levels:
                pose, bone_aux = self._kinematic_update(pose, o, d, conf, fov, g, gate_logstd, inp.bone_len)
                aux.update(bone_aux)
        return x, pose, aux

    def _raydn_logits(self, value: torch.Tensor, inp: RefineInputs):
        """Score the GT joint against ``raydn_negatives`` points displaced along its viewing ray (RayDN, ECCV 2024,
        adapted to stereo matching): the fused left/right features of every candidate are scored by ``raydn_head``;
        the LightningModule applies a cross-entropy with the GT candidate (index 0) as the target.
        Returns logits [N, J, 1 + Kn] and a validity mask [N, J] (finite GT, GT inside both views)."""
        gt = inp.gt.float()
        N, J, _ = gt.shape
        Kn = self.raydn_negatives
        finite = torch.isfinite(gt).all(-1)                                            # [N, J]
        gt = torch.nan_to_num(gt)
        center = inp.center.float()
        rel = gt - center
        dist = rel.norm(dim=-1, keepdim=True).clamp_min(1.0)
        r = rel / dist
        # offsets: sign x magnitude in [0.3, 1] x frac x distance (never closer than 30 % of the max offset)
        mag = (0.3 + 0.7 * torch.rand(N, J, Kn, device=gt.device)) * self.raydn_depth_frac * dist
        sign = torch.where(torch.rand(N, J, Kn, device=gt.device) < 0.5, -1.0, 1.0)
        cand = torch.cat([gt[:, :, None], gt[:, :, None] + (sign * mag)[..., None] * r[:, :, None]], dim=2)   # [N, J, 1+Kn, 3]
        uv, fov = inp.project_fn(cand.reshape(N, J * (1 + Kn), 3))
        uv = uv.reshape(N, self.num_views, J, 1 + Kn, 2)[:, :, :, None]                # [N, V, J, 1, K', 2] (H = 1 layout)
        fov = fov.reshape(N, self.num_views, J, 1 + Kn)[:, :, :, None]                 # [N, V, J, 1, K']
        Nn, V, H, hd, h, w = value.shape
        # sample every head at the same locations (expand the H axis of the uv grid)
        uv_h = uv.expand(N, V, J, H, 1 + Kn, 2)
        fov_h = fov.expand(N, V, J, H, 1 + Kn)
        fused = self.ray_attn.fuse_views(self.ray_attn.sample(value, uv_h), fov_h)     # [N, J, K', H, hd]
        logits = self.raydn_head(fused.reshape(N, J, 1 + Kn, H * hd).to(value.dtype)).squeeze(-1)  # [N, J, K']
        valid = finite & fov[:, :, :, 0, 0].all(dim=1)                                 # GT candidate inside both views
        return logits.float(), valid

    def _kinematic_update(self, pose, o, d, conf, fov, g, logstd, bone_len):
        """P5: level by level down the kinematic tree, intersect the best view's ray of every child joint with the
        sphere of its memorised bone length around the (already updated) parent and blend the result in with the
        bone gate. o [N, V, 1, 3], d [N, V, J, 3] rays through the residual-corrected 2D points; conf / fov [N, V, J];
        g [N, J] geometry gate; logstd [N, J]; bone_len [N, num_bones] (cm)."""
        N, V, J, _ = d.shape
        pose = pose.clone()
        score = conf * fov                                                              # [N, V, J]
        best = score.argmax(dim=1)                                                       # [N, J] view index
        ar = torch.arange(N, device=pose.device)[:, None]
        gates = torch.zeros(N, J, dtype=pose.dtype, device=pose.device)
        hits = torch.zeros(N, J, dtype=torch.bool, device=pose.device)
        p_bone_all = pose.detach().clone()
        for joints, parents, bones in self.kin_levels:
            jt = torch.as_tensor(joints, device=pose.device)
            v = best[:, jt]                                                              # [N, M]
            o_j = o.expand(N, V, J, 3)[ar, v, jt]                                        # [N, M, 3]
            d_j = d[ar, v, jt]                                                           # [N, M, 3]
            c_j = score[ar, v, jt]                                                       # [N, M]
            f_j = fov[ar, v, jt]
            center = pose[:, parents]                                                    # updated parents [N, M, 3]
            radius = bone_len[:, bones].float()                                          # [N, M]
            prior = pose[:, jt]
            X, hit, gap = intersect_ray_sphere(o_j, d_j, center, radius, prior)
            consist = torch.log((X - prior).norm(dim=-1).clamp_min(1e-2)) / 5.0
            feats = torch.stack([c_j, f_j, hit.float(), gap / 10.0, consist, radius / 100.0, g[:, jt], logstd[:, jt].float(),
                                 gates[:, parents]], dim=-1)                              # [N, M, 9]
            g_b = self.bone_gate(feats, f_j > 0)                                          # [N, M]
            pose[:, jt] = prior + g_b[..., None] * (X - prior)
            gates[:, jt], hits[:, jt], p_bone_all[:, jt] = g_b, hit, X.detach()
        return pose, {"bone_gate": gates, "bone_hit": hits, "bone_pred": p_bone_all}


class EgoRayFormer(nn.Module):
    def __init__(
        self,
        encoder: Dict,
        rigs: Sequence[str] = ("unrealego",),
        num_joints: int = NUM_JOINTS,
        num_views: int = 2,
        embed_dims: int = 128,
        num_heads: int = 4,
        ffn_dims: int = 512,
        image_size: int = 256,
        proposal_layers: int = 1,
        refine_layers: int = 3,
        deform_points: int = 16,
        e2e_grad: bool = True,
        iter_reproj: bool = True,
        conf_gate: bool = True,
        kin_bias: bool = True,
        uncertainty: bool = True,
        temporal_window: int = 1,
        track_prior: bool = False,
        track_prior_dropout: float = 0.3,
        track_prior_noise_cm: float = 3.0,
        dataset_embed: bool = True,
        num_datasets: int = 8,
        heatmap_head: bool = True,
        denoise_groups: int = 0,
        denoise_std_cm: float = 5.0,
        dropout: float = 0.0,
        to_mm: float = 10.0,
        # ---- redesign switches ----
        sampling: str = "dsa",            # "dsa" (v1 per-view 2D deformable) | "ray_consistent" (P2 shared 3D hypotheses)
        geo_branch: bool = False,         # P2: two-ray intersection + uncertainty gate (needs sampling == "ray_consistent")
        geo_gate_init: float = -2.0,
        traj_window: int = 0,             # P3: trajectory-aligned sampling of the past w frames (0 = off)
        temporal_pos: str = "rope",       # "rope" | "age" | "rope+age"
        temporal_mixer: bool = False,     # TSR-Ego TFM (pixel-aligned causal temporal conv) as a control
        tfm_blocks: int = 2,
        tfm_kernel: int = 3,
        tfm_dilations: Sequence[int] = (1, 2),
        refine_mode: str = "residual",    # "residual": P_l = P_{l-1} + dP | "absolute": P_l = P0 + dP_l (TSR-Ego)
        refine_layer_order: str = "cross_first",  # "temporal_first": TSR-Ego order (temporal -> joint -> stereo -> FFN)
        uncertainty_per_layer: bool = True,
        track_prior_uses_uncertainty: bool = True,
        # ---- multi-dataset / multi-device extensions ----
        skeleton: str = "unrealego16",    # canonical skeleton of the joint queries (see data/skeleton.py registry)
        output_skeletons: Sequence[str] = (),  # extra dataset skeletons mapped from the canonical one (JointMapper)
        aux_cond: bool = False,           # headset / gravity conditioning of the body token (EPFv2-style auxiliary query)
        visibility_head: bool = False,    # per-view joint visibility logits (EvaPose-style), also used by the geometry gate
        traj_last_layer_only: bool = False,  # P3 only in the last refinement layer (latency / accuracy trade-off)
        # ---- geometrically informative joints, gate inputs ----
        geo_info_min_angle_deg: float = 0.0,  # joints with a smaller ray angle are not "geometrically informative" (0 = off)
        geo_info_max_dist_cm: float = 0.0,    # joints farther from the device are not informative (0 = off)
        gate_extra_inputs: bool = False,      # GeoGate also sees geo/learned consistency, along-ray sigma, informativeness
        # ---- P2 hypotheses / matching (RayStereoAttn) ----
        p2_depth_hyp: str = "pm",             # "pm" (+/-3 cm alternating, v1 of the redesign) | "layered" (depth levels incl. 0)
        p2_depth_frac: float = 0.15,          # layered: delta = frac * reference distance (scaled with the anchor distance)
        p2_content_attn: bool = False,        # query x point-feature dot product added to the point logits
        p2_corr_channel: bool = False,        # left-right cosine similarity per hypothesis in the view fusion
        # ---- Heatmap local soft-argmax as a second 2D residual source of the geometry branch ----
        hm_softargmax: bool = False,
        hm_softargmax_window: int = 7,
        # ---- Calibration robustness ----
        learn_extrinsics: bool = False,       # learnable per-rig SE(3) refinement of every camera (zero-initialised;
                                              # unfreeze on real-device data only, trained by the reprojection losses)
        max_rigs: int = 16,                   # size of the learnable extrinsic table (rigs added later must fit)
        calib_jitter_rot_deg: float = 0.0,    # training-time calibration jitter: rotation std (deg) per rig and forward
        calib_jitter_trans_cm: float = 0.0,   # ... and translation std (cm); covers the measured UnrealEgo-RW session drift
        # ---- P5: kinematic ray geometry + online bone-length memory ----
        kin_prior: bool = False,              # ray-sphere candidate P_bone for child joints, blended by the bone gate
        bone_memory: str = "causal_mean",     # "causal_mean" (running mean over the sequence) | "frame" (per-frame prediction)
        # ---- RayDN-style along-ray hard negatives (training-only auxiliary classification) ----
        raydn_negatives: int = 0,             # number of displaced candidates per joint (0 = off)
        raydn_depth_frac: float = 0.15,       # max displacement along the ray as a fraction of the joint distance
        # ---- Uncertainty expressed in the anchor's ray-local frame (only meaningful with the uncertainty head) ----
        unc_ray_aligned: bool = False,
        # ---- Geometric proposal initialisation from the heatmap peaks ----
        proposal_init: str = "learned",       # "learned" (P0 of the proposal decoder) | "triangulated" | "hybrid" (confidence blend)
        proposal_init_conf: float = 0.3,      # min heatmap peak in both views for the triangulated anchor
        # ---- Exposure bias of the trajectory sampler (in-framework stand-in for cross-training caching) ----
        traj_anchor_noise_cm: float = 0.0,    # training: the past anchors P3 samples around are perturbed by N(0, sigma)
        traj_anchor_noise_p: float = 0.0,     # ... on this fraction of the (frame, joint) pairs, so the sampler sees
                                              # realistic (validation-sized) errors instead of the over-fitted training ones
        # Track token log-std channel. In training the channel is the constant
        # log(track_prior_noise_cm) (or the sampled noise level with ``track_noise_sample``); feeding the *predicted*
        # log-std at test ("predicted") shifted the token off-distribution and cost 12-14 mm on every P4 run, so the
        # default now feeds the training constant at test as well.
        track_logstd_at_test: str = "train_constant",
        track_noise_sample: bool = False,     # training: per-(frame, joint) noise level sigma ~ LogUniform(1, 8) cm on the
                                              # previous pose, log sigma in the channel (gives the channel a meaning)
        # exposure bias: i.i.d. noise on the teacher-forced previous pose is "seen through" (GT + 3 cm noise
        # is only 0.15 mm worse than clean GT) while the streamed own prediction is 13 mm worse: "correlated" draws a
        # per-clip per-joint offset plus AR(1) noise whose scale grows with the joint's distance to the device.
        track_noise_mode: str = "iid",        # "iid" | "correlated"
        # ---- P4 as a pure side output: no uncertainty gradient / input into the pose path ----
        unc_detach_tokens: bool = False,      # the uncertainty head reads detached tokens (NLL cannot shape the trunk)
        gate_uses_logstd: bool = True,        # False: the geometry gate sees the constant reference log-std
        traj_unc_scale_fixed: bool = False,   # True: P3 weights ignore the past anchors' log-std (unc_scale = 0 buffer)
        # ---- valid views, directional fusion, depth-hypothesis steps, rig augmentation ----
        p2_mask_invalid_views: bool = False,  # out-of-image P2 / P3 samples -> learnt missing-view vector, no correlation
        geo_fusion: str = "scalar",           # "directional" = gates along / across the anchor's viewing ray
        p2_disp_step_px: float = 0.5,         # p2_depth_hyp "jacobian": disparity per along-ray step (feature px)
        p2_depth_step_min_cm: float = 0.5,    # ... clamped to [min, max] cm (no search beyond max where depth is unobservable)
        p2_depth_step_max_cm: float = 15.0,
        geo_gate_eval_override: float = -1.0,  # >= 0: constant gate at inference where the intersection is valid
        geo_gate_eval_override_layers: str = "all",         # "last": the override acts in the last refinement layer only
        geo_gate_eval_override_feedback: str = "modified",  # "original": the streamed track prior keeps the learned gate's
                                                            # output (the override changes the reported pose only)
        p2_depth_scaled: bool = True,         # False: P2 offsets in cm independent of the anchor distance
        p2_depth_along_scaled: bool = True,   # False: only the along-ray offsets in fixed cm ("fixed" step-rule control)
        fov_circle: bool = False,             # projections outside the image circle are outside the view
    ):
        super().__init__()
        assert sampling in ("dsa", "ray_consistent") and refine_mode in ("residual", "absolute")
        if geo_branch and sampling != "ray_consistent":
            warnings.warn("geo_branch requires sampling='ray_consistent'; disabling it")
            geo_branch = False
        if traj_window > 0 and sampling != "ray_consistent":
            warnings.warn("traj_window requires sampling='ray_consistent'; disabling trajectory sampling")
            traj_window = 0
        self.skeleton = get_skeleton(skeleton)
        if num_joints != self.skeleton.num_joints:
            warnings.warn(f"num_joints={num_joints} overridden by skeleton {skeleton!r} ({self.skeleton.num_joints} joints)")
            num_joints = self.skeleton.num_joints
        self.J, self.V, self.C = num_joints, num_views, embed_dims
        self.image_size = image_size
        self.e2e_grad, self.iter_reproj = e2e_grad, iter_reproj
        self.use_uncertainty, self.use_heatmap = uncertainty, heatmap_head
        self.temporal_window = temporal_window
        self.track_prior, self.track_dropout, self.track_noise = track_prior, track_prior_dropout, track_prior_noise_cm
        self.track_uses_unc = track_prior_uses_uncertainty
        # single source of truth for the track token's log-std channel: training constant = eval constant = fallback
        self.track_ls_const = math.log(max(track_prior_noise_cm, 0.3))
        self.track_noise_sample = track_noise_sample
        self.track_noise_range = (math.log(1.0), math.log(8.0))   # cm, log-uniform
        assert track_noise_mode in ("iid", "correlated"), track_noise_mode
        self.track_noise_mode = track_noise_mode
        self.denoise_groups, self.denoise_std = denoise_groups, denoise_std_cm
        self.to_mm = to_mm
        self.sampling, self.geo_branch, self.traj_window = sampling, geo_branch, traj_window
        self.refine_mode = refine_mode
        self.use_aux_cond, self.use_visibility = aux_cond, visibility_head
        self.geo_info_enabled = geo_info_min_angle_deg > 0 or geo_info_max_dist_cm > 0  # geometric-loss masks active
        self.geo_info_min_cos = math.cos(math.radians(geo_info_min_angle_deg)) if geo_info_min_angle_deg > 0 else None
        self.geo_info_max_dist = geo_info_max_dist_cm if geo_info_max_dist_cm > 0 else None
        self.num_extra = 1 + (1 if track_prior else 0)  # body token (+ track token)

        # ---- encoder & rigs ----
        enc_cfg = dict(encoder)
        enc_cfg.setdefault("num_views", num_views)
        if enc_cfg.get("type") == "stereo_fisheye_vit":
            # single source of truth: the encoder token grid follows the model input resolution
            if enc_cfg.get("img_size") not in (None, image_size):
                warnings.warn(f"encoder img_size={enc_cfg['img_size']} overridden by model image_size={image_size}")
            enc_cfg["img_size"] = image_size
        self.encoder = build_encoder(enc_cfg)
        self.feat_proj = nn.Conv2d(self.encoder.out_channels, embed_dims, 1)
        self.heatmap_joint_ids = list(self.skeleton.heatmap_joint_ids)
        self.heatmap_head = HeatmapHead(self.encoder.out_channels, len(self.heatmap_joint_ids)) if heatmap_head else None
        self.tfm = CausalTemporalFeatureMixer(embed_dims, tfm_blocks, tfm_kernel, tfm_dilations) if temporal_mixer else None
        self.rig_names = list(rigs)
        self.rigs = nn.ModuleList([get_rig(r) for r in self.rig_names])
        for r in self.rigs:
            assert r.num_views == num_views, f"rig {r.name} has {r.num_views} views, model expects {num_views}"
        self.fov_circle = fov_circle
        self.rig_bank = RigBank(list(self.rigs))  # loop-free batched projection over rigs and views
        self.rig_bank.circle_fov = fov_circle
        self._geom_cache: Dict[Tuple, Dict[str, torch.Tensor]] = {}
        # Per-rig learnable extrinsic refinement (rotvec rad, translation cm) and calibration jitter
        self.learn_extrinsics, self.max_rigs = learn_extrinsics, max_rigs
        assert len(self.rigs) <= max_rigs, f"{len(self.rigs)} rigs > max_rigs={max_rigs}"
        self.extr_delta = nn.Parameter(torch.zeros(max_rigs, num_views, 6)) if learn_extrinsics else None
        self.calib_jitter = (float(calib_jitter_rot_deg), float(calib_jitter_trans_cm))
        self._jitter: Optional[torch.Tensor] = None  # [R, V, 6] sampled once per training forward
        # rig augmentation: while ``rig_augmentation`` is active, rig ids >= _aug_base are per-sample
        # virtual rigs; _aug_src maps them to their source rig, _aug_ray_rot rotates the encoder's device-frame rays
        self._aug_base: Optional[int] = None
        self._aug_src: Optional[torch.Tensor] = None
        self._aug_ray_rot: Optional[torch.Tensor] = None

        # ---- tokens ----
        self.joint_embed = nn.Embedding(num_joints, embed_dims)
        self.body_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        nn.init.normal_(self.body_token, std=0.02)
        self.dataset_embed = nn.Embedding(num_datasets, embed_dims) if dataset_embed else None
        if self.dataset_embed is not None:
            nn.init.zeros_(self.dataset_embed.weight)
        self.anchor_embed = MLP(3, embed_dims, embed_dims, num_layers=2)  # xyz (m) -> C
        if track_prior:
            in_dim = num_joints * 3 + (num_joints if track_prior_uses_uncertainty else 0)
            self.track_mlp = MLP(in_dim, embed_dims, embed_dims, num_layers=2)
            self.track_null = nn.Parameter(torch.zeros(1, 1, embed_dims))
            nn.init.normal_(self.track_null, std=0.02)
        if aux_cond:  # headset pose / gravity conditioning: zero-initialised so that it starts as a no-op
            self.cond_mlp = MLP(AUX_COND_DIM, embed_dims, embed_dims, num_layers=2)
            nn.init.zeros_(self.cond_mlp.net[-1].weight)
            nn.init.zeros_(self.cond_mlp.net[-1].bias)
            self.cond_null = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.kin_bias = KinematicAttentionBias(num_heads, num_joints, self.num_extra, dist=self.skeleton.tree_distance()) if kin_bias else None
        self.joint_mapper = JointMapper(self.skeleton, output_skeletons) if output_skeletons else None
        # P5: bone-length head on the body token + kinematic levels (child, parent, bone index) from the skeleton
        assert proposal_init in ("learned", "triangulated", "hybrid")
        self.proposal_init, self.proposal_init_conf = (proposal_init if heatmap_head else "learned"), proposal_init_conf
        self.traj_anchor_noise = (float(traj_anchor_noise_cm), float(traj_anchor_noise_p))
        assert track_logstd_at_test in ("predicted", "train_constant")
        self.track_logstd_at_test = track_logstd_at_test
        self.kin_prior = kin_prior and geo_branch and sampling == "ray_consistent"
        assert geo_gate_eval_override_layers in ("all", "last"), geo_gate_eval_override_layers
        assert geo_gate_eval_override_feedback in ("modified", "original"), geo_gate_eval_override_feedback
        if geo_gate_eval_override_feedback == "original" and (geo_gate_eval_override_layers != "last" or self.kin_prior):
            raise ValueError("geo_gate_eval_override_feedback='original' needs geo_gate_eval_override_layers='last' and no "
                             "kin_prior (the learned-gate pose of the last layer is what the stream state keeps)")
        self.override_feedback = geo_gate_eval_override_feedback
        self.bone_memory = bone_memory
        self.bones = list(self.skeleton.bones)                         # (child, parent) pairs among real joints
        self.kin_levels = self._kinematic_levels() if self.kin_prior else []
        self.bone_head = BoneLengthHead(embed_dims, len(self.bones)) if self.kin_prior else None

        # ---- decoders ----
        self.proposal_layers = nn.ModuleList(
            [ProposalLayer(embed_dims, num_heads, ffn_dims, num_views, temporal_window, dropout, temporal_pos) for _ in range(proposal_layers)]
        )
        # (the former ``proposal_unc`` head was removed: it received no gradient - its only consumer,
        # the layer-0 P3 past frames, detaches it - and therefore always output the constant LOG_REF_STD)
        if proposal_layers > 0:
            self.proposal_head = OffsetHead(embed_dims)
            self.p0 = None
        else:  # TSR-Ego style single-stage decoder: learned initial 3D reference pose (zero-initialised)
            self.proposal_head = None
            self.p0 = nn.Parameter(torch.zeros(num_joints, 3))
        self.refine_layers = nn.ModuleList([
            RefineLayer(
                embed_dims, num_heads, ffn_dims, num_views, num_joints, deform_points, conf_gate, temporal_window, dropout,
                sampling=sampling, geo_branch=geo_branch, geo_gate_init=geo_gate_init,
                traj_window=traj_window if (not traj_last_layer_only or i == refine_layers - 1) else 0,
                temporal_pos=temporal_pos, uncertainty=uncertainty and (uncertainty_per_layer or i == refine_layers - 1),
                visibility_head=visibility_head, geo_info_min_angle_deg=geo_info_min_angle_deg,
                geo_info_max_dist_cm=geo_info_max_dist_cm, gate_extra_inputs=gate_extra_inputs,
                p2_kwargs=dict(depth_hyp=p2_depth_hyp, depth_frac=p2_depth_frac, content_attn=p2_content_attn, corr_channel=p2_corr_channel,
                               mask_invalid_views=p2_mask_invalid_views, depth_scaled=p2_depth_scaled,
                               along_scaled=p2_depth_along_scaled),
                hm_softargmax=hm_softargmax and heatmap_head, hm_softargmax_window=hm_softargmax_window,
                heatmap_joint_ids=self.heatmap_joint_ids, kin_prior=self.kin_prior, kin_levels=self.kin_levels,
                raydn_negatives=raydn_negatives, raydn_depth_frac=raydn_depth_frac, unc_ray_aligned=unc_ray_aligned,
                unc_detach_tokens=unc_detach_tokens, gate_uses_logstd=gate_uses_logstd, traj_unc_scale_fixed=traj_unc_scale_fixed,
                geo_fusion=geo_fusion, p2_disp_step_px=p2_disp_step_px, p2_depth_step_min_cm=p2_depth_step_min_cm,
                p2_depth_step_max_cm=p2_depth_step_max_cm,
                geo_gate_eval_override=(geo_gate_eval_override if geo_gate_eval_override_layers == "all" or i == refine_layers - 1
                                        else -1.0),
                keep_unmodified=geo_gate_eval_override_feedback == "original" and i == refine_layers - 1,
                layer_order=refine_layer_order,
            )
            for i in range(refine_layers)
        ])
        self._pos_cache: Dict[Tuple, torch.Tensor] = {}
        # (extent_cm, step_cm): forward_streaming adds the read-only probe of the last layer to its outputs; set by the LightningModule for diagnostic evaluations, never saved
        self.diag_probe: Optional[Tuple[float, float]] = None

    # ------------------------------------------------------------------ rigs
    def add_rigs(self, names: Sequence[str]) -> List[str]:
        """Register additional camera rigs (names or rig.json paths) at test / fine-tuning time - e.g. evaluating a
        UnrealEgo2-trained model zero-shot on the FRAME or Ego4View rig. Calibration buffers are not part of the
        checkpoint, so this only rebuilds the RigBank. Returns the names that were added."""
        added = []
        for n in names:
            if n in self.rig_names:
                continue
            rig = get_rig(n)
            assert rig.num_views == self.V, f"rig {rig.name} has {rig.num_views} views, model expects {self.V}"
            self.rig_names.append(n)
            self.rigs.append(rig.to(self.rig_bank.R.device))
            added.append(n)
        if added:
            assert self.extr_delta is None or len(self.rigs) <= self.max_rigs, f"more than max_rigs={self.max_rigs} rigs with learn_extrinsics"
            self.rig_bank = RigBank(list(self.rigs)).to(self.rig_bank.R.device)
            self.rig_bank.circle_fov = self.fov_circle
            self._geom_cache.clear()
        return added

    # ------------------------------------------------------------------ Geometric proposal init
    def init_anchors(self, proposal: torch.Tensor, hm: Optional[torch.Tensor], origin_cam: torch.Tensor,
                     rig_ids: torch.Tensor, rot: Optional[torch.Tensor]) -> torch.Tensor:
        """Anchors of the first refinement layer. ``learned``: the proposal P0. ``triangulated`` / ``hybrid``: the
        heatmap peaks of both views are intersected (two-ray least squares with the proposal as prior); joints whose
        peaks are confident in both views take the triangulated point (``triangulated``) or a confidence-weighted
        blend (``hybrid``); joints without a heatmap channel (head) keep P0. Everything in fp32, gradients flow to
        the proposal through the prior and the blend."""
        if self.proposal_init == "learned" or hm is None:
            return proposal
        ids = self.heatmap_joint_ids
        N = proposal.shape[0]
        uv, peak = heatmap_softargmax(hm.detach().float())                        # [N, V, K, 2], [N, V, K]
        o, d = self.rays(uv, origin_cam, rig_ids, rot)                             # [N, V, 1, 3], [N, V, K, 3]
        conf = (peak >= self.proposal_init_conf).float() * peak                     # [N, V, K]
        p_tri, rho, _ = intersect_rays(o, d, conf, prior=proposal[:, ids].float())  # [N, K, 3], [N, V, K]
        both = (peak >= self.proposal_init_conf).all(dim=1) & (rho > 0).all(dim=1)  # [N, K] confident and in front of both cameras
        # same informativeness thresholds as the geometric losses: far / near-parallel joints keep P0
        if self.geo_info_min_cos is not None:
            both = both & (pairwise_ray_cosine(d, both[:, None].expand(N, self.V, -1)) <= self.geo_info_min_cos)
        if self.geo_info_max_dist is not None:
            center = self.center(origin_cam, rig_ids, rot)                          # [N, 1, 3]
            both = both & ((proposal[:, ids].float() - center).norm(dim=-1) <= self.geo_info_max_dist)
        w = peak.amin(dim=1) if self.proposal_init == "hybrid" else torch.ones_like(both, dtype=torch.float32)
        w = (w * both.float())[..., None]
        anchors = proposal.float().clone()
        anchors[:, ids] = proposal[:, ids].float() + w * ((p_tri - proposal[:, ids].float()).clamp(-60.0, 60.0))
        return anchors

    # ------------------------------------------------------------------ P5 helpers
    def _kinematic_levels(self) -> List[Tuple[List[int], List[int], List[int]]]:
        """Group child joints by depth below the first joint of their chain that has no real parent: level 1 =
        children of chain roots (e.g. upper arms of the neck, calves of the thighs), level 2 their children, ...
        Every entry is (child joints, parent joints, bone indices into ``self.bones``)."""
        parents = list(self.skeleton.parents)
        J = self.skeleton.num_joints
        bone_idx = {(c, p): i for i, (c, p) in enumerate(self.bones)}
        depth = {}

        def dep(j):
            if j in depth:
                return depth[j]
            p = parents[j]
            depth[j] = 0 if not (0 <= p < J) else dep(p) + 1
            return depth[j]

        levels: Dict[int, Tuple[List[int], List[int], List[int]]] = {}
        for j in range(J):
            if dep(j) >= 1:
                lv = levels.setdefault(dep(j), ([], [], []))
                lv[0].append(j)
                lv[1].append(parents[j])
                lv[2].append(bone_idx[(j, parents[j])])
        return [levels[k] for k in sorted(levels)]

    def bone_lengths(self, x: torch.Tensor) -> torch.Tensor:
        """Per-frame bone lengths [N, num_bones] (cm) from the body token (position J) of the token sequence."""
        return self.bone_head(x[:, self.J].float())

    def _bone_memory_parallel(self, bone: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """Causal running mean over the clip (identical to the streaming state): [N, nb] -> [N, nb]."""
        if self.bone_memory != "causal_mean":
            return bone
        b = bone.reshape(B, T, -1)
        cnt = torch.arange(1, T + 1, device=bone.device, dtype=bone.dtype)[None, :, None]
        return (b.cumsum(1) / cnt).reshape(B * T, -1)

    # ------------------------------------------------------------------ deployment helpers
    def compile_decoder(self, mode: str = "reduce-overhead", dynamic: Optional[bool] = None) -> int:
        """``torch.compile`` the launch-bound decoder leaves (ray-consistent sampling, trajectory sampling,
        residual / offset / uncertainty heads and the temporal attention) in place. The streaming loop itself keeps
        its Python state (deques, KV caches), so only the per-layer tensor kernels are compiled - the part that
        dominates the 40 ms streaming step at B = 1 (many small kernels). Returns the number of compiled modules;
        ``mode="reduce-overhead"`` uses CUDA graphs once the shapes are stable (after the first ``window`` steps)."""
        def clone_out(x):  # CUDA-graph outputs are overwritten by the next replay: detach them by cloning
            if torch.is_tensor(x):
                return x.clone()
            if isinstance(x, dict):
                return {k: clone_out(v) for k, v in x.items()}
            if isinstance(x, (tuple, list)):
                return type(x)(clone_out(v) for v in x)
            return x

        def wrap(obj, attr):
            """Compile ``obj.<attr>`` in place; only methods with pure tensor arguments (a fresh closure such as the
            per-frame ``project_fn`` would trigger a recompilation on every call)."""
            if hasattr(obj, f"_orig_{attr}"):
                return 0
            fn = getattr(obj, attr)
            setattr(obj, f"_orig_{attr}", fn)
            compiled = torch.compile(fn, mode=mode, dynamic=dynamic)
            if mode == "reduce-overhead":
                setattr(obj, attr, lambda *a, **k: clone_out(compiled(*a, **k)))
            else:
                setattr(obj, attr, compiled)
            return 1

        n = 0
        for layer in self.refine_layers:
            for name in ("dsa", "self_attn", "ffn", "offset_head", "unc_head", "res_head", "vis_head"):
                mod = getattr(layer, name, None)
                if mod is not None:
                    n += wrap(mod, "forward")
            if layer.ray_attn is not None:  # tensor-only pieces of the ray-consistent sampling (shared with P3)
                for name in ("project_values", "sample", "fuse_views", "to_world", "offsets_local"):
                    n += wrap(layer.ray_attn, name)
        for layer in self.proposal_layers:
            for name in ("cross", "self_attn", "ffn"):
                mod = getattr(layer, name, None)
                if mod is not None:
                    n += wrap(mod, "forward")
        self._decoder_compiled = n > 0
        return n

    # ------------------------------------------------------------------ geometry
    def _rig_geometry(self, rig_idx: int, H: int, W: int, device) -> Dict[str, torch.Tensor]:
        key = (rig_idx, H, W, str(device))
        if key not in self._geom_cache:
            rig: StereoRig = self.rigs[rig_idx]
            g: Dict[str, torch.Tensor] = {}
            needs = self.encoder.needs_geometry
            # cached for the whole run from inside the bf16-mixed forward: must be exact fp32
            with torch.autocast(device_type=torch.device(device).type, enabled=False):
                if "gnomonic" in needs:
                    patch = getattr(self.encoder, "patch_size", 16)
                    grid, rays, _ = rig.gnomonic_grids(H // patch, patch)
                    g["grid"], g["rays"] = grid.to(device), rays.to(device)
                if "lut" in needs:  # real-layout -> virtual sampling grid at the encoder's output stride
                    patch = getattr(self.encoder, "patch_size", 16)
                    stride = getattr(self.encoder, "out_stride", 4)
                    g["lut"] = rig.real_to_virtual_luts(H // patch, patch, (H // stride, W // stride)).to(device)
                if "ray_map" in needs:
                    g["ray_map"] = rig.ray_maps(H, W).to(device)
            self._geom_cache[key] = g
        return self._geom_cache[key]

    def _gather_geometry(self, rig_ids_bt: torch.Tensor, H: int, W: int) -> Optional[Dict[str, torch.Tensor]]:
        if not self.encoder.needs_geometry:
            return None
        per_rig = [self._rig_geometry(i, H, W, rig_ids_bt.device) for i in range(len(self.rigs))]
        src = self.source_rig_ids(rig_ids_bt)
        out = {}
        for k in per_rig[0]:
            stacked = torch.stack([g[k] for g in per_rig], dim=0)  # [R, V, ...]
            sel = stacked.index_select(0, src)                    # [N, V, ...]
            if k in ("rays", "ray_map") and self._aug_base is not None:   # B: rays of the rotated cameras (device frame)
                virt = rig_ids_bt >= self._aug_base
                rot = torch.eye(3, device=sel.device).expand(sel.shape[0], sel.shape[1], 3, 3).clone()
                rot[virt] = self._aug_ray_rot.index_select(0, (rig_ids_bt[virt] - self._aug_base)).to(rot.dtype)
                with torch.autocast(device_type=sel.device.type, enabled=False):   # cached ray geometry stays fp32
                    sel = torch.einsum("nvij,nvjhw->nvihw", rot, sel.float()).to(sel.dtype)
            out[k] = sel.reshape(-1, *sel.shape[2:])              # [N*V, ...]
        return out

    # All camera geometry runs in float32 regardless of autocast: the polynomial fisheye model and the ray
    # intersection are far too sensitive for bf16 (0.4 % relative error = several pixels at 1024 px). The RigBank
    # evaluates a mixed batch (several rigs) in one shot - no Python loops over rigs or views.
    def sample_calib_jitter(self) -> None:
        """Draw one SE(3) perturbation per (rig, view) for the current training forward (None outside training)."""
        rot_deg, trans_cm = self.calib_jitter
        if not self.training or (rot_deg <= 0 and trans_cm <= 0):
            self._jitter = None
            return
        dev = self.rig_bank.R.device
        j = torch.zeros(self.max_rigs, self.V, 6, device=dev)
        j[..., :3] = torch.randn(self.max_rigs, self.V, 3, device=dev) * math.radians(rot_deg)
        j[..., 3:] = torch.randn(self.max_rigs, self.V, 3, device=dev) * trans_cm
        self._jitter = j

    # ------------------------------------------------------------------ rig augmentation
    def source_rig_ids(self, rig_ids: torch.Tensor) -> torch.Tensor:
        """Rig ids with every per-sample virtual rig (``rig_augmentation``) replaced by the rig it was derived from."""
        if self._aug_base is None:
            return rig_ids
        virt = rig_ids >= self._aug_base
        return torch.where(virt, self._aug_src[(rig_ids - self._aug_base).clamp_min(0)], rig_ids)

    @contextmanager
    def rig_augmentation(self, rig_ids: torch.Tensor, dR: Optional[torch.Tensor], check_source: bool = True):
        """Geometry of a rotated rig: inside the context the cameras of sample n
        are rotated about their optical centres by dR [n, v] (p_cam_new = dR p_cam_old); yields the rig ids to use
        (virtual per-sample rigs). Projection, rays and centres follow the rotated cameras, the encoder's cached
        geometry is taken from the source rig (gnomonic grid and feature LUT depend on the intrinsics only) and its
        device-frame ray directions are rotated by R^T dR^T R. The caller rotates ``origin_cam`` (origin' = dR origin)
        and warps the images when the augmentation is geometry-consistent. ``dR`` None: no-op."""
        if dR is None:
            yield rig_ids
            return
        orig_bank, base = self.rig_bank, self.rig_bank.num_rigs
        self.rig_bank, virt = orig_bank.augmented(rig_ids, dR, check_source=check_source)
        with torch.autocast(device_type=dR.device.type, enabled=False):
            R0 = orig_bank.R.index_select(0, rig_ids).float()                          # [n, V, 3, 3] target -> camera
            self._aug_ray_rot = R0.transpose(-1, -2) @ dR.float().transpose(-1, -2) @ R0  # d_dev' = R^T dR^T R d_dev
        self._aug_base, self._aug_src = base, rig_ids
        try:
            yield virt
        finally:
            self.rig_bank = orig_bank
            self._aug_base = self._aug_src = self._aug_ray_rot = None

    def extrinsic_delta(self, rig_ids: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """(dR [N, V, 3, 3], dt [N, V, 3]) composing the learnable refinement and the sampled jitter; None when idle."""
        rig_ids = self.source_rig_ids(rig_ids)
        parts = []
        if self.extr_delta is not None:
            parts.append(self.extr_delta.index_select(0, rig_ids))
        if self._jitter is not None:
            parts.append(self._jitter.index_select(0, rig_ids))
        if not parts:
            return None
        dR, dt = None, None
        for xi in parts:  # apply the learned refinement first, then the jitter: p' = Rj (Rl p + tl) + tj
            R_i, t_i = so3_exp(xi[..., :3].float()), xi[..., 3:].float()
            if dR is None:
                dR, dt = R_i, t_i
            else:
                dt = torch.einsum("nvij,nvj->nvi", R_i, dt) + t_i
                dR = R_i @ dR
        return dR, dt

    def project(self, points: torch.Tensor, origin_cam: torch.Tensor, rig_ids: torch.Tensor, rot: Optional[torch.Tensor] = None):
        """points [N, M, 3] target frame; origin_cam [N, V, 1, 3]; rig_ids [N]; rot [N, 3, 3] (target -> device, optional)
        -> uv [N, V, M, 2] in [0,1], in_fov [N, V, M]."""
        with torch.autocast(device_type=points.device.type, enabled=False):
            return self.rig_bank.project(points.float(), origin_cam.float(), rig_ids, rot, delta=self.extrinsic_delta(rig_ids))

    def rays(self, uv: torch.Tensor, origin_cam: torch.Tensor, rig_ids: torch.Tensor, rot: Optional[torch.Tensor] = None):
        """uv [N, V, M, 2] normalised -> ray origins o [N, V, 1, 3] and unit directions d [N, V, M, 3] (target frame)."""
        with torch.autocast(device_type=uv.device.type, enabled=False):
            return self.rig_bank.rays(uv.float(), origin_cam.float(), rig_ids, rot, delta=self.extrinsic_delta(rig_ids))

    def center(self, origin_cam: torch.Tensor, rig_ids: torch.Tensor, rot: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Device centre (mean of the camera centres) in the target frame -> [N, 1, 3]."""
        with torch.autocast(device_type=origin_cam.device.type, enabled=False):
            return self.rig_bank.centers(origin_cam.float(), rig_ids, rot, delta=self.extrinsic_delta(rig_ids)).mean(1, keepdim=True)

    def focal_px(self, rig_ids: torch.Tensor) -> torch.Tensor:
        """Pixels per radian near the image centre for every (sample, view) -> [N, V]: |d rho / d theta| of the
        Scaramuzza W2C polynomial at the optical axis (theta = -pi/2 in this parametrisation), fx for OpenCV fisheye."""
        rb = self.rig_bank
        w2c = rb.w2c.index_select(0, rig_ids).float()                                         # [N, V, D]
        theta0 = torch.full_like(w2c[..., 0], -math.pi / 2)
        k = torch.arange(1, w2c.shape[-1], device=w2c.device, dtype=w2c.dtype)
        dpoly = (w2c[..., 1:] * k * theta0[..., None] ** (k - 1)).sum(-1).abs()                # |sum_k k a_k theta^(k-1)|
        fx = rb.K4.index_select(0, rig_ids)[..., 0].float()
        f = torch.where(rb.model.index_select(0, rig_ids) == MODEL_SCARAMUZZA, dpoly, fx)
        return f.clamp_min(1.0)

    def angular_sigma(self, rig_ids: torch.Tensor, feat_w: int) -> torch.Tensor:
        """Angular uncertainty of half a feature pixel per (sample, view) -> [N, V, 1] rad: 0.5 * calib width / feat_w
        / focal (replaces the former 1024 px / FOCAL_PX constant, which was wrong for the 872 px UnrealEgo-RW
        calibration and for any other rig)."""
        with torch.autocast(device_type=rig_ids.device.type, enabled=False):
            width = self.rig_bank.size.index_select(0, rig_ids)[..., 0].float()               # [N, V] calibration px
            return (0.5 * width / float(feat_w) / self.focal_px(rig_ids))[..., None]

    def _pos_embed(self, h: int, w: int, device) -> torch.Tensor:
        key = (h, w, str(device))
        if key not in self._pos_cache:
            self._pos_cache[key] = sine_pos_embed_2d(h, w, self.C, device=device)[None, None]  # [1, 1, hw, C]
        return self._pos_cache[key]

    # ------------------------------------------------------------------ encoder
    def encode(self, img: torch.Tensor, rig_ids_bt: torch.Tensor):
        """img [N, V, 3, H, W] -> feats [N, V, C, h, w], heatmaps [N, V, 15, h, w] or None."""
        N, V, _, H, W = img.shape
        geom = self._gather_geometry(rig_ids_bt, H, W)
        raw = self.encoder(img.flatten(0, 1), geom)  # [N*V, Ce, h, w]
        hm = self.heatmap_head(raw) if self.heatmap_head is not None else None
        feats = self.feat_proj(raw)
        feats = feats.reshape(N, V, *feats.shape[1:])
        hm = hm.reshape(N, V, *hm.shape[1:]) if hm is not None else None
        return feats, hm

    # ------------------------------------------------------------------ tokens
    def _init_tokens(self, N: int, dataset_ids: Optional[torch.Tensor], track_pose: Optional[torch.Tensor],
                     track_logstd: Optional[torch.Tensor], device, aux_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        joints = self.joint_embed.weight[None].expand(N, -1, -1)
        body = self.body_token.expand(N, -1, -1)
        if self.dataset_embed is not None and dataset_ids is not None:
            body = body + self.dataset_embed(dataset_ids)[:, None]
        if self.use_aux_cond:  # headset / gravity conditioning; NaN rows (no headset pose) -> learned null token
            if aux_cond is None:
                body = body + self.cond_null.expand(N, -1, -1)
            else:
                ok = torch.isfinite(aux_cond).all(dim=-1)                                       # [N]
                c = self.cond_mlp(torch.nan_to_num(aux_cond, nan=0.0).to(body.dtype))[:, None]  # [N, 1, C]
                body = body + torch.where(ok[:, None, None], c, self.cond_null.expand(N, -1, -1))
        toks = [joints, body]
        if self.track_prior:
            if track_pose is None:
                tr = self.track_null.expand(N, -1, -1)
            else:
                valid = torch.isfinite(track_pose).all(dim=(-1, -2))  # [N]
                tp = torch.nan_to_num(track_pose, nan=0.0) / 100.0
                inp = tp.reshape(N, -1)
                if self.track_uses_unc:  # log-std channel: the training constant unless the caller provides one
                    ls = track_logstd if track_logstd is not None else tp.new_full((N, self.J), self.track_ls_const)
                    inp = torch.cat([inp, torch.nan_to_num(ls, nan=self.track_ls_const).reshape(N, -1).to(inp.dtype)], dim=-1)
                tr = self.track_mlp(inp)[:, None]
                tr = torch.where(valid[:, None, None], tr, self.track_null.expand(N, -1, -1))
            toks.append(tr)
        return torch.cat(toks, dim=1)  # [N, J+E, C]

    def _conf(self, hm: Optional[torch.Tensor], uv: torch.Tensor, in_fov: torch.Tensor) -> Optional[torch.Tensor]:
        if hm is None:
            return None
        conf = in_fov.to(torch.float32).clone()  # joints without a heatmap channel (head) -> in_fov
        ids = self.heatmap_joint_ids
        vals = sample_heatmap_at(hm.detach().float(), uv[:, :, ids].float())
        conf[:, :, ids] = vals.clamp(0.0, 1.0)
        return conf

    # ------------------------------------------------------------------ decoders
    def _run_proposal(self, x, feats, kin_bias, temporal_fn):
        """Returns tokens, proposal [N, J, 3] (cm) and its per-joint log-std [N, J]."""
        N = x.shape[0]
        if self.p0 is not None:  # single-stage: learned initial reference pose, no image cross-attention
            prop = self.p0[None].expand(N, -1, -1).float()
            return x, prop, prop.new_full((N, self.J), LOG_REF_STD)
        _, V, C, h, w = feats.shape
        f = feats.flatten(3).transpose(2, 3)  # [N, V, hw, C]
        pos = self._pos_embed(h, w, feats.device)
        for layer in self.proposal_layers:
            x = layer(x, f, pos, kin_bias, temporal_fn)
        proposal = self.proposal_head(x[:, : self.J]).float()  # [N, J, 3] (cm); poses are kept in float32
        # the proposal has no uncertainty head (it would receive no gradient, see __init__): constant reference log-std
        return x, proposal, proposal.new_full((N, self.J), LOG_REF_STD, dtype=torch.float32)

    def _run_refine(self, x, anchors, prev_logstd, feats, hm, origin_cam, rig_ids, kin_bias, temporal_fn,
                    past_builder_factory=None, rot=None, bone_len=None, gt=None):
        """Iterative refinement.

        ``past_builder_factory(stage_idx, prev_pred, prev_logstd, value) -> PastFrames | None`` provides the P3 past
        frames of the previous decoder stage (proposal for the first layer). ``rot`` [N, 3, 3] is the optional
        target -> device rotation (gravity-aligned target frame). Returns per-layer poses [N, J, 3], final tokens and
        a list of per-layer auxiliary dicts.
        """
        preds, auxs = [], []
        anchors = anchors.float()
        x = torch.cat([x[:, : self.J] + self.anchor_embed((anchors / 100.0).to(x.dtype)), x[:, self.J:]], dim=1)
        cur = anchors
        prev_pred, prev_ls = anchors, prev_logstd
        center = self.center(origin_cam, rig_ids, rot)
        project_fn = lambda pts: self.project(pts, origin_cam, rig_ids, rot)  # noqa: E731
        rays_fn = lambda uv: self.rays(uv, origin_cam, rig_ids, rot)  # noqa: E731
        ang_sigma = self.angular_sigma(rig_ids, feats.shape[-1]) if self.geo_branch else None
        for i, layer in enumerate(self.refine_layers):
            ref = cur if self.iter_reproj else anchors
            base = cur if self.refine_mode == "residual" else anchors
            uv, fov = self.project(ref.detach(), origin_cam, rig_ids, rot)
            conf = self._conf(hm, uv, fov)
            past_builder = None
            if past_builder_factory is not None and layer.traj is not None:
                past_builder = lambda value, _i=i, _p=prev_pred, _l=prev_ls: past_builder_factory(_i, _p, _l, value)  # noqa: E731
            inp = RefineInputs(feats=feats, ref=ref.detach(), base=base, uv_ref=uv, fov_ref=fov, conf_hm=conf, center=center,
                               project_fn=project_fn, rays_fn=rays_fn, kin_bias=kin_bias, temporal_fn=temporal_fn,
                               project_with_origin=self.project, past_builder=past_builder, hm=hm, bone_len=bone_len, gt=gt,
                               angular_sigma=ang_sigma,
                               diag=self.diag_probe is not None and i == len(self.refine_layers) - 1)
            x, pose, aux = layer(x, inp)
            aux.update(uv_ref=uv, fov_ref=fov, conf_hm=conf)
            cur = pose.float()
            preds.append(cur)
            auxs.append(aux)
            prev_pred, prev_ls = cur, aux["logstd"]
        return preds, x, auxs

    # ------------------------------------------------------------------ forward (parallel over T)
    def map_joints(self, pose: torch.Tensor, skeleton_ids: Optional[torch.Tensor]) -> torch.Tensor:
        """Canonical-skeleton pose [B, T, J, 3] -> per-sample output skeleton [B, T, J_max, 3] (NaN-padded)."""
        if self.joint_mapper is None or skeleton_ids is None:
            return pose
        return self.joint_mapper(pose, skeleton_ids)

    def forward(
        self,
        img: torch.Tensor,
        origin_cam: torch.Tensor,
        rig_ids: torch.Tensor,
        dataset_ids: Optional[torch.Tensor] = None,
        prev_pose: Optional[torch.Tensor] = None,
        gt_pose: Optional[torch.Tensor] = None,
        target_rot: Optional[torch.Tensor] = None,
        aux_cond: Optional[torch.Tensor] = None,
        skeleton_ids: Optional[torch.Tensor] = None,
        track_clean: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """img [B,T,V,3,H,W]; origin_cam [B,T,V,1,3]; rig_ids/dataset_ids/skeleton_ids [B]; prev_pose/gt_pose [B,T,J,3]
        (cm, canonical skeleton); target_rot [B,T,3,3] target -> device (optional); aux_cond [B,T,AUX_COND_DIM] (optional);
        track_clean [B] bool: clips whose prev_pose is the model's own streamed prediction (no synthetic noise added)."""
        self.sample_calib_jitter()  # one calibration perturbation per rig for this forward (training only)
        try:
            return self._forward(img, origin_cam, rig_ids, dataset_ids, prev_pose, gt_pose, target_rot, aux_cond, skeleton_ids,
                                 track_clean)
        finally:
            self._jitter = None

    def _correlated_track_noise(self, prev_pose: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        """prev_pose [B, T, J, 3], center [B*T, 1, 3] (device centre in the target frame) -> noise [B, T, J, 3] (cm).

        Half of the variance is a per-clip per-joint offset, half an AR(1) process (rho 0.8) over the clip; the per-axis
        sigma is track_prior_noise_cm * (0.5 + distance / 1 m): ~0.6x at the head, ~2x at the feet."""
        B, T, J = prev_pose.shape[:3]
        dist = (prev_pose.reshape(B * T, J, 3) - center).norm(dim=-1).nan_to_num(100.0).reshape(B, T, J)
        sigma = self.track_noise * (0.5 + dist / 100.0)
        rho = 0.8
        ar = [torch.randn(B, J, 3, device=prev_pose.device)]
        for _ in range(1, T):
            ar.append(rho * ar[-1] + math.sqrt(1.0 - rho * rho) * torch.randn(B, J, 3, device=prev_pose.device))
        unit = math.sqrt(0.5) * torch.randn(B, 1, J, 3, device=prev_pose.device) + math.sqrt(0.5) * torch.stack(ar, 1)
        return unit * sigma[..., None]

    def _forward(self, img, origin_cam, rig_ids, dataset_ids, prev_pose, gt_pose, target_rot, aux_cond, skeleton_ids,
                 track_clean=None) -> Dict[str, torch.Tensor]:
        B, T = img.shape[:2]
        N = B * T
        rig_bt = rig_ids[:, None].expand(B, T).reshape(N)
        ds_bt = dataset_ids[:, None].expand(B, T).reshape(N) if dataset_ids is not None else None
        rot_bt = target_rot.reshape(N, 3, 3) if target_rot is not None else None
        cond_bt = aux_cond.reshape(N, -1) if aux_cond is not None else None
        feats, hm = self.encode(img.flatten(0, 1), rig_bt)
        if self.tfm is not None:
            feats = self.tfm(feats.reshape(B, T, *feats.shape[1:])).reshape(N, *feats.shape[1:])
        origin_bt = origin_cam.reshape(N, self.V, 1, 3)
        kin_bias = self.kin_bias() if self.kin_bias is not None else None

        def temporal_fn(module: CausalTemporalAttention, x: torch.Tensor) -> torch.Tensor:
            return module(x.reshape(B, T, *x.shape[1:])).reshape(N, *x.shape[1:])

        past_factory = None
        if self.traj_window > 0:
            center_bt = self.center(origin_bt, rig_bt, rot_bt)

            def past_factory(stage, prev_pred, prev_ls, value):  # noqa: F811
                sigma, p = self.traj_anchor_noise
                if self.training and sigma > 0 and p > 0:  # realistic errors on the past anchors
                    hit = (torch.rand(prev_pred.shape[:2], device=prev_pred.device) < p).float()[..., None]
                    prev_pred = prev_pred + hit * torch.randn_like(prev_pred) * sigma
                return build_past_parallel(B, T, self.traj_window, prev_pred, prev_ls, value, center_bt, origin_bt, rig_bt, rot_bt)

        track, track_ls = None, None
        if self.track_prior and prev_pose is not None:
            track = prev_pose.reshape(N, self.J, 3)
            if self.training:  # teacher forcing: GT previous pose + noise; the log-std channel states the noise level
                if self.track_noise_sample:
                    lo, hi = self.track_noise_range
                    track_ls = torch.rand(N, self.J, device=track.device) * (hi - lo) + lo          # log sigma per joint
                    noise = torch.randn_like(track) * track_ls.exp()[..., None]
                elif self.track_noise_mode == "correlated":
                    center = self.center(origin_bt, rig_bt, rot_bt)
                    noise = self._correlated_track_noise(prev_pose, center).reshape(N, self.J, 3)
                    track_ls = track.new_full((N, self.J), self.track_ls_const)
                else:
                    noise = torch.randn_like(track) * self.track_noise
                    track_ls = track.new_full((N, self.J), self.track_ls_const)
                if track_clean is not None:  # own streamed predictions already carry realistic errors
                    noise = noise * (~track_clean)[:, None].expand(B, T).reshape(N, 1, 1).to(noise.dtype)
                track = track + noise
                drop = torch.rand(N, device=track.device) < self.track_dropout
                track = torch.where(drop[:, None, None], torch.full_like(track, float("nan")), track)

        x = self._init_tokens(N, ds_bt, track, track_ls, img.device, cond_bt)
        x, proposal, prop_ls = self._run_proposal(x, feats, kin_bias, temporal_fn)
        anchors = proposal if self.e2e_grad else proposal.detach()
        anchors = self.init_anchors(anchors, hm, origin_bt, rig_bt, rot_bt)   # (identity for "learned")
        bone, bone_mem = None, None
        if self.kin_prior:  # P5: per-frame bone lengths from the body token, causal running mean as the prior
            bone = self.bone_lengths(x)
            bone_mem = self._bone_memory_parallel(bone, B, T)
        gt_bt = gt_pose.reshape(N, self.J, 3) if (self.training and gt_pose is not None and gt_pose.shape[-2] == self.J) else None
        preds, x_final, auxs = self._run_refine(x, anchors, prop_ls, feats, hm, origin_bt, rig_bt, kin_bias, temporal_fn, past_factory, rot_bt,
                                                bone_len=bone_mem, gt=gt_bt)

        def bt(t: torch.Tensor, *tail) -> torch.Tensor:
            return t.reshape(B, T, *tail)

        out: Dict[str, torch.Tensor] = {
            "proposal": bt(proposal, self.J, 3),
            "refined": [bt(p, self.J, 3) for p in preds],
            "pred": bt(preds[-1], self.J, 3),
            "anchors_2d": [(bt(a["uv_ref"], self.V, self.J, 2), bt(a["fov_ref"], self.V, self.J)) for a in auxs],
        }
        if bone is not None:
            out["bone_len"], out["bone_mem"] = bt(bone, -1), bt(bone_mem, -1)
            if "bone_gate" in auxs[-1]:
                out["bone_gates"] = [bt(a["bone_gate"], self.J) for a in auxs]
                out["bone_pred"] = bt(auxs[-1]["bone_pred"], self.J, 3)
        if "raydn_logits" in auxs[-1]:  # hard-negative logits per layer [B, T, J, 1 + Kn] + validity [B, T, J]
            out["raydn_logits"] = [bt(a["raydn_logits"], self.J, -1) for a in auxs if "raydn_logits" in a]
            out["raydn_valid"] = [bt(a["raydn_valid"], self.J) for a in auxs if "raydn_valid" in a]
        if hm is not None:
            out["heatmaps"] = bt(hm, self.V, *hm.shape[2:])
        if "unc_raw" in auxs[-1]:
            layers = [l for l, a in zip(self.refine_layers, auxs) if "unc_raw" in a]
            raws = [a["unc_raw"] for a in auxs if "unc_raw" in a]
            rots = [a.get("unc_rot") for a in auxs if "unc_raw" in a]
            out["uncertainty"] = bt(raws[-1], self.J, 6)
            chols = []
            for l, r, R in zip(layers, raws, rots):  # ray-local factors are rotated into the target frame
                Lc = l.unc_head.cholesky(r)
                chols.append(bt(l.unc_head.world_cholesky(Lc, R) if R is not None else Lc, self.J, 3, 3))
            out["uncertainty_chol_layers"] = chols
            out["uncertainty_chol"] = out["uncertainty_chol_layers"][-1]
            if rots[-1] is not None:
                out["uncertainty_rot"] = bt(rots[-1], self.J, 3, 3)
                out["uncertainty_local_std"] = bt(layers[-1].unc_head.local_std(raws[-1]), self.J, 3)
        if "geo_pred" in auxs[-1]:
            out["geo_pred"] = [bt(a["geo_pred"], self.J, 3) for a in auxs]
            out["gates"] = [bt(a["gate"], self.J) for a in auxs]
            out["res2d"] = [bt(a["uv_res"], self.V, self.J, 2) for a in auxs]
            out["view_conf"] = [bt(a["view_conf"], self.V, self.J) for a in auxs]
            out["geo_info"] = [bt(a["geo_info"], self.J) for a in auxs]       # geometric-loss masks / gate statistics
            out["anchor_dist"] = bt(auxs[-1]["anchor_dist"], self.J)
        if auxs[-1].get("conf_hm") is not None:  # heatmap confidence at the final anchors, max over views [B, T, J]
            out["hm_conf"] = bt(auxs[-1]["conf_hm"].float().amax(dim=1), self.J)
        out.update(self._diag_outputs(auxs[-1], lambda t, *tail: bt(t, *tail)))
        if "geo_pred" in out:  # last-layer aliases of the per-layer lists (same keys as the streaming path)
            out["geo_pred_last"], out["gate"], out["geo_info_last"] = out["geo_pred"][-1], out["gates"][-1], out["geo_info"][-1]
        if "vis_logit" in auxs[-1]:
            out["vis_logits"] = [bt(a["vis_logit"], self.V, self.J) for a in auxs if "vis_logit" in a]
            out["visibility"] = torch.sigmoid(out["vis_logits"][-1].float())
        if self.joint_mapper is not None and skeleton_ids is not None:
            out["pred_mapped"] = self.map_joints(out["pred"], skeleton_ids)
            out["refined_mapped"] = [self.map_joints(p, skeleton_ids) for p in out["refined"]]
            out["proposal_mapped"] = self.map_joints(out["proposal"], skeleton_ids)

        if self.training and self.denoise_groups > 0 and gt_pose is not None and gt_pose.shape[-2] == self.J:
            # denoising queries per *frame*: frames without a complete GT pose (unlabeled UDA clips, other skeletons,
            # mocap drop-outs) are masked out of the DN loss via ``dn_valid`` instead of disabling DN for the batch
            gt_all = gt_pose.reshape(N, self.J, 3)
            dn_ok = torch.isfinite(gt_all).flatten(1).all(-1)                              # [N]
            if bool(dn_ok.any()):
                gt_fill = torch.nan_to_num(gt_all)
                dn_preds, dn_targets = [], []
                for _ in range(self.denoise_groups):
                    noised = gt_fill + torch.randn_like(gt_fill) * self.denoise_std
                    ls = noised.new_full((N, self.J), math.log(self.denoise_std), dtype=torch.float32)
                    x0 = self._init_tokens(N, ds_bt, track, track_ls, img.device, cond_bt)
                    p_dn, _, _ = self._run_refine(x0, noised, ls, feats, hm, origin_bt, rig_bt, kin_bias, temporal_fn, past_factory, rot_bt,
                                                  bone_len=bone_mem.detach() if bone_mem is not None else None)
                    dn_preds.append([bt(p, self.J, 3) for p in p_dn])
                    dn_targets.append(gt_pose)
                out["dn_refined"] = dn_preds
                out["dn_targets"] = dn_targets
                out["dn_valid"] = bt(dn_ok)                                                    # [B, T]
        return out

    _DIAG_KEYS = {"learn_pred": "learn_pred_last", "geo_cand": "geo_cand_last", "anchor": "anchor_last", "center": "center_last",
                  "n_valid": "n_valid_last", "ray_cos": "ray_cos_last", "conf_min": "conf_min_last", "geo_eig": "geo_eig_last",
                  "disp_rate": "disp_rate_last", "hyp_min": "hyp_min_last", "hyp_max": "hyp_max_last",
                  "gate_par": "gate_par_last", "gate_perp": "gate_perp_last", "depth_unit": "depth_unit_last"}

    def _diag_outputs(self, aux: Dict[str, torch.Tensor], reshape: Callable) -> Dict[str, torch.Tensor]:
        """Last-layer diagnostics (error / gate analysis): [N, ...] tensors of ``aux`` -> [B, T, ...] via ``reshape``."""
        out = {}
        for src, dst in self._DIAG_KEYS.items():
            if src in aux:
                t = aux[src].detach().float()
                out[dst] = reshape(t, *t.shape[1:])
        return out

    def _probe_step(self, aux: Dict[str, torch.Tensor], hm: Optional[torch.Tensor], origin_cam: torch.Tensor,
                    rig_ids: torch.Tensor, rot: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Read-only probe of the last refinement layer for one streamed frame (``diag_probe`` = (extent, step) cm).

        Points on three lines through the layer's anchor - its viewing ray from the device centre and the two lateral axes
        of the same ray-local basis as the P2 offsets - at -extent..+extent cm in steps of ``step``. For every point: the
        left-right cosine of the layer's value maps (all heads' channels) at its projections and the summed heatmap value of
        the joint over the views that see it. Also returns the layer's sampled hypotheses with their projections. Nothing
        here feeds back into the model."""
        extent, step = self.diag_probe
        a, c = aux["anchor"].float(), aux["center"].float()                               # [B, J, 3], [B, 1, 3]
        B, J = a.shape[:2]
        r, e1, e2 = ray_local_frame(a, c)
        axes = torch.stack([r, e1, e2], dim=2)                                             # [B, J, 3 (line), 3]
        offs = torch.arange(-extent, extent + 0.5 * step, step, device=a.device, dtype=torch.float32)
        S = offs.numel()
        pts = a[:, :, None, None] + offs[None, None, None, :, None] * axes[:, :, :, None]  # [B, J, 3, S, 3]
        uv, fov = self.project(pts.reshape(B, J * 3 * S, 3), origin_cam, rig_ids, rot)    # [B, V, M, 2], [B, V, M]
        V = uv.shape[1]
        grid = (2.0 * uv - 1.0).reshape(B * V, 1, -1, 2)
        value = aux["p2_value"].float().flatten(2, 3)                                      # [B, V, C, h, w]
        C, h, w = value.shape[2:]
        s = F.grid_sample(value.reshape(B * V, C, h, w), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        s = F.normalize(s.reshape(B, V, C, -1).permute(0, 1, 3, 2), dim=-1)                # [B, V, M, C]
        pairs = [(i, k) for i in range(V) for k in range(i + 1, V)]
        cos = torch.stack([(s[:, i] * s[:, k]).sum(-1) for i, k in pairs], 0).mean(0)       # [B, M]
        hm_score = torch.full((B, J, 3, S), float("nan"), device=a.device)
        if hm is not None:
            ids = self.heatmap_joint_ids
            hh, hw = hm.shape[-2:]
            g = grid.reshape(B, V, J, 3 * S, 2)[:, :, ids].reshape(B * V * len(ids), 1, 3 * S, 2)
            vals = F.grid_sample(hm.float().reshape(B * V * len(ids), 1, hh, hw), g, mode="bilinear", padding_mode="zeros",
                                 align_corners=False).reshape(B, V, len(ids), 3, S)
            seen = fov.reshape(B, V, J, 3, S)[:, :, ids].to(vals.dtype)
            hm_score[:, ids] = (vals * seen).sum(1)
        return {"probe_cos_last": cos.reshape(B, J, 3, S), "probe_hm_last": hm_score,
                "probe_fov_last": fov.reshape(B, V, J, 3, S), "probe_axes_last": axes,
                "cand_points_last": aux["p2_points"].float(), "cand_uv_last": aux["p2_uv"].float(), "cand_fov_last": aux["p2_fov"]}

    def _streamed(self, pose: torch.Tensor, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        """The pose a refinement layer leaves in the streaming state (track prior, P3 history): its output, or - with
        geo_gate_eval_override_feedback 'original' - the learned-gate pose that the override replaced in the output."""
        if self.override_feedback == "original" and "pose_unmodified" in aux:
            return aux["pose_unmodified"]
        return pose

    # ------------------------------------------------------------------ streaming inference
    @torch.no_grad()
    def forward_streaming(
        self,
        img: torch.Tensor,
        origin_cam: torch.Tensor,
        rig_ids: torch.Tensor,
        dataset_ids: Optional[torch.Tensor] = None,
        state: Optional[Dict] = None,
        target_rot: Optional[torch.Tensor] = None,
        aux_cond: Optional[torch.Tensor] = None,
        skeleton_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """Causal frame-by-frame decoding with KV caches (features of the whole clip are computed at once).

        ``state`` carries the temporal-attention KV caches, the TFM caches, the feature / anchor histories used by
        the trajectory sampler, the step counter and the previous prediction (track prior) across calls, so
        consecutive clips of one sequence can be streamed. Numerically equivalent to ``forward``.
        """
        B, T = img.shape[:2]
        N = B * T
        rig_bt = rig_ids[:, None].expand(B, T).reshape(N)
        feats_all, hm_all = self.encode(img.flatten(0, 1), rig_bt)
        feats_all = feats_all.reshape(B, T, *feats_all.shape[1:])
        hm_all = hm_all.reshape(B, T, *hm_all.shape[1:]) if hm_all is not None else None
        kin_bias = self.kin_bias() if self.kin_bias is not None else None

        n_layers = len(self.refine_layers)
        w = max(self.traj_window, 1)
        state = state or {
            "step": 0, "caches": {}, "tfm": {}, "prev": None, "prev_logstd": None,
            # P3 histories (newest last): camera origins / device centres / target rotations per frame, per-stage
            # anchors + log-std (stage 0 = proposal, stage i+1 = refine layer i) and per-layer cached value projections
            "origin_hist": deque(maxlen=w), "center_hist": deque(maxlen=w), "rot_hist": deque(maxlen=w),
            "anchor_hist": [deque(maxlen=w) for _ in range(n_layers + 1)],
            "value_hist": [deque(maxlen=w + 1) for _ in range(n_layers)],  # + the current frame's value
        }
        caches: Dict[str, Dict] = state["caches"]

        preds, props, chols, viss = [], [], [], []
        gates, geos, infos, hmc = [], [], [], []   # last-layer gate statistics for the validation analysis
        bgs, bms = [], []                          # P5 bone gate / memorised bone lengths per step
        lstd = []                                  # ray-local stds (along ray, e1, e2) per step
        diags: Dict[str, List[torch.Tensor]] = {}  # last-layer diagnostics per step (error / gate analysis)
        for t in range(T):
            step = state["step"]
            if getattr(self, "_decoder_compiled", False):  # CUDA graphs (compile_decoder): new replay generation per step
                torch.compiler.cudagraph_mark_step_begin()
            feats_t = feats_all[:, t]
            if self.tfm is not None:
                feats_t = self.tfm.forward_step(feats_t, state["tfm"])
            hm_t = hm_all[:, t] if hm_all is not None else None
            origin_t = origin_cam[:, t]
            rot_t = target_rot[:, t] if target_rot is not None else None
            cond_t = aux_cond[:, t] if aux_cond is not None else None

            def temporal_fn(module: CausalTemporalAttention, x: torch.Tensor, _step=step) -> torch.Tensor:
                cache = caches.setdefault(str(id(module)), {})
                return module.forward_step(x, cache, _step)

            past_factory = None
            if self.traj_window > 0:
                origins = list(state["origin_hist"])[::-1]                  # newest first: age 1, 2, ...
                centers = list(state["center_hist"])[::-1]
                rots = list(state["rot_hist"])[::-1]

                def past_factory(stage, prev_pred, prev_ls, value, _o=origins, _c=centers, _r=rots):  # noqa: F811
                    # cache this frame's value projection of the stage for the next steps, then batch the history
                    state["value_hist"][stage].append(value)
                    anchors_ls = list(state["anchor_hist"][stage])[::-1]
                    values = list(state["value_hist"][stage])[::-1][1:]      # exclude the value just appended (current frame)
                    return PastFrames.from_history([a for a, _ in anchors_ls], [l for _, l in anchors_ls], values, _c, _o, rig_ids, _r)

            prev_ls = state["prev_logstd"]
            if prev_ls is not None and self.track_logstd_at_test == "train_constant":
                prev_ls = torch.full_like(prev_ls, self.track_ls_const)   # what the token saw in training
            x = self._init_tokens(B, dataset_ids, state["prev"], prev_ls, img.device, cond_t)
            x, proposal, prop_ls = self._run_proposal(x, feats_t, kin_bias, temporal_fn)
            bone_mem = None
            if self.kin_prior:  # P5: running mean of the per-frame bone lengths over the whole streamed sequence
                bone = self.bone_lengths(x)
                if self.bone_memory == "causal_mean":
                    state["bone_sum"] = bone if state.get("bone_sum") is None else state["bone_sum"] + bone
                    state["bone_cnt"] = state.get("bone_cnt", 0) + 1
                    bone_mem = state["bone_sum"] / state["bone_cnt"]
                else:
                    bone_mem = bone
            anchors_t = self.init_anchors(proposal, hm_t, origin_t, rig_ids, rot_t)
            p, _, auxs = self._run_refine(x, anchors_t, prop_ls, feats_t, hm_t, origin_t, rig_ids, kin_bias, temporal_fn, past_factory, rot_t,
                                          bone_len=bone_mem)
            preds.append(p[-1])
            props.append(proposal)
            if "unc_raw" in auxs[-1]:
                head = self.refine_layers[-1].unc_head
                Lc = head.cholesky(auxs[-1]["unc_raw"])
                chols.append(head.world_cholesky(Lc, auxs[-1]["unc_rot"]) if "unc_rot" in auxs[-1] else Lc)
                if "unc_rot" in auxs[-1]:
                    lstd.append(head.local_std(auxs[-1]["unc_raw"]))
            if "vis_logit" in auxs[-1]:
                viss.append(torch.sigmoid(auxs[-1]["vis_logit"].float()))
            if "gate" in auxs[-1]:
                gates.append(auxs[-1]["gate"].float())
                geos.append(auxs[-1]["geo_pred"].float())
                infos.append(auxs[-1]["geo_info"])
            if auxs[-1].get("conf_hm") is not None:
                hmc.append(auxs[-1]["conf_hm"].float().amax(dim=1))
            if "bone_gate" in auxs[-1]:
                bgs.append(auxs[-1]["bone_gate"].float())
                bms.append(bone_mem)
            for k, v in self._diag_outputs(auxs[-1], lambda t, *tail: t).items():
                diags.setdefault(k, []).append(v)
            if self.diag_probe is not None and "p2_points" in auxs[-1]:
                for k, v in self._probe_step(auxs[-1], hm_t, origin_t, rig_ids, rot_t).items():
                    diags.setdefault(k, []).append(v)
            if self.traj_window > 0:
                state["origin_hist"].append(origin_t)
                state["center_hist"].append(self.center(origin_t, rig_ids, rot_t))
                state["rot_hist"].append(rot_t)
                state["anchor_hist"][0].append((anchors_t.detach(), prop_ls.detach()))
                for i, a in enumerate(auxs):
                    state["anchor_hist"][i + 1].append((self._streamed(p[i], a).detach(), a["logstd"].detach()))
            if self.track_prior:
                state["prev"] = self._streamed(p[-1], auxs[-1])
                state["prev_logstd"] = auxs[-1]["logstd"]
            state["step"] = step + 1
        out = {"pred": torch.stack(preds, 1), "proposal": torch.stack(props, 1)}
        if chols:
            out["uncertainty_chol"] = torch.stack(chols, 1)
        if viss:
            out["visibility"] = torch.stack(viss, 1)
        if gates:  # last layer only ([B, T, J] / [B, T, J, 3]); the training path returns per-layer lists
            out["gate"], out["geo_pred_last"], out["geo_info_last"] = torch.stack(gates, 1), torch.stack(geos, 1), torch.stack(infos, 1)
        if hmc:
            out["hm_conf"] = torch.stack(hmc, 1)
        if bgs:
            out["bone_gate"], out["bone_mem"] = torch.stack(bgs, 1), torch.stack(bms, 1)
        if lstd:
            out["uncertainty_local_std"] = torch.stack(lstd, 1)
        for k, vs in diags.items():
            out[k] = torch.stack(vs, 1)
        if self.joint_mapper is not None and skeleton_ids is not None:
            out["pred_mapped"] = self.map_joints(out["pred"], skeleton_ids)
        return out, state

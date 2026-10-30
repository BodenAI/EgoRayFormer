"""Typed configuration dataclasses.

jsonargparse merges dataclass fields across multiple ``--config`` files, whereas plain ``Dict``
hyper-parameters are replaced wholesale; using dataclasses lets ablation configs override single
fields (e.g. ``model.model.temporal_window``) without repeating the whole block.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class ModelConfig:
    encoder: Dict[str, Any] = field(default_factory=lambda: {"type": "r18_fpn", "model_name": "resnet18", "pretrained": True})
    rigs: List[str] = field(default_factory=lambda: ["unrealego"])
    num_joints: int = 16
    num_views: int = 2
    embed_dims: int = 128
    num_heads: int = 4
    ffn_dims: int = 512
    image_size: int = 256
    proposal_layers: int = 1
    refine_layers: int = 3
    deform_points: int = 16
    e2e_grad: bool = True
    iter_reproj: bool = True
    conf_gate: bool = True
    kin_bias: bool = True
    uncertainty: bool = True
    temporal_window: int = 1
    track_prior: bool = False
    track_prior_dropout: float = 0.3
    track_prior_noise_cm: float = 3.0
    dataset_embed: bool = True
    num_datasets: int = 8
    heatmap_head: bool = True
    denoise_groups: int = 0
    denoise_std_cm: float = 5.0
    dropout: float = 0.0
    to_mm: float = 10.0
    # ---- redesign switches (see models/egorayformer.py) ----
    sampling: str = "dsa"                 # "dsa" (v1 per-view 2D deformable) | "ray_consistent" (P2 shared 3D hypotheses)
    geo_branch: bool = False              # P2 two-ray intersection + uncertainty gate
    geo_gate_init: float = -2.0
    traj_window: int = 0                  # P3 trajectory-aligned past-frame sampling (0 = off)
    temporal_pos: str = "rope"            # "rope" | "age" | "rope+age"
    temporal_mixer: bool = False          # TSR-Ego TFM control
    tfm_blocks: int = 2
    tfm_kernel: int = 3
    tfm_dilations: List[int] = field(default_factory=lambda: [1, 2])
    refine_mode: str = "residual"         # "residual" | "absolute" (TSR-Ego layer-wise full-pose heads)
    refine_layer_order: str = "cross_first"  # "temporal_first": TSR-Ego decoder layer order (temporal -> joint -> stereo -> FFN)
    uncertainty_per_layer: bool = True    # P4
    track_prior_uses_uncertainty: bool = True
    # ---- multi-dataset / multi-device extensions ----
    skeleton: str = "unrealego16"         # canonical skeleton of the joint queries
    output_skeletons: List[str] = field(default_factory=list)  # e.g. ["mo2cap2_15", "momentum26"] (JointMapper)
    aux_cond: bool = False                # headset pose / gravity conditioning of the body token
    visibility_head: bool = False         # per-view visibility logits (also feed the geometry gate)
    traj_last_layer_only: bool = False    # P3 only in the last refinement layer
    # ---- geometrically informative joints, trajectory-anchor noise ----
    geo_info_min_angle_deg: float = 0.0   # geometric losses / statistics only for joints whose rays subtend >= this angle (0 = off)
    geo_info_max_dist_cm: float = 0.0     # ... and whose anchor is closer than this to the device (0 = off)
    gate_extra_inputs: bool = False       # GeoGate: + geo/learned consistency, along-ray sigma, informativeness, depth stats (9 -> 14 inputs)
    # ---- P2 hypotheses / matching ----
    p2_depth_hyp: str = "pm"              # "pm" | "layered" (depth levels incl. the anchor, delta = p2_depth_frac * ref distance)
    p2_depth_frac: float = 0.15
    p2_content_attn: bool = False         # query x point-feature dot product in the point attention
    p2_corr_channel: bool = False         # left-right cosine similarity channel in the per-point view fusion
    # ---- Heatmap local soft-argmax residual ----
    hm_softargmax: bool = False           # blend the 7x7 heatmap soft-argmax estimate into the per-view 2D residual
    hm_softargmax_window: int = 7
    # ---- Calibration robustness ----
    learn_extrinsics: bool = False        # learnable per-rig camera SE(3) refinement (real-device fine-tuning only)
    max_rigs: int = 16
    calib_jitter_rot_deg: float = 0.0     # training-time calibration jitter (std, deg) - 0.5 covers the RW session drift
    calib_jitter_trans_cm: float = 0.0    # ... translation std (cm)
    # ---- P5: kinematic ray geometry + online bone-length memory ----
    kin_prior: bool = False               # ray-sphere candidate (best view's ray x bone-length sphere around the parent) + bone gate
    bone_memory: str = "causal_mean"      # "causal_mean" running mean over the sequence | "frame" per-frame prediction
    # ---- RayDN-style along-ray hard negatives (training only) ----
    raydn_negatives: int = 0              # displaced candidates per joint scored by the P2 matching features (0 = off)
    raydn_depth_frac: float = 0.15
    # ---- Ray-aligned uncertainty (Cholesky in the (r, e1, e2) frame, rotated to the target frame) ----
    unc_ray_aligned: bool = False
    # ---- Geometric proposal initialisation (heatmap peaks -> two-ray intersection) ----
    proposal_init: str = "learned"        # "learned" | "triangulated" | "hybrid"
    proposal_init_conf: float = 0.3
    # ---- P3 exposure bias - noise on the past anchors during training (cross-training-caching stand-in) ----
    traj_anchor_noise_cm: float = 0.0
    traj_anchor_noise_p: float = 0.0
    # ---- track token ----
    # The track token's log-std channel is a constant in training; feeding the predicted log-std at test
    # ("predicted") cost 12-14 mm on every P4 run -> the training constant is fed at test as well by default.
    track_logstd_at_test: str = "train_constant"
    track_noise_sample: bool = False      # training: sigma ~ LogUniform(1, 8) cm per (frame, joint), log sigma in the channel
    track_noise_mode: str = "iid"         # "correlated" = per-clip offset + AR(1), scale growing with distance
    # P4 as a pure side output: no NLL gradient into the trunk, no uncertainty input to the gate / P3 weights
    # (combine with track_prior_uses_uncertainty: false and loss.nll_detach_pose: true)
    unc_detach_tokens: bool = False
    gate_uses_logstd: bool = True
    traj_unc_scale_fixed: bool = False
    # ---- valid views, directional fusion, depth-hypothesis steps, rig augmentation ----
    # P2 / P3 samples of a view in which the hypothesis falls outside the image are replaced by a learnt
    # missing-view vector (the clamped projection used to read the image border), and the left-right correlation of
    # such a pair is zeroed; the geometric correction is already closed without two valid views (gate x enough).
    p2_mask_invalid_views: bool = False
    # Fusion of the triangulated update and the learnt offset - "scalar" (one gate per joint) or "directional" (one
    # gate along the anchor's viewing ray and one for the lateral axes, DirectionalGeoGate with observability inputs)
    geo_fusion: str = "scalar"
    # p2_depth_hyp "jacobian" - along-ray hypotheses in disparity steps of the real fisheye projection at the
    # predicted anchor: one step = p2_disp_step_px feature pixels, clamped to [min, max] cm
    p2_disp_step_px: float = 0.5
    p2_depth_step_min_cm: float = 0.5
    p2_depth_step_max_cm: float = 15.0
    # inference only: >= 0 replaces the gate by this constant where the intersection is valid
    geo_gate_eval_override: float = -1.0
    geo_gate_eval_override_layers: str = "all"         # "last": only the last refinement layer
    geo_gate_eval_override_feedback: str = "modified"  # "original": the streamed track prior keeps the learned gate's pose
    # False keeps every P2 offset (lateral and along the ray) in cm whatever the anchor distance
    p2_depth_scaled: bool = True
    # "fixed" step-rule control: False keeps only the along-ray offsets in cm; the lateral ones stay distance-scaled
    # as in the linear and jacobian variants, so the three differ only in the along-ray step rule
    p2_depth_along_scaled: bool = True
    # A projection outside the image circle is outside the view (invalid-view masks, gate view count, warped
    # labels); required by the rig augmentation, whose warped images are black outside the circle
    fov_circle: bool = False


@dataclass
class LossConfig:
    w_mpjpe: float = 1.0
    w_proposal: float = 1.0
    w_aux_layers: float = 1.0
    w_nll: float = 1.0
    w_nll_aux: float = 0.0        # Student-t NLL on intermediate layers (P4 per-layer uncertainty)
    nll_nu: float = 3.0
    nll_warmup_frac: float = 0.5
    # ---- P4 loss variants ----
    nll_type: str = "student_t"   # "student_t" (EPFv2) | "beta_gauss" (beta-NLL, Seitzer et al. 2022)
    nll_beta: float = 0.5         # beta of the beta-NLL (0 = plain Gaussian NLL, 1 = MSE-like weighting of the mean)
    nll_detach_pose: bool = False # NLL trains the uncertainty head only (pose detached): uncertainty as a pure side output
    # frames of held-out training sequences (data.unc_holdout_frac > 0) carry only the NLL (pose detached), never the
    # pose losses, so the uncertainty head sees realistic (not over-fitted) errors - a single-run stand-in for FRAME's
    # cross-training caching
    w_heatmap: float = 10.0
    w_reproj: float = 0.0
    w_res2d: float = 0.0          # P2: L1 (px) between the predicted per-view 2D residual points and GT 2D
    w_geo: float = 0.0            # P2: MPJPE of the triangulated pose on joints visible in both views
    w_bone: float = 0.0
    bone_sym: float = 0.5
    w_vel: float = 0.0
    w_jerk: float = 0.0
    w_dn: float = 1.0
    w_vis: float = 0.0            # BCE of the per-view visibility head against the projected-visibility labels
    vis_pseudo_from_heatmap: bool = False  # occlusion pseudo-labels from the heatmap response at the GT 2D joint
    vis_pseudo_thresh: float = 0.3
    w_bone_len: float = 0.0       # P5: L1 (cm) between the predicted per-frame bone lengths and the GT bone lengths
    w_raydn: float = 0.0          # cross-entropy of the GT candidate against the along-ray hard negatives
    # ---- Self-supervision on unlabeled frames (data sources with ``unlabeled: true``) ----
    w_ssl_geo: float = 0.0        # L1 to the detached two-ray intersection on informative, confident joints
    w_ssl_reproj: float = 0.0     # L1 (px) between the projected prediction and the confident heatmap peaks
    w_ssl_temporal: float = 0.0   # jerk (third temporal difference) of the prediction
    w_ssl_bone: float = 0.0       # constant + symmetric bone lengths over the clip
    ssl_conf_thresh: float = 0.5  # min heatmap peak / view confidence for the pseudo-labels


@dataclass
class OptimConfig:
    lr: float = 1e-3
    weight_decay: float = 5e-4
    encoder_lr_scale: float = 1.0
    layer_decay: float = 1.0
    warmup_steps: int = 500
    schedule: str = "multistep"  # multistep | cosine
    milestones: List[float] = field(default_factory=lambda: [8, 10])   # epochs, fractions allowed (milestone_steps)
    gamma: float = 0.1
    min_lr_ratio: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    grad_clip: Optional[float] = None


def milestone_steps(milestones: Sequence[float], steps_per_epoch: int) -> List[int]:
    """Optimizer step at which each multistep milestone (in epochs, fractions allowed) takes effect: the first integer
    step >= milestone * steps_per_epoch, i.e. the ceiling (exact for integer milestones)."""
    return [int(math.ceil(float(m) * steps_per_epoch)) for m in milestones]


def to_dict(cfg) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    return asdict(cfg)

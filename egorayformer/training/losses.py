"""Loss terms. All poses in centimetres, shapes [B, T, J, 3] unless noted; ``valid`` is [B, T] bool.

Joints whose ground truth is NaN (missing annotation, or padding of a smaller dataset skeleton) are masked out
per joint, so mixed-skeleton batches and partially annotated frames are handled uniformly."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..data.skeleton import BONES, SYMMETRIC_BONE_PAIRS


def _frame_mask(gt: torch.Tensor, valid: Optional[torch.Tensor]) -> torch.Tensor:
    """[B, T] frames with at least one annotated joint (and ``valid``)."""
    m = torch.isfinite(gt).all(dim=-1).any(dim=-1)
    return m & valid if valid is not None else m


def _joint_mask(gt: torch.Tensor, valid: Optional[torch.Tensor]) -> torch.Tensor:
    """[B, T, J] annotated joints of valid frames."""
    m = torch.isfinite(gt).all(dim=-1)
    return m & valid[..., None] if valid is not None else m


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x [..., *], mask broadcastable to x's leading dims."""
    while mask.ndim < x.ndim:
        mask = mask[..., None]
    m = mask.to(x.dtype).expand_as(x)
    return (x * m).sum() / m.sum().clamp_min(1.0)


def match_joints(pred: torch.Tensor, gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Trim / NaN-pad the joint axis so that ``pred`` and ``gt`` [B, T, J, *] have the same J."""
    jp, jg = pred.shape[-2], gt.shape[-2]
    if jp == jg:
        return pred, gt
    if jp > jg:
        pad = torch.full((*gt.shape[:-2], jp - jg, gt.shape[-1]), float("nan"), dtype=gt.dtype, device=gt.device)
        return pred, torch.cat([gt, pad], dim=-2)
    return pred, gt[..., :jp, :]


def mpjpe_loss(pred: torch.Tensor, gt: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    pred, gt = match_joints(pred, gt)
    m = _joint_mask(gt, valid) & torch.isfinite(pred).all(dim=-1)
    d = torch.linalg.norm(torch.nan_to_num(pred.float()) - torch.nan_to_num(gt.float()), dim=-1)  # [B, T, J]
    return masked_mean(d, m)


def mpjpe_joint_masked(pred: torch.Tensor, gt: torch.Tensor, joint_mask: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """MPJPE restricted to joints where ``joint_mask`` [B, T, J] is True (e.g. visible in both views)."""
    pred, gt = match_joints(pred, gt)
    jm = joint_mask[..., : gt.shape[-2]]
    if jm.shape[-1] < gt.shape[-2]:
        jm = F.pad(jm, (0, gt.shape[-2] - jm.shape[-1]), value=False)
    m = _joint_mask(gt, valid) & jm & torch.isfinite(pred).all(dim=-1)
    d = torch.linalg.norm(torch.nan_to_num(pred.float()) - torch.nan_to_num(gt.float()), dim=-1)
    return masked_mean(d, m)


def visibility_loss(logits: torch.Tensor, target_vis: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """BCE between per-view visibility logits [B, T, V, J] and 0/1 targets, restricted to ``mask`` [B, T, V, J]."""
    m = mask.to(logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits.float(), target_vis.to(logits.dtype).float(), reduction="none")
    return (bce * m).sum() / m.sum().clamp_min(1.0)


def student_t_nll(
    pred: torch.Tensor, gt: torch.Tensor, chol: torch.Tensor, valid: Optional[torch.Tensor] = None, nu: float = 3.0
) -> torch.Tensor:
    """Multivariate Student-t NLL with covariance L L^T (EgoPoseFormer v2 eq. 10-11). chol [B, T, J, 3, 3]."""
    pred, gt = match_joints(pred, gt)
    m = _joint_mask(gt, valid)
    pred, gt, chol = pred.float(), gt.float(), chol.float()
    delta = (torch.nan_to_num(gt) - pred)[..., None]  # [B, T, J, 3, 1]
    z = torch.linalg.solve_triangular(chol, delta, upper=False)[..., 0]  # L^-1 delta
    maha = (z * z).sum(-1)
    logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
    d = 3.0
    nll = 0.5 * (nu + d) * torch.log1p(maha / nu) + 0.5 * logdet
    return masked_mean(nll, m)


def beta_gaussian_nll(
    pred: torch.Tensor, gt: torch.Tensor, chol: torch.Tensor, valid: Optional[torch.Tensor] = None, beta: float = 0.5
) -> torch.Tensor:
    """Multivariate Gaussian NLL with covariance L L^T, re-weighted per joint by stopgrad(sigma_bar^(2 beta)) - the
    beta-NLL of Seitzer et al. (ICLR 2022) generalised to 3D. With beta = 0 this is the plain NLL, whose gradient on
    the mean is scaled by 1/sigma^2 and therefore vanishes on the hard joints the network is least sure about (the
    lower body on UnrealEgo-RW); beta = 1 recovers an MSE-like weighting on the mean while sigma is still fitted by
    the NLL. sigma_bar^2 = tr(L L^T) / 3 (cm^2)."""
    pred, gt = match_joints(pred, gt)
    m = _joint_mask(gt, valid)
    pred, gt, chol = pred.float(), gt.float(), chol.float()
    delta = (torch.nan_to_num(gt) - pred)[..., None]
    z = torch.linalg.solve_triangular(chol, delta, upper=False)[..., 0]
    maha = (z * z).sum(-1)
    logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
    nll = 0.5 * maha + 0.5 * logdet
    if beta > 0:
        var_bar = (chol * chol).sum(dim=(-1, -2)) / 3.0                      # tr(L L^T) / 3
        nll = nll * var_bar.detach().clamp_min(1e-6) ** beta
    return masked_mean(nll, m)


def heatmap_loss(pred_hm: torch.Tensor, target_hm: torch.Tensor, valid: Optional[torch.Tensor] = None,
                 channel_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """pred/target [B, T, V, K, h, w]; MSE summed over views (v1 style), mean elsewhere.

    ``channel_mask`` [B, T, V, K]: channels whose joint has a 2D annotation in that view. An
    annotated joint outside the image keeps its empty target (a negative example), an *unannotated* joint is left out
    of the loss instead of being pushed towards an empty map; the normaliser counts the supervised entries only, so
    with every channel annotated the loss is unchanged."""
    se = (pred_hm.float() - target_hm.float()) ** 2
    if valid is None and channel_mask is None:
        return se.mean() * pred_hm.shape[2]
    m = torch.ones(se.shape[:4], dtype=torch.bool, device=se.device)
    if valid is not None:
        m = m & valid[:, :, None, None]
    if channel_mask is not None:
        m = m & channel_mask.bool()
    m = m[..., None, None].to(se.dtype).expand_as(se)
    return (se * m).sum() / (m.sum() / pred_hm.shape[2]).clamp_min(1.0)


def reprojection_loss(
    uv_pred: torch.Tensor, uv_gt: torch.Tensor, valid2d: torch.Tensor, valid: Optional[torch.Tensor] = None, image_size: float = 256.0
) -> torch.Tensor:
    """uv_* [B, T, V, J, 2] normalised; valid2d [B, T, V, J] -> mean L1 in pixels of a ``image_size`` image."""
    m = valid2d & torch.isfinite(uv_gt).all(-1)
    if valid is not None:
        m = m & valid[:, :, None, None]
    d = (uv_pred.float() - torch.nan_to_num(uv_gt.float())).abs().sum(-1) * image_size
    return masked_mean(d, m)


def bone_loss(
    pred: torch.Tensor, gt: torch.Tensor, valid: Optional[torch.Tensor] = None, sym_weight: float = 0.5,
    bones: Optional[Sequence[Tuple[int, int]]] = None, sym_pairs: Optional[Sequence[Tuple[int, int]]] = None,
) -> torch.Tensor:
    """Bone-length consistency with GT (cm) + left/right symmetry of predicted bone lengths. ``bones`` / ``sym_pairs``
    default to the 16-joint UnrealEgo skeleton; pass the skeleton's own lists for other joint sets. Bones with a
    missing endpoint in the GT are masked."""
    bones = list(bones) if bones is not None else BONES
    sym_pairs = list(sym_pairs) if sym_pairs is not None else SYMMETRIC_BONE_PAIRS
    pred, gt = match_joints(pred, gt)
    jm = _joint_mask(gt, valid)
    pred, gt = pred.float(), gt.float()
    a = torch.tensor([b[0] for b in bones], device=pred.device)
    b = torch.tensor([b[1] for b in bones], device=pred.device)
    lp = torch.linalg.norm(pred[..., a, :] - pred[..., b, :], dim=-1)  # [B, T, nb]
    g = torch.nan_to_num(gt)
    lg = torch.linalg.norm(g[..., a, :] - g[..., b, :], dim=-1)
    bm = jm[..., a] & jm[..., b]                                        # [B, T, nb]
    loss = masked_mean((lp - lg).abs(), bm)
    if sym_weight > 0 and sym_pairs:
        l_idx = torch.tensor([p[0] for p in sym_pairs], device=pred.device)
        r_idx = torch.tensor([p[1] for p in sym_pairs], device=pred.device)
        loss = loss + sym_weight * masked_mean((lp[..., l_idx] - lp[..., r_idx]).abs(), bm[..., l_idx] & bm[..., r_idx])
    return loss


def raydn_loss(logits: torch.Tensor, valid_joint: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Cross-entropy over the (1 + Kn) along-ray candidates of every joint, target = candidate 0 (the GT
    joint). logits [B, T, J, 1 + Kn]; valid_joint [B, T, J]; valid [B, T] frames."""
    m = valid_joint & valid[..., None] if valid is not None else valid_joint
    ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), torch.zeros(logits[..., 0].numel(), dtype=torch.long, device=logits.device),
                         reduction="none").reshape(logits.shape[:-1])
    return masked_mean(ce, m)


def bone_length_loss(
    pred_len: torch.Tensor, gt: torch.Tensor, bones: Sequence[Tuple[int, int]], valid: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """P5: L1 (cm) between predicted bone lengths [B, T, nb] and the GT bone lengths derived from ``gt`` [B, T, J, 3]
    along ``bones`` (child, parent); bones with a missing endpoint are masked."""
    jm = _joint_mask(gt, valid)
    a = torch.tensor([b[0] for b in bones], device=gt.device)
    b = torch.tensor([b[1] for b in bones], device=gt.device)
    g = torch.nan_to_num(gt.float())
    lg = torch.linalg.norm(g[..., a, :] - g[..., b, :], dim=-1)          # [B, T, nb]
    return masked_mean((pred_len.float() - lg).abs(), jm[..., a] & jm[..., b])


def velocity_loss(pred: torch.Tensor, gt: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    if pred.shape[1] < 2:
        return pred.new_zeros(())
    pred, gt = match_joints(pred, gt)
    m = _joint_mask(gt, valid)
    pred, gt = pred.float(), gt.float()
    mv = m[:, 1:] & m[:, :-1]
    vp = torch.nan_to_num(pred)[:, 1:] - torch.nan_to_num(pred)[:, :-1]
    vg = torch.nan_to_num(gt)[:, 1:] - torch.nan_to_num(gt)[:, :-1]
    return masked_mean(torch.linalg.norm(vp - vg, dim=-1), mv)


def jerk_loss(pred: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Smoothness prior: mean |third temporal difference| of the prediction (EgoBody3M / EgoPoseFormer v2 jerk loss)."""
    if pred.shape[1] < 4:
        return pred.new_zeros(())
    pred = torch.nan_to_num(pred.float())
    jerk = pred[:, 3:] - 3 * pred[:, 2:-1] + 3 * pred[:, 1:-2] - pred[:, :-3]
    if valid is None:
        return jerk.norm(dim=-1).mean()
    m = valid[:, 3:] & valid[:, 2:-1] & valid[:, 1:-2] & valid[:, :-3]
    return masked_mean(jerk.norm(dim=-1), m)


# ------------------------------------------------------------------ Self-supervision on unlabeled frames
def pseudo_geo_loss(pred: torch.Tensor, geo_pred: torch.Tensor, weight: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """L1 (cm) between the prediction and the detached triangulated pose on joints weighted by ``weight`` [B, T, J]
    (informativeness x view confidence): the two-ray intersection acts as a pseudo-label where it is reliable."""
    d = (pred.float() - geo_pred.detach().float()).norm(dim=-1)              # [B, T, J]
    w = weight.float()
    if valid is not None:
        w = w * valid[..., None].float()
    return (d * w).sum() / w.sum().clamp_min(1.0)


def heatmap_reprojection_loss(
    uv_pred: torch.Tensor, uv_peak: torch.Tensor, peak: torch.Tensor, conf_thresh: float = 0.5,
    valid: Optional[torch.Tensor] = None, image_size: float = 256.0,
) -> torch.Tensor:
    """Stereo self-consistency without labels: L1 (px) between the projections of the predicted 3D joints
    ``uv_pred`` [B, T, V, K, 2] and the (detached) heatmap peaks ``uv_peak`` where the peak value exceeds the threshold."""
    m = peak.detach() >= conf_thresh
    if valid is not None:
        m = m & valid[:, :, None, None]
    d = (uv_pred.float() - uv_peak.detach().float()).abs().sum(-1) * image_size
    return masked_mean(d, m)


def bone_consistency_loss(
    pred: torch.Tensor, valid: Optional[torch.Tensor], bones: Sequence[Tuple[int, int]],
    sym_pairs: Optional[Sequence[Tuple[int, int]]] = None, sym_weight: float = 0.5,
) -> torch.Tensor:
    """Label-free bone prior: bone lengths must be constant over the clip (deviation from the clip mean) and
    left / right symmetric. pred [B, T, J, 3]; valid [B, T]."""
    a = torch.tensor([b[0] for b in bones], device=pred.device)
    b = torch.tensor([b[1] for b in bones], device=pred.device)
    lp = torch.linalg.norm(pred.float()[..., a, :] - pred.float()[..., b, :], dim=-1)   # [B, T, nb]
    m = valid.float()[..., None] if valid is not None else torch.ones_like(lp[..., :1])
    mean = (lp * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1.0)
    loss = ((lp - mean).abs() * m).sum() / (m.sum() * lp.shape[-1]).clamp_min(1.0)
    if sym_pairs:
        l_idx = torch.tensor([p[0] for p in sym_pairs], device=pred.device)
        r_idx = torch.tensor([p[1] for p in sym_pairs], device=pred.device)
        loss = loss + sym_weight * ((lp[..., l_idx] - lp[..., r_idx]).abs() * m).sum() / (m.sum() * len(sym_pairs)).clamp_min(1.0)
    return loss


def cosine_mix(progress: float) -> float:
    """0 -> 1 following half a cosine; used to shift weight from MSE to the NLL term (EgoPoseFormer v2)."""
    p = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 - math.cos(math.pi * p))

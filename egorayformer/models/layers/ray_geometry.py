"""Differentiable ray geometry for the stereo refinement (P2 of the redesign).

* ``ray_local_frame``    - orthonormal basis (r, e1, e2) around the ray from the device centre to an anchor; 3D
                           sampling offsets are expressed in this frame (depth along the ray + two lateral axes) so
                           that "along-ray" and "lateral" hypotheses are disentangled for every joint.
* ``intersect_rays``     - confidence-weighted least-squares intersection of V rays (closed form, differentiable)
                           with the analytic covariance of the intersection point given an angular noise level.
* ``ray_angle``          - angle between the two viewing rays (near-parallel rays -> ill-conditioned depth).

All quantities live in the *target* frame of the model (pelvis- / root- / device-relative); camera centres and
directions are obtained from ``StereoRig.rays`` which handles the extrinsics.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F


def ray_local_frame(anchor: torch.Tensor, center: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """anchor [..., 3], center [..., 3] (broadcastable) -> unit r (centre -> anchor), e1, e2 [..., 3].

    e1 is chosen horizontal-ish (perpendicular to r and to the image "down" axis y); e2 = r x e1.
    Degenerate rays (parallel to y) fall back to the x axis.
    """
    r = anchor - center
    r = r / r.norm(dim=-1, keepdim=True).clamp_min(eps)
    down = torch.zeros_like(r)
    down[..., 1] = 1.0
    e1 = torch.cross(down, r, dim=-1)
    bad = e1.norm(dim=-1, keepdim=True) < 1e-3
    alt = torch.zeros_like(r)
    alt[..., 0] = 1.0
    e1 = torch.where(bad, torch.cross(alt, r, dim=-1), e1)
    e1 = F.normalize(e1, dim=-1, eps=eps)
    e2 = torch.cross(r, e1, dim=-1)
    return r, e1, e2


def ray_cosine(d: torch.Tensor) -> torch.Tensor:
    """d [..., V, M, 3] unit directions -> cosine of the angle between view 0 and view 1 [..., M] (safe to differentiate)."""
    return (d[..., 0, :, :] * d[..., 1, :, :]).sum(-1).clamp(-1.0, 1.0)


def pairwise_ray_cosine(d: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """d [..., V, M, 3] unit directions, valid [..., V, M] bool -> the largest |cosine| over all pairs of valid views
    [..., M] (the most parallel pair bounds the depth conditioning). Equals ``ray_cosine`` for V = 2 (up to the
    absolute value); 1.0 when fewer than two views are valid. Any number of views."""
    V = d.shape[-3]
    if V == 2 and valid is None:
        return ray_cosine(d).abs()
    out = torch.full(d.shape[:-3] + d.shape[-2:-1], -1.0, dtype=d.dtype, device=d.device)
    for i in range(V):
        for j in range(i + 1, V):
            c = (d[..., i, :, :] * d[..., j, :, :]).sum(-1).abs().clamp(max=1.0)
            if valid is not None:
                c = torch.where(valid[..., i, :] & valid[..., j, :], c, torch.full_like(c, -1.0))
            out = torch.maximum(out, c)
    return torch.where(out < 0, torch.ones_like(out), out)


def ray_angle(d: torch.Tensor) -> torch.Tensor:
    """d [..., V, M, 3] unit directions -> angle (rad) between view 0 and view 1 [..., M].

    The cosine is clamped strictly inside (-1, 1): acos has an infinite derivative at +-1, which happens exactly
    when both projections are clamped to the same image corner (joint outside both FOVs) and would poison the
    gradients with NaNs. Use ``ray_cosine`` inside the network; ``ray_angle`` is for analysis.
    """
    return torch.acos(ray_cosine(d).clamp(-1.0 + 1e-6, 1.0 - 1e-6))


def inv3x3(A: torch.Tensor, det_floor: float = 1e-30) -> torch.Tensor:
    """Closed-form (adjugate) inverse of [..., 3, 3] matrices, computed in float64 and returned as float32.

    ``torch.linalg.inv`` / ``solve`` dispatch to batched cuSOLVER kernels that dominate the decoder latency for the
    tiny systems of the ray intersection; the adjugate is a handful of fused elementwise kernels. float64 keeps the
    cancellation-prone cofactors exact for the ill-conditioned regularised systems used here (det ~ eps^3)."""
    A = A.double()
    a, b, c = A[..., 0, 0], A[..., 0, 1], A[..., 0, 2]
    d, e, f = A[..., 1, 0], A[..., 1, 1], A[..., 1, 2]
    g, h, i = A[..., 2, 0], A[..., 2, 1], A[..., 2, 2]
    c00, c01, c02 = e * i - f * h, -(d * i - f * g), d * h - e * g
    det = a * c00 + b * c01 + c * c02
    det = torch.where(det.abs() < det_floor, torch.full_like(det, det_floor), det)
    adj = torch.stack([
        torch.stack([c00, -(b * i - c * h), b * f - c * e], dim=-1),
        torch.stack([c01, a * i - c * g, -(a * f - c * d)], dim=-1),
        torch.stack([c02, -(a * h - b * g), a * e - b * d], dim=-1),
    ], dim=-2)
    return (adj / det[..., None, None]).float()


def intersect_rays(
    o: torch.Tensor,
    d: torch.Tensor,
    w: torch.Tensor,
    prior: Optional[torch.Tensor] = None,
    eps: float = 2e-5,
    angular_sigma: Union[float, torch.Tensor] = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weighted least-squares point closest to V rays, softly tied to a prior point.

    o [B, V, 1|M, 3] ray origins, d [B, V, M, 3] unit directions, w [B, V, M] non-negative weights,
    prior [B, M, 3] (default: origin of the frame).
    Minimises sum_v w_v || (I - d_v d_v^T)(X - o_v) ||^2 + eps || X - prior ||^2  ->  A X = b with
    A = sum_v w_v (I - d d^T) + eps I. The tiny prior term keeps the system solvable when the rays are (nearly)
    parallel or only one view is valid: the unobservable along-ray coordinate then stays at the prior (the
    current anchor in the refinement) instead of exploding. For two rays subtending an angle a the depth
    eigenvalue of A is ~a^2/2, so eps = 2e-5 biases a joint 1.5 m away (a ~ 0.08 rad, feet) by ~0.6% towards
    the prior and a joint at 60 cm by < 0.1%.

    Returns X [B, M, 3], depth rho [B, V, M] (signed distance of X along each ray; <= 0 means the solution lies
    behind that camera - diverging rays) and the covariance Sigma [B, M, 3, 3] of X under independent angular
    noise ``angular_sigma`` (rad, scalar or a tensor broadcastable to [B, V, M]) on every ray: the lateral
    position error of ray v at depth rho_v is rho_v * sigma, so Sigma = (sum_v w_v (I - d d^T) / (rho_v sigma)^2)^-1.

    Numerics: the system is solved for the offset from the prior, Y = X - prior (A Y = sum_v w_v P_v (o_v - prior);
    the eps term cancels), so the right-hand side stays small and the cancellation-prone depth direction is not
    amplified. Autocast is disabled inside: under Lightning's bf16-mixed the two matmuls used to be cast to bf16,
    which gave a median 4 cm / p90 14 cm error on noise-free rays.
    """
    with torch.autocast(device_type=o.device.type, enabled=False):
        o, d, w = o.float(), d.float(), w.float()
        B, V, M, _ = d.shape
        eye = torch.eye(3, device=d.device, dtype=d.dtype)
        P = eye - d[..., :, None] * d[..., None, :]                        # [B, V, M, 3, 3] projector onto the normal plane
        A = (w[..., None, None] * P).sum(1) + eps * eye                    # [B, M, 3, 3]
        p0 = prior.float() if prior is not None else o.new_zeros(B, M, 3)
        rel = o.expand(B, V, M, 3) - p0[:, None]                           # [B, V, M, 3] origins relative to the prior
        b = (w[..., None, None] * (P @ rel[..., None])).sum(1)             # [B, M, 3, 1]
        X = p0 + (inv3x3(A) @ b)[..., 0]                                   # [B, M, 3]
        rho = ((X[:, None] - o) * d).sum(-1)                               # [B, V, M]
        sig = angular_sigma if torch.is_tensor(angular_sigma) else torch.tensor(float(angular_sigma), device=d.device)
        inv_var = w / (rho.abs().clamp_min(1.0) * sig.float()) ** 2        # [B, V, M]
        A_cov = (inv_var[..., None, None] * P).sum(1) + eps * eye
        Sigma = inv3x3(A_cov)
        return X, rho, Sigma


def intersect_ray_sphere(
    o: torch.Tensor, d: torch.Tensor, center: torch.Tensor, radius: torch.Tensor, prior: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-form, differentiable intersection of rays with spheres (a joint seen in a
    single view lies on that view's ray *and* at bone-length distance from its parent, cf. KITRO / Structural
    Triangulation).

    o, d [B, M, 3] ray origins / unit directions, center [B, M, 3] sphere centres (parent joints), radius [B, M]
    (bone lengths, cm), prior [B, M, 3] (current anchors). Of the two roots the one closer to the prior is returned;
    when the ray misses the sphere the point of closest approach is pushed radially onto the sphere. Returns
    X [B, M, 3], ``hit`` [B, M] (real roots exist) and ``gap`` [B, M] (distance ray - sphere, 0 when hit)."""
    o, d, center, radius, prior = o.float(), d.float(), center.float(), radius.float(), prior.float()
    oc = o - center
    b = (d * oc).sum(-1)                                   # [B, M]
    cc = (oc * oc).sum(-1) - radius * radius
    disc = b * b - cc
    hit = disc >= 0
    root = torch.sqrt(disc.clamp_min(eps))
    t1, t2 = -b - root, -b + root
    X1, X2 = o + t1[..., None] * d, o + t2[..., None] * d
    pick2 = (X2 - prior).norm(dim=-1) < (X1 - prior).norm(dim=-1)
    X_hit = torch.where(pick2[..., None], X2, X1)
    X_near = o + (-b)[..., None] * d                       # closest approach when the ray misses the sphere
    X_miss = center + radius[..., None] * F.normalize(X_near - center, dim=-1, eps=eps)
    X = torch.where(hit[..., None], X_hit, X_miss)
    gap = torch.sqrt((-disc).clamp_min(0.0) + eps) * (~hit).float()
    return X, hit, gap


def so3_exp(rotvec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Rodrigues: rotation vectors [..., 3] (rad) -> rotation matrices [..., 3, 3]; exact for small angles
    (used for the calibration perturbations / learnable extrinsic refinement of ``RigBank``)."""
    theta = rotvec.norm(dim=-1, keepdim=True).clamp_min(eps)                 # [..., 1]
    k = rotvec / theta
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = torch.zeros_like(kx)
    K = torch.stack([torch.stack([zero, -kz, ky], -1), torch.stack([kz, zero, -kx], -1), torch.stack([-ky, kx, zero], -1)], -2)
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device).expand(K.shape)
    s, c = torch.sin(theta)[..., None], torch.cos(theta)[..., None]
    return eye + s * K + (1.0 - c) * (K @ K)


def normal_matrix_eigvals(d: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Eigenvalues (ascending) of the intersection's normal matrix A = sum_v w_v (I - d_v d_v^T) -> [B, M, 3].

    d [B, V, M, 3] unit ray directions, w [B, V, M] weights. The eigenvectors of A are the observable directions of the
    joint: for two rays subtending a small angle a, the smallest eigenvalue ~ a^2 / 2 belongs to the (nearly
    unobservable) depth direction, the two others (~ number of valid views) to the lateral directions. fp32, no autocast."""
    with torch.autocast(device_type=d.device.type, enabled=False):
        d, w = d.float(), w.float()
        eye = torch.eye(3, device=d.device, dtype=d.dtype)
        A = (w[..., None, None] * (eye - d[..., :, None] * d[..., None, :])).sum(1)    # [B, M, 3, 3]
        return torch.linalg.eigvalsh(A)


def disparity_rate(anchor: torch.Tensor, center: torch.Tensor, project_fn, feat_hw: Tuple[int, int],
                   delta_cm: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Image motion of a joint hypothesis moved along its viewing ray, from the real camera model.

    anchor [N, J, 3], center [N, 1, 3] (device centre); ``project_fn`` points [N, M, 3] -> uv [N, V, M, 2] in [0, 1],
    fov [N, V, M]. A central finite difference of +/- delta/2 cm along r = (anchor - centre) / |.| gives every view's
    displacement in *feature pixels per cm*; the disparity rate is the sum of the two largest view displacements
    (= the change of the stereo disparity for V = 2, the most sensitive pair otherwise). Views where either end of the
    difference leaves the image contribute 0. Returns (rate [N, J], per-view rate [N, V, J]); no gradient, fp32."""
    with torch.no_grad(), torch.autocast(device_type=anchor.device.type, enabled=False):
        anchor, center = anchor.float(), center.float()
        N, J, _ = anchor.shape
        r = anchor - center
        r = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pts = torch.cat([anchor - 0.5 * delta_cm * r, anchor + 0.5 * delta_cm * r], dim=1)      # [N, 2J, 3]
        uv, fov = project_fn(pts)
        h, w = feat_hw
        du = (uv[:, :, J:] - uv[:, :, :J]).float() * uv.new_tensor([float(w), float(h)])        # [N, V, J, 2] feature px
        per_view = du.norm(dim=-1) / float(delta_cm)
        per_view = per_view * (fov[:, :, :J] & fov[:, :, J:]).to(per_view.dtype)
        V = per_view.shape[1]
        rate = per_view.topk(min(2, V), dim=1).values.sum(1) if V > 1 else per_view[:, 0]
        return rate, per_view


def log_trace(Sigma: torch.Tensor, floor: float = 1e-4) -> torch.Tensor:
    """[..., 3, 3] -> log(trace) [...] (a scalar uncertainty summary)."""
    return torch.log(Sigma.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(floor))


def pixel_to_angular_sigma(pixel_sigma: float, focal_px: float = 516.8) -> float:
    """Convert an image-plane noise level (px in the 1024 calibration image) to radians near the image centre
    using the first W2C polynomial coefficient of the UnrealEgo calibration (~517 px / rad)."""
    return pixel_sigma / focal_px


__all__ = ["ray_local_frame", "ray_angle", "ray_cosine", "pairwise_ray_cosine", "inv3x3", "intersect_rays", "intersect_ray_sphere",
           "so3_exp", "log_trace", "pixel_to_angular_sigma", "normal_matrix_eigvals", "disparity_rate"]

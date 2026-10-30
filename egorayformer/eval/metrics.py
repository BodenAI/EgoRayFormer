"""3D pose metrics: MPJPE, PA-MPJPE, 3D PCK, AUC, MPJVE (all in millimetres; MPJVE in mm per frame) plus the analysis
metrics used in the paper: root(mid-hip)-relative MPJPE (translation-free protocol), per-joint / body-group errors,
visibility strata (joint seen by both views / one view / none), image-motion strata (proxy for fast head motion) and
uncertainty calibration (Gaussian / Student-t NLL, Mahalanobis coverage, AUSE sparsification error).

Joints without a ground truth are masked per joint: a frame with one unannotated joint keeps
its other joints, PA-MPJPE aligns on the annotated joints (>= 4 of them), and the summaries report how many frames and
joint samples entered. A non-finite *prediction* of an annotated joint is counted as an infinite error, so a diverged
model shows up as ``inf`` instead of silently dropping out. Velocity errors also cover the frame pairs that straddle two
consecutive evaluation chunks of a sequence (``seq_ids`` / ``frame_ids``), so the clip-wise and the continuous streaming
evaluation see the same pairs.

``batch_procrustes`` is adapted from EgoPoseFormer v1 ``pose_metric.py`` (Apache-2.0), itself
derived from the widely used SPIN / UnrealEgo ``compute_similarity_transform``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..data.records import MIDHIP_JOINTS
from ..data.skeleton import BODY_PARTS, JOINT_NAMES, get_skeleton

CM_TO_MM = 10.0
UPPER_BODY = [0, 1, 2, 3, 4, 5, 6, 7]
LOWER_BODY = [8, 9, 10, 11, 12, 13, 14, 15]
CHI2_3_68, CHI2_3_95 = 3.5268, 7.8147  # chi-square (3 dof) quantiles at 68.27 % / 95 % = Gaussian coverage thresholds
COV_LEVELS = (0.6827, 0.95)
DEFAULT_SKELETON = "unrealego16"
_TORSO_KEYS = ("neck", "upperarm", "shoulder", "thigh", "hip", "upleg", "spine")
MIN_PA_JOINTS = 4  # a similarity transform needs >= 4 non-degenerate correspondences


def torso_joint_ids(sk) -> List[int]:
    """Joints that move rigidly with the trunk (neck, shoulders / upper arms, hips / thighs, spine) of any skeleton."""
    return [j for j, n in enumerate(sk.joint_names) if any(k in n.lower() for k in _TORSO_KEYS)]


def _regularized_beta(z: float, a: float, b: float, n: int = 20000) -> float:
    """I_z(a, b) by Simpson integration of t^(a-1) (1-t)^(b-1) / B(a, b) on [0, z] (a, b >= 1 here)."""
    if z <= 0.0:
        return 0.0
    if z >= 1.0:
        return 1.0
    t = np.linspace(0.0, z, 2 * n + 1)
    f = t ** (a - 1.0) * (1.0 - t) ** (b - 1.0)
    h = z / (2 * n)
    integral = h / 3.0 * (f[0] + f[-1] + 4.0 * f[1:-1:2].sum() + 2.0 * f[2:-1:2].sum())
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    return float(min(1.0, integral / math.exp(log_beta)))


def maha_quantile(q: float, dim: int = 3, nu: float = float("inf")) -> float:
    """Quantile of the squared Mahalanobis distance (x - mu)^T S^-1 (x - mu) of a ``dim``-variate Student-t with
    ``nu`` degrees of freedom and scale matrix S (chi-square with ``dim`` dof for nu = inf, the Gaussian case).

    For a t distribution maha / dim ~ F(dim, nu); the Gaussian chi-square thresholds under-cover a *calibrated*
    t model: with nu = 3 a perfectly calibrated model reaches only 0.55 at the chi2 68 % threshold and 0.77 at
    the 95 % one, so coverage must be judged against the quantiles of the training likelihood.
    """
    if not math.isfinite(nu):
        lo, hi = 0.0, 200.0
        for _ in range(100):  # chi-square CDF = regularized lower gamma; bisection on the incomplete beta is not needed
            mid = 0.5 * (lo + hi)
            cdf = _regularized_gamma(dim / 2.0, mid / 2.0)
            lo, hi = (mid, hi) if cdf < q else (lo, mid)
        return 0.5 * (lo + hi)
    a, b = dim / 2.0, nu / 2.0
    lo, hi = 0.0, 1e4
    for _ in range(100):
        x = 0.5 * (lo + hi)                  # F value
        z = dim * x / (dim * x + nu)
        cdf = _regularized_beta(z, a, b)
        lo, hi = (x, hi) if cdf < q else (lo, x)
    return dim * 0.5 * (lo + hi)


def _regularized_gamma(s: float, x: float, n: int = 20000) -> float:
    """P(s, x) = gamma(s, x) / Gamma(s) by Simpson integration (s = 1.5 for the 3-dof chi-square)."""
    if x <= 0.0:
        return 0.0
    t = np.linspace(0.0, x, 2 * n + 1)
    f = t ** (s - 1.0) * np.exp(-t)
    h = x / (2 * n)
    integral = h / 3.0 * (f[0] + f[-1] + 4.0 * f[1:-1:2].sum() + 2.0 * f[2:-1:2].sum())
    return float(min(1.0, integral / math.exp(math.lgamma(s))))


def batch_procrustes(S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
    """Similarity transform (scale, R, t) aligning S1 to S2. S1, S2: [B, J, 3] -> aligned S1."""
    S1 = S1.double().permute(0, 2, 1)  # [B, 3, J]
    S2 = S2.double().permute(0, 2, 1)
    mu1 = S1.mean(dim=-1, keepdim=True)
    mu2 = S2.mean(dim=-1, keepdim=True)
    X1, X2 = S1 - mu1, S2 - mu2
    var1 = (X1 ** 2).sum(dim=(1, 2))
    K = X1 @ X2.transpose(1, 2)
    U, _, Vh = torch.linalg.svd(K)
    V = Vh.transpose(1, 2)
    Z = torch.eye(3, dtype=S1.dtype, device=S1.device).expand(S1.shape[0], 3, 3).clone()
    Z[:, -1, -1] = torch.sign(torch.linalg.det(U @ V.transpose(1, 2)))
    R = V @ Z @ U.transpose(1, 2)
    scale = torch.einsum("bii->b", R @ K) / var1.clamp_min(1e-12)
    t = mu2 - scale[:, None, None] * (R @ mu1)
    S1_hat = scale[:, None, None] * (R @ S1) + t
    return S1_hat.permute(0, 2, 1).float()


def batch_procrustes_masked(S1: torch.Tensor, S2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Similarity transform aligning S1 to S2 estimated on the joints where ``mask`` [B, J] is True only (equal to
    ``batch_procrustes`` when every joint is valid). S1, S2: [B, J, 3] (NaN allowed outside the mask) -> aligned S1."""
    w = mask.double()[..., None]                                            # [B, J, 1]
    A = torch.nan_to_num(S1.double()) * w
    Bm = torch.nan_to_num(S2.double()) * w
    n = w.sum(dim=1, keepdim=True).clamp_min(1.0)
    mu1, mu2 = A.sum(1, keepdim=True) / n, Bm.sum(1, keepdim=True) / n      # [B, 1, 3]
    X1, X2 = (A - mu1) * w, (Bm - mu2) * w
    var1 = (X1 ** 2).sum(dim=(1, 2))
    K = X1.transpose(1, 2) @ X2                                             # sum_j x1_j x2_j^T  [B, 3, 3]
    U, _, Vh = torch.linalg.svd(K)
    V = Vh.transpose(1, 2)
    Z = torch.eye(3, dtype=K.dtype, device=K.device).expand(K.shape[0], 3, 3).clone()
    Z[:, -1, -1] = torch.sign(torch.linalg.det(U @ V.transpose(1, 2)))
    R = V @ Z @ U.transpose(1, 2)
    scale = torch.einsum("bii->b", R @ K) / var1.clamp_min(1e-12)
    t = mu2.transpose(1, 2) - scale[:, None, None] * (R @ mu1.transpose(1, 2))  # [B, 3, 1]
    S1_hat = scale[:, None, None] * (R @ S1.double().transpose(1, 2)) + t
    return S1_hat.transpose(1, 2).float()


def per_joint_error_mm(pred_cm: torch.Tensor, gt_cm: torch.Tensor) -> torch.Tensor:
    """[..., J, 3] -> [..., J] Euclidean error in mm."""
    return torch.linalg.norm(pred_cm - gt_cm, dim=-1) * CM_TO_MM


def masked_joint_error_mm(pred_cm: torch.Tensor, gt_cm: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Per-joint error [..., J] in mm: NaN where ``valid`` (annotated joint) is False, +inf where the joint is annotated
    but the prediction (or the error) is not finite - a diverged model must not drop out of the average."""
    e = per_joint_error_mm(pred_cm, gt_cm)
    e = torch.where(valid & ~torch.isfinite(e), torch.full_like(e, float("inf")), e)
    return e.masked_fill(~valid, float("nan"))


def root_center(pose_cm: torch.Tensor, joints: Sequence[int] = MIDHIP_JOINTS) -> torch.Tensor:
    """Subtract the mid-hip (mean of the given joints) -> translation-free pose."""
    return pose_cm - pose_cm[..., list(joints), :].mean(dim=-2, keepdim=True)


def pa_per_joint_error_mm(pred_cm: torch.Tensor, gt_cm: torch.Tensor) -> torch.Tensor:
    shape = pred_cm.shape
    p = pred_cm.reshape(-1, shape[-2], 3)
    g = gt_cm.reshape(-1, shape[-2], 3)
    aligned = batch_procrustes(p, g)
    return (torch.linalg.norm(aligned - g, dim=-1) * CM_TO_MM).reshape(shape[:-1])


def pa_masked_joint_error_mm(pred_cm: torch.Tensor, gt_cm: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """PA-MPJPE per joint [N, J] aligned on the annotated joints of every frame; frames with fewer than
    ``MIN_PA_JOINTS`` annotated joints (or a non-finite prediction) get NaN / inf rows like ``masked_joint_error_mm``."""
    N, J = valid.shape
    out = torch.full((N, J), float("nan"), dtype=torch.float32, device=pred_cm.device)
    pred_ok = torch.isfinite(pred_cm).all(-1)                                   # [N, J]
    enough = valid.sum(-1) >= MIN_PA_JOINTS
    good = enough & (pred_ok | ~valid).all(-1)
    if good.any():
        p, g, m = pred_cm[good], gt_cm[good], valid[good]
        aligned = batch_procrustes_masked(p, g, m)
        e = torch.linalg.norm(aligned - torch.nan_to_num(g), dim=-1) * CM_TO_MM
        out[good] = e.float().masked_fill(~m, float("nan"))
    bad_pred = enough & ~good                                                   # annotated joints with a non-finite prediction
    if bad_pred.any():
        out[bad_pred] = torch.where(valid[bad_pred], torch.full_like(out[bad_pred], float("inf")), out[bad_pred])
    return out


def _finite_or_inf(x: torch.Tensor) -> torch.Tensor:
    """Entries that took part in the evaluation (finite or +inf), i.e. not NaN."""
    return ~torch.isnan(x)


def pck(err_mm: torch.Tensor, threshold_mm: float = 100.0) -> torch.Tensor:
    e = err_mm[_finite_or_inf(err_mm)]
    return (e <= threshold_mm).float().mean() if e.numel() else torch.tensor(float("nan"))


def auc(err_mm: torch.Tensor, max_threshold_mm: float = 150.0, steps: int = 31) -> torch.Tensor:
    e = err_mm[_finite_or_inf(err_mm)]
    if e.numel() == 0:
        return torch.tensor(float("nan"))
    thr = torch.linspace(0.0, max_threshold_mm, steps, device=e.device)
    pcks = torch.stack([(e <= t).float().mean() for t in thr])
    return pcks.mean()


def mpjve_mm(pred_cm: torch.Tensor, gt_cm: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean per-joint velocity error over consecutive frames (mm per frame, not per second). pred/gt [B, T, J, 3],
    valid [B, T]."""
    vp = pred_cm[:, 1:] - pred_cm[:, :-1]
    vg = gt_cm[:, 1:] - gt_cm[:, :-1]
    err = torch.linalg.norm(vp - vg, dim=-1) * CM_TO_MM  # [B, T-1, J]
    if valid is not None:
        m = (valid[:, 1:] & valid[:, :-1]).float()[..., None]
        return (err * m).sum() / (m.sum() * err.shape[-1]).clamp_min(1.0)
    return err.mean()


def _nanmean(x: Optional[torch.Tensor]) -> float:
    """Mean over the evaluated entries (NaN = not evaluated); +inf propagates (diverged predictions)."""
    if x is None or x.numel() == 0:
        return float("nan")
    m = _finite_or_inf(x)
    if not bool(m.any()):
        return float("nan")
    return float(x[m].double().mean())


def _nanmean_rows(e: torch.Tensor) -> torch.Tensor:
    """Per-frame mean over the evaluated joints of [N, J] -> [N] (NaN for frames without any)."""
    m = _finite_or_inf(e)
    s = torch.where(m, e, torch.zeros_like(e)).sum(dim=1)
    c = m.sum(dim=1)
    return torch.where(c > 0, s / c.clamp_min(1), torch.full_like(s, float("nan")))


# ------------------------------------------------------------------------------------------ uncertainty
def gaussian_nll_mm(delta_mm: torch.Tensor, chol_mm: torch.Tensor) -> torch.Tensor:
    """delta [..., 3] (gt - pred, mm), chol [..., 3, 3] lower-triangular (mm) -> per-joint Gaussian NLL."""
    z = torch.linalg.solve_triangular(chol_mm, delta_mm[..., None], upper=False)[..., 0]
    maha = (z * z).sum(-1)
    logdet = 2.0 * torch.log(torch.diagonal(chol_mm, dim1=-2, dim2=-1)).sum(-1)
    return 0.5 * (maha + logdet + 3.0 * math.log(2.0 * math.pi)), maha


def student_t_nll_mm(maha: torch.Tensor, chol_mm: torch.Tensor, nu: float = 3.0) -> torch.Tensor:
    logdet = 2.0 * torch.log(torch.diagonal(chol_mm, dim1=-2, dim2=-1)).sum(-1)
    d = 3.0
    const = math.lgamma((nu + d) / 2) - math.lgamma(nu / 2) - 0.5 * d * math.log(nu * math.pi)
    return -const + 0.5 * (nu + d) * torch.log1p(maha / nu) + 0.5 * logdet


def sparsification_ause(err: torch.Tensor, unc: torch.Tensor, steps: int = 20) -> float:
    """Area Under the Sparsification Error curve.

    Remove the fraction f of samples with the highest predicted uncertainty (resp. highest true error for the
    oracle) and track the mean error of the rest; AUSE = mean_f (curve_unc - curve_oracle) normalised by the
    overall mean error. 0 = perfect ranking.
    """
    ok = torch.isfinite(err) & torch.isfinite(unc)
    err, unc = err[ok], unc[ok]
    n = err.numel()
    if n < 2:
        return float("nan")
    order_u = torch.argsort(unc, descending=True)
    order_e = torch.argsort(err, descending=True)
    e_u, e_o = err[order_u], err[order_e]
    # suffix means: removing the first k elements leaves mean(e[k:])
    suf_u = torch.flip(torch.cumsum(torch.flip(e_u, [0]), 0), [0])
    suf_o = torch.flip(torch.cumsum(torch.flip(e_o, [0]), 0), [0])
    diffs = []
    for i in range(steps):
        k = int(round(i / steps * n))
        rem = n - k
        diffs.append(float(suf_u[k] / rem - suf_o[k] / rem))
    return float(np.mean(diffs) / max(float(err.mean()), 1e-9))


# ------------------------------------------------------------------------------------------ accumulator
# per-frame stores (one row per evaluated frame of a tag; aligned with ``err``) and per-velocity-pair stores
_FRAME_STORES = ("err", "err_root", "pa_err", "vis", "motion", "nll_g", "nll_t", "maha", "unc", "dist", "conf", "gate",
                 "geo_err", "geo_info", "err_ray", "err_lat", "std_local", "head_motion", "fid")
_LABEL_STORES = ("seq",)                 # per-frame python labels (plus every kind of ``labels``)
_PAIR_STORES = ("vel", "vel_fid")


class PoseMetricAccumulator:
    """Collects per-frame joint errors (and velocity errors) grouped by dataset name.

    ``update`` optionally receives per-view joint visibility ``vis`` [B, T, V, J], normalised 2D keypoints
    ``pts2d`` [B, T, V, J, 2] (image-motion proxy) and the predicted Cholesky factor ``unc_chol`` [B, T, J, 3, 3]
    (cm) for calibration metrics. With ``seq_ids`` / ``frame_ids`` it also keeps per-sequence error sums
    (``per_sequence``, the input of a paired sequence bootstrap) and the boundary frames
    of every chunk (velocity errors across consecutive chunks). ``export_state`` / ``load_states`` merge the
    accumulators of several ranks exactly (duplicated frames of a padded DistributedSampler are dropped).
    """

    def __init__(
        self,
        pck_threshold_mm: float = 100.0,
        auc_max_mm: float = 150.0,
        motion_bins_px: Sequence[float] = (5.0, 20.0),
        image_size_px: float = 1024.0,
        nll_nu: float = 3.0,
        dist_bins_cm: Sequence[float] = (60.0, 100.0),
        conf_threshold: float = 0.5,
        head_motion_bins_cm: Sequence[float] = (1.0, 3.0),
    ):
        self.pck_thr = pck_threshold_mm
        self.auc_max = auc_max_mm
        self.motion_bins = tuple(motion_bins_px)
        self.image_size_px = image_size_px
        self.nll_nu = nll_nu
        # coverage thresholds of the *training* likelihood: Student-t(nu) when the model is trained with the
        # Student-t NLL, chi-square (nu = inf) for a Gaussian NLL; the chi-square ones are always reported as *_gauss
        self.maha_thr = tuple(maha_quantile(q, 3, nll_nu) for q in COV_LEVELS)
        # scale matrix -> covariance factor of a t distribution (variance = nu / (nu - 2) * S)
        self.t_std_factor = math.sqrt(nll_nu / (nll_nu - 2.0)) if (math.isfinite(nll_nu) and nll_nu > 2.0) else 1.0
        self.dist_bins = tuple(dist_bins_cm)   # near / mid / far joints (distance from the device centre)
        self.conf_thr = conf_threshold         # high / low heatmap (or predicted-visibility) confidence
        # per-frame mean displacement (cm) of the torso joints in the device frame = motion of the headset
        # relative to the body (the image-motion strata mix it with the limbs' own motion)
        self.head_motion_bins = tuple(head_motion_bins_cm)
        self.reset()

    def reset(self) -> None:
        self.err: Dict[str, List[torch.Tensor]] = defaultdict(list)       # [N, J] device / target-frame error (NaN = no GT)
        self.err_root: Dict[str, List[torch.Tensor]] = defaultdict(list)  # [N, J] mid-hip centred error
        self.pa_err: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.vel: Dict[str, List[torch.Tensor]] = defaultdict(list)       # [M, J] mm per frame
        self.vel_fid: Dict[str, List[torch.Tensor]] = defaultdict(list)   # [M] frame id of the later frame of the pair
        self.vel_seq: Dict[str, List[str]] = defaultdict(list)            # [M] sequence of the pair
        self.vis: Dict[str, List[torch.Tensor]] = defaultdict(list)       # [N, J] int8: number of views seeing the joint
        self.motion: Dict[str, List[torch.Tensor]] = defaultdict(list)    # [N] px (NaN when undefined)
        self.nll_g: Dict[str, List[torch.Tensor]] = defaultdict(list)     # [N, J]
        self.nll_t: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.maha: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.unc: Dict[str, List[torch.Tensor]] = defaultdict(list)       # [N, J] sqrt(trace Sigma) mm
        self.dist: Dict[str, List[torch.Tensor]] = defaultdict(list)      # [N, J] cm: GT joint distance from the device centre
        self.conf: Dict[str, List[torch.Tensor]] = defaultdict(list)      # [N, J] heatmap / visibility confidence in [0, 1]
        self.gate: Dict[str, List[torch.Tensor]] = defaultdict(list)      # [N, J] geometry gate of the last layer
        self.geo_err: Dict[str, List[torch.Tensor]] = defaultdict(list)   # [N, J] mm: error of the triangulated pose P_geo
        self.geo_info: Dict[str, List[torch.Tensor]] = defaultdict(list)  # [N, J] bool: geometrically informative joint
        self.err_ray: Dict[str, List[torch.Tensor]] = defaultdict(list)   # [N, J] mm: error component along the viewing ray
        self.err_lat: Dict[str, List[torch.Tensor]] = defaultdict(list)   # [N, J] mm: lateral error component
        self.std_local: Dict[str, List[torch.Tensor]] = defaultdict(list) # [N, J, 3] mm: predicted (ray, e1, e2) stds
        self.head_motion: Dict[str, List[torch.Tensor]] = defaultdict(list)  # [N] cm (NaN when undefined)
        self.fid: Dict[str, List[torch.Tensor]] = defaultdict(list)       # [N] int64 frame id (-1 when unknown)
        self.seq: Dict[str, List[str]] = defaultdict(list)                # [N] sequence id ("" when unknown)
        self.labels: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))  # kind -> tag -> [N] labels
        # first / last evaluated frame of every chunk: (seq, stride, first fid, p, g, last fid, p, g) per tag
        self.bounds: Dict[str, List[Tuple]] = defaultdict(list)
        self.skeleton_of: Dict[str, str] = {}                              # tag -> skeleton name

    @property
    def groups(self) -> Dict[str, List[str]]:
        """Backwards-compatible view of the per-frame session labels."""
        return self.labels["session"]

    @staticmethod
    def _head_motion(dev_pose: torch.Tensor, ids: Sequence[int]) -> torch.Tensor:
        """dev_pose [T, J, 3] GT in the device frame -> per-frame mean displacement (cm) of the torso joints ``ids``
        between t-1 and t; NaN at t=0 and where undefined."""
        T = dev_pose.shape[0]
        out = torch.full((T,), float("nan"), device=dev_pose.device)
        if T < 2 or not ids:
            return out
        x = dev_pose[:, list(ids)]
        out[1:] = (x[1:] - x[:-1]).norm(dim=-1).mean(dim=-1)  # NaN propagates for joints without GT
        return out

    @staticmethod
    def _image_motion(pts2d: torch.Tensor, vis: Optional[torch.Tensor], image_size_px: float) -> torch.Tensor:
        """pts2d [T, V, J, 2] normalised -> per-frame mean 2D displacement (px) of joints visible in t-1 and t; NaN at t=0."""
        T = pts2d.shape[0]
        out = torch.full((T,), float("nan"), device=pts2d.device)
        if T < 2:
            return out
        d = (pts2d[1:] - pts2d[:-1]).norm(dim=-1) * image_size_px  # [T-1, V, J]
        ok = torch.isfinite(d)
        if vis is not None:
            ok = ok & vis[1:] & vis[:-1]
        cnt = ok.sum(dim=(1, 2)).clamp_min(1)
        out[1:] = torch.where(ok, d, torch.zeros_like(d)).sum(dim=(1, 2)) / cnt
        out[1:][ok.sum(dim=(1, 2)) == 0] = float("nan")
        return out

    @staticmethod
    def _frame_stride(fids: torch.Tensor) -> int:
        """Frame-id step of a chunk's time axis (all its valid frames, annotated or not); 1 when undetermined."""
        d = fids[1:] - fids[:-1] if fids.numel() > 1 else fids.new_zeros(0)
        d = d[d > 0]
        return int(d.min()) if d.numel() else 1

    @staticmethod
    def _velocity_error(p: torch.Tensor, g: torch.Tensor, gm: torch.Tensor) -> torch.Tensor:
        """Velocity error (mm / frame) between consecutive rows of p / g [n, J, 3] with GT masks gm [n, J] -> [n-1, J]."""
        ve = torch.linalg.norm((p[1:] - p[:-1]) - (g[1:] - g[:-1]), dim=-1) * CM_TO_MM
        ok = gm[1:] & gm[:-1]
        ve = torch.where(ok & ~torch.isfinite(ve), torch.full_like(ve, float("inf")), ve)
        return ve.masked_fill(~ok, float("nan"))

    @torch.no_grad()
    def update(
        self,
        pred_cm: torch.Tensor,
        gt_cm: torch.Tensor,
        valid_frames: Optional[torch.Tensor] = None,
        tags: Optional[List[str]] = None,
        vis: Optional[torch.Tensor] = None,
        pts2d: Optional[torch.Tensor] = None,
        unc_chol: Optional[torch.Tensor] = None,
        skeletons: Optional[List[str]] = None,
        dist_cm: Optional[torch.Tensor] = None,
        conf: Optional[torch.Tensor] = None,
        gate: Optional[torch.Tensor] = None,
        geo_pred_cm: Optional[torch.Tensor] = None,
        geo_info: Optional[torch.Tensor] = None,
        ray_dir: Optional[torch.Tensor] = None,
        local_std_cm: Optional[torch.Tensor] = None,
        image_size_px: Optional[torch.Tensor] = None,
        dev_pose_cm: Optional[torch.Tensor] = None,
        groups: Optional[List[str]] = None,
        seq_ids: Optional[List[str]] = None,
        frame_ids: Optional[torch.Tensor] = None,
        labels: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """pred/gt [B, T, J, 3] (J may be NaN-padded beyond the skeleton's joints); valid_frames [B, T]; tags: dataset
        name per batch item; skeletons: skeleton name per batch item (default 16-joint UnrealEgo).

        Analysis inputs (all [B, T, J] unless noted, optional): ``dist_cm`` distance of the GT joint from the device
        centre, ``conf`` heatmap / visibility confidence, ``gate`` geometry gate, ``geo_pred_cm`` [B, T, J, 3]
        triangulated pose, ``geo_info`` informative-joint flag;
        ``ray_dir`` [B, T, J, 3] unit viewing ray of the GT joint and ``local_std_cm`` [B, T, J, 3] predicted stds
        (along ray, e1, e2) decompose error and uncertainty along vs. across the ray.
        ``image_size_px`` [B] original image width (the scale of the normalised ``pts2d``; default the constructor's),
        ``dev_pose_cm`` [B, T, J, 3] GT in the device frame (head-motion strata). ``labels`` {kind: [B] label} gives the
        per-group breakdowns ``<kind>/<label>`` (``groups`` = the "session" kind, kept for older callers);
        ``seq_ids`` [B] / ``frame_ids`` [B, T] enable the per-sequence sums and the cross-chunk velocity pairs."""
        B, T = pred_cm.shape[:2]
        pred_cm, gt_cm = pred_cm.float(), gt_cm.float()
        if valid_frames is None:
            valid_frames = torch.ones(B, T, dtype=torch.bool, device=pred_cm.device)
        tags = tags or ["all"] * B
        skeletons = skeletons or [DEFAULT_SKELETON] * B
        kinds = dict(labels or {})
        if groups is not None:
            kinds.setdefault("session", groups)
        for b in range(B):
            sk = get_skeleton(skeletons[b])
            J = sk.num_joints
            tag = tags[b]
            self.skeleton_of.setdefault(tag, sk.name)
            gb, pb = gt_cm[b, :, :J], pred_cm[b, :, :J]
            gm_all = torch.isfinite(gb).all(dim=-1)                                    # [T, J] annotated joints
            vf = valid_frames[b] & gm_all.any(dim=-1)
            n = int(vf.sum())
            if n == 0:
                continue
            p, g, gm = pb[vf], gb[vf], gm_all[vf]
            self.err[tag].append(masked_joint_error_mm(p, g, gm).cpu())
            root_ok = gm[:, list(sk.root_joints)].all(dim=-1, keepdim=True)            # [n, 1] mid-hip defined
            er = masked_joint_error_mm(root_center(p, sk.root_joints), root_center(g, sk.root_joints), gm & root_ok)
            self.err_root[tag].append(er.cpu())
            self.pa_err[tag].append(pa_masked_joint_error_mm(p, g, gm).cpu())
            fids = frame_ids[b][vf].long().cpu() if frame_ids is not None else torch.full((n,), -1, dtype=torch.long)
            sid = str(seq_ids[b]) if seq_ids is not None else ""
            self.fid[tag].append(fids)
            self.seq[tag].extend([sid] * n)
            for kind, lab in kinds.items():
                self.labels[kind][tag].extend([lab[b]] * n)
            if vis is not None:
                self.vis[tag].append(vis[b][vf][:, :, :J].to(torch.int8).sum(dim=1).cpu())  # [n, J] views seeing the joint
            if pts2d is not None:
                size = float(image_size_px[b]) if image_size_px is not None else self.image_size_px
                mo = self._image_motion(pts2d[b][:, :, :J].float(), vis[b][:, :, :J] if vis is not None else None, size)
                self.motion[tag].append(mo[vf].cpu())
            if dev_pose_cm is not None:
                self.head_motion[tag].append(self._head_motion(dev_pose_cm[b][:, :J].float(), torso_joint_ids(sk))[vf].cpu())
            if unc_chol is not None and unc_chol.shape[-3] == J:
                L = unc_chol[b][vf].float() * CM_TO_MM
                delta = torch.nan_to_num((g - p) * CM_TO_MM)
                nll_g, maha = gaussian_nll_mm(delta, L)
                nll_t = student_t_nll_mm(maha, L, self.nll_nu)
                miss = ~gm
                self.nll_g[tag].append(nll_g.masked_fill(miss, float("nan")).cpu())
                self.maha[tag].append(maha.masked_fill(miss, float("nan")).cpu())
                self.nll_t[tag].append(nll_t.masked_fill(miss, float("nan")).cpu())
                self.unc[tag].append((L ** 2).sum(dim=(-2, -1)).sqrt().masked_fill(miss, float("nan")).cpu())   # sqrt(trace L L^T)
            if dist_cm is not None and dist_cm.shape[-1] >= J:
                self.dist[tag].append(dist_cm[b][vf][:, :J].float().cpu())
            if conf is not None and conf.shape[-1] >= J:
                self.conf[tag].append(conf[b][vf][:, :J].float().cpu())
            if gate is not None and gate.shape[-1] >= J:
                self.gate[tag].append(gate[b][vf][:, :J].float().cpu())
            if geo_pred_cm is not None and geo_pred_cm.shape[-2] >= J:
                self.geo_err[tag].append(masked_joint_error_mm(geo_pred_cm[b][vf][:, :J].float(), g, gm).cpu())
            if geo_info is not None and geo_info.shape[-1] >= J:
                self.geo_info[tag].append(geo_info[b][vf][:, :J].bool().cpu())
            if ray_dir is not None and ray_dir.shape[-2] >= J:
                rd = ray_dir[b][vf][:, :J].float()
                delta = (p - g)
                along = (delta * rd).sum(-1, keepdim=True)
                self.err_ray[tag].append((along.abs()[..., 0] * CM_TO_MM).masked_fill(~gm, float("nan")).cpu())
                self.err_lat[tag].append(((delta - along * rd).norm(dim=-1) * CM_TO_MM).masked_fill(~gm, float("nan")).cpu())
            if local_std_cm is not None and local_std_cm.shape[-2] >= J:
                self.std_local[tag].append((local_std_cm[b][vf][:, :J].float() * CM_TO_MM).cpu())
            # velocity: consecutive frames of the chunk's *time axis* with GT in both (the stride used to be
            # inferred from the frames left after dropping the unannotated ones, so a gap looked adjacent); the pairs
            # that straddle two chunks are added in ``_cross_chunk_velocity`` from the chunk boundaries
            stride = self._frame_stride(frame_ids[b][valid_frames[b]].long().cpu()) if frame_ids is not None else 1
            if n > 1:
                consec = torch.ones(n - 1, dtype=torch.bool)
                if frame_ids is not None:
                    consec = (fids[1:] - fids[:-1]) == stride
                ve = self._velocity_error(p, g, gm).cpu()[consec]
                if ve.numel():
                    self.vel[tag].append(ve)
                    self.vel_fid[tag].append(fids[1:][consec])
                    self.vel_seq[tag].extend([sid] * int(consec.sum()))
            if seq_ids is not None and frame_ids is not None:
                self.bounds[tag].append((sid, stride, int(fids[0]), p[0].cpu(), g[0].cpu(), gm[0].cpu(),
                                         int(fids[-1]), p[-1].cpu(), g[-1].cpu(), gm[-1].cpu()))

    # ------------------------------------------------------------------ cross-chunk velocity / per-sequence sums
    def _cross_chunk_velocity(self, tag: str) -> Optional[torch.Tensor]:
        """Velocity errors [M, J] of the frame pairs (last frame of a chunk, first frame of the next chunk of the same
        sequence) - deduplicated by (sequence, first frame id) so a padded DistributedSampler does not count twice."""
        by_seq: Dict[str, Dict[int, Tuple]] = defaultdict(dict)
        for bnd in self.bounds.get(tag, []):
            by_seq[bnd[0]].setdefault(bnd[2], bnd)
        rows = []
        for sid, chunks in by_seq.items():
            if not sid:
                continue
            order = sorted(chunks)
            for a, b in zip(order[:-1], order[1:]):
                ca, cb = chunks[a], chunks[b]
                if cb[2] != ca[6] + ca[1]:      # not adjacent (a gap or an overlapping chunk)
                    continue
                p = torch.stack([ca[7], cb[3]])
                g = torch.stack([ca[8], cb[4]])
                gm = torch.stack([ca[9], cb[5]])
                rows.append(self._velocity_error(p, g, gm))
        return torch.cat(rows, 0) if rows else None

    def per_sequence(self) -> List[Dict]:
        """Per (dataset tag, sequence): frames, evaluated joint samples and the error sums (mm) of the main metrics -
        sums and counts, not means, so a paired bootstrap over sequences reproduces the frame-weighted metrics."""
        out: List[Dict] = []
        for tag in sorted(self.err):
            e = self._cat(self.err, tag)
            seqs = self.seq.get(tag, [])
            if e is None or len(seqs) != e.shape[0]:
                continue
            sk = get_skeleton(self.skeleton_of.get(tag, DEFAULT_SKELETON))
            pa, er = self._cat(self.pa_err, tag), self._cat(self.err_root, tag)
            dist = self._cat(self.dist, tag)
            dist = dist if (dist is not None and dist.shape == e.shape) else None
            era, erl = self._cat(self.err_ray, tag), self._cat(self.err_lat, tag)
            ray_ok = era is not None and era.shape == e.shape
            sessions = self.labels["session"].get(tag, [])
            subjects = self.labels["subject"].get(tag, [])
            arr = np.asarray(seqs)
            for sid in dict.fromkeys(seqs):
                m = torch.from_numpy(arr == sid)
                first = int(np.argmax(arr == sid))
                row = {"tag": tag, "seq_id": sid, "session": sessions[first] if len(sessions) == len(seqs) else "",
                       "subject": subjects[first] if len(subjects) == len(seqs) else "", "frames": int(m.sum())}

                def add(name: str, x: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None) -> None:
                    if x is None:
                        return
                    xs = x[m] if mask is None else x[m][mask[m]]
                    ok = _finite_or_inf(xs)
                    row[f"sum_{name}"] = float(xs[ok].double().sum())
                    row[f"n_{name}"] = int(ok.sum())

                add("mpjpe", e)
                add("pa_mpjpe", pa)
                add("mpjpe_root", er)
                add("upper", e[:, list(sk.upper_body)])
                add("lower", e[:, list(sk.lower_body)])
                if dist is not None:
                    lo, hi = self.dist_bins
                    for name, dm in (("near", dist < lo), ("mid", (dist >= lo) & (dist < hi)), ("far", dist >= hi)):
                        add(f"dist_{name}", torch.where(dm, e, torch.full_like(e, float("nan"))))
                        if ray_ok:
                            add(f"along_{name}", torch.where(dm, era, torch.full_like(era, float("nan"))))
                            add(f"lateral_{name}", torch.where(dm, erl, torch.full_like(erl, float("nan"))))
                if ray_ok:
                    add("along", era)
                    add("lateral", erl)
                out.append(row)
        return out

    # ------------------------------------------------------------------ multi-rank merge
    def export_state(self) -> Dict:
        """Picklable snapshot of everything accumulated (CPU tensors / lists) for ``torch.distributed.all_gather_object``."""
        st: Dict = {"skeleton_of": dict(self.skeleton_of)}
        for name in _FRAME_STORES + _PAIR_STORES:
            st[name] = {tag: list(v) for tag, v in getattr(self, name).items()}
        st["seq"] = {tag: list(v) for tag, v in self.seq.items()}
        st["vel_seq"] = {tag: list(v) for tag, v in self.vel_seq.items()}
        st["labels"] = {kind: {tag: list(v) for tag, v in d.items()} for kind, d in self.labels.items()}
        st["bounds"] = {tag: list(v) for tag, v in self.bounds.items()}
        return st

    def load_states(self, states: Sequence[Dict]) -> None:
        """Replace the accumulated data by the union of ``states`` (one per rank). Frames that occur on several ranks
        (same tag, sequence and frame id - DistributedSampler pads the last batch by repeating samples) are kept once."""
        self.reset()
        for st in states:
            self.skeleton_of.update(st.get("skeleton_of", {}))
            for name in _FRAME_STORES + _PAIR_STORES:
                for tag, v in st.get(name, {}).items():
                    getattr(self, name)[tag].extend(v)
            for tag, v in st.get("seq", {}).items():
                self.seq[tag].extend(v)
            for tag, v in st.get("vel_seq", {}).items():
                self.vel_seq[tag].extend(v)
            for kind, d in st.get("labels", {}).items():
                for tag, v in d.items():
                    self.labels[kind][tag].extend(v)
            for tag, v in st.get("bounds", {}).items():
                self.bounds[tag].extend(v)
        self._dedupe()

    def _dedupe(self) -> None:
        for tag in list(self.err):
            fid = self._cat(self.fid, tag)
            seqs = self.seq.get(tag, [])
            e = self._cat(self.err, tag)
            if fid is None or e is None or len(seqs) != e.shape[0] or fid.shape[0] != e.shape[0] or bool((fid < 0).any()):
                continue
            seen, keep = set(), []
            for i, key in enumerate(zip(seqs, fid.tolist())):
                keep.append(key not in seen)
                seen.add(key)
            keep_t = torch.tensor(keep, dtype=torch.bool)
            if bool(keep_t.all()):
                continue
            for name in _FRAME_STORES:
                t = self._cat(getattr(self, name), tag)
                if t is not None and t.shape[0] == keep_t.shape[0]:
                    getattr(self, name)[tag] = [t[keep_t]]
                elif t is not None:
                    getattr(self, name)[tag] = []          # misaligned optional store: drop it rather than mis-pair rows
            self.seq[tag] = [s for s, k in zip(seqs, keep) if k]
            for kind in list(self.labels):
                lab = self.labels[kind].get(tag, [])
                if len(lab) == len(keep):
                    self.labels[kind][tag] = [s for s, k in zip(lab, keep) if k]
            vf = self._cat(self.vel_fid, tag)
            vs = self.vel_seq.get(tag, [])
            v = self._cat(self.vel, tag)
            if vf is not None and v is not None and len(vs) == v.shape[0] == vf.shape[0]:
                seen, vk = set(), []
                for key in zip(vs, vf.tolist()):
                    vk.append(key not in seen)
                    seen.add(key)
                vk_t = torch.tensor(vk, dtype=torch.bool)
                self.vel[tag], self.vel_fid[tag] = [v[vk_t]], [vf[vk_t]]
                self.vel_seq[tag] = [s for s, k in zip(vs, vk) if k]

    def raw_calibration(self) -> Dict[str, np.ndarray]:
        """Per-(frame, joint) calibration inputs pooled over all tags - squared Mahalanobis distance under the predicted
        scale matrix, Euclidean error (mm), sqrt(trace S) (mm), GT distance from the device (cm) - for post-hoc
        temperature scaling. Empty when no uncertainty was accumulated."""
        out: Dict[str, np.ndarray] = {}
        if not any(self.maha.values()):
            return out
        for name, store in (("maha", self.maha), ("err", self.err), ("unc", self.unc), ("dist", self.dist)):
            t = self._cat(store, None)
            if t is not None:
                out[name] = t.numpy()
        if "err" in out and out["err"].shape != out["maha"].shape:  # tags without uncertainty would misalign the arrays
            out.pop("err")
        if "dist" in out and out["dist"].shape != out["maha"].shape:
            out.pop("dist")
        return out

    def compute(self, prefix: str = "") -> Dict[str, float]:
        out: Dict[str, float] = {}
        tags = sorted(self.err)
        for tag in tags:
            out.update(self._summary_tag(tag, f"{prefix}{tag}/"))
        # the pooled "all" summary only makes sense for tags sharing one skeleton
        skels = {self.skeleton_of.get(t, DEFAULT_SKELETON) for t in tags}
        if len(tags) > 1 and len(skels) == 1:
            out.update(self._summary_tag(None, f"{prefix}all/"))
        if len(tags) == 1:  # convenience aliases without the dataset name
            tag = tags[0]
            for k in list(out):
                if k.startswith(f"{prefix}{tag}/"):
                    out[k.replace(f"{tag}/", "", 1)] = out[k]
        return out

    def _cat(self, store: Dict[str, List[torch.Tensor]], tag: Optional[str]) -> Optional[torch.Tensor]:
        keys = [tag] if tag is not None else sorted(store)
        parts = [t for k in keys for t in store.get(k, [])]
        return torch.cat(parts, 0) if parts else None

    def _labels(self, kind: str, tag: Optional[str]) -> List[str]:
        d = self.labels.get(kind, {})
        return [g for k in ([tag] if tag is not None else sorted(self.err)) for g in d.get(k, [])]

    def _summary_tag(self, tag: Optional[str], p: str) -> Dict[str, float]:
        e = self._cat(self.err, tag)
        if e is None:
            return {}
        sk_name = self.skeleton_of.get(tag, DEFAULT_SKELETON) if tag is not None else next(iter(self.skeleton_of.values()), DEFAULT_SKELETON)
        sk = get_skeleton(sk_name)
        pa = self._cat(self.pa_err, tag)
        er = self._cat(self.err_root, tag)
        d: Dict[str, float] = {
            f"{p}mpjpe": _nanmean(e),
            f"{p}pa_mpjpe": _nanmean(pa),
            f"{p}mpjpe_root": _nanmean(er),  # mid-hip centred (translation-free, TSR-Ego style)
            f"{p}pck": float(pck(e, self.pck_thr)),
            f"{p}pck150": float(pck(e, 150.0)),   # MPI-INF-3DHP / Mo2Cap2 convention (the leaderboard threshold is to be confirmed)
            f"{p}auc": float(auc(e, self.auc_max)),
            f"{p}num_frames": float(e.shape[0]),
            f"{p}num_joints": float(_finite_or_inf(e).sum()),   # evaluated (frame, joint) samples
            f"{p}num_pa_frames": float(_finite_or_inf(pa).any(dim=1).sum()) if pa is not None else 0.0,
        }
        for part, ids in sk.body_parts.items():
            d[f"{p}part/{part}"] = _nanmean(e[:, ids])
        for j, name in enumerate(sk.joint_names):
            d[f"{p}joint/{name}"] = _nanmean(e[:, j])
        d[f"{p}group/upper"] = _nanmean(e[:, list(sk.upper_body)])
        d[f"{p}group/lower"] = _nanmean(e[:, list(sk.lower_body)])
        d[f"{p}group/lower_root"] = _nanmean(er[:, list(sk.lower_body)])
        vel_parts = [t for t in ([self._cat(self.vel, tag)] if tag is not None else [self._cat(self.vel, None)]) if t is not None]
        cross = [c for c in ([self._cross_chunk_velocity(tag)] if tag is not None else
                             [self._cross_chunk_velocity(t) for t in sorted(self.err)]) if c is not None]
        if vel_parts or cross:
            v_all = torch.cat(vel_parts + cross, 0)
            d[f"{p}mpjve"] = _nanmean(v_all)            # mm per frame (the leaderboards' MPJVE convention)
            d[f"{p}mpjve_pairs"] = float(_finite_or_inf(v_all).any(dim=1).sum())
            d[f"{p}mpjve_cross_chunk_pairs"] = float(sum(int(_finite_or_inf(c).any(dim=1).sum()) for c in cross))
        vis = self._cat(self.vis, tag)
        if vis is not None and vis.shape == e.shape:
            for k, name in ((2, "both"), (1, "one"), (0, "none")):
                m = (vis >= 2) if k == 2 else (vis == k)  # "both" = seen by at least two views (any number of cameras)
                m = m & _finite_or_inf(e)
                d[f"{p}vis/{name}"] = _nanmean(e[m]) if m.any() else float("nan")
                d[f"{p}vis/{name}_root"] = _nanmean(er[m]) if m.any() else float("nan")
                d[f"{p}vis/{name}_frac"] = float(m.float().sum() / _finite_or_inf(e).float().sum().clamp_min(1.0))
        fe = _nanmean_rows(e)
        mo = self._cat(self.motion, tag)
        if mo is not None and mo.shape[0] == e.shape[0]:
            lo, hi = self.motion_bins
            bins = {"low": mo < lo, "mid": (mo >= lo) & (mo < hi), "high": mo >= hi}
            for name, m in bins.items():
                m = m & torch.isfinite(mo)
                d[f"{p}motion/{name}"] = _nanmean(fe[m]) if m.any() else float("nan")
                d[f"{p}motion/{name}_frac"] = float(m.float().mean())
        hmo = self._cat(self.head_motion, tag)
        if hmo is not None and hmo.shape[0] == e.shape[0]:
            lo, hi = self.head_motion_bins
            for name, m in (("low", hmo < lo), ("mid", (hmo >= lo) & (hmo < hi)), ("high", hmo >= hi)):
                m = m & torch.isfinite(hmo)
                d[f"{p}headmotion/{name}"] = _nanmean(fe[m]) if m.any() else float("nan")
                d[f"{p}headmotion/{name}_frac"] = float(m.float().mean())
        for kind in sorted(self.labels):
            labels = self._labels(kind, tag)
            if len(labels) != e.shape[0] or not labels:
                continue
            arr = np.asarray(labels)
            for gname in sorted(set(labels)):
                if not gname:
                    continue
                m = torch.from_numpy(arr == gname)
                d[f"{p}{kind}/{gname}"] = _nanmean(e[m])
                d[f"{p}{kind}/{gname}_frac"] = float(m.float().mean())
        # Distance strata (the FOV-based visibility strata are degenerate on 170-degree fisheye rigs)
        dist = self._cat(self.dist, tag)
        if dist is not None and dist.shape == e.shape:
            lo, hi = self.dist_bins
            bins = {"near": dist < lo, "mid": (dist >= lo) & (dist < hi), "far": dist >= hi}
            ev = _finite_or_inf(e)
            for name, m in bins.items():
                m = m & torch.isfinite(dist) & ev
                d[f"{p}dist/{name}"] = _nanmean(e[m]) if m.any() else float("nan")
                d[f"{p}dist/{name}_root"] = _nanmean(er[m]) if m.any() else float("nan")
                d[f"{p}dist/{name}_frac"] = float(m.float().sum() / ev.float().sum().clamp_min(1.0))
        conf = self._cat(self.conf, tag)
        if conf is not None and conf.shape == e.shape:
            for name, m in (("high", conf >= self.conf_thr), ("low", conf < self.conf_thr)):
                m = m & _finite_or_inf(e)
                d[f"{p}conf/{name}"] = _nanmean(e[m]) if m.any() else float("nan")
                d[f"{p}conf/{name}_frac"] = float(m.float().mean())
        # Where does the geometry gate open, and how good is the triangulated pose there
        gate = self._cat(self.gate, tag)
        if gate is not None and gate.shape == e.shape:
            ev = _finite_or_inf(e)
            d[f"{p}geo/gate_mean"] = _nanmean(gate[ev])
            d[f"{p}geo/gate_open_frac"] = float((gate[ev] > 0.5).float().mean()) if ev.any() else float("nan")
            if dist is not None and dist.shape == e.shape:
                lo, hi = self.dist_bins
                for name, m in (("near", dist < lo), ("mid", (dist >= lo) & (dist < hi)), ("far", dist >= hi)):
                    m = m & ev
                    d[f"{p}geo/gate_{name}"] = _nanmean(gate[m]) if m.any() else float("nan")
            info = self._cat(self.geo_info, tag)
            if info is not None and info.shape == e.shape:
                info = info & ev
                d[f"{p}geo/info_frac"] = float(info.float().sum() / ev.float().sum().clamp_min(1.0))
                d[f"{p}geo/gate_info"] = _nanmean(gate[info]) if info.any() else float("nan")
                ge = self._cat(self.geo_err, tag)
                if ge is not None and ge.shape == e.shape:
                    d[f"{p}geo/mpjpe_geo_info"] = _nanmean(ge[info]) if info.any() else float("nan")
                    d[f"{p}geo/mpjpe_pred_info"] = _nanmean(e[info]) if info.any() else float("nan")
                    d[f"{p}geo/mpjpe_geo_all"] = _nanmean(ge)
        # Along-ray vs lateral decomposition of the error (and of the predicted sigma when ray-aligned)
        er_ray = self._cat(self.err_ray, tag)
        if er_ray is not None and er_ray.shape == e.shape:
            el = self._cat(self.err_lat, tag)
            d[f"{p}ray/err_along"] = _nanmean(er_ray)
            d[f"{p}ray/err_lateral"] = _nanmean(el)
            if dist is not None and dist.shape == e.shape:
                lo, hi = self.dist_bins
                for name, m in (("near", dist < lo), ("mid", (dist >= lo) & (dist < hi)), ("far", dist >= hi)):
                    d[f"{p}ray/err_along_{name}"] = _nanmean(er_ray[m]) if m.any() else float("nan")
                    d[f"{p}ray/err_lateral_{name}"] = _nanmean(el[m]) if m.any() else float("nan")
            sl = self._cat(self.std_local, tag)
            if sl is not None and sl.shape[:2] == e.shape:
                ok = torch.isfinite(er_ray)
                d[f"{p}ray/sigma_along"] = float(sl[..., 0][ok].mean()) if ok.any() else float("nan")
                d[f"{p}ray/sigma_lateral"] = float(sl[..., 1:].pow(2).sum(-1).sqrt()[ok].mean()) if ok.any() else float("nan")
                # E|z| of the training likelihood: 0.80 for N(0,1), 1.10 for a Student-t with nu = 3 (scale = 1)
                d[f"{p}ray/ratio_along"] = float((er_ray[ok] / sl[..., 0][ok].clamp_min(1e-3)).mean()) if ok.any() else float("nan")
                d[f"{p}ray/spearman_along"] = _spearman(er_ray[ok], sl[..., 0][ok])
        nll_g = self._cat(self.nll_g, tag)
        if nll_g is not None and nll_g.shape == e.shape:
            maha = self._cat(self.maha, tag)
            unc = self._cat(self.unc, tag)
            ok = torch.isfinite(maha)
            d[f"{p}calib/nll_gauss"] = _nanmean(nll_g)
            d[f"{p}calib/nll_t"] = _nanmean(self._cat(self.nll_t, tag))
            # coverage against the thresholds of the training likelihood (calibrated model -> 0.68 / 0.95) ...
            d[f"{p}calib/cov68"] = float((maha[ok] <= self.maha_thr[0]).float().mean())
            d[f"{p}calib/cov95"] = float((maha[ok] <= self.maha_thr[1]).float().mean())
            # ... and against the Gaussian chi-square ones (a calibrated Student-t(3) model only reaches 0.55 / 0.77)
            d[f"{p}calib/cov68_gauss"] = float((maha[ok] <= CHI2_3_68).float().mean())
            d[f"{p}calib/cov95_gauss"] = float((maha[ok] <= CHI2_3_95).float().mean())
            d[f"{p}calib/ause"] = sparsification_ause(e[ok].flatten(), unc[ok].flatten())
            d[f"{p}calib/mean_sigma"] = _nanmean(unc)                      # sqrt(trace S), mm (scale matrix)
            d[f"{p}calib/mean_std_t"] = _nanmean(unc) * self.t_std_factor  # sqrt(trace Cov) of the t predictive
            d[f"{p}calib/spearman"] = _spearman(e[ok].flatten(), unc[ok].flatten())
        return d


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ok = torch.isfinite(a) & torch.isfinite(b)
    a, b = a[ok], b[ok]
    if a.numel() < 3:
        return float("nan")
    ra = torch.argsort(torch.argsort(a)).double()
    rb = torch.argsort(torch.argsort(b)).double()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()).clamp_min(1e-12))

"""Fisheye camera models, stereo / multi-view rigs and the batched ``RigBank`` used inside the model.

Two camera models are implemented behind one interface (``FisheyeCamera``):

* ``ScaramuzzaCamera``   - omnidirectional polynomial model (UnrealEgo / UnrealEgo2 / UnrealEgo-RW / Ego4View
                           calibrations). Projection follows EgoPoseFormer v1 ``pose_estimation/models/utils/camera_models.py``
                           (Apache-2.0, Chenhongyi Yang) and EgoRear ``pose_estimation/utils/camera_models.py``
                           (Apache-2.0, Hiroyasu Akada) so that our 2D anchors match theirs exactly on UnrealEgo.
* ``OpenCVFisheyeCamera`` - OpenCV / Kannala-Brandt equidistant model with k1..k4 (the alternative calibration shipped
                           with UnrealEgo(-RW), typical for consumer headsets such as the Quest 3 rig of FRAME).

Both provide projection, back-projection (unit rays), ray maps for resized images (ray positional encoding) and
gnomonic (tangent-plane) patch sampling grids for fisheye-aware ViT tokenization (FisheyeViT idea, Wang et al. CVPR 2024).

``StereoRig`` groups the calibrated cameras of one device; ``RigBank`` packs the cameras of *all* rigs used by a model
into padded parameter tensors so that projection / back-projection of a mixed batch (several rigs, any number of
views) is a single batched computation without Python loops over rigs or views - the decoder projects tens of
thousands of sampling points per layer and the per-view / per-rig loops were the dominant latency term.

Conventions: OpenCV camera frame (x right, y down, z forward), lengths in centimetres, pixel coordinates in the
*original* calibration resolution, ``uv_norm`` in [0, 1]. ``p_cam = R @ p_device + t``.
"""

from __future__ import annotations

import functools
import json
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _fp32_geometry(fn):
    """no_grad + autocast off for the dense geometry cached by the encoders (token grids, LUTs, ray maps).

    They are first built inside the bf16-mixed forward and then reused for the whole run: under autocast the matmuls
    run in bf16 and the nearest-block argmax of ``real_to_virtual_lut`` picked a wrong block for 9-28 % of the
    UnrealEgo-RW pixels.
    """

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with torch.no_grad(), torch.autocast(device_type=self.center.device.type, enabled=False):
            return fn(self, *args, **kwargs)

    return wrapper

# --------------------------------------------------------------------------------------
# UnrealEgo / UnrealEgo2 calibration (hiroyasuakada/UnrealEgo utils/fisheye_calibration_UEP.json).
# Left and right cameras share intrinsics; the device frame is the midpoint of the two cameras
# (baseline 12 cm along x, verified from EgoPoseFormer's pelvis_pos.pkl: right - left = +12 cm).
# --------------------------------------------------------------------------------------
UNREALEGO_CALIB = {
    "name": "unreal_ego_pose",
    "size": [1024, 1024],
    "imageCircleRadius": 512,
    "image_center": [511.1183388444314, 510.8730105600536],
    "polynomialC2W": [
        -516.7691300636768, 0.0, 0.001827567433476, -0.000009458375642153957,
        4.460350926140045e-8, -9.434841474792276e-11, 8.099831344095236e-14,
    ],
    "polynomialW2C": [
        541.084422, 133.996745, -53.833198, 60.96083, -24.78051, 12.451492,
        -30.240511, 26.90122, 116.38499, -133.991117, -141.904687, 184.05592,
        107.45616, -125.552875, -55.66342, 44.209519, 18.234651, -6.410899, -2.737066,
    ],
    "affine": [0.999997845608931, -0.00003641214346960503, -0.0000492182252254407],
}
# p_cam = R @ p_device + t   (cm). Left camera sits at device x = +6, right at x = -6.
UNREALEGO_EXTRINSICS = {
    "left": {"R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "t": [-6.0, 0.0, 0.0]},
    "right": {"R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "t": [6.0, 0.0, 0.0]},
}

MODEL_SCARAMUZZA, MODEL_OPENCV = 0, 1
_NEWTON_ITERS = 8


def polyval(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """sum_i coeffs[i] * x**i (Horner). ``coeffs`` [D] shared, or [..., D] broadcasting against ``x[..., None]``
    over the leading dims (batched cameras: coeffs [N, V, 1, D] vs x [N, V, M])."""
    if coeffs.ndim == 1:
        out = torch.zeros_like(x)
        for a in torch.flip(coeffs, dims=[0]):
            out = out * x + a
        return out
    out = torch.zeros_like(x)
    for k in range(coeffs.shape[-1] - 1, -1, -1):
        out = out * x + coeffs[..., k]
    return out


# --------------------------------------------------------------------------------------
# functional, batched projection / back-projection (parameters broadcast against the point dims)
# --------------------------------------------------------------------------------------
def scaramuzza_project(p_cam: torch.Tensor, w2c: torch.Tensor, affine: torch.Tensor, center: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """p_cam [..., 3]; w2c [..., D], affine [..., 3], center [..., 2] broadcastable to p_cam[..., 0] -> uv [..., 2] (px)."""
    x, y, z = p_cam[..., 0], p_cam[..., 1], p_cam[..., 2]
    norm = torch.sqrt(x * x + y * y).clamp_min(eps)
    theta = torch.atan2(-z, norm)
    rho = polyval(w2c, theta)
    us, vs = x / norm * rho, y / norm * rho
    c, d, e = affine[..., 0], affine[..., 1], affine[..., 2]
    u = c * us + d * vs + center[..., 0]
    v = e * us + vs + center[..., 1]
    return torch.stack((u, v), dim=-1)


def scaramuzza_unproject(uv: torch.Tensor, c2w: torch.Tensor, affine: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """Pixel coords [..., 2] -> unit rays [..., 3] in the camera frame."""
    du, dv = uv[..., 0] - center[..., 0], uv[..., 1] - center[..., 1]
    c, d, e = affine[..., 0], affine[..., 1], affine[..., 2]
    det = c - d * e
    xs = (du - d * dv) / det
    ys = (-e * du + c * dv) / det
    rho = torch.sqrt(xs * xs + ys * ys)
    zs = polyval(c2w, rho)  # Scaramuzza z: negative along the optical axis
    return F.normalize(torch.stack((xs, ys, -zs), dim=-1), dim=-1)


def opencv_project(p_cam: torch.Tensor, K: torch.Tensor, D: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """OpenCV fisheye (Kannala-Brandt). p_cam [..., 3]; K [..., 4] = (fx, fy, cx, cy); D [..., 4] = (k1..k4).
    theta = atan2(sqrt(x^2 + y^2), z) so that points beyond 90 deg (fisheye FOV > 180) project consistently."""
    x, y, z = p_cam[..., 0], p_cam[..., 1], p_cam[..., 2]
    rho = torch.sqrt(x * x + y * y).clamp_min(eps)
    theta = torch.atan2(rho, z)
    t2 = theta * theta
    theta_d = theta * (1.0 + t2 * (D[..., 0] + t2 * (D[..., 1] + t2 * (D[..., 2] + t2 * D[..., 3]))))
    xd, yd = theta_d * x / rho, theta_d * y / rho
    u = K[..., 0] * xd + K[..., 2]
    v = K[..., 1] * yd + K[..., 3]
    return torch.stack((u, v), dim=-1)


def opencv_unproject(uv: torch.Tensor, K: torch.Tensor, D: torch.Tensor, iters: int = _NEWTON_ITERS) -> torch.Tensor:
    """Pixel coords [..., 2] -> unit rays [..., 3]: invert theta_d(theta) with Newton iterations (differentiable)."""
    xd = (uv[..., 0] - K[..., 2]) / K[..., 0]
    yd = (uv[..., 1] - K[..., 3]) / K[..., 1]
    theta_d = torch.sqrt(xd * xd + yd * yd)
    k1, k2, k3, k4 = D[..., 0], D[..., 1], D[..., 2], D[..., 3]
    theta = theta_d.clone()
    for _ in range(iters):
        t2 = theta * theta
        f = theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - theta_d
        fp = 1.0 + t2 * (3.0 * k1 + t2 * (5.0 * k2 + t2 * (7.0 * k3 + t2 * 9.0 * k4)))
        theta = theta - f / fp.clamp_min(1e-6)
    scale = torch.where(theta_d > 1e-9, torch.sin(theta) / theta_d.clamp_min(1e-9), torch.ones_like(theta_d))
    return F.normalize(torch.stack((xd * scale, yd * scale, torch.cos(theta)), dim=-1), dim=-1)


# --------------------------------------------------------------------------------------
# camera modules
# --------------------------------------------------------------------------------------
class FisheyeCamera(nn.Module):
    """One calibrated fisheye camera rigidly attached to the device frame. Parameters are buffers so the camera
    moves with the model. Subclasses implement ``project_cam`` / ``unproject_pixel`` and ``calib_dict``."""

    model_id: int = -1

    def __init__(
        self,
        name: str,
        size_wh: Sequence[int],
        image_center: Sequence[float],
        R: Optional[Sequence[Sequence[float]]] = None,
        t: Optional[Sequence[float]] = None,
        image_circle_radius: Optional[float] = None,
    ):
        super().__init__()
        self.name = name
        self.width, self.height = int(size_wh[0]), int(size_wh[1])
        self.radius = float(image_circle_radius) if image_circle_radius else min(self.width, self.height) / 2.0
        self.register_buffer("center", torch.tensor(image_center, dtype=torch.float32), persistent=False)
        R = torch.eye(3) if R is None else torch.tensor(R, dtype=torch.float32)
        t = torch.zeros(3) if t is None else torch.tensor(t, dtype=torch.float32)
        self.register_buffer("R", R, persistent=False)
        self.register_buffer("t", t, persistent=False)

    # ---------------- rigid transforms ----------------
    def device_to_cam(self, p_dev: torch.Tensor) -> torch.Tensor:
        return p_dev @ self.R.T + self.t

    def cam_to_device(self, p_cam: torch.Tensor) -> torch.Tensor:
        return (p_cam - self.t) @ self.R

    def ray_cam_to_device(self, d_cam: torch.Tensor) -> torch.Tensor:
        return d_cam @ self.R  # R^T d

    # ---------------- projection (subclass) ----------------
    def project_cam_px(self, p_cam: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def unproject_pixel(self, uv: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def calib_dict(self) -> Dict:  # pragma: no cover - abstract
        raise NotImplementedError

    def project_cam(self, p_cam: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Camera-frame points [..., 3] -> pixel coords [..., 2] (original resolution) and in-FOV mask."""
        uv = self.project_cam_px(p_cam)
        u, v = uv[..., 0], uv[..., 1]
        in_fov = (u > 0) & (u < self.width) & (v > 0) & (v < self.height)
        return uv, in_fov

    def project_cam_norm(self, p_cam: torch.Tensor, clamp: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Same as project_cam but normalised to [0, 1] (EgoPoseFormer v1 convention)."""
        uv, in_fov = self.project_cam(p_cam)
        uv = uv / uv.new_tensor([self.width, self.height])
        if clamp:
            uv = uv.clamp(0.0, 1.0)
        return uv, in_fov

    def unproject_norm(self, uv_norm: torch.Tensor) -> torch.Tensor:
        """Normalised [0,1] image coords [..., 2] -> unit rays [..., 3] in the camera frame (differentiable)."""
        uv = uv_norm * uv_norm.new_tensor([float(self.width), float(self.height)])
        return self.unproject_pixel(uv)

    def pixel_in_circle(self, uv: torch.Tensor) -> torch.Tensor:
        return ((uv - self.center) ** 2).sum(-1).sqrt() <= self.radius

    # ---------------- dense geometry ----------------
    def pixel_centers(self, h: int, w: int) -> torch.Tensor:
        """Pixel centres of an (h, w) resized image expressed in original pixel coordinates -> [h, w, 2]."""
        dev = self.center.device
        us = (torch.arange(w, device=dev, dtype=torch.float32) + 0.5) * (self.width / w)
        vs = (torch.arange(h, device=dev, dtype=torch.float32) + 0.5) * (self.height / h)
        vv, uu = torch.meshgrid(vs, us, indexing="ij")
        return torch.stack((uu, vv), dim=-1)

    @_fp32_geometry
    def ray_map(self, h: int, w: int, frame: str = "device") -> torch.Tensor:
        """Unit ray direction per pixel of an (h, w) image -> [3, h, w]; zeros outside the image circle."""
        uv = self.pixel_centers(h, w)
        rays = self.unproject_pixel(uv)
        if frame == "device":
            rays = self.ray_cam_to_device(rays)
        rays = rays * self.pixel_in_circle(uv)[..., None]
        return rays.permute(2, 0, 1).contiguous()

    @_fp32_geometry
    def _gnomonic_basis(self, n: int, fov_scale: float = 1.0):
        """Per token block: centre ray r, image-aligned tangent basis (e_u, e_v), tangent half-extent s, validity."""
        dev = self.center.device
        centers = self.pixel_centers(n, n)  # [n, n, 2]
        valid = self.pixel_in_circle(centers)
        r = self.unproject_pixel(centers)  # [n, n, 3]

        # image-aligned local basis: e_v along increasing v, e_u = e_v x r (x right, y down, z fwd)
        step = 0.5 * self.height / n
        below = self.unproject_pixel(centers + centers.new_tensor([0.0, step]))
        e_v = below - r
        e_v = e_v - (e_v * r).sum(-1, keepdim=True) * r
        e_v = F.normalize(e_v, dim=-1)
        e_u = F.normalize(torch.cross(e_v, r, dim=-1), dim=-1)

        # angular half-extent of a token from neighbouring centre rays
        def ang(a, b):
            return torch.acos((a * b).sum(-1).clamp(-1.0, 1.0))

        ang_h = torch.zeros(n, n, device=dev)
        ang_v = torch.zeros(n, n, device=dev)
        ang_h[:, :-1] = ang(r[:, :-1], r[:, 1:])
        ang_h[:, -1] = ang_h[:, -2]
        ang_v[:-1, :] = ang(r[:-1, :], r[1:, :])
        ang_v[-1, :] = ang_v[-2, :]
        alpha = 0.5 * (ang_h + ang_v)
        alpha = torch.where(valid, alpha, torch.full_like(alpha, math.radians(5.0)))
        s = torch.tan((0.5 * alpha * fov_scale).clamp(max=math.radians(80.0)))  # [n, n]
        return r, e_u, e_v, s, valid

    @_fp32_geometry
    def real_to_virtual_lut(self, n_tokens: int, patch_px: int, out_hw: Tuple[int, int], fov_scale: float = 1.0) -> torch.Tensor:
        """Exact inverse of ``gnomonic_grid`` on a grid of *real* pixels.

        For every pixel centre of an (h, w) real-layout map, back-project to a ray, pick the token block whose centre
        ray is closest, and project the ray gnomonically onto that block's tangent plane: the tangent coordinates
        (a, b) in [-1, 1] give the position inside the block. Returns [h, w, 2] normalised virtual-image coordinates in
        [-1, 1] (``F.grid_sample`` convention), -2 outside the image circle. Used by StereoFisheyeViT(feature_lut=True)
        to resample its virtual-layout feature map into the real image layout that the decoder samples with real uv.
        """
        n, p = n_tokens, patch_px
        h, w = out_hw
        r, e_u, e_v, s, valid = self._gnomonic_basis(n, fov_scale)              # [n, n, ...]
        uv = self.pixel_centers(h, w)                                             # real pixel centres [h, w, 2]
        d = self.unproject_pixel(uv)                                              # [h, w, 3]
        ok = self.pixel_in_circle(uv)
        rc = r.reshape(-1, 3)                                                     # [n*n, 3]
        dots = d.reshape(-1, 3) @ rc.T                                            # [h*w, n*n]
        dots = dots.masked_fill(~valid.reshape(1, -1), -2.0)
        blk = dots.argmax(dim=1)                                                  # nearest block per real pixel
        bi, bj = blk // n, blk % n
        rb, eu, ev, sb = r[bi, bj], e_u[bi, bj], e_v[bi, bj], s[bi, bj]           # [h*w, 3] / [h*w]
        dd = d.reshape(-1, 3)
        depth = (dd * rb).sum(-1).clamp_min(1e-6)
        a = (dd * eu).sum(-1) / depth / sb                                        # tangent coords in units of the half patch
        b = (dd * ev).sum(-1) / depth / sb
        a, b = a.clamp(-1.0, 1.0), b.clamp(-1.0, 1.0)
        # virtual pixel (continuous): block origin + (coord + 1) / 2 * p ; then normalised to [-1, 1] over n*p pixels
        vx = (bj.float() + (a + 1.0) * 0.5) * p
        vy = (bi.float() + (b + 1.0) * 0.5) * p
        lut = torch.stack([vx / (n * p) * 2.0 - 1.0, vy / (n * p) * 2.0 - 1.0], dim=-1).reshape(h, w, 2)
        return torch.where(ok[..., None], lut, torch.full_like(lut, -2.0))

    @_fp32_geometry
    def gnomonic_grid(
        self, n_tokens: int, patch_px: int, fov_scale: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tangent-plane (gnomonic) patch sampling grid for fisheye-aware ViT tokenization.

        The image is divided into n_tokens x n_tokens tokens. For each token the centre ray is
        back-projected, a tangent plane is spanned by an image-aligned local basis and a
        patch_px x patch_px grid on that plane is projected back to the fisheye image. Feeding
        ``F.grid_sample(img, grid[None], align_corners=False)`` yields a "virtual" undistorted
        image of size (n_tokens*patch_px)^2 whose 16x16 blocks are undistorted patches.

        Returns:
            grid      [n*p, n*p, 2]  normalised sampling coords in [-1, 1] (outside -> < -1)
            rays_dev  [3, n*p, n*p]  device-frame unit ray of every virtual pixel (0 if invalid)
            valid     [n, n]         token validity (centre inside the image circle)
        """
        n, p = n_tokens, patch_px
        dev = self.center.device
        r, e_u, e_v, s, valid = self._gnomonic_basis(n, fov_scale)

        lin = (torch.arange(p, device=dev, dtype=torch.float32) + 0.5) / p * 2.0 - 1.0
        bb, aa = torch.meshgrid(lin, lin, indexing="ij")  # bb rows (v), aa cols (u): [p, p]
        dirs = (
            r[:, :, None, None, :]
            + s[:, :, None, None, None] * (aa[None, None, :, :, None] * e_u[:, :, None, None, :]
                                            + bb[None, None, :, :, None] * e_v[:, :, None, None, :])
        )  # [n, n, p, p, 3]
        dirs = F.normalize(dirs, dim=-1)
        uv, in_fov = self.project_cam(dirs)
        ok = in_fov & self.pixel_in_circle(uv) & valid[:, :, None, None]
        gx = uv[..., 0] / self.width * 2.0 - 1.0
        gy = uv[..., 1] / self.height * 2.0 - 1.0
        grid = torch.stack((gx, gy), dim=-1)
        grid = torch.where(ok[..., None], grid, torch.full_like(grid, -2.0))
        rays_dev = self.ray_cam_to_device(dirs) * ok[..., None]

        # [n, n, p, p, *] -> [n*p, n*p, *]
        grid = grid.permute(0, 2, 1, 3, 4).reshape(n * p, n * p, 2).contiguous()
        rays_dev = rays_dev.permute(4, 0, 2, 1, 3).reshape(3, n * p, n * p).contiguous()
        return grid, rays_dev, valid

    def extra_repr(self) -> str:
        return f"name={self.name}, model={type(self).__name__}, size=({self.width}, {self.height}), t={self.t.tolist()}"


class ScaramuzzaCamera(FisheyeCamera):
    model_id = MODEL_SCARAMUZZA

    def __init__(
        self,
        name: str,
        size_wh: Sequence[int],
        image_center: Sequence[float],
        poly_w2c: Sequence[float],
        poly_c2w: Sequence[float],
        affine: Sequence[float] = (1.0, 0.0, 0.0),
        R: Optional[Sequence[Sequence[float]]] = None,
        t: Optional[Sequence[float]] = None,
        image_circle_radius: Optional[float] = None,
    ):
        super().__init__(name, size_wh, image_center, R, t, image_circle_radius)
        self.register_buffer("w2c", torch.tensor(poly_w2c, dtype=torch.float32), persistent=False)
        self.register_buffer("c2w", torch.tensor(poly_c2w, dtype=torch.float32), persistent=False)
        self.register_buffer("affine", torch.tensor(affine, dtype=torch.float32), persistent=False)

    def project_cam_px(self, p_cam: torch.Tensor) -> torch.Tensor:
        return scaramuzza_project(p_cam, self.w2c, self.affine, self.center)

    def unproject_pixel(self, uv: torch.Tensor) -> torch.Tensor:
        return scaramuzza_unproject(uv, self.c2w, self.affine, self.center)

    def calib_dict(self) -> Dict:
        return {
            "model": "scaramuzza",
            "size": [self.width, self.height],
            "image_center": self.center.tolist(),
            "polynomialW2C": self.w2c.tolist(),
            "polynomialC2W": self.c2w.tolist(),
            "affine": self.affine.tolist(),
            "imageCircleRadius": self.radius,
        }


class OpenCVFisheyeCamera(FisheyeCamera):
    """OpenCV fisheye model: K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], D = [k1, k2, k3, k4]."""

    model_id = MODEL_OPENCV

    def __init__(
        self,
        name: str,
        size_wh: Sequence[int],
        K: Sequence[Sequence[float]],
        D: Sequence[float],
        R: Optional[Sequence[Sequence[float]]] = None,
        t: Optional[Sequence[float]] = None,
        image_circle_radius: Optional[float] = None,
    ):
        K = torch.tensor(K, dtype=torch.float32).reshape(3, 3)
        super().__init__(name, size_wh, (float(K[0, 2]), float(K[1, 2])), R, t, image_circle_radius)
        self.register_buffer("K4", torch.stack([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]), persistent=False)
        d = torch.tensor(D, dtype=torch.float32).reshape(-1)
        d = torch.cat([d, d.new_zeros(max(0, 4 - d.numel()))])[:4]
        self.register_buffer("D4", d, persistent=False)

    def project_cam_px(self, p_cam: torch.Tensor) -> torch.Tensor:
        return opencv_project(p_cam, self.K4, self.D4)

    def unproject_pixel(self, uv: torch.Tensor) -> torch.Tensor:
        return opencv_unproject(uv, self.K4, self.D4)

    def calib_dict(self) -> Dict:
        fx, fy, cx, cy = self.K4.tolist()
        return {
            "model": "opencv_fisheye",
            "size": [self.width, self.height],
            "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            "D": self.D4.tolist(),
            "imageCircleRadius": self.radius,
        }


# --------------------------------------------------------------------------------------
# rigs
# --------------------------------------------------------------------------------------
class StereoRig(nn.Module):
    """A set of calibrated cameras rigidly attached to the device frame (any number of views)."""

    def __init__(self, cameras: Sequence[FisheyeCamera], name: str = "rig"):
        super().__init__()
        self.name = name
        self.cameras = nn.ModuleList(list(cameras))

    def __len__(self) -> int:
        return len(self.cameras)

    @property
    def num_views(self) -> int:
        return len(self.cameras)

    @property
    def names(self) -> List[str]:
        return [c.name for c in self.cameras]

    @property
    def translations(self) -> torch.Tensor:
        """Device origin expressed in every camera frame -> [V, 3] (this is ``t``)."""
        return torch.stack([c.t for c in self.cameras], dim=0)

    @property
    def rotations(self) -> torch.Tensor:
        return torch.stack([c.R for c in self.cameras], dim=0)  # [V, 3, 3]

    def project(
        self, p_target: torch.Tensor, origin_cam: Optional[torch.Tensor] = None, normalize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project target-frame points into every view.

        p_target   [B, J, 3]        points in the target frame (device- or pelvis-relative)
        origin_cam [B, V, 1, 3]     target-frame origin expressed in each camera frame
                                    (None -> device frame, i.e. origin = camera t)
        returns    uv [B, V, J, 2] (normalised to [0,1] if normalize), in_fov [B, V, J]
        """
        outs, fovs = [], []
        for v, cam in enumerate(self.cameras):
            p_cam = p_target @ cam.R.T
            p_cam = p_cam + (cam.t if origin_cam is None else origin_cam[:, v])
            uv, in_fov = cam.project_cam_norm(p_cam) if normalize else cam.project_cam(p_cam)
            outs.append(uv)
            fovs.append(in_fov)
        return torch.stack(outs, dim=1), torch.stack(fovs, dim=1)

    def rays(self, uv_norm: torch.Tensor, origin_cam: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Back-project normalised image points of every view into the *target* frame.

        uv_norm    [B, V, M, 2]   normalised [0,1] coordinates in each view
        origin_cam [B, V, 1, 3]   target-frame origin expressed in each camera frame (as in ``project``)
        returns    o [B, V, 1, 3] camera centres in the target frame, d [B, V, M, 3] unit ray directions
        """
        os_, ds = [], []
        for v, cam in enumerate(self.cameras):
            d_cam = cam.unproject_norm(uv_norm[:, v])          # [B, M, 3]
            ds.append(d_cam @ cam.R)                            # R^T d
            os_.append(-(origin_cam[:, v] @ cam.R))             # p_cam = 0  ->  p_target = -R^T origin
        return torch.stack(os_, dim=1), torch.stack(ds, dim=1)

    def centers(self, origin_cam: torch.Tensor) -> torch.Tensor:
        """Camera centres in the target frame -> [B, V, 3]."""
        return torch.stack([-(origin_cam[:, v, 0] @ cam.R) for v, cam in enumerate(self.cameras)], dim=1)

    @torch.no_grad()
    def ray_maps(self, h: int, w: int) -> torch.Tensor:
        return torch.stack([c.ray_map(h, w, frame="device") for c in self.cameras], dim=0)  # [V, 3, h, w]

    @torch.no_grad()
    def gnomonic_grids(self, n_tokens: int, patch_px: int, fov_scale: float = 1.0):
        grids, rays, valids = zip(*[c.gnomonic_grid(n_tokens, patch_px, fov_scale) for c in self.cameras])
        return torch.stack(grids), torch.stack(rays), torch.stack(valids)  # [V, np, np, 2], [V, 3, np, np], [V, n, n]

    @torch.no_grad()
    def real_to_virtual_luts(self, n_tokens: int, patch_px: int, out_hw: Tuple[int, int], fov_scale: float = 1.0) -> torch.Tensor:
        """[V, h, w, 2] real-layout -> virtual-image sampling grids (see ``FisheyeCamera.real_to_virtual_lut``)."""
        return torch.stack([c.real_to_virtual_lut(n_tokens, patch_px, out_hw, fov_scale) for c in self.cameras])


class RigBank(nn.Module):
    """All rigs of a model packed into padded per-(rig, view) parameter buffers for loop-free batched geometry.

    ``project`` / ``rays`` / ``centers`` take ``rig_ids`` [N] (index into the rig list) and an optional per-sample
    rotation ``R_target`` [N, 3, 3] mapping the *target* frame to the device frame (``p_dev = R_target @ p_target``;
    identity for the pelvis / root / device protocols, the gravity-aligned rotation for the floor protocol).
    All rigs must have the same number of views. Both camera models may be mixed; each family is evaluated on the
    whole batch and selected per (sample, view) - negligible cost for the small point sets involved.
    """

    def __init__(self, rigs: Sequence[StereoRig]):
        super().__init__()
        assert len(rigs) > 0
        V = rigs[0].num_views
        assert all(r.num_views == V for r in rigs), "all rigs of one model must have the same number of views"
        self.num_rigs, self.num_views = len(rigs), V
        self.names = [r.name for r in rigs]
        cams = [c for r in rigs for c in r.cameras]
        Dw = max([c.w2c.numel() for c in cams if isinstance(c, ScaramuzzaCamera)] + [1])
        Dc = max([c.c2w.numel() for c in cams if isinstance(c, ScaramuzzaCamera)] + [1])

        def pack(fn, shape):
            out = torch.zeros(len(rigs), V, *shape, dtype=torch.float32)
            for i, r in enumerate(rigs):
                for v, c in enumerate(r.cameras):
                    val = fn(c)
                    if val is not None:
                        val = torch.as_tensor(val, dtype=torch.float32)
                        out[i, v][tuple(slice(0, s) for s in val.shape)] = val
            return out

        self.register_buffer("R", pack(lambda c: c.R, (3, 3)), persistent=False)
        self.register_buffer("t", pack(lambda c: c.t, (3,)), persistent=False)
        self.register_buffer("center", pack(lambda c: c.center, (2,)), persistent=False)
        self.register_buffer("size", pack(lambda c: torch.tensor([float(c.width), float(c.height)]), (2,)), persistent=False)
        self.register_buffer("radius", pack(lambda c: torch.tensor(c.radius), ()), persistent=False)
        self.register_buffer("model", pack(lambda c: torch.tensor(float(c.model_id)), ()).long(), persistent=False)
        self.register_buffer("w2c", pack(lambda c: c.w2c if isinstance(c, ScaramuzzaCamera) else None, (Dw,)), persistent=False)
        self.register_buffer("c2w", pack(lambda c: c.c2w if isinstance(c, ScaramuzzaCamera) else None, (Dc,)), persistent=False)
        self.register_buffer("affine", pack(lambda c: c.affine if isinstance(c, ScaramuzzaCamera) else torch.tensor([1.0, 0.0, 0.0]), (3,)), persistent=False)
        self.register_buffer("K4", pack(lambda c: c.K4 if isinstance(c, OpenCVFisheyeCamera) else torch.tensor([1.0, 1.0, 0.0, 0.0]), (4,)), persistent=False)
        self.register_buffer("D4", pack(lambda c: c.D4 if isinstance(c, OpenCVFisheyeCamera) else None, (4,)), persistent=False)
        self.has_scaramuzza = bool((self.model == MODEL_SCARAMUZZA).any())
        self.has_opencv = bool((self.model == MODEL_OPENCV).any())
        # With ``circle_fov`` a projection is inside a view only within the image circle as well
        # (outside it a fisheye image is black); off = the image rectangle only (the default)
        self.circle_fov = False

    def valid_px(self, uv: torch.Tensor, rig_ids: torch.Tensor, circle: Optional[bool] = None) -> torch.Tensor:
        """uv [N, V, M, 2] original pixels -> [N, V, M] inside the image rectangle (and the image circle if ``circle``,
        default ``self.circle_fov``). The one validity rule shared by projections, warped images and warped labels."""
        size = self.size.index_select(0, rig_ids)[:, :, None]                 # [N, V, 1, 2]
        ok = (uv > 0).all(-1) & (uv < size).all(-1)
        if self.circle_fov if circle is None else circle:
            c = self.center.index_select(0, rig_ids)[:, :, None]
            r = self.radius.index_select(0, rig_ids)[:, :, None]
            ok = ok & ((uv - c).norm(dim=-1) <= r)
        return ok

    # ---------------- helpers ----------------
    def _cam_rot(self, rig_ids: torch.Tensor, R_target: Optional[torch.Tensor]) -> torch.Tensor:
        """Rotation target -> camera for every (sample, view): R_cam @ R_target -> [N, V, 3, 3]."""
        R = self.R.index_select(0, rig_ids)  # [N, V, 3, 3]
        if R_target is not None:
            R = R @ R_target.float()[:, None]
        return R

    def _project_px(self, p_cam: torch.Tensor, rig_ids: torch.Tensor) -> torch.Tensor:
        """p_cam [N, V, M, 3] -> uv [N, V, M, 2] (original pixels)."""
        g = lambda buf: buf.index_select(0, rig_ids)[:, :, None]  # noqa: E731 - [N, V, 1, ...]
        if self.has_scaramuzza and not self.has_opencv:
            return scaramuzza_project(p_cam, g(self.w2c), g(self.affine), g(self.center))
        if self.has_opencv and not self.has_scaramuzza:
            return opencv_project(p_cam, g(self.K4), g(self.D4))
        uv_s = scaramuzza_project(p_cam, g(self.w2c), g(self.affine), g(self.center))
        uv_o = opencv_project(p_cam, g(self.K4), g(self.D4))
        sel = (self.model.index_select(0, rig_ids) == MODEL_OPENCV)[:, :, None, None]
        return torch.where(sel, uv_o, uv_s)

    def _unproject_px(self, uv: torch.Tensor, rig_ids: torch.Tensor) -> torch.Tensor:
        g = lambda buf: buf.index_select(0, rig_ids)[:, :, None]  # noqa: E731
        if self.has_scaramuzza and not self.has_opencv:
            return scaramuzza_unproject(uv, g(self.c2w), g(self.affine), g(self.center))
        if self.has_opencv and not self.has_scaramuzza:
            return opencv_unproject(uv, g(self.K4), g(self.D4))
        d_s = scaramuzza_unproject(uv, g(self.c2w), g(self.affine), g(self.center))
        d_o = opencv_unproject(uv, g(self.K4), g(self.D4))
        sel = (self.model.index_select(0, rig_ids) == MODEL_OPENCV)[:, :, None, None]
        return torch.where(sel, d_o, d_s)

    # ---------------- batched geometry ----------------
    # ``delta`` is an optional SE(3) perturbation of every camera *in its own
    # frame*: p_cam_true = dR @ p_cam_calib + dt with dR [N, V, 3, 3], dt [N, V, 3]. It carries the learnable
    # per-rig extrinsic refinement and the training-time calibration jitter of ``EgoRayFormer`` (the UnrealEgo-RW
    # sessions differ by 0.14-1.31 deg / 0.1-0.26 cm from the official calibration).
    def project(
        self,
        points: torch.Tensor,
        origin_cam: Optional[torch.Tensor],
        rig_ids: torch.Tensor,
        R_target: Optional[torch.Tensor] = None,
        normalize: bool = True,
        clamp: bool = True,
        delta: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """points [N, M, 3] (target frame); origin_cam [N, V, 1, 3] (target origin in each camera frame; None ->
        device frame); rig_ids [N] -> uv [N, V, M, 2] ([0,1] if normalize), in_fov [N, V, M]."""
        points = points.float()
        Rc = self._cam_rot(rig_ids, R_target)                                  # [N, V, 3, 3]
        p_cam = torch.einsum("nvij,nmj->nvmi", Rc, points)                    # [N, V, M, 3]
        origin = self.t.index_select(0, rig_ids)[:, :, None] if origin_cam is None else origin_cam.float()
        p_cam = p_cam + origin
        if delta is not None:
            dR, dt = delta
            p_cam = torch.einsum("nvij,nvmj->nvmi", dR.float(), p_cam) + dt.float()[:, :, None]
        uv = self._project_px(p_cam, rig_ids)
        size = self.size.index_select(0, rig_ids)[:, :, None]                 # [N, V, 1, 2]
        in_fov = self.valid_px(uv, rig_ids)
        aug_dR = getattr(self, "aug_dR", None)
        if aug_dR is not None:  # rig augmentation: the warped image only has content where the original camera saw
            p_src = torch.einsum("nvji,nvmj->nvmi", aug_dR.index_select(0, rig_ids), p_cam)   # dR^T p_cam
            in_fov = in_fov & self.valid_px(self._project_px(p_src, rig_ids), rig_ids)
        if normalize:
            uv = uv / size
            if clamp:
                uv = uv.clamp(0.0, 1.0)
        return uv, in_fov

    def rays(
        self, uv_norm: torch.Tensor, origin_cam: torch.Tensor, rig_ids: torch.Tensor, R_target: Optional[torch.Tensor] = None,
        delta: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """uv_norm [N, V, M, 2] -> camera centres o [N, V, 1, 3] and unit directions d [N, V, M, 3] in the target frame."""
        uv = uv_norm.float() * self.size.index_select(0, rig_ids)[:, :, None]
        d_cam = self._unproject_px(uv, rig_ids)                               # [N, V, M, 3]
        Rc = self._cam_rot(rig_ids, R_target)                                  # target -> cam
        origin = origin_cam.float()
        if delta is not None:  # p_cam_true = dR p_cam + dt  ->  d_cam = dR^T d_true, centre at p_cam = -dR^T dt
            dR, dt = delta
            d_cam = torch.einsum("nvji,nvmj->nvmi", dR.float(), d_cam)
            origin = origin + torch.einsum("nvji,nvj->nvi", dR.float(), dt.float())[:, :, None]
        d = torch.einsum("nvji,nvmj->nvmi", Rc, d_cam)                        # Rc^T d
        o = -torch.einsum("nvji,nvkj->nvki", Rc, origin)                      # p_cam = 0 -> -Rc^T origin, [N, V, 1, 3]
        return o, d

    def centers(
        self, origin_cam: torch.Tensor, rig_ids: torch.Tensor, R_target: Optional[torch.Tensor] = None,
        delta: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Camera centres in the target frame -> [N, V, 3]."""
        Rc = self._cam_rot(rig_ids, R_target)
        origin = origin_cam.float()[:, :, 0]
        if delta is not None:
            dR, dt = delta
            origin = origin + torch.einsum("nvji,nvj->nvi", dR.float(), dt.float())
        return -torch.einsum("nvji,nvj->nvi", Rc, origin)

    def augmented(self, rig_ids: torch.Tensor, dR: torch.Tensor, check_source: bool = True) -> Tuple["RigBank", torch.Tensor]:
        """Per-sample virtual rigs for the geometry-consistent rig augmentation.

        Every camera of sample n is rotated about its optical centre by dR [n, v] in its own frame, with the convention
        of ``delta``: p_cam_new = dR p_cam_old, i.e. R' = dR R and t' = dR t (target -> camera), same intrinsics. The
        camera centres do not move, so ``rays`` / ``centers`` keep their origins and the target frame is unchanged.
        Returns a new bank (the original rigs followed by the B virtual ones) and the virtual rig ids [n]. With
        ``check_source`` a point counts as inside a view only if the *original* camera also saw it - the warped image
        has no content elsewhere. A caller passing ``origin_cam`` must rotate it as well (origin' = dR origin)."""
        n = rig_ids.shape[0]
        out = RigBank.__new__(RigBank)
        nn.Module.__init__(out)
        out.num_rigs, out.num_views = self.num_rigs + n, self.num_views
        out.names = list(self.names) + [f"aug{i}" for i in range(n)]
        out.has_scaramuzza, out.has_opencv = self.has_scaramuzza, self.has_opencv
        out.circle_fov = self.circle_fov
        dR = dR.float()
        # the composed extrinsics are called from inside the bf16-mixed training step: autocast would round R' / t'
        with torch.autocast(device_type=dR.device.type, enabled=False):
            for name, buf in self._buffers.items():
                if buf is None or name == "aug_dR":
                    continue
                sel = buf.index_select(0, rig_ids)
                if name == "R":
                    sel = dR @ sel.float()
                elif name == "t":
                    sel = torch.einsum("nvij,nvj->nvi", dR, sel.float())
                out.register_buffer(name, torch.cat([buf, sel.to(buf.dtype)], 0), persistent=False)
        if check_source:
            eye = torch.eye(3, dtype=dR.dtype, device=dR.device).expand(self.num_rigs, self.num_views, 3, 3)
            out.register_buffer("aug_dR", torch.cat([eye, dR], 0), persistent=False)
        return out, self.num_rigs + torch.arange(n, device=rig_ids.device)

    def extra_repr(self) -> str:
        return f"rigs={self.names}, views={self.num_views}"


# --------------------------------------------------------------------------------------
# constructors / registry
# --------------------------------------------------------------------------------------
def camera_from_calib_dict(calib: Dict, name: str, R=None, t=None) -> FisheyeCamera:
    """Build a camera from a calibration dict: Scaramuzza (``polynomialW2C``) or OpenCV fisheye (``K`` + ``D``)."""
    model = calib.get("model")
    if "polynomialW2C" in calib and model in (None, "scaramuzza"):
        return ScaramuzzaCamera(
            name=name,
            size_wh=calib["size"],
            image_center=calib["image_center"],
            poly_w2c=calib["polynomialW2C"],
            poly_c2w=calib["polynomialC2W"],
            affine=calib.get("affine", (1.0, 0.0, 0.0)),
            R=R,
            t=t,
            image_circle_radius=calib.get("imageCircleRadius"),
        )
    if "K" in calib and "D" in calib:
        return OpenCVFisheyeCamera(
            name=name, size_wh=calib["size"], K=calib["K"], D=calib["D"], R=R, t=t,
            image_circle_radius=calib.get("imageCircleRadius"),
        )
    raise ValueError(f"unrecognised calibration for camera {name!r}: keys {sorted(calib)}")


def unrealego_rig() -> StereoRig:
    cams = [
        camera_from_calib_dict(UNREALEGO_CALIB, "left", **UNREALEGO_EXTRINSICS["left"]),
        camera_from_calib_dict(UNREALEGO_CALIB, "right", **UNREALEGO_EXTRINSICS["right"]),
    ]
    return StereoRig(cams, name="unrealego")


def rig_from_json(path: str) -> StereoRig:
    """Load a rig description written by the preprocessing scripts.

    {"name": ..., "cameras": [{"name": "left", "calib": {...scaramuzza or opencv...}, "R": [[..]], "t": [..]}, ...]}
    """
    with open(path, "r") as f:
        spec = json.load(f)
    cams = [camera_from_calib_dict(c["calib"], c["name"], R=c.get("R"), t=c.get("t")) for c in spec["cameras"]]
    return StereoRig(cams, name=spec.get("name", os.path.splitext(os.path.basename(path))[0]))


def rig_to_json(rig: StereoRig, path: str) -> None:
    spec = {"name": rig.name, "num_views": rig.num_views, "cameras": []}
    for cam in rig.cameras:
        spec["cameras"].append({"name": cam.name, "calib": cam.calib_dict(), "R": cam.R.tolist(), "t": cam.t.tolist()})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(spec, f, indent=2)


_RIG_BUILDERS = {"unrealego": unrealego_rig, "unrealego2": unrealego_rig}


def register_rig(name: str, builder) -> None:
    """Register a named rig (``builder`` is a zero-argument callable returning a StereoRig, or a json path)."""
    _RIG_BUILDERS[name] = builder if callable(builder) else (lambda p=builder: rig_from_json(p))


def get_rig(name_or_path: str) -> StereoRig:
    """'unrealego' / 'unrealego2' / a registered name, or a path to a rig json."""
    if name_or_path in _RIG_BUILDERS:
        return _RIG_BUILDERS[name_or_path]()
    if os.path.isfile(name_or_path):
        return rig_from_json(name_or_path)
    raise ValueError(f"unknown rig {name_or_path!r}")


def fit_rigid_transform(src: torch.Tensor, dst: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Least-squares R, t with dst ~= src @ R.T + t (Kabsch). src, dst: [N, 3]."""
    src = src.double()
    dst = dst.double()
    mu_s, mu_d = src.mean(0), dst.mean(0)
    H = (src - mu_s).T @ (dst - mu_d)
    U, _, Vh = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vh.T @ U.T))
    D = torch.diag(torch.tensor([1.0, 1.0, d], dtype=torch.float64))
    R = Vh.T @ D @ U.T
    t = mu_d - mu_s @ R.T
    return R.float(), t.float()


def gravity_alignment(gravity_dev: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Rotation ``R_target`` (target -> device) of the gravity-aligned target frame.

    ``gravity_dev`` [..., 3] is the gravity (down) direction expressed in the device frame (from the headset pose).
    The target frame keeps the device's forward axis as far as possible while its +y axis points along gravity, so
    that poses are expressed "upright" regardless of head pitch / roll (FRAME-style floor alignment without the
    floor height). Returns [..., 3, 3] with ``p_dev = R_target @ p_target``.
    """
    g = F.normalize(gravity_dev.float(), dim=-1, eps=eps)                 # target +y in device coordinates
    fwd = torch.zeros_like(g)
    fwd[..., 2] = 1.0
    z = fwd - (fwd * g).sum(-1, keepdim=True) * g
    bad = z.norm(dim=-1, keepdim=True) < 1e-3                            # looking straight down / up
    alt = torch.zeros_like(g)
    alt[..., 1] = -1.0
    z = torch.where(bad, alt - (alt * g).sum(-1, keepdim=True) * g, z)
    z = F.normalize(z, dim=-1, eps=eps)
    x = F.normalize(torch.cross(g, z, dim=-1), dim=-1, eps=eps)
    return torch.stack((x, g, z), dim=-1)                                # columns = target axes in device coords


__all__ = [
    "UNREALEGO_CALIB", "UNREALEGO_EXTRINSICS", "polyval", "FisheyeCamera", "ScaramuzzaCamera", "OpenCVFisheyeCamera",
    "StereoRig", "RigBank", "camera_from_calib_dict", "unrealego_rig", "rig_from_json", "rig_to_json", "register_rig",
    "get_rig", "fit_rigid_transform", "gravity_alignment", "scaramuzza_project", "scaramuzza_unproject",
    "opencv_project", "opencv_unproject",
]

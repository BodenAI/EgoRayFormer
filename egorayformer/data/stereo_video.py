"""Clip dataset over unified sequence records (T=1 is the single-frame special case)."""

from __future__ import annotations

import hashlib
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .camera import get_rig, gravity_alignment
from .records import IMAGENET_MEAN, IMAGENET_STD, DatasetIndex, SequenceRecord, derive_root_dev
from .skeleton import NUM_JOINTS, get_skeleton, skeleton_id

# stable ids for the dataset embedding in the body token
DATASET_IDS: Dict[str, int] = {
    "unrealego": 0, "unrealego2": 1, "unrealego_rw": 2, "synthetic": 3,
    "frame": 4, "ego4view_syn": 5, "ego4view_rw": 6, "egobody3m": 7,
}
AUX_COND_DIM = 10  # gravity_dev (3) + device height / 100 (1) + 6D rotation of the gravity frame (6)

# Target coordinate frames (protocols):
#   pelvis  - pelvis-relative (UnrealEgo v1 implementation)
#   device  - device-relative, includes the global translation (UnrealEgo2 / UnrealEgo-RW official leaderboard)
#   root    - root-relative as used by TSR-Ego: pelvis where annotated (UnrealEgo2), mid-hip otherwise (UnrealEgo-RW)
#   gravity - device-relative translation, but axes rotated so that +y points along gravity (needs the headset pose /
#             gravity direction of the record; falls back to ``device`` frame by frame when it is unavailable)
TARGET_FRAMES = ("pelvis", "device", "root", "gravity")
SUPERVISION_TYPES = ("full", "2d_only", "none")


def resolve_rig_swap(rig: str, swap: Optional[Dict[str, str]]) -> str:
    """Rig used in place of ``rig`` under ``swap`` {rig name or file name: replacement} (a replacement without a directory
    lives next to the original rig file); unchanged when the rig is not listed."""
    if not swap:
        return rig
    import os

    key = os.path.basename(rig)
    new = swap.get(rig, swap.get(key))
    if new is None:
        return rig
    return new if (os.sep in new or not os.path.dirname(rig)) else os.path.join(os.path.dirname(rig), new)


class PhotometricAugment:
    """Clip-consistent photometric jitter on float images in [0, 1] (same params for all views/frames)."""

    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3, hue_shift=0.03, noise_std=0.02, gray_p=0.05):
        self.brightness, self.contrast, self.saturation = brightness, contrast, saturation
        self.hue_shift, self.noise_std, self.gray_p = hue_shift, noise_std, gray_p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:  # img [T, V, 3, H, W] in [0,1]
        b = 1.0 + random.uniform(-self.brightness, self.brightness)
        c = 1.0 + random.uniform(-self.contrast, self.contrast)
        s = 1.0 + random.uniform(-self.saturation, self.saturation)
        img = img * b
        mean = img.mean(dim=(-1, -2), keepdim=True)
        img = (img - mean) * c + mean
        gray = img.mean(dim=2, keepdim=True)
        img = (img - gray) * s + gray
        if self.hue_shift > 0:
            shift = torch.empty(3).uniform_(-self.hue_shift, self.hue_shift).view(1, 1, 3, 1, 1)
            img = img + shift
        if random.random() < self.gray_p:
            img = img.mean(dim=2, keepdim=True).expand_as(img)
        if self.noise_std > 0:
            img = img + torch.randn_like(img) * random.uniform(0.0, self.noise_std)
        return img.clamp(0.0, 1.0)


class StereoVideoDataset(Dataset):
    """Yields clips of T stereo frames with 3D / 2D annotations.

    Each item (all tensors, cm / normalised coordinates; J = ``num_joints_pad`` >= joints of the source skeleton,
    NaN-padded):
        img          float32 [T, V, 3, H, W]   ImageNet-normalised
        pose         float32 [T, J, 3]         target-frame joints (NaN if unavailable / padded)
        origin_cam   float32 [T, V, 1, 3]      target-frame origin in each camera frame
        target_rot   float32 [T, 3, 3]         rotation target -> device (identity except for the gravity frame)
        aux_cond     float32 [T, AUX_COND_DIM] headset / gravity conditioning vector (NaN when unavailable)
        pts2d        float32 [T, V, J, 2]      normalised [0,1] 2D joints (NaN if unavailable)
        pts2d_valid  bool    [T, V, J]         in image, in front of the camera (per-view joint visibility)
        root_dev     float32 [T, 3]            root (pelvis or mid-hip) in the device frame (NaN if unavailable)
        valid_frames bool    [T]               False for padded frames at sequence ends
        dataset_id   int64   [], skeleton_id int64 []
        frame_ids    int64   [T]
        seq_id       str, rig str, skeleton str
    """

    def __init__(
        self,
        root: str,
        sources: Sequence[Dict],
        clip_len: int = 1,
        frame_stride: int = 1,
        mode: str = "train",
        target_frame: str = "pelvis",
        img_size: int = 256,
        augment: bool = True,
        clips_per_epoch_scale: float = 1.0,
        num_joints_pad: Optional[int] = None,
        unc_holdout_frac: float = 0.0,
        rig_swap: Optional[Dict[str, str]] = None,
    ):
        assert mode in ("train", "eval")
        assert target_frame in TARGET_FRAMES, f"target_frame must be one of {TARGET_FRAMES}"
        self.root = root
        self.clip_len = int(clip_len)
        self.frame_stride = int(frame_stride)
        self.mode = mode
        self.target_frame = target_frame
        self.img_size = int(img_size)
        self.augment = PhotometricAugment() if (augment and mode == "train") else None
        self.clips_per_epoch_scale = clips_per_epoch_scale
        # Held-out *sequences* for the uncertainty head: a stable hash of the sequence id selects
        # the fraction; their clips are flagged ``unc_holdout`` and the LightningModule applies only the (pose-detached)
        # NLL to them, so the uncertainty is fitted to realistic errors instead of the over-fitted training residuals.
        self.unc_holdout_frac = float(unc_holdout_frac) if mode == "train" else 0.0

        self.indices: List[DatasetIndex] = [DatasetIndex(root, s["dataset"], s["split"]) for s in sources]
        # evaluation with a neighbouring session's calibration - the situation
        # of the test-only UnrealEgo-RW session 2023_01_17 - reproduced on validation sequences ({rig: replacement})
        self.rig_swap = dict(rig_swap or {})
        if self.rig_swap:
            for idx in self.indices:
                for seq in idx.sequences:
                    seq.rig = resolve_rig_swap(seq.rig, self.rig_swap)
        self.source_weights: List[float] = [float(s.get("weight", 1.0)) for s in sources]
        # Supervision of a source: "full" (3D + 2D), "2d_only" (the 3D annotation is hidden, the 2D
        # joints stay - heatmap / reprojection supervision only) or "none" (UDA target: every annotation is
        # hidden so that only the self-supervised losses see it). ``unlabeled: true`` keeps meaning "none".
        self.source_supervision: List[str] = [str(s.get("supervision", "none" if s.get("unlabeled", False) else "full")) for s in sources]
        for sup in self.source_supervision:
            if sup not in SUPERVISION_TYPES:
                raise ValueError(f"supervision must be one of {SUPERVISION_TYPES}, got {sup!r}")
        self.source_unlabeled: List[bool] = [sup == "none" for sup in self.source_supervision]
        if any(sup != "full" for sup in self.source_supervision) and target_frame not in ("device", "gravity"):
            raise ValueError("sources without 3D supervision require target_frame 'device' (or 'gravity'): the root / pelvis "
                             "origins depend on the GT")
        self.rigs = {r: get_rig(r) for idx in self.indices for r in ([idx.rig] + sorted({s.rig for s in idx.sequences}))}
        self.rig_translations = {name: rig.translations.numpy() for name, rig in self.rigs.items()}  # [V, 3]
        self.rig_R = {name: torch.stack([c.R for c in rig.cameras]).numpy() for name, rig in self.rigs.items()}  # [V, 3, 3]
        self.skeletons = {idx.skeleton: get_skeleton(idx.skeleton) for idx in self.indices}
        self.num_joints_pad = int(num_joints_pad or max(s.num_joints for s in self.skeletons.values()))
        assert all(s.num_joints <= self.num_joints_pad for s in self.skeletons.values()), "num_joints_pad too small"

        # entries: (source_index, sequence_index, start_frame)
        self.entries: List[Tuple[int, int, int]] = []
        self.entry_weights: List[float] = []
        span = (self.clip_len - 1) * self.frame_stride + 1
        for si, idx in enumerate(self.indices):
            for qi, seq in enumerate(idx.sequences):
                n = seq.num_frames
                if mode == "train":
                    n_clips = max(1, int(math.ceil(n / span * clips_per_epoch_scale)))
                    for _ in range(n_clips):
                        self.entries.append((si, qi, -1))  # -1: random start at __getitem__
                        self.entry_weights.append(self.source_weights[si])
                else:
                    for start in range(0, n, span):
                        self.entries.append((si, qi, start))
                        self.entry_weights.append(self.source_weights[si])
        self.span = span

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def num_frames(self) -> int:
        return sum(idx.num_frames for idx in self.indices)

    def is_unc_holdout(self, seq_id: str) -> bool:
        """Deterministic per-sequence membership of the uncertainty hold-out set (independent of the process seed)."""
        if self.unc_holdout_frac <= 0.0:
            return False
        h = int(hashlib.md5(seq_id.encode("utf-8")).hexdigest()[:8], 16) / float(0xFFFFFFFF)
        return h < self.unc_holdout_frac

    # ---------------------------------------------------------------------------------
    def _frame_indices(self, seq: SequenceRecord, start: int) -> Tuple[np.ndarray, np.ndarray]:
        n = seq.num_frames
        if start < 0:  # random start covering [0, n - span]
            hi = max(0, n - self.span)
            start = random.randint(0, hi)
        idx = start + np.arange(self.clip_len) * self.frame_stride
        valid = idx < n
        idx = np.minimum(idx, n - 1)
        return idx, valid

    def _load_images(self, seq: SequenceRecord, idx: np.ndarray) -> torch.Tensor:
        imgs = seq.images(idx)  # uint8 [T, V, H, W, 3]
        T, V, H, W, _ = imgs.shape
        if H != self.img_size or W != self.img_size:
            flat = imgs.reshape(T * V, H, W, 3)
            interp = cv2.INTER_AREA if self.img_size < H else cv2.INTER_LINEAR
            flat = np.stack([cv2.resize(im, (self.img_size, self.img_size), interpolation=interp) for im in flat])
            imgs = flat.reshape(T, V, self.img_size, self.img_size, 3)
        x = torch.from_numpy(np.ascontiguousarray(imgs)).permute(0, 1, 4, 2, 3).float() / 255.0  # [T,V,3,H,W]
        if self.augment is not None:
            x = self.augment(x)
        mean = torch.from_numpy(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
        std = torch.from_numpy(IMAGENET_STD).view(1, 1, 3, 1, 1)
        return (x - mean) / std

    def _pad_joints(self, arr: np.ndarray, axis: int) -> np.ndarray:
        """NaN-pad the joint axis of ``arr`` to ``num_joints_pad``."""
        J = arr.shape[axis]
        if J == self.num_joints_pad:
            return arr
        pad = [(0, 0)] * arr.ndim
        pad[axis] = (0, self.num_joints_pad - J)
        return np.pad(arr.astype(np.float32), pad, constant_values=np.nan)

    @staticmethod
    def _aux_cond(gravity_dev: np.ndarray, height: np.ndarray, rot: np.ndarray) -> np.ndarray:
        """[T, 3] gravity, [T] height (cm), [T, 3, 3] rotation -> [T, AUX_COND_DIM] (NaN rows when no headset pose)."""
        T = gravity_dev.shape[0]
        out = np.full((T, AUX_COND_DIM), np.nan, dtype=np.float32)
        ok = np.isfinite(gravity_dev).all(-1)
        out[ok, :3] = gravity_dev[ok]
        h = np.where(np.isfinite(height), height, 0.0) / 100.0
        out[ok, 3] = h[ok]
        out[ok, 4:] = rot[ok][:, :, :2].reshape(-1, 6)  # 6D rotation representation (first two columns)
        return out

    def __getitem__(self, i: int) -> Dict:
        si, qi, start = self.entries[i]
        index = self.indices[si]
        seq = index.sequences[qi]
        idx, valid = self._frame_indices(seq, start)
        meta = seq.meta
        img = self._load_images(seq, idx)
        skel = self.skeletons[index.skeleton]

        T, V = len(idx), img.shape[1]
        pose_dev = meta["pose_dev"][idx]                          # [T, J, 3]
        root_dev = (meta["root_dev"][idx] if "root_dev" in meta else derive_root_dev(pose_dev, meta["pose_pelvis"][idx], skel.root_joints))
        rot = np.broadcast_to(np.eye(3, dtype=np.float32), (T, 3, 3)).copy()  # target -> device
        gravity = meta["gravity_dev"][idx].astype(np.float32) if "gravity_dev" in meta else np.full((T, 3), np.nan, np.float32)
        height = meta["device_height"][idx].astype(np.float32) if "device_height" in meta else np.full((T,), np.nan, np.float32)
        if self.target_frame == "pelvis":
            pose = meta["pose_pelvis"][idx]                       # [T, J, 3]
            origin = meta["pelvis_cam"][idx][:, :, None, :]       # [T, V, 1, 3]
        elif self.target_frame == "root":
            pose = pose_dev - root_dev[:, None, :]
            R, t = self.rig_R[seq.rig], self.rig_translations[seq.rig]              # [V, 3, 3], [V, 3]
            origin = (np.einsum("tk,vjk->tvj", root_dev, R) + t[None])[:, :, None, :]  # root in each camera: R r + t
        else:
            pose = pose_dev
            t = self.rig_translations[seq.rig]                    # [V, 3]
            origin = np.broadcast_to(t[None, :, None, :], (T, V, 1, 3)).copy()
            if self.target_frame == "gravity":
                ok = np.isfinite(gravity).all(-1)
                if ok.any():
                    Rg = gravity_alignment(torch.from_numpy(gravity[ok])).numpy()      # [n, 3, 3] target -> device
                    rot[ok] = Rg
                    pose = pose.copy()
                    pose[ok] = np.einsum("tji,tkj->tki", Rg, pose_dev[ok])            # R^T p_dev
        if self.target_frame == "gravity":
            aux = self._aux_cond(gravity, height, rot)
        else:  # conditioning with the true gravity frame even when the target frame is not rotated
            rot_g = np.broadcast_to(np.eye(3, dtype=np.float32), (T, 3, 3)).copy()
            okg = np.isfinite(gravity).all(-1)
            if okg.any():
                rot_g[okg] = gravity_alignment(torch.from_numpy(gravity[okg])).numpy()
            aux = self._aux_cond(gravity, height, rot_g)

        pts2d = meta["pts2d"][idx].astype(np.float32)             # [T, V, J, 2] original pixels
        ow, oh = index.orig_size
        pts2d_norm = pts2d / np.array([ow, oh], dtype=np.float32)
        pts3d_cam = meta["pts3d_cam"][idx]
        front = np.isfinite(pts3d_cam[..., 2]) & (pts3d_cam[..., 2] > 0) if np.isfinite(pts3d_cam).any() else np.ones(pts2d.shape[:-1], bool)
        pts2d_valid = np.isfinite(pts2d_norm).all(-1) & (pts2d_norm >= 0).all(-1) & (pts2d_norm <= 1).all(-1) & front
        if "visibility" in meta:  # dataset-provided labels (e.g. self-occlusion aware) refine the FOV-based flag
            vis_lbl = meta["visibility"][idx]
            pts2d_valid = pts2d_valid & ~(np.isfinite(vis_lbl) & (vis_lbl < 0.5))

        pose = self._pad_joints(np.asarray(pose, np.float32), axis=1)
        pts2d_norm = self._pad_joints(pts2d_norm, axis=2)
        pts2d_valid = np.pad(pts2d_valid, [(0, 0), (0, 0), (0, self.num_joints_pad - pts2d_valid.shape[2])], constant_values=False)
        unlabeled = self.source_unlabeled[si]
        supervision = self.source_supervision[si]
        if supervision != "full":  # no 3D annotation for "2d_only" and "none" sources
            pose = np.full_like(pose, np.nan)
        if unlabeled:  # strip every annotation of an unlabeled (UDA target) source
            pts2d_norm = np.full_like(pts2d_norm, np.nan)
            pts2d_valid = np.zeros_like(pts2d_valid)

        return {
            "img": img,
            "pose": torch.from_numpy(np.ascontiguousarray(pose, dtype=np.float32)),
            "origin_cam": torch.from_numpy(np.ascontiguousarray(origin, dtype=np.float32)),
            "target_rot": torch.from_numpy(np.ascontiguousarray(rot, dtype=np.float32)),
            "aux_cond": torch.from_numpy(np.ascontiguousarray(aux, dtype=np.float32)),
            "pts2d": torch.from_numpy(np.ascontiguousarray(pts2d_norm)),
            "pts2d_valid": torch.from_numpy(np.ascontiguousarray(pts2d_valid)),
            "root_dev": torch.from_numpy(np.ascontiguousarray(root_dev, dtype=np.float32)),
            "valid_frames": torch.from_numpy(valid),
            "dataset_id": torch.tensor(DATASET_IDS.get(index.dataset, len(DATASET_IDS)), dtype=torch.long),
            "unc_holdout": torch.tensor(self.is_unc_holdout(seq.seq_id)),
            "unlabeled": torch.tensor(unlabeled),
            "supervision": supervision,
            "skeleton_id": torch.tensor(skeleton_id(index.skeleton), dtype=torch.long),
            "frame_ids": torch.from_numpy(meta["frame_ids"][idx].astype(np.int64)),
            "orig_size": torch.tensor([float(ow), float(oh)]),   # pixels of the calibration / annotation (pts2d scale)
            "seq_id": seq.seq_id,
            "rig": seq.rig,
            "skeleton": index.skeleton,
            "aux_provenance": index.aux_provenance,              # origin of headset_T / gravity / height (GT-free audit)
        }


def collate_clips(batch: List[Dict]) -> Dict:
    out: Dict = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        out[k] = torch.stack(vals, 0) if torch.is_tensor(vals[0]) else vals
    return out

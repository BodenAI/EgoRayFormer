"""Unified on-disk format shared by all datasets (UnrealEgo / UnrealEgo2 / UnrealEgo-RW / FRAME / Ego4View / EgoBody3M).

    <root>/<dataset>/<split>/index.json
    <root>/<dataset>/<split>/<seq_dir>/left.npy    uint8 [N, H, W, 3]   (memory-mapped; V != 2 -> view0.npy, view1.npy, ...)
                                      /right.npy   uint8 [N, H, W, 3]
                                      /meta.npz    per-frame annotations (see META_KEYS)

index.json: dataset, split, rig (name or rig.json path), image_size, orig_size, num_views (default 2),
            skeleton (default "unrealego16"), num_frames, sequences [{id, dir, num_frames}].

meta.npz keys (float32 unless noted, NaN where unavailable, J = joints of the dataset skeleton):
    frame_ids     int32   [N]
    pose_dev      [N, J, 3]      device-relative joints (cm)
    pose_pelvis   [N, J, 3]      pelvis-relative joints (cm)
    pelvis_cam    [N, V, 3]      pelvis position in each camera frame (cm)
    pts3d_cam     [N, V, J, 3]   joints in each camera frame (cm)
    pts2d         [N, V, J, 2]   joints in original pixel coordinates
    root_dev      [N, 3]         root of the root-relative protocol: the pelvis when the dataset provides one
                                 (UnrealEgo / UnrealEgo2), otherwise the mid-hip (UnrealEgo-RW, TSR-Ego convention).
                                 Older records derive it on the fly via ``derive_root_dev``.
    headset_T     [N, 4, 4]      (optional) world_from_device rigid transform (cm) from the headset tracking
    gravity_dev   [N, 3]         (optional) unit gravity (down) direction in the device frame
    device_height [N]            (optional) height of the device origin above the floor (cm)
    visibility    [N, V, J]      (optional, float 0/1) dataset-provided per-view visibility labels
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

META_KEYS = ("frame_ids", "pose_dev", "pose_pelvis", "pelvis_cam", "pts3d_cam", "pts2d", "root_dev",
             "headset_T", "gravity_dev", "device_height", "visibility")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MIDHIP_JOINTS = (8, 9)  # thigh_l / thigh_r (UnrealEgo-RW: LeftUpLeg / RightUpLeg)

# Where the optional headset fields (headset_T / gravity_dev / device_height) of a dataset come from:
# "gt_derived" = fitted from / rendered with the ground truth (not available to a deployed model), "sensor" = the
# on-device tracking of the headset, "external_tracking" = an external mocap system tracking the headset. index.json
# may override it with an ``aux_provenance`` field (written by the preprocessing scripts).
DEFAULT_AUX_PROVENANCE: Dict[str, str] = {
    "unrealego": "gt_derived", "unrealego2": "gt_derived", "unrealego_rw": "gt_derived", "synthetic": "gt_derived",
    "frame": "sensor", "ego4view_syn": "gt_derived", "ego4view_rw": "gt_derived", "egobody3m": "external_tracking",
}


def seq_id_to_dirname(seq_id: str) -> str:
    return seq_id.strip("/").replace("/", "__")


def view_names_for(num_views: int) -> List[str]:
    return ["left", "right"] if num_views == 2 else [f"view{v}" for v in range(num_views)]


def derive_root_dev(pose_dev: np.ndarray, pose_pelvis: Optional[np.ndarray] = None, root_joints: Sequence[int] = MIDHIP_JOINTS) -> np.ndarray:
    """Root position in the device frame for the root-relative protocol -> [..., 3].

    Pelvis if it can be recovered (``pose_dev - pose_pelvis`` is the same for every joint), otherwise the mid-hip
    (midpoint of both hip joints), which is exactly the TSR-Ego root for UnrealEgo-RW where no pelvis is annotated.
    """
    pose_dev = np.asarray(pose_dev, np.float32)
    midhip = pose_dev[..., list(root_joints), :].mean(axis=-2)
    if pose_pelvis is None:
        return midhip
    pelvis = pose_dev[..., 0, :] - np.asarray(pose_pelvis, np.float32)[..., 0, :]
    return np.where(np.isfinite(pelvis).all(axis=-1, keepdims=True), pelvis, midhip).astype(np.float32)


class SequenceWriter:
    """Accumulates frames of one sequence and writes the view .npy files and meta.npz."""

    def __init__(self, out_dir: str, num_views: int = 2, num_joints: int = 16, root_joints: Sequence[int] = MIDHIP_JOINTS):
        self.out_dir = out_dir
        self.num_views, self.num_joints, self.root_joints = num_views, num_joints, tuple(root_joints)
        self.images: List[List[np.ndarray]] = [[] for _ in range(num_views)]
        self.meta: Dict[str, List[np.ndarray]] = {k: [] for k in META_KEYS}

    def add_frame(
        self,
        frame_id: int,
        images: Sequence[np.ndarray],
        pose_dev: Optional[np.ndarray] = None,
        pose_pelvis: Optional[np.ndarray] = None,
        pelvis_cam: Optional[np.ndarray] = None,
        pts3d_cam: Optional[np.ndarray] = None,
        pts2d: Optional[np.ndarray] = None,
        headset_T: Optional[np.ndarray] = None,
        gravity_dev: Optional[np.ndarray] = None,
        device_height: Optional[float] = None,
        visibility: Optional[np.ndarray] = None,
    ) -> None:
        assert len(images) == self.num_views
        for v, im in enumerate(images):
            assert im.dtype == np.uint8 and im.ndim == 3 and im.shape[2] == 3, im.shape
            self.images[v].append(im)
        V, J = self.num_views, self.num_joints
        nan = lambda *s: np.full(s, np.nan, dtype=np.float32)  # noqa: E731
        put = lambda key, val, *shape: self.meta[key].append(np.asarray(val, np.float32) if val is not None else nan(*shape))  # noqa: E731
        self.meta["frame_ids"].append(np.int32(frame_id))
        put("pose_dev", pose_dev, J, 3)
        put("pose_pelvis", pose_pelvis, J, 3)
        put("pelvis_cam", pelvis_cam, V, 3)
        put("pts3d_cam", pts3d_cam, V, J, 3)
        put("pts2d", pts2d, V, J, 2)
        self.meta["root_dev"].append(derive_root_dev(self.meta["pose_dev"][-1], pose_pelvis, self.root_joints) if pose_dev is not None else nan(3))
        put("headset_T", headset_T, 4, 4)
        put("gravity_dev", gravity_dev, 3)
        put("device_height", device_height)
        put("visibility", visibility, V, J)

    def __len__(self) -> int:
        return len(self.meta["frame_ids"])

    def finalize(self) -> int:
        n = len(self)
        if n == 0:
            return 0
        os.makedirs(self.out_dir, exist_ok=True)
        for v, name in enumerate(view_names_for(self.num_views)):
            np.save(os.path.join(self.out_dir, f"{name}.npy"), np.stack(self.images[v], axis=0))
        meta = {k: np.stack(v, axis=0) for k, v in self.meta.items() if len(v) == n}
        # optional keys that are NaN everywhere are not written (keeps old readers / small files)
        for k in ("headset_T", "gravity_dev", "device_height", "visibility"):
            if k in meta and not np.isfinite(meta[k]).any():
                del meta[k]
        np.savez(os.path.join(self.out_dir, "meta.npz"), **meta)
        return n


@dataclass
class SequenceRecord:
    """Lazy reader for one sequence directory."""

    path: str
    seq_id: str
    num_frames: int
    rig: str
    dataset: str
    num_views: int = 2
    _images: Optional[List[np.memmap]] = field(default=None, repr=False)
    _meta: Optional[Dict[str, np.ndarray]] = field(default=None, repr=False)

    @property
    def view_names(self) -> List[str]:
        return view_names_for(self.num_views)

    def _load(self) -> None:
        if self._images is None:
            self._images = [np.load(os.path.join(self.path, f"{n}.npy"), mmap_mode="r") for n in self.view_names]
            with np.load(os.path.join(self.path, "meta.npz")) as z:
                self._meta = {k: z[k] for k in z.files}

    @property
    def meta(self) -> Dict[str, np.ndarray]:
        self._load()
        return self._meta  # type: ignore[return-value]

    def images(self, idx: Sequence[int]) -> np.ndarray:
        """-> uint8 [T, V, H, W, 3]"""
        self._load()
        idx = np.asarray(idx)
        return np.stack([np.asarray(im[idx]) for im in self._images], axis=1)  # type: ignore[index]

    def image_size(self) -> tuple:
        self._load()
        return self._images[0].shape[2], self._images[0].shape[1]  # (W, H)  # type: ignore[index]

    def __len__(self) -> int:
        return self.num_frames


class DatasetIndex:
    """index.json of one <root>/<dataset>/<split>."""

    def __init__(self, root: str, dataset: str, split: str):
        self.root, self.dataset, self.split = root, dataset, split
        self.dir = os.path.join(root, dataset, split)
        with open(os.path.join(self.dir, "index.json"), "r") as f:
            self.info = json.load(f)
        self.rig: str = self.info["rig"]
        self.image_size = tuple(self.info["image_size"])  # (W, H) of stored images
        self.orig_size = tuple(self.info["orig_size"])    # (W, H) of the calibration / pts2d
        self.num_views: int = int(self.info.get("num_views", 2))
        self.skeleton: str = self.info.get("skeleton", "unrealego16")
        self.aux_provenance: str = str(self.info.get("aux_provenance", DEFAULT_AUX_PROVENANCE.get(dataset, "unknown")))
        self.sequences: List[SequenceRecord] = [
            SequenceRecord(
                path=os.path.join(self.dir, s["dir"]),
                seq_id=s["id"],
                num_frames=int(s["num_frames"]),
                rig=s.get("rig", self.rig),  # per-sequence (capture-session) rig, see scripts/fit_session_rigs.py
                dataset=dataset,
                num_views=self.num_views,
            )
            for s in self.info["sequences"]
        ]

    @property
    def num_frames(self) -> int:
        return sum(s.num_frames for s in self.sequences)

    @property
    def all_rigs(self) -> List[str]:
        """Every rig used by this split (the default one first)."""
        extra = sorted({s.rig for s in self.sequences} - {self.rig})
        return [self.rig] + extra

    @staticmethod
    def write(
        dir_path: str,
        dataset: str,
        split: str,
        rig: str,
        image_size: Sequence[int],
        orig_size: Sequence[int],
        sequences: List[Dict],
        extra: Optional[Dict] = None,
        num_views: int = 2,
        skeleton: str = "unrealego16",
    ) -> None:
        os.makedirs(dir_path, exist_ok=True)
        info = {
            "dataset": dataset,
            "split": split,
            "rig": rig,
            "image_size": list(image_size),
            "orig_size": list(orig_size),
            "num_views": int(num_views),
            "skeleton": skeleton,
            "num_frames": int(sum(s["num_frames"] for s in sequences)),
            "sequences": sequences,
        }
        if extra:
            info.update(extra)
        with open(os.path.join(dir_path, "index.json"), "w") as f:
            json.dump(info, f, indent=1)

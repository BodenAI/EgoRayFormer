#!/usr/bin/env python
"""UnrealEgo2 / UnrealEgo-RW raw data (png + json per frame) -> unified sequence records.

Expected raw layout (UnrealEgo2 website):
    <rgb_root>/<seq path>/fisheye_final_image/<left|right ...>/final_<i>.png
    <pose_root>/<seq path>/json/frame_<i>.json
json keys: camera_left_pts3d, camera_right_pts3d, camera_left_pts2d, camera_right_pts2d, device_pts3d
(values: {joint_name: [x, y, z]} dicts or 16-element lists). The exact image sub-folder names are
discovered automatically (anything containing "left" / "right").

    # UnrealEgo2 (same camera as UnrealEgo)
    python scripts/preprocess_unrealego_raw.py --dataset unrealego2 --rgb_root .../UnrealEgoData2_train_val_rgb \
        --pose_root .../UnrealEgoData2_train_val_pose --split_txt .../train.txt --split_name train \
        --out_root /data/unrealego/processed --workers 32
    # UnrealEgo-RW (own calibration; extrinsics fitted from device_pts3d vs camera_*_pts3d)
    python scripts/preprocess_unrealego_raw.py --dataset unrealego_rw ... --calib_left rw_left.json --calib_right rw_right.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from multiprocessing import Pool
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from egorayformer.data.camera import StereoRig, camera_from_calib_dict, fit_rigid_transform, get_rig, rig_to_json  # noqa: E402
from egorayformer.data.records import DatasetIndex, SequenceWriter, seq_id_to_dirname  # noqa: E402
from egorayformer.data.skeleton import UNREALEGO_JSON_NAMES, UNREALEGO_RW_JSON_NAMES, resolve_joint_dict  # noqa: E402

IMG_SUBDIR = "fisheye_final_image"
JSON_SUBDIR = "json"
_PREFIX = re.compile(r"^\.?/?UnrealEgoData[^/]*/")


def parse_split_file(path: str) -> List[str]:
    with open(path, "r") as f:
        return [_PREFIX.sub("", ln.strip()).strip("/") for ln in f if ln.strip()]


def discover_sequences(rgb_root: str) -> List[str]:
    seqs = []
    for dirpath, dirnames, _ in os.walk(rgb_root):
        if IMG_SUBDIR in dirnames:
            seqs.append(os.path.relpath(dirpath, rgb_root))
            dirnames[:] = []
    return sorted(seqs)


def find_view_dirs(img_dir: str) -> Tuple[str, str]:
    subs = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]
    left = [d for d in subs if "left" in d.lower()]
    right = [d for d in subs if "right" in d.lower()]
    if len(left) == 1 and len(right) == 1:
        return os.path.join(img_dir, left[0]), os.path.join(img_dir, right[0])
    raise FileNotFoundError(f"cannot identify left/right sub-folders in {img_dir}: {subs}")


def _indexed_files(directory: str, pattern: re.Pattern) -> Dict[int, str]:
    """Map frame id -> path. Duplicate ids are an error: the last file must not silently win."""
    out: Dict[int, str] = {}
    if not os.path.isdir(directory):
        return out
    for fn in os.listdir(directory):
        m = pattern.search(fn)
        if not m:
            continue
        fid = int(m.group(1))
        if fid in out:
            raise RuntimeError(f"duplicate frame id {fid} in {directory}: {fn} and {os.path.basename(out[fid])}")
        out[fid] = os.path.join(directory, fn)
    return out


def list_frames(view_dir: str) -> Dict[int, str]:
    return _indexed_files(view_dir, re.compile(r"(\d+)\.(png|jpg|jpeg)$"))


def list_json_frames(json_dir: str) -> Dict[int, str]:
    return _indexed_files(json_dir, re.compile(r"frame_(\d+)\.json$"))


def _brief_ids(ids: set) -> str:
    shown = sorted(ids)[:8]
    text = ",".join(str(i) for i in shown) if shown else "none"
    extra = len(ids) - len(shown)
    return f"{len(ids)} [{text}{', ...' if extra else ''}]"


def require_complete_frames(seq_id: str, left: Dict[int, str], right: Dict[int, str], jsons: Dict[int, str]) -> List[int]:
    """Left images, right images and json must carry the same frame ids. A missing view or annotation used to be
    skipped, so frames [1, 3] were stored as a complete sequence and never filled in on resume."""
    left_ids, right_ids, json_ids = set(left), set(right), set(jsons)
    if not left_ids or left_ids != right_ids or left_ids != json_ids:
        raise RuntimeError(
            f"{seq_id}: refusing incomplete frames. "
            f"left {_brief_ids(left_ids)}, right {_brief_ids(right_ids)}, json {_brief_ids(json_ids)}; "
            f"missing right {_brief_ids(left_ids - right_ids)}, missing left {_brief_ids(right_ids - left_ids)}, "
            f"missing json {_brief_ids((left_ids | right_ids) - json_ids)}, "
            f"json without image {_brief_ids(json_ids - left_ids - right_ids)}"
        )
    return sorted(left_ids)


def joints_from_json(val, name_table) -> np.ndarray:
    if isinstance(val, dict):
        return resolve_joint_dict(val, name_table)
    arr = np.asarray(val, dtype=np.float32)
    if arr.shape[0] != 16:
        raise ValueError(f"expected 16 joints, got {arr.shape}")
    return arr


def joints_key(js: Dict, key: str, name_table) -> np.ndarray:
    """Read one per-joint quantity from either json schema -> [16, D].

    * flat (UnrealEgo2 website description):  js[key] = {joint_name: [x, y, z]} or a 16-row list
    * nested (released UnrealEgo-RW / Ego4View): js["joints"][joint_name][key] = [x, y, z]
    """
    if key in js:
        return joints_from_json(js[key], name_table)
    if "joints" in js:
        raw = {n: v[key] for n, v in js["joints"].items() if isinstance(v, dict) and key in v}
        if raw:
            return resolve_joint_dict(raw, name_table)
    raise KeyError(f"{key} not found in json (keys: {list(js)[:8]})")


def has_joints_key(js: Dict, key: str) -> bool:
    if key in js:
        return True
    return "joints" in js and any(isinstance(v, dict) and key in v for v in js["joints"].values())


def detect_up_axis(world: np.ndarray) -> int:
    """World axis pointing up: the one along which the head is farthest above the feet ([16, 3] UnrealEgo joint order:
    head = 0, ball_l / ball_r = 14 / 15). UnrealEgo-RW's mocap world is y-up; UE4 worlds are z-up."""
    d = world[0] - world[[14, 15]].mean(0)
    return int(np.argmax(d))


def headset_from_world_joints(dev: np.ndarray, world: np.ndarray, up_axis: int, max_resid_cm: float = 1.0):
    """Recover the headset pose from the same joints given in the device frame and in the mocap world frame:
    world = R dev + t. Returns (headset_T [4,4] world_from_device, gravity_dev [3], device_height along ``up_axis``) or
    None when the two annotations are not rigidly consistent.

    The result is fitted from the *ground-truth* joints: it is not the output of an on-device tracker and
    must not be fed to a model whose numbers are reported as RGB-only / deployable - index.json records it as
    ``aux_provenance: gt_derived`` and eval/gt_audit.py flags every inference input built from it."""
    R, t = fit_rigid_transform(torch.from_numpy(dev).float(), torch.from_numpy(world).float())
    resid = (torch.from_numpy(dev).float() @ R.T + t - torch.from_numpy(world).float()).norm(dim=-1).mean().item()
    if not np.isfinite(resid) or resid > max_resid_cm:
        return None
    R, t = R.numpy(), t.numpy()
    T = np.eye(4, dtype=np.float32)
    T[:3, :3], T[:3, 3] = R, t
    down = np.zeros(3, np.float32)
    down[up_axis] = -1.0
    gravity_dev = (R.T @ down).astype(np.float32)
    return T, gravity_dev, float(t[up_axis])


def load_image(path: str, size: int) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if im.shape[0] != size or im.shape[1] != size:
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA if size < im.shape[0] else cv2.INTER_LINEAR)
    return im


_CFG: Dict = {}


def _init_worker(cfg: Dict) -> None:
    global _CFG
    torch.set_num_threads(1)
    _CFG = cfg


def process_sequence(seq_id: str) -> Tuple[str, str, int, Optional[float]]:
    cfg = _CFG
    name_table = UNREALEGO_RW_JSON_NAMES if cfg["dataset"] == "unrealego_rw" else UNREALEGO_JSON_NAMES
    img_dir = os.path.join(cfg["rgb_root"], seq_id, IMG_SUBDIR)
    json_dir = os.path.join(cfg["pose_root"], seq_id, JSON_SUBDIR)
    left_dir, right_dir = find_view_dirs(img_dir)
    left, right = list_frames(left_dir), list_frames(right_dir)
    jsons = list_json_frames(json_dir)
    fids = require_complete_frames(seq_id, left, right, jsons)
    # out_splits: one archive holds several splits (UnrealEgo2 train and validation). Absent -> the single out_split.
    out_dir = os.path.join(cfg.get("out_splits", {}).get(seq_id, cfg["out_split"]), seq_id_to_dirname(seq_id))
    writer = SequenceWriter(out_dir)
    rig: StereoRig = get_rig(cfg["rig"]) if cfg["rig"] else None
    reproj_err = []
    up_axis = None
    for fid in fids:
        with open(jsons[fid], "r") as f:
            js = json.load(f)
        dev = joints_key(js, "device_pts3d", name_table)
        cam3d = np.stack([joints_key(js, f"camera_{v}_pts3d", name_table) for v in ("left", "right")])
        cam2d = np.stack([joints_key(js, f"camera_{v}_pts2d", name_table)[:, :2] for v in ("left", "right")])
        pelvis_dev = None
        for key in ("pelvis", "Hips", "hips"):
            src = js.get("device_pts3d")
            if isinstance(src, dict) and key in src:
                pelvis_dev = np.asarray(src[key], np.float32)
                break
            if "joints" in js and key in js["joints"] and "device_pts3d" in js["joints"][key]:
                pelvis_dev = np.asarray(js["joints"][key]["device_pts3d"], np.float32)
                break
        pose_pelvis = dev - pelvis_dev if pelvis_dev is not None else None
        pelvis_cam = None
        if pelvis_dev is not None and rig is not None:
            pelvis_cam = np.stack([rig.cameras[v].device_to_cam(torch.from_numpy(pelvis_dev)).numpy() for v in range(2)])
        headset = None
        if has_joints_key(js, "trans"):  # UnrealEgo-RW: joints also given in the mocap world frame -> headset pose
            world = joints_key(js, "trans", name_table)
            if up_axis is None:
                up_axis = detect_up_axis(world)
            headset = headset_from_world_joints(dev, world, up_axis)
        if rig is not None and len(reproj_err) < 32:
            uv, _ = rig.cameras[0].project_cam(torch.from_numpy(cam3d[0]))
            reproj_err.append(float((uv.numpy() - cam2d[0]).__abs__().mean()))
        writer.add_frame(
            fid, [load_image(left[fid], cfg["img_size"]), load_image(right[fid], cfg["img_size"])],
            pose_dev=dev, pose_pelvis=pose_pelvis, pelvis_cam=pelvis_cam, pts3d_cam=cam3d, pts2d=cam2d,
            headset_T=headset[0] if headset else None, gravity_dev=headset[1] if headset else None,
            device_height=headset[2] if headset else None,
        )
    n = writer.finalize()
    return seq_id, os.path.basename(out_dir), n, (float(np.mean(reproj_err)) if reproj_err else None)


def fit_rw_rig(rgb_root: str, pose_root: str, seqs: Sequence[str], calib_left: str, calib_right: str, max_frames: int = 400) -> StereoRig:
    """Fit per-camera R, t (device -> camera) from json annotations and attach the RW Scaramuzza calibration."""
    src, dst = [[], []], [[], []]
    n = 0
    for seq in seqs:
        json_dir = os.path.join(pose_root, seq, JSON_SUBDIR)
        if not os.path.isdir(json_dir):
            continue
        for jp in sorted(glob.glob(os.path.join(json_dir, "frame_*.json")))[:20]:
            with open(jp, "r") as f:
                js = json.load(f)
            dev = joints_key(js, "device_pts3d", UNREALEGO_RW_JSON_NAMES)
            for v, name in enumerate(("left", "right")):
                cam = joints_key(js, f"camera_{name}_pts3d", UNREALEGO_RW_JSON_NAMES)
                src[v].append(dev)
                dst[v].append(cam)
            n += 1
            if n >= max_frames:
                break
        if n >= max_frames:
            break
    cams = []
    for v, (name, calib_path) in enumerate((("left", calib_left), ("right", calib_right))):
        s = torch.from_numpy(np.concatenate(src[v])).float()
        d = torch.from_numpy(np.concatenate(dst[v])).float()
        R, t = fit_rigid_transform(s, d)
        resid = (s @ R.T + t - d).norm(dim=-1).mean().item()
        print(f"[rig] {name}: fitted R,t from {s.shape[0]} joints, mean residual {resid:.3f} cm, t={t.tolist()}")
        with open(calib_path, "r") as f:
            calib = json.load(f)
        cams.append(camera_from_calib_dict(calib, name, R=R.tolist(), t=t.tolist()))
    return StereoRig(cams, name="unrealego_rw")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["unrealego2", "unrealego_rw"])
    ap.add_argument("--rgb_root", required=True)
    ap.add_argument("--pose_root", default=None)
    ap.add_argument("--split_txt", default=None, help="sequence list; if omitted all sequences under rgb_root are used")
    ap.add_argument("--split_name", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--img_size", type=int, default=512, help="stored resolution; training resizes on the fly (256/384/512)")
    ap.add_argument("--orig_size", type=int, default=1024)
    ap.add_argument("--env_filter", default=None, help="only sequences whose path contains this string (chunked staging)")
    ap.add_argument("--calib_left", default=None)
    ap.add_argument("--calib_right", default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    pose_root = args.pose_root or args.rgb_root

    seqs = parse_split_file(args.split_txt) if args.split_txt else discover_sequences(args.rgb_root)
    seqs = [s for s in seqs if os.path.isdir(os.path.join(args.rgb_root, s, IMG_SUBDIR))]
    if args.env_filter:
        seqs = [s for s in seqs if args.env_filter in s]
    if args.limit:
        seqs = seqs[: args.limit]
    print(f"{len(seqs)} sequences to process")

    out_ds = os.path.join(args.out_root, args.dataset)
    out_split = os.path.join(out_ds, args.split_name)
    os.makedirs(out_split, exist_ok=True)

    if args.dataset == "unrealego2":
        rig_name = "unrealego2"
    else:
        rig_path = os.path.join(out_ds, "rig.json")
        if not os.path.isfile(rig_path):
            if not (args.calib_left and args.calib_right):
                raise SystemExit("UnrealEgo-RW needs --calib_left/--calib_right (Scaramuzza json from the dataset page) to build rig.json")
            # The dataset rig's extrinsics are *fitted* from GT device / camera joints - only ever from the training
            # split, so that no test annotation enters a calibration.
            if args.split_name != "train":
                raise SystemExit(f"{rig_path} does not exist and this is the {args.split_name!r} split: process the train "
                                 "split first so the rig extrinsics are fitted on training data only")
            rig = fit_rw_rig(args.rgb_root, pose_root, seqs, args.calib_left, args.calib_right)
            rig_to_json(rig, rig_path)
            with open(rig_path, "r") as f:
                js = json.load(f)
            js["fit_split"], js["fit_sequences"] = args.split_name, [s for s in seqs if os.path.isdir(os.path.join(pose_root, s, JSON_SUBDIR))][:20]
            with open(rig_path, "w") as f:
                json.dump(js, f, indent=1)
            print(f"[rig] wrote {rig_path} (extrinsics fitted on the first 20 {args.split_name} sequences)")
        rig_name = rig_path

    cfg = dict(dataset=args.dataset, rgb_root=args.rgb_root, pose_root=pose_root, out_split=out_split, img_size=args.img_size, rig=rig_name)
    entries, errs = [], []
    with Pool(args.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        for i, (seq_id, dirname, n, err) in enumerate(pool.imap_unordered(process_sequence, seqs, chunksize=1)):
            if n > 0:
                entries.append({"id": seq_id, "dir": dirname, "num_frames": n})
            if err is not None:
                errs.append(err)
            if (i + 1) % 20 == 0 or i + 1 == len(seqs):
                msg = f" | json-vs-projected pts2d mean abs err {np.mean(errs):.2f} px" if errs else ""
                print(f"[{args.split_name}] {i + 1}/{len(seqs)} sequences, {sum(e['num_frames'] for e in entries)} frames{msg}")

    # merge with an existing index (chunked staging appends environment by environment)
    index_path = os.path.join(out_split, "index.json")
    if os.path.isfile(index_path):
        with open(index_path, "r") as f:
            old = {s["id"]: s for s in json.load(f)["sequences"]}
        for e in entries:
            old[e["id"]] = e
        entries = list(old.values())
    entries.sort(key=lambda e: e["id"])
    DatasetIndex.write(out_split, args.dataset, args.split_name, rig=rig_name, image_size=(args.img_size, args.img_size),
                       orig_size=(args.orig_size, args.orig_size), sequences=entries,
                       extra={"aux_provenance": "gt_derived"})   # headset_T / gravity / height fitted from the GT joints
    print(f"done: {len(entries)} sequences, {sum(e['num_frames'] for e in entries)} frames -> {out_split}")


if __name__ == "__main__":
    main()

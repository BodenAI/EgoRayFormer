#!/usr/bin/env python
"""Fit one stereo rig per capture session from training annotations and assign it to every sequence of the records.

UnrealEgo-RW was recorded on six days and the camera-from-device transform differs between them (the cameras were
re-mounted): a single rig is off by 0.4-2.4 cm (median over a sequence's joints) on the other days, while it is stable
within a sequence (std ~0.1 cm). This script fits the extrinsics per session from the stored device-frame joints
(``pose_dev``) and camera-frame joints (``pts3d_cam``) of the ``--fit_splits`` only (never the test split), keeps the
intrinsics of the dataset rig and writes

    <root>/<dataset>/rigs/rig_<session>.json     one per session with fitting data
    <root>/<dataset>/rigs/rig_train_all.json     fitted on all fitting data (single-rig baseline without test data)

A session without fitting data (UnrealEgo-RW: 2023_01_17, test subject S1 only) takes the rig of the nearest fitted
session in chronological order (``--fallback nearest``, default; 2023_01_18, residual 0.34 cm) or the dataset rig
(``--fallback dataset``). NOTE: the dataset rig ``rig.json`` is *not* an official calibration - only
its intrinsics come from the UnrealEgo-RW calibration files, its extrinsics are fitted by preprocess_unrealego_raw.py
from the first 20 sequences of whichever split was processed first; if that was not the train split, it was fitted on
test annotations. ``--fallback dataset`` is therefore
only leak-free when ``rig.json`` carries ``fit_split == "train"``; the script refuses it otherwise. Every split's
index.json gets ``"rig"`` per sequence (the top-level ``"rig"`` stays as the default; ``dataset``-fallback sequences
simply have no per-sequence entry); images and annotations are not touched. The residual table reports how well each
rig transfers to the test sequences of its session (they are not used for fitting). By default only the train split
is used for fitting (``--fit_splits train``): fitting on validation GT would tune the validation rigs on the very
sequences used for model selection.

    python scripts/fit_session_rigs.py --root /data/unrealego/processed --dataset unrealego_rw [--dry_run]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from egorayformer.data.camera import StereoRig, camera_from_calib_dict, fit_rigid_transform, get_rig, rig_to_json  # noqa: E402

SPLITS = ("train", "validation", "test")


def _residual_cm(R: torch.Tensor, t: torch.Tensor, src: np.ndarray, dst: np.ndarray) -> float:
    s, d = torch.from_numpy(src).float(), torch.from_numpy(dst).float()
    return float((s @ R.T + t - d).norm(dim=-1).median())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--dataset", default="unrealego_rw")
    ap.add_argument("--session_regex", default=r"^[^/]+/([^/]+)/", help="first group = session key of a sequence id")
    ap.add_argument("--fit_splits", default="train", help="splits used for fitting (comma separated; never test, "
                    "and validation only if you accept tuning the validation rigs on the model-selection data)")
    ap.add_argument("--max_frames_per_seq", type=int, default=50)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--rigs_only", action="store_true", help="write the rig files but leave the index.json files unchanged")
    ap.add_argument("--fallback", choices=("dataset", "nearest"), default="nearest",
                    help="rig of a session without fitting data: the nearest fitted session (default) or the dataset rig "
                         "(only accepted when rig.json records fit_split == train)")
    args = ap.parse_args()
    fit_splits = [s for s in args.fit_splits.split(",") if s]
    if "test" in fit_splits:
        raise SystemExit("refusing to fit rigs on the test split")
    ds_dir = os.path.join(args.root, args.dataset)
    splits = [s for s in SPLITS if os.path.isfile(os.path.join(ds_dir, s, "index.json"))]
    infos = {s: json.load(open(os.path.join(ds_dir, s, "index.json"))) for s in splits}
    base_name = infos[splits[0]]["rig"]
    base = get_rig(base_name)
    if args.fallback == "dataset" and os.path.isfile(base_name):
        with open(base_name, "r") as f:
            fit_split = json.load(f).get("fit_split")
        if fit_split != "train":
            raise SystemExit(f"--fallback dataset refused: {base_name} records fit_split={fit_split!r} (its extrinsics were "
                             "fitted on data of unknown / non-train origin, see the module docstring); use --fallback nearest "
                             "or rebuild rig.json from the train split")
    V = base.num_views
    pat = re.compile(args.session_regex)

    # (session, split) -> per view [src (device), dst (camera)] joint arrays
    data = defaultdict(lambda: [[[], []] for _ in range(V)])
    seq_session, n_seq = {}, defaultdict(int)
    for sp in splits:
        for s in infos[sp]["sequences"]:
            m = pat.search(s["id"])
            if not m:
                raise SystemExit(f"session regex {args.session_regex!r} does not match sequence {s['id']!r}")
            key = m.group(1)
            seq_session[(sp, s["id"])] = key
            n_seq[(key, sp)] += 1
            with np.load(os.path.join(ds_dir, sp, s["dir"], "meta.npz")) as z:
                pose, cam = z["pose_dev"], z["pts3d_cam"]
            sel = np.linspace(0, len(pose) - 1, min(len(pose), args.max_frames_per_seq)).astype(int)
            for v in range(V):
                p, c = pose[sel].reshape(-1, 3), cam[sel, v].reshape(-1, 3)
                ok = np.isfinite(p).all(-1) & np.isfinite(c).all(-1)
                data[(key, sp)][v][0].append(p[ok])
                data[(key, sp)][v][1].append(c[ok])
    cat = {k: [(np.concatenate(a), np.concatenate(b)) for a, b in vs] for k, vs in data.items()}
    sessions = sorted({k for k, _ in cat})

    def fit(keys):
        out = []
        for v in range(V):
            src = np.concatenate([cat[k][v][0] for k in keys])
            dst = np.concatenate([cat[k][v][1] for k in keys])
            out.append(fit_rigid_transform(torch.from_numpy(src).float(), torch.from_numpy(dst).float()))
        return out

    extr = {"train_all": fit([k for k in cat if k[1] in fit_splits])}
    fitted = [s for s in sessions if any((s, sp) in cat for sp in fit_splits)]
    for s in fitted:
        extr[s] = fit([(s, sp) for sp in fit_splits if (s, sp) in cat])
    DATASET_RIG = "dataset"  # pseudo source key: keep the dataset rig (no per-sequence rig entry)
    default = [(c.R, c.t) for c in base.cameras]
    extr[DATASET_RIG] = default
    source = {s: s for s in fitted}
    for s in sessions:
        if s not in source:
            if args.fallback == "dataset":
                source[s] = DATASET_RIG
                print(f"[{s}] no {'/'.join(fit_splits)} data -> keeping the dataset rig ({base.name}, fitted on the train split)")
            else:
                i = sessions.index(s)
                source[s] = min(fitted, key=lambda f: (abs(sessions.index(f) - i), sessions.index(f)))
                print(f"[{s}] no {'/'.join(fit_splits)} data -> using the rig of session {source[s]}")
    print(f"\nmedian joint residual (cm, left/right); fitting splits = {fit_splits}")
    print(f"{'session':12s} {'split':10s} {'#seq':>5s} {'rig':>12s} {'dataset rig':>14s} {'train_all':>14s} {'session rig':>14s}")
    for s in sessions:
        for sp in splits:
            if (s, sp) not in cat:
                continue
            cols = []
            for rig in (default, extr["train_all"], extr[source[s]]):
                cols.append("/".join(f"{_residual_cm(R, t, *cat[(s, sp)][v]):.2f}" for v, (R, t) in enumerate(rig)))
            print(f"{s:12s} {sp:10s} {n_seq[(s, sp)]:5d} {source[s]:>12s} {cols[0]:>14s} {cols[1]:>14s} {cols[2]:>14s}")
    print("\nsession rig vs train_all rig: rotation (deg) / translation (cm), left | right")
    for s in fitted:
        parts = []
        for v in range(V):
            (R0, t0), (R1, t1) = extr["train_all"][v], extr[s][v]
            ang = math.degrees(math.acos(max(-1.0, min(1.0, (float(torch.trace(R1 @ R0.T)) - 1) / 2))))
            parts.append(f"{ang:.2f} / {float((t1 - t0).norm()):.2f}")
        print(f"{s:12s} " + " | ".join(parts))

    if args.dry_run:
        return
    os.makedirs(os.path.join(ds_dir, "rigs"), exist_ok=True)
    paths = {}
    for key, cams_rt in extr.items():
        if key == DATASET_RIG:
            continue
        cams = [camera_from_calib_dict(base.cameras[v].calib_dict(), base.cameras[v].name, R=R.tolist(), t=t.tolist())
                for v, (R, t) in enumerate(cams_rt)]
        paths[key] = os.path.join(ds_dir, "rigs", f"rig_{key}.json")
        rig_to_json(StereoRig(cams, name=f"{args.dataset}_{key}"), paths[key])
        with open(paths[key], "r") as f:  # provenance: which splits / sessions the extrinsics were fitted on
            js = json.load(f)
        js["fit_splits"], js["fit_sessions"] = fit_splits, ([key] if key != "train_all" else fitted)
        with open(paths[key], "w") as f:
            json.dump(js, f, indent=2)
    print(f"single rig without test data: {paths['train_all']}")
    if args.rigs_only:
        return
    for sp in splits:
        for s in infos[sp]["sequences"]:
            src = source[seq_session[(sp, s["id"])]]
            if src == DATASET_RIG:
                s.pop("rig", None)  # no per-sequence entry -> the loader uses the dataset's official rig
            else:
                s["rig"] = paths[src]
        with open(os.path.join(ds_dir, sp, "index.json"), "w") as f:
            json.dump(infos[sp], f, indent=1)
        n = defaultdict(int)
        for s in infos[sp]["sequences"]:
            n[os.path.basename(s["rig"]) if "rig" in s else f"<dataset rig {infos[sp]['rig']}>"] += 1
        print(f"[{sp}] per-sequence rigs written: {dict(n)}")


if __name__ == "__main__":
    main()

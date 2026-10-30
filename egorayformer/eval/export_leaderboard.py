"""Write predictions in the UnrealEgo2 / UnrealEgo-RW benchmark submission format.

    <out_root>/<sequence path>/json/frame_<idx>.json   with {"joint_name": [x, y, z], ...} (cm)
"""

from __future__ import annotations

import json
import os
import zipfile
from typing import Iterable, Sequence

import numpy as np

from ..data.skeleton import JOINT_NAMES, UNREALEGO_JSON_NAMES, UNREALEGO_RW_JSON_NAMES

NAME_TABLES = {"unrealego2": UNREALEGO_JSON_NAMES, "unrealego_rw": UNREALEGO_RW_JSON_NAMES}


def write_sequence_predictions(
    out_root: str, seq_id: str, frame_ids: Sequence[int], pred_cm: np.ndarray, name_style: str = "unrealego2"
) -> int:
    """pred_cm [N, 16, 3] device-relative joints in cm -> N json files. Returns number written."""
    table = NAME_TABLES[name_style]
    json_dir = os.path.join(out_root, seq_id.strip("/"), "json")
    os.makedirs(json_dir, exist_ok=True)
    for fid, p in zip(frame_ids, pred_cm):
        d = {table[name][0]: [float(v) for v in p[j]] for j, name in enumerate(JOINT_NAMES)}
        with open(os.path.join(json_dir, f"frame_{int(fid)}.json"), "w") as f:
            json.dump(d, f)
    return len(frame_ids)


def zip_directory(dir_path: str, zip_path: str) -> str:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(dir_path):
            for fn in files:
                full = os.path.join(root, fn)
                zf.write(full, os.path.relpath(full, os.path.dirname(dir_path)))
    return zip_path

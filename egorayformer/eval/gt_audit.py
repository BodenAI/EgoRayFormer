"""GT-free input audit.

A device-protocol number is a deployable prediction only if nothing the model reads at inference is derived from the
ground truth. The protocol name alone does not establish that: the root / pelvis protocols project with the GT root
(``origin_cam``), the gravity protocol rotates the target frame with a gravity direction that is fitted from the GT
joints on UnrealEgo-RW (``target_rot``, passed at inference regardless of ``aux_cond``), and the auxiliary headset
conditioning may carry GT-derived headset poses. ``audit_batch`` lists every such input of an evaluated batch;
``merge_audits`` pools the batches of an epoch; the LightningModule writes the result next to the per-sequence errors
and into the export directory (``protocol.json``).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

ROT_TOL = 1e-5     # |target_rot - I| above this counts as a rotated (gravity) target frame
ORIGIN_TOL = 1e-3  # cm: device-protocol origin_cam must be the rig's camera translation


def audit_batch(batch: Dict, target_frame: str, model, rig_ids: torch.Tensor) -> Dict:
    """-> {"target_frame", "gt_free", "gt_inputs" [...], "sensor_inputs" [...], "batches": 1} for one batch."""
    gt_inputs, sensor_inputs = set(), set()
    prov: Optional[List[str]] = batch.get("aux_provenance")
    B = batch["img"].shape[0]
    if target_frame in ("root", "pelvis"):
        gt_inputs.add(f"origin_cam = GT {target_frame} position in every camera frame (oracle-root projection)")
    else:
        oc = batch["origin_cam"].float()                                              # [B, T, V, 1, 3]
        t = model.rig_bank.t.index_select(0, rig_ids.to(model.rig_bank.t.device)).to(oc.device)   # [B, V, 3]
        dev = float((oc[:, :, :, 0, :] - t[:, None]).abs().amax()) if oc.numel() else 0.0
        if dev > ORIGIN_TOL:
            gt_inputs.add(f"origin_cam deviates from the rig camera translation by {dev:.3g} cm")
    rot = batch.get("target_rot")
    if rot is not None:
        eye = torch.eye(3, dtype=rot.dtype, device=rot.device)
        moved = (rot - eye).abs().amax(dim=(-1, -2)) > ROT_TOL                        # [B, T]
        for b in range(B):
            if bool(moved[b].any()):
                p = prov[b] if prov is not None else "unknown"
                (sensor_inputs if p == "sensor" else gt_inputs).add(f"target_rot (gravity-aligned frame) from {p} gravity")
    aux = batch.get("aux_cond")
    if getattr(model, "use_aux_cond", False) and aux is not None:
        has = torch.isfinite(aux).all(dim=-1).any(dim=-1)                              # [B]
        for b in range(B):
            if bool(has[b]):
                p = prov[b] if prov is not None else "unknown"
                (sensor_inputs if p == "sensor" else gt_inputs).add(f"aux_cond (gravity / height / rotation) from {p} headset data")
    return {"target_frame": target_frame, "gt_free": not gt_inputs, "gt_inputs": sorted(gt_inputs),
            "sensor_inputs": sorted(sensor_inputs), "batches": 1}


def merge_audits(a: Dict, b: Dict) -> Dict:
    frames = sorted({a["target_frame"], b["target_frame"]})
    gt = sorted(set(a["gt_inputs"]) | set(b["gt_inputs"]))
    return {"target_frame": frames[0] if len(frames) == 1 else "+".join(frames), "gt_free": not gt, "gt_inputs": gt,
            "sensor_inputs": sorted(set(a["sensor_inputs"]) | set(b["sensor_inputs"])),
            "batches": int(a.get("batches", 0)) + int(b.get("batches", 0))}


def format_audit(a: Dict) -> str:
    head = f"target_frame={a['target_frame']} gt_free={a['gt_free']} ({a.get('batches', 0)} batches)"
    parts = [head]
    if a["gt_inputs"]:
        parts.append("GT-derived inputs: " + "; ".join(a["gt_inputs"]))
    if a["sensor_inputs"]:
        parts.append("sensor inputs: " + "; ".join(a["sensor_inputs"]))
    return " | ".join(parts)

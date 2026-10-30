"""LightningModule wrapping EgoRayFormer: heatmap pre-training stage and end-to-end stage."""

from __future__ import annotations

import copy
import math
import os
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from ..data.heatmaps import render_gaussian_heatmaps
from ..data.records import IMAGENET_MEAN, IMAGENET_STD
from ..data.skeleton import SKELETON_IDS, skeleton_id
from ..eval.export_leaderboard import write_sequence_predictions
from ..eval.metrics import PoseMetricAccumulator
from ..models.egorayformer import EgoRayFormer
from . import losses as L
from .config import LossConfig, ModelConfig, OptimConfig, milestone_steps, to_dict


class EgoRayFormerLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: ModelConfig = ModelConfig(),
        stage: str = "e2e",
        loss: LossConfig = LossConfig(),
        optim: OptimConfig = OptimConfig(),
        encoder_pretrained: Optional[str] = None,
        heatmap_size: int = 64,
        heatmap_sigma: float = 1.5,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        ema_horizon_epochs: float = 0.0,
        eval_streaming: bool = False,
        export_dir: Optional[str] = None,
        export_name_style: str = "unrealego2",
        pck_threshold_mm: float = 100.0,
        calib_dump: Optional[str] = None,     # path prefix: <prefix>_{val,test}.npz with the per-joint Mahalanobis
                                              # distances / errors of the epoch (post-hoc temperature scaling)
        # exposure bias: with probability p per clip the track token is fed the model's own streamed
        # prediction of the previous frame (no-grad eval-mode pass, as at test) instead of noised GT; p ramps up
        # linearly over the first ``track_self_cond_ramp`` of training
        track_self_cond_p: float = 0.0,
        track_self_cond_ramp: float = 0.5,
        eval_precision: str = "auto",         # "auto" (trainer precision) | "fp32": evaluate outside autocast
        # path prefix -> <prefix>_{val,test}.csv with per-sequence error sums and
        # counts (input of a paired sequence bootstrap)
        seq_error_dump: Optional[str] = None,
        # path prefix -> <prefix>_{val,test}.npz with the per-joint predictions of the last refinement layer (final,
        # raw P_geo, the actual geometric candidate, P_learn, gate, GT, anchor, rays; error / gate analysis)
        diag_dump: Optional[str] = None,
        # with diag_dump and > 0: the read-only probe of the last layer in the same streamed forward - lines of +-extent cm in steps of diag_probe_step_cm along / across the anchor's ray, and the
        # nearest sampled hypothesis to the GT (3D, and 2D in input pixels per view); needs continuous streaming
        diag_probe_extent_cm: float = 0.0,
        diag_probe_step_cm: float = 1.0,
        gt_audit: bool = True,                # record which inference inputs carry GT information (eval/gt_audit.py)
        # rig augmentation: every training clip (probability rig_aug_p) gets one rotation per camera about
        # its optical centre, rotation vector ~ N(0, rig_aug_rot_deg^2) per axis, shared by all frames of the clip.
        # "consistent": images and 2D labels warped + the model's rig rotated; "image_only": images / labels warped, rig
        # unchanged; "calib_only": images unchanged, the model's rig rotated (calibration error)
        rig_aug_rot_deg: float = 0.0,
        rig_aug_mode: str = "consistent",
        rig_aug_p: float = 1.0,
        # evaluation under a known rig rotation (perturbation curves): angle per camera, random axis per sequence
        eval_rig_rot_deg: float = 0.0,
        eval_rig_mode: str = "consistent",
        # selection metric ``val/mpjpe`` (the checkpoint monitor): this dataset's validation MPJPE - an error when it is
        # not evaluated or not finite; unset: the frame-pooled MPJPE of a multi-dataset validation,
        # the dataset itself for a single-dataset one
        select_val_dataset: Optional[str] = None,
        # Weights-only initialisation from another e2e run's checkpoint (UE2 -> RW fine-tuning): the
        # model weights (the EMA copy by default) are loaded, every key must match; optimizer, scheduler, epoch and step
        # start fresh - unlike --ckpt_path, which resumes the source training. Applied at the start of a new fit only
        # (on_fit_start): rebuilding the module from a checkpoint never reads the source file
        init_from: Optional[str] = None,
        init_from_ema: bool = True,
    ):
        super().__init__()
        assert stage in ("heatmap", "e2e")
        self.save_hyperparameters()
        self.stage = stage
        model_cfg = {**to_dict(ModelConfig()), **to_dict(model)}
        self.loss_cfg = {**to_dict(LossConfig()), **to_dict(loss)}
        self.optim_cfg = {**to_dict(OptimConfig()), **to_dict(optim)}
        self.heatmap_size, self.heatmap_sigma = heatmap_size, heatmap_sigma
        self.eval_streaming = eval_streaming
        self.export_dir, self.export_name_style = export_dir, export_name_style
        self.calib_dump = calib_dump
        self.track_self_cond_p, self.track_self_cond_ramp = track_self_cond_p, track_self_cond_ramp
        assert eval_precision in ("auto", "fp32"), eval_precision
        self.eval_precision = eval_precision
        self.seq_error_dump, self.diag_dump, self.gt_audit = seq_error_dump, diag_dump, gt_audit
        if float(diag_probe_extent_cm) > 0 and (not diag_dump or float(diag_probe_step_cm) <= 0):
            raise ValueError("model.diag_probe_extent_cm needs model.diag_dump and diag_probe_step_cm > 0")
        self.diag_probe = (float(diag_probe_extent_cm), float(diag_probe_step_cm)) if float(diag_probe_extent_cm) > 0 else None
        for mode in (rig_aug_mode, eval_rig_mode):
            assert mode in ("consistent", "image_only", "calib_only"), mode
        if (float(rig_aug_rot_deg) > 0 or float(eval_rig_rot_deg) > 0) and not to_dict(model).get("fov_circle", False):
            # the warped images are black outside the image circle: labels and projections must use the same region
            raise ValueError("rig augmentation / evaluation under a rig rotation needs model.model.fov_circle: true "
                             "(configs/fov_circle.yaml), so that warped images, 2D labels and projections share one valid region")
        self.rig_aug_rot_deg, self.rig_aug_mode, self.rig_aug_p = float(rig_aug_rot_deg), rig_aug_mode, float(rig_aug_p)
        self.eval_rig_rot_deg, self.eval_rig_mode = float(eval_rig_rot_deg), eval_rig_mode
        self.select_val_dataset = select_val_dataset
        self._diag_rows: List[Dict[str, Any]] = []
        self._audit: Optional[Dict[str, Any]] = None
        self._stream_key, self._stream_state, self._stream_next_fid, self._stream_breaks = None, None, None, 0

        self.model = EgoRayFormer(**model_cfg)
        if encoder_pretrained:
            self.load_encoder_pretrained(encoder_pretrained)
        elif not init_from:
            self._check_backbone_weights()
        self.init_from, self.init_from_ema = init_from, init_from_ema
        self.use_ema = use_ema
        self.ema_decay, self.ema_horizon_epochs = ema_decay, ema_horizon_epochs
        self.ema: Optional[AveragedModel] = None
        if use_ema:
            self.ema = AveragedModel(self.model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay), use_buffers=True)
        # calibration coverage thresholds follow the training likelihood (Student-t nu vs. Gaussian), see metrics.py
        self.metrics = PoseMetricAccumulator(pck_threshold_mm=pck_threshold_mm,
                                             nll_nu=self.loss_cfg["nll_nu"] if self.loss_cfg.get("nll_type", "student_t") == "student_t" else float("inf"))

    def on_fit_start(self) -> None:
        """``init_from`` is a one-off initialisation of a new training: it runs here, at the start of a
        fit that does not resume a checkpoint (``--ckpt_path`` restores the weights itself, before this hook), and the
        EMA copy starts from the same weights. load_from_checkpoint, validate and test only rebuild the module and never
        read the source checkpoint, so a fine-tuned checkpoint is complete without it."""
        if not self.init_from:
            return
        if self.trainer.ckpt_path:
            print(f"[EgoRayFormer] resuming {self.trainer.ckpt_path}: init_from {self.init_from} is not applied")
            return
        self.load_init_weights(self.init_from, prefer_ema=self.init_from_ema)
        if self.ema is not None:
            self.ema.module.load_state_dict(self.model.state_dict())

    def on_train_start(self) -> None:
        """Tie the EMA decay to the step budget. With ``ema_horizon_epochs`` > 0 the averaging horizon
        1 / (1 - decay) is set to that many epochs of optimizer steps (a fixed 0.999 lags several epochs behind when
        an epoch has only ~300 steps, as in the 4-GPU video runs)."""
        if self.ema is None or self.ema_horizon_epochs <= 0:
            return
        try:
            steps_per_epoch = int(self.trainer.num_training_batches) // max(1, int(self.trainer.accumulate_grad_batches))
        except Exception:  # noqa: BLE001
            return
        if steps_per_epoch <= 0 or steps_per_epoch == float("inf"):
            return
        horizon = max(10.0, self.ema_horizon_epochs * steps_per_epoch)
        decay = 1.0 - 1.0 / horizon
        self.ema.multi_avg_fn = get_ema_multi_avg_fn(decay)
        print(f"[EgoRayFormer] EMA decay {decay:.5f} (horizon {horizon:.0f} steps = {self.ema_horizon_epochs} epochs x {steps_per_epoch} steps)")

    # ------------------------------------------------------------------ weights
    def load_encoder_pretrained(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        clean = {}
        for k, v in sd.items():
            k2 = k
            for prefix in ("model.model.", "ema.module.", "model."):
                if k2.startswith(prefix):
                    k2 = k2[len(prefix):]
                    break
            if k2.startswith("encoder.") or k2.startswith("heatmap_head.") or k2.startswith("feat_proj."):
                clean[k2] = v
        res = self.model.load_state_dict(clean, strict=False)
        own = set(self.model.state_dict())
        n_loaded = len(set(clean) & own)
        n_enc_own = sum(k.startswith("encoder.") for k in own)
        n_enc_loaded = sum(k.startswith("encoder.") for k in set(clean) & own)
        print(f"[EgoRayFormer] encoder_pretrained: loaded {n_loaded} tensors from {path} "
              f"(encoder {n_enc_loaded}/{n_enc_own}, unexpected {len(res.unexpected_keys)})")
        if n_loaded == 0:
            raise RuntimeError(f"{path} contains no encoder weights")
        if n_enc_own > 0 and n_enc_loaded == 0:
            # e.g. an R18 heatmap checkpoint for a ViT encoder: only heatmap_head / feat_proj would match
            raise RuntimeError(f"{path} matched none of the {n_enc_own} encoder tensors: stage-1 checkpoint of another encoder")
        if n_enc_loaded < n_enc_own:
            import warnings

            warnings.warn(
                f"encoder_pretrained only matched {n_enc_loaded}/{n_enc_own} encoder tensors - the heatmap "
                "pre-training checkpoint was probably produced with a different encoder configuration."
            )

    def load_init_weights(self, path: str, prefer_ema: bool = True) -> None:
        """Weights-only start from a Lightning checkpoint of the same architecture. Calibration
        buffers are not part of a checkpoint (rigs are added from the datamodule); every other key must match in name
        and shape - a fine-tuning run of a different configuration fails here instead of training partly from scratch."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        prefix = "ema.module." if prefer_ema and any(k.startswith("ema.module.") for k in sd) else "model."
        src = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        src = {k: v for k, v in src.items() if ".rigs." not in k and "rig_bank." not in k and ".proposal_unc." not in k}
        own = self.model.state_dict()
        missing = sorted(set(own) - set(src))
        unexpected = sorted(set(src) - set(own))
        shape = sorted(k for k in set(src) & set(own) if src[k].shape != own[k].shape)
        if missing or unexpected or shape:
            raise RuntimeError(f"init_from {path}: the checkpoint does not match this model ({len(missing)} missing, "
                               f"{len(unexpected)} unexpected, {len(shape)} shape mismatches; e.g. "
                               f"{(missing + unexpected + shape)[:3]})")
        self.model.load_state_dict(src, strict=True)
        print(f"[EgoRayFormer] init_from: loaded {len(src)} tensors ({prefix[:-1]} weights, epoch {ckpt.get('epoch')}, "
              f"step {ckpt.get('global_step')}) from {path}; optimizer, scheduler and step counters start fresh")

    def _check_backbone_weights(self) -> None:
        """Without a stage-1 checkpoint the backbone must start from the pretrained weights it asked for:
        ``create_timm_model`` silently falls back to ImageNet or random init (EGORAYFORMER_ALLOW_WEIGHT_FALLBACK=1 allows it)."""
        enc = self.model.encoder
        if not getattr(enc, "requested_pretrained", False) or os.environ.get("EGORAYFORMER_ALLOW_WEIGHT_FALLBACK") == "1":
            return
        if not enc.loaded_pretrained or enc.loaded_name != enc.requested_name:
            got = enc.loaded_name if enc.loaded_pretrained else "random initialisation"
            raise RuntimeError(f"encoder {enc.requested_name!r}: pretrained weights requested but got {got!r} - put the file "
                               "into weights/timm (EGORAYFORMER_TIMM_WEIGHTS_DIR) or set EGORAYFORMER_ALLOW_WEIGHT_FALLBACK=1")

    def eval_model(self) -> nn.Module:
        return self.ema.module if (self.ema is not None and not self.training) else self.model

    # ------------------------------------------------------------------ rigs of the attached datamodule
    def setup(self, stage: Optional[str] = None) -> None:
        """Register the camera rigs of every dataset source of the datamodule that the model does not know yet, so a
        model trained on one device can be evaluated / fine-tuned on another rig without editing ``model.rigs``."""
        dm = getattr(getattr(self, "trainer", None), "datamodule", None)
        if dm is None:
            return
        from ..data.records import DatasetIndex
        from ..data.stereo_video import resolve_rig_swap

        rigs = []
        for src in list(getattr(dm, "train_sources", [])) + list(getattr(dm, "val_sources", [])) + list(getattr(dm, "test_sources", [])):
            try:
                rigs.extend(r for r in DatasetIndex(dm.root, src["dataset"], src["split"]).all_rigs if r not in rigs)
            except FileNotFoundError:
                continue
        swap = getattr(dm, "rig_swap", None)   # neighbouring-session calibrations of the rig-swap evaluation
        rigs.extend(s for s in dict.fromkeys(resolve_rig_swap(r, swap) for r in list(rigs)) if s not in rigs)
        for m in ([self.model] + ([self.ema.module] if self.ema is not None else [])):
            added = m.add_rigs([r for r in rigs if r not in m.rig_names])
            if added:
                print(f"[EgoRayFormer] registered additional rigs from the datamodule: {added}")

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # calibration buffers are derived from the rig files (non-persistent); drop stale copies of older checkpoints
        sd = checkpoint.get("state_dict", {})
        for k in [k for k in sd if ".rigs." in k or ".rig_bank." in k]:
            del sd[k]
        # the former ``proposal_unc`` head was removed (it never received a gradient); older checkpoints
        # still carry its constant weights
        for k in [k for k in sd if ".proposal_unc." in k]:
            del sd[k]
        # use_ema=False on a checkpoint trained with EMA evaluates the raw weights (EMA does not change the training
        # trajectory, so this is the exact "without EMA" ablation); the opposite case initialises the EMA from the model
        ema_keys = [k for k in sd if k.startswith("ema.")]
        if self.ema is None and ema_keys:
            for k in ema_keys:
                del sd[k]
        elif self.ema is not None and not ema_keys:
            for k in [k for k in sd if k.startswith("model.")]:
                sd["ema.module." + k[len("model."):]] = sd[k].clone()
            sd.setdefault("ema.n_averaged", torch.tensor(0, dtype=torch.long))

    # ------------------------------------------------------------------ batch prep
    def _rig_ids(self, batch: Dict) -> torch.Tensor:
        names = self.model.rig_names
        return torch.tensor([names.index(r) for r in batch["rig"]], device=self.device, dtype=torch.long)

    def _skeleton_ids(self, batch: Dict) -> Optional[torch.Tensor]:
        if "skeleton_id" in batch:
            return batch["skeleton_id"].to(self.device)
        return None

    def _canonical_2d(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """2D joints / visibility of the batch expressed on the model's canonical skeleton [B, T, V, J_canon, *].

        Samples whose skeleton is the canonical one are copied; for mapped skeletons only 1:1 target joints transfer
        their 2D annotation to the canonical joint they copy (others are NaN / invalid)."""
        pts2d, valid = batch["pts2d"], batch["pts2d_valid"]
        J = self.model.J
        canon = skeleton_id(self.model.skeleton.name)
        sk = self._skeleton_ids(batch)
        if sk is None or bool((sk == canon).all()):
            return pts2d[..., :J, :], valid[..., :J]
        B, T, V = pts2d.shape[:3]
        out = torch.full((B, T, V, J, 2), float("nan"), dtype=pts2d.dtype, device=pts2d.device)
        out_v = torch.zeros((B, T, V, J), dtype=torch.bool, device=pts2d.device)
        mapper = self.model.joint_mapper
        for b in range(B):
            if int(sk[b]) == canon:
                out[b], out_v[b] = pts2d[b, :, :, :J], valid[b, :, :, :J]
            elif mapper is not None:
                src = mapper.one_to_one[int(sk[b])]                 # [Jmax] canonical joint copied by target joint k
                for k in range(min(src.numel(), pts2d.shape[3])):
                    c = int(src[k])
                    if c >= 0:
                        out[b, :, :, c], out_v[b, :, :, c] = pts2d[b, :, :, k], valid[b, :, :, k]
        return out, out_v

    def _heatmap_targets(self, batch: Dict, like: torch.Tensor) -> torch.Tensor:
        """Render GT heatmaps at the resolution of the predicted heatmaps ``like`` [B, T, V, K, h, w]."""
        pts2d, valid = self._canonical_2d(batch)
        ids = self.model.heatmap_joint_ids
        uv = pts2d[..., ids, :]                                   # [B, T, V, K, 2]
        valid = valid[..., ids]
        h, w = like.shape[-2:]
        sigma = self.heatmap_sigma * (h / float(self.heatmap_size))  # sigma defined at heatmap_size (64 for 256 input)
        return render_gaussian_heatmaps(uv, valid, (h, w), sigma).to(like.dtype)

    def _heatmap_annotated(self, batch: Dict) -> torch.Tensor:
        """[B, T, V, K] heatmap channels whose joint carries a 2D annotation in the view (inside the image or not);
        unannotated channels are excluded from the heatmap loss instead of being trained towards 0."""
        pts2d, _ = self._canonical_2d(batch)
        return torch.isfinite(pts2d[..., self.model.heatmap_joint_ids, :]).all(dim=-1)

    def _heatmap_visible(self, heatmaps: torch.Tensor, pts2d_c: torch.Tensor, thresh: float) -> torch.Tensor:
        """Occlusion pseudo-label from the predicted heatmaps: [B, T, V, J] True where the (detached) heatmap of the
        joint at its GT 2D location exceeds ``thresh``; joints without a heatmap channel (head) stay True."""
        from ..data.heatmaps import sample_heatmap_at

        B, T, V = heatmaps.shape[:3]
        J = pts2d_c.shape[-2]
        hm = heatmaps.detach().float().flatten(0, 1)                                   # [N, V, K, h, w]
        ids = self.model.heatmap_joint_ids
        uv = torch.nan_to_num(pts2d_c[..., ids, :].float(), nan=0.0).flatten(0, 1)      # [N, V, K, 2]
        vals = sample_heatmap_at(hm, uv)                                                 # [N, V, K]
        vis = torch.ones(B * T, V, J, dtype=torch.bool, device=heatmaps.device)
        vis[:, :, ids] = vals >= thresh
        return vis.reshape(B, T, V, J)

    def _canonical_gt(self, batch: Dict) -> torch.Tensor:
        """GT on the canonical skeleton [B, T, J_canon, 3]; NaN for samples of another skeleton."""
        gt = batch["pose"]
        J = self.model.J
        out = gt[..., :J, :]
        if gt.shape[-2] < J:
            out = torch.cat([gt, torch.full((*gt.shape[:-2], J - gt.shape[-2], 3), float("nan"), dtype=gt.dtype, device=gt.device)], dim=-2)
        sk = self._skeleton_ids(batch)
        if sk is not None:
            other = sk != skeleton_id(self.model.skeleton.name)
            if bool(other.any()):
                out = out.clone()
                out[other] = float("nan")
        return out

    def _pred_for_gt(self, out: Dict, key: str, batch: Dict):
        """Prediction(s) in the skeleton of the batch GT: mapped when the model has a JointMapper and the batch
        contains non-canonical skeletons, canonical otherwise."""
        mapped = f"{key}_mapped"
        if mapped in out and self._skeleton_ids(batch) is not None:
            return out[mapped]
        return out[key]

    @staticmethod
    def _prev_pose(gt: torch.Tensor) -> torch.Tensor:
        prev = torch.full_like(gt, float("nan"))
        prev[:, 1:] = gt[:, :-1]
        return prev

    # ------------------------------------------------------------------ training
    def _progress(self) -> float:
        try:
            total = int(getattr(self.trainer, "estimated_stepping_batches", 0) or 0)
            step = int(getattr(self.trainer, "global_step", 0) or 0)
        except Exception:  # noqa: BLE001 - no trainer attached (unit tests)
            return 0.0
        return float(step) / float(max(1, total))

    def _self_cond_prob(self) -> float:
        p = float(self.track_self_cond_p)
        if p <= 0:
            return 0.0
        ramp = float(self.track_self_cond_ramp)
        return p * min(1.0, self._progress() / ramp) if ramp > 0 else p

    @torch.no_grad()
    def _streamed_prev_pose(self, batch: Dict, sel: torch.Tensor, rig_ids: torch.Tensor) -> torch.Tensor:
        """Previous-frame pose as the track token sees it at test: the model's own causal streamed prediction (eval
        mode: no dropout / DN / synthetic noise, BatchNorm running statistics), shifted by one frame -> [n, T, J, 3]."""
        model = self.model
        was_training = model.training
        model.eval()
        try:
            sk = self._skeleton_ids(batch)
            rot, cond = batch.get("target_rot"), batch.get("aux_cond")
            out, _ = model.forward_streaming(
                batch["img"][sel], batch["origin_cam"][sel], rig_ids[sel], batch["dataset_id"][sel],
                target_rot=rot[sel] if rot is not None else None,
                aux_cond=cond[sel] if (cond is not None and model.use_aux_cond) else None,
                skeleton_ids=sk[sel] if (sk is not None and model.joint_mapper is not None) else None,
            )
        finally:
            model.train(was_training)
        return self._prev_pose(out["pred"].float())

    def _unc_holdout_enabled(self) -> bool:
        """True when the attached datamodule reserves training sequences for the uncertainty head."""
        try:
            return float(getattr(self.trainer.datamodule, "unc_holdout_frac", 0.0) or 0.0) > 0.0
        except Exception:  # noqa: BLE001 - no trainer / datamodule (unit tests)
            return False

    # ------------------------------------------------------------------ rig augmentation
    def _sample_rig_rotation(self, B: int, V: int, device, rot_deg: float) -> torch.Tensor:
        """One rotation per (clip, camera): rotation vector ~ N(0, rot_deg^2) per axis -> dR [B, V, 3, 3], float32 even
        under autocast (the K @ K of so3_exp ran in bf16)."""
        from ..models.layers.ray_geometry import so3_exp

        device = torch.device(device)
        with torch.autocast(device_type=device.type, enabled=False):
            return so3_exp(torch.randn(B, V, 3, device=device, dtype=torch.float32) * math.radians(rot_deg))

    def _fixed_rig_rotation(self, seq_ids: List[str], V: int, device, angle_deg: float) -> torch.Tensor:
        """Evaluation: a rotation of exactly ``angle_deg`` per camera about an axis drawn from a hash of the sequence id
        (the same rotation for every chunk of a sequence) -> dR [B, V, 3, 3]."""
        import hashlib

        from ..models.layers.ray_geometry import so3_exp

        axes = []
        for s in seq_ids:
            g = torch.Generator().manual_seed(int(hashlib.md5(str(s).encode()).hexdigest()[:8], 16))
            a = torch.randn(V, 3, generator=g)
            axes.append(a / a.norm(dim=-1, keepdim=True))
        device = torch.device(device)
        with torch.autocast(device_type=device.type, enabled=False):      # float32, as _sample_rig_rotation
            return so3_exp(torch.stack(axes).to(device=device, dtype=torch.float32) * math.radians(angle_deg))

    def _apply_fov_circle(self, batch: Dict, rig_ids: torch.Tensor) -> Dict:
        """``model.fov_circle``: 2D labels outside the image circle are outside the view - the same rule as
        the model's projections; an annotated joint there stays a negative example of the heatmap loss."""
        if not self.model.fov_circle or "pts2d" not in batch:
            return batch
        pts, vis = batch["pts2d"], batch["pts2d_valid"]                             # [B, T, V, J, 2] normalised
        B, T, V, J, _ = pts.shape
        bank = self.model.rig_bank
        size = bank.size.index_select(0, rig_ids)[:, :, None]                      # [B, V, 1, 2]
        u = torch.nan_to_num(pts.float()).permute(0, 2, 1, 3, 4).reshape(B, V, T * J, 2) * size
        ok = bank.valid_px(u, rig_ids, circle=True).reshape(B, V, T, J).permute(0, 2, 1, 3)
        return dict(batch, pts2d_valid=vis & ok)

    @torch.no_grad()
    def _warp_for_rig_rotation(self, batch: Dict, rig_ids: torch.Tensor, dR: torch.Tensor) -> Dict:
        """Images and 2D labels as seen by the cameras rotated by dR [B, V] about their optical centres (p_new = dR p_old):
        a new pixel's ray is d_new = unproject(u_new), the old camera saw it at project(dR^T d_new) (inverse map for the
        images), and an annotated 2D joint moves to project(dR unproject(u_old)) (forward map, no GT-3D re-projection).
        Images, labels and the model's projections (``RigBank.valid_px`` with the image circle) share one
        valid region: a pixel / joint counts only if its position in the new view *and* its source in the original
        view are inside the image rectangle and circle; elsewhere the image is black and the label invalid."""
        bank = self.model.rig_bank
        img = batch["img"]                                                          # [B, T, V, 3, H, W] normalised
        B, T, V, C, H, W = img.shape
        size = bank.size.index_select(0, rig_ids)[:, :, None]                      # [B, V, 1, 2] calibration px
        ys = (torch.arange(H, device=img.device, dtype=torch.float32) + 0.5) / H
        xs = (torch.arange(W, device=img.device, dtype=torch.float32) + 0.5) / W
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        uv_new = torch.stack([gx, gy], -1).reshape(1, 1, H * W, 2).expand(B, V, H * W, 2) * size
        with torch.autocast(device_type=img.device.type, enabled=False):
            d_new = bank._unproject_px(uv_new, rig_ids)                             # [B, V, HW, 3]
            d_old = torch.einsum("bvji,bvmj->bvmi", dR.float(), d_new)             # dR^T d_new
            uv_old = bank._project_px(d_old, rig_ids)
        ok = bank.valid_px(uv_new, rig_ids, circle=True) & bank.valid_px(uv_old, rig_ids, circle=True) & (d_old[..., 2] > -1.0)
        grid = (2.0 * uv_old / size - 1.0).reshape(B * V, H, W, 2)
        src = img.permute(0, 2, 1, 3, 4, 5).reshape(B * V, T * C, H, W).float()
        warped = F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        black = torch.from_numpy(-IMAGENET_MEAN / IMAGENET_STD).to(img.device).float()   # pixel value 0 after normalisation
        fill = black.view(1, 1, C, 1, 1).expand(B * V, T, C, H, W).reshape(B * V, T * C, H, W)
        warped = torch.where(ok.reshape(B * V, 1, H, W), warped, fill)
        out = dict(batch, img=warped.reshape(B, V, T, C, H, W).permute(0, 2, 1, 3, 4, 5).to(img.dtype))
        pts, vis = batch["pts2d"], batch["pts2d_valid"]                             # [B, T, V, J, 2] normalised, [B, T, V, J]
        J = pts.shape[3]
        u = torch.nan_to_num(pts.float()).permute(0, 2, 1, 3, 4).reshape(B, V, T * J, 2) * size
        with torch.autocast(device_type=img.device.type, enabled=False):
            d0 = bank._unproject_px(u, rig_ids)
            d1 = torch.einsum("bvij,bvmj->bvmi", dR.float(), d0)                    # dR d_old
            u1_px = bank._project_px(d1, rig_ids)
        inside = bank.valid_px(u1_px, rig_ids, circle=True) & bank.valid_px(u, rig_ids, circle=True) & (d1[..., 2] > -1.0)
        u1 = (u1_px / size).reshape(B, V, T, J, 2).permute(0, 2, 1, 3, 4)
        inside = inside.reshape(B, V, T, J).permute(0, 2, 1, 3)
        finite = torch.isfinite(pts).all(-1)
        out["pts2d"] = torch.where(finite[..., None], u1.to(pts.dtype), pts)
        out["pts2d_valid"] = vis & inside
        return out

    def _augment_rig(self, batch: Dict, rig_ids: torch.Tensor, dR: torch.Tensor, mode: str) -> Tuple[Dict, bool]:
        """Apply the rig rotation dR of ``mode`` to a batch: warped images / labels for "consistent" and "image_only",
        origin_cam rotated (origin' = dR origin) for the modes in which the model's rig is rotated. Returns the batch and
        whether the model geometry must be rotated (``rig_augmentation`` context)."""
        if mode in ("consistent", "image_only"):
            batch = self._warp_for_rig_rotation(batch, rig_ids, dR)
        geometry = mode in ("consistent", "calib_only")
        if geometry:
            oc = batch["origin_cam"]                                                 # [B, T, V, 1, 3]
            with torch.autocast(device_type=oc.device.type, enabled=False):          # no bf16 rounding
                batch = dict(batch, origin_cam=torch.einsum("bvij,btvkj->btvki", dR.float(), oc.float()).to(oc.dtype))
        return batch, geometry

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        if self.model.fov_circle:
            batch = self._apply_fov_circle(batch, self._rig_ids(batch))
        if self.stage == "e2e" and self.rig_aug_rot_deg > 0:
            rig_ids = self._rig_ids(batch)
            B, V = batch["img"].shape[0], batch["img"].shape[2]
            dR = self._sample_rig_rotation(B, V, batch["img"].device, self.rig_aug_rot_deg)
            if self.rig_aug_p < 1.0:  # clips without augmentation keep the identity rotation
                keep = torch.rand(B, device=dR.device) >= self.rig_aug_p
                dR = torch.where(keep[:, None, None, None], torch.eye(3, device=dR.device).expand_as(dR), dR)
            batch, geometry = self._augment_rig(batch, rig_ids, dR, self.rig_aug_mode)
            if geometry:
                with self.model.rig_augmentation(rig_ids, dR, check_source=self.rig_aug_mode == "consistent") as virt:
                    return self._training_step(batch, batch_idx, virt)
            return self._training_step(batch, batch_idx, rig_ids)
        return self._training_step(batch, batch_idx, None)

    def _training_step(self, batch: Dict, batch_idx: int, rig_ids: Optional[torch.Tensor]) -> torch.Tensor:
        img, gt, valid = batch["img"], batch["pose"], batch["valid_frames"]
        B, T = img.shape[:2]
        rig_ids = self._rig_ids(batch) if rig_ids is None else rig_ids
        cfg = self.loss_cfg
        logs: Dict[str, torch.Tensor] = {}

        if self.stage == "heatmap":
            rig_bt = rig_ids[:, None].expand(B, T).reshape(-1)
            _, hm = self.model.encode(img.flatten(0, 1), rig_bt)
            hm = hm.reshape(B, T, *hm.shape[1:])
            target = self._heatmap_targets(batch, hm)
            loss = cfg["w_heatmap"] * L.heatmap_loss(hm, target, valid, self._heatmap_annotated(batch))
            self.log("train/heatmap", loss, prog_bar=True)
            return loss

        gt_canon = self._canonical_gt(batch)                      # NaN for samples of another skeleton
        sk_ids = self._skeleton_ids(batch)
        # Frames of uncertainty hold-out sequences (data.unc_holdout_frac) get only the pose-detached NLL: ``valid``
        # (pose losses) excludes them, ``valid_unc`` (NLL) is restricted to them when any are present in the batch.
        valid_unc, valid_all = valid, valid
        holdout_active = False                                    # explicit flag (an identity check on the masks broke under UDA)
        hold = batch.get("unc_holdout")
        prev_pose = self._prev_pose(gt_canon) if self.model.track_prior else None
        track_clean = None
        p_sc = self._self_cond_prob() if prev_pose is not None else 0.0
        if p_sc > 0:
            sc = torch.rand(B, device=img.device) < p_sc
            if bool(sc.any()):
                prev_pose = prev_pose.clone()
                prev_pose[sc] = self._streamed_prev_pose(batch, sc, rig_ids).to(prev_pose.dtype)
                track_clean = sc
            self.log("train/self_cond_frac", sc.float().mean(), on_step=True, on_epoch=False)
        if hold is not None and self._unc_holdout_enabled():
            hold_bt = hold.to(valid.device)[:, None].expand_as(valid)
            valid, valid_unc = valid & ~hold_bt, valid & hold_bt   # NLL only ever sees hold-out frames
            holdout_active = True
            if cfg.get("holdout_strict", False):
                # hold-out frames must not be easier than test frames: no 2D supervision of the encoder on them and
                # no teacher-forced GT track prior (the null token, as at the start of a streamed clip)
                valid_all = valid
                if prev_pose is not None:
                    prev_pose = torch.where(hold_bt[:, :, None, None], torch.full_like(prev_pose, float("nan")), prev_pose)
        unl = batch.get("unlabeled")
        if unl is not None and bool(unl.any()):  # unlabeled frames get no supervised loss (incl. heatmaps)
            unl_bt = unl.to(valid.device)[:, None].expand_as(valid)
            valid, valid_unc, valid_all = valid & ~unl_bt, valid_unc & ~unl_bt, valid_all & ~unl_bt
        out = self.model(
            img, batch["origin_cam"], rig_ids, batch["dataset_id"],
            prev_pose=prev_pose, gt_pose=gt_canon,
            target_rot=batch.get("target_rot"), aux_cond=batch.get("aux_cond") if self.model.use_aux_cond else None,
            skeleton_ids=sk_ids if self.model.joint_mapper is not None else None, track_clean=track_clean,
        )
        # predictions in the GT skeleton (mapped for non-canonical datasets), canonical 2D annotations
        pred = self._pred_for_gt(out, "pred", batch)
        refined = self._pred_for_gt(out, "refined", batch)
        proposal = self._pred_for_gt(out, "proposal", batch)
        pts2d_c, vis_c = self._canonical_2d(batch)
        total = img.new_zeros(())
        # proposal + refinement layers (deep supervision)
        if cfg["w_proposal"] > 0 and self.model.p0 is None:
            logs["proposal"] = cfg["w_proposal"] * L.mpjpe_loss(proposal, gt, valid)
        nll_active = "uncertainty_chol" in out and cfg["w_nll"] > 0
        # P4 loss variants: Student-t (EPFv2) or beta-NLL; optionally detach the pose so the NLL only
        # trains the uncertainty head (always detached on hold-out frames, which never receive pose losses).
        detach = bool(cfg.get("nll_detach_pose", False)) or holdout_active
        # With a pose-supervising NLL the last layer follows the L1 -> NLL curriculum below instead of its plain L1 (it
        # used to get both: double weight early in training). A detached NLL does not supervise the pose, so the last
        # layer keeps its L1 in that case - otherwise it would lose every pose loss after the warm-up.
        blend_last = nll_active and not detach
        for i, p in enumerate(refined):
            last = i == len(refined) - 1
            if last and blend_last:
                continue
            w = cfg["w_mpjpe"] if last else cfg["w_aux_layers"]
            logs[f"refine{i}"] = w * L.mpjpe_loss(p, gt, valid)
        if nll_active:  # uncertainty lives on the canonical joints
            mix = L.cosine_mix(self._progress() / max(cfg["nll_warmup_frac"], 1e-6))

            def nll(p, c):
                p = p.detach() if detach else p
                if cfg.get("nll_type", "student_t") == "beta_gauss":
                    return L.beta_gaussian_nll(p, gt_canon, c, valid_unc, cfg.get("nll_beta", 0.5))
                return L.student_t_nll(p, gt_canon, c, valid_unc, cfg["nll_nu"])

            logs["nll"] = cfg["w_nll"] * mix * nll(out["pred"], out["uncertainty_chol"])
            if blend_last:
                logs["refine_last"] = cfg["w_mpjpe"] * (1.0 - mix) * L.mpjpe_loss(pred, gt, valid)
            if cfg["w_nll_aux"] > 0 and len(out.get("uncertainty_chol_layers", [])) > 1:
                n_aux = len(out["uncertainty_chol_layers"]) - 1
                aux = sum(nll(p, c) for p, c in zip(out["refined"][-n_aux - 1:-1], out["uncertainty_chol_layers"][:-1]))
                logs["nll_aux"] = cfg["w_nll_aux"] * mix * aux / n_aux
        if "heatmaps" in out and cfg["w_heatmap"] > 0:  # 2D supervision stays on every frame (hold-out included)
            logs["heatmap"] = cfg["w_heatmap"] * L.heatmap_loss(out["heatmaps"], self._heatmap_targets(batch, out["heatmaps"]), valid_all,
                                                                self._heatmap_annotated(batch))
        if cfg["w_reproj"] > 0:
            N = B * T
            uv_pred, _ = self.model.project(
                out["pred"].reshape(N, -1, 3), batch["origin_cam"].reshape(N, self.model.V, 1, 3),
                rig_ids[:, None].expand(B, T).reshape(N),
                batch["target_rot"].reshape(N, 3, 3) if "target_rot" in batch else None,
            )
            logs["reproj"] = cfg["w_reproj"] * L.reprojection_loss(
                uv_pred.reshape(B, T, self.model.V, -1, 2), pts2d_c, vis_c, valid, self.model.image_size
            )
        # The geometric losses only act on "informative" joints (both rays valid, ray angle / anchor distance
        # thresholds of model.geo_info_*); without thresholds ``geo_info`` is all-True and the masks below reduce to
        # the previous GT-visibility masks.
        infos = out.get("geo_info") if self.model.geo_info_enabled else None
        if "res2d" in out and cfg["w_res2d"] > 0:  # P2: supervise the per-view 2D residual points of every layer
            res = 0.0
            for li, uv in enumerate(out["res2d"]):
                m2d = vis_c & infos[li][:, :, None, :] if infos is not None else vis_c
                res = res + L.reprojection_loss(uv, pts2d_c, m2d, valid, self.model.image_size)
            logs["res2d"] = cfg["w_res2d"] * res / len(out["res2d"])
        if "geo_pred" in out and cfg["w_geo"] > 0:  # P2: the triangulated pose itself, where >= 2 views see the joint
            both = vis_c.sum(dim=2) >= 2  # [B, T, J]
            geo = sum(L.mpjpe_joint_masked(p, gt_canon, both & infos[li] if infos is not None else both, valid)
                      for li, p in enumerate(out["geo_pred"]))
            logs["geo"] = cfg["w_geo"] * geo / len(out["geo_pred"])
        if "vis_logits" in out and cfg["w_vis"] > 0:  # visibility head: supervise where the anchor projects inside the view
            lbl_known = torch.isfinite(pts2d_c).all(-1)           # [B, T, V, J]
            target_vis = vis_c
            if cfg.get("vis_pseudo_from_heatmap", False) and "heatmaps" in out:
                # Datasets without occlusion labels (UnrealEgo-RW) - a joint inside the FOV counts as
                # occluded when the (detached) heatmap has no response at its GT 2D location
                target_vis = vis_c & self._heatmap_visible(out["heatmaps"], pts2d_c, cfg.get("vis_pseudo_thresh", 0.3))
            # every layer has its own head and every layer's logits scale the intersection weights: supervise all of
            # them (only the last one was trained before, the others stayed at their init)
            fovs = [f for _, f in out["anchors_2d"]][-len(out["vis_logits"]):]
            vis_loss = sum(L.visibility_loss(lg, target_vis, fov & lbl_known & valid[:, :, None, None])
                           for lg, fov in zip(out["vis_logits"], fovs))
            logs["vis"] = cfg["w_vis"] * vis_loss / len(out["vis_logits"])
        if cfg["w_bone"] > 0:
            logs["bone"] = cfg["w_bone"] * L.bone_loss(out["pred"], gt_canon, valid, cfg["bone_sym"],
                                                       self.model.skeleton.bones, self.model.skeleton.symmetric_bone_pairs)
        if "raydn_logits" in out and cfg.get("w_raydn", 0.0) > 0:  # along-ray hard negatives, every layer
            rd = sum(L.raydn_loss(lg, vj, valid) for lg, vj in zip(out["raydn_logits"], out["raydn_valid"]))
            logs["raydn"] = cfg["w_raydn"] * rd / len(out["raydn_logits"])
        if "bone_len" in out and cfg.get("w_bone_len", 0.0) > 0:  # P5: per-frame bone-length head against the GT lengths
            logs["bone_len"] = cfg["w_bone_len"] * L.bone_length_loss(out["bone_len"], gt_canon, self.model.bones, valid)
            with torch.no_grad():
                self.log("train/bone_len_mem_err", L.bone_length_loss(out["bone_mem"], gt_canon, self.model.bones, valid), on_step=True, on_epoch=False)
        if cfg["w_vel"] > 0 and T > 1:
            logs["vel"] = cfg["w_vel"] * L.velocity_loss(pred, gt, valid)
        if cfg["w_jerk"] > 0 and T > 3:
            logs["jerk"] = cfg["w_jerk"] * L.jerk_loss(out["pred"], valid)
        # Self-supervision on the frames of unlabeled (UDA target) sources
        if unl is not None and bool(unl.any()) and any(cfg.get(k, 0.0) > 0 for k in ("w_ssl_geo", "w_ssl_reproj", "w_ssl_temporal", "w_ssl_bone")):
            unl_frames = batch["valid_frames"] & unl.to(valid.device)[:, None].expand_as(valid)
            logs.update(self._ssl_losses(out, batch, unl_frames, rig_ids))
        if "dn_refined" in out and cfg["w_dn"] > 0:
            dn = img.new_zeros(())
            dn_valid = valid & out["dn_valid"] if "dn_valid" in out else valid   # frames with a complete GT pose
            for group, target in zip(out["dn_refined"], out["dn_targets"]):
                for p in group:
                    dn = dn + L.mpjpe_loss(p, target, dn_valid)
            logs["dn"] = cfg["w_dn"] * dn / max(1, len(out["dn_refined"]) * len(out["dn_refined"][0]))

        for k, v in logs.items():
            total = total + v
            self.log(f"train/{k}", v, on_step=True, on_epoch=False)
        self.log("train/total", total, prog_bar=True)
        with torch.no_grad():
            self.log("train/mpjpe_mm", L.mpjpe_loss(pred, gt, valid) * self.model.to_mm, prog_bar=True)
            if "gates" in out:
                self.log("train/geo_gate", out["gates"][-1].float().mean(), on_step=True, on_epoch=False)
        return total

    def _ssl_losses(self, out: Dict, batch: Dict, unl_valid: torch.Tensor, rig_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Label-free losses on unlabeled frames (``unl_valid`` [B, T]): geometric pseudo-labels from the two-ray
        intersection, stereo re-projection onto the confident heatmap peaks, temporal smoothness and bone constancy."""
        from ..data.heatmaps import heatmap_softargmax

        cfg = self.loss_cfg
        pred = out["pred"]
        B, T = pred.shape[:2]
        N = B * T
        logs: Dict[str, torch.Tensor] = {}
        thr = cfg.get("ssl_conf_thresh", 0.5)
        if cfg.get("w_ssl_geo", 0.0) > 0 and "geo_pred" in out:
            info = out["geo_info"][-1] if "geo_info" in out else torch.ones_like(pred[..., 0], dtype=torch.bool)
            vconf = out["view_conf"][-1].amin(dim=2)                              # [B, T, J] min over views
            weight = info.float() * (vconf.detach() >= thr).float() * vconf.detach()
            logs["ssl_geo"] = cfg["w_ssl_geo"] * L.pseudo_geo_loss(pred, out["geo_pred"][-1], weight, unl_valid)
        if cfg.get("w_ssl_reproj", 0.0) > 0 and "heatmaps" in out:
            ids = self.model.heatmap_joint_ids
            uv_peak, peak = heatmap_softargmax(out["heatmaps"].detach().float().flatten(0, 1))   # [N, V, K, 2], [N, V, K]
            uv_pred, _ = self.model.project(
                pred.reshape(N, -1, 3)[:, ids], batch["origin_cam"].reshape(N, self.model.V, 1, 3),
                rig_ids[:, None].expand(B, T).reshape(N), batch["target_rot"].reshape(N, 3, 3) if "target_rot" in batch else None,
            )
            logs["ssl_reproj"] = cfg["w_ssl_reproj"] * L.heatmap_reprojection_loss(
                uv_pred.reshape(B, T, self.model.V, -1, 2), uv_peak.reshape(B, T, self.model.V, -1, 2),
                peak.reshape(B, T, self.model.V, -1), thr, unl_valid, self.model.image_size)
        if cfg.get("w_ssl_temporal", 0.0) > 0 and T > 3:
            logs["ssl_temporal"] = cfg["w_ssl_temporal"] * L.jerk_loss(pred, unl_valid)
        if cfg.get("w_ssl_bone", 0.0) > 0:
            logs["ssl_bone"] = cfg["w_ssl_bone"] * L.bone_consistency_loss(pred, unl_valid, self.model.skeleton.bones,
                                                                           self.model.skeleton.symmetric_bone_pairs)
        return logs

    def on_before_optimizer_step(self, optimizer) -> None:
        clip = self.optim_cfg.get("grad_clip")
        if clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)
        if self.ema is not None:
            self.ema.update_parameters(self.model)

    # ------------------------------------------------------------------ evaluation
    def _stream_sequences(self) -> bool:
        """data.eval_clip_len 0: consecutive clips of a sequence arrive one per batch; the streaming state is carried."""
        try:
            return bool(getattr(self.trainer.datamodule, "stream_sequences", False))
        except Exception:  # noqa: BLE001 - no trainer attached (unit tests)
            return False

    def _predict(self, batch: Dict, rig_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        model = self.eval_model()
        rig_ids = self._rig_ids(batch) if rig_ids is None else rig_ids
        kw = dict(
            target_rot=batch.get("target_rot"), aux_cond=batch.get("aux_cond") if model.use_aux_cond else None,
            skeleton_ids=self._skeleton_ids(batch) if model.joint_mapper is not None else None,
        )
        # paired ablations differ by 0.2-0.5 mm while bf16 inference moves single joints by up to ~1.5 cm
        fp32 = torch.autocast(device_type=batch["img"].device.type, enabled=False) if self.eval_precision == "fp32" else nullcontext()
        model.diag_probe = self.diag_probe
        try:
            with fp32:
                if self._stream_sequences():
                    state = self._continued_stream_state(batch)
                    out, state = model.forward_streaming(batch["img"], batch["origin_cam"], rig_ids, batch["dataset_id"], state=state, **kw)
                    self._remember_stream_state(batch, state)
                elif self.eval_streaming or model.track_prior:
                    out, _ = model.forward_streaming(batch["img"], batch["origin_cam"], rig_ids, batch["dataset_id"], **kw)
                else:
                    out = model(batch["img"], batch["origin_cam"], rig_ids, batch["dataset_id"], **kw)
        finally:
            model.diag_probe = None
        return out

    def _continued_stream_state(self, batch: Dict) -> Optional[Dict]:
        """Streaming state of the previous chunk if (and only if) this chunk continues the same sequence at the next
        frame (the state used to be reused for any chunk of the same sequence, so a skipped or reordered
        chunk was silently treated as the immediate history)."""
        key = batch["seq_id"][0]
        if key != self._stream_key or self._stream_state is None:
            return None
        fids = batch.get("frame_ids")
        if fids is None or self._stream_next_fid is None:
            return self._stream_state
        first = int(fids[0, 0])
        if first == self._stream_next_fid:
            return self._stream_state
        self._stream_breaks += 1
        if self._stream_breaks <= 3:
            print(f"[EgoRayFormer] streaming: chunk of {key} starts at frame {first}, expected {self._stream_next_fid} - state reset")
        return None

    def _remember_stream_state(self, batch: Dict, state: Dict) -> None:
        self._stream_key, self._stream_state = batch["seq_id"][0], state
        fids = batch.get("frame_ids")
        if fids is None:
            self._stream_next_fid = None
            return
        vf = batch["valid_frames"][0]
        f = fids[0][vf] if bool(vf.any()) else fids[0]
        stride = int(f[1] - f[0]) if f.numel() > 1 else 1
        self._stream_next_fid = int(f[-1]) + max(stride, 1)

    @staticmethod
    def _seq_parts(seq_id: str) -> List[str]:
        s = str(seq_id).strip("/")
        return [p for p in (s.split("/") if "/" in s else s.split("__")) if p]

    @classmethod
    def _session_of(cls, seq_id: str) -> str:
        """Recording session of a sequence id: the capture date of UnrealEgo-RW ("studio/2023_01_17/S1/seq_1-0-2" or the
        directory form "studio__2023_01_17__S1__seq_1-0-2" -> "2023_01_17"), the environment otherwise (UnrealEgo2
        "ArchVisInterior_ArchVis_RT/Day/..." -> "ArchVisInterior_ArchVis_RT")."""
        import re

        parts = cls._seq_parts(seq_id)
        if len(parts) >= 3 and re.fullmatch(r"\d{4}_\d{2}_\d{2}", parts[1]):
            return parts[1]
        return parts[0] if parts else ""

    @classmethod
    def _subject_of(cls, seq_id: str) -> str:
        """Subject of an UnrealEgo-RW sequence ("studio/2023_01_17/S1/seq_1-0-2" -> "S1"); "" when the id has none."""
        import re

        parts = cls._seq_parts(seq_id)
        if len(parts) >= 4 and re.fullmatch(r"\d{4}_\d{2}_\d{2}", parts[1]) and re.fullmatch(r"S\d+", parts[2]):
            return parts[2]
        return ""

    def _eval_step(self, batch: Dict, prefix: str) -> None:
        if self.model.fov_circle:
            batch = self._apply_fov_circle(batch, self._rig_ids(batch))
        if self.stage == "heatmap":
            B, T = batch["img"].shape[:2]
            rig_bt = self._rig_ids(batch)[:, None].expand(B, T).reshape(-1)
            _, hm = self.eval_model().encode(batch["img"].flatten(0, 1), rig_bt)
            hm = hm.reshape(B, T, *hm.shape[1:])
            loss = L.heatmap_loss(hm, self._heatmap_targets(batch, hm), batch["valid_frames"], self._heatmap_annotated(batch))
            self.log(f"{prefix}/heatmap", loss, sync_dist=True, on_epoch=True)
            return
        if self.eval_rig_rot_deg > 0:  # evaluation with every camera rotated by a known angle
            rig0 = self._rig_ids(batch)
            dR = self._fixed_rig_rotation(list(batch["seq_id"]), batch["img"].shape[2], batch["img"].device, self.eval_rig_rot_deg)
            batch_in, geometry = self._augment_rig(batch, rig0, dR, self.eval_rig_mode)
            if geometry:
                with self.eval_model().rig_augmentation(rig0, dR, check_source=self.eval_rig_mode == "consistent") as virt:
                    out = self._predict(batch_in, virt)
            else:
                out = self._predict(batch_in, rig0)
        else:
            out = self._predict(batch)
        pred = self._pred_for_gt(out, "pred", batch)
        tags = [f"{ds}" for ds in self._dataset_names(batch)]
        skels = list(batch["skeleton"]) if "skeleton" in batch else None
        canon = pred.shape[-2] == self.model.J
        unc = out.get("uncertainty_chol") if canon else None  # canonical joints only
        # analysis inputs: GT distance from the device centre, heatmap confidence, last-layer gate
        gt = batch["pose"]
        B, T = gt.shape[:2]
        model = self.eval_model()
        with torch.no_grad():
            center = model.center(batch["origin_cam"].reshape(B * T, model.V, 1, 3), self._rig_ids(batch)[:, None].expand(B, T).reshape(-1),
                                  batch["target_rot"].reshape(B * T, 3, 3) if "target_rot" in batch else None)
            rel = gt.reshape(B * T, -1, 3).float() - center
            dist = rel.norm(dim=-1).reshape(B, T, -1)                                          # [B, T, J]
            ray_dir = torch.nn.functional.normalize(rel, dim=-1).reshape(B, T, -1, 3)          # unit ray of the GT joint
            rot = batch.get("target_rot")                                                       # target -> device axes
            dev = rel if rot is None else torch.einsum("nij,nkj->nki", rot.reshape(B * T, 3, 3).float(), rel)
            dev = dev.reshape(B, T, -1, 3)                                                      # GT in the device frame
        gate = out.get("gate") if canon else None
        if gate is None and canon and isinstance(out.get("gates"), list):    # parallel forward: per-layer lists
            gate = out["gates"][-1]
        geo = out.get("geo_pred_last") if canon else None
        if geo is None and canon and isinstance(out.get("geo_pred"), list):
            geo = out["geo_pred"][-1]
        info = out.get("geo_info_last") if canon else None
        if info is None and canon and isinstance(out.get("geo_info"), list):
            info = out["geo_info"][-1]
        seq_ids = list(batch["seq_id"]) if "seq_id" in batch else None
        labels = ({"session": [self._session_of(s) for s in seq_ids], "subject": [self._subject_of(s) for s in seq_ids]}
                  if seq_ids is not None else None)
        self.metrics.update(pred, gt, batch["valid_frames"], tags, vis=batch.get("pts2d_valid"),
                            pts2d=batch.get("pts2d"), unc_chol=unc, skeletons=skels, dist_cm=dist,
                            conf=out.get("hm_conf") if canon else None, gate=gate, geo_pred_cm=geo, geo_info=info,
                            ray_dir=ray_dir if canon else None, local_std_cm=out.get("uncertainty_local_std") if canon else None,
                            image_size_px=batch["orig_size"][:, 0] if "orig_size" in batch else None, dev_pose_cm=dev,
                            seq_ids=seq_ids, frame_ids=batch.get("frame_ids"), labels=labels)
        if self.gt_audit:
            self._update_audit(batch)
        if self.diag_dump and canon:
            self._collect_diag(batch, out, pred, gt, prefix)
        if prefix == "test" and self.export_dir:
            self._export(batch, pred)

    def _update_audit(self, batch: Dict) -> None:
        """Accumulate the GT-free input audit of the evaluated batches (eval/gt_audit.py)."""
        from ..eval.gt_audit import audit_batch, merge_audits

        a = audit_batch(batch, self._target_frame(), self.eval_model(), self._rig_ids(batch))
        self._audit = a if self._audit is None else merge_audits(self._audit, a)

    def _collect_diag(self, batch: Dict, out: Dict, pred: torch.Tensor, gt: torch.Tensor, prefix: str) -> None:
        """Keep the per-joint last-layer candidates of the valid frames for the error / gate analysis."""
        keys = {"pred": pred, "gt": gt, "geo_raw": out.get("geo_pred_last"), "geo_cand": out.get("geo_cand_last"),
                "learn": out.get("learn_pred_last"), "gate": out.get("gate"), "anchor": out.get("anchor_last"),
                "center": out.get("center_last"), "n_valid": out.get("n_valid_last"), "ray_cos": out.get("ray_cos_last"),
                "conf_min": out.get("conf_min_last"), "eig": out.get("geo_eig_last"), "disp_rate": out.get("disp_rate_last"),
                "hyp_min": out.get("hyp_min_last"), "hyp_max": out.get("hyp_max_last")}
        vf = batch["valid_frames"]
        row: Dict[str, Any] = {k: v[vf].detach().float().cpu().numpy() for k, v in keys.items() if torch.is_tensor(v)}
        B, T = vf.shape
        row["seq_id"] = [s for b, s in enumerate(batch["seq_id"]) for _ in range(int(vf[b].sum()))]
        row["frame_id"] = batch["frame_ids"][vf].cpu().numpy()
        row["split"] = prefix
        if self.diag_probe is not None:
            row.update(self._probe_rows(batch, out, gt, vf))
        self._diag_rows.append(row)

    _PROBE_KEYS = ("probe_cos_last", "probe_hm_last", "probe_fov_last", "probe_axes_last", "cand_points_last", "cand_uv_last",
                   "cand_fov_last")

    def _probe_rows(self, batch: Dict, out: Dict, gt: torch.Tensor, vf: torch.Tensor) -> Dict[str, Any]:
        """Per valid frame: the probe of the last layer (scores on the lines through the anchor, which views see each
        point, the line axes) and the distances of the GT joint to the layer's nearest sampled hypothesis - 3D over all
        hypotheses and over those inside every view, and 2D (input pixels) per view over the hypotheses that view sees."""
        missing = [k for k in self._PROBE_KEYS if k not in out]
        if missing:
            raise RuntimeError(f"model.diag_probe_extent_cm: the forward returned no {missing} - the probe runs in the streamed "
                               "forward (data.eval_clip_len 0 or a track-prior model) of a ray-consistent model")
        B, T = vf.shape
        pts, fov_c = out["cand_points_last"], out["cand_fov_last"]                  # [B, T, J, M, 3], [B, T, V, J, M]
        J, V = pts.shape[2], fov_c.shape[2]
        g = gt[..., :J, :].float()
        inside_all = fov_c.all(dim=2)                                                # [B, T, J, M]
        d3 = (pts - g[..., None, :]).norm(dim=-1)

        def nearest(d: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
            m = torch.where(ok, d, torch.full_like(d, float("inf"))).amin(-1)
            return torch.where(torch.isinf(m), torch.full_like(m, float("nan")), m)

        rot = batch.get("target_rot")
        with torch.no_grad():
            uv_gt, fov_gt = self.eval_model().project(
                g.reshape(B * T, J, 3), batch["origin_cam"].reshape(B * T, V, 1, 3).float(),
                self._rig_ids(batch)[:, None].expand(B, T).reshape(-1), rot.reshape(B * T, 3, 3).float() if rot is not None else None)
        d2 = (out["cand_uv_last"] - uv_gt.reshape(B, T, V, J, 1, 2)).norm(dim=-1) * float(batch["img"].shape[-1])
        sel = lambda t: t[vf].detach().cpu().numpy()  # noqa: E731
        return {"probe_cos": sel(out["probe_cos_last"].float()), "probe_hm": sel(out["probe_hm_last"].float()),
                "probe_fov": sel(out["probe_fov_last"]), "probe_axes": sel(out["probe_axes_last"].float()),
                "cand_d3_full": sel(nearest(d3, torch.ones_like(inside_all))), "cand_d3_valid": sel(nearest(d3, inside_all)),
                "cand_n_valid": sel(inside_all.sum(-1)), "cand_d2_view": sel(nearest(d2, fov_c)),
                "gt_fov_view": sel(fov_gt.reshape(B, T, V, J))}

    def _dataset_names(self, batch: Dict) -> List[str]:
        # seq_id encodes the source only indirectly; use the rig + dataset id registry
        from ..data.stereo_video import DATASET_IDS

        inv = {v: k for k, v in DATASET_IDS.items()}
        return [inv.get(int(i), f"ds{int(i)}") for i in batch["dataset_id"]]

    def _target_frame(self) -> str:
        """Training target frame of the attached datamodule (``device`` when no trainer / datamodule is attached)."""
        try:
            return str(getattr(self.trainer.datamodule, "target_frame", "device"))
        except Exception:  # noqa: BLE001 - no trainer attached (unit tests)
            return str(getattr(self, "_target_frame_override", "device"))

    def _to_device_frame(self, batch: Dict, pred: torch.Tensor) -> torch.Tensor:
        """Predictions in the training target frame -> device frame (the leaderboard convention), [B, T, J, 3] cm:
        p_dev = R_target p (gravity protocol) + root_dev (root / pelvis protocols). Exporting root-relative
        coordinates used to go unnoticed."""
        p = pred.detach().float()
        rot = batch.get("target_rot")                                  # [B, T, 3, 3] target -> device (identity if unused)
        if rot is not None:
            p = torch.einsum("btij,btkj->btki", rot.float(), p)
        frame = self._target_frame()
        if frame in ("root", "pelvis"):
            root = batch.get("root_dev")
            vf = batch["valid_frames"]
            if root is None or not bool(torch.isfinite(root[vf]).all()):
                raise RuntimeError(f"export in the {frame} protocol needs a finite device-frame root for every valid frame")
            p = p + root.float()[:, :, None, :]
        elif frame not in ("device", "gravity"):
            raise RuntimeError(f"export does not know how to map target_frame={frame!r} back to the device frame")
        return p

    def _export(self, batch: Dict, pred: torch.Tensor) -> None:
        pred = self._to_device_frame(batch, pred).cpu().numpy()
        for b, seq_id in enumerate(batch["seq_id"]):
            vf = batch["valid_frames"][b].cpu().numpy()
            fids = batch["frame_ids"][b].cpu().numpy()[vf]
            write_sequence_predictions(self.export_dir, seq_id, fids, pred[b][vf], self.export_name_style)

    def _eval_epoch_start(self) -> None:
        self._stream_key, self._stream_state, self._stream_next_fid, self._stream_breaks = None, None, None, 0
        self._diag_rows, self._audit = [], None
        if self._stream_sequences() and self._world_size() > 1:
            # chunks of one sequence would be interleaved over the ranks (DistributedSampler), so no rank sees the
            # history it carries; the per-rank means were also averaged instead of pooled
            raise RuntimeError("continuous streaming evaluation (data.eval_clip_len 0) needs a single device: "
                               "run validate / test with --trainer.devices 1")
        if self.diag_dump and self._world_size() > 1:
            # rank 0 would write the samples of its own shard next to metrics pooled over every rank
            raise RuntimeError("model.diag_dump needs a single device: run validate with --trainer.devices 1")

    def _world_size(self) -> int:
        try:
            return int(self.trainer.world_size)
        except Exception:  # noqa: BLE001 - no trainer attached (unit tests)
            return 1

    def on_validation_epoch_start(self) -> None:
        self._eval_epoch_start()

    def on_test_epoch_start(self) -> None:
        self._eval_epoch_start()

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        self._eval_step(batch, "val")

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        self._eval_step(batch, "test")

    def _all_gather(self, obj: Any) -> List[Any]:
        """``obj`` of every rank (a one-element list on a single process)."""
        if self._world_size() <= 1 or not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return [obj]
        out: List[Any] = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(out, obj)
        return out

    def _gather_metrics(self) -> None:
        """Pool the per-frame errors of all ranks before computing the metrics (averaging per-rank means is
        not the frame-weighted mean; DistributedSampler duplicates are dropped by (sequence, frame id))."""
        states = self._all_gather(self.metrics.export_state())
        if len(states) > 1:
            self.metrics.load_states(states)

    def _gather_audit(self) -> None:
        """GT-free audit over every rank: a GT-derived input seen by any rank marks the whole evaluation,
        whichever rank writes protocol.json."""
        from ..eval.gt_audit import merge_audits

        audits = [a for a in self._all_gather(self._audit) if a is not None]
        merged = None
        for a in audits:
            merged = a if merged is None else merge_audits(merged, a)
        self._audit = merged

    def _selection_metric(self, res: Dict[str, float], prefix: str) -> None:
        """``val/mpjpe``, the checkpoint-selection metric. With ``select_val_dataset`` it is that
        dataset's MPJPE - an error when that dataset was not evaluated or its MPJPE is not finite (it used to fall back
        to another metric, or keep the alias of whatever single dataset was evaluated). Unset: the frame-pooled MPJPE of
        a multi-dataset validation (single dataset: the alias); without a pooled value (different skeletons) no
        selection metric is logged - a checkpoint monitor then fails loudly instead of choosing a dataset."""
        if prefix != "val":
            return
        key = "val/mpjpe"
        datasets = sorted(k.split("/")[1] for k in res if k.endswith("/mpjpe") and k.count("/") == 2 and k != "val/all/mpjpe")
        if self.select_val_dataset:
            src = f"val/{self.select_val_dataset}/mpjpe"
            v = res.get(src)
            if v is None or not math.isfinite(float(v)):
                raise RuntimeError(f"model.select_val_dataset={self.select_val_dataset!r}: {src} is "
                                   f"{'missing' if v is None else v} (validation sets evaluated: {datasets})")
            res[key] = float(v)
            return
        if key in res:
            return
        if "val/all/mpjpe" in res:
            res[key] = res["val/all/mpjpe"]
        elif len(datasets) > 1:
            print(f"[EgoRayFormer] val/mpjpe not logged: validation sets {datasets} without a pooled MPJPE - set "
                  "model.select_val_dataset to select checkpoints")

    def _write_seq_errors(self, prefix: str) -> None:
        import csv

        rows = self.metrics.per_sequence()
        if not rows:
            return
        path = f"{self.seq_error_dump}_{prefix}.csv"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"[EgoRayFormer] per-sequence errors: {path} ({len(rows)} sequences)")

    def _write_diag(self, prefix: str) -> None:
        import numpy as np

        rows = [r for r in self._diag_rows if r.get("split") == prefix]
        if not rows:
            return
        keys = [k for k in rows[0] if k not in ("split",)]
        data = {}
        for k in keys:
            vals = [r[k] for r in rows if k in r]
            if len(vals) != len(rows):
                continue
            data[k] = np.concatenate([np.asarray(v) for v in vals], 0)
        if self.diag_probe is not None:
            data["probe_extent_step_cm"] = np.asarray(self.diag_probe, dtype=np.float64)
        path = f"{self.diag_dump}_{prefix}.npz"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(path, **data)
        print(f"[EgoRayFormer] diagnostic dump: {path} ({data['pred'].shape[0]} frames, keys {sorted(data)})")

    def _epoch_end(self, prefix: str) -> None:
        if self.stage == "heatmap":
            return
        self._gather_metrics()
        if self.gt_audit:
            self._gather_audit()
        res = self.metrics.compute(prefix=f"{prefix}/")
        self._selection_metric(res, prefix)
        if self._stream_breaks:
            res[f"{prefix}/stream_state_resets"] = float(self._stream_breaks)
        if self._audit is not None:
            res[f"{prefix}/audit/gt_free"] = 1.0 if self._audit.get("gt_free") else 0.0
        if self.global_rank == 0:
            if self.calib_dump:
                raw = self.metrics.raw_calibration()
                if raw:
                    import numpy as np

                    os.makedirs(os.path.dirname(os.path.abspath(self.calib_dump)), exist_ok=True)
                    np.savez(f"{self.calib_dump}_{prefix}.npz", nu=self.metrics.nll_nu, **raw)
                    print(f"[EgoRayFormer] calibration dump: {self.calib_dump}_{prefix}.npz ({raw['maha'].size} joint samples)")
            if self.seq_error_dump:
                self._write_seq_errors(prefix)
            if self.diag_dump:
                self._write_diag(prefix)
            if self._audit is not None:
                self._write_audit(prefix)
        self.metrics.reset()
        self._diag_rows = []
        # every rank holds the pooled metrics now (or is the only rank): log them as they are
        for k, v in res.items():
            self.log(k, torch.tensor(v, device=self.device), sync_dist=False, rank_zero_only=False,
                     prog_bar=k.endswith("/mpjpe") and "/all/" not in k)

    def _write_audit(self, prefix: str) -> None:
        import json

        from ..eval.gt_audit import format_audit

        a = dict(self._audit, split=prefix)
        print(f"[EgoRayFormer] GT-free input audit ({prefix}): {format_audit(a)}")
        targets = []
        if self.export_dir and prefix == "test":
            targets.append(os.path.join(self.export_dir, "protocol.json"))
        if self.seq_error_dump:
            targets.append(f"{self.seq_error_dump}_{prefix}_audit.json")
        for path in targets:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w") as f:
                json.dump(a, f, indent=1)

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    # ------------------------------------------------------------------ optimisation
    def configure_optimizers(self):
        cfg = self.optim_cfg
        lr, wd = cfg["lr"], cfg["weight_decay"]
        enc_params = set(id(p) for p in self.model.encoder.parameters())
        groups = self.model.encoder.param_groups(lr, wd, layer_decay=cfg["layer_decay"], lr_scale=cfg["encoder_lr_scale"])
        decay, no_decay, extr = [], [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad or id(p) in enc_params:
                continue
            if n.endswith("extr_delta"):  # camera refinement (rad / cm) - small lr, no decay
                extr.append(p)
                continue
            nd = p.ndim <= 1 or n.endswith(".bias") or "token" in n or "embed" in n or "table" in n
            (no_decay if nd else decay).append(p)
        groups += [
            {"params": decay, "lr": lr, "weight_decay": wd, "name": "decoder"},
            {"params": no_decay, "lr": lr, "weight_decay": 0.0, "name": "decoder_no_decay"},
        ]
        if extr:
            groups.append({"params": extr, "lr": lr * 0.1, "weight_decay": 0.0, "name": "extrinsics"})
        for g in groups:
            g["base_lr"] = g["lr"]
        opt = torch.optim.AdamW(groups, lr=lr, betas=tuple(cfg["betas"]))

        warm = int(cfg["warmup_steps"])
        total = int(getattr(self.trainer, "estimated_stepping_batches", 0) or 0)
        steps_per_epoch = max(1, total // max(1, self.trainer.max_epochs or 1))
        milestones = milestone_steps(cfg["milestones"], steps_per_epoch)

        def lr_lambda(step: int) -> float:
            if step < warm:
                return float(step + 1) / float(warm)
            if cfg["schedule"] == "cosine":
                p = (step - warm) / max(1, total - warm)
                return cfg["min_lr_ratio"] + (1 - cfg["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
            return cfg["gamma"] ** sum(step >= m for m in milestones)

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

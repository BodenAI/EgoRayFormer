"""LightningDataModule for the unified stereo clip datasets."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler

from .records import DatasetIndex
from .skeleton import get_skeleton
from .stereo_video import StereoVideoDataset, collate_clips


class EgoRayFormerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        train: Sequence[Dict] = (),
        val: Sequence[Dict] = (),
        test: Sequence[Dict] = (),
        clip_len: int = 1,
        eval_clip_len: Optional[int] = None,  # None: clip_len; 0: continuous streaming over whole sequences (clips of
        eval_stream_chunk: int = 32,          # eval_stream_chunk frames, batch 1, the model state carried between them)
        frame_stride: int = 1,
        target_frame: str = "pelvis",
        img_size: int = 256,
        batch_size: int = 32,
        eval_batch_size: Optional[int] = None,
        num_workers: int = 8,
        augment: bool = True,
        weighted_sampling: bool = False,
        clips_per_epoch_scale: float = 1.0,
        pin_memory: bool = True,
        unc_holdout_frac: float = 0.0,
        rig_swap: Optional[Dict[str, str]] = None,   # evaluation only: {rig: replacement} (neighbouring-session calibration)
        # clips drawn per training epoch, with replacement (by the source weights under weighted_sampling); None: one
        # pass over the clip entries (shuffled) or len(entries) weighted draws, as before
        train_samples_per_epoch: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.root = root
        self.train_sources, self.val_sources, self.test_sources = list(train), list(val), list(test)
        self.clip_len = clip_len
        self.stream_sequences = eval_clip_len == 0
        if eval_clip_len is None:
            self.eval_clip_len = clip_len
        else:
            self.eval_clip_len = eval_stream_chunk if self.stream_sequences else eval_clip_len
        self.frame_stride = frame_stride
        self.target_frame = target_frame
        self.img_size = img_size
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size
        self.num_workers = num_workers
        self.augment = augment
        self.weighted_sampling = weighted_sampling
        self.clips_per_epoch_scale = clips_per_epoch_scale
        self.pin_memory = pin_memory
        self.unc_holdout_frac = unc_holdout_frac  # fraction of training sequences reserved for the uncertainty head
        self.rig_swap = dict(rig_swap or {})
        if train_samples_per_epoch is not None and int(train_samples_per_epoch) < batch_size:
            raise ValueError(f"data.train_samples_per_epoch={train_samples_per_epoch} is smaller than one batch ({batch_size})")
        self.train_samples_per_epoch = None if train_samples_per_epoch is None else int(train_samples_per_epoch)
        self.train_ds = self.val_ds = self.test_ds = None
        self._num_joints_pad: Optional[int] = None

    @property
    def num_joints_pad(self) -> int:
        """Joint padding shared by all loaders: the largest skeleton among every source (train / val / test)."""
        if self._num_joints_pad is None:
            skels = set()
            for s in self.train_sources + self.val_sources + self.test_sources:
                try:
                    skels.add(DatasetIndex(self.root, s["dataset"], s["split"]).skeleton)
                except FileNotFoundError:
                    pass
            self._num_joints_pad = max([get_skeleton(k).num_joints for k in skels] + [16])
        return self._num_joints_pad

    def _make(self, sources, mode, clip_len):
        if not sources:
            return None
        return StereoVideoDataset(
            root=self.root,
            sources=sources,
            clip_len=clip_len,
            frame_stride=self.frame_stride,
            mode=mode,
            target_frame=self.target_frame,
            img_size=self.img_size,
            augment=self.augment,
            clips_per_epoch_scale=self.clips_per_epoch_scale,
            num_joints_pad=self.num_joints_pad,
            unc_holdout_frac=self.unc_holdout_frac if mode == "train" else 0.0,
            rig_swap=self.rig_swap if mode == "eval" else None,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit") and self.train_ds is None:
            self.train_ds = self._make(self.train_sources, "train", self.clip_len)
        if stage in (None, "fit", "validate") and self.val_ds is None:
            self.val_ds = self._make(self.val_sources, "eval", self.eval_clip_len)
        if stage in (None, "test", "predict") and self.test_ds is None:
            if not self.test_sources and self.val_sources:
                import warnings

                warnings.warn("data.test is empty: the *validation* sources are evaluated under the test/ metric prefix "
                              f"({self.val_sources}) - set data.test for a real test-split number")
            self.test_ds = self._make(self.test_sources or self.val_sources, "eval", self.eval_clip_len)

    def train_dataloader(self) -> DataLoader:
        sampler = None
        shuffle = True
        n = self.train_samples_per_epoch
        if self.weighted_sampling and self.train_ds is not None:
            w = torch.tensor(self.train_ds.entry_weights, dtype=torch.double)
            sampler = WeightedRandomSampler(w, num_samples=len(self.train_ds) if n is None else n, replacement=True)
            shuffle = False
        elif n is not None and self.train_ds is not None:
            sampler = RandomSampler(self.train_ds, replacement=True, num_samples=n)
            shuffle = False
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=shuffle, sampler=sampler,
            num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=True,
            collate_fn=collate_clips, persistent_workers=self.num_workers > 0,
        )

    def _eval_loader(self, ds) -> DataLoader:
        # continuous streaming: consecutive clips of a sequence must arrive in order, one per batch
        return DataLoader(
            ds, batch_size=1 if self.stream_sequences else self.eval_batch_size, shuffle=False, num_workers=self.num_workers,
            pin_memory=self.pin_memory, drop_last=False, collate_fn=collate_clips,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return self._eval_loader(self.val_ds)

    def test_dataloader(self) -> DataLoader:
        return self._eval_loader(self.test_ds)

    def predict_dataloader(self) -> DataLoader:
        return self._eval_loader(self.test_ds)

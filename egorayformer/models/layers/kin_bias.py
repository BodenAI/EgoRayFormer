"""Learnable attention bias indexed by kinematic-tree distance (EgoTAP-style skeletal propagation prior)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ...data.skeleton import NUM_JOINTS, tree_distance_matrix


class KinematicAttentionBias(nn.Module):
    """Returns an additive bias [1, H, N, N] for tokens = [J joints, E extra tokens].

    ``dist`` is the [J, J] hop-distance matrix of the canonical skeleton (``Skeleton.tree_distance()``); when omitted
    the 16-joint UnrealEgo tree is used for J = 16 and a flat (uninformative) prior otherwise.
    """

    def __init__(self, num_heads: int, num_joints: int = NUM_JOINTS, num_extra_tokens: int = 1, max_dist: int = 6,
                 dist: Optional[np.ndarray] = None):
        super().__init__()
        self.num_heads = num_heads
        self.max_dist = max_dist
        n = num_joints + num_extra_tokens
        idx = torch.full((n, n), max_dist + 1, dtype=torch.long)  # joint <-> extra
        if dist is None:
            dist = tree_distance_matrix() if num_joints == NUM_JOINTS else np.full((num_joints, num_joints), max_dist, dtype=np.int64)
        dist_t = torch.as_tensor(np.asarray(dist), dtype=torch.long)
        assert dist_t.shape == (num_joints, num_joints), f"dist {tuple(dist_t.shape)} vs {num_joints} joints"
        idx[:num_joints, :num_joints] = dist_t.clamp(max=max_dist)
        idx[num_joints:, num_joints:] = max_dist + 2  # extra <-> extra
        self.register_buffer("index", idx)
        self.table = nn.Embedding(max_dist + 3, num_heads)
        nn.init.zeros_(self.table.weight)

    def forward(self) -> torch.Tensor:
        return self.table(self.index).permute(2, 0, 1)[None]  # [1, H, N, N]

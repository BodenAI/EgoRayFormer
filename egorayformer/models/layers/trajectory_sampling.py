"""Trajectory-aligned causal temporal sampling (P3 of the redesign).

TSR-Ego enriches the *pixel-aligned* feature maps with a causal temporal convolution, which assumes that a body
part stays at the same image location over time - violated as soon as the head rotates. Here the joint query of
frame t samples the stereo features of the past frames t-1 ... t-w around the joint's *own past anchors*
(the previous decoder stage's predictions at those frames), using the same 3D ray-local offsets as the current
frame. Every past frame is projected with its own camera origin, so the samples stay aligned with the joint even
under head motion. Point weights come from the query (per age and point), plus an age bias and a learned penalty
on the cached per-joint log-std of the past anchors (uncertain past evidence is down-weighted).

All past frames of a window are batched along the leading dimension (age-major, index = age * N + n) so that
one projection and one ``grid_sample`` call serve the whole window. ``build_past_parallel`` assembles the batch
from a clip processed in parallel (training); ``PastFrames.from_history`` builds it from the streaming state at
inference, where the value projections of past frames are cached per layer. Both paths give identical results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .ray_stereo_attn import RayStereoAttn

LOG_REF_STD = math.log(3.0)  # cm; log-std of a "normal" joint

ProjectWithOrigin = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]]
# (points [M, P, 3], origin [M, V, 1, 3], rig_ids [M], rot [M, 3, 3] | None) -> uv [M, V, P, 2], fov [M, V, P]


@dataclass
class PastFrames:
    """Past frames of one decoder stage, batched age-major: row a*N + n is frame t_n - (a+1) of query n."""

    ages: int
    valid: torch.Tensor      # [A*N] bool  - the past frame exists for this query
    anchor: torch.Tensor     # [A*N, J, 3] - previous-stage prediction at the past frame (detached)
    logstd: torch.Tensor     # [A*N, J]    - its per-joint mean log-std (detached)
    value: torch.Tensor      # [A*N, V, H, hd, h, w] - value-projected features of the past frame
    center: torch.Tensor     # [A*N, 1, 3] - device centre in the target frame at the past frame
    origin: torch.Tensor     # [A*N, V, 1, 3] - camera origins of the past frame
    rig_ids: torch.Tensor    # [A*N]
    rot: Optional[torch.Tensor] = None  # [A*N, 3, 3] target -> device rotation of the past frame (optional)

    @staticmethod
    def from_history(
        anchors: Sequence[torch.Tensor], logstds: Sequence[torch.Tensor], values: Sequence[torch.Tensor],
        centers: Sequence[torch.Tensor], origins: Sequence[torch.Tensor], rig_ids: torch.Tensor,
        rots: Optional[Sequence[Optional[torch.Tensor]]] = None,
    ) -> Optional["PastFrames"]:
        """Streaming: lists ordered newest first (age 1, 2, ...), each element batched over B."""
        A = min(len(anchors), len(values))
        if A == 0:
            return None
        B = rig_ids.shape[0]
        rot = None
        if rots is not None and len(rots) >= A and all(r is not None for r in rots[:A]):
            rot = torch.cat(list(rots[:A]), 0)
        return PastFrames(
            ages=A,
            valid=torch.ones(A * B, dtype=torch.bool, device=rig_ids.device),
            anchor=torch.cat([a.detach() for a in anchors[:A]], 0),
            logstd=torch.cat([l.detach() for l in logstds[:A]], 0),
            value=torch.cat(list(values[:A]), 0),
            center=torch.cat(list(centers[:A]), 0),
            origin=torch.cat(list(origins[:A]), 0),
            rig_ids=rig_ids.repeat(A),
            rot=rot,
        )


def past_indices(B: int, T: int, delta: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Frame index n = b*T + t -> source index of frame t-delta (clamped) and validity mask, both [B*T]."""
    t = torch.arange(T, device=device)
    src_t = (t - delta).clamp_min(0)
    valid = (t - delta >= 0)
    base = (torch.arange(B, device=device) * T)[:, None]
    return (base + src_t[None]).reshape(-1), valid[None].expand(B, T).reshape(-1)


def build_past_parallel(
    B: int, T: int, window: int, prev_anchor: torch.Tensor, prev_logstd: torch.Tensor, value: torch.Tensor,
    center: torch.Tensor, origin: torch.Tensor, rig_ids: torch.Tensor, rot: Optional[torch.Tensor] = None,
) -> Optional[PastFrames]:
    """All tensors are per frame with N = B*T rows (n = b*T + t)."""
    A = min(window, T - 1)
    if A <= 0:
        return None
    srcs, valids = zip(*[past_indices(B, T, delta, value.device) for delta in range(1, A + 1)])
    src, valid = torch.cat(srcs), torch.cat(valids)
    return PastFrames(
        ages=A, valid=valid,
        anchor=prev_anchor.detach().index_select(0, src),
        logstd=prev_logstd.detach().index_select(0, src),
        value=value.index_select(0, src),
        center=center.index_select(0, src),
        origin=origin.index_select(0, src),
        rig_ids=rig_ids.index_select(0, src),
        rot=rot.index_select(0, src) if rot is not None else None,
    )


class TrajectoryAttention(nn.Module):
    def __init__(self, embed_dims: int, num_heads: int, num_points: int, window: int, unc_scale_fixed: bool = False):
        """``unc_scale_fixed``: the penalty on the past anchors' log-std is a zero buffer instead of a learned scale,
        i.e. the past evidence is weighted without any uncertainty input (P4 as a pure side output)."""
        super().__init__()
        self.H, self.K, self.w = num_heads, num_points, window
        self.attn_w = nn.Linear(embed_dims, num_heads * window * num_points)
        self.age_bias = nn.Parameter(torch.zeros(window, num_heads))
        if unc_scale_fixed:
            self.register_buffer("unc_scale", torch.zeros(num_heads), persistent=False)
        else:
            self.unc_scale = nn.Parameter(torch.ones(num_heads))
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        nn.init.zeros_(self.attn_w.weight)
        nn.init.zeros_(self.attn_w.bias)
        nn.init.zeros_(self.out_proj.weight)   # identity start: the module adds nothing until it learns to
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, query: torch.Tensor, delta_local: torch.Tensor, attn: RayStereoAttn, past: Optional[PastFrames],
                project: ProjectWithOrigin, depth_unit: Optional[torch.Tensor] = None) -> torch.Tensor:
        """query [N, J, C]; delta_local [N, J, H, K, 3] (from the current-frame RayStereoAttn); ``depth_unit`` [N, J] cm per
        disparity step of the current frame (the same ray-local hypotheses are reused at the past anchors) -> [N, J, C]."""
        N, J, C = query.shape
        if past is None:
            return query.new_zeros(N, J, C)
        A, H, K = past.ages, self.H, self.K
        du = depth_unit.repeat(A, 1) if depth_unit is not None else None
        points = attn.to_world(delta_local.repeat(A, 1, 1, 1, 1), past.anchor, past.center, du)  # [A*N, J, H, K, 3]
        uv, fov = project(points.reshape(A * N, J * H * K, 3), past.origin, past.rig_ids, past.rot)
        uv = uv.reshape(A * N, attn.V, J, H, K, 2)
        fov = fov.reshape(A * N, attn.V, J, H, K)
        fused = attn.fuse_views(attn.sample(past.value, uv), fov)                                 # [A*N, J, K, H, hd]
        fused = fused.view(A, N, J, K, H, -1).permute(1, 2, 4, 0, 3, 5)                          # [N, J, H, A, K, hd]
        logits = self.attn_w(query).view(N, J, H, self.w, K).float()[:, :, :, :A]                # [N, J, H, A, K]
        logits = logits + self.age_bias[:A].t()[None, None, :, :, None]
        unc = (LOG_REF_STD - past.logstd.float()).view(A, N, J).permute(1, 2, 0)                 # [N, J, A]
        logits = logits + self.unc_scale[None, None, :, None, None] * unc[:, :, None, :, None]
        valid = past.valid.view(A, N).t()                                                        # [N, A]
        logits = logits.masked_fill(~valid[:, None, None, :, None], -1e4)
        wts = torch.softmax(logits.flatten(3), dim=-1).view_as(logits)
        agg = (fused * wts[..., None]).sum(dim=(3, 4))                                           # [N, J, H, hd]
        out = self.out_proj(agg.reshape(N, J, C).to(query.dtype))
        # frames without any past evidence contribute exactly nothing (also matches the streaming path at t = 0)
        return out * valid.any(dim=1)[:, None, None].to(out.dtype)

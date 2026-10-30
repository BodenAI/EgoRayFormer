"""16-joint body skeleton shared by UnrealEgo / UnrealEgo2 / UnrealEgo-RW.

Joint order follows UnrealEgo (`utils/loss.py::list_joints`) so that predictions are directly
comparable with EgoPoseFormer v1 and the UnrealEgo2 / UnrealEgo-RW leaderboards. The pelvis is
*not* one of the 16 joints: in the pelvis-relative protocol it is the origin, in the
device-relative protocol it is simply absent. For kinematic reasoning we add a virtual pelvis
node (index 16) connected to the neck and both thighs.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

JOINT_NAMES: List[str] = [
    "head",        # 0
    "neck_01",     # 1
    "upperarm_l",  # 2
    "upperarm_r",  # 3
    "lowerarm_l",  # 4
    "lowerarm_r",  # 5
    "hand_l",      # 6
    "hand_r",      # 7
    "thigh_l",     # 8
    "thigh_r",     # 9
    "calf_l",      # 10
    "calf_r",      # 11
    "foot_l",      # 12
    "foot_r",      # 13
    "ball_l",      # 14
    "ball_r",      # 15
]
NUM_JOINTS = len(JOINT_NAMES)
VIRTUAL_PELVIS = NUM_JOINTS  # index 16 in the extended tree

# UnrealEgo heatmaps have 15 channels: all joints except the head (same order otherwise).
HEATMAP_JOINT_IDS: List[int] = list(range(1, NUM_JOINTS))
NUM_HEATMAP_JOINTS = len(HEATMAP_JOINT_IDS)

# Candidate key names in the raw json files, in priority order (the UnrealEgo2 website spells
# thigh as "thigt" in the leaderboard description; accept both).
UNREALEGO_JSON_NAMES: Dict[str, Sequence[str]] = {
    "head": ("head",),
    "neck_01": ("neck_01",),
    "upperarm_l": ("upperarm_l",),
    "upperarm_r": ("upperarm_r",),
    "lowerarm_l": ("lowerarm_l",),
    "lowerarm_r": ("lowerarm_r",),
    "hand_l": ("hand_l",),
    "hand_r": ("hand_r",),
    "thigh_l": ("thigh_l", "thigt_l"),
    "thigh_r": ("thigh_r", "thigt_r"),
    "calf_l": ("calf_l",),
    "calf_r": ("calf_r",),
    "foot_l": ("foot_l",),
    "foot_r": ("foot_r",),
    "ball_l": ("ball_l",),
    "ball_r": ("ball_r",),
}

# UnrealEgo-RW uses a mocap-studio skeleton with different names (leaderboard json format).
UNREALEGO_RW_JSON_NAMES: Dict[str, Sequence[str]] = {
    "head": ("Head",),
    "neck_01": ("Neck",),
    "upperarm_l": ("LeftArm",),
    "upperarm_r": ("RightArm",),
    "lowerarm_l": ("LeftForeArm",),
    "lowerarm_r": ("RightForeArm",),
    "hand_l": ("LeftHand",),
    "hand_r": ("RightHand",),
    "thigh_l": ("LeftUpLeg",),
    "thigh_r": ("RightUpLeg",),
    "calf_l": ("LeftLeg",),
    "calf_r": ("RightLeg",),
    "foot_l": ("LeftFoot",),
    "foot_r": ("RightFoot",),
    "ball_l": ("LeftToeBase",),
    "ball_r": ("RightToeBase",),
}

# Parent of each joint in the extended tree (virtual pelvis = 16 is the root).
PARENTS_EXT: List[int] = [
    1,               # head -> neck
    VIRTUAL_PELVIS,  # neck -> pelvis (spine)
    1, 1,            # upper arms -> neck
    2, 3,            # lower arms -> upper arms
    4, 5,            # hands -> lower arms
    VIRTUAL_PELVIS, VIRTUAL_PELVIS,  # thighs -> pelvis
    8, 9,            # calves -> thighs
    10, 11,          # feet -> calves
    12, 13,          # balls -> feet
]

# Bones with both endpoints among the 16 predicted joints (used for bone-length losses).
BONES: List[Tuple[int, int]] = [
    (0, 1),
    (1, 2), (1, 3),
    (2, 4), (3, 5),
    (4, 6), (5, 7),
    (8, 10), (9, 11),
    (10, 12), (11, 13),
    (12, 14), (13, 15),
    (8, 9),   # hip width
    (2, 3),   # shoulder width
]
# Left/right bone pairs whose lengths should match.
SYMMETRIC_BONE_PAIRS: List[Tuple[int, int]] = [
    (BONES.index((1, 2)), BONES.index((1, 3))),
    (BONES.index((2, 4)), BONES.index((3, 5))),
    (BONES.index((4, 6)), BONES.index((5, 7))),
    (BONES.index((8, 10)), BONES.index((9, 11))),
    (BONES.index((10, 12)), BONES.index((11, 13))),
    (BONES.index((12, 14)), BONES.index((13, 15))),
]
LEFT_RIGHT_PAIRS: List[Tuple[int, int]] = [(2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]

BODY_PARTS: Dict[str, List[int]] = {
    "head": [0], "neck": [1], "upper_arm": [2, 3], "lower_arm": [4, 5], "hand": [6, 7],
    "thigh": [8, 9], "calf": [10, 11], "foot": [12, 13], "ball": [14, 15],
}


def tree_distance_matrix(include_virtual_pelvis: bool = False) -> np.ndarray:
    """Hop distance between joints along the kinematic tree.

    Returns an int64 matrix of shape [16, 16] (or [17, 17] when the virtual pelvis is included).
    """
    n = NUM_JOINTS + 1
    adj = np.zeros((n, n), dtype=bool)
    for j, p in enumerate(PARENTS_EXT):
        adj[j, p] = adj[p, j] = True
    dist = np.full((n, n), 10**6, dtype=np.int64)
    for s in range(n):
        dist[s, s] = 0
        frontier = [s]
        while frontier:
            nxt = []
            for u in frontier:
                for v in np.nonzero(adj[u])[0]:
                    if dist[s, v] > dist[s, u] + 1:
                        dist[s, v] = dist[s, u] + 1
                        nxt.append(int(v))
            frontier = nxt
    return dist if include_virtual_pelvis else dist[:NUM_JOINTS, :NUM_JOINTS]


def resolve_joint_dict(raw: Dict[str, Sequence[float]], name_table: Dict[str, Sequence[str]], joint_names: Sequence[str] = None) -> np.ndarray:
    """Pick the joints (in ``joint_names`` order, default the 16 UnrealEgo joints) out of a raw {name: [x, y(, z)]}
    dictionary -> [J, D]."""
    rows = []
    for name in (joint_names or JOINT_NAMES):
        for cand in name_table[name]:
            if cand in raw:
                rows.append(np.asarray(raw[cand], dtype=np.float32).reshape(-1))
                break
        else:
            raise KeyError(f"joint {name} not found; tried {name_table[name]}; available: {list(raw)[:10]}...")
    return np.stack(rows, axis=0)


# ======================================================================================================
# Skeleton registry: the canonical query skeleton of the model and the output skeletons of other datasets.
#
# * ``unrealego16``  - UnrealEgo / UnrealEgo2 / UnrealEgo-RW / Ego4View (16 joints, above)
# * ``mo2cap2_15``   - the 15-joint Mo2Cap2 / xR-EgoPose / SceneEgo ("EgoScene") skeleton used by FRAME (no head)
# * ``momentum26``   - the 26 Momentum body-model joints of EgoBody3M (EPFv2 protocol)
#
# ``JOINT_MAPPINGS[(src, dst)]`` gives, for every joint of ``dst``, the ``src`` joints (with weights) that
# initialise the linear JointMapper; joints without a geometric counterpart start from a plausible interpolation
# and are learned. 1:1 correspondences (weight 1.0 on a single source joint) are also used to transfer 2D
# supervision between skeletons.
# ======================================================================================================
from dataclasses import dataclass, field  # noqa: E402


@dataclass(frozen=True)
class Skeleton:
    name: str
    joint_names: Tuple[str, ...]
    parents: Tuple[int, ...]                 # parent index per joint; == num_joints -> virtual root, -1 -> none
    heatmap_joint_ids: Tuple[int, ...]       # joints with a 2D heatmap channel
    left_right_pairs: Tuple[Tuple[int, int], ...]
    root_joints: Tuple[int, ...]             # averaged for the translation-free (mid-hip) protocol
    upper_body: Tuple[int, ...]
    lower_body: Tuple[int, ...]
    body_parts: Dict[str, List[int]] = field(default_factory=dict)

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    def index(self, name: str) -> int:
        return self.joint_names.index(name)

    @property
    def bones(self) -> List[Tuple[int, int]]:
        """Parent-child pairs among real joints."""
        return [(j, p) for j, p in enumerate(self.parents) if 0 <= p < self.num_joints]

    @property
    def symmetric_bone_pairs(self) -> List[Tuple[int, int]]:
        bones = self.bones
        lr = dict(self.left_right_pairs)
        lr.update({r: l for l, r in self.left_right_pairs})
        out = []
        for i, (a, b) in enumerate(bones):
            mirrored = (lr.get(a, a), lr.get(b, b))
            if mirrored in bones and bones.index(mirrored) > i:
                out.append((i, bones.index(mirrored)))
        return out

    def tree_distance(self) -> np.ndarray:
        """Hop distance along the kinematic tree (virtual root included as a node but not returned) -> [J, J] int64."""
        n = self.num_joints + 1
        adj = np.zeros((n, n), dtype=bool)
        for j, p in enumerate(self.parents):
            if p >= 0:
                adj[j, p] = adj[p, j] = True
        dist = np.full((n, n), 10**6, dtype=np.int64)
        for s in range(n):
            dist[s, s] = 0
            frontier = [s]
            while frontier:
                nxt = []
                for u in frontier:
                    for v in np.nonzero(adj[u])[0]:
                        if dist[s, v] > dist[s, u] + 1:
                            dist[s, v] = dist[s, u] + 1
                            nxt.append(int(v))
                frontier = nxt
        return dist[: self.num_joints, : self.num_joints]


UNREALEGO16 = Skeleton(
    name="unrealego16",
    joint_names=tuple(JOINT_NAMES),
    parents=tuple(PARENTS_EXT),
    heatmap_joint_ids=tuple(HEATMAP_JOINT_IDS),
    left_right_pairs=tuple(LEFT_RIGHT_PAIRS),
    root_joints=(8, 9),
    upper_body=(0, 1, 2, 3, 4, 5, 6, 7),
    lower_body=(8, 9, 10, 11, 12, 13, 14, 15),
    body_parts={k: list(v) for k, v in BODY_PARTS.items()},
)

# Mo2Cap2 / xR-EgoPose / SceneEgo 15-joint set in the order used by FRAME (`keypoint_map/order.py::MO2CAP2_KEYPOINTS`:
# Neck, LeftArm, LeftForeArm, LeftHand, RightArm, RightForeArm, RightHand, LeftUpLeg, LeftLeg, LeftFoot, LeftToeBase,
# RightUpLeg, RightLeg, RightFoot, RightToeBase) so that per-joint numbers line up with the FRAME paper.
MO2CAP2_JOINT_NAMES: List[str] = [
    "neck", "left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist",
    "left_hip", "left_knee", "left_ankle", "left_foot", "right_hip", "right_knee", "right_ankle", "right_foot",
]
# raw FRAME (Mixamo-style) joint names for every mo2cap2_15 joint; "Hips" is the pelvis (root, not predicted)
FRAME_JSON_NAMES: Dict[str, Sequence[str]] = {
    "neck": ("Neck",), "left_shoulder": ("LeftArm",), "left_elbow": ("LeftForeArm",), "left_wrist": ("LeftHand",),
    "right_shoulder": ("RightArm",), "right_elbow": ("RightForeArm",), "right_wrist": ("RightHand",),
    "left_hip": ("LeftUpLeg",), "left_knee": ("LeftLeg",), "left_ankle": ("LeftFoot",), "left_foot": ("LeftToeBase",),
    "right_hip": ("RightUpLeg",), "right_knee": ("RightLeg",), "right_ankle": ("RightFoot",), "right_foot": ("RightToeBase",),
}
MO2CAP2_15 = Skeleton(
    name="mo2cap2_15",
    joint_names=tuple(MO2CAP2_JOINT_NAMES),
    parents=(15, 0, 1, 2, 0, 4, 5, 15, 7, 8, 9, 15, 11, 12, 13),  # 15 = virtual pelvis
    heatmap_joint_ids=tuple(range(15)),
    left_right_pairs=((1, 4), (2, 5), (3, 6), (7, 11), (8, 12), (9, 13), (10, 14)),
    root_joints=(7, 11),
    upper_body=(0, 1, 2, 3, 4, 5, 6),
    lower_body=(7, 8, 9, 10, 11, 12, 13, 14),
    body_parts={"neck": [0], "shoulder": [1, 4], "elbow": [2, 5], "wrist": [3, 6], "hip": [7, 11], "knee": [8, 12],
                "ankle": [9, 13], "foot": [10, 14]},
)

# EgoBody3M / Momentum body model (facebookresearch/EgoBody3M README order).
MOMENTUM_JOINT_NAMES: List[str] = [
    "b_head", "b_l_shoulder", "b_l_arm", "b_l_forearm", "b_l_wrist", "b_r_shoulder", "b_r_arm", "b_r_forearm",
    "b_r_wrist", "b_neck0", "b_spine0", "b_spine1", "b_spine2", "b_spine3", "b_r_upleg", "b_r_leg", "b_r_foot_twist",
    "b_r_talocrural", "b_r_subtalar", "b_r_transversetarsal", "b_l_upleg", "b_l_leg", "b_l_foot_twist",
    "b_l_talocrural", "b_l_subtalar", "b_l_transversetarsal",
]
MOMENTUM26 = Skeleton(
    name="momentum26",
    joint_names=tuple(MOMENTUM_JOINT_NAMES),
    parents=(9, 13, 1, 2, 3, 13, 5, 6, 7, 13, 26, 10, 11, 12, 10, 14, 15, 16, 17, 18, 10, 20, 21, 22, 23, 24),  # 26 = virtual root
    heatmap_joint_ids=tuple(range(26)),
    left_right_pairs=((1, 5), (2, 6), (3, 7), (4, 8), (20, 14), (21, 15), (22, 16), (23, 17), (24, 18), (25, 19)),
    root_joints=(14, 20),
    upper_body=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
    lower_body=(14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25),
    body_parts={"head": [0], "shoulder": [1, 5], "arm": [2, 6], "forearm": [3, 7], "wrist": [4, 8], "neck": [9],
                "spine": [10, 11, 12, 13], "upleg": [14, 20], "leg": [15, 21], "ankle": [16, 17, 22, 23],
                "foot": [18, 19, 24, 25]},
)

SKELETONS: Dict[str, Skeleton] = {s.name: s for s in (UNREALEGO16, MO2CAP2_15, MOMENTUM26)}
SKELETON_IDS: Dict[str, int] = {"unrealego16": 0, "mo2cap2_15": 1, "momentum26": 2}
MAX_JOINTS = max(s.num_joints for s in SKELETONS.values())


def get_skeleton(name: str) -> Skeleton:
    if name not in SKELETONS:
        raise KeyError(f"unknown skeleton {name!r}; registered: {sorted(SKELETONS)}")
    return SKELETONS[name]


def skeleton_id(name: str) -> int:
    return SKELETON_IDS[name]


# (source skeleton, target skeleton) -> {target joint name: [(source joint name, weight), ...]}
JOINT_MAPPINGS: Dict[Tuple[str, str], Dict[str, List[Tuple[str, float]]]] = {
    ("unrealego16", "mo2cap2_15"): {
        "neck": [("neck_01", 1.0)],
        "right_shoulder": [("upperarm_r", 1.0)], "right_elbow": [("lowerarm_r", 1.0)], "right_wrist": [("hand_r", 1.0)],
        "left_shoulder": [("upperarm_l", 1.0)], "left_elbow": [("lowerarm_l", 1.0)], "left_wrist": [("hand_l", 1.0)],
        "right_hip": [("thigh_r", 1.0)], "right_knee": [("calf_r", 1.0)], "right_ankle": [("foot_r", 1.0)], "right_foot": [("ball_r", 1.0)],
        "left_hip": [("thigh_l", 1.0)], "left_knee": [("calf_l", 1.0)], "left_ankle": [("foot_l", 1.0)], "left_foot": [("ball_l", 1.0)],
    },
    ("unrealego16", "momentum26"): {
        "b_head": [("head", 1.0)], "b_neck0": [("neck_01", 1.0)],
        "b_l_shoulder": [("neck_01", 0.5), ("upperarm_l", 0.5)], "b_l_arm": [("upperarm_l", 1.0)],
        "b_l_forearm": [("lowerarm_l", 1.0)], "b_l_wrist": [("hand_l", 1.0)],
        "b_r_shoulder": [("neck_01", 0.5), ("upperarm_r", 0.5)], "b_r_arm": [("upperarm_r", 1.0)],
        "b_r_forearm": [("lowerarm_r", 1.0)], "b_r_wrist": [("hand_r", 1.0)],
        # spine chain: interpolate between the mid-hip and the neck
        "b_spine0": [("thigh_l", 0.5), ("thigh_r", 0.5)],
        "b_spine1": [("thigh_l", 0.375), ("thigh_r", 0.375), ("neck_01", 0.25)],
        "b_spine2": [("thigh_l", 0.25), ("thigh_r", 0.25), ("neck_01", 0.5)],
        "b_spine3": [("thigh_l", 0.125), ("thigh_r", 0.125), ("neck_01", 0.75)],
        "b_r_upleg": [("thigh_r", 1.0)], "b_r_leg": [("calf_r", 1.0)], "b_r_foot_twist": [("calf_r", 0.25), ("foot_r", 0.75)],
        "b_r_talocrural": [("foot_r", 1.0)], "b_r_subtalar": [("foot_r", 0.75), ("ball_r", 0.25)], "b_r_transversetarsal": [("ball_r", 1.0)],
        "b_l_upleg": [("thigh_l", 1.0)], "b_l_leg": [("calf_l", 1.0)], "b_l_foot_twist": [("calf_l", 0.25), ("foot_l", 0.75)],
        "b_l_talocrural": [("foot_l", 1.0)], "b_l_subtalar": [("foot_l", 0.75), ("ball_l", 0.25)], "b_l_transversetarsal": [("ball_l", 1.0)],
    },
}


HUB_SKELETON = "unrealego16"  # every mapping is specified from / composed through the UnrealEgo skeleton


def _direct_mapping(src: str, dst: str) -> Tuple[np.ndarray, np.ndarray]:
    spec = JOINT_MAPPINGS[(src, dst)]
    s, d = get_skeleton(src), get_skeleton(dst)
    W = np.zeros((d.num_joints, s.num_joints), dtype=np.float32)
    one_to_one = np.full(d.num_joints, -1, dtype=np.int64)
    for j, name in enumerate(d.joint_names):
        entries = spec[name]
        for src_name, w in entries:
            W[j, s.index(src_name)] += w
        if len(entries) == 1 and abs(entries[0][1] - 1.0) < 1e-6:
            one_to_one[j] = s.index(entries[0][0])
    return W, one_to_one


def mapping_matrix(src: str, dst: str) -> Tuple[np.ndarray, np.ndarray]:
    """Linear map W [J_dst, J_src] initialising the JointMapper and the 1:1 index map ``src_of_dst`` [J_dst]
    (source joint index when the target joint is an exact copy of one source joint, else -1).

    Mappings are specified from the hub skeleton (``unrealego16``); any other pair is composed through the hub with
    the least-squares inverse of the hub -> src map (W_dst = W_hub->dst @ pinv(W_hub->src)), which is exact for the
    1:1 joints and a sensible initialisation for the interpolated ones."""
    if src == dst:
        n = get_skeleton(src).num_joints
        return np.eye(n, dtype=np.float32), np.arange(n)
    if (src, dst) in JOINT_MAPPINGS:
        return _direct_mapping(src, dst)
    for name in (src, dst):
        if name != HUB_SKELETON and (HUB_SKELETON, name) not in JOINT_MAPPINGS:
            raise KeyError(f"no joint mapping between {src!r} and {dst!r} (define JOINT_MAPPINGS[({HUB_SKELETON!r}, {name!r})])")
    W_hs, one_hs = (np.eye(get_skeleton(src).num_joints, dtype=np.float32), np.arange(get_skeleton(src).num_joints)) \
        if src == HUB_SKELETON else _direct_mapping(HUB_SKELETON, src)
    W_hd, one_hd = (np.eye(get_skeleton(dst).num_joints, dtype=np.float32), np.arange(get_skeleton(dst).num_joints)) \
        if dst == HUB_SKELETON else _direct_mapping(HUB_SKELETON, dst)
    W = (W_hd @ np.linalg.pinv(W_hs)).astype(np.float32)
    # 1:1 through the hub: dst joint j copies hub joint h; the first src joint copying the same hub joint
    one_to_one = np.full(W_hd.shape[0], -1, dtype=np.int64)
    for j in range(W_hd.shape[0]):
        h = one_hd[j]
        if h >= 0:
            ks = np.nonzero(one_hs == h)[0]
            if ks.size:
                one_to_one[j] = int(ks[0])
                W[j] = 0.0
                W[j, ks[0]] = 1.0
    return W, one_to_one

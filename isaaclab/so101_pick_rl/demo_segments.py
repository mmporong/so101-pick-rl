"""Conservative carry-only candidates; never promote an episode to full success."""
from __future__ import annotations

import math
import numpy as np


def carry_ranges(features, cfg):
    """Return inclusive zero-based state ranges spanning the required hold time.

    Both endpoints must be observed in one uninterrupted run. N states contain
    N-1 supervised transitions; a failed gap is never bridged or smoothed.
    These ranges omit approach and release and are not a complete BC dataset.
    """
    dt = float(cfg["physics_dt_s"])
    hold = float(cfg["minimum_lift_hold_s"])
    if not math.isfinite(dt) or not math.isfinite(hold) or dt <= 0 or hold <= 0:
        raise ValueError("invalid segment timing")
    required_states = math.ceil(hold / dt) + 1
    good = []
    for f in features:
        z, lift = float(f["approach_world_z"]), float(f["lift_m"])
        if not math.isfinite(z) or not math.isfinite(lift):
            raise ValueError("invalid carry state")
        good.append(all(f[k] for k in ("bilateral", "opposing_sides", "inside_pad_planes",
                                      "penetration_safe", "hand_box_clear"))
                    and -1 - 1e-6 <= z < 0 and lift >= cfg["minimum_lift_m"])
    ranges, start = [], None
    for index, valid in enumerate(good + [False]):
        if valid and start is None:
            start = index
        if not valid and start is not None:
            if index - start >= required_states:
                ranges.append((start, index - 1))
            start = None
    return ranges


def state_action_pair(previous, current):
    """Pair a trace's actual pre-state with its executed absolute target.

    The previous sample validates the pre-state. Never substitute demonstration
    observations after modifying actions. Units remain rad, m, and seconds.
    """
    if current["step"] != previous["step"] + 1:
        raise ValueError("nonconsecutive trace")
    if any(row.get("phase") != "recorded_transition" or row.get("control_boundary") is not True
           for row in (previous, current)):
        raise ValueError("only consecutive source control-boundary samples are supported")
    pre = current["pre_step_state"]
    for key, prior in (("joint_position_rad", "joint_pos"),
                       ("previous_target_rad", "executed_target"),
                       ("cube_pose_w", "cube_pose")):
        left, right = np.asarray(pre[key]), np.asarray(previous[prior])
        if left.shape != right.shape or not np.allclose(left, right, atol=1e-6, rtol=0):
            raise ValueError("trace pre-state mismatch: " + key)
    fields = (("joint_position_rad", 6), ("joint_velocity_rad_s", 6),
              ("previous_target_rad", 6), ("cube_pose_w", 7),
              ("cube_velocity_w", 6), ("box_position_w_m", 3))
    values = []
    for key, length in fields:
        value = np.asarray(pre[key], dtype=float)
        if value.shape != (length,) or not np.isfinite(value).all():
            raise ValueError("invalid observation field: " + key)
        values.append(value)
    target = np.asarray(current["executed_target"], dtype=float)
    if target.shape != (6,) or not np.isfinite(target).all():
        raise ValueError("invalid executed target")
    return np.concatenate(values), target.copy()

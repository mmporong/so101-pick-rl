"""Read-only checks of reset labels and supported-place teacher coverage.

Opening is a measured joint-angle crossing, not proof that contacts released.
These checks never relabel a kinematic proxy as a successful manipulation.
"""
from __future__ import annotations

import numpy as np


def _matrix(value, columns, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != columns or len(value) < 2 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite (T, {columns}) array, T >= 2")
    return value


def audit_episode(q_pre, targets, q_post, cube_post, box_post, cube_velocity_post,
                  initial_cube_z, *, edge_m=.03, floor_half_height_m=.004):
    q_pre = _matrix(q_pre, 6, "q_pre")
    targets = _matrix(targets, 6, "targets")
    q_post = _matrix(q_post, 6, "q_post")
    cube = _matrix(cube_post, 7, "cube_post")
    box = _matrix(box_post, 7, "box_post")
    velocity = _matrix(cube_velocity_post, 6, "cube_velocity_post")
    if len({len(v) for v in (q_pre, targets, q_post, cube, box, velocity)}) != 1:
        raise ValueError("all arrays must share the recorded time axis")
    if (not np.isfinite(initial_cube_z) or not np.isfinite(edge_m)
            or not np.isfinite(floor_half_height_m) or not edge_m > 0 or not floor_half_height_m > 0):
        raise ValueError("invalid geometry or initial height")
    if not np.allclose(q_pre[1:], q_post[:-1], atol=1e-6, rtol=0):
        raise ValueError("pre/post joint state alignment mismatch")
    if not np.allclose(np.linalg.norm(cube[:, 3:], axis=1), 1, atol=1e-4, rtol=0):
        raise ValueError("cube quaternions must be normalized wxyz")
    if (not np.allclose(np.abs(box[:, 3]), 1, atol=1e-6, rtol=0)
            or not np.allclose(box[:, 4:], 0, atol=1e-6, rtol=0)):
        raise ValueError("source box must be axis aligned for this position-only proxy")
    result = {
        "initial_joint_norm_rad": float(np.linalg.norm(q_pre[0])),
        "initial_previous_target_norm_rad": float(np.linalg.norm(targets[0])),
        "first_command_jump_l2_rad": float(np.linalg.norm(targets[1] - targets[0])),
        "first_transition_rejected_by_existing_continuity_filter": bool(np.linalg.norm(targets[1] - targets[0]) >= 1),
        "first_command_rad": targets[1].tolist(),
        "opening_proxy": None,
    }
    lifted = np.maximum.accumulate(cube[:, 2] - float(initial_cube_z)) >= .08
    crossings = np.flatnonzero(lifted[1:] & (q_post[1:, -1] > .26) & (q_post[:-1, -1] <= .26)) + 1
    if not len(crossings):
        return result
    index = int(crossings[0])
    w, x, y, z = cube[index, 3:]
    world_z_row = np.array([2 * (x*z-w*y), 2 * (y*z+w*x), 1-2*(x*x+y*y)])
    half_height = edge_m / 2 * np.abs(world_z_row).sum()
    clearance = cube[index, 2] - half_height - (box[index, 2] + floor_half_height_m)
    result["opening_proxy"] = {
        "poststate_index": index,
        "bottom_clearance_m": float(clearance),
        # Source success window on the center, not rotated-volume containment.
        "center_in_source_xy_window": bool((np.abs(cube[index, :2] - box[index, :2]) <= .045).all()),
        "linear_speed_m_s": float(np.linalg.norm(velocity[index, :3])),
        "angular_speed_rad_s": float(np.linalg.norm(velocity[index, 3:])),
        "contact_verified_release": False,
        "supported_place_success_claimed": False,
    }
    return result


def summarize(episodes):
    if not episodes:
        raise ValueError("at least one episode is required")
    rows = list(episodes.values())
    openings = [row["opening_proxy"] for row in rows if row["opening_proxy"] is not None]
    clearance = [row["bottom_clearance_m"] for row in openings]
    return {
        "episodes": len(rows),
        "zero_joint_reset_episodes": sum(row["initial_joint_norm_rad"] == 0 for row in rows),
        "first_transition_continuity_rejected": sum(row["first_transition_rejected_by_existing_continuity_filter"] for row in rows),
        "first_command_jump_median_rad": float(np.median([row["first_command_jump_l2_rad"] for row in rows])),
        "episodes_with_opening_proxy": len(openings),
        "opening_clearance_quantiles_m": np.quantile(clearance, [0, .25, .5, .75, 1]).tolist() if clearance else None,
        "opening_above_1cm": sum(row["bottom_clearance_m"] > .01 for row in openings),
        "opening_center_in_source_xy_window": sum(row["center_in_source_xy_window"] for row in openings),
        "opening_speed_above_3cm_s": sum(row["linear_speed_m_s"] > .03 for row in openings),
        "contact_verified_release_count": None,
        "full_supported_place_success_count": None,
    }

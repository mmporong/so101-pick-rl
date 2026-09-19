"""Conservative replay diagnostics, not a policy or hardware success certificate."""
from __future__ import annotations

import math

import numpy as np

from .grasp_audit import transform_points


# Preserve the existing feasibility diagnostic thresholds for this comparison.
# These are design limits, not measurements or hardware calibration constants.
DIAGNOSTIC_LIMITS = {
    "maximum_finger_cube_penetration_m": 0.001,
    "maximum_finger_table_penetration_m": 0.001,
    "maximum_abs_wrist_flex_deg": 75.0,
    "loaded_contact_threshold_n": 0.2,
}
CONTACT_PAIRS = ("gripper_cube", "jaw_cube", "gripper_table", "jaw_table")
SIDE_LIMITS = {"edge_margin_m": .003, "surface_tolerance_m": .002,
               "minimum_axis_alignment": .965, "maximum_normal_dot": -.95}


def loaded_opposing_sides(contacts, pose, size, force_threshold):
    """Check cube-local side-face interiors using only committed geometry helpers."""
    pose, size = np.asarray(pose, dtype=float), np.asarray(size, dtype=float)
    if (pose.shape != (7,) or size.shape != (3,) or not np.isfinite(pose).all()
            or not np.isfinite(size).all() or np.any(size <= .006)
            or not np.isclose(np.linalg.norm(pose[3:]), 1., atol=1e-4)
            or not math.isfinite(force_threshold) or force_threshold <= 0):
        raise ValueError("invalid cube pose, size or force threshold")
    inverse = pose[3:] * np.array([1., -1., -1., -1.])
    candidates = []
    for pair in CONTACT_PAIRS[:2]:
        sides = []
        for row in contacts[pair]:
            forces = np.asarray(row["normal_force_n"], dtype=float)
            points = np.asarray(row["points_w_m"], dtype=float)
            normals = np.asarray(row["normals_w"], dtype=float)
            if (forces.ndim != 1 or points.shape != (len(forces), 3) or normals.shape != points.shape
                    or not all(np.isfinite(x).all() for x in (forces, points, normals))):
                raise ValueError("invalid active contact arrays")
            for force, point, normal in zip(forces, points, normals):
                if force <= force_threshold:
                    continue
                point = transform_points([point - pose[:3]], np.zeros(3), inverse)[0]
                normal = transform_points([normal], np.zeros(3), inverse)[0]
                length = np.linalg.norm(normal)
                if length < 1e-8:
                    continue
                normal /= length
                axis = int(np.abs(normal).argmax())
                tangential = [i for i in range(3) if i != axis]
                if (axis == 2 or abs(normal[axis]) < SIDE_LIMITS["minimum_axis_alignment"]
                        or abs(abs(point[axis]) - size[axis] / 2) > SIDE_LIMITS["surface_tolerance_m"]
                        or np.any(np.abs(point[tangential]) > size[tangential] / 2 - SIDE_LIMITS["edge_margin_m"])
                        or point[axis] * normal[axis] <= 0):
                    continue
                sides.append((axis, np.sign(point[axis]), normal))
        candidates.append(sides)
    return any(a[0] == b[0] and a[1] != b[1] and np.dot(a[2], b[2]) <= SIDE_LIMITS["maximum_normal_dot"]
               for a in candidates[0] for b in candidates[1])


def stable_release_candidate(relative_position, gripper_rad, linear_velocity, angular_velocity):
    """Source stable_release_v2 instantaneous predicate; duration is counted separately."""
    position = np.asarray(relative_position, dtype=float)
    linear = np.asarray(linear_velocity, dtype=float)
    angular = np.asarray(angular_velocity, dtype=float)
    if (any(x.shape != (3,) for x in (position, linear, angular))
            or not all(np.isfinite(x).all() for x in (position, linear, angular))
            or not math.isfinite(gripper_rad)):
        raise ValueError("release state must be finite xyz vectors and a gripper angle")
    return bool((np.abs(position[:2]) < .045).all() and .012 < position[2] < .075
                and gripper_rad > .26 and np.linalg.norm(linear) <= .03
                and np.linalg.norm(angular) <= .5)


def summarize_contact_trace(records, *, dt_s=1 / 60, cube_size_m=(.03, .03, .03)):
    """Fail closed on incomplete samples; expose exclusions before any BC/PPO use.

    An all-pass diagnostic is necessary, not sufficient: aperture, approach and
    controlled release ordering still need a dedicated phase gate.
    """
    if not records or not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("nonempty trace and positive finite time step required")
    separation_min = {pair: None for pair in CONTACT_PAIRS}
    opposing_steps, lifted_opposing_steps = [], []
    longest, run, max_wrist = 0, 0, 0.0
    for expected, record in enumerate(records, 1):
        if record["step"] != expected or not set(CONTACT_PAIRS).issubset(record["contacts"]):
            raise ValueError("trace must contain contiguous steps and all contact pairs")
        q = np.asarray(record["joint_pos"], dtype=float)
        pose = np.asarray(record["cube_pose"], dtype=float)
        if (q.shape != (6,) or pose.shape != (7,) or not np.isfinite(q).all()
                or not np.isfinite(pose).all() or not math.isfinite(record["lift_m"])):
            raise ValueError("invalid trace pose or joint state")
        max_wrist = max(max_wrist, abs(math.degrees(float(q[3]))))
        for pair in CONTACT_PAIRS:
            values = [v for row in record["contacts"][pair] for v in row["separation_m"]]
            if not all(math.isfinite(v) for v in values):
                raise ValueError("nonfinite contact separation")
            if values:
                old = separation_min[pair]
                separation_min[pair] = min(values) if old is None else min(old, min(values))
        opposing = loaded_opposing_sides(record["contacts"], pose, cube_size_m,
                                         DIAGNOSTIC_LIMITS["loaded_contact_threshold_n"])
        if opposing:
            opposing_steps.append(expected)
            if record["lift_m"] > .03:
                lifted_opposing_steps.append(expected)
        run = run + 1 if opposing else 0
        longest = max(longest, run)
    def bounded(pairs, limit):
        return all(separation_min[pair] is None or separation_min[pair] >= -limit for pair in pairs)
    checks = {
        "finger_cube_penetration_bounded": bounded(CONTACT_PAIRS[:2], DIAGNOSTIC_LIMITS["maximum_finger_cube_penetration_m"]),
        "finger_table_penetration_bounded": bounded(CONTACT_PAIRS[2:], DIAGNOSTIC_LIMITS["maximum_finger_table_penetration_m"]),
        "wrist_flex_bounded": max_wrist <= DIAGNOSTIC_LIMITS["maximum_abs_wrist_flex_deg"],
        "loaded_opposing_side_contact_observed": bool(opposing_steps),
        "loaded_opposing_side_contact_while_lifted_observed": bool(lifted_opposing_steps),
    }
    return {
        "schema": "so101_pick_rl.demo_replay_diagnostics.v1",
        "limits": dict(DIAGNOSTIC_LIMITS), "side_limits": dict(SIDE_LIMITS),
        "samples": len(records), "sampling_dt_s": dt_s,
        "maximum_abs_wrist_flex_deg": max_wrist, "minimum_separation_m": separation_min,
        "opposing_contact_steps": opposing_steps,
        "lifted_opposing_contact_steps": lifted_opposing_steps,
        "longest_opposing_contact_seconds": longest * dt_s,
        "checks": checks, "diagnostic_checks_pass": all(checks.values()),
        "normal_grasp_gate_pass": False, "training_ready": False,
        "limitations": ["Not a full open-approach-close-lift-carry-release sequence gate",
                        "Joint flex limit is not a measurement of world-frame gripper orientation",
                        "PhysX signed distances and point forces do not certify hardware feasibility"],
    }

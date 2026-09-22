"""Small deterministic commands and contact summaries for pose-only probes."""
from __future__ import annotations

import numpy as np
import re


def validate_pad_geometry(geometry, robot_sha256):
    if (geometry.get("schema") != "so101_pick_rl.demo_box_pad_geometry.v1"
            or geometry.get("valid_for") != robot_sha256
            or geometry.get("body_order") != ["gripper", "jaw"]
            or geometry.get("inward_normals") != [[1, 0, 0], [-1, 0, 0]]):
        raise ValueError("pad geometry does not match the asset-bound search")
    offsets = np.asarray(geometry.get("pad_offsets_m"), dtype=float)
    if offsets.shape != (2, 3) or not np.isfinite(offsets).all():
        raise ValueError("invalid pad offsets")


def validate_candidate_names(candidates):
    names = [candidate.get("name", "") for candidate in candidates]
    if (not names or any(not isinstance(name, str) or re.fullmatch(r"[a-zA-Z0-9_-]+", name) is None
                         for name in names) or len(set(names)) != len(names)):
        raise ValueError("candidate names must be unique safe filename stems")


def gravity_compensated_target(reference, gravity, stiffness, lower, upper):
    """Offset arm position commands by model gravity / existing PD stiffness."""
    arrays = [np.asarray(x, dtype=float) for x in (reference, gravity, stiffness, lower, upper)]
    if any(x.shape != (6,) or not np.isfinite(x).all() for x in arrays):
        raise ValueError("gravity target inputs must be six finite values")
    reference, gravity, stiffness, lower, upper = arrays
    if (stiffness[:5] <= 0).any() or (lower >= upper).any():
        raise ValueError("invalid stiffness or joint limits")
    target = reference.copy()
    correction = gravity[:5] / stiffness[:5]
    if np.abs(correction).max() > .25:
        raise ValueError("gravity offset exceeds diagnostic bound")
    target[:5] += correction
    if ((target < lower) | (target > upper)).any():
        raise ValueError("gravity target violates joint limits")
    return target


def separating_axis_clearance(vertices, normals, rotation, position, box_centers, box_half_sizes):
    """Conservative separation witness for a convex hand hull versus axis-aligned boxes.

    Positive distance proves disjoint projections along at least one tested axis.
    Omitted edge-cross axes can reject a safe pose; they cannot certify overlap as clear.
    This does not certify a different PhysX cooked hull or later dynamics.
    """
    vertices = np.asarray(vertices, dtype=float) @ np.asarray(rotation).T + position
    axes = np.concatenate((np.eye(3), np.asarray(normals, dtype=float) @ np.asarray(rotation).T))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    projection = vertices @ axes.T
    centers = np.asarray(box_centers) @ axes.T
    radius = np.asarray(box_half_sizes) @ np.abs(axes).T
    gap = np.maximum(projection.min(axis=0) - (centers + radius),
                     (centers - radius) - projection.max(axis=0))
    return gap.max(axis=1)


def upright_cube_projected_width(rotation, edge_m):
    """Cube width along the fixed finger's world normal (local +X)."""
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all() or not np.isfinite(edge_m) or edge_m <= 0:
        raise ValueError("invalid cube projection inputs")
    return float(edge_m * np.abs(rotation[:, 0]).sum())


def opening_schedule(arm, *, closed_rad=0., open_rad=1.35, hold_steps=60, opening_steps=450):
    arm = np.asarray(arm, dtype=float)
    if arm.shape != (5,) or not np.isfinite(arm).all():
        raise ValueError("arm must contain five finite angles")
    if not (np.isfinite(closed_rad) and np.isfinite(open_rad) and closed_rad < open_rad):
        raise ValueError("gripper bounds must be finite and increasing")
    if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in (hold_steps, opening_steps)):
        raise ValueError("phase lengths must be positive integers")
    grip = np.r_[np.full(hold_steps, closed_rad),
                 np.linspace(closed_rad, open_rad, opening_steps + 1)[1:],
                 np.full(hold_steps, open_rad)]
    commands = np.c_[np.repeat(arm[None, :], len(grip), axis=0), grip]
    phases = ["hold_closed"] * hold_steps + ["opening"] * opening_steps + ["hold_open"] * hold_steps
    return commands, phases


def contact_summary(rows):
    forces = np.asarray([x for row in rows for x in row["normal_force_n"]], dtype=float)
    depths = np.asarray([x for row in rows for x in row["separation_m"]], dtype=float)
    if not np.isfinite(forces).all() or not np.isfinite(depths).all() or (forces < 0).any():
        raise ValueError("invalid contact sample")
    return {"normal_force_n": float(forces.sum()),
            "minimum_separation_m": float(depths.min()) if len(depths) else None,
            "contact_count": int(len(depths))}

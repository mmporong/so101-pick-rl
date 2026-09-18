"""Absolute-target demonstration contract, independent of the legacy delta task.

The source recorder convention assigns ``target[t + 1]`` to the transition
starting at ``obs[t]``. Numerical state alignment alone does not prove this
command convention: source recorder provenance and replay must validate it
before training. The missing final target is not fabricated. This module audits
commands; it never clips or smooths them.
"""

from __future__ import annotations

import numpy as np


JOINT_NAMES = (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper",
)


def _matrix(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != len(JOINT_NAMES) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite (T, 6) array")
    return result


def _limits(lower, upper) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    if (lower.shape != (6,) or upper.shape != (6,) or not np.isfinite(lower).all()
            or not np.isfinite(upper).all() or not (lower < upper).all()):
        raise ValueError("limits must be finite ordered six-joint bounds")
    return lower, upper


def aligned_targets(observed_q, recorded_target, post_step_q, *, atol: float = 1e-6) -> np.ndarray:
    """Check state-stream alignment and apply the source target-index convention.

    This does not certify that targets drove those states; replay is required.
    """
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("alignment tolerance must be finite and non-negative")
    observed = _matrix(observed_q, "observed_q")
    target = _matrix(recorded_target, "recorded_target")
    post_step = _matrix(post_step_q, "post_step_q")
    if observed.shape != target.shape or observed.shape != post_step.shape or len(observed) < 2:
        raise ValueError("recorder arrays must have matching shapes with at least two frames")
    if not np.allclose(observed[1:], post_step[:-1], atol=atol, rtol=0):
        raise ValueError("pre-step observations do not match previous post-step states")
    return target[1:].copy()


def normalize_targets(target, lower, upper) -> np.ndarray:
    """Map in-range absolute radians to [-1, 1]; refuse lossy clipping."""
    target = _matrix(target, "target")
    lower, upper = _limits(lower, upper)
    if ((target < lower) | (target > upper)).any():
        raise ValueError("target outside joint limits; audit the episode before training")
    return 2 * (target - lower) / (upper - lower) - 1


def denormalize_targets(action, lower, upper) -> np.ndarray:
    action = _matrix(action, "action")
    lower, upper = _limits(lower, upper)
    if (np.abs(action) > 1).any():
        raise ValueError("normalized action outside [-1, 1]")
    return lower + (action + 1) * (upper - lower) / 2


def audit_targets(target, lower, upper, *, maximum_step_norm_rad: float = 1.0) -> dict:
    """Report bounds and the source converter's continuity criterion, not safety."""
    target = _matrix(target, "target")
    lower, upper = _limits(lower, upper)
    if len(target) == 0:
        raise ValueError("empty target sequence")
    if not np.isfinite(maximum_step_norm_rad) or maximum_step_norm_rad <= 0:
        raise ValueError("continuity threshold must be positive and finite")
    outside = (target < lower) | (target > upper)
    excess = np.maximum(np.maximum(lower - target, target - upper), 0)
    steps = np.linalg.norm(np.diff(target, axis=0), axis=1)
    discontinuities = np.flatnonzero(steps > maximum_step_norm_rad)
    return {
        "frames": len(target),
        "out_of_bounds_frames": int(outside.any(axis=1).sum()),
        "out_of_bounds_values": int(outside.sum()),
        "maximum_bound_excess_rad": float(excess.max()),
        "maximum_step_norm_rad": float(steps.max()) if len(steps) else 0.0,
        "continuity_threshold_rad": maximum_step_norm_rad,
        "discontinuous_transition_indices": discontinuities.tolist(),
        "label_quality_pass": not bool(outside.any()) and len(discontinuities) == 0,
        "physics_replay_validated": False,
        "real_robot_safe": False,
    }


def validate_source_timing(env_args: dict) -> None:
    """The initial demo-aligned experiment preserves source 60 Hz, no resampling."""
    sim_args = env_args.get("sim_args", {})
    if sim_args.get("decimation") != 1 or isinstance(sim_args.get("decimation"), bool):
        raise ValueError("source decimation must be 1")
    dt = sim_args.get("dt")
    if not isinstance(dt, (int, float)) or isinstance(dt, bool) or not np.isclose(dt, 1 / 60, rtol=0, atol=1e-12):
        raise ValueError("source physics time step must be 1/60 second")

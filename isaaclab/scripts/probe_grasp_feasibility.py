#!/usr/bin/env python3
"""Probe whether one SO-101 can complete a physical pick-place sequence.

This is a scripted simulation diagnostic, not a policy evaluation.  It never
attaches or teleports the cube and never writes robot state after reset.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "isaaclab"))
from so101_pick_rl.run_contract import GRASP_GATE_LIMITS


def validate_collision_offsets(contact_offset_m: float | None, rest_offset_m: float | None) -> None:
    """Validate optional per-run cube collision offsets."""
    for name, value in (("contact offset", contact_offset_m), ("rest offset", rest_offset_m)):
        if value is not None and (not math.isfinite(value) or value < 0.0):
            raise ValueError(f"{name} must be finite and non-negative")
    if contact_offset_m is not None and rest_offset_m is not None and rest_offset_m > contact_offset_m:
        raise ValueError("rest offset must not exceed contact offset")


def point_linear_jacobian(
    spatial_jacobian: torch.Tensor, point_offset_world_m: torch.Tensor
) -> torch.Tensor:
    """Return the linear Jacobian at a world-offset point on a rigid body."""
    if spatial_jacobian.ndim != 2 or spatial_jacobian.shape[0] != 6:
        raise ValueError("spatial_jacobian must have shape (6, joints)")
    if point_offset_world_m.shape != (3,):
        raise ValueError("point_offset_world_m must have shape (3,)")
    rx, ry, rz = point_offset_world_m
    skew = torch.stack(
        (
            torch.stack((rx * 0, -rz, ry)),
            torch.stack((rz, ry * 0, -rx)),
            torch.stack((-ry, rx, rz * 0)),
        )
    )
    # v_point = v_body + omega x r = v_body - skew(r) omega.
    return spatial_jacobian[:3] - skew @ spatial_jacobian[3:]


def damped_position_step(
    jacobian: torch.Tensor,
    position_error_m: torch.Tensor,
    *,
    damping: float,
    gain: float,
    max_joint_step_rad: float,
    joint_position_rad: torch.Tensor | None = None,
    reference_joint_position_rad: torch.Tensor | None = None,
    posture_gain: float = 0.0,
) -> torch.Tensor:
    """Compute a bounded DLS task step with optional null-space posture bias."""
    if (
        jacobian.ndim != 2
        or position_error_m.ndim != 1
        or jacobian.shape[0] != position_error_m.shape[0]
    ):
        raise ValueError("Jacobian rows and task-error length must match")
    if damping <= 0 or gain <= 0 or max_joint_step_rad <= 0 or posture_gain < 0:
        raise ValueError("controller gains and limits are invalid")
    identity = torch.eye(jacobian.shape[0], device=jacobian.device, dtype=jacobian.dtype)
    inverse = torch.linalg.solve(jacobian @ jacobian.T + damping**2 * identity, identity)
    pseudo_inverse = jacobian.T @ inverse
    step = pseudo_inverse @ (gain * position_error_m)
    if posture_gain:
        if joint_position_rad is None or reference_joint_position_rad is None:
            raise ValueError("posture state and reference are required when posture_gain is non-zero")
        null_projector = torch.eye(
            jacobian.shape[1], device=jacobian.device, dtype=jacobian.dtype
        ) - pseudo_inverse @ jacobian
        step = step + null_projector @ (
            posture_gain * (reference_joint_position_rad - joint_position_rad)
        )
    return step.clamp(-max_joint_step_rad, max_joint_step_rad)


def contact_extrema(contacts: dict) -> dict[str, dict[str, float | int | None]]:
    """Reduce an audit contact sample without turning no-contact into zero overlap."""
    result = {}
    for pair, rows in contacts.items():
        separations = [value for row in rows for value in row["separation_m"]]
        forces = [value for row in rows for value in row["normal_force_n"]]
        result[pair] = {
            "contact_count": len(separations),
            "minimum_separation_m": min(separations) if separations else None,
            "peak_normal_force_n": max(forces) if forces else None,
        }
    return result


def bounded_accumulator_action(
    current_joint_position_rad: torch.Tensor,
    accumulated_target_rad: torch.Tensor,
    desired_joint_delta_rad: torch.Tensor,
    action_scale_rad: float,
    maximum_tracking_error_rad: float,
    soft_joint_lower_rad: torch.Tensor,
    soft_joint_upper_rad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integrate a target while bounding its distance from state and soft limits."""
    if not action_scale_rad > 0 or not math.isfinite(action_scale_rad):
        raise ValueError("action_scale_rad must be finite and positive")
    if not maximum_tracking_error_rad > 0 or not math.isfinite(maximum_tracking_error_rad):
        raise ValueError("maximum_tracking_error_rad must be finite and positive")
    if not (
        current_joint_position_rad.shape
        == accumulated_target_rad.shape
        == desired_joint_delta_rad.shape
        == soft_joint_lower_rad.shape
        == soft_joint_upper_rad.shape
    ):
        raise ValueError("joint state, target, delta, and limit shapes must match")
    if bool((soft_joint_lower_rad > soft_joint_upper_rad).any()):
        raise ValueError("soft joint lower limits must not exceed upper limits")
    proposed_target = accumulated_target_rad + desired_joint_delta_rad
    bounded_lower = torch.maximum(
        soft_joint_lower_rad, current_joint_position_rad - maximum_tracking_error_rad
    )
    bounded_upper = torch.minimum(
        soft_joint_upper_rad, current_joint_position_rad + maximum_tracking_error_rad
    )
    if bool((bounded_lower > bounded_upper).any()):
        raise ValueError("tracking-error interval does not intersect soft joint limits")
    bounded_target = torch.minimum(torch.maximum(proposed_target, bounded_lower), bounded_upper)
    tracking_saturated = ~torch.isclose(
        bounded_target, proposed_target, rtol=1e-6, atol=1e-7
    )
    raw_action = ((bounded_target - accumulated_target_rad) / action_scale_rad).clamp(-1.0, 1.0)
    return raw_action, bounded_target, tracking_saturated


def ramp_position(start_m, goal_m, elapsed_s: float, speed_m_s: float):
    """Advance along the straight segment at constant target speed (zero selects a step)."""
    if not math.isfinite(speed_m_s) or speed_m_s < 0 or not math.isfinite(elapsed_s) or elapsed_s < 0:
        raise ValueError("ramp speed and elapsed time must be non-negative and finite")
    if speed_m_s == 0:
        return goal_m, True  # The diagnostic's original step-target mode.
    distance_m = float(torch.linalg.vector_norm(goal_m - start_m))
    fraction = min(1.0, speed_m_s * elapsed_s / distance_m) if distance_m > 0 else 1.0
    return start_m + fraction * (goal_m - start_m), fraction >= 1.0


def arm_delta_for_hold(delta_rad, freeze_targets):
    """Keep the bounded target controller but request no additional arm motion."""
    return torch.zeros_like(delta_rad) if freeze_targets else delta_rad


def retained_posture_targets(arm_position_rad, tip_height_m, enabled):
    """Snapshot the phase-start posture; later state mutation must not move this reference."""
    if not enabled:
        return None, 0.0
    if tip_height_m is None or not math.isfinite(float(tip_height_m)):
        raise ValueError("Retained posture requires a finite measured fingertip height difference")
    return arm_position_rad.clone(), float(tip_height_m)


def validate_retained_posture_mode(enabled, grasp_frame, level_fingertips):
    if enabled and (grasp_frame != "fingertip-midpoint" or not level_fingertips):
        raise ValueError("--retain-lift-posture requires --grasp-frame fingertip-midpoint and --level-fingertips")


def closure_step_budget(joint_span_rad, fine_step_rad):
    """Allow a full fine-mode traversal plus contact confirmation, without relaxing gates."""
    if not math.isfinite(joint_span_rad) or joint_span_rad <= 0 or not math.isfinite(fine_step_rad) or fine_step_rad <= 0:
        raise ValueError("Joint span and fine step must be finite and positive")
    return max(250, math.ceil(joint_span_rad / fine_step_rad) + 10)


def phase_decision(*, contact_lost, penetration_safe, condition_steps,
                   elapsed_steps, minimum_steps, ramp_finished):
    if contact_lost:
        return "bilateral_contact_lost_after_latch"
    if not penetration_safe:
        return "penetration_guard_during_hold_or_carry"
    if condition_steps >= 5 and elapsed_steps >= minimum_steps and ramp_finished:
        return "complete"
    return None


def load_tip_offsets(report_path: Path) -> tuple[dict[str, list[float]], str]:
    """Load measured body-local fingertip offsets and bind them to their source bytes."""
    payload_bytes = report_path.read_bytes()
    payload = json.loads(payload_bytes)
    offsets = payload.get("mesh_derived_tip_offsets_m")
    if not isinstance(offsets, dict):
        raise ValueError("tip-offset report has no mesh_derived_tip_offsets_m mapping")
    result = {}
    for body in ("gripper", "jaw"):
        values = offsets.get(body)
        if (
            not isinstance(values, list)
            or len(values) != 3
            or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values)
        ):
            raise ValueError(f"tip-offset report has invalid {body!r} offset")
        result[body] = [float(value) for value in values]
    return result, hashlib.sha256(payload_bytes).hexdigest()


def pair_penetration_bounded(
    minimum_separation_by_pair_m: dict[str, float],
    pairs: tuple[str, ...],
    maximum_penetration_m: float,
    *,
    sampling_valid: bool,
) -> bool:
    """Accept absent contacts but reject sampled overlap beyond a diagnostic bound."""
    if not sampling_valid or maximum_penetration_m < 0 or not math.isfinite(maximum_penetration_m):
        return False
    values = [minimum_separation_by_pair_m[pair] for pair in pairs if pair in minimum_separation_by_pair_m]
    return not values or min(values) >= -maximum_penetration_m


def contact_latched_gripper_command(
    *,
    unilateral_contact_seen: bool,
    latched_target_rad: float | None,
    accumulated_target_rad: float,
    coarse_close_step_rad: float,
    fine_close_step_rad: float,
) -> tuple[float, float | None, str]:
    """Select coarse, fine, or target-hold closure without advancing a latched target."""
    if not (0.0 < fine_close_step_rad <= coarse_close_step_rad):
        raise ValueError("fine close step must be positive and no larger than coarse close step")
    if latched_target_rad is not None:
        return latched_target_rad - accumulated_target_rad, latched_target_rad, "latched_hold"
    if unilateral_contact_seen:
        return -fine_close_step_rad, None, "fine_close_after_unilateral_contact"
    return -coarse_close_step_rad, None, "coarse_close"


def update_contact_latch(
    *,
    unilateral_contact_seen: bool,
    latched_target_rad: float | None,
    fixed_force_n: float,
    moving_force_n: float,
    contact_threshold_n: float,
    accumulated_target_rad: float,
) -> tuple[bool, float | None, str | None]:
    """Latch the exact accumulated target at the first bilateral contact sample."""
    fixed_contact = fixed_force_n > contact_threshold_n
    moving_contact = moving_force_n > contact_threshold_n
    if latched_target_rad is not None:
        return True, latched_target_rad, None
    if fixed_contact and moving_contact:
        return True, accumulated_target_rad, "bilateral_target_latched"
    if fixed_contact or moving_contact:
        return True, None, "first_unilateral_contact" if not unilateral_contact_seen else None
    return unilateral_contact_seen, None, None


def latched_contact_maintained(
    latched_target_rad: float | None,
    fixed_force_n: float,
    moving_force_n: float,
    contact_threshold_n: float,
) -> bool:
    """Require both contacts whenever a grasp target has been latched."""
    return latched_target_rad is not None and fixed_force_n > contact_threshold_n and moving_force_n > contact_threshold_n


def fingertip_midpoint_kinematics(
    fixed_tip_position_m: torch.Tensor,
    moving_tip_position_m: torch.Tensor,
    fixed_tip_jacobian: torch.Tensor,
    moving_tip_jacobian: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build midpoint translation and one fingertip-height leveling task row."""
    if fixed_tip_position_m.shape != (3,) or moving_tip_position_m.shape != (3,):
        raise ValueError("tip positions must be 3-vectors")
    if fixed_tip_jacobian.shape != moving_tip_jacobian.shape or fixed_tip_jacobian.shape[0] != 3:
        raise ValueError("tip Jacobians must have matching (3, joints) shapes")
    return (
        0.5 * (fixed_tip_position_m + moving_tip_position_m),
        0.5 * (fixed_tip_jacobian + moving_tip_jacobian),
        moving_tip_position_m[2] - fixed_tip_position_m[2],
        (moving_tip_jacobian - fixed_tip_jacobian)[2],
    )


def _build_parser():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Summary JSON (must not exist)")
    parser.add_argument("--trace", type=Path, default=None, help="Per-policy-step JSONL")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--position-tolerance-m", type=float, default=0.012)
    parser.add_argument("--max-joint-step-rad", type=float, default=0.035)
    parser.add_argument(
        "--fine-gripper-step-rad",
        type=float,
        default=0.0035,
        help="Diagnostic close increment used after the first unilateral contact",
    )
    parser.add_argument("--max-target-tracking-error-rad", type=float, default=0.10)
    parser.add_argument("--damping", type=float, default=0.05)
    parser.add_argument("--position-gain", type=float, default=0.65)
    parser.add_argument("--posture-gain", type=float, default=0.08)
    parser.add_argument("--pregrasp-height-m", type=float, default=0.10)
    parser.add_argument("--lift-height-m", type=float, default=0.11)
    parser.add_argument("--grasp-height-offset-m", type=float, default=0.0,
                        help="Diagnostic approach offset from initial cube center; not a calibrated pad frame")
    parser.add_argument("--prelift-hold-seconds", type=float, default=0.0,
                        help="Request zero arm deltas with latched gripper target before moving; zero preserves baseline")
    parser.add_argument("--lift-speed-m-s", type=float, default=0.0,
                        help="Straight-line lift target speed; zero preserves step-target baseline")
    parser.add_argument("--retain-lift-posture", action="store_true",
                        help="Anchor lift posture bias and measured tip height difference; requires fingertip-midpoint and --level-fingertips")
    parser.add_argument("--transport-height-m", type=float, default=0.12)
    for name, default in GRASP_GATE_LIMITS.items():
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=default)
    parser.add_argument("--cube-contact-offset-m", type=float, default=None)
    parser.add_argument("--cube-rest-offset-m", type=float, default=None)
    parser.add_argument(
        "--grasp-frame",
        choices=("configured-jaw-frame", "fingertip-midpoint"),
        default="configured-jaw-frame",
    )
    parser.add_argument("--tip-offset-report", type=Path, default=None)
    parser.add_argument(
        "--level-fingertips",
        action="store_true",
        help="Add one task row that drives the measured fingertip height difference to zero",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    validate_retained_posture_mode(args.retain_lift_posture, args.grasp_frame, args.level_fingertips)
    if not math.isfinite(args.grasp_height_offset_m):
        raise ValueError("Grasp height offset must be finite")
    if any(not math.isfinite(value) or value < 0 for value in
           (args.prelift_hold_seconds, args.lift_speed_m_s)):
        raise ValueError("Hold duration and lift speed must be finite and non-negative")
    validate_collision_offsets(args.cube_contact_offset_m, args.cube_rest_offset_m)
    if args.grasp_frame == "fingertip-midpoint" and args.tip_offset_report is None:
        raise ValueError("--tip-offset-report is required for fingertip-midpoint control")
    if args.level_fingertips and args.grasp_frame != "fingertip-midpoint":
        raise ValueError("--level-fingertips requires --grasp-frame fingertip-midpoint")
    if not args.max_target_tracking_error_rad > 0:
        raise ValueError("--max-target-tracking-error-rad must be positive")
    if not 0.0 < args.fine_gripper_step_rad <= args.max_joint_step_rad:
        raise ValueError(
            "--fine-gripper-step-rad must be positive and no larger than --max-joint-step-rad"
        )
    if args.output.exists():
        raise FileExistsError(args.output)
    trace_path = args.trace or ROOT / "artifacts" / "grasp_feasibility" / (
        args.output.stem + ".jsonl"
    )
    if trace_path.exists():
        raise FileExistsError(trace_path)

    from runtime_metrics import ResourceSampler, enforce_resource_guard, write_json
    from so101_pick_rl.kit_log import bind_kit_log, summarize_kit_log
    from isaaclab.app import AppLauncher

    started = datetime.now(timezone.utc)
    report = {
        "schema": "so101_pick_rl.grasp_feasibility_probe.v1",
        "status": "failed",
        "classification": "scripted_simulation_diagnostic_not_policy_evaluation",
        "started_at_utc": started.isoformat(),
        "command": subprocess.list2cmdline(sys.argv),
        "output": str(args.output.resolve()),
        "trace": str(trace_path.resolve()),
        "gate_limits": {name: getattr(args, name) for name in GRASP_GATE_LIMITS},
        "retention_experiment": {"prelift_hold_seconds": args.prelift_hold_seconds,
                                 "lift_speed_m_s": args.lift_speed_m_s,
                                 "grasp_height_offset_m": args.grasp_height_offset_m,
                                 "retain_lift_posture": args.retain_lift_posture,
                                 "classification": "diagnostic design settings; no physics or success threshold changes"},
        "gates": {
            "opened_near_cube_before_grasp": False,
            "cube_not_pushed_before_close": False,
            "bilateral_contact": False,
            "finger_cube_penetration_bounded": False,
            "finger_table_penetration_bounded": False,
            "cube_lifted": False,
            "cube_reached_target_xy": False,
            "wrist_flex_bounded": False,
            "controlled_release_observed": False,
            "full_task_success": False,
            "simulator_error_free": False,
        },
        "phases": [],
        "limitations": [
            "Position-only differential IK; no unverified full-pose target is imposed on the five-DOF arm.",
            "Neutral-posture bias and a wrist-flex gate detect but do not prove real-world approach orientation.",
            "Native contacts are sampled at 30 Hz after env.step and expose only the last 120 Hz physics substep.",
            "PhysX signed separation and authored meshes do not expose the exact cooked convex hull geometry.",
            "The configured grasp-center frame is attached to the moving jaw, not recomputed as a symmetric fingertip midpoint.",
        ],
    }
    sampler = ResourceSampler()
    app = env = trace_stream = None
    binding = {"path": None}
    diagnostic_completed = False
    try:
        report["preflight_resources"] = enforce_resource_guard(40, 4096, 70)
        report["git_commit"] = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
        report["git_dirty"] = bool(
            subprocess.check_output(
                ["git", "-C", str(ROOT), "status", "--porcelain"], text=True
            ).strip()
        )
        sampler.start()
        app = AppLauncher(args).app
        binding = bind_kit_log(started.timestamp(), args.output.name)

        import gymnasium as gym
        import so101_pick_rl.tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg
        from isaaclab.utils.math import quat_apply
        from so101_pick_rl.grasp_audit import GraspAudit
        from so101_pick_rl.run_contract import PICK_PLACE_TASK_ID, contract_sha256

        cfg = parse_env_cfg(PICK_PLACE_TASK_ID, device=args.device, num_envs=1, use_fabric=True)
        cfg.seed = args.seed
        cfg.episode_length_s = 60.0
        if args.cube_contact_offset_m is not None:
            cfg.scene.cube.spawn.collision_props.contact_offset = args.cube_contact_offset_m
        if args.cube_rest_offset_m is not None:
            cfg.scene.cube.spawn.collision_props.rest_offset = args.cube_rest_offset_m
        env = gym.make(PICK_PLACE_TASK_ID, cfg=cfg)
        env.reset(seed=args.seed)
        raw = env.unwrapped
        robot = raw.scene["robot"]
        cube = raw.scene["cube"]
        ee_sensor = raw.scene.sensors["ee_frame"]
        audit = GraspAudit(raw)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_stream = trace_path.open("x", encoding="utf-8")

        joint_names = list(robot.joint_names)
        action_joint_names = list(raw.pick_place_spec["robot"]["joint_order"])
        ordered_joint_ids = [joint_names.index(name) for name in action_joint_names]
        arm_names = action_joint_names[:-1]
        arm_ids = [joint_names.index(name) for name in arm_names]
        arm_action_ids = [action_joint_names.index(name) for name in arm_names]
        gripper_name = action_joint_names[-1]
        gripper_id = joint_names.index(gripper_name)
        gripper_action_id = action_joint_names.index(gripper_name)
        action_term = raw.action_manager.get_term("joint_position_delta")
        action_scale_rad = float(raw.pick_place_spec["control"]["action"]["clip_abs_rad"])
        if action_term._target.shape[1] != len(action_joint_names):
            raise RuntimeError("Action accumulator order does not match the task contract")
        wrist_flex_arm_index = arm_names.index("wrist_flex")
        fixed_body_id = robot.body_names.index("gripper")
        jaw_body_id = robot.body_names.index("jaw")
        fixed_jacobian_body_id = fixed_body_id - 1 if robot.is_fixed_base else fixed_body_id
        jaw_jacobian_body_id = jaw_body_id - 1 if robot.is_fixed_base else jaw_body_id
        tip_offsets = tip_offsets_sha256 = None
        if args.tip_offset_report is not None:
            tip_offsets, tip_offsets_sha256 = load_tip_offsets(args.tip_offset_report.resolve())
        limits = robot.data.soft_joint_pos_limits[0]
        ordered_lower_limits = limits[ordered_joint_ids, 0]
        ordered_upper_limits = limits[ordered_joint_ids, 1]
        gripper_lower = float(limits[gripper_id, 0])
        gripper_upper = float(limits[gripper_id, 1])
        if not gripper_upper > gripper_lower:
            raise RuntimeError("Invalid gripper soft joint limits")

        def fingertip_state():
            if tip_offsets is None:
                raise RuntimeError("Fingertip offsets were not loaded")
            jacobians = robot.root_physx_view.get_jacobians()
            fixed_offset = torch.tensor(tip_offsets["gripper"], device=raw.device)
            moving_offset = torch.tensor(tip_offsets["jaw"], device=raw.device)
            fixed_world_offset = quat_apply(
                robot.data.body_quat_w[0, fixed_body_id].unsqueeze(0), fixed_offset.unsqueeze(0)
            )[0]
            moving_world_offset = quat_apply(
                robot.data.body_quat_w[0, jaw_body_id].unsqueeze(0), moving_offset.unsqueeze(0)
            )[0]
            fixed_position = robot.data.body_pos_w[0, fixed_body_id] + fixed_world_offset
            moving_position = robot.data.body_pos_w[0, jaw_body_id] + moving_world_offset
            fixed_spatial = jacobians[0, fixed_jacobian_body_id][:, arm_ids]
            moving_spatial = jacobians[0, jaw_jacobian_body_id][:, arm_ids]
            fixed_jacobian = point_linear_jacobian(fixed_spatial, fixed_world_offset)
            moving_jacobian = point_linear_jacobian(moving_spatial, moving_world_offset)
            return fixed_position, moving_position, fixed_jacobian, moving_jacobian

        def control_state():
            if args.grasp_frame == "configured-jaw-frame":
                jacobians = robot.root_physx_view.get_jacobians()
                jaw_position = robot.data.body_pos_w[0, jaw_body_id]
                position = ee_sensor.data.target_pos_w[0, 0]
                spatial = jacobians[0, jaw_jacobian_body_id][:, arm_ids]
                return position, point_linear_jacobian(spatial, position - jaw_position), None
            fixed_tip, moving_tip, fixed_jacobian, moving_jacobian = fingertip_state()
            position, jacobian, height_error, leveling_jacobian = fingertip_midpoint_kinematics(
                fixed_tip, moving_tip, fixed_jacobian, moving_jacobian
            )
            return position, jacobian, {
                "fixed_tip_position_w_m": fixed_tip,
                "moving_tip_position_w_m": moving_tip,
                "moving_minus_fixed_tip_height_m": height_error,
                "leveling_jacobian": leveling_jacobian,
            }

        initial_cube = cube.data.root_pos_w[0].clone()
        initial_control, _, initial_tip_state = control_state()
        initial_ee = initial_control.clone()
        initial_ee_quat = ee_sensor.data.target_quat_w[0, 0].clone()
        neutral_arm = robot.data.joint_pos[0, arm_ids].clone()
        target_position = raw.target_pos_w[0].clone()
        local_axes = torch.eye(3, device=raw.device)
        world_axes = [quat_apply(initial_ee_quat.unsqueeze(0), axis.unsqueeze(0))[0] for axis in local_axes]
        world_down = torch.tensor([0.0, 0.0, -1.0], device=raw.device)
        report.update(
            {
                "contract_sha256": contract_sha256(PICK_PLACE_TASK_ID),
                "device": str(raw.device),
                "seed": args.seed,
                "joint_names": joint_names,
                "action_joint_names": action_joint_names,
                "action_scale_rad": action_scale_rad,
                "controller": {
                    "target_update": "bounded_accumulator",
                    "maximum_tracking_error_rad": args.max_target_tracking_error_rad,
                    "tracking_error_bound_classification": (
                        "diagnostic design bound, not a measured mechanical limit"
                    ),
                },
                "body_names": list(robot.body_names),
                "control_frame": {
                    "mode": args.grasp_frame,
                    "level_fingertips": args.level_fingertips,
                    "tip_offset_report": (
                        str(args.tip_offset_report.resolve()) if args.tip_offset_report else None
                    ),
                    "tip_offset_report_sha256": tip_offsets_sha256,
                    "tip_offsets_body_local_m": tip_offsets,
                    "initial_tip_state": (
                        {key: value.tolist() for key, value in initial_tip_state.items()}
                        if initial_tip_state is not None
                        else None
                    ),
                    "leveling_constraint": (
                        "task error adds -(moving_tip_z-fixed_tip_z); Jacobian adds "
                        "(J_moving-J_fixed)[z]. This is a measured-tip kinematic constraint, "
                        "not proof of a finger-pad frame or real calibration."
                        if args.level_fingertips
                        else None
                    ),
                },
                "jacobian_bodies": {
                    "gripper": {"body_id": fixed_body_id, "view_id": fixed_jacobian_body_id},
                    "jaw": {"body_id": jaw_body_id, "view_id": jaw_jacobian_body_id},
                },
                "initial_cube_position_w_m": initial_cube.tolist(),
                "initial_control_position_w_m": initial_control.tolist(),
                "initial_configured_ee_position_w_m": ee_sensor.data.target_pos_w[0, 0].tolist(),
                "initial_configured_ee_quaternion_wxyz": initial_ee_quat.tolist(),
                "approach_orientation_diagnostic": {
                    "commanded_translation_direction_w": world_down.tolist(),
                    "initial_configured_grasp_frame_axes_w": [axis.tolist() for axis in world_axes],
                    "absolute_axis_alignment_with_world_down": [
                        abs(float(torch.dot(axis, world_down))) for axis in world_axes
                    ],
                    "full_orientation_commanded": False,
                    "single_fingertip_leveling_constraint_commanded": args.level_fingertips,
                    "reason": (
                        "Only measured fingertip height equality is constrained; no finger-pad frame is claimed."
                        if args.level_fingertips
                        else "No repository contract verifies which grasp-frame axis should be constrained for this five-DOF arm."
                    ),
                },
                "target_position_w_m": target_position.tolist(),
                "collision_probe": {
                    "cube_contact_offset_m": float(cfg.scene.cube.spawn.collision_props.contact_offset),
                    "cube_rest_offset_m": float(cfg.scene.cube.spawn.collision_props.rest_offset),
                    "source_override": args.cube_contact_offset_m is not None
                    or args.cube_rest_offset_m is not None,
                },
                "contact_schema": audit.metadata,
            }
        )

        all_pair_minimums: dict[str, float] = {}
        all_pair_peaks: dict[str, float] = {}
        success = raw.pick_place_spec["task"]["success"]
        both_contact_observed = False
        preclose_max_cube_motion = 0.0
        max_abs_wrist_flex = 0.0
        maximum_cube_lift = 0.0
        minimum_target_xy_error = math.inf
        maximum_open_fraction = 0.0
        pregrasp_open_near_cube_observed = False
        grasp_sequence_valid_observed = False
        scripted_grasp_sequence_valid_observed = False
        contract_pregrasp_opened_observed = False
        controlled_release_state_observed = False
        full_success_observed = False
        episode_ended = False
        last_pose_target = initial_ee.clone()
        grasp_offset = torch.zeros(3, device=raw.device)
        global_step = 0
        controller_stats = {
            "target_bound_saturation_joint_events": 0,
            "target_bound_saturation_steps": 0,
            "raw_action_saturation_joint_events": 0,
            "maximum_unbounded_tracking_error_rad": 0.0,
            "maximum_requested_tracking_error_rad": 0.0,
            "maximum_applied_tracking_error_rad": 0.0,
            "fine_gripper_step_rad": args.fine_gripper_step_rad,
            "fine_gripper_step_classification": "diagnostic design value",
            "unilateral_contact_event": None,
            "bilateral_latch_event": None,
            "latched_gripper_target_rad": None,
            "latched_hold_steps": 0,
            "contact_loss_event": None,
        }
        unilateral_contact_seen = False
        latched_gripper_target_rad = None

        def open_fraction() -> float:
            value = float(robot.data.joint_pos[0, gripper_id])
            return min(1.0, max(0.0, (value - gripper_lower) / (gripper_upper - gripper_lower)))

        def position_action(
            requested_position,
            gripper_delta_rad,
            *,
            gripper_target_override_rad=None,
            gripper_control_mode="direct_delta",
            freeze_arm_targets=False,
            posture_reference_rad=None,
            fingertip_height_target_m=0.0,
        ):
            control_position, task_jacobian, tip_state = control_state()
            task_error = requested_position - control_position
            if args.level_fingertips:
                if tip_state is None:
                    raise RuntimeError("Fingertip leveling requires a midpoint tip state")
                task_jacobian = torch.cat(
                    (task_jacobian, tip_state["leveling_jacobian"].unsqueeze(0)), dim=0
                )
                task_error = torch.cat(
                    (task_error, (fingertip_height_target_m - tip_state["moving_minus_fixed_tip_height_m"]).unsqueeze(0))
                )
            current_arm = robot.data.joint_pos[0, arm_ids]
            delta = damped_position_step(
                task_jacobian,
                task_error,
                damping=args.damping,
                gain=args.position_gain,
                max_joint_step_rad=args.max_joint_step_rad,
                joint_position_rad=current_arm,
                reference_joint_position_rad=(neutral_arm if posture_reference_rad is None else posture_reference_rad),
                posture_gain=args.posture_gain,
            )
            delta = arm_delta_for_hold(delta, freeze_arm_targets)
            current_ordered = robot.data.joint_pos[0, ordered_joint_ids]
            desired_delta = torch.zeros(len(action_joint_names), device=raw.device)
            desired_delta[arm_action_ids] = delta
            desired_delta[gripper_action_id] = gripper_delta_rad
            if gripper_target_override_rad is not None:
                desired_delta[gripper_action_id] = (
                    gripper_target_override_rad - action_term._target[0, gripper_action_id]
                )
            controller_stats["maximum_unbounded_tracking_error_rad"] = max(
                controller_stats["maximum_unbounded_tracking_error_rad"],
                float(
                    (
                        action_term._target[0]
                        + desired_delta
                        - current_ordered
                    ).abs().max()
                ),
            )
            raw_action, bounded_target, tracking_saturated = bounded_accumulator_action(
                current_ordered,
                action_term._target[0],
                desired_delta,
                action_scale_rad,
                args.max_target_tracking_error_rad,
                ordered_lower_limits,
                ordered_upper_limits,
            )
            saturated_joints = int(tracking_saturated.sum())
            controller_stats["target_bound_saturation_joint_events"] += saturated_joints
            controller_stats["target_bound_saturation_steps"] += int(saturated_joints > 0)
            controller_stats["raw_action_saturation_joint_events"] += int(
                (raw_action.abs() >= 1.0 - 1e-6).sum()
            )
            controller_stats["maximum_requested_tracking_error_rad"] = max(
                controller_stats["maximum_requested_tracking_error_rad"],
                float((bounded_target - current_ordered).abs().max()),
            )
            action = raw_action.unsqueeze(0)
            return action, task_jacobian, tip_state, {
                "bounded_target_rad": bounded_target,
                "tracking_saturated": tracking_saturated,
                "gripper_control_mode": gripper_control_mode,
                "requested_gripper_delta_rad": float(desired_delta[gripper_action_id]),
                "gripper_target_override_rad": gripper_target_override_rad,
            }

        def update_partial_report():
            finger_minimum = min(
                all_pair_minimums.get("gripper_cube", math.inf),
                all_pair_minimums.get("jaw_cube", math.inf),
            )
            table_minimum = min(
                all_pair_minimums.get("gripper_table", math.inf),
                all_pair_minimums.get("jaw_table", math.inf),
            )
            report["measurements"] = {
                "steps": global_step,
                "simulation_seconds": global_step * raw.step_dt,
                "maximum_gripper_open_fraction": maximum_open_fraction,
                "pregrasp_open_near_cube_observed": pregrasp_open_near_cube_observed,
                "grasp_sequence_valid_observed": grasp_sequence_valid_observed,
                "scripted_grasp_sequence_valid_observed": (
                    scripted_grasp_sequence_valid_observed
                ),
                "contract_pregrasp_opened_observed": contract_pregrasp_opened_observed,
                "maximum_preclose_cube_motion_m": preclose_max_cube_motion,
                "bilateral_contact_observed": both_contact_observed,
                "minimum_finger_cube_separation_m": (
                    finger_minimum if math.isfinite(finger_minimum) else None
                ),
                "finger_cube_penetration_guard": {
                    "configured_limit_m": args.max_allowed_penetration_m,
                    "classification": "diagnostic design gate, not a measured real-robot tolerance",
                },
                "minimum_finger_table_separation_m": (
                    table_minimum if math.isfinite(table_minimum) else None
                ),
                "finger_table_penetration_guard": {
                    "configured_limit_m": args.max_allowed_finger_table_penetration_m,
                    "classification": "diagnostic design gate, not a measured real-robot tolerance",
                },
                "minimum_separation_by_pair_m": all_pair_minimums,
                "peak_normal_force_by_pair_n": all_pair_peaks,
                "maximum_cube_lift_m": maximum_cube_lift,
                "minimum_target_xy_error_m": (
                    minimum_target_xy_error if math.isfinite(minimum_target_xy_error) else None
                ),
                "maximum_abs_wrist_flex_rad": max_abs_wrist_flex,
                "wrist_flex_guard": {
                    "maximum_abs_rad": max_abs_wrist_flex,
                    "configured_limit_deg": args.max_abs_wrist_flex_deg,
                    "classification": "configurable diagnostic posture guard, not a measured mechanical limit",
                },
                "controlled_release_state_observed": controlled_release_state_observed,
                "full_task_success_observed": full_success_observed,
                "grasp_center_offset_from_cube_after_close_m": grasp_offset.tolist(),
                "controller": dict(controller_stats),
            }
            report["gates"].update(
                {
                    "opened_near_cube_before_grasp": pregrasp_open_near_cube_observed,
                    "cube_not_pushed_before_close": (
                        global_step > 0
                        and preclose_max_cube_motion <= args.max_preclose_cube_motion_m
                    ),
                    "bilateral_contact": both_contact_observed
                    and scripted_grasp_sequence_valid_observed,
                    "finger_cube_penetration_bounded": math.isfinite(finger_minimum)
                    and finger_minimum >= -args.max_allowed_penetration_m,
                    "finger_table_penetration_bounded": pair_penetration_bounded(
                        all_pair_minimums,
                        ("gripper_table", "jaw_table"),
                        args.max_allowed_finger_table_penetration_m,
                        sampling_valid=global_step > 0,
                    ),
                    "cube_lifted": maximum_cube_lift >= success["minimum_delta_z_m"],
                    "cube_reached_target_xy": minimum_target_xy_error
                    <= success["placement_xy_tolerance_m"],
                    "wrist_flex_bounded": max_abs_wrist_flex
                    <= math.radians(args.max_abs_wrist_flex_deg),
                    "controlled_release_observed": controlled_release_state_observed,
                    "full_task_success": full_success_observed,
                }
            )

        def run_phase(
            name,
            requested_position,
            gripper_direction,
            max_steps,
            completion,
            *,
            contact_latched_close=False,
            hold_latched_gripper=False,
            freeze_arm_targets=False,
            minimum_phase_steps=5,
            ramp_speed_m_s=0.0,
            retain_posture=False,
        ):
            nonlocal global_step, both_contact_observed, preclose_max_cube_motion
            nonlocal max_abs_wrist_flex, maximum_cube_lift, minimum_target_xy_error
            nonlocal maximum_open_fraction, controlled_release_state_observed
            nonlocal full_success_observed, episode_ended
            nonlocal pregrasp_open_near_cube_observed, grasp_sequence_valid_observed
            nonlocal scripted_grasp_sequence_valid_observed
            nonlocal contract_pregrasp_opened_observed
            nonlocal unilateral_contact_seen, latched_gripper_target_rad
            errors = []
            condition_steps = 0
            phase = {"name": name, "requested_position_w_m": requested_position.tolist()}
            phase["maximum_steps"] = max_steps
            phase_last_control = control_state()[0].clone()
            ramp_start_m = phase_last_control.clone()
            start_tip_state = control_state()[2]
            posture_reference, tip_height_target_m = retained_posture_targets(
                robot.data.joint_pos[0, arm_ids],
                start_tip_state["moving_minus_fixed_tip_height_m"] if start_tip_state is not None else None,
                retain_posture)
            phase["retained_posture_reference_rad"] = posture_reference.tolist() if posture_reference is not None else None
            phase["fingertip_height_target_m"] = tip_height_target_m
            phase["freeze_arm_targets"] = freeze_arm_targets
            phase["ramp_speed_m_s"] = ramp_speed_m_s
            phase["position_error_semantics"] = "distance to final goal, not ramp waypoint tracking error"
            for phase_step in range(max_steps):
                commanded_position, ramp_finished = ramp_position(
                    ramp_start_m, requested_position, (phase_step + 1) * raw.step_dt, ramp_speed_m_s)
                latch_active_before_step = latched_gripper_target_rad is not None
                if contact_latched_close:
                    gripper_delta_rad, target_override, gripper_control_mode = (
                        contact_latched_gripper_command(
                            unilateral_contact_seen=unilateral_contact_seen,
                            latched_target_rad=latched_gripper_target_rad,
                            accumulated_target_rad=float(
                                action_term._target[0, gripper_action_id]
                            ),
                            coarse_close_step_rad=args.max_joint_step_rad,
                            fine_close_step_rad=args.fine_gripper_step_rad,
                        )
                    )
                elif hold_latched_gripper:
                    if latched_gripper_target_rad is None:
                        raise RuntimeError("Cannot hold gripper before bilateral contact latch")
                    gripper_delta_rad = 0.0
                    target_override = latched_gripper_target_rad
                    gripper_control_mode = "latched_hold"
                else:
                    gripper_delta_rad = gripper_direction * args.max_joint_step_rad
                    target_override = None
                    gripper_control_mode = "direct_delta"
                action, task_jacobian, tip_state, controller_step = position_action(
                    commanded_position,
                    gripper_delta_rad,
                    gripper_target_override_rad=target_override,
                    gripper_control_mode=gripper_control_mode,
                    freeze_arm_targets=freeze_arm_targets,
                    posture_reference_rad=posture_reference,
                    fingertip_height_target_m=tip_height_target_m,
                )
                if gripper_control_mode == "latched_hold":
                    controller_stats["latched_hold_steps"] += 1
                accumulator_before = action_term._target[0].clone()
                pre_step_control = control_state()[0].clone()
                pre_step_error = float(
                    torch.linalg.vector_norm(requested_position - pre_step_control)
                )
                _, _, terminated, truncated, _ = env.step(action)
                global_step += 1
                ended = bool(terminated[0]) or bool(truncated[0])
                success_flag = bool(raw.termination_manager.get_term("pick_place_success")[0])
                if ended:
                    errors.append(pre_step_error)
                    phase_last_control = pre_step_control
                    full_success_observed |= success_flag
                    controlled_release_state_observed |= bool(
                        raw.last_episode_metrics["released"][0]
                    )
                    episode_ended = True
                    trace_stream.write(
                        json.dumps(
                            {
                                "state_index": global_step,
                                "phase": name,
                                "phase_step": phase_step,
                                "requested_control_position_w_m": requested_position.tolist(),
                                "commanded_control_position_w_m": commanded_position.tolist(),
                                "ramp_finished": ramp_finished,
                                "last_pre_step_final_goal_position_error_m": pre_step_error,
                                "last_pre_step_commanded_position_error_m": float(
                                    torch.linalg.vector_norm(commanded_position - pre_step_control)),
                                "last_pre_step_control_position_w_m": pre_step_control.tolist(),
                                "issued_raw_action": action[0].tolist(),
                                "joint_target_before_action_rad": accumulator_before.tolist(),
                                "bounded_requested_joint_target_rad": controller_step[
                                    "bounded_target_rad"
                                ].tolist(),
                                "tracking_saturated_joints": controller_step[
                                    "tracking_saturated"
                                ].tolist(),
                                "gripper_control_mode": gripper_control_mode,
                                "requested_gripper_delta_rad": controller_step[
                                    "requested_gripper_delta_rad"
                                ],
                                "latched_gripper_target_rad": latched_gripper_target_rad,
                                "post_step_state_omitted": "environment auto-reset occurred",
                                "terminated": bool(terminated[0]),
                                "truncated": bool(truncated[0]),
                                "full_success_pre_reset_flag": success_flag,
                                "released_pre_reset_metric": bool(
                                    raw.last_episode_metrics["released"][0]
                                ),
                            },
                            allow_nan=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    update_partial_report()
                    phase["environment_ended"] = True
                    break
                audit_sample = audit.sample(global_step)
                applied_tracking_error = (
                    action_term._target[0] - robot.data.joint_pos[0, ordered_joint_ids]
                ).abs()
                controller_stats["maximum_applied_tracking_error_rad"] = max(
                    controller_stats["maximum_applied_tracking_error_rad"],
                    float(applied_tracking_error.max()),
                )
                extrema = contact_extrema(audit_sample["contacts"])
                control_position, _, tip_state = control_state()
                configured_ee_position = ee_sensor.data.target_pos_w[0, 0]
                cube_position = cube.data.root_pos_w[0]
                error = float(torch.linalg.vector_norm(requested_position - control_position))
                errors.append(error)
                phase_last_control = control_position.clone()
                current_open = open_fraction()
                control_cube_distance = float(
                    torch.linalg.vector_norm(control_position - cube_position)
                )
                configured_ee_cube_distance = float(
                    torch.linalg.vector_norm(configured_ee_position - cube_position)
                )
                current_lift = float(cube_position[2] - initial_cube[2])
                current_xy_error = float(torch.linalg.vector_norm(cube_position[:2] - target_position[:2]))
                wrist = abs(float(robot.data.joint_pos[0, arm_ids[wrist_flex_arm_index]]))
                maximum_open_fraction = max(maximum_open_fraction, current_open)
                maximum_cube_lift = max(maximum_cube_lift, current_lift)
                minimum_target_xy_error = min(minimum_target_xy_error, current_xy_error)
                max_abs_wrist_flex = max(max_abs_wrist_flex, wrist)
                if name in ("open", "pregrasp", "descend"):
                    preclose_max_cube_motion = max(
                        preclose_max_cube_motion,
                        float(torch.linalg.vector_norm(cube_position - initial_cube)),
                    )
                for pair, values in extrema.items():
                    separation = values["minimum_separation_m"]
                    force = values["peak_normal_force_n"]
                    if separation is not None:
                        all_pair_minimums[pair] = min(all_pair_minimums.get(pair, math.inf), separation)
                    if force is not None:
                        all_pair_peaks[pair] = max(all_pair_peaks.get(pair, 0.0), force)
                fixed_force = extrema.get("gripper_cube", {}).get("peak_normal_force_n") or 0.0
                moving_force = extrema.get("jaw_cube", {}).get("peak_normal_force_n") or 0.0
                contact_threshold = success["contact_threshold_n"]
                both_contact = fixed_force > contact_threshold and moving_force > contact_threshold
                both_contact_observed |= both_contact
                if contact_latched_close:
                    unilateral_contact_seen, new_latch, latch_event = update_contact_latch(
                        unilateral_contact_seen=unilateral_contact_seen,
                        latched_target_rad=latched_gripper_target_rad,
                        fixed_force_n=fixed_force,
                        moving_force_n=moving_force,
                        contact_threshold_n=contact_threshold,
                        accumulated_target_rad=float(
                            action_term._target[0, gripper_action_id]
                        ),
                    )
                    if latch_event == "first_unilateral_contact":
                        controller_stats["unilateral_contact_event"] = {
                            "state_index": global_step,
                            "phase_step": phase_step,
                            "fixed_force_n": fixed_force,
                            "moving_force_n": moving_force,
                        }
                    if latch_event == "bilateral_target_latched":
                        latched_gripper_target_rad = new_latch
                        controller_stats["latched_gripper_target_rad"] = new_latch
                        controller_stats["bilateral_latch_event"] = {
                            "state_index": global_step,
                            "phase_step": phase_step,
                            "fixed_force_n": fixed_force,
                            "moving_force_n": moving_force,
                            "gripper_open_fraction": current_open,
                            "latched_target_rad": new_latch,
                        }
                contact_must_hold = hold_latched_gripper or (
                    contact_latched_close and latch_active_before_step
                )
                contact_lost = contact_must_hold and not latched_contact_maintained(
                    latched_gripper_target_rad,
                    fixed_force,
                    moving_force,
                    contact_threshold,
                )
                if contact_lost and controller_stats["contact_loss_event"] is None:
                    controller_stats["contact_loss_event"] = {
                        "state_index": global_step,
                        "phase": name,
                        "phase_step": phase_step,
                        "fixed_force_n": fixed_force,
                        "moving_force_n": moving_force,
                    }
                pregrasp_open_near_cube_observed |= (
                    not both_contact
                    and control_cube_distance <= success["pregrasp_maximum_ee_distance_m"]
                    and current_open >= success["pregrasp_open_fraction"]
                )
                grasp_sequence_valid_observed |= bool(raw.pick_place_state.grasp_sequence_valid[0])
                contract_pregrasp_opened_observed |= bool(raw.pick_place_state.pregrasp_opened[0])
                scripted_grasp_sequence_valid_observed |= (
                    pregrasp_open_near_cube_observed
                    and both_contact
                    and current_open <= success["maximum_grasp_open_fraction"]
                )
                controlled_release_state_observed |= bool(raw.pick_place_state.released[0])
                full_success_observed |= success_flag
                audit_sample.update(
                    {
                        "phase": name,
                        "phase_step": phase_step,
                        "grasp_frame_mode": args.grasp_frame,
                        "requested_control_position_w_m": requested_position.tolist(),
                        "commanded_control_position_w_m": commanded_position.tolist(),
                        "ramp_finished": ramp_finished,
                        "achieved_control_position_w_m": control_position.tolist(),
                        "configured_ee_position_w_m": configured_ee_position.tolist(),
                        "control_position_error_m": error,
                        "final_goal_position_error_m": error,
                        "commanded_position_error_m": float(
                            torch.linalg.vector_norm(commanded_position - control_position)),
                        "control_cube_distance_m": control_cube_distance,
                        "configured_ee_cube_distance_m": configured_ee_cube_distance,
                        "cube_position_w_m": cube_position.tolist(),
                        "target_xy_error_m": current_xy_error,
                        "gripper_open_fraction": current_open,
                        "issued_raw_action": action[0].tolist(),
                        "processed_joint_delta_rad": action_term.processed_actions[0].tolist(),
                        "commanded_joint_target_rad": action_term._target[0].tolist(),
                        "bounded_requested_joint_target_rad": controller_step[
                            "bounded_target_rad"
                        ].tolist(),
                        "tracking_saturated_joints": controller_step[
                            "tracking_saturated"
                        ].tolist(),
                        "gripper_control_mode": gripper_control_mode,
                        "requested_gripper_delta_rad": controller_step[
                            "requested_gripper_delta_rad"
                        ],
                        "latched_gripper_target_rad": latched_gripper_target_rad,
                        "applied_target_tracking_error_rad": applied_tracking_error.tolist(),
                        "contact_extrema": extrema,
                        "task_jacobian_singular_values": torch.linalg.svdvals(task_jacobian).tolist(),
                        "tip_state": (
                            {key: value.tolist() for key, value in tip_state.items()}
                            if tip_state is not None
                            else None
                        ),
                        "terminated": bool(terminated[0]),
                        "truncated": bool(truncated[0]),
                        "full_success": success_flag,
                    }
                )
                GraspAudit.write_sample(trace_stream, audit_sample)
                update_partial_report()
                penetration_safe = not hold_latched_gripper or (
                    pair_penetration_bounded(all_pair_minimums, ("gripper_cube", "jaw_cube"),
                                             args.max_allowed_penetration_m, sampling_valid=True)
                    and pair_penetration_bounded(all_pair_minimums, ("gripper_table", "jaw_table"),
                                                args.max_allowed_finger_table_penetration_m, sampling_valid=True)
                )
                condition_steps = condition_steps + 1 if completion(error, current_open, both_contact) else 0
                decision = phase_decision(
                    contact_lost=contact_lost, penetration_safe=penetration_safe,
                    condition_steps=condition_steps, elapsed_steps=phase_step + 1,
                    minimum_steps=minimum_phase_steps, ramp_finished=ramp_finished)
                if decision == "complete":
                    phase["completion_condition_met"] = True
                    break
                if decision is not None:
                    phase["failure_reason"] = decision
                    break
            if not phase.get("completion_condition_met") and not phase.get("environment_ended"):
                phase.setdefault("failure_reason", "maximum_phase_steps_exhausted")
            phase.update(
                {
                    "steps": len(errors),
                    "minimum_position_error_m": min(errors),
                    "final_position_error_m": errors[-1],
                    "achieved_position_w_m": phase_last_control.tolist(),
                    "gripper_open_fraction": None if episode_ended else open_fraction(),
                }
            )
            report["phases"].append(phase)
            return (
                bool(phase.get("completion_condition_met"))
                and not phase.get("environment_ended", False)
            ) or (bool(phase.get("environment_ended")) and full_success_observed)

        # All targets are world-frame positions for the configured grasp-center frame.
        pregrasp = initial_cube + torch.tensor([0.0, 0.0, args.pregrasp_height_m], device=raw.device)
        descend = initial_cube.clone()
        descend[2] += args.grasp_height_offset_m
        if not run_phase("open", initial_ee, +1.0, 80, lambda _e, opened, _c: opened >= 0.85):
            raise RuntimeError("Gripper did not open while holding the initial pose")
        if not run_phase("pregrasp", pregrasp, +1.0, 180, lambda e, _o, _c: e <= args.position_tolerance_m):
            raise RuntimeError("Pregrasp position was not reached")
        last_pose_target = pregrasp
        if not run_phase("descend", descend, +1.0, 140, lambda e, _o, _c: e <= args.position_tolerance_m):
            raise RuntimeError("Grasp-center position was not reached")
        last_pose_target = descend
        if not run_phase(
            "close",
            last_pose_target,
            -1.0,
            closure_step_budget(gripper_upper - gripper_lower, args.fine_gripper_step_rad),
            lambda _e, opened, contact: contact
            and opened <= success["maximum_grasp_open_fraction"],
            contact_latched_close=True,
        ):
            raise RuntimeError(
                "Bilateral finger contact was not established and maintained during closure"
            )
        if not pair_penetration_bounded(
            all_pair_minimums,
            ("gripper_cube", "jaw_cube"),
            args.max_allowed_penetration_m,
            sampling_valid=global_step > 0,
        ):
            raise RuntimeError("Finger-cube penetration guard failed before lift")
        if not pair_penetration_bounded(
            all_pair_minimums,
            ("gripper_table", "jaw_table"),
            args.max_allowed_finger_table_penetration_m,
            sampling_valid=global_step > 0,
        ):
            raise RuntimeError("Finger-table penetration guard failed before lift")
        if args.prelift_hold_seconds > 0:
            hold_steps = max(5, math.ceil(args.prelift_hold_seconds / raw.step_dt))
            if not run_phase("stationary_hold", control_state()[0].clone(), 0.0, hold_steps,
                             lambda _e, _o, contact: contact, hold_latched_gripper=True,
                             freeze_arm_targets=True, minimum_phase_steps=hold_steps):
                raise RuntimeError("Stationary grasp retention failed before lift")
        grasp_offset = control_state()[0] - cube.data.root_pos_w[0]
        lift = control_state()[0].clone()
        lift[2] += args.lift_height_m
        if not run_phase(
            "lift",
            lift,
            0.0,
            (math.ceil(args.lift_height_m / args.lift_speed_m_s / raw.step_dt) + 180
             if args.lift_speed_m_s > 0 else 180),
            lambda e, _o, _c: e <= args.position_tolerance_m,
            hold_latched_gripper=True,
            ramp_speed_m_s=args.lift_speed_m_s,
            retain_posture=args.retain_lift_posture,
        ):
            raise RuntimeError("Lift aborted or target was not reached with bilateral contact held")
        transport = target_position + grasp_offset
        transport[2] = target_position[2] + grasp_offset[2] + args.transport_height_m
        if not run_phase(
            "transport",
            transport,
            0.0,
            220,
            lambda e, _o, _c: e <= args.position_tolerance_m,
            hold_latched_gripper=True,
        ):
            raise RuntimeError("Transport aborted or target was not reached with bilateral contact held")
        lower = target_position + grasp_offset
        if not run_phase(
            "lower",
            lower,
            0.0,
            180,
            lambda e, _o, _c: e <= args.position_tolerance_m,
            hold_latched_gripper=True,
        ):
            raise RuntimeError("Lower aborted or target was not reached with bilateral contact held")
        if not run_phase("release", lower, +1.0, 100, lambda _e, opened, _c: opened >= 0.85):
            raise RuntimeError("Gripper did not open at the target")
        if not episode_ended:
            retreat = control_state()[0].clone()
            retreat[2] += max(0.10, success["minimum_ee_clearance_m"])
            run_phase(
                "retreat",
                retreat,
                +1.0,
                180,
                lambda e, _o, _c: e <= args.position_tolerance_m,
            )
            if not episode_ended:
                run_phase(
                    "stabilize",
                    retreat,
                    +1.0,
                    80,
                    lambda _e, _o, _c: full_success_observed,
                )
        update_partial_report()
        diagnostic_completed = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    finally:
        if trace_stream is not None:
            trace_stream.close()
        if env is not None:
            env.close()
        report["resources"] = sampler.stop()
        report["simulator_log"] = summarize_kit_log(binding)
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        simulator_clean = report["simulator_log"].get("error_count") == 0
        report["gates"]["simulator_error_free"] = simulator_clean
        feasible = diagnostic_completed and bool(report["gates"]) and all(report["gates"].values())
        report["status"] = "feasible" if feasible else "not_feasible_or_incomplete"
        write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        if app is not None:
            app.close()
    return 0 if report["status"] == "feasible" else 1


if __name__ == "__main__":
    raise SystemExit(main())

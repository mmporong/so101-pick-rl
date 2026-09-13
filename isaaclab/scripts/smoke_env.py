#!/usr/bin/env python3
"""Run a bounded Isaac Lab environment smoke test and write measured evidence."""

from __future__ import annotations

import argparse
import hashlib
import math
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ISAACLAB_PROJECT_DIR = SCRIPT_DIR.parent
REPOSITORY_ROOT = ISAACLAB_PROJECT_DIR.parent
sys.path.insert(0, str(ISAACLAB_PROJECT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from runtime_metrics import ResourceSampler, enforce_resource_guard, write_json
from so101_pick_rl.kit_log import bind_kit_log, summarize_kit_log
from so101_pick_rl.run_contract import (
    LIFT_TASK_ID,
    PICK_PLACE_TASK_ID,
    contract_sha256,
    load_spec,
)
from so101_pick_rl.validation import count_reset_events, finalize_smoke_checks, summarize_reset_failures


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="SO101-LiftCube-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--physics_steps", type=int, default=120)
parser.add_argument("--action_mode", choices=("zero", "random"), default="zero")
parser.add_argument("--random_action_amplitude", type=float, default=0.25)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--max_baseline_gpu_util", type=float, default=40.0)
parser.add_argument("--minimum_free_vram_mib", type=float, default=4096.0)
parser.add_argument("--max_baseline_cpu_util", type=float, default=70.0)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

started_at = datetime.now(timezone.utc)
output_path = args.output.expanduser().resolve()
command = subprocess.list2cmdline(sys.argv)
try:
    preflight_resources = enforce_resource_guard(
        args.max_baseline_gpu_util,
        args.minimum_free_vram_mib,
        args.max_baseline_cpu_util,
    )
except Exception as exc:
    write_json(
        output_path,
        {
            "schema": "so101_pick_rl.windows_smoke.v1",
            "status": "blocked_resource_guard",
            "classification": "not_run",
            "command": command,
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
        },
    )
    print(f"SMOKE_REPORT={output_path}", flush=True)
    raise SystemExit(3)
sampler = ResourceSampler(interval_seconds=1.0)
sampler.start()
simulation_app = None
kit_log_binding: dict[str, object] = {"path": None, "binding": None, "reason": "launcher not started"}
try:
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    kit_log_binding = bind_kit_log(started_at.timestamp(), output_path.name)
except Exception as exc:
    kit_log_binding = bind_kit_log(started_at.timestamp(), output_path.name)
    write_json(
        output_path,
        {
            "schema": "so101_pick_rl.windows_smoke.v1",
            "status": "failed",
            "classification": "not_run",
            "command": command,
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"AppLauncher failed: {type(exc).__name__}: {exc}",
            "resources": sampler.stop(),
            "simulator_log": summarize_kit_log(kit_log_binding),
        },
    )
    if simulation_app is not None:
        simulation_app.close()
    print(f"SMOKE_REPORT={output_path}", flush=True)
    raise SystemExit(4)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
import so101_pick_rl.tasks  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def git_metadata() -> dict[str, object]:
    def run(*command: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), *command],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "dirty_paths": status.splitlines(),
    }


def run_config_sha256(command: str) -> str:
    digest = hashlib.sha256()
    digest.update((REPOSITORY_ROOT / "configs" / "isaaclab" / "windows_runtime.json").read_bytes())
    digest.update(b"\0")
    digest.update(command.encode("utf-8"))
    return digest.hexdigest()


def report_path(path: Path) -> str:
    try:
        return path.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(path)


def tensor_is_finite(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(tensor_is_finite(item) for item in value.values())
    return True


def per_environment_finite_mask(value, num_envs: int, device) -> torch.Tensor:
    """Return one finite-state flag per environment for nested observations."""
    result = torch.ones(num_envs, dtype=torch.bool, device=device)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0 or value.shape[0] != num_envs:
            return torch.zeros(num_envs, dtype=torch.bool, device=device)
        finite = torch.isfinite(value)
        if finite.ndim > 1:
            finite = finite.reshape(num_envs, -1).all(dim=1)
        return finite
    if isinstance(value, dict):
        for item in value.values():
            result &= per_environment_finite_mask(item, num_envs, device)
    return result


def per_environment_nonzero_mask(value: torch.Tensor) -> torch.Tensor:
    """Return one non-zero flag per environment for a state tensor."""
    if value.ndim == 1:
        return value != 0
    return (value != 0).reshape(value.shape[0], -1).any(dim=1)


def main() -> int:
    git = git_metadata()
    task_spec = load_spec(args.task)
    task_contract_sha = contract_sha256(args.task)
    joint_order = tuple(task_spec["robot"]["joint_order"])
    physics_hz = int(task_spec["control"]["physics_hz"])
    policy_hz = int(task_spec["control"]["policy_hz"])
    expected_decimation = physics_hz // policy_hz
    action_scale_rad = float(task_spec["control"]["action"]["clip_abs_rad"])
    expected_observation_dimension = int(task_spec.get("observation", {}).get("dimension", 22))
    pick_place_target_cfg = task_spec["task"].get("target")
    payload: dict[str, object] = {
        "schema": "so101_pick_rl.windows_smoke.v1",
        "status": "failed",
        "classification": "not_run",
        "command": command,
        "started_at_utc": started_at.isoformat(),
        "task": args.task,
        "seed": args.seed,
        "requested_num_envs": args.num_envs,
        "requested_physics_steps": args.physics_steps,
        "action_mode": args.action_mode,
        "preflight_resources": preflight_resources,
        "git": git,
        "contract_sha256": task_contract_sha,
    }
    env = None
    exit_code = 1
    loop_started = None
    try:
        torch.manual_seed(args.seed)
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, use_fabric=True)
        env_cfg.seed = args.seed
        env = gym.make(args.task, cfg=env_cfg)
        observations, _ = env.reset(seed=args.seed)

        unwrapped = env.unwrapped
        is_so101_task = args.task in (LIFT_TASK_ID, PICK_PLACE_TASK_ID)
        is_pick_place_task = args.task == PICK_PLACE_TASK_ID
        decimation = int(unwrapped.cfg.decimation)
        policy_steps = math.ceil(args.physics_steps / decimation)
        action_shape = env.action_space.shape
        robot = unwrapped.scene["robot"]
        cube = unwrapped.scene["cube"] if "cube" in unwrapped.scene.rigid_objects else None
        initial_joint_positions = robot.data.joint_pos.clone()
        initial_cube_positions = cube.data.root_pos_w.clone() if cube is not None else None
        initial_target_positions = unwrapped.target_pos_w.clone() if is_pick_place_task else None
        initial_pick_place_state_zero = None
        if is_pick_place_task:
            state = unwrapped.pick_place_state
            contact_term_index = task_spec["observation"]["terms"].index("finger_contact_force_n")
            contact_start = sum(task_spec["observation"]["term_dimensions"][:contact_term_index])
            contact_slice = slice(contact_start, contact_start + task_spec["observation"]["term_dimensions"][contact_term_index])
            state_fields = ("picked", "carry_valid", "released", "lift_steps", "stable_steps", "previous_release_ready",
                            "pregrasp_opened", "grasp_sequence_valid")
            initial_pick_place_state_zero = all(
                bool(torch.count_nonzero(getattr(state, field)).item() == 0) for field in state_fields
            )

        non_finite_steps = 0
        reward_non_finite_steps = 0
        terminated_count = 0
        truncated_count = 0
        automatic_reset_count = 0
        reset_failure_count = 0
        reset_failure_reason_counts = {
            "returned_observation_non_finite": 0,
            "episode_length_not_zero": 0,
            "robot_state_non_finite": 0,
            "cube_state_non_finite": 0,
            "cube_initial_height_non_finite": 0,
            "cube_initial_height_mismatch": 0,
        }
        if is_pick_place_task:
            reset_failure_reason_counts.update(
                {
                    "picked_not_zero": 0,
                    "carry_valid_not_zero": 0,
                    "released_not_zero": 0,
                    "lift_steps_not_zero": 0,
                    "stable_steps_not_zero": 0,
                    "previous_release_ready_not_zero": 0,
                    "pregrasp_opened_not_zero": 0,
                    "grasp_sequence_valid_not_zero": 0,
                    "reset_contact_observation_not_zero": 0,
                    "target_non_finite": 0,
                    "target_x_out_of_range": 0,
                    "target_y_out_of_range": 0,
                    "target_too_close_to_start": 0,
                }
            )
        termination_term_counts = {name: 0 for name in unwrapped.termination_manager.active_terms}
        max_abs_joint_position = float(torch.abs(robot.data.joint_pos).max().item())
        max_abs_joint_velocity = float(torch.abs(robot.data.joint_vel).max().item())
        max_abs_joint_delta = 0.0
        max_abs_processed_action_delta = 0.0
        joint_position_limit_violation_steps = 0
        joint_velocity_limit_violation_steps = 0
        non_finite_contact_steps = 0
        tabletop_penetration_steps = 0
        minimum_cube_height = float(cube.data.root_pos_w[:, 2].min().item()) if cube is not None else None
        maximum_cube_height = float(cube.data.root_pos_w[:, 2].max().item()) if cube is not None else None
        peak_filtered_contact_force_n = (
            {"fixed_finger_contact": 0.0, "moving_finger_contact": 0.0} if is_so101_task else {}
        )
        loop_started = time.perf_counter()
        for _ in range(policy_steps):
            if args.action_mode == "zero":
                actions = torch.zeros(action_shape, device=unwrapped.device)
            else:
                actions = (
                    2.0 * torch.rand(action_shape, device=unwrapped.device) - 1.0
                ) * args.random_action_amplitude
            observations, rewards, terminated, truncated, _ = env.step(actions)
            terminated_flags = terminated.detach().cpu().tolist()
            truncated_flags = truncated.detach().cpu().tolist()
            reset_mask = torch.logical_or(terminated, truncated)
            step_reset_count = count_reset_events(terminated_flags, truncated_flags)
            terminated_count += sum(bool(flag) for flag in terminated_flags)
            truncated_count += sum(bool(flag) for flag in truncated_flags)
            automatic_reset_count += step_reset_count
            if step_reset_count:
                reset_env_ids = torch.nonzero(reset_mask, as_tuple=False).squeeze(-1)
                observation_finite = per_environment_finite_mask(
                    observations,
                    unwrapped.num_envs,
                    unwrapped.device,
                )
                robot_state_finite = torch.isfinite(robot.data.joint_pos).all(dim=1) & torch.isfinite(
                    robot.data.joint_vel
                ).all(dim=1)
                reason_masks = {
                    "returned_observation_non_finite": ~observation_finite[reset_env_ids],
                    "episode_length_not_zero": unwrapped.episode_length_buf[reset_env_ids] != 0,
                    "robot_state_non_finite": ~robot_state_finite[reset_env_ids],
                }
                if is_so101_task and cube is not None:
                    cube_state_finite = torch.isfinite(cube.data.root_state_w).all(dim=1)
                    cube_initial_height = unwrapped._so101_cube_initial_z
                    initial_height_finite = torch.isfinite(cube_initial_height)
                    reason_masks.update(
                        {
                            "cube_state_non_finite": ~cube_state_finite[reset_env_ids],
                            "cube_initial_height_non_finite": ~initial_height_finite[reset_env_ids],
                            "cube_initial_height_mismatch": ~torch.isclose(
                                cube.data.root_pos_w[reset_env_ids, 2],
                                cube_initial_height[reset_env_ids],
                                atol=1.0e-5,
                                rtol=0.0,
                            ),
                        }
                    )
                if is_pick_place_task and cube is not None:
                    state = unwrapped.pick_place_state
                    target = unwrapped.target_pos_w
                    relative_target = target - unwrapped.scene.env_origins
                    x_range = pick_place_target_cfg["position_range_m"]["x"]
                    y_range = pick_place_target_cfg["position_range_m"]["y"]
                    minimum_separation = float(pick_place_target_cfg["minimum_start_distance_m"])
                    target_separation = torch.linalg.vector_norm(
                        target[:, :2] - cube.data.root_pos_w[:, :2], dim=1
                    )
                    reason_masks.update(
                        {
                            "picked_not_zero": per_environment_nonzero_mask(state.picked)[reset_env_ids],
                            "carry_valid_not_zero": per_environment_nonzero_mask(state.carry_valid)[reset_env_ids],
                            "released_not_zero": per_environment_nonzero_mask(state.released)[reset_env_ids],
                            "lift_steps_not_zero": per_environment_nonzero_mask(state.lift_steps)[reset_env_ids],
                            "stable_steps_not_zero": per_environment_nonzero_mask(state.stable_steps)[reset_env_ids],
                            "previous_release_ready_not_zero": per_environment_nonzero_mask(
                                state.previous_release_ready
                            )[reset_env_ids],
                            "pregrasp_opened_not_zero": state.pregrasp_opened[reset_env_ids],
                            "grasp_sequence_valid_not_zero": state.grasp_sequence_valid[reset_env_ids],
                            "reset_contact_observation_not_zero": observations["policy"][reset_env_ids, contact_slice].ne(0).any(dim=1),
                            "target_non_finite": ~torch.isfinite(target).all(dim=1)[reset_env_ids],
                            "target_x_out_of_range": ~(
                                (relative_target[:, 0] >= float(x_range[0]))
                                & (relative_target[:, 0] <= float(x_range[1]))
                            )[reset_env_ids],
                            "target_y_out_of_range": ~(
                                (relative_target[:, 1] >= float(y_range[0]))
                                & (relative_target[:, 1] <= float(y_range[1]))
                            )[reset_env_ids],
                            "target_too_close_to_start": (
                                target_separation < minimum_separation
                            )[reset_env_ids],
                        }
                    )
                step_reset_failures = summarize_reset_failures(
                    {
                        reason: flags.detach().cpu().tolist()
                        for reason, flags in reason_masks.items()
                    }
                )
                reset_failure_count += int(step_reset_failures["failure_count"])
                for reason, count in step_reset_failures["reason_counts"].items():
                    reset_failure_reason_counts[reason] += int(count)
            if is_so101_task:
                processed_actions = unwrapped.action_manager.get_term("joint_position_delta").processed_actions
                max_abs_processed_action_delta = max(
                    max_abs_processed_action_delta,
                    float(torch.abs(processed_actions).max().item()),
                )
            if not tensor_is_finite(observations):
                non_finite_steps += 1
            if not tensor_is_finite(rewards):
                reward_non_finite_steps += 1
            for term_name in termination_term_counts:
                termination_term_counts[term_name] += int(
                    torch.count_nonzero(unwrapped.termination_manager.get_term(term_name)).item()
                )
            max_abs_joint_position = max(max_abs_joint_position, float(torch.abs(robot.data.joint_pos).max().item()))
            max_abs_joint_velocity = max(max_abs_joint_velocity, float(torch.abs(robot.data.joint_vel).max().item()))
            position_limits = robot.data.soft_joint_pos_limits
            position_violation = torch.logical_or(
                robot.data.joint_pos < position_limits[..., 0] - 1.0e-3,
                robot.data.joint_pos > position_limits[..., 1] + 1.0e-3,
            )
            joint_position_limit_violation_steps += int(torch.any(position_violation).item())
            velocity_limits = robot.data.soft_joint_vel_limits
            velocity_violation = torch.abs(robot.data.joint_vel) > velocity_limits * 1.01
            joint_velocity_limit_violation_steps += int(torch.any(velocity_violation).item())
            max_abs_joint_delta = max(
                max_abs_joint_delta,
                float(torch.abs(robot.data.joint_pos - initial_joint_positions).max().item()),
            )
            if cube is not None:
                minimum_cube_height = min(minimum_cube_height, float(cube.data.root_pos_w[:, 2].min().item()))
                maximum_cube_height = max(maximum_cube_height, float(cube.data.root_pos_w[:, 2].max().item()))
                if is_so101_task:
                    relative_cube_positions = cube.data.root_pos_w - unwrapped.scene.env_origins
                    on_table_xy = (
                        (torch.abs(relative_cube_positions[:, 0]) <= 0.30)
                        & (relative_cube_positions[:, 1] >= -0.08)
                        & (relative_cube_positions[:, 1] <= 0.52)
                    )
                    penetrated_table = on_table_xy & (relative_cube_positions[:, 2] < 0.015)
                    tabletop_penetration_steps += int(torch.any(penetrated_table).item())
                for sensor_name in peak_filtered_contact_force_n:
                    sensor = unwrapped.scene.sensors[sensor_name]
                    if sensor.data.force_matrix_w is None or not bool(
                        torch.isfinite(sensor.data.force_matrix_w).all().item()
                    ):
                        non_finite_contact_steps += 1
                    else:
                        peak_filtered_contact_force_n[sensor_name] = max(
                            peak_filtered_contact_force_n[sensor_name],
                            float(torch.linalg.vector_norm(sensor.data.force_matrix_w, dim=-1).max().item()),
                        )

        loop_seconds = time.perf_counter() - loop_started
        completed_physics_steps = policy_steps * decimation
        unique_cube_xy = None
        unique_cube_xy_ratio = None
        cube_xy_standard_deviation = None
        cube_position_range = None
        if initial_cube_positions is not None:
            relative_positions = initial_cube_positions - unwrapped.scene.env_origins
            rounded_xy = torch.round(relative_positions[:, :2] * 10000) / 10000
            unique_cube_xy = int(torch.unique(rounded_xy, dim=0).shape[0])
            unique_cube_xy_ratio = unique_cube_xy / args.num_envs
            cube_xy_standard_deviation = relative_positions[:, :2].std(
                dim=0, unbiased=False
            ).tolist()
            cube_position_range = {
                "minimum": relative_positions.min(dim=0).values.tolist(),
                "maximum": relative_positions.max(dim=0).values.tolist(),
            }

        unique_target_xy = None
        unique_target_xy_ratio = None
        target_xy_standard_deviation = None
        target_position_range = None
        target_finite = None
        target_within_xy_range = None
        target_separated_from_cube = None
        minimum_initial_target_separation_m = None
        if initial_target_positions is not None and initial_cube_positions is not None:
            relative_targets = initial_target_positions - unwrapped.scene.env_origins
            rounded_target_xy = torch.round(relative_targets[:, :2] * 10000) / 10000
            unique_target_xy = int(torch.unique(rounded_target_xy, dim=0).shape[0])
            unique_target_xy_ratio = unique_target_xy / args.num_envs
            target_xy_standard_deviation = relative_targets[:, :2].std(
                dim=0, unbiased=False
            ).tolist()
            target_position_range = {
                "minimum": relative_targets.min(dim=0).values.tolist(),
                "maximum": relative_targets.max(dim=0).values.tolist(),
            }
            target_finite = bool(torch.isfinite(initial_target_positions).all().item())
            target_cfg = pick_place_target_cfg
            x_range = target_cfg["position_range_m"]["x"]
            y_range = target_cfg["position_range_m"]["y"]
            target_within_xy_range = bool(
                (
                    (relative_targets[:, 0] >= float(x_range[0]))
                    & (relative_targets[:, 0] <= float(x_range[1]))
                    & (relative_targets[:, 1] >= float(y_range[0]))
                    & (relative_targets[:, 1] <= float(y_range[1]))
                ).all().item()
            )
            minimum_initial_target_separation_m = float(target_cfg["minimum_start_distance_m"])
            initial_separation = torch.linalg.vector_norm(
                initial_target_positions[:, :2] - initial_cube_positions[:, :2], dim=1
            )
            target_separated_from_cube = bool(
                (initial_separation >= minimum_initial_target_separation_m).all().item()
            )

        contact_shapes = {}
        for sensor_name in ("fixed_finger_contact", "moving_finger_contact"):
            if sensor_name in unwrapped.scene.sensors:
                sensor = unwrapped.scene.sensors[sensor_name]
                force_matrix = sensor.data.force_matrix_w
                contact_shapes[sensor_name] = list(force_matrix.shape) if force_matrix is not None else None

        observation_shape = list(observations["policy"].shape) if isinstance(observations, dict) else None
        completed_step_contract = args.physics_steps <= completed_physics_steps < args.physics_steps + decimation
        runtime_checks = {
            "requested_environment_count": unwrapped.num_envs == args.num_envs,
            "requested_physics_steps": completed_step_contract,
            "finite_observations": non_finite_steps == 0,
            "finite_rewards": reward_non_finite_steps == 0,
            "joint_position_limits": joint_position_limit_violation_steps == 0,
            "joint_velocity_limits": joint_velocity_limit_violation_steps == 0,
        }
        if is_so101_task:
            runtime_checks.update(
                {
                    "physics_rate_120_hz": math.isclose(
                        float(unwrapped.physics_dt), 1.0 / physics_hz, abs_tol=1.0e-9
                    ),
                    "policy_rate_30_hz": math.isclose(
                        float(unwrapped.step_dt), 1.0 / policy_hz, abs_tol=1.0e-9
                    ),
                    "decimation_contract": decimation == expected_decimation,
                    "action_dimension": action_shape[-1] == len(joint_order),
                    "observation_dimension": (
                        observation_shape is not None
                        and observation_shape[-1] == expected_observation_dimension
                    ),
                    "action_delta_clip": max_abs_processed_action_delta <= action_scale_rad + 1.0e-6,
                    "finite_contact_sensors": non_finite_contact_steps == 0,
                    "no_tabletop_penetration": tabletop_penetration_steps == 0,
                    "independent_initial_cube_randomization": (
                        unique_cube_xy_ratio >= 0.95
                        and (
                            args.num_envs == 1
                            or min(cube_xy_standard_deviation) >= 1.0e-4
                        )
                    ),
                }
            )
        if is_pick_place_task:
            runtime_checks.update(
                {
                    "pick_place_state_zero_after_reset": initial_pick_place_state_zero is True,
                    "finite_initial_targets": target_finite is True,
                    "targets_within_contract_xy_range": target_within_xy_range is True,
                    "initial_target_separation": target_separated_from_cube is True,
                    "independent_initial_target_randomization": (
                        unique_target_xy_ratio >= 0.95
                        and (
                            args.num_envs == 1
                            or min(target_xy_standard_deviation) >= 1.0e-4
                        )
                    ),
                }
            )
        preliminary_passed = all(runtime_checks.values())
        if is_so101_task and args.num_envs == 1 and args.physics_steps >= 1000:
            gate_evaluation = {"gate": "G2", "eligible": True, "passed": None}
        elif is_so101_task and args.num_envs == 64 and args.physics_steps >= 10000:
            gate_evaluation = {"gate": "G3", "eligible": True, "passed": None}
        else:
            gate_evaluation = {"gate": None, "eligible": False, "passed": None, "reason": "bounded diagnostic"}
        payload.update(
            {
                "status": "pending_evidence" if preliminary_passed else "failed",
                "classification": "not_run" if preliminary_passed else "failed",
                "actual_num_envs": unwrapped.num_envs,
                "device": unwrapped.device,
                "observation_device": (
                    str(observations["policy"].device) if isinstance(observations, dict) else None
                ),
                "target_device": str(initial_target_positions.device) if initial_target_positions is not None else None,
                "physics_dt_seconds": float(unwrapped.physics_dt),
                "policy_dt_seconds": float(unwrapped.step_dt),
                "decimation": decimation,
                "policy_steps": policy_steps,
                "completed_physics_steps": completed_physics_steps,
                "wall_clock_seconds": loop_seconds,
                "physics_frames_per_second": completed_physics_steps / loop_seconds,
                "aggregate_sim_steps_per_second": completed_physics_steps * unwrapped.num_envs / loop_seconds,
                "observation_space": str(env.observation_space),
                "action_space": str(env.action_space),
                "observation_tensor_shape": observation_shape,
                "joint_names": list(robot.joint_names),
                "body_names": list(robot.body_names),
                "soft_joint_position_limits_rad": robot.data.soft_joint_pos_limits[0].tolist(),
                "non_finite_observation_steps": non_finite_steps,
                "non_finite_reward_steps": reward_non_finite_steps,
                "terminated_count": terminated_count,
                "truncated_count": truncated_count,
                "automatic_reset_count": automatic_reset_count,
                "reset_failure_count": reset_failure_count,
                "reset_failure_reason_counts": reset_failure_reason_counts,
                "termination_term_counts": termination_term_counts,
                "max_abs_joint_position_rad": max_abs_joint_position,
                "max_abs_joint_velocity_rad_s": max_abs_joint_velocity,
                "max_abs_joint_delta_from_initial_rad": max_abs_joint_delta,
                "max_abs_processed_action_delta_rad": max_abs_processed_action_delta,
                "joint_position_limit_violation_steps": joint_position_limit_violation_steps,
                "joint_velocity_limit_violation_steps": joint_velocity_limit_violation_steps,
                "non_finite_contact_steps": non_finite_contact_steps,
                "tabletop_penetration_steps": tabletop_penetration_steps,
                "minimum_cube_height_m": minimum_cube_height,
                "maximum_cube_height_m": maximum_cube_height,
                "peak_filtered_contact_force_n": peak_filtered_contact_force_n if cube is not None else None,
                "unique_initial_cube_xy_count": unique_cube_xy,
                "unique_initial_cube_xy_ratio": unique_cube_xy_ratio,
                "initial_cube_xy_standard_deviation_m": cube_xy_standard_deviation,
                "initial_cube_position_range_m": cube_position_range,
                "pick_place_state_zero_after_reset": initial_pick_place_state_zero,
                "unique_initial_target_xy_count": unique_target_xy,
                "unique_initial_target_xy_ratio": unique_target_xy_ratio,
                "initial_target_xy_standard_deviation_m": target_xy_standard_deviation,
                "initial_target_position_range_m": target_position_range,
                "minimum_initial_target_separation_m": minimum_initial_target_separation_m,
                "contact_force_matrix_shapes": contact_shapes,
                "runtime_checks": runtime_checks,
                "gate_evaluation": gate_evaluation,
            }
        )
        exit_code = 0 if preliminary_passed else 2
    except Exception as exc:
        payload["classification"] = "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()
        raise
    finally:
        if env is not None:
            env.close()
        payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        payload["resources"] = sampler.stop()
        payload["simulator_log"] = summarize_kit_log(kit_log_binding)
        if "runtime_checks" in payload:
            is_g3_run = args.task in (LIFT_TASK_ID, PICK_PLACE_TASK_ID) and args.num_envs == 64 and args.physics_steps >= 10000
            final_checks = finalize_smoke_checks(
                payload["runtime_checks"],
                payload["termination_term_counts"],
                is_so101_task=args.task in (LIFT_TASK_ID, PICK_PLACE_TASK_ID),
                is_g3_run=is_g3_run,
                automatic_reset_count=payload["automatic_reset_count"],
                reset_failure_count=payload["reset_failure_count"],
                resource_sample_count=payload["resources"].get("sample_count", 0),
                simulator_log_path=payload["simulator_log"].get("path"),
                simulator_error_count=payload["simulator_log"].get("error_count"),
            )
            passed = all(final_checks.values())
            payload["runtime_checks"] = final_checks
            payload["status"] = "passed" if passed else "failed"
            payload["classification"] = "runtime_validated" if passed else "failed"
            if payload["gate_evaluation"]["eligible"]:
                payload["gate_evaluation"]["passed"] = passed
            if not passed:
                failed_checks = [name for name, value in final_checks.items() if not value]
                payload.setdefault("error", f"runtime checks failed: {failed_checks}")
            exit_code = 0 if passed else 2
        write_json(output_path, payload)
        manifest_path = output_path.with_name(output_path.stem + "_manifest.json")
        manifest = {
            "schema_version": "0.1.0",
            "run_id": output_path.stem,
            "git_commit": git["commit"],
            "git_dirty": git["dirty"],
            "task_contract_sha256": task_contract_sha,
            "platform": "windows",
            "simulator": "isaac_sim_physx",
            "algorithm": "random" if args.action_mode == "random" else "scripted",
            "seed": args.seed,
            "num_envs": args.num_envs,
            "status": "completed" if payload["status"] == "passed" else "failed",
            "started_at_utc": payload["started_at_utc"],
            "finished_at_utc": payload["finished_at_utc"],
            "config_sha256": run_config_sha256(command),
            "checkpoint_uri": None,
            "checkpoint_sha256": None,
            "metrics_path": report_path(output_path),
            "notes": f"bounded smoke; requested_physics_steps={args.physics_steps}; classification={payload['classification']}",
        }
        write_json(manifest_path, manifest)
        print(f"SMOKE_REPORT={output_path}", flush=True)
        print(f"RUN_MANIFEST={manifest_path}", flush=True)
    return exit_code


try:
    raise SystemExit(main())
finally:
    if simulation_app is not None:
        simulation_app.close()

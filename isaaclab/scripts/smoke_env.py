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

from runtime_metrics import ResourceSampler, enforce_resource_guard, summarize_latest_kit_log, write_json
from so101_pick_rl.task_contract import ACTION_SCALE_RAD, DECIMATION, JOINT_ORDER, PHYSICS_HZ, POLICY_HZ, TASK_ID


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
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

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


def contract_sha256() -> str:
    completed = subprocess.run(
        [sys.executable, str(REPOSITORY_ROOT / "scripts" / "validate_contract.py")],
        check=True,
        capture_output=True,
        text=True,
    )
    return next(
        line.split("=", 1)[1] for line in completed.stdout.splitlines() if line.startswith("contract_sha256=")
    )


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


def main() -> int:
    git = git_metadata()
    task_contract_sha = contract_sha256()
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
        is_so101_task = args.task == TASK_ID
        decimation = int(unwrapped.cfg.decimation)
        policy_steps = math.ceil(args.physics_steps / decimation)
        action_shape = env.action_space.shape
        robot = unwrapped.scene["robot"]
        cube = unwrapped.scene["cube"] if "cube" in unwrapped.scene.rigid_objects else None
        initial_joint_positions = robot.data.joint_pos.clone()
        initial_cube_positions = cube.data.root_pos_w.clone() if cube is not None else None

        non_finite_steps = 0
        reward_non_finite_steps = 0
        terminated_count = 0
        truncated_count = 0
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
        peak_filtered_contact_force_n = {"fixed_finger_contact": 0.0, "moving_finger_contact": 0.0}
        loop_started = time.perf_counter()
        for _ in range(policy_steps):
            if args.action_mode == "zero":
                actions = torch.zeros(action_shape, device=unwrapped.device)
            else:
                actions = (
                    2.0 * torch.rand(action_shape, device=unwrapped.device) - 1.0
                ) * args.random_action_amplitude
            observations, rewards, terminated, truncated, _ = env.step(actions)
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
            terminated_count += int(torch.count_nonzero(terminated).item())
            truncated_count += int(torch.count_nonzero(truncated).item())
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
        cube_position_range = None
        if initial_cube_positions is not None:
            relative_positions = initial_cube_positions - unwrapped.scene.env_origins
            rounded_xy = torch.round(relative_positions[:, :2] * 10000) / 10000
            unique_cube_xy = int(torch.unique(rounded_xy, dim=0).shape[0])
            cube_position_range = {
                "minimum": relative_positions.min(dim=0).values.tolist(),
                "maximum": relative_positions.max(dim=0).values.tolist(),
            }

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
                        float(unwrapped.physics_dt), 1.0 / PHYSICS_HZ, abs_tol=1.0e-9
                    ),
                    "policy_rate_30_hz": math.isclose(
                        float(unwrapped.step_dt), 1.0 / POLICY_HZ, abs_tol=1.0e-9
                    ),
                    "decimation_contract": decimation == DECIMATION,
                    "action_dimension": action_shape[-1] == len(JOINT_ORDER),
                    "observation_dimension": observation_shape is not None and observation_shape[-1] == 22,
                    "action_delta_clip": max_abs_processed_action_delta <= ACTION_SCALE_RAD + 1.0e-6,
                    "finite_contact_sensors": non_finite_contact_steps == 0,
                    "no_tabletop_penetration": tabletop_penetration_steps == 0,
                    "independent_initial_cube_randomization": unique_cube_xy == args.num_envs,
                }
            )
        passed = all(runtime_checks.values())
        if args.num_envs == 1 and args.physics_steps >= 1000:
            gate_evaluation = {"gate": "G2", "eligible": True, "passed": passed}
        elif args.num_envs == 64 and args.physics_steps >= 10000:
            gate_evaluation = {"gate": "G3", "eligible": True, "passed": passed}
        else:
            gate_evaluation = {"gate": None, "eligible": False, "passed": None, "reason": "bounded diagnostic"}
        payload.update(
            {
                "status": "passed" if passed else "failed",
                "classification": "runtime_validated" if passed else "failed",
                "actual_num_envs": unwrapped.num_envs,
                "device": unwrapped.device,
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
                "initial_cube_position_range_m": cube_position_range,
                "contact_force_matrix_shapes": contact_shapes,
                "runtime_checks": runtime_checks,
                "gate_evaluation": gate_evaluation,
            }
        )
        exit_code = 0 if passed else 2
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
        payload["simulator_log"] = summarize_latest_kit_log(started_at.timestamp())
        if payload["resources"].get("sample_count", 0) == 0:
            payload["status"] = "failed"
            payload["classification"] = "failed"
            payload["error"] = "resource sampler produced no measurements"
            exit_code = 2
        if payload["simulator_log"].get("error_count") not in (0, None):
            payload["status"] = "failed"
            payload["classification"] = "failed"
            payload["error"] = "Isaac Sim log contains error-level messages"
            exit_code = 2
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
    simulation_app.close()

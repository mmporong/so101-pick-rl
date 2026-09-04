#!/usr/bin/env python3
"""Validate action clipping, partial-reset state, contact sensing, and success hold semantics."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ISAACLAB_PROJECT_DIR = SCRIPT_DIR.parent
REPOSITORY_ROOT = ISAACLAB_PROJECT_DIR.parent
sys.path.insert(0, str(ISAACLAB_PROJECT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from runtime_metrics import ResourceSampler, enforce_resource_guard, summarize_latest_kit_log, write_json
from so101_pick_rl.task_contract import ACTION_SCALE_RAD, POLICY_HZ, SUCCESS_HOLD_SECONDS, TASK_ID


parser = argparse.ArgumentParser(description=__doc__)
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
            "schema": "so101_pick_rl.windows_semantics.v1",
            "status": "blocked_resource_guard",
            "classification": "not_run",
            "command": command,
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
        },
    )
    print(f"SEMANTICS_REPORT={output_path}", flush=True)
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
    def run(*git_args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), *git_args],
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


def config_sha256() -> str:
    digest = hashlib.sha256()
    digest.update((REPOSITORY_ROOT / "configs" / "isaaclab" / "windows_runtime.json").read_bytes())
    digest.update(b"\0")
    digest.update(command.encode("utf-8"))
    return digest.hexdigest()


def main() -> int:
    payload: dict[str, object] = {
        "schema": "so101_pick_rl.windows_semantics.v1",
        "status": "failed",
        "classification": "not_run",
        "command": command,
        "started_at_utc": started_at.isoformat(),
        "task": TASK_ID,
        "seed": args.seed,
        "num_envs": 2,
        "preflight_resources": preflight_resources,
        "git": git_metadata(),
        "contract_sha256": contract_sha256(),
    }
    env = None
    exit_code = 1
    try:
        torch.manual_seed(args.seed)
        env_cfg = parse_env_cfg(TASK_ID, device=args.device, num_envs=2, use_fabric=True)
        env_cfg.seed = args.seed
        env = gym.make(TASK_ID, cfg=env_cfg)
        observations, _ = env.reset(seed=args.seed)
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        cube = unwrapped.scene["cube"]
        zero_actions = torch.zeros(env.action_space.shape, device=unwrapped.device)

        raw_actions = torch.full(env.action_space.shape, 10.0, device=unwrapped.device)
        observations, _, _, _, _ = env.step(raw_actions)
        action_term = unwrapped.action_manager.get_term("joint_position_delta")
        processed = action_term.processed_actions.clone()
        observed_previous_action = observations["policy"][:, 12:18]
        action_clip_check = bool(
            torch.allclose(processed, torch.full_like(processed, ACTION_SCALE_RAD), atol=1.0e-6)
            and torch.allclose(observed_previous_action, processed, atol=1.0e-6)
        )

        env.reset(seed=args.seed)
        baseline_before = unwrapped._so101_cube_initial_z.clone()
        unwrapped._so101_cube_initial_z[0] = torch.nan
        env_ids = torch.tensor([0], device=unwrapped.device, dtype=torch.long)
        unwrapped._reset_idx(env_ids)
        baseline_after = unwrapped._so101_cube_initial_z.clone()
        partial_reset_check = bool(
            torch.isfinite(baseline_after[0]).item()
            and torch.equal(baseline_after[1], baseline_before[1])
        )

        env.reset(seed=args.seed)
        fixed_body_index = robot.body_names.index("gripper")
        moving_body_index = robot.body_names.index("jaw")
        midpoint = 0.5 * (
            robot.data.body_pos_w[:, fixed_body_index] + robot.data.body_pos_w[:, moving_body_index]
        )
        cube_pose = cube.data.root_pose_w.clone()
        cube_pose[:, :3] = midpoint
        cube.write_root_pose_to_sim(cube_pose)
        cube.write_root_velocity_to_sim(torch.zeros_like(cube.data.root_vel_w))
        peak_contact_force = {"fixed_finger_contact": 0.0, "moving_finger_contact": 0.0}
        for _ in range(5):
            env.step(zero_actions)
            for sensor_name in peak_contact_force:
                force_matrix = unwrapped.scene.sensors[sensor_name].data.force_matrix_w
                if force_matrix is not None:
                    peak_contact_force[sensor_name] = max(
                        peak_contact_force[sensor_name],
                        float(torch.linalg.vector_norm(force_matrix, dim=-1).max().item()),
                    )
        contact_sensor_check = all(force > 0.2 for force in peak_contact_force.values())

        env.reset(seed=args.seed)
        required_hold_steps = int(round(SUCCESS_HOLD_SECONDS * POLICY_HZ))
        termination_steps: list[int] = []
        for step_index in range(1, required_hold_steps + 1):
            lifted_pose = cube.data.root_pose_w.clone()
            lifted_pose[:, 2] = unwrapped._so101_cube_initial_z + 0.10
            cube.write_root_pose_to_sim(lifted_pose)
            cube.write_root_velocity_to_sim(torch.zeros_like(cube.data.root_vel_w))
            _, _, terminated, _, _ = env.step(zero_actions)
            if bool(torch.any(terminated).item()):
                termination_steps.append(step_index)
        _, _, terminated_after_reset, _, _ = env.step(zero_actions)
        success_hold_check = termination_steps == [required_hold_steps] and not bool(
            torch.any(terminated_after_reset).item()
        )

        checks = {
            "raw_action_is_clipped_to_contract_delta": action_clip_check,
            "previous_action_matches_applied_delta": bool(
                torch.allclose(observed_previous_action, processed, atol=1.0e-6)
            ),
            "partial_reset_repairs_only_selected_baseline": partial_reset_check,
            "both_filtered_contact_sensors_positive": contact_sensor_check,
            "success_hold_boundary_and_reset": success_hold_check,
        }
        passed = all(checks.values())
        payload.update(
            {
                "status": "passed" if passed else "failed",
                "classification": "runtime_validated" if passed else "failed",
                "checks": checks,
                "raw_action_value": 10.0,
                "maximum_processed_action_delta_rad": float(torch.abs(processed).max().item()),
                "previous_action_observation_matches": checks["previous_action_matches_applied_delta"],
                "partial_reset_baseline_before_m": baseline_before.tolist(),
                "partial_reset_baseline_after_m": baseline_after.tolist(),
                "peak_filtered_contact_force_n": peak_contact_force,
                "required_success_hold_steps": required_hold_steps,
                "observed_success_termination_steps": termination_steps,
            }
        )
        exit_code = 0 if passed else 2
    except Exception as exc:
        payload["classification"] = "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()
        exit_code = 2
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
            "git_commit": payload["git"]["commit"],
            "git_dirty": payload["git"]["dirty"],
            "task_contract_sha256": payload["contract_sha256"],
            "platform": "windows",
            "simulator": "isaac_sim_physx",
            "algorithm": "scripted",
            "seed": args.seed,
            "num_envs": 2,
            "status": "completed" if payload["status"] == "passed" else "failed",
            "started_at_utc": payload["started_at_utc"],
            "finished_at_utc": payload["finished_at_utc"],
            "config_sha256": config_sha256(),
            "checkpoint_uri": None,
            "checkpoint_sha256": None,
            "metrics_path": output_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "notes": f"task semantics validation; classification={payload['classification']}",
        }
        write_json(manifest_path, manifest)
        print(f"SEMANTICS_REPORT={output_path}", flush=True)
        print(f"RUN_MANIFEST={manifest_path}", flush=True)
    return exit_code


try:
    raise SystemExit(main())
finally:
    simulation_app.close()

#!/usr/bin/env python3
# Training flow adapted from Isaac Lab v2.1.1 (BSD-3-Clause).
"""Train the external SO-101 task with RSL-RL and write a measured run manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ISAACLAB_PROJECT_DIR = SCRIPT_DIR.parent
REPOSITORY_ROOT = ISAACLAB_PROJECT_DIR.parent
sys.path.insert(0, str(ISAACLAB_PROJECT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from runtime_metrics import ResourceSampler, enforce_resource_guard, write_json
from so101_pick_rl.kit_log import bind_kit_log, summarize_kit_log
from so101_pick_rl.policy_evaluation import evaluate_policy
from so101_pick_rl.run_contract import (
    LIFT_TASK_ID,
    PICK_PLACE_TASK_ID,
    contract_sha256,
    load_resume_binding,
    load_spec,
    run_binding,
)


REQUIRED_ITERATION_SCALAR_TAGS = (
    "Loss/value_function",
    "Loss/surrogate",
    "Policy/mean_noise_std",
)
EPISODIC_SCALAR_TAGS = (
    "Train/mean_reward",
    "Episode_Termination/non_finite",
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="SO101-LiftCube-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_iterations", type=int, default=10)
parser.add_argument("--save_interval", type=int, default=5)
parser.add_argument("--run_name", default="smoke")
parser.add_argument("--log_root", type=Path, default=ISAACLAB_PROJECT_DIR / "logs" / "rsl_rl")
parser.add_argument("--resume_checkpoint", type=Path, default=None)
parser.add_argument("--evaluate_episodes", type=int, default=0)
parser.add_argument("--evaluation_seed", type=int, default=0)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--max_baseline_gpu_util", type=float, default=40.0)
parser.add_argument("--minimum_free_vram_mib", type=float, default=4096.0)
parser.add_argument("--max_baseline_cpu_util", type=float, default=70.0)
parser.add_argument("--video_dir", type=Path, default=None)
parser.add_argument("--video_width", type=int, default=1280)
parser.add_argument("--video_height", type=int, default=720)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video_dir is not None:
    if not args.headless or not args.enable_cameras:
        parser.error("--video_dir requires --headless --enable_cameras")
    if args.video_dir.expanduser().resolve().exists():
        parser.error("--video_dir must not already exist")
    if min(args.video_width, args.video_height) < 64 or args.video_width % 2 or args.video_height % 2:
        parser.error("Video dimensions must be even and at least 64")

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
            "schema": "so101_pick_rl.windows_ppo.v1",
            "status": "blocked_resource_guard",
            "classification": "not_run",
            "command": command,
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
        },
    )
    print(f"PPO_REPORT={output_path}", flush=True)
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
            "schema": "so101_pick_rl.windows_ppo.v1",
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
    print(f"PPO_REPORT={output_path}", flush=True)
    raise SystemExit(4)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
import so101_pick_rl.tasks  # noqa: E402, F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from isaaclab.utils.io import dump_pickle, dump_yaml  # noqa: E402


class InstrumentedRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """Accumulate training safety evidence on-device before automatic resets obscure it."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        device = self.unwrapped.device
        self._non_finite_observation_steps = torch.zeros((), dtype=torch.long, device=device)
        self._non_finite_reward_steps = torch.zeros((), dtype=torch.long, device=device)
        self._termination_counts = {
            name: torch.zeros((), dtype=torch.long, device=device)
            for name in self.unwrapped.termination_manager.active_terms
        }
        self.training_video = None

    def step(self, actions):
        observations, rewards, dones, extras = super().step(actions)
        self._non_finite_observation_steps += (~torch.isfinite(observations).all()).long()
        self._non_finite_reward_steps += (~torch.isfinite(rewards).all()).long()
        termination_manager = self.unwrapped.termination_manager
        for name, count in self._termination_counts.items():
            count += torch.count_nonzero(termination_manager.get_term(name))
        if self.training_video is not None:
            self.training_video.capture_post_step(self.unwrapped)
        return observations, rewards, dones, extras

    def runtime_snapshot(self) -> dict[str, Any]:
        return {
            "non_finite_observation_steps": int(self._non_finite_observation_steps.item()),
            "non_finite_reward_steps": int(self._non_finite_reward_steps.item()),
            "termination_term_counts": {
                name: int(count.item()) for name, count in self._termination_counts.items()
            },
        }


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def effective_config_sha256(command: str, *paths: Path) -> str:
    digest = hashlib.sha256()
    digest.update(command.encode("utf-8"))
    for path in paths:
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def report_path(path: Path) -> str:
    try:
        return path.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(path)


def all_checkpoint_tensors_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(all_checkpoint_tensors_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_checkpoint_tensors_finite(item) for item in value)
    return True


def inspect_tensorboard(log_dir: Path) -> dict[str, Any]:
    event_files = sorted(log_dir.glob("events.out.tfevents.*"))
    result: dict[str, Any] = {
        "event_files": [str(path) for path in event_files],
        "scalar_tags": [],
        "scalar_event_counts": {},
        "non_finite_scalar_count": 0,
        "last_scalars": {},
        "scalar_ranges": {},
    }
    if not event_files:
        return result
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalar_tags = accumulator.Tags().get("scalars", [])
    non_finite = 0
    last_scalars = {}
    scalar_event_counts = {}
    scalar_ranges = {}
    for tag in scalar_tags:
        events = accumulator.Scalars(tag)
        if events:
            values = [event.value for event in events]
            last_scalars[tag] = events[-1].value
            scalar_event_counts[tag] = len(events)
            finite_values = [value for value in values if math.isfinite(value)]
            if finite_values:
                scalar_ranges[tag] = {"minimum": min(finite_values), "maximum": max(finite_values)}
            non_finite += sum(not math.isfinite(value) for value in values)
    result.update(
        {
            "scalar_tags": scalar_tags,
            "scalar_event_counts": scalar_event_counts,
            "non_finite_scalar_count": non_finite,
            "last_scalars": last_scalars,
            "scalar_ranges": scalar_ranges,
        }
    )
    return result


def main() -> int:
    git = git_metadata()
    if args.evaluate_episodes < 0:
        raise ValueError("--evaluate_episodes must be non-negative")
    if args.video_dir is not None and args.task != PICK_PLACE_TASK_ID:
        raise ValueError("--video_dir is supported only for SO101-PickPlace-v0")
    load_spec(args.task)
    task_contract_sha = contract_sha256(args.task)
    fallback_config_path = REPOSITORY_ROOT / "configs" / "isaaclab" / "windows_runtime.json"
    payload: dict[str, Any] = {
        "schema": "so101_pick_rl.windows_ppo.v1",
        "status": "failed",
        "classification": "not_run",
        "command": command,
        "started_at_utc": started_at.isoformat(),
        "task": args.task,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "max_iterations": args.max_iterations,
        "preflight_resources": preflight_resources,
        "git": git,
        "contract_sha256": task_contract_sha,
    }
    env = None
    wrapped_env = None
    training_video = None
    exit_code = 1
    try:
        torch.manual_seed(args.seed)
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, use_fabric=True)
        env_cfg.seed = args.seed
        if args.video_dir is not None:
            env_cfg.viewer.resolution = (args.video_width, args.video_height)
        agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        agent_cfg.seed = args.seed
        agent_cfg.max_iterations = args.max_iterations
        agent_cfg.save_interval = args.save_interval
        agent_cfg.run_name = args.run_name
        agent_cfg.logger = "tensorboard"

        experiment_root = args.log_root.expanduser().resolve() / agent_cfg.experiment_name
        log_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.run_name:
            log_name += f"_{args.run_name}"
        log_dir = experiment_root / log_name
        log_dir.mkdir(parents=True, exist_ok=False)
        payload["log_dir"] = str(log_dir)

        env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video_dir else None)
        wrapped_env = InstrumentedRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        observation_dimension = int(wrapped_env.num_obs)
        action_dimension = int(wrapped_env.num_actions)
        binding = run_binding(args.task, observation_dimension, action_dimension)
        binding["device"] = str(wrapped_env.unwrapped.device)
        payload.update(
            {
                "observation_dimension": observation_dimension,
                "action_dimension": action_dimension,
                "device": str(wrapped_env.unwrapped.device),
                "runner_device": str(agent_cfg.device),
            }
        )
        runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
        payload["policy_device"] = str(next(runner.alg.policy.parameters()).device)
        runner.add_git_repo_to_log(str(Path(__file__).resolve()))

        resume_checkpoint = args.resume_checkpoint.expanduser().resolve() if args.resume_checkpoint else None
        start_iteration = 0
        if resume_checkpoint is not None:
            if not resume_checkpoint.is_file():
                raise FileNotFoundError(resume_checkpoint)
            resume_sidecar_path = resume_checkpoint.parent / "run_contract.json"
            resume_binding = load_resume_binding(resume_checkpoint, binding)
            resume_payload = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
            resume_from_iteration = int(resume_payload["iter"])
            runner.load(str(resume_checkpoint))
            start_iteration = resume_from_iteration + 1
            runner.current_learning_iteration = start_iteration
            payload["resume_checkpoint"] = {
                "path": str(resume_checkpoint),
                "sha256": file_sha256(resume_checkpoint),
                "iteration": resume_from_iteration,
                "run_contract_path": str(resume_sidecar_path) if resume_binding is not None else None,
            }
        payload["start_iteration"] = start_iteration
        write_json(log_dir / "run_contract.json", binding)

        params_dir = log_dir / "params"
        params_dir.mkdir(parents=True, exist_ok=True)
        dump_yaml(str(params_dir / "env.yaml"), env_cfg)
        dump_yaml(str(params_dir / "agent.yaml"), agent_cfg)
        dump_pickle(str(params_dir / "env.pkl"), env_cfg)
        dump_pickle(str(params_dir / "agent.pkl"), agent_cfg)
        payload["config_sha256"] = effective_config_sha256(
            command, params_dir / "env.yaml", params_dir / "agent.yaml"
        )

        if args.video_dir is not None:
            from so101_pick_rl.parallel_video import ParallelTrainingVideo

            training_video = ParallelTrainingVideo(
                output_dir=args.video_dir.expanduser().resolve(),
                width=args.video_width,
                height=args.video_height,
                fps=int(wrapped_env.unwrapped.pick_place_spec["control"]["policy_hz"]),
                num_envs=args.num_envs,
                num_steps_per_env=agent_cfg.num_steps_per_env,
                start_iteration=start_iteration,
                max_iterations=args.max_iterations,
                environment_origins=wrapped_env.unwrapped.scene.env_origins,
                provenance={
                    "git": git,
                    "task": args.task,
                    "contract_sha256": task_contract_sha,
                    "resume_checkpoint": payload.get("resume_checkpoint"),
                    "training_report": str(output_path),
                    "log_dir": str(log_dir),
                },
            )
            training_video.warm_up(wrapped_env.unwrapped)
            wrapped_env.training_video = training_video
            payload["parallel_training_video"] = {
                "directory": str(training_video.output_dir),
                "manifest": str(training_video.manifest_path),
                "expected_frames": training_video.expected_frames,
                "fps": training_video.fps,
                "resolution": [training_video.width, training_video.height],
                "camera_views": [training_video.overview.__dict__, training_video.detail.__dict__],
            }

        train_started = time.perf_counter()
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        training_seconds = time.perf_counter() - train_started
        if training_video is not None:
            wrapped_env.training_video = None
            training_video.close(completed=True)
            video_manifest = json.loads(training_video.manifest_path.read_text(encoding="utf-8"))
            payload["parallel_training_video"].update({
                "frames": training_video.frames,
                "status": video_manifest["status"],
                "simulation_duration_seconds": video_manifest["simulation_duration_seconds"],
                "files": video_manifest["files"],
            })
        training_runtime = wrapped_env.runtime_snapshot()

        if args.evaluate_episodes and args.task != PICK_PLACE_TASK_ID:
            raise ValueError("--evaluate_episodes is currently supported only for SO101-PickPlace-v0")
        evaluation = evaluate_policy(
            runner,
            wrapped_env,
            args.evaluate_episodes,
            seed=args.evaluation_seed,
        ) if args.task == PICK_PLACE_TASK_ID else {
            "requested_episodes": args.evaluate_episodes,
            "completed_episodes": 0,
            "successes": 0,
            "failures": 0,
            "invalid_episode_count": 0,
            "success_rate": None,
            "policy_success_claimed": False,
        }

        checkpoints = sorted(log_dir.glob("model_*.pt"), key=lambda path: path.stat().st_mtime)
        checkpoint_rows = []
        checkpoints_finite = True
        for checkpoint in checkpoints:
            checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            finite = all_checkpoint_tensors_finite(checkpoint_payload)
            checkpoints_finite &= finite
            checkpoint_rows.append(
                {
                    "path": str(checkpoint),
                    "sha256": file_sha256(checkpoint),
                    "size_bytes": checkpoint.stat().st_size,
                    "all_tensors_finite": finite,
                    "iteration": int(checkpoint_payload["iter"]),
                }
            )

        tensorboard = inspect_tensorboard(log_dir)
        checkpoint_rows.sort(key=lambda row: row["iteration"])
        final_iteration = checkpoint_rows[-1]["iteration"] if checkpoint_rows else None
        expected_final_iteration = start_iteration + args.max_iterations - 1
        required_scalar_tags_present = all(
            tag in tensorboard["scalar_tags"] for tag in REQUIRED_ITERATION_SCALAR_TAGS
        )
        required_scalar_event_counts_ok = all(
            tensorboard["scalar_event_counts"].get(tag, 0) >= args.max_iterations
            for tag in REQUIRED_ITERATION_SCALAR_TAGS
        )
        non_finite_termination_max = tensorboard["scalar_ranges"].get(
            "Episode_Termination/non_finite", {}
        ).get("maximum")
        validation_checks = {
            "checkpoints_present": bool(checkpoint_rows),
            "tensorboard_event_present": bool(tensorboard["event_files"]),
            "checkpoint_tensors_finite": checkpoints_finite,
            "tensorboard_scalars_finite": tensorboard["non_finite_scalar_count"] == 0,
            "required_scalar_tags_present": required_scalar_tags_present,
            "required_scalar_event_counts": required_scalar_event_counts_ok,
            "finite_training_observations": training_runtime["non_finite_observation_steps"] == 0,
            "finite_training_rewards": training_runtime["non_finite_reward_steps"] == 0,
            "no_non_finite_termination": (
                training_runtime["termination_term_counts"].get("non_finite") == 0
            ),
            "expected_final_iteration": final_iteration == expected_final_iteration,
            "evaluation_episode_count": (
                args.evaluate_episodes == 0
                or evaluation["completed_episodes"] == args.evaluate_episodes
            ),
        }
        preliminary_passed = all(validation_checks.values())
        gate_eligible = (
            args.task in (LIFT_TASK_ID, PICK_PLACE_TASK_ID)
            and args.num_envs == 64
            and args.max_iterations >= 10
        )
        payload.update(
            {
                "status": "pending_evidence" if preliminary_passed else "failed",
                "classification": "not_run" if preliminary_passed else "failed",
                "training_wall_clock_seconds": training_seconds,
                "transitions": args.num_envs * agent_cfg.num_steps_per_env * args.max_iterations,
                "num_steps_per_env": agent_cfg.num_steps_per_env,
                "checkpoints": checkpoint_rows,
                "tensorboard": tensorboard,
                "evaluation": evaluation,
                "training_runtime": training_runtime,
                "checkpoint_tensors_finite": checkpoints_finite,
                "required_scalar_tags": list(REQUIRED_ITERATION_SCALAR_TAGS),
                "episodic_scalar_tags": list(EPISODIC_SCALAR_TAGS),
                "episodic_scalar_tags_present": {
                    tag: tag in tensorboard["scalar_tags"] for tag in EPISODIC_SCALAR_TAGS
                },
                "required_scalar_tags_present": required_scalar_tags_present,
                "required_scalar_event_counts_ok": required_scalar_event_counts_ok,
                "non_finite_termination_max": non_finite_termination_max,
                "expected_final_iteration": expected_final_iteration,
                "final_iteration": final_iteration,
                "completed_iterations": (
                    final_iteration - start_iteration + 1 if final_iteration is not None else 0
                ),
                "validation_checks": validation_checks,
                "gate_evaluation": {
                    "gate": "G4" if gate_eligible else None,
                    "eligible": gate_eligible,
                    "passed": None,
                    "reason": None if gate_eligible else "bounded diagnostic",
                },
            }
        )
        exit_code = 0 if preliminary_passed else 2
    except Exception as exc:
        payload["classification"] = "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()
        raise
    finally:
        if training_video is not None and not training_video.closed:
            if wrapped_env is not None:
                wrapped_env.training_video = None
            try:
                training_video.close(
                    completed=False, error=payload.get("error", "training did not complete")
                )
            except Exception as video_close_exc:
                payload.setdefault(
                    "video_cleanup_error", f"{type(video_close_exc).__name__}: {video_close_exc}"
                )
        if wrapped_env is not None:
            wrapped_env.close()
        elif env is not None:
            env.close()
        payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        payload["resources"] = sampler.stop()
        payload["simulator_log"] = summarize_kit_log(kit_log_binding)
        if "validation_checks" in payload:
            validation_checks = dict(payload["validation_checks"])
            validation_checks.update(
                {
                    "resource_samples_present": payload["resources"].get("sample_count", 0) > 0,
                    "simulator_log_captured": bool(payload["simulator_log"].get("path")),
                    "simulator_error_free": payload["simulator_log"].get("error_count") == 0,
                }
            )
            passed = all(validation_checks.values())
            payload["validation_checks"] = validation_checks
            payload["status"] = "passed" if passed else "failed"
            payload["classification"] = "measured" if passed else "failed"
            if payload["gate_evaluation"]["eligible"]:
                payload["gate_evaluation"]["passed"] = passed
            if not passed:
                failed_checks = [name for name, value in validation_checks.items() if not value]
                payload.setdefault("error", f"training validation failed: {failed_checks}")
            exit_code = 0 if passed else 2
        write_json(output_path, payload)
        checkpoints = payload.get("checkpoints", [])
        final_checkpoint = checkpoints[-1] if checkpoints else None
        manifest_path = output_path.with_name(output_path.stem + "_manifest.json")
        manifest = {
            "schema_version": "0.1.0",
            "run_id": output_path.stem,
            "git_commit": git["commit"],
            "git_dirty": git["dirty"],
            "task_contract_sha256": task_contract_sha,
            "platform": "windows",
            "simulator": "isaac_sim_physx",
            "algorithm": "ppo",
            "seed": args.seed,
            "num_envs": args.num_envs,
            "status": "completed" if payload["status"] == "passed" else "failed",
            "started_at_utc": payload["started_at_utc"],
            "finished_at_utc": payload["finished_at_utc"],
            "config_sha256": payload.get("config_sha256")
            or effective_config_sha256(command, fallback_config_path),
            "checkpoint_uri": final_checkpoint["path"] if final_checkpoint else None,
            "checkpoint_sha256": final_checkpoint["sha256"] if final_checkpoint else None,
            "metrics_path": report_path(output_path),
            "notes": f"RSL-RL PPO; iterations={args.max_iterations}; classification={payload['classification']}",
        }
        write_json(manifest_path, manifest)
        print(f"PPO_REPORT={output_path}", flush=True)
        print(f"RUN_MANIFEST={manifest_path}", flush=True)
    return exit_code


try:
    raise SystemExit(main())
finally:
    if simulation_app is not None:
        simulation_app.close()

#!/usr/bin/env python3
"""Evaluate a Pick & Place checkpoint without performing optimizer training."""

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

from runtime_metrics import ResourceSampler, enforce_resource_guard, write_json
from so101_pick_rl.kit_log import bind_kit_log, summarize_kit_log
from so101_pick_rl.policy_evaluation import evaluate_policy
from so101_pick_rl.run_contract import (
    PICK_PLACE_TASK_ID,
    contract_sha256,
    load_resume_binding,
    load_spec,
    run_binding,
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=PICK_PLACE_TASK_ID)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, required=True)
parser.add_argument("--evaluation_seed", type=int, default=0)
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
            "schema": "so101_pick_rl.windows_policy_evaluation.v1",
            "status": "blocked_resource_guard",
            "classification": "not_run",
            "command": command,
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
        },
    )
    print(f"EVALUATION_REPORT={output_path}", flush=True)
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
            "schema": "so101_pick_rl.windows_policy_evaluation.v1",
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
    print(f"EVALUATION_REPORT={output_path}", flush=True)
    raise SystemExit(4)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
import so101_pick_rl.tasks  # noqa: E402, F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.utils.io import load_yaml  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def main() -> int:
    checkpoint = args.checkpoint.expanduser().resolve()
    payload = {
        "schema": "so101_pick_rl.windows_policy_evaluation.v1",
        "status": "failed",
        "classification": "not_run",
        "evaluation_scope": "diagnostic",
        "command": command,
        "started_at_utc": started_at.isoformat(),
        "task": args.task,
        "evaluation_seed": args.evaluation_seed,
        "requested_episodes": args.episodes,
        "num_envs": args.num_envs,
        "preflight_resources": preflight_resources,
        "checkpoint": {
            "path": str(checkpoint),
        },
    }
    env = None
    wrapped_env = None
    exit_code = 1
    try:
        if args.task != PICK_PLACE_TASK_ID:
            raise ValueError(f"independent evaluator supports only {PICK_PLACE_TASK_ID}")
        if args.episodes <= 0:
            raise ValueError("--episodes must be positive")
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        payload["contract_sha256"] = contract_sha256(args.task)
        spec = load_spec(args.task)
        payload["formal_evaluation"] = {
            "complete": False,
            "required_seeds": spec["evaluation"]["seeds"],
            "required_episodes_per_seed": spec["evaluation"]["episodes_per_seed"],
            "reason": "this command evaluates one checkpoint and one seed",
        }
        payload["git"] = git_metadata()
        payload["checkpoint"].update(
            {
                "sha256": file_sha256(checkpoint),
                "size_bytes": checkpoint.stat().st_size,
            }
        )
        torch.manual_seed(args.evaluation_seed)
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, use_fabric=True)
        env_cfg.seed = args.evaluation_seed
        env = gym.make(args.task, cfg=env_cfg)
        wrapped_env = RslRlVecEnvWrapper(env, clip_actions=None)
        binding = run_binding(args.task, int(wrapped_env.num_obs), int(wrapped_env.num_actions))
        sidecar_binding = load_resume_binding(checkpoint, binding)
        sidecar_path = checkpoint.parent / "run_contract.json"

        saved_agent_cfg = checkpoint.parent / "params" / "agent.yaml"
        if not saved_agent_cfg.is_file():
            raise FileNotFoundError(
                f"checkpoint is missing its saved runner configuration: {saved_agent_cfg}"
            )
        agent_cfg = load_yaml(str(saved_agent_cfg))
        clip_actions = agent_cfg.get("clip_actions")
        wrapped_env.clip_actions = clip_actions
        runner = OnPolicyRunner(wrapped_env, agent_cfg, log_dir=None, device=agent_cfg["device"])
        runner.load(str(checkpoint), load_optimizer=False)
        evaluation = evaluate_policy(
            runner,
            wrapped_env,
            episodes=args.episodes,
            seed=args.evaluation_seed,
        )
        payload.update(
            {
                "status": "pending_evidence",
                "classification": "not_run",
                "observation_dimension": int(wrapped_env.num_obs),
                "action_dimension": int(wrapped_env.num_actions),
                "environment_device": str(wrapped_env.unwrapped.device),
                "policy_device": str(next(runner.alg.policy.parameters()).device),
                "run_contract_sidecar": {
                    "path": str(sidecar_path),
                    "sha256": file_sha256(sidecar_path),
                    "binding": sidecar_binding,
                },
                "agent_config": {
                    "source": "checkpoint_params",
                    "path": str(saved_agent_cfg),
                    "sha256": file_sha256(saved_agent_cfg),
                },
                "evaluation": evaluation,
            }
        )
        exit_code = 0
    except Exception as exc:
        payload["classification"] = "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()
        raise
    finally:
        if wrapped_env is not None:
            wrapped_env.close()
        elif env is not None:
            env.close()
        payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        payload["resources"] = sampler.stop()
        payload["simulator_log"] = summarize_kit_log(kit_log_binding)
        if payload["status"] == "pending_evidence":
            checks = {
                "requested_episode_count": (
                    payload["evaluation"]["completed_episodes"] == args.episodes
                ),
                "resource_samples_present": payload["resources"].get("sample_count", 0) > 0,
                "simulator_log_captured": bool(payload["simulator_log"].get("path")),
                "simulator_error_free": payload["simulator_log"].get("error_count") == 0,
            }
            passed = all(checks.values())
            payload["validation_checks"] = checks
            payload["status"] = "passed" if passed else "failed"
            payload["classification"] = "measured_diagnostic" if passed else "failed"
            if not passed:
                payload["error"] = f"evaluation validation failed: {[k for k, v in checks.items() if not v]}"
            exit_code = 0 if passed else 2
        write_json(output_path, payload)
        print(f"EVALUATION_REPORT={output_path}", flush=True)
    return exit_code


try:
    raise SystemExit(main())
finally:
    if simulation_app is not None:
        simulation_app.close()

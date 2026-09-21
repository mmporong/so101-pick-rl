#!/usr/bin/env python3
"""Validate the separate DemoBox batch interface, not policy success."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "isaaclab"))
from so101_pick_rl.demo_action_contract import aligned_targets, normalize_targets, validate_source_timing
from so101_pick_rl.demo_bc import file_sha256, load_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=ROOT / "common/demo_box_spec.json")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, choices=(1, 64), required=True)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--episode", type=int, default=12)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.steps < 120:
        raise ValueError("smoke requires at least 120 control steps")
    contract, contract_sha = load_contract(args.contract)
    if file_sha256(args.dataset) != contract["dataset"]["expected_sha256"]:
        raise ValueError("dataset SHA mismatch")
    for relative, key in (("assets/robots/so101_follower.usd", "robot_sha256"),
                          ("assets/scenes/table_with_cube/scene.usd", "scene_sha256")):
        if file_sha256(args.snapshot / relative) != contract["scene"][key]:
            raise ValueError(f"asset SHA mismatch: {relative}")
    with h5py.File(args.dataset, "r") as dataset:
        episode = dataset[f"data/demo_{args.episode}"]
        index = int(episode.attrs["source_index"])
        validate_source_timing(json.loads(dataset[f"source_metadata/source_{index:04d}/data_attrs"].attrs["env_args"]))
        initial = {kind: {name: {field: ds[:] for field, ds in entity.items()}
                          for name, entity in entities.items()}
                   for kind, entities in episode["initial_state"].items()}
        previous = episode["obs/joint_pos_target"][0]
        q = episode["obs/joint_pos"][:]
        targets = aligned_targets(q, episode["obs/joint_pos_target"][:],
                                  episode["states/articulation/robot/joint_position"][:])
        actions = normalize_targets(targets[:args.steps], contract["action"]["lower_rad"], contract["action"]["upper_rad"])
        if len(actions) != args.steps:
            raise ValueError("episode too short for smoke")
        expected = np.concatenate((q[0], episode["obs/joint_vel"][0], previous,
                                   initial["rigid_object"]["cube"]["root_pose"][0],
                                   initial["rigid_object"]["cube"]["root_velocity"][0],
                                   initial["rigid_object"]["box_target"]["root_pose"][0, :3]))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    code_paths = [Path(__file__), ROOT / "isaaclab/so101_pick_rl/demo_box_runtime.py",
                  ROOT / "isaaclab/so101_pick_rl/demo_bc.py", ROOT / "isaaclab/so101_pick_rl/demo_action_contract.py",
                  args.contract]
    if contract["policy"].get("variant") == "previous_target_residual":
        code_paths.append(ROOT / "isaaclab/so101_pick_rl/demo_bc_residual.py")
    report = {
        "schema": "so101_pick_rl.demo_box_smoke.v1", "status": "running",
        "kind": "batched_source_command_interface_smoke", "training": False,
        "num_envs": args.num_envs, "task_contract_sha256": contract_sha,
        "dataset_sha256": contract["dataset"]["expected_sha256"],
        "robot_sha256": contract["scene"]["robot_sha256"], "scene_sha256": contract["scene"]["scene_sha256"],
        "code_sha256": {str(p.resolve().relative_to(ROOT)).replace("\\", "/"): file_sha256(p) for p in code_paths},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
        "seed": 0, "episode": args.episode, "checks": {},
        "limitations": ["identical source initialization translated across environments",
                        "interface smoke does not measure independent reset distributions or policy success",
                        "no contact-sequence success claim, no PPO updates"],
    }
    def save():
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    def check(name, value):
        report["checks"][name] = bool(value)
        if not value:
            raise AssertionError(name)
    save()
    app = runtime = None
    try:
        app = AppLauncher(args).app
        import torch
        from so101_pick_rl.demo_box_runtime import DemoBoxRuntime
        torch.manual_seed(0)
        runtime = DemoBoxRuntime(args.snapshot, args.num_envs, device=args.device, contract_path=args.contract)
        device = runtime.device
        if not str(device).startswith("cuda"):
            raise ValueError("GPU smoke requires CUDA")
        report["runtime"] = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
                             "isaaclab": importlib.metadata.version("isaaclab"), "isaacsim": "4.5.0",
                             "device": str(device)}
        torch.cuda.reset_peak_memory_stats()
        state = {kind: {name: {field: torch.as_tensor(value, device=device).repeat(args.num_envs, 1)
                               for field, value in entity.items()}
                        for name, entity in entities.items()} for kind, entities in initial.items()}
        prev = torch.as_tensor(previous, device=device).repeat(args.num_envs, 1)
        obs = runtime.reset(state, prev)
        check("observation_shape", tuple(obs.shape) == (args.num_envs, 34))
        check("observation_finite", torch.isfinite(obs).all().item())
        reset_error = float((obs - torch.as_tensor(expected, device=device)).abs().max().item())
        report["initial_observation_max_abs_error"] = reset_error
        check("source_initial_state_and_local_coordinates", reset_error < 1e-5)
        check("env_origins_distinct", len(torch.unique(runtime.scene.env_origins, dim=0)) == args.num_envs)
        check("collision_filter_enabled", runtime.scene.cfg.filter_collisions)
        check("cuda_batch", obs.is_cuda and obs.shape[0] == args.num_envs)
        before = obs.clone()
        invalid_rejected = 0
        for value in (float("nan"), 1.1):
            invalid = torch.zeros((args.num_envs, 6), device=device)
            invalid[0, 0] = value
            try:
                runtime.step(invalid)
            except ValueError:
                invalid_rejected += 1
        check("invalid_action_rejected_before_step", invalid_rejected == 2 and torch.equal(before, runtime.observe()))
        torch.cuda.synchronize()
        started = time.perf_counter()
        for command in actions:
            obs = runtime.step(torch.as_tensor(command, dtype=torch.float32, device=device).repeat(args.num_envs, 1))
            if tuple(obs.shape) != (args.num_envs, 34) or not torch.isfinite(obs).all():
                raise AssertionError("nonfinite or malformed step observation")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        check("finite_step_observations", True)
        check("previous_target_matches_executed_command", torch.allclose(obs[:, 12:18], torch.as_tensor(targets[args.steps-1], dtype=torch.float32, device=device), atol=1e-6, rtol=0))
        check("joint_motion_observed", ((obs[:, :6] - before[:, :6]).norm(dim=1) > .001).all().item())
        reset_obs = runtime.reset(state, prev)
        check("repeat_reset_matches_initial", torch.allclose(reset_obs, before, atol=1e-5, rtol=0))
        report.update(status="passed", control_steps=args.steps, transitions=args.steps * args.num_envs,
                      wall_seconds=elapsed, simulation_steps_per_second=args.steps * args.num_envs / elapsed,
                      peak_torch_vram_bytes=torch.cuda.max_memory_allocated(),
                      vram_scope="Torch allocator only; excludes PhysX/renderer")
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()
        if runtime is not None:
            runtime.close()
        if app is not None:
            app.close(wait_for_replicator=False)
    print(json.dumps({"status": report["status"], "report": str(args.output_dir / "report.json")}))


if __name__ == "__main__":
    main()

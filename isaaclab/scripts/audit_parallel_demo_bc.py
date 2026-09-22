#!/usr/bin/env python3
"""Audit selected contact traces while one BC policy drives all 64 environments."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "isaaclab"))
sys.path.insert(0, str(ROOT / "isaaclab/scripts"))

from audit_demo_bc import phase_flags  # noqa: E402
from evaluate_demo_bc import diagnostic_flags  # noqa: E402
from so101_pick_rl.demo_action_contract import validate_source_timing  # noqa: E402
from so101_pick_rl.demo_bc import build_policy, file_sha256, load_contract  # noqa: E402
from so101_pick_rl.demo_sequence import DemoSequenceGate, sequence_features  # noqa: E402
from so101_pick_rl.grasp_audit import unpack_contacts  # noqa: E402
from so101_pick_rl.demo_reset_bootstrap import controller_action, reset_bootstrap_metadata  # noqa: E402


def fixed_episode_ids() -> list[int]:
    """Return the frozen 64-reset evaluation population."""
    return list(range(32)) + list(range(400, 432))


def selected_environment_indices(selected: list[int]) -> dict[int, int]:
    """Map selected episode IDs to their original index in the fixed batch."""
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("selected episodes must be non-empty and unique")
    index_by_episode = {episode_id: index for index, episode_id in enumerate(fixed_episode_ids())}
    missing = [episode_id for episode_id in selected if episode_id not in index_by_episode]
    if missing:
        raise ValueError(f"selected episodes are outside the fixed 64-reset population: {missing}")
    return {episode_id: index_by_episode[episode_id] for episode_id in selected}


def _snapshot_paths(contract_path: Path) -> dict[str, Path]:
    paths = {
        "isaaclab/scripts/audit_parallel_demo_bc.py": Path(__file__),
        "isaaclab/scripts/audit_demo_bc.py": ROOT / "isaaclab/scripts/audit_demo_bc.py",
        "isaaclab/scripts/evaluate_demo_bc.py": ROOT / "isaaclab/scripts/evaluate_demo_bc.py",
        "isaaclab/so101_pick_rl/demo_bc.py": ROOT / "isaaclab/so101_pick_rl/demo_bc.py",
        "isaaclab/so101_pick_rl/demo_bc_residual.py": ROOT / "isaaclab/so101_pick_rl/demo_bc_residual.py",
        "isaaclab/so101_pick_rl/demo_box_runtime.py": ROOT / "isaaclab/so101_pick_rl/demo_box_runtime.py",
        "isaaclab/so101_pick_rl/demo_sequence.py": ROOT / "isaaclab/so101_pick_rl/demo_sequence.py",
        "isaaclab/so101_pick_rl/demo_action_contract.py": ROOT / "isaaclab/so101_pick_rl/demo_action_contract.py",
        "isaaclab/so101_pick_rl/demo_replay_metrics.py": ROOT / "isaaclab/so101_pick_rl/demo_replay_metrics.py",
        "isaaclab/so101_pick_rl/grasp_audit.py": ROOT / "isaaclab/so101_pick_rl/grasp_audit.py",
        "isaaclab/so101_pick_rl/demo_reset_bootstrap.py": ROOT / "isaaclab/so101_pick_rl/demo_reset_bootstrap.py",
        "configs/isaaclab/demo_box_pad_geometry.json": ROOT / "configs/isaaclab/demo_box_pad_geometry.json",
        "configs/evaluation/demo_box_replay_gate_v2.json": ROOT / "configs/evaluation/demo_box_replay_gate_v2.json",
        "contract.json": contract_path,
    }
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selected-episodes", nargs="+", type=int, default=[30, 412])
    parser.add_argument("--reset-bootstrap", choices=("none", "home-open-one-step"), default="none")
    help_requested = any(argument in ("-h", "--help") for argument in sys.argv[1:])
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    if help_requested:
        parser.print_help()
        return 0
    args = parser.parse_args()

    episode_ids = fixed_episode_ids()
    selected_indices = selected_environment_indices(args.selected_episodes)
    contract_path = args.contract.expanduser().resolve()
    dataset_path = args.dataset.expanduser().resolve()
    snapshot_path = args.snapshot.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    training_report_path = args.training_report.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse audit output directory: {output_dir}")

    geometry_path = ROOT / "configs/isaaclab/demo_box_pad_geometry.json"
    gate_path = ROOT / "configs/evaluation/demo_box_replay_gate_v2.json"
    contract, contract_sha = load_contract(contract_path)
    if contract["policy"].get("variant") != "previous_target_residual":
        raise ValueError("parallel audit requires the previous-target residual policy contract")
    dataset_sha = file_sha256(dataset_path)
    if dataset_sha != contract["dataset"]["expected_sha256"]:
        raise ValueError("dataset SHA-256 mismatch")
    asset_paths = {
        "robot": snapshot_path / "assets/robots/so101_follower.usd",
        "scene": snapshot_path / "assets/scenes/table_with_cube/scene.usd",
    }
    for name, path in asset_paths.items():
        expected = contract["scene"][f"{name}_sha256"]
        if file_sha256(path) != expected:
            raise ValueError(f"{name} asset SHA-256 mismatch")
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    gate_cfg = json.loads(gate_path.read_text(encoding="utf-8"))
    if geometry.get("valid_for") != contract["scene"]["robot_sha256"]:
        raise ValueError("contact geometry is not bound to the contracted robot")

    checkpoint_sha = file_sha256(checkpoint_path)
    training = json.loads(training_report_path.read_text(encoding="utf-8"))
    if (training.get("status") != "passed"
            or training.get("checkpoint", {}).get("sha256") != checkpoint_sha
            or training.get("contract", {}).get("sha256") != contract_sha
            or training.get("dataset", {}).get("sha256") != dataset_sha
            or training.get("policy_variant") != "previous_target_residual"):
        raise ValueError("checkpoint, training report, contract, or dataset provenance mismatch")

    states: list[dict] = []
    targets: list[np.ndarray] = []
    source_indices: list[int] = []
    with h5py.File(dataset_path, "r") as dataset:
        for episode_id in episode_ids:
            group = dataset[f"data/demo_{episode_id}"]
            source_index = int(group.attrs["source_index"])
            metadata = dataset[f"source_metadata/source_{source_index:04d}/data_attrs"].attrs["env_args"]
            validate_source_timing(json.loads(metadata))
            states.append({
                kind: {
                    name: {field: value[:] for field, value in entity.items()}
                    for name, entity in entities.items()
                }
                for kind, entities in group["initial_state"].items()
            })
            targets.append(group["obs/joint_pos_target"][0])
            source_indices.append(source_index)

    output_dir.mkdir(parents=True, exist_ok=False)
    code_dir = output_dir / "code"
    code_dir.mkdir()
    code_sha256: dict[str, str] = {}
    for logical_path, source_path in _snapshot_paths(contract_path).items():
        destination = code_dir / logical_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)
        code_sha256[logical_path] = file_sha256(source_path)

    report = {
        "schema": "so101_pick_rl.demo_bc_parallel_contact_audit.v1",
        "status": "running",
        "classification": "batched_instrumentation_diagnostic_not_policy_success",
        "num_envs": 64,
        "control_steps": 900,
        "episodes": episode_ids,
        "source_indices": source_indices,
        "selected_episode_to_environment": {str(key): value for key, value in selected_indices.items()},
        "training": False,
        "ppo_updates": 0,
        "hardware_validated": False,
        "task_contract_sha256": contract_sha,
        "dataset_sha256": dataset_sha,
        "checkpoint_sha256": checkpoint_sha,
        "training_report_sha256": file_sha256(training_report_path),
        "robot_sha256": contract["scene"]["robot_sha256"],
        "scene_sha256": contract["scene"]["scene_sha256"],
        "geometry_sha256": file_sha256(geometry_path),
        "sequence_gate_sha256": file_sha256(gate_path),
        "code_sha256": code_sha256,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
        "state_write_scope": "one_initial_reset_for_all_64_environments",
        "reset_bootstrap": reset_bootstrap_metadata(args.reset_bootstrap),
        "limitations": [
            "selected traces instrument only their original environment indices inside the same 64-environment rollout",
            "the result discriminates instrumentation or batching sensitivity but does not infer that a one-environment failure invalidates 64 environments",
            "source release and lift predicates are diagnostics; only the selected contact traces run the full sequence gate",
            "completion is not a policy-success claim",
        ],
        "selected_results": [],
    }

    def save_report() -> None:
        (output_dir / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    save_report()
    app = runtime = None
    trace_streams = {}
    try:
        app = AppLauncher(args).app
        import torch
        from pxr import Usd, UsdPhysics
        from so101_pick_rl.demo_box_runtime import DemoBoxRuntime

        torch.manual_seed(0)
        runtime = DemoBoxRuntime(
            snapshot_path, 64, args.device, contract_path=contract_path, enable_contact_sensors=True
        )
        if runtime.device.type != "cuda":
            raise ValueError("parallel contact audit requires a CUDA runtime")
        checkpoint = torch.load(checkpoint_path, map_location=runtime.device, weights_only=True)
        if (checkpoint.get("contract_sha256") != contract_sha
                or checkpoint.get("dataset_sha256") != dataset_sha
                or checkpoint.get("policy_variant") != "previous_target_residual"):
            raise ValueError("embedded checkpoint provenance mismatch")
        mean = checkpoint.get("observation_mean")
        std = checkpoint.get("observation_std")
        if (not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor)
                or mean.shape != (34,) or std.shape != (34,)
                or not torch.isfinite(mean).all() or not torch.isfinite(std).all()
                or not (std > 0).all()):
            raise ValueError("invalid checkpoint observation normalizer")
        policy = build_policy(contract, mean, std).to(runtime.device).eval()
        policy.load_state_dict(checkpoint["model_state_dict"])
        if (not torch.equal(policy.observation_mean, mean)
                or not torch.equal(policy.observation_std, std)):
            raise ValueError("checkpoint normalizer copies disagree")

        state = {
            kind: {
                name: {
                    field: torch.as_tensor(
                        np.concatenate([item[kind][name][field] for item in states]),
                        device=runtime.device,
                    )
                    for field in entity
                }
                for name, entity in entities.items()
            }
            for kind, entities in states[0].items()
        }
        previous = torch.as_tensor(np.stack(targets), dtype=torch.float32, device=runtime.device)
        observation = runtime.reset(state, previous)
        initial_z = observation[:, 20].clone()

        selected_runtime = {}
        for episode_id, env_index in selected_indices.items():
            root = f"/World/envs/env_{env_index}"
            cube_path = runtime.source_rigid_paths["cube"].replace("{ENV_REGEX_NS}", root)
            table_paths = [
                str(prim.GetPath())
                for prim in Usd.PrimRange(runtime.sim.stage.GetPrimAtPath(root + "/Scene"))
                if prim.HasAPI(UsdPhysics.CollisionAPI)
                and not str(prim.GetPath()).startswith(cube_path)
            ]
            if not table_paths:
                raise ValueError(f"table collision paths unresolved for episode {episode_id}")
            walls = [root + "/BoxTarget"] + [
                root + "/Box" + name for name in ("Left", "Right", "Front", "Back")
            ]
            views = {}
            view_specs = (
                ("gripper_cube", root + "/Robot/gripper", [cube_path]),
                ("jaw_cube", root + "/Robot/jaw", [cube_path]),
                ("gripper_table", root + "/Robot/gripper", table_paths),
                ("jaw_table", root + "/Robot/jaw", table_paths),
                ("gripper_box", root + "/Robot/gripper", walls),
                ("jaw_box", root + "/Robot/jaw", walls),
                ("cube_box", cube_path, [root + "/BoxTarget"]),
            )
            for name, body_path, filters in view_specs:
                view = runtime.sim.physics_sim_view.create_rigid_contact_view(
                    body_path, filter_patterns=filters, max_contact_data_count=4096
                )
                if view.sensor_count != 1 or view.filter_count < 1:
                    raise ValueError(f"unresolved contact view for episode {episode_id}: {name}")
                views[name] = view
            initial_cube = runtime.cube.data.root_pos_w[env_index].cpu().numpy().copy()
            selected_runtime[episode_id] = {
                "env_index": env_index,
                "views": views,
                "gate": DemoSequenceGate(gate_cfg, initial_cube),
                "counts": dict(
                    near=0, near_open=0, bilateral=0, opposing=0,
                    valid_grasp=0, valid_lift=0, penetration_unsafe=0,
                ),
                "first": {},
                "minimum_distance_m": float("inf"),
                "minimum_gap_m": float("inf"),
                "maximum_lift_m": 0.0,
            }
            trace_streams[episode_id] = (output_dir / f"demo_{episode_id}_trace.jsonl").open(
                "x", encoding="utf-8"
            )

        peak_lift = torch.zeros(64, device=runtime.device)
        lift_run = torch.zeros(64, dtype=torch.int64, device=runtime.device)
        release_run = torch.zeros_like(lift_run)
        longest_lift = torch.zeros_like(lift_run)
        longest_release = torch.zeros_like(lift_run)
        max_normalized_observation = torch.zeros(64, device=runtime.device)
        torch.cuda.reset_peak_memory_stats(runtime.device)
        torch.cuda.synchronize(runtime.device)
        started = time.perf_counter()
        with torch.inference_mode():
            for step in range(900):
                prestep = observation.clone()
                normalized = (observation - mean) / std
                max_normalized_observation = torch.maximum(
                    max_normalized_observation, normalized.abs().max(dim=1).values
                )
                action = controller_action(
                    args.reset_bootstrap, step, observation, mean, std, policy,
                    contract["action"]["lower_rad"], contract["action"]["upper_rad"],
                )
                observation = runtime.step(action)
                lift, release = diagnostic_flags(observation, initial_z)
                peak_lift = torch.maximum(peak_lift, lift)
                lift_run = torch.where(lift >= 0.08, lift_run + 1, 0)
                release_run = torch.where(release, release_run + 1, 0)
                longest_lift = torch.maximum(longest_lift, lift_run)
                longest_release = torch.maximum(longest_release, release_run)

                for episode_id, selected in selected_runtime.items():
                    env_index = selected["env_index"]
                    contacts = {}
                    for name, view in selected["views"].items():
                        buffers = [value.cpu().numpy().copy() for value in view.get_contact_data(runtime.dt)]
                        if int(buffers[4].sum()) >= view.max_contact_data_count:
                            raise ValueError(
                                f"contact buffer may be truncated for episode {episode_id}: {name}"
                            )
                        contacts[name] = unpack_contacts(buffers)
                    finger_ids = [runtime.robot.body_names.index(name) for name in ("gripper", "jaw")]
                    record = {
                        "step": step + 1,
                        "environment_index": env_index,
                        "pre_step_state": {
                            "joint_position_rad": prestep[env_index, :6].cpu().tolist(),
                            "previous_target_rad": prestep[env_index, 12:18].cpu().tolist(),
                            "box_position_w_m": runtime.box.data.root_pos_w[env_index].cpu().tolist(),
                        },
                        "joint_pos": observation[env_index, :6].cpu().tolist(),
                        "executed_target": observation[env_index, 12:18].cpu().tolist(),
                        "contacts": contacts,
                        "finger_body_position_w_m": runtime.robot.data.body_pos_w[
                            env_index, finger_ids
                        ].cpu().tolist(),
                        "finger_body_quaternion_wxyz": runtime.robot.data.body_quat_w[
                            env_index, finger_ids
                        ].cpu().tolist(),
                        "cube_pose": runtime.cube.data.root_state_w[env_index, :7].cpu().tolist(),
                        "cube_linear_velocity": runtime.cube.data.root_lin_vel_w[env_index].cpu().tolist(),
                        "cube_angular_velocity": runtime.cube.data.root_ang_vel_w[env_index].cpu().tolist(),
                        "lift_m": float(lift[env_index].item()),
                    }
                    feature = sequence_features(record, geometry, gate_cfg)
                    selected["gate"].update(feature)
                    flags = phase_flags(feature, gate_cfg)
                    for name, flag in flags.items():
                        if flag:
                            selected["counts"][name] += 1
                            selected["first"].setdefault(name, step + 1)
                    selected["minimum_distance_m"] = min(
                        selected["minimum_distance_m"], feature["distance_m"]
                    )
                    selected["minimum_gap_m"] = min(selected["minimum_gap_m"], feature["gap_m"])
                    selected["maximum_lift_m"] = max(
                        selected["maximum_lift_m"], feature["lift_m"]
                    )
                    record["phase_flags"] = flags
                    record["distance_m"] = feature["distance_m"]
                    record["gap_m"] = feature["gap_m"]
                    record["sequence_gate"] = selected["gate"].report()
                    trace_streams[episode_id].write(json.dumps(record, allow_nan=False) + "\n")
        torch.cuda.synchronize(runtime.device)
        elapsed = time.perf_counter() - started

        results = []
        for index, episode_id in enumerate(episode_ids):
            results.append({
                "episode": episode_id,
                "environment_index": index,
                "source_index": source_indices[index],
                "split": "train_shard_reset" if source_indices[index] < 16 else "validation_shard_reset",
                "maximum_lift_m": float(peak_lift[index].item()),
                "lift_hold_steps": int(longest_lift[index].item()),
                "source_release_hold_steps": int(longest_release[index].item()),
                "lift_8cm_0_2s": bool(longest_lift[index] >= 12),
                "source_stable_release": bool(longest_release[index] >= 30),
                "maximum_abs_normalized_observation": float(max_normalized_observation[index].item()),
            })
        for episode_id, selected in selected_runtime.items():
            trace_streams[episode_id].close()
            del trace_streams[episode_id]
            trace_path = output_dir / f"demo_{episode_id}_trace.jsonl"
            report["selected_results"].append({
                "episode": episode_id,
                "environment_index": selected["env_index"],
                "source_index": source_indices[selected["env_index"]],
                "phase_steps": selected["counts"],
                "first_phase_step": selected["first"],
                "minimum_pad_cube_distance_m": selected["minimum_distance_m"],
                "minimum_gap_m": selected["minimum_gap_m"],
                "maximum_lift_m": selected["maximum_lift_m"],
                "sequence_gate": selected["gate"].report(),
                "trace_sha256": file_sha256(trace_path),
            })
        report["results"] = results
        report["summary"] = {
            split: {
                "episodes": len(rows),
                "lift_8cm_0_2s": sum(row["lift_8cm_0_2s"] for row in rows),
                "source_stable_release": sum(row["source_stable_release"] for row in rows),
            }
            for split in ("train_shard_reset", "validation_shard_reset")
            for rows in ([row for row in results if row["split"] == split],)
        }
        report.update(
            status="completed",
            wall_seconds=elapsed,
            transitions=64 * 900,
            simulation_steps_per_second=64 * 900 / elapsed,
            peak_torch_vram_bytes=torch.cuda.max_memory_allocated(runtime.device),
            vram_scope="Torch allocator only; excludes PhysX and renderer",
            runtime={
                "gpu": torch.cuda.get_device_name(runtime.device),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "isaaclab": importlib.metadata.version("isaaclab"),
                "isaacsim": "4.5.0",
                "device": str(runtime.device),
                "dt": runtime.dt,
                "seed": 0,
            },
        )
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for stream in trace_streams.values():
            stream.close()
        save_report()
        if runtime is not None:
            runtime.close()
        if app is not None:
            app.close(wait_for_replicator=False)
    print(json.dumps({"status": report["status"], "summary": report["summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

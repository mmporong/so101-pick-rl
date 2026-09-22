#!/usr/bin/env python3
"""Single-env phase/contact audit of BC or recorded commands, without state corrections."""
from __future__ import annotations

import argparse
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
from so101_pick_rl.demo_bc import load_contract, file_sha256, build_policy
from so101_pick_rl.demo_action_contract import aligned_targets, normalize_targets, validate_source_timing
from so101_pick_rl.demo_sequence import DemoSequenceGate, sequence_features
from so101_pick_rl.grasp_audit import unpack_contacts
from so101_pick_rl.demo_reset_bootstrap import (
    controller_action, reset_bootstrap_metadata, validate_reset_bootstrap,
)


def phase_flags(feature, cfg):
    """Instantaneous phase diagnostics; sequence validity remains a separate gate."""
    near = feature["distance_m"] <= cfg["pregrasp_maximum_distance_m"]
    grasp = (near and feature["bilateral"] and feature["opposing_sides"] and feature["inside_pad_planes"]
             and feature["penetration_safe"] and feature["approach_world_z"] < 0)
    return {"near": near, "near_open": near and feature["opened"], "bilateral": feature["bilateral"],
            "opposing": feature["opposing_sides"], "valid_grasp": grasp,
            "valid_lift": grasp and feature["lift_m"] >= cfg["minimum_lift_m"],
            "penetration_unsafe": not feature["penetration_safe"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--controller", choices=("bc", "source"), default="bc")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--contract", type=Path, default=ROOT / "common/demo_box_spec.json")
    parser.add_argument("--episodes", nargs="+", type=int, default=[1, 12, 400])
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reset-bootstrap", choices=("none", "home-open-one-step"), default="none")
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    validate_reset_bootstrap(args.reset_bootstrap, args.controller)
    if args.steps < 1 or len(set(args.episodes)) != len(args.episodes) or min(args.episodes) < 0:
        raise ValueError("invalid steps or episode selection")
    contract_path = args.contract
    geometry_path = ROOT / "configs/isaaclab/demo_box_pad_geometry.json"
    gate_path = ROOT / "configs/evaluation/demo_box_replay_gate_v2.json"
    contract, digest = load_contract(contract_path)
    geometry = json.loads(geometry_path.read_text())
    cfg = json.loads(gate_path.read_text())
    if file_sha256(args.dataset) != contract["dataset"]["expected_sha256"]:
        raise ValueError("dataset SHA mismatch")
    if geometry["valid_for"] != contract["scene"]["robot_sha256"]:
        raise ValueError("geometry binding mismatch")
    checkpoint_sha = None
    if args.controller == "bc":
        if not args.checkpoint or not args.training_report:
            raise ValueError("BC audit requires checkpoint and training report")
        checkpoint_sha = file_sha256(args.checkpoint)
        training = json.loads(args.training_report.read_text())
        if training["status"] != "passed" or training["checkpoint"]["sha256"] != checkpoint_sha or training["contract"]["sha256"] != digest:
            raise ValueError("checkpoint/training provenance mismatch")
    with h5py.File(args.dataset, "r") as dataset:
        for index in args.episodes:
            source = int(dataset[f"data/demo_{index}"].attrs["source_index"])
            validate_source_timing(json.loads(dataset[f"source_metadata/source_{source:04d}/data_attrs"].attrs["env_args"]))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"schema": "so101_pick_rl.demo_bc_contact_audit.v1", "status": "running",
              "controller": args.controller, "checkpoint_sha256": checkpoint_sha,
              "task_contract_sha256": digest, "dataset_sha256": contract["dataset"]["expected_sha256"],
              "episodes": args.episodes, "steps_requested": args.steps, "num_envs": 1,
              "training": False, "hardware_validated": False, "results": [],
              "reset_bootstrap": reset_bootstrap_metadata(args.reset_bootstrap, args.controller),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))}
    code = args.output_dir / "code"
    code.mkdir()
    paths = [Path(__file__), contract_path, geometry_path, gate_path] + [ROOT / "isaaclab/so101_pick_rl" / name for name in
            ("demo_bc.py", "demo_box_runtime.py", "demo_sequence.py", "demo_action_contract.py", "demo_replay_metrics.py", "grasp_audit.py",
             "demo_reset_bootstrap.py")]
    if contract["policy"].get("variant") == "previous_target_residual":
        paths.append(ROOT / "isaaclab/so101_pick_rl/demo_bc_residual.py")
    report["code_sha256"] = {}
    for path in paths:
        shutil.copyfile(path, code / path.name)
        report["code_sha256"][path.name] = file_sha256(path)
    def save():
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    save()
    app = runtime = None
    try:
        app = AppLauncher(args).app
        import torch
        from pxr import Usd, UsdPhysics
        from so101_pick_rl.demo_box_runtime import DemoBoxRuntime
        runtime = DemoBoxRuntime(args.snapshot, 1, args.device, contract_path=contract_path, enable_contact_sensors=True)
        robot, cube, box = runtime.robot, runtime.cube, runtime.box
        finger_ids = [robot.body_names.index(name) for name in ("gripper", "jaw")]
        root = "/World/envs/env_0"
        cube_path = runtime.source_rigid_paths["cube"].replace("{ENV_REGEX_NS}", root)
        table = [str(p.GetPath()) for p in Usd.PrimRange(runtime.sim.stage.GetPrimAtPath(root + "/Scene"))
                 if p.HasAPI(UsdPhysics.CollisionAPI) and not str(p.GetPath()).startswith(cube_path)]
        if not table:
            raise ValueError("table collision filters unresolved")
        walls = [root + "/BoxTarget"] + [root + "/Box" + name for name in ("Left", "Right", "Front", "Back")]
        views = {}
        for name, body, filters in (("gripper_cube", "gripper", [cube_path]), ("jaw_cube", "jaw", [cube_path]),
                                    ("gripper_table", "gripper", table), ("jaw_table", "jaw", table),
                                    ("gripper_box", "gripper", walls), ("jaw_box", "jaw", walls),
                                    ("cube_box", None, [root + "/BoxTarget"])):
            view = runtime.sim.physics_sim_view.create_rigid_contact_view(
                root + "/Robot/" + body if body else cube_path,
                filter_patterns=filters, max_contact_data_count=4096)
            if view.sensor_count != 1 or view.filter_count < 1:
                raise ValueError(f"unresolved contact view: {name}")
            views[name] = view
        if args.controller == "bc":
            ck = torch.load(args.checkpoint, map_location=runtime.device, weights_only=True)
            if ck["contract_sha256"] != digest or ck["dataset_sha256"] != report["dataset_sha256"]:
                raise ValueError("embedded checkpoint provenance mismatch")
            mean, std = ck["observation_mean"], ck["observation_std"]
            model = build_policy(contract, mean, std).to(runtime.device).eval()
            model.load_state_dict(ck["model_state_dict"])
        report["runtime"] = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "dt": runtime.dt}
        with h5py.File(args.dataset, "r") as dataset, torch.inference_mode():
            for index in args.episodes:
                group = dataset[f"data/demo_{index}"]
                state = {kind: {name: {field: torch.as_tensor(ds[:], device=runtime.device) for field, ds in entity.items()}
                                for name, entity in entities.items()} for kind, entities in group["initial_state"].items()}
                previous = torch.as_tensor(group["obs/joint_pos_target"][0:1], device=runtime.device)
                obs = runtime.reset(state, previous)
                initial_cube = cube.data.root_pos_w[0].cpu().numpy().copy()
                gate = DemoSequenceGate(cfg, initial_cube)
                actions = None
                if args.controller == "source":
                    targets = aligned_targets(group["obs/joint_pos"][:], group["obs/joint_pos_target"][:], group["states/articulation/robot/joint_position"][:])
                    actions = normalize_targets(targets, contract["action"]["lower_rad"], contract["action"]["upper_rad"])
                steps = min(args.steps, len(actions)) if actions is not None else args.steps
                counts = dict(near=0, near_open=0, bilateral=0, opposing=0, valid_grasp=0, valid_lift=0, penetration_unsafe=0)
                first = {}
                min_distance, max_lift, min_gap = float("inf"), 0., float("inf")
                started = time.perf_counter()
                trace_path = args.output_dir / f"demo_{index}_trace.jsonl"
                with trace_path.open("x", encoding="utf-8") as trace:
                    for step in range(steps):
                        pre = {"joint_position_rad": obs[0, :6].cpu().tolist(), "previous_target_rad": obs[0, 12:18].cpu().tolist(),
                               "box_position_w_m": box.data.root_pos_w[0].cpu().tolist()}
                        if actions is None:
                            action = controller_action(
                                args.reset_bootstrap, step, obs, mean, std, model,
                                contract["action"]["lower_rad"], contract["action"]["upper_rad"],
                            )
                        else:
                            action = torch.as_tensor(actions[step:step+1], dtype=torch.float32, device=runtime.device)
                        obs = runtime.step(action)
                        contacts = {}
                        for name, view in views.items():
                            buffers = [x.cpu().numpy().copy() for x in view.get_contact_data(runtime.dt)]
                            if int(buffers[4].sum()) >= view.max_contact_data_count:
                                raise ValueError("contact buffer may be truncated")
                            contacts[name] = unpack_contacts(buffers)
                        record = {"step": step + 1, "pre_step_state": pre, "joint_pos": obs[0, :6].cpu().tolist(),
                                  "executed_target": obs[0, 12:18].cpu().tolist(), "contacts": contacts,
                                  "finger_body_position_w_m": robot.data.body_pos_w[0, finger_ids].cpu().tolist(),
                                  "finger_body_quaternion_wxyz": robot.data.body_quat_w[0, finger_ids].cpu().tolist(),
                                  "cube_pose": cube.data.root_state_w[0, :7].cpu().tolist(),
                                  "cube_linear_velocity": cube.data.root_lin_vel_w[0].cpu().tolist(),
                                  "cube_angular_velocity": cube.data.root_ang_vel_w[0].cpu().tolist(),
                                  "lift_m": float(cube.data.root_pos_w[0, 2].item() - initial_cube[2])}
                        feature = sequence_features(record, geometry, cfg)
                        gate.update(feature)
                        flags = phase_flags(feature, cfg)
                        for name, flag in flags.items():
                            if flag:
                                counts[name] += 1
                                first.setdefault(name, step + 1)
                        min_distance = min(min_distance, feature["distance_m"])
                        min_gap = min(min_gap, feature["gap_m"])
                        max_lift = max(max_lift, feature["lift_m"])
                        record["phase_flags"] = flags
                        record["distance_m"] = feature["distance_m"]
                        record["gap_m"] = feature["gap_m"]
                        record["sequence_gate"] = gate.report()
                        trace.write(json.dumps(record, allow_nan=False) + "\n")
                report["results"].append({"episode": index, "source_index": int(group.attrs["source_index"]), "steps": steps,
                                          "phase_steps": counts, "first_phase_step": first, "minimum_pad_cube_distance_m": min_distance,
                                          "minimum_gap_m": min_gap, "maximum_lift_m": max_lift,
                                          "sequence_gate": gate.report(), "trace_sha256": file_sha256(trace_path),
                                          "wall_seconds": time.perf_counter() - started})
                save()
        report["status"] = "completed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()
        if runtime is not None:
            runtime.close()
        if app is not None:
            app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()

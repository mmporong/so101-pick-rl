"""Reset-assisted empty-hand placement-pose probe; never a task success or training demonstration."""
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
from so101_pick_rl.demo_bc import load_contract, file_sha256
from so101_pick_rl.demo_pose_probe import opening_schedule, contact_summary, validate_candidate_names, gravity_compensated_target, validate_pad_geometry
from so101_pick_rl.grasp_audit import unpack_contacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=ROOT / "common/demo_box_spec.json")
    parser.add_argument("--gravity-target-compensation", action="store_true",
                        help="offset arm position commands by model gravity / unchanged PD stiffness")
    parser.add_argument("--probe-rejected-attempts", type=int, default=0,
                        help="diagnose up to three rejected search attempts; never treats them as geometry passes")
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    contract, contract_sha = load_contract(args.contract)
    candidates = json.loads(args.candidates.read_text())
    generators = {"verified_point_kinematics_only": "verify_demo_workspace_candidates.py",
                  "sampled_geometry_only": "search_demo_place_geometry.py",
                  "convex_geometry_only": "search_demo_place_geometry.py"}
    generator = generators.get(candidates.get("status"))
    if (generator is None
            or candidates.get("robot_sha256") != contract["scene"]["robot_sha256"]
            or candidates.get("dataset_sha256") != file_sha256(args.dataset)
            or candidates["dataset_sha256"] != contract["dataset"]["expected_sha256"]
            or (generator == "search_demo_place_geometry.py" and candidates.get("task_contract_sha256") != contract_sha)
            or candidates.get("script_sha256") != file_sha256(ROOT / "isaaclab/scripts" / generator)):
        raise ValueError("candidate provenance mismatch")
    if not 0 <= args.probe_rejected_attempts <= 3:
        raise ValueError("rejected-attempt diagnostic is bounded to three poses")
    selected = (candidates.get("best_attempts", [])[:args.probe_rejected_attempts]
                if args.probe_rejected_attempts else candidates["candidates"])
    validate_candidate_names(selected)
    gate_path = ROOT / "configs/evaluation/demo_box_replay_gate_v2.json"
    cfg = json.loads(gate_path.read_text())
    pad_path = ROOT / "configs/isaaclab/demo_box_pad_geometry.json"
    pad_geometry = json.loads(pad_path.read_text())
    validate_pad_geometry(pad_geometry, contract["scene"]["robot_sha256"])
    if generator == "search_demo_place_geometry.py" and (
            candidates.get("gate_sha256") != file_sha256(gate_path)
            or candidates.get("pad_geometry_sha256") != file_sha256(pad_path)
            or candidates.get("helper_sha256") != file_sha256(ROOT / "isaaclab/so101_pick_rl/demo_pose_probe.py")):
        raise ValueError("candidate geometry or gate provenance mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"schema": "so101_pick_rl.demo_place_pose_probe.v1", "status": "running",
              "kind": "reset_assisted_empty_hand_static_pose_and_opening_probe",
              "training": False, "task_success_claimed": False, "num_envs": 1,
              "gravity_target_compensation": args.gravity_target_compensation,
              "open_target_rad": candidates.get("search", {}).get("open_rad", 1.35),
              "input_candidate_selection": "rejected_search_attempts_not_geometry_pass" if args.probe_rejected_attempts else "search_candidates_not_physics_certificates",
              "task_contract_sha256": contract_sha, "dataset_sha256": candidates["dataset_sha256"],
              "candidates_sha256": file_sha256(args.candidates), "gate_sha256": file_sha256(gate_path),
              "pad_geometry_sha256": file_sha256(pad_path),
              "physical_state_writes": "initial reset for each independent pose; cube remains at original table reset",
              "limitations": ["No approach path, grasp, carried cube, supported placement, or retreat validation",
                              "Contact response may move the reset pose; joint tracking is recorded",
                              "Fixed base/table contacts reported but excluded from moving-link diagnostic"],
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
              "results": [], "code_sha256": {}}
    code = args.output_dir / "code"
    code.mkdir()
    for path in [Path(__file__), args.contract, gate_path, pad_path, ROOT / "isaaclab/scripts" / generator,
                 *(ROOT / "isaaclab/so101_pick_rl" / n for n in
                   ("demo_box_runtime.py", "demo_pose_probe.py", "demo_bc.py", "demo_action_contract.py", "grasp_audit.py", "demo_sequence.py", "demo_replay_metrics.py"))]:
        shutil.copyfile(path, code / path.name)
        report["code_sha256"][path.name] = file_sha256(path)
    shutil.copyfile(args.candidates, code / "input_candidates.json")
    def save():
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    save()
    app = runtime = None
    try:
        app = AppLauncher(args).app
        import torch
        from pxr import Usd, UsdPhysics
        from so101_pick_rl.demo_box_runtime import DemoBoxRuntime
        from so101_pick_rl.demo_sequence import finger_geometry
        runtime = DemoBoxRuntime(args.snapshot, 1, args.device, contract_path=args.contract, enable_contact_sensors=True)
        robot = runtime.robot
        finger_indices = [robot.body_names.index(n) for n in ("gripper", "jaw")]
        root = "/World/envs/env_0"
        cube_path = runtime.source_rigid_paths["cube"].replace("{ENV_REGEX_NS}", root)
        table = [str(p.GetPath()) for p in Usd.PrimRange(runtime.sim.stage.GetPrimAtPath(root + "/Scene"))
                 if p.HasAPI(UsdPhysics.CollisionAPI) and not str(p.GetPath()).startswith(cube_path)]
        if not table:
            raise ValueError("table collision prims unresolved")
        groups = {"box": [root + "/BoxTarget"] + [root + "/Box" + n for n in ("Left", "Right", "Front", "Back")],
                  "table": table, "cube": [cube_path]}
        views = {}
        for body in robot.body_names:
            others = [root + "/Robot/" + n for n in robot.body_names if n != body]
            for name, filters in {**groups, "self": others}.items():
                view = runtime.sim.physics_sim_view.create_rigid_contact_view(root + "/Robot/" + body,
                    filter_patterns=filters, max_contact_data_count=4096)
                if view.sensor_count != 1 or view.filter_count != len(filters):
                    raise ValueError(f"unresolved view: {body}/{name}")
                views[body + "/" + name] = view
        report["body_names"] = robot.body_names
        report["contact_filters"] = {key: view.filter_count for key, view in views.items()}
        report["box_filter_order"] = groups["box"]
        report["runtime"] = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(runtime.device)
                             if str(runtime.device).startswith("cuda") else "cpu", "dt": runtime.dt}
        with h5py.File(args.dataset, "r") as data, torch.inference_mode():
            source = data["data/demo_0/initial_state"]
            for candidate in selected:
                targets, phases = opening_schedule(candidate["joint_position_rad"], open_rad=report["open_target_rad"])
                if ((targets < np.array(contract["action"]["lower_rad"])) |
                        (targets > np.array(contract["action"]["upper_rad"]))).any():
                    raise ValueError("command schedule outside limits")
                state = {kind: {name: {field: torch.as_tensor(ds[:], device=runtime.device).clone()
                         for field, ds in entity.items()} for name, entity in entities.items()} for kind, entities in source.items()}
                target0 = torch.as_tensor(targets[0:1], dtype=torch.float32, device=runtime.device)
                state["articulation"]["robot"]["joint_position"] = target0.clone()
                state["articulation"]["robot"]["joint_velocity"] = torch.zeros_like(target0)
                runtime.reset(state, target0)
                result = {"name": candidate["name"], "target_point_world_m": candidate["target_world_m"],
                          "assumed_cube_center_in_gripper_m": candidate.get("assumed_cube_center_in_gripper_m", candidates.get("assumed_cube_center_in_gripper_m")),
                          "first_failure_step": {}, "contacts": {}, "maximum_arm_tracking_error_rad": 0.,
                          "maximum_arm_reference_error_rad": 0.}
                path = args.output_dir / (candidate["name"] + "_trace.jsonl")
                started = time.perf_counter()
                with path.open("x", encoding="utf-8") as trace:
                    for step, (target, phase) in enumerate(zip(targets, phases), 1):
                        reference = target.copy()
                        if args.gravity_target_compensation:
                            target = gravity_compensated_target(reference,
                                robot.root_physx_view.get_gravity_compensation_forces()[0].cpu().numpy(),
                                robot.data.joint_stiffness[0].cpu().numpy(),
                                runtime.lower.cpu().numpy(), runtime.upper.cpu().numpy())
                        target = torch.as_tensor(target[None], dtype=torch.float32, device=runtime.device)
                        obs = runtime.step(2 * (target - runtime.lower) / (runtime.upper - runtime.lower) - 1)
                        contacts = {}
                        for key, view in views.items():
                            buffers = [v.cpu().numpy().copy() for v in view.get_contact_data(runtime.dt)]
                            if int(buffers[4].sum()) >= view.max_contact_data_count:
                                raise ValueError("contact buffer may be truncated")
                            contacts[key] = unpack_contacts(buffers)
                            metrics = contact_summary(contacts[key])
                            aggregate = result["contacts"].setdefault(key, {"maximum_force_n": 0., "minimum_separation_m": None})
                            aggregate["maximum_force_n"] = max(aggregate["maximum_force_n"], metrics["normal_force_n"])
                            depth = metrics["minimum_separation_m"]
                            if depth is not None:
                                old = aggregate["minimum_separation_m"]
                                aggregate["minimum_separation_m"] = depth if old is None else min(old, depth)
                            if key == "base/table":
                                continue
                            if metrics["normal_force_n"] > cfg["maximum_hand_box_force_n"]:
                                result["first_failure_step"].setdefault(key + "/loaded_contact", step)
                            if depth is not None and depth < -cfg["maximum_penetration_m"]:
                                result["first_failure_step"].setdefault(key + "/penetration", step)
                        hand_force = sum(contact_summary(contacts[key])["normal_force_n"] for key in ("gripper/box", "jaw/box"))
                        if hand_force > cfg["maximum_hand_box_force_n"]:
                            result["first_failure_step"].setdefault("combined_hand/box/loaded_contact", step)
                        tracking = float(torch.abs(robot.data.joint_pos[0, :5] - target[0, :5]).max())
                        result["maximum_arm_tracking_error_rad"] = max(result["maximum_arm_tracking_error_rad"], tracking)
                        reference_error = float(np.abs(robot.data.joint_pos[0, :5].cpu().numpy() - reference[:5]).max())
                        result["maximum_arm_reference_error_rad"] = max(result["maximum_arm_reference_error_rad"], reference_error)
                        if "assumed_cube_orientation_wxyz" in candidate:
                            hypothetical = finger_geometry({
                                "finger_body_position_w_m": robot.data.body_pos_w[0, finger_indices].cpu().numpy(),
                                "finger_body_quaternion_wxyz": robot.data.body_quat_w[0, finger_indices].cpu().numpy(),
                                "cube_pose": candidate["target_world_m"] + candidate["assumed_cube_orientation_wxyz"]},
                                pad_geometry, cfg["cube_size_m"])
                            result["final_hypothetical_open_gap_m"] = hypothetical["gap_m"]
                            result["final_hypothetical_cube_projected_width_m"] = hypothetical["projected_cube_width_m"]
                            result["final_hypothetical_aperture_sufficient"] = hypothetical["gap_m"] >= hypothetical["projected_cube_width_m"] + cfg["minimum_open_clearance_m"]
                        trace.write(json.dumps({"step": step, "phase": phase, "joint_position_rad": obs[0, :6].cpu().tolist(),
                            "target_rad": target[0].cpu().tolist(), "reference_rad": reference.tolist(), "contacts": contacts,
                            "body_positions_w_m": robot.data.body_pos_w[0].cpu().tolist(),
                            "body_quaternions_wxyz": robot.data.body_quat_w[0].cpu().tolist()}, allow_nan=False) + "\n")
                if result.get("final_hypothetical_aperture_sufficient") is False:
                    result["first_failure_step"]["final_hypothetical_aperture_insufficient"] = len(targets)
                result.update(steps=len(targets), wall_seconds=time.perf_counter()-started,
                              trace_sha256=file_sha256(path), task_success_claimed=False)
                report["results"].append(result)
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

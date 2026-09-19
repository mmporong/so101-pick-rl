#!/usr/bin/env python3
"""Cross-version replay of preserved LeIsaac absolute motor targets, not training.

Scene/actuator constants follow the transferred LeIsaac source snapshot
(Apache-2.0). No object or joint state is overwritten after each episode reset.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "isaaclab"))
from so101_pick_rl.demo_action_contract import JOINT_NAMES, aligned_targets, audit_targets, validate_source_timing
from so101_pick_rl.grasp_audit import unpack_contacts
from so101_pick_rl.demo_replay_metrics import stable_release_candidate, summarize_contact_trace


def read_initial_state(group, device):
    import torch
    return {kind: {name: {field: torch.as_tensor(ds[:], device=device)
                         for field, ds in entity.items()}
                   for name, entity in entities.items()}
            for kind, entities in group.items()}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--target-policy", choices=["reject", "bounded-derived"], default="reject")
    parser.add_argument("--max-steps", type=int, default=0, help="0 replays all recorded T-1 transitions")
    parser.add_argument("--settle-steps", type=int, default=0,
                        help="Extra final-target hold steps, reported separately from recorded transitions")
    parser.add_argument("--wrist-flex-limit-deg", type=float, default=None,
                        help="Experimental command bound; creates a derived replay, never changes source labels")
    parser.add_argument("--solver-position-iterations", type=int, choices=(4, 16), default=4,
                        help="4 preserves the source robot; 16 is a separately reported solver comparison")
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.max_steps < 0 or len(set(args.episodes)) != len(args.episodes) or min(args.episodes) < 0:
        raise ValueError("invalid step count or duplicate/negative episode selection")
    if args.settle_steps < 0 or (args.settle_steps and args.max_steps):
        raise ValueError("settle steps require a full recorded replay and a nonnegative count")
    if args.wrist_flex_limit_deg is not None and not 0 < args.wrist_flex_limit_deg <= 95:
        raise ValueError("wrist command limit must be in (0, 95] degrees")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dataset = args.dataset.resolve()
    snapshot = args.snapshot.resolve()
    robot_path = snapshot / "assets/robots/so101_follower.usd"
    scene_path = snapshot / "assets/scenes/table_with_cube/scene.usd"
    report = {
        "schema": "so101_pick_rl.demo_box_replay.v1", "status": "running", "kind": "scripted_replay",
        "training": False, "dataset": str(dataset), "dataset_sha256": sha256(dataset),
        "robot_sha256": sha256(robot_path), "scene_sha256": sha256(scene_path),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
        "replay_script_sha256": sha256(__file__),
        "target_policy": args.target_policy, "episodes_requested": args.episodes, "results": [],
        "wrist_flex_command_limit_deg": args.wrist_flex_limit_deg,
        "additional_terminal_hold_steps": args.settle_steps,
        "solver_position_iterations": args.solver_position_iterations,
        "limitations": ["Isaac Sim 4.5 replay differs from source 5.1; equality is measured, not assumed",
                        "Stable box release does not alone prove normal grasp geometry",
                        "Contact separations are PhysX estimates, not hardware validation"],
    }
    if args.wrist_flex_limit_deg is not None:
        report["kind"] = "constrained_target_replay"
    if args.settle_steps:
        report["kind"] += "_with_terminal_hold"
    if args.solver_position_iterations != 4:
        report["kind"] += "_solver_comparison"
    code_paths = [Path(__file__), ROOT / "isaaclab/so101_pick_rl/demo_action_contract.py",
                  ROOT / "isaaclab/so101_pick_rl/demo_replay_metrics.py",
                  ROOT / "isaaclab/so101_pick_rl/grasp_audit.py"]
    code_dir = args.output_dir / "code"
    code_dir.mkdir()
    report["code_sha256"] = {}
    for source in code_paths:
        shutil.copyfile(source, code_dir / source.name)
        report["code_sha256"][source.name] = sha256(source)
    def save():
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    save()
    app = None
    try:
        launcher = AppLauncher(args)
        app = launcher.app
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.utils import configclass
        from pxr import Usd, UsdGeom, UsdPhysics
        from isaacsim.core.version import get_version
        report["runtime"] = {"isaac_sim": get_version()[0],
                             "isaaclab": importlib.metadata.version("isaaclab"),
                             "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}

        @configclass
        class SceneCfg(InteractiveSceneCfg):
            robot = ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/Robot",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(robot_path), activate_contact_sensors=True,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=True, solver_position_iteration_count=args.solver_position_iterations,
                        solver_velocity_iteration_count=4, fix_root_link=True)),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(0.35, -0.64, 0.01), rot=(0., 0., 0., 1.), joint_pos={n: 0. for n in JOINT_NAMES}),
                actuators={"arm": ImplicitActuatorCfg(joint_names_expr=list(JOINT_NAMES[:-1]),
                                effort_limit_sim=10., velocity_limit_sim=10., stiffness=17.8, damping=0.60),
                           "gripper": ImplicitActuatorCfg(joint_names_expr=["gripper"],
                                effort_limit_sim=10., velocity_limit_sim=10., stiffness=17.8, damping=0.60)},
                soft_joint_pos_limit_factor=1.0)
            source_scene = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Scene",
                                       spawn=sim_utils.UsdFileCfg(usd_path=str(scene_path)))
            box_target = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BoxTarget",
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.58, -0.35, 0.0455)),
                spawn=sim_utils.CuboidCfg(size=(0.12, 0.12, 0.008),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.45, 0.85))))
            light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1000.))

        cfg = SceneCfg(num_envs=1, env_spacing=8.)
        stage_asset = Usd.Stage.Open(str(scene_path))
        rigid_paths = {}
        for prim in stage_asset.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                name = prim.GetName()
                matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                rotation = matrix.ExtractRotationQuat().GetNormalized()
                relative = str(prim.GetPath()).split("/", 2)[-1]
                path = "{ENV_REGEX_NS}/Scene/" + relative
                setattr(cfg, name, RigidObjectCfg(prim_path=path, spawn=None,
                    init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(matrix.ExtractTranslation()),
                        rot=(rotation.GetReal(), *rotation.GetImaginary()))))
                rigid_paths[name] = path
        if set(rigid_paths) != {"cube"}:
            raise ValueError(f"unexpected source scene rigid entities: {rigid_paths}")
        for name, size, pos in [
            ("Left", (.012, .132, .07), (.514, -.35, .0815)),
            ("Right", (.012, .132, .07), (.646, -.35, .0815)),
            ("Front", (.12, .012, .07), (.58, -.416, .0815)),
            ("Back", (.12, .012, .07), (.58, -.284, .0815)),
        ]:
            setattr(cfg, "box_wall_" + name, AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Box" + name,
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
                spawn=sim_utils.CuboidCfg(size=size, collision_props=sim_utils.CollisionPropertiesCfg())))

        sim_cfg = sim_utils.SimulationCfg(dt=1/60, render_interval=1, device=args.device)
        sim_cfg.physx.bounce_threshold_velocity = 0.01
        sim_cfg.physx.friction_correlation_distance = 0.00625
        sim = sim_utils.SimulationContext(sim_cfg)
        scene = InteractiveScene(cfg)
        sim.reset()
        scene.update(sim_cfg.dt)
        robot, cube, box = scene["robot"], scene["cube"], scene["box_target"]
        finger_body_ids = [robot.body_names.index(name) for name in ("gripper", "jaw")]
        if tuple(robot.joint_names) != JOINT_NAMES:
            raise ValueError(f"joint order mismatch: {robot.joint_names}")
        lower, upper = robot.data.soft_joint_pos_limits[0].unbind(-1)
        report.update({"joint_names": robot.joint_names, "limits_rad": robot.data.soft_joint_pos_limits[0].cpu().tolist(),
                       "device": str(robot.device), "physics_dt_s": sim_cfg.dt,
                       "cube_mass_kg": cube.data.default_mass.cpu().tolist(), "num_envs": 1,
                       "source_rigid_paths": rigid_paths})
        root = "/World/envs/env_0"
        cube_prim_path = rigid_paths["cube"].replace("{ENV_REGEX_NS}", root)
        table_colliders = [str(p.GetPath()) for p in Usd.PrimRange(sim.stage.GetPrimAtPath(root + "/Scene"))
                           if p.HasAPI(UsdPhysics.CollisionAPI) and not str(p.GetPath()).startswith(cube_prim_path)]
        if not table_colliders:
            raise ValueError("source table collision shapes not found")
        views = {}
        for name, body, filters in [("gripper_cube", "gripper", [cube_prim_path]),
                                     ("jaw_cube", "jaw", [cube_prim_path]),
                                     ("gripper_table", "gripper", table_colliders),
                                     ("jaw_table", "jaw", table_colliders)]:
            view = sim.physics_sim_view.create_rigid_contact_view(root + "/Robot/" + body,
                filter_patterns=filters, max_contact_data_count=4096)
            if view.sensor_count != 1 or view.filter_count < 1:
                raise ValueError(f"contact view unresolved: {name}")
            views[name] = view
        report["contact_views"] = {k: {"sensors": list(v.sensor_paths), "filters": v.filter_paths} for k,v in views.items()}
        save()
        with h5py.File(dataset, "r") as f, torch.inference_mode():
            for index in args.episodes:
                d = f[f"data/demo_{index}"]
                source_index = int(d.attrs["source_index"])
                env_args = json.loads(f[f"source_metadata/source_{source_index:04d}/data_attrs"].attrs["env_args"])
                validate_source_timing(env_args)
                commands = aligned_targets(d["obs/joint_pos"][:], d["obs/joint_pos_target"][:],
                                           d["states/articulation/robot/joint_position"][:])
                quality = audit_targets(commands, lower.cpu().numpy(), upper.cpu().numpy())
                derived = np.clip(commands, lower.cpu().numpy(), upper.cpu().numpy())
                if args.wrist_flex_limit_deg is not None:
                    wrist_limit = np.deg2rad(args.wrist_flex_limit_deg)
                    derived[:, 3] = np.clip(derived[:, 3], -wrist_limit, wrist_limit)
                derived_quality = audit_targets(derived, lower.cpu().numpy(), upper.cpu().numpy())
                if args.target_policy == "reject" and quality["out_of_bounds_values"]:
                    raise ValueError(f"demo_{index} has out-of-range targets; explicit derived policy required")
                state = read_initial_state(d["initial_state"], sim.device)
                scene.reset()
                scene.reset_to(state, is_relative=True)
                robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel))
                scene.write_data_to_sim()
                sim.forward()
                scene.update(sim_cfg.dt)
                initial_error = float((robot.data.joint_pos[0] - torch.tensor(d["obs/joint_pos"][0], device=sim.device)).abs().max())
                initial_z = float(cube.data.root_pos_w[0, 2])
                max_lift, stable_count, max_stable, first_success = 0., 0, 0, None
                errors, records, separation_min = [], [], {}
                steps = min(len(commands), args.max_steps) if args.max_steps else len(commands)
                start = time.monotonic()
                recorded_success = False
                for step in range(steps + args.settle_steps):
                    # Capture the actual constrained rollout state, never pair a
                    # changed command with the old demonstration's observation.
                    pre_step_state = {
                        "joint_position_rad": robot.data.joint_pos[0].cpu().tolist(),
                        "joint_velocity_rad_s": robot.data.joint_vel[0].cpu().tolist(),
                        "previous_target_rad": robot.data.joint_pos_target[0].cpu().tolist(),
                        "cube_pose_w": cube.data.root_state_w[0, :7].cpu().tolist(),
                        "cube_velocity_w": cube.data.root_state_w[0, 7:13].cpu().tolist(),
                        "box_position_w_m": box.data.root_pos_w[0].cpu().tolist(),
                    }
                    command_index = min(step, steps - 1)
                    action = torch.tensor(derived[command_index], dtype=torch.float32, device=sim.device).unsqueeze(0)
                    bounded = action.clamp(lower, upper)
                    if args.target_policy == "reject" and not torch.equal(action, bounded):
                        raise ValueError("unexpected target clipping")
                    # Reproduce source nearest-object effort selection without its single-env indexing assumption.
                    objects = list(scene.rigid_objects.values())
                    positions = torch.stack([o.data.root_pos_w for o in objects])
                    masses = torch.stack([o.data.default_mass[:, 0].to(sim.device) for o in objects])
                    nearest = torch.linalg.vector_norm(positions - robot.data.body_pos_w[:, -1].unsqueeze(0), dim=-1).argmin(0)
                    effort = masses[nearest, torch.arange(scene.num_envs, device=sim.device)] / .15
                    current = robot.data.joint_effort_limits[:, -1]
                    effort = torch.where((effort - current).abs() > .1, effort, current)
                    robot.write_joint_effort_limit_to_sim(effort.unsqueeze(-1), joint_ids=[5])
                    robot.set_joint_position_target(bounded)
                    scene.write_data_to_sim()
                    sim.step(render=False)
                    scene.update(sim_cfg.dt)
                    actual = robot.data.joint_pos[0].cpu().numpy()
                    if not np.isfinite(actual).all():
                        raise ValueError("nonfinite robot state")
                    if step < steps:
                        errors.append(actual - d["states/articulation/robot/joint_position"][step])
                    lift = float(cube.data.root_pos_w[0, 2]) - initial_z
                    max_lift = max(max_lift, lift)
                    relative = (cube.data.root_pos_w - box.data.root_pos_w)[0]
                    candidate = stable_release_candidate(relative.cpu().numpy(), float(robot.data.joint_pos[0, -1]),
                        cube.data.root_lin_vel_w[0].cpu().numpy(), cube.data.root_ang_vel_w[0].cpu().numpy())
                    stable_count = stable_count + 1 if candidate else 0
                    max_stable = max(max_stable, stable_count)
                    if stable_count >= 30 and first_success is None:
                        first_success = step + 1
                    if step < steps:
                        recorded_success = first_success is not None
                    contacts = {}
                    for name, view in views.items():
                        buffers = [x.cpu().numpy().copy() for x in view.get_contact_data(sim_cfg.dt)]
                        if int(buffers[4].sum()) >= view.max_contact_data_count:
                            raise ValueError("contact buffer may be truncated")
                        contacts[name] = unpack_contacts(buffers)
                        values = [v for row in contacts[name] for v in row["separation_m"]]
                        if values:
                            separation_min[name] = min(separation_min.get(name, min(values)), min(values))
                    records.append({"step": step + 1, "joint_pos": actual.tolist(),
                        "pre_step_state": pre_step_state,
                        "phase": "recorded_transition" if step < steps else "additional_terminal_hold",
                        "executed_target": bounded[0].cpu().tolist(),
                        "finger_body_position_w_m": robot.data.body_pos_w[0, finger_body_ids].cpu().tolist(),
                        "finger_body_quaternion_wxyz": robot.data.body_quat_w[0, finger_body_ids].cpu().tolist(),
                        "cube_pose": cube.data.root_state_w[0, :7].cpu().tolist(), "lift_m": lift,
                        "gripper_effort_limit": float(effort[0]), "contacts": contacts,
                        "stable_release_candidate": candidate,
                        "cube_linear_velocity": cube.data.root_lin_vel_w[0].cpu().tolist(),
                        "cube_angular_velocity": cube.data.root_ang_vel_w[0].cpu().tolist()})
                error = np.asarray(errors)
                result = {"episode": f"demo_{index}", "source_episode": str(d.attrs["source_episode"]),
                    "steps": steps, "full_recorded_horizon": steps == len(commands),
                    "total_executed_steps": steps + args.settle_steps,
                    "recorded_horizon_stable_box_release": recorded_success,
                    "initial_joint_error_rad": initial_error, "joint_rmse_rad": float(np.sqrt(np.mean(error**2))),
                    "source_index": source_index, "source_path": str(d.attrs["source_path"]),
                    "target_audit": quality, "executed_target_audit": derived_quality,
                    "clipped_values": int(np.count_nonzero(derived != commands)),
                    "maximum_command_adjustment_rad": float(np.abs(derived - commands).max()),
                    "max_lift_m": max_lift, "stable_box_release": first_success is not None,
                    "first_stable_release_step": first_success, "maximum_stable_seconds": max_stable / 60,
                    "minimum_separation_m": separation_min, "normal_grasp_gate_pass": False,
                    "final_cube_xyz": cube.data.root_pos_w[0].cpu().tolist(), "wall_seconds": time.monotonic() - start,
                    "physical_diagnostics": summarize_contact_trace(records)}
                with (args.output_dir / f"demo_{index}_trace.jsonl").open("x", encoding="utf-8") as stream:
                    for record in records:
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                report["results"].append(result)
                save()
                print(json.dumps(result, allow_nan=False), flush=True)
        report["status"] = "replay_complete_physical_gate_pending"
        save()
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise
    finally:
        if app is not None:
            print("Replay finished; closing simulation (no render writers).", flush=True)
            if "sim" in locals():
                # Release Isaac Lab's standalone STOP render-loop subscription
                # before stopping; otherwise headless shutdown never returns.
                sim.clear_all_callbacks()
                sim.clear_instance()
                sim.stop()
            # This state-only replay never starts a Replicator writer/render job.
            app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()

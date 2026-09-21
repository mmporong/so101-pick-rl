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
from so101_pick_rl.demo_sequence import DemoSequenceGate, sequence_features
from so101_pick_rl.demo_place_controller import ControlledPlace, bounded_cartesian_step


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
    parser.add_argument("--sequence-gate", type=Path,
                        default=ROOT / "configs/evaluation/demo_box_replay_gate_v2.json",
                        help="Versioned sequence audit; does not alter executed targets")
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--target-policy", choices=["reject", "bounded-derived"], default="reject")
    parser.add_argument("--max-steps", type=int, default=0, help="0 replays all recorded T-1 transitions")
    parser.add_argument("--settle-steps", type=int, default=0,
                        help="Extra final-target hold steps, reported separately from recorded transitions")
    parser.add_argument("--wrist-flex-limit-deg", type=float, default=None,
                        help="Experimental command bound; creates a derived replay, never changes source labels")
    parser.add_argument("--solver-position-iterations", type=int, choices=(4, 16), default=4,
                        help="4 preserves the source robot; 16 is a separately reported solver comparison")
    parser.add_argument("--solver-type", choices=("tgs", "pgs"), default="tgs")
    parser.add_argument("--max-velocity-iterations", type=int, choices=(0, 4), default=None,
                        help="Optional scene-wide solver velocity-iteration cap for a reported comparison")
    parser.add_argument("--min-position-iterations", type=int, choices=(1, 64), default=1,
                        help="Scene-wide minimum; unlike a robot-only request this also bounds the GPU island")
    parser.add_argument("--controlled-place", action="store_true",
                        help="Replace an impending high release with supported slow placement")
    parser.add_argument("--external-forces-every-iteration", action="store_true",
                        help="Separate TGS stability variant; preserves the source setting unless requested")
    parser.add_argument("--place-offset-xy", type=float, nargs=2, default=(0., 0.), metavar=("X_M", "Y_M"))
    parser.add_argument("--box-offset-xy", type=float, nargs=2, default=(0., 0.), metavar=("X_M", "Y_M"),
                        help="Explicit scene variant: translate box floor and walls together at reset")
    parser.add_argument("--physics-substeps", type=int, choices=(1, 4), default=1,
                        help="Physics steps per unchanged 60 Hz command; audit every physics step")
    parser.add_argument("--supported-gripper-effort", type=float, default=None,
                        help="Optional Nm cap ramped down only after supported placement")
    parser.add_argument("--level-cube", action="store_true",
                        help="Correct carried cube Z tilt instead of gripper shaft tilt before support")
    parser.add_argument("--grasp-height-offset", type=float, default=0.,
                        help="Derived source target offset in meters, ramped before closure; no state writes")
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.max_steps < 0 or len(set(args.episodes)) != len(args.episodes) or min(args.episodes) < 0:
        raise ValueError("invalid step count or duplicate/negative episode selection")
    if args.settle_steps < 0 or (args.settle_steps and args.max_steps):
        raise ValueError("settle steps require a full recorded replay and a nonnegative count")
    if args.controlled_place and args.max_steps:
        raise ValueError("controlled placement requires an untruncated source prefix")
    if not np.isfinite(args.box_offset_xy).all() or np.max(np.abs(args.box_offset_xy)) > .10:
        raise ValueError("box scene comparison is limited to finite offsets within 10 cm per axis")
    if args.supported_gripper_effort is not None and not (args.controlled_place and 0 < args.supported_gripper_effort <= .0666667):
        raise ValueError("supported effort requires controlled placement and a positive cap at or below the source cap")
    if not 0 <= args.grasp_height_offset <= .015:
        raise ValueError("grasp height comparison must be finite and within 0..15 mm")
    if args.wrist_flex_limit_deg is not None and not 0 < args.wrist_flex_limit_deg <= 95:
        raise ValueError("wrist command limit must be in (0, 95] degrees")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dataset = args.dataset.resolve()
    snapshot = args.snapshot.resolve()
    robot_path = snapshot / "assets/robots/so101_follower.usd"
    scene_path = snapshot / "assets/scenes/table_with_cube/scene.usd"
    gate_path = args.sequence_gate.resolve()
    geometry_path = ROOT / "configs/isaaclab/demo_box_pad_geometry.json"
    gate_cfg = json.loads(gate_path.read_text(encoding="utf-8"))
    gate_cfg["physics_dt_s"] /= args.physics_substeps
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    if geometry["valid_for"] != sha256(robot_path):
        raise ValueError("pad geometry does not match the robot asset")
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
        "solver_type": args.solver_type,
        "max_velocity_iterations_override": args.max_velocity_iterations,
        "min_position_iterations": args.min_position_iterations,
        "sequence_gate_sha256": sha256(gate_path), "geometry_sha256": sha256(geometry_path),
        "controlled_place": args.controlled_place,
        "external_forces_every_iteration": args.external_forces_every_iteration,
        "place_offset_xy_m": args.place_offset_xy,
        "box_offset_xy_m": args.box_offset_xy,
        "control_dt_s": 1 / 60, "physics_substeps": args.physics_substeps,
        "supported_gripper_effort_limit_nm": args.supported_gripper_effort,
        "level_cube": args.level_cube,
        "grasp_height_offset_m": args.grasp_height_offset,
        "placement_translation_control_point": "observed_cube_center_local_grasp_assumption",
        "effective_sequence_gate": gate_cfg,
        "effective_sequence_gate_sha256": hashlib.sha256(json.dumps(gate_cfg, sort_keys=True).encode()).hexdigest(),
        "gripper_effort_policy": "cube_mass_only" if args.controlled_place else "source_nearest_rigid_object",
        "box_walls_kinematic_for_gpu_contact_audit": True,
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
    if args.controlled_place:
        report["kind"] += "_controlled_place"
    code_paths = [Path(__file__), ROOT / "isaaclab/so101_pick_rl/demo_action_contract.py",
                  ROOT / "isaaclab/so101_pick_rl/demo_replay_metrics.py",
                  ROOT / "isaaclab/so101_pick_rl/grasp_audit.py",
                  ROOT / "isaaclab/so101_pick_rl/demo_sequence.py",
                  ROOT / "isaaclab/so101_pick_rl/demo_place_controller.py", gate_path, geometry_path]
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
        # Reject missing timing before starting Kit; never infer source timing
        # from the current scene or a different dataset's metadata.
        with h5py.File(dataset, "r") as preflight:
            for index in args.episodes:
                source_index = int(preflight[f"data/demo_{index}"].attrs["source_index"])
                metadata = preflight[f"source_metadata/source_{source_index:04d}/data_attrs"]
                validate_source_timing(json.loads(metadata.attrs["env_args"]))
        launcher = AppLauncher(args)
        app = launcher.app
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.utils import configclass
        from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
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
        cfg.box_target.init_state.pos = (.58 + args.box_offset_xy[0], -.35 + args.box_offset_xy[1], .0455)
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
            pos = (pos[0] + args.box_offset_xy[0], pos[1] + args.box_offset_xy[1], pos[2])
            setattr(cfg, "box_wall_" + name, AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Box" + name,
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
                spawn=sim_utils.CuboidCfg(size=size, collision_props=sim_utils.CollisionPropertiesCfg(),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True))))

        sim_cfg = sim_utils.SimulationCfg(dt=1/(60 * args.physics_substeps), render_interval=args.physics_substeps, device=args.device)
        sim_cfg.physx.bounce_threshold_velocity = 0.01
        sim_cfg.physx.solver_type = 1 if args.solver_type == "tgs" else 0
        sim_cfg.physx.min_position_iteration_count = args.min_position_iterations
        if args.max_velocity_iterations is not None:
            sim_cfg.physx.max_velocity_iteration_count = args.max_velocity_iterations
        sim_cfg.physx.friction_correlation_distance = 0.00625
        sim = sim_utils.SimulationContext(sim_cfg)
        physics_api = PhysxSchema.PhysxSceneAPI(sim.stage.GetPrimAtPath(sim_cfg.physics_prim_path))
        force_attr = physics_api.CreateEnableExternalForcesEveryIterationAttr(args.external_forces_every_iteration)
        if force_attr.Get() != args.external_forces_every_iteration:
            raise ValueError("TGS external force setting was not applied")
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
        report["authored_actor_solver_iterations"] = {str(p.GetPath()):
            {a.GetName(): a.Get() for a in p.GetAttributes() if "solver" in a.GetName().lower()
             and "iteration" in a.GetName().lower()} for p in Usd.PrimRange(sim.stage.GetPrimAtPath(root))
            if any("solver" in a.GetName().lower() and "iteration" in a.GetName().lower() for a in p.GetAttributes())}
        cube_prim_path = rigid_paths["cube"].replace("{ENV_REGEX_NS}", root)
        table_colliders = [str(p.GetPath()) for p in Usd.PrimRange(sim.stage.GetPrimAtPath(root + "/Scene"))
                           if p.HasAPI(UsdPhysics.CollisionAPI) and not str(p.GetPath()).startswith(cube_prim_path)]
        if not table_colliders:
            raise ValueError("source table collision shapes not found")
        views = {}
        for name, body, filters in [("gripper_cube", "gripper", [cube_prim_path]),
                                     ("jaw_cube", "jaw", [cube_prim_path]),
                                     ("gripper_table", "gripper", table_colliders),
                                     ("jaw_table", "jaw", table_colliders),
                                     ("gripper_box", "gripper", [root + "/BoxTarget"] + [root + "/Box" + x for x in ("Left", "Right", "Front", "Back")]),
                                     ("jaw_box", "jaw", [root + "/BoxTarget"] + [root + "/Box" + x for x in ("Left", "Right", "Front", "Back")]),
                                     ("cube_box", None, [root + "/BoxTarget"])]:
            sensor_path = cube_prim_path if body is None else root + "/Robot/" + body
            view = sim.physics_sim_view.create_rigid_contact_view(sensor_path,
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
                state["rigid_object"]["box_target"]["root_pose"][:, :2] += torch.tensor(args.box_offset_xy, device=sim.device)
                scene.reset()
                scene.reset_to(state, is_relative=True)
                robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel))
                scene.write_data_to_sim()
                sim.forward()
                scene.update(sim_cfg.dt)
                initial_error = float((robot.data.joint_pos[0] - torch.tensor(d["obs/joint_pos"][0], device=sim.device)).abs().max())
                initial_z = float(cube.data.root_pos_w[0, 2])
                gate = DemoSequenceGate(gate_cfg, cube.data.root_pos_w[0].cpu().numpy())
                controller = ControlledPlace(geometry, gate_cfg, args.place_offset_xy, args.level_cube,
                    minimum_support_force_n=.8 * float(cube.data.default_mass[0, 0]) * abs(sim_cfg.gravity[2])) if args.controlled_place else None
                last_feature = None
                control_upper, control_lower = upper.cpu().numpy().copy(), lower.cpu().numpy().copy()
                if args.wrist_flex_limit_deg is not None:
                    control_upper[3] = min(control_upper[3], np.deg2rad(args.wrist_flex_limit_deg))
                    control_lower[3] = max(control_lower[3], -np.deg2rad(args.wrist_flex_limit_deg))
                max_lift, stable_count, max_stable, first_success = 0., 0, 0, None
                errors, records, separation_min = [], [], {}
                steps = min(len(commands), args.max_steps) if args.max_steps else len(commands)
                start = time.monotonic()
                recorded_success = False
                budget = (steps + args.settle_steps + (2400 if controller else 0)) * args.physics_substeps
                for step in range(budget):
                    if controller and controller.done:
                        break
                    control_step = step // args.physics_substeps
                    if control_step >= steps + args.settle_steps and not (controller and controller.active):
                        break
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
                    command_index = min(control_step, steps - 1)
                    if step % args.physics_substeps == 0:
                        target = derived[command_index].copy()
                        if args.grasp_height_offset and not (controller and controller.active):
                            spatial = robot.root_physx_view.get_jacobians()[0, finger_body_ids[0] - 1].cpu().numpy()
                            quat = robot.data.body_quat_w[0, finger_body_ids[0]].cpu().numpy()
                            offset_m = args.grasp_height_offset * np.clip((control_step - 40) / 40, 0., 1.)
                            target[:5] += bounded_cartesian_step(spatial[:3, :5], [0, 0, offset_m],
                                spatial[3:, :5], quat, quat, max_cartesian_step_m=.015, max_joint_step_rad=.15,
                                delta_lower_rad=control_lower[:5] - target[:5], delta_upper_rad=control_upper[:5] - target[:5])
                    if controller and step % args.physics_substeps == 0:
                        target = controller.action(proposed_target_rad=target,
                            previous_target_rad=pre_step_state["previous_target_rad"],
                            q_rad=pre_step_state["joint_position_rad"], cube_position_m=pre_step_state["cube_pose_w"][:3],
                            box_position_m=pre_step_state["box_position_w_m"], feature=last_feature,
                            body_quaternions=robot.data.body_quat_w[0, finger_body_ids].cpu().numpy(),
                            body_positions_m=robot.data.body_pos_w[0, finger_body_ids].cpu().numpy(),
                            cube_quaternion=pre_step_state["cube_pose_w"][3:],
                            spatial_jacobians=robot.root_physx_view.get_jacobians()[0, [i - 1 for i in finger_body_ids]].cpu().numpy(),
                            lower_rad=control_lower, upper_rad=control_upper)
                    action = torch.tensor(target, dtype=torch.float32, device=sim.device).unsqueeze(0)
                    bounded = action.clamp(lower, upper)
                    if args.target_policy == "reject" and not torch.equal(action, bounded):
                        raise ValueError("unexpected target clipping")
                    # Reproduce source nearest-object effort selection without its single-env indexing assumption.
                    objects = list(scene.rigid_objects.values())
                    positions = torch.stack([o.data.root_pos_w for o in objects])
                    masses = torch.stack([o.data.default_mass[:, 0].to(sim.device) for o in objects])
                    nearest = torch.linalg.vector_norm(positions - robot.data.body_pos_w[:, -1].unsqueeze(0), dim=-1).argmin(0)
                    effort = masses[nearest, torch.arange(scene.num_envs, device=sim.device)] / .15
                    if controller:
                        effort = cube.data.default_mass[:, 0].to(sim.device) / .15
                    current = robot.data.joint_effort_limits[:, -1]
                    if controller and args.supported_gripper_effort is not None and controller.support_started:
                        effort = torch.clamp(current - .002 / args.physics_substeps, min=args.supported_gripper_effort)
                    elif not controller:
                        effort = torch.where((effort - current).abs() > .1, effort, current)
                    robot.write_joint_effort_limit_to_sim(effort.unsqueeze(-1), joint_ids=[5])
                    robot.set_joint_position_target(bounded)
                    scene.write_data_to_sim()
                    sim.step(render=False)
                    scene.update(sim_cfg.dt)
                    actual = robot.data.joint_pos[0].cpu().numpy()
                    if not np.isfinite(actual).all():
                        raise ValueError("nonfinite robot state")
                    if (control_step < steps and (step + 1) % args.physics_substeps == 0
                            and not (controller and controller.active)):
                        errors.append(actual - d["states/articulation/robot/joint_position"][control_step])
                    lift = float(cube.data.root_pos_w[0, 2]) - initial_z
                    max_lift = max(max_lift, lift)
                    relative = (cube.data.root_pos_w - box.data.root_pos_w)[0]
                    candidate = stable_release_candidate(relative.cpu().numpy(), float(robot.data.joint_pos[0, -1]),
                        cube.data.root_lin_vel_w[0].cpu().numpy(), cube.data.root_ang_vel_w[0].cpu().numpy())
                    stable_count = stable_count + 1 if candidate else 0
                    max_stable = max(max_stable, stable_count)
                    if stable_count >= 30 * args.physics_substeps and first_success is None:
                        first_success = step + 1
                    if control_step < steps and not (controller and controller.active):
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
                    records.append({"step": step + 1, "control_step": control_step + 1,
                        "control_boundary": step % args.physics_substeps == 0, "joint_pos": actual.tolist(),
                        "pre_step_state": pre_step_state,
                        "phase": controller.phase if controller and controller.active else (
                            "recorded_transition" if control_step < steps else "additional_terminal_hold"),
                        "executed_target": bounded[0].cpu().tolist(),
                        "finger_body_position_w_m": robot.data.body_pos_w[0, finger_body_ids].cpu().tolist(),
                        "finger_body_quaternion_wxyz": robot.data.body_quat_w[0, finger_body_ids].cpu().tolist(),
                        "finger_spatial_jacobians_w": robot.root_physx_view.get_jacobians()[0, [i - 1 for i in finger_body_ids]].cpu().tolist(),
                        "cube_pose": cube.data.root_state_w[0, :7].cpu().tolist(), "lift_m": lift,
                        "gripper_effort_limit": float(effort[0]), "contacts": contacts,
                        "stable_release_candidate": candidate,
                        "cube_linear_velocity": cube.data.root_lin_vel_w[0].cpu().tolist(),
                        "cube_angular_velocity": cube.data.root_ang_vel_w[0].cpu().tolist()})
                    last_feature = sequence_features(records[-1], geometry, gate_cfg)
                    gate.update(last_feature)
                    records[-1]["sequence_gate"] = gate.report()
                    if controller and gate.failures:
                        # Preserve the first invalid sample and stop this trial;
                        # later settling must never recover a failed trajectory.
                        break
                    if (step + 1) % 240 == 0:
                        report["progress"] = {"episode": index, "step": step + 1,
                                              "phase": records[-1]["phase"], "gate": gate.report(),
                                              "cube_position_m": records[-1]["cube_pose"][:3],
                                              "approach_world_z": last_feature["approach_world_z"],
                                              "height_above_rest_m": last_feature["height_above_rest_m"],
                                              "supported": last_feature["supported"],
                                              "support_force_n": last_feature["support_force_n"],
                                              "hand_box_force_n": last_feature["hand_box_force_n"],
                                              "linear_speed_m_s": last_feature["linear_speed_m_s"],
                                              "angular_speed_rad_s": last_feature["angular_speed_rad_s"]}
                        save()
                error = np.asarray(errors)
                result = {"episode": f"demo_{index}", "source_episode": str(d.attrs["source_episode"]),
                    "steps": steps, "full_recorded_horizon": (steps == len(commands)
                        and len(records) >= steps * args.physics_substeps and not (controller and controller.active)),
                    "source_comparison_steps": len(errors), "total_executed_steps": len(records),
                    "recorded_horizon_stable_box_release": recorded_success,
                    "initial_joint_error_rad": initial_error,
                    "joint_rmse_rad": float(np.sqrt(np.mean(error**2))) if len(errors) else None,
                    "source_index": source_index, "source_path": str(d.attrs["source_path"]),
                    "target_audit": quality, "derived_source_target_audit": derived_quality,
                    "executed_target_audit": audit_targets(np.array([r["executed_target"] for r in records]),
                                                           lower.cpu().numpy(), upper.cpu().numpy()),
                    "source_adjustment_scope": "clipping metrics below cover derived source labels, not corrective controller commands",
                    "clipped_values": int(np.count_nonzero(derived != commands)),
                    "maximum_command_adjustment_rad": float(np.abs(derived - commands).max()),
                    "max_lift_m": max_lift, "stable_box_release": first_success is not None,
                    "outcomes": {
                        "source_stable_release_criterion_pass": first_success is not None,
                        "sequence_audit_pass": gate.report()["normal_grasp_gate_pass"],
                        "hardware_validated": False,
                        "training_ready": False,
                        "source_criterion_scope": "configured stable_release_v2 on this rollout, not generation-version equivalence",
                    },
                    "first_stable_release_step": first_success, "maximum_stable_seconds": max_stable * sim_cfg.dt,
                    "minimum_separation_m": separation_min, "normal_grasp_gate_pass": gate.report()["normal_grasp_gate_pass"],
                    "termination": ("sequence_pass" if gate.report()["normal_grasp_gate_pass"] else
                                    "sequence_failure" if gate.failures else
                                    "correction_budget_exhausted" if controller and controller.active else "recorded_horizon_end"),
                    "termination_scope": "sequence_audit_not_source_task_or_process_exit",
                    "sequence_gate": gate.report(),
                    "placement_controller": {"active": controller.active, "done": controller.done,
                        "phase_history": controller.history, "final_phase": controller.phase,
                        "minimum_support_force_n": controller.minimum_support_force_n} if controller else None,
                    "final_cube_xyz": cube.data.root_pos_w[0].cpu().tolist(), "wall_seconds": time.monotonic() - start,
                    "physical_diagnostics": summarize_contact_trace(records, dt_s=sim_cfg.dt)}
                with (args.output_dir / f"demo_{index}_trace.jsonl").open("x", encoding="utf-8") as stream:
                    for record in records:
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                result["trace_sha256"] = sha256(args.output_dir / f"demo_{index}_trace.jsonl")
                report["results"].append(result)
                save()
                print(json.dumps({k: result[k] for k in ("episode", "termination", "total_executed_steps",
                    "sequence_gate", "placement_controller", "final_cube_xyz", "wall_seconds")}, allow_nan=False), flush=True)
        report["status"] = "replay_complete_sequence_audited"
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

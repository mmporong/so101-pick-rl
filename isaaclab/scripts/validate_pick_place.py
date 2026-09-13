#!/usr/bin/env python3
"""Live task wiring fixtures, explicitly separate from policy success evaluation."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "isaaclab"))
from runtime_metrics import ResourceSampler, enforce_resource_guard, write_json
from so101_pick_rl.kit_log import bind_kit_log, summarize_kit_log

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--capture", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
started = datetime.now(timezone.utc)
report = {"schema": "so101_pick_rl.pick_place_fixtures.v1", "status": "failed",
          "classification": "diagnostic_only_not_policy_evaluation", "started_at_utc": started.isoformat(),
          "command": subprocess.list2cmdline(sys.argv), "checks": {}}
app = env = None
sampler = ResourceSampler()
binding = {"path": None}
try:
    report["preflight_resources"] = enforce_resource_guard(40, 4096, 70)
    report["git_commit"] = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    report["git_dirty"] = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip())
    sampler.start()
    app = AppLauncher(args).app
    binding = bind_kit_log(started.timestamp(), args.output.name)
    import torch
    import gymnasium as gym
    import isaaclab_tasks
    import so101_pick_rl.tasks
    from isaaclab_tasks.utils import parse_env_cfg
    from so101_pick_rl.tasks.pick_place import mdp

    cfg = parse_env_cfg("SO101-PickPlace-v0", device=args.device, num_envs=2)
    cfg.seed = 0
    env = gym.make("SO101-PickPlace-v0", cfg=cfg, render_mode="rgb_array" if args.capture else None)
    obs, _ = env.reset(seed=0)
    raw = env.unwrapped
    checks = report["checks"]
    checks["observation_shape"] = list(obs["policy"].shape) == [2, raw.pick_place_spec["observation"]["dimension"]]
    checks["cuda_environment"] = str(raw.device).startswith("cuda")
    checks["lift_is_not_termination"] = "lift_held" not in raw.termination_manager.active_terms
    checks["full_success_termination"] = "pick_place_success" in raw.termination_manager.active_terms
    from so101_pick_rl.run_contract import contract_sha256
    report["contract_sha256"] = contract_sha256("SO101-PickPlace-v0")
    checks["contract_control_rate"] = raw.step_dt == 1 / raw.pick_place_spec["control"]["policy_hz"]
    checks["initial_target_separation"] = bool((torch.linalg.vector_norm(
        mdp.cube_to_target(raw)[:, :2], dim=1) >= 0.06).all())
    # Poison one slot, then prove reset repairs only it, including task history and target.
    target_before = raw.target_pos_w.clone()
    raw.pick_place_state.picked[:] = True
    raw.pick_place_state.carry_valid[:] = True
    raw.target_pos_w[0] = torch.nan
    ids = torch.tensor([0], device=raw.device)
    raw._reset_idx(ids)
    checks["partial_reset_target_isolation"] = bool(torch.isfinite(raw.target_pos_w).all()
        and torch.equal(raw.target_pos_w[1], target_before[1]))
    checks["partial_reset_history_isolation"] = bool(not raw.pick_place_state.picked[0]
        and raw.pick_place_state.picked[1] and not raw.pick_place_state.carry_valid[0])
    env.reset(seed=0)
    actions = torch.zeros(env.action_space.shape, device=raw.device)
    robot = raw.scene["robot"]
    cube = raw.scene["cube"]
    fixed_idx = robot.body_names.index("gripper")
    moving_idx = robot.body_names.index("jaw")
    midpoint = 0.5 * (robot.data.body_pos_w[:, fixed_idx] + robot.data.body_pos_w[:, moving_idx])
    pose = cube.data.root_pose_w.clone()
    pose[:, :3] = midpoint
    cube.write_root_pose_to_sim(pose)
    cube.write_root_velocity_to_sim(torch.zeros_like(cube.data.root_vel_w))
    forces = torch.zeros(2, device=raw.device)
    for _ in range(5):
        env.step(actions)
        forces = torch.maximum(forces, mdp.contact_forces(raw).amax(dim=0))
    report["peak_contact_forces_n"] = forces.tolist()
    checks["physical_contact_sensor_positive"] = bool((forces > 0.2).all())
    env.reset(seed=0)
    checks["reset_contact_observation_cleared"] = bool((mdp.contact_forces(raw) == 0).all())
    checks["reset_grasp_sequence_cleared"] = bool(
        not raw.pick_place_state.pregrasp_opened.any() and not raw.pick_place_state.grasp_sequence_valid.any())
    # Compare distal points rather than joint/body origins: the jaw rotates about its origin.
    from isaaclab.utils.math import quat_apply
    from pxr import UsdGeom, Usd, UsdPhysics, Gf
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    geometry = []
    tip_offsets = {}
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if "/env_0/Robot/" not in path or not any(f"/{name}/" in path for name in ("jaw", "gripper")):
            continue
        if prim.IsA(UsdGeom.Mesh):
            points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            if points:
                coordinates = torch.tensor([list(p) for p in points])
                geometry.append({"path": path, "minimum": coordinates.amin(dim=0).tolist(),
                                 "maximum": coordinates.amax(dim=0).tolist(),
                                 "collision": prim.HasAPI(UsdPhysics.CollisionAPI)})
                body_name = "jaw" if "/jaw/" in path else "gripper"
                if "/visuals/" in path and ("moving_jaw" in path or "wrist_roll_follower" in path):
                    if body_name == "jaw":
                        tip_points = coordinates[coordinates[:, 1] < coordinates[:, 1].min() + 0.005]
                    else:
                        tip_points = coordinates[coordinates[:, 2] > coordinates[:, 2].max() - 0.005]
                    body_prim = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{body_name}")
                    mesh_transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    body_transform = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    local_tip = (mesh_transform * body_transform.GetInverse()).Transform(
                        Gf.Vec3d(*tip_points.mean(dim=0).tolist()))
                    tip_offsets[body_name] = list(local_tip)
    report["finger_mesh_local_bounds"] = geometry
    env.reset(seed=0)
    grip_idx = robot.joint_names.index("gripper")
    distances = []
    for bound in (0, 1):
        positions = robot.data.default_joint_pos.clone()
        positions[:, grip_idx] = robot.data.soft_joint_pos_limits[:, grip_idx, bound]
        robot.write_joint_state_to_sim(positions, torch.zeros_like(positions))
        raw.action_manager.get_term("joint_position_delta").reset()
        raw.scene.write_data_to_sim()
        raw.sim.forward()
        raw.scene.update(raw.physics_dt)
        if args.capture:
            from PIL import Image
            pixels = env.render()
            picture_path = args.output.with_name(args.output.stem + f"_gripper_{bound}.png")
            Image.fromarray(pixels).save(picture_path)
        fixed_tip = robot.data.body_pos_w[0:1, fixed_idx] + quat_apply(
            robot.data.body_quat_w[0:1, fixed_idx], torch.tensor([tip_offsets["gripper"]], device=raw.device))
        moving_tip = robot.data.body_pos_w[0:1, moving_idx] + quat_apply(
            robot.data.body_quat_w[0:1, moving_idx], torch.tensor([tip_offsets["jaw"]], device=raw.device))
        distances.append(float(torch.linalg.vector_norm(fixed_tip - moving_tip)))
    report["distal_probe_separation_at_lower_upper_limit_m"] = distances
    report["mesh_derived_tip_offsets_m"] = tip_offsets
    checks["upper_joint_limit_opens_distal_probe"] = distances[1] > distances[0] + 0.01
    # Synthetic past history ONLY: prove physical rest + full-success wiring and auto-reset.
    # This is never included in learned-policy evaluation.
    env.reset(seed=0)
    positions = robot.data.default_joint_pos.clone()
    positions[:, grip_idx] = robot.data.soft_joint_pos_limits[:, grip_idx, 1]
    robot.write_joint_state_to_sim(positions, torch.zeros_like(positions))
    raw.action_manager.get_term("joint_position_delta").reset()
    pose = cube.data.root_pose_w.clone()
    pose[:, :3] = raw.target_pos_w
    cube.write_root_pose_to_sim(pose)
    cube.write_root_velocity_to_sim(torch.zeros_like(cube.data.root_vel_w))
    raw.pick_place_state.picked[0] = True
    raw.pick_place_state.carry_valid[0] = True
    raw.pick_place_state.released[0] = True
    success_steps = []
    terminal_bonus_rewards = []
    terminal_bonus_components = []
    pushed_successes = 0
    required_policy_steps = math.ceil(raw.pick_place_state.required_stable_steps / raw.cfg.decimation)
    updates_before = raw.history_updates
    for i in range(1, required_policy_steps + 16):
        _, rewards, _, _, _ = env.step(actions)
        flags = raw.termination_manager.get_term("pick_place_success")
        if bool(flags[0]):
            success_steps.append(i)
            terminal_bonus_rewards.append(float(rewards[0]))
            components = dict(raw.reward_manager.get_active_iterable_terms(0))
            terminal_bonus_components.append(components["terminal_success"][0] * raw.step_dt)
        pushed_successes += int(flags[1])
    report["synthetic_history_stable_success_steps"] = success_steps
    report["synthetic_history_terminal_step_rewards"] = terminal_bonus_rewards
    report["synthetic_history_terminal_bonus_components"] = terminal_bonus_components
    checks["terminal_bonus_paid"] = bool(terminal_bonus_components) and all(
        math.isclose(value, raw.pick_place_spec["reward"]["terminal_success_bonus"], rel_tol=1e-5)
        for value in terminal_bonus_components)
    report["no_history_target_placement_success_count"] = pushed_successes
    checks["stable_placement_terminates_and_resets"] = len(success_steps) == 1 and success_steps[0] >= required_policy_steps
    checks["history_updated_every_physics_step"] = raw.history_updates - updates_before == i * raw.cfg.decimation
    checks["placement_without_pick_history_rejected"] = pushed_successes == 0
    # Adversarial substep fixture: an already achieved hold must survive until the
    # policy boundary, while a later bounce cannot create a new success by itself.
    env.reset(seed=0)
    positions = robot.data.default_joint_pos.clone()
    positions[:, grip_idx] = robot.data.soft_joint_pos_limits[:, grip_idx, 1]
    robot.write_joint_state_to_sim(positions, torch.zeros_like(positions))
    raw.action_manager.get_term("joint_position_delta").reset()
    pose = cube.data.root_pose_w.clone()
    pose[:, :3] = raw.target_pos_w
    cube.write_root_pose_to_sim(pose)
    cube.write_root_velocity_to_sim(torch.zeros_like(cube.data.root_vel_w))
    raw.pick_place_state.picked[0] = True
    raw.pick_place_state.carry_valid[0] = True
    raw.pick_place_state.released[0] = True
    raw.pick_place_state.stable_steps[0] = raw.pick_place_state.required_stable_steps - 2
    original_update = mdp.update_history
    substep_successes = []

    def injected_bounce_update(environment):
        if len(substep_successes) == 2:
            velocity = cube.data.root_vel_w.clone()
            velocity[0, 0] = 1.0
            cube.write_root_velocity_to_sim(velocity)
        result = original_update(environment)
        substep_successes.append(bool(result[0]))
        return result

    mdp.update_history = injected_bounce_update
    try:
        _, rewards, _, _, _ = env.step(actions)
    finally:
        mdp.update_history = original_update
    report["injected_bounce_substep_successes"] = substep_successes
    checks["intermediate_success_latched_to_policy_boundary"] = (
        any(substep_successes) and not substep_successes[-1]
        and bool(raw.termination_manager.get_term("pick_place_success")[0])
        and math.isclose(dict(raw.reward_manager.get_active_iterable_terms(0))["terminal_success"][0] * raw.step_dt,
                         raw.pick_place_spec["reward"]["terminal_success_bonus"], rel_tol=1e-5))
    report["device"] = str(raw.device)
    report["observation_dimension"] = obs["policy"].shape[-1]
except Exception as exc:
    report["error"] = str(exc)
    report["traceback"] = traceback.format_exc()
finally:
    if env is not None:
        env.close()
    report["resources"] = sampler.stop()
    report["simulator_log"] = summarize_kit_log(binding)
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["checks"]["simulator_error_free"] = report["simulator_log"].get("error_count") == 0
    report["status"] = "passed" if not report.get("error") and all(report["checks"].values()) else "failed"
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if app is not None:
        app.close()
raise SystemExit(0 if report["status"] == "passed" else 1)

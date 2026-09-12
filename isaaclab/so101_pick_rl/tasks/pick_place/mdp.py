"""Observation, reset and reward terms for full Pick & Place."""

import torch

from ..lift_cube.mdp.events import reset_cube_pose


def contact_forces(env):
    return torch.stack([
        torch.linalg.vector_norm(env.scene.sensors[name].data.force_matrix_w, dim=-1).amax(dim=(1, 2))
        for name in ("fixed_finger_contact", "moving_finger_contact")
    ], dim=1)


def cube_to_target(env):
    return env.target_pos_w - env.scene["cube"].data.root_pos_w


def cube_velocity(env):
    return env.scene["cube"].data.root_vel_w


def phase_state(env):
    return env.pick_place_state.observation()


def gripper_open_fraction(env):
    robot = env.scene["robot"]
    idx = robot.joint_names.index("gripper")
    limits = robot.data.soft_joint_pos_limits[:, idx]
    return ((robot.data.joint_pos[:, idx] - limits[:, 0]) / (limits[:, 1] - limits[:, 0])).clamp(0, 1)


def ee_distance(env):
    ee = env.scene.sensors["ee_frame"].data.target_pos_w[:, 0]
    return torch.linalg.vector_norm(ee - env.scene["cube"].data.root_pos_w, dim=1)


def reset_pick_place(env, env_ids, pose_range, velocity_range, asset_cfg):
    reset_cube_pose(env, env_ids, pose_range, velocity_range, asset_cfg)
    env.pick_place_state.reset(env_ids)
    for value in env.episode_metrics.values():
        value[env_ids] = False
    ranges = env.pick_place_spec["task"]["target"]["position_range_m"]
    origin = env.scene.env_origins[env_ids]
    for axis, name in enumerate(("x", "y")):
        low, high = ranges[name]
        env.target_pos_w[env_ids, axis] = origin[:, axis] + low + (high - low) * torch.rand(
            len(env_ids), device=env.device)
    env.target_pos_w[env_ids, 2] = origin[:, 2] + env.pick_place_spec["task"]["target"]["rest_height_m"]
    marker_pose = env.scene["target"].data.root_pose_w[env_ids].clone()
    marker_pose[:, :3] = env.target_pos_w[env_ids]
    marker_pose[:, 2] = origin[:, 2] + 0.001
    env.scene["target"].write_root_pose_to_sim(marker_pose, env_ids=env_ids)


def update_history(env):
    cube = env.scene["cube"]
    success = env.pick_place_state.update(
        lift_height_m=cube.data.root_pos_w[:, 2] - env._so101_cube_initial_z,
        contact_forces_n=contact_forces(env), target_delta_m=cube_to_target(env),
        linear_speed_m_s=torch.linalg.vector_norm(cube.data.root_lin_vel_w, dim=1),
        angular_speed_rad_s=torch.linalg.vector_norm(cube.data.root_ang_vel_w, dim=1),
        ee_distance_m=ee_distance(env), gripper_open_fraction=gripper_open_fraction(env),
    )
    metrics = env.episode_metrics
    metrics["reached"] |= ee_distance(env) < 0.06
    metrics["contacted"] |= (contact_forces(env) > env.pick_place_spec["task"]["success"]["contact_threshold_n"]).all(dim=1)
    metrics["picked"] |= env.pick_place_state.picked
    metrics["at_target"] |= (env.pick_place_state.carry_valid &
        (torch.linalg.vector_norm(cube_to_target(env)[:, :2], dim=1) <=
         env.pick_place_spec["task"]["success"]["placement_xy_tolerance_m"]) &
        (cube_to_target(env)[:, 2].abs() <= env.pick_place_spec["task"]["success"]["placement_z_tolerance_m"]))
    metrics["released"] |= env.pick_place_state.released
    return success


def pick_place_success(env):
    # History is sampled at every physics substep, not only at the policy boundary.
    # Clone prevents automatic reset from erasing TerminationManager's retained term.
    return env.policy_success.clone()


def lift_with_contact(env):
    height = env.scene["cube"].data.root_pos_w[:, 2] - env._so101_cube_initial_z
    contact = (contact_forces(env) > env.pick_place_spec["task"]["success"]["contact_threshold_n"]).all(dim=1)
    return (height / env.pick_place_spec["task"]["success"]["minimum_delta_z_m"]).clamp(0, 1) * contact


def transport(env):
    delta = cube_to_target(env)
    return env.pick_place_state.carry_valid.float() * (1 - torch.tanh(
        torch.linalg.vector_norm(delta[:, :2], dim=1) / 0.10))


def placement(env):
    delta = cube_to_target(env)
    return env.pick_place_state.carry_valid.float() * (1 - torch.tanh(
        torch.linalg.vector_norm(delta, dim=1) / 0.06))


def release_and_retreat(env):
    at_target = torch.linalg.vector_norm(cube_to_target(env), dim=1) < 0.04
    return (env.pick_place_state.released.float() * at_target
            * gripper_open_fraction(env) * (ee_distance(env) / 0.08).clamp(0, 1))


def stable_placement(env):
    return (env.pick_place_state.stable_steps / env.pick_place_state.required_stable_steps).clamp(0, 1)


def terminal_success_bonus(env):
    # RewardManager multiplies rates by step_dt; cancel it for this one-shot event.
    return env.policy_success.float() / env.step_dt

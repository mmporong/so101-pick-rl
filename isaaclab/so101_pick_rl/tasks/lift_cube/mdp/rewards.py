"""Shaped rewards for scratch PPO without camera observations."""

from __future__ import annotations

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, FrameTransformer

from .state import cube_initial_height


def _ee_cube_delta(env, ee_frame_cfg: SceneEntityCfg, cube_cfg: SceneEntityCfg) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene.sensors[ee_frame_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]
    return cube.data.root_pos_w - ee_frame.data.target_pos_w[:, 0, :]


def reaching_cube(
    env,
    std: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    distance = torch.linalg.vector_norm(_ee_cube_delta(env, ee_frame_cfg, cube_cfg), dim=1)
    return 1.0 - torch.tanh(distance / std)


def gripper_cube_alignment(
    env,
    xy_std: float,
    desired_vertical_gap_m: float,
    z_std: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    delta = _ee_cube_delta(env, ee_frame_cfg, cube_cfg)
    xy_error = torch.linalg.vector_norm(delta[:, :2], dim=1)
    vertical_error = torch.abs(delta[:, 2] + desired_vertical_gap_m)
    return (1.0 - torch.tanh(xy_error / xy_std)) * (1.0 - torch.tanh(vertical_error / z_std))


def valid_finger_contact(
    env,
    threshold_n: float,
    fixed_sensor_cfg: SceneEntityCfg = SceneEntityCfg("fixed_finger_contact"),
    moving_sensor_cfg: SceneEntityCfg = SceneEntityCfg("moving_finger_contact"),
) -> torch.Tensor:
    fixed: ContactSensor = env.scene.sensors[fixed_sensor_cfg.name]
    moving: ContactSensor = env.scene.sensors[moving_sensor_cfg.name]
    fixed_force = torch.linalg.vector_norm(fixed.data.force_matrix_w, dim=-1).amax(dim=(1, 2))
    moving_force = torch.linalg.vector_norm(moving.data.force_matrix_w, dim=-1).amax(dim=(1, 2))
    return torch.logical_and(fixed_force > threshold_n, moving_force > threshold_n).float()


def lift_progress(
    env,
    target_delta_z_m: float,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    cube: RigidObject = env.scene[cube_cfg.name]
    initial_z = cube_initial_height(env)
    delta_z = cube.data.root_pos_w[:, 2] - initial_z
    return torch.clamp(delta_z / target_delta_z_m, min=0.0, max=1.0)


def object_lifted(
    env,
    minimum_delta_z_m: float,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    cube: RigidObject = env.scene[cube_cfg.name]
    initial_z = cube_initial_height(env)
    return (cube.data.root_pos_w[:, 2] - initial_z >= minimum_delta_z_m).float()

"""Privileged state observations in the order fixed by task_spec.json."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from .state import cube_initial_height


def joint_position_rad(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.joint_pos[:, asset_cfg.joint_ids]


def joint_velocity_rad_s(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.joint_vel[:, asset_cfg.joint_ids]


def previous_action_rad(env, action_name: str = "joint_position_delta") -> torch.Tensor:
    """Return the clipped delta that was actually accumulated by the action term."""
    return env.action_manager.get_term(action_name).processed_actions


def end_effector_to_cube_position_m(
    env,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene.sensors[ee_frame_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]
    return cube.data.root_pos_w - ee_frame.data.target_pos_w[:, 0, :]


def cube_height_relative_to_initial_m(
    env, cube_cfg: SceneEntityCfg = SceneEntityCfg("cube")
) -> torch.Tensor:
    cube: RigidObject = env.scene[cube_cfg.name]
    initial_z = cube_initial_height(env)
    return (cube.data.root_pos_w[:, 2] - initial_z).unsqueeze(-1)

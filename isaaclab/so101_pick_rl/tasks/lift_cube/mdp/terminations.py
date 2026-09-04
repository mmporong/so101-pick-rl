"""Termination conditions for finite-state safety and held lift success."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

from .state import cube_initial_height


def non_finite_state(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]
    finite = torch.isfinite(robot.data.joint_pos).all(dim=1)
    finite &= torch.isfinite(robot.data.joint_vel).all(dim=1)
    finite &= torch.isfinite(cube.data.root_state_w).all(dim=1)
    return torch.logical_not(finite)


def workspace_exit(
    env,
    xy_limit_m: float,
    minimum_z_m: float,
    maximum_z_m: float,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    cube: RigidObject = env.scene[cube_cfg.name]
    position = cube.data.root_pos_w - env.scene.env_origins
    outside_xy = torch.any(torch.abs(position[:, :2]) > xy_limit_m, dim=1)
    outside_z = torch.logical_or(position[:, 2] < minimum_z_m, position[:, 2] > maximum_z_m)
    return torch.logical_or(outside_xy, outside_z)


class lift_held_success(ManagerTermBase):
    """Require the lift threshold for consecutive policy steps."""

    def __init__(self, cfg: TerminationTermCfg, env) -> None:
        super().__init__(cfg, env)
        self._consecutive_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._required_steps = max(1, math.ceil(cfg.params["minimum_hold_seconds"] / env.step_dt))

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._consecutive_steps[env_ids] = 0

    def __call__(
        self,
        env,
        minimum_delta_z_m: float,
        minimum_hold_seconds: float,
        cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ) -> torch.Tensor:
        del minimum_hold_seconds
        cube: RigidObject = env.scene[cube_cfg.name]
        initial_z = cube_initial_height(env)
        above_threshold = cube.data.root_pos_w[:, 2] - initial_z >= minimum_delta_z_m
        self._consecutive_steps = torch.where(
            above_threshold, self._consecutive_steps + 1, torch.zeros_like(self._consecutive_steps)
        )
        return self._consecutive_steps >= self._required_steps

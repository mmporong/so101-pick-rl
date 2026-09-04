"""Reset events that preserve the task's per-episode initial height."""

from __future__ import annotations

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs.mdp.events import reset_root_state_uniform
from isaaclab.managers import SceneEntityCfg


def reset_cube_pose(
    env,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> None:
    """Reset the cube and capture its episode-specific reference height."""
    reset_root_state_uniform(env, env_ids, pose_range, velocity_range, asset_cfg)
    cube: RigidObject = env.scene[asset_cfg.name]
    if not hasattr(env, "_so101_cube_initial_z"):
        env._so101_cube_initial_z = torch.zeros(env.num_envs, device=env.device)
    env._so101_cube_initial_z[env_ids] = cube.data.root_pos_w[env_ids, 2]

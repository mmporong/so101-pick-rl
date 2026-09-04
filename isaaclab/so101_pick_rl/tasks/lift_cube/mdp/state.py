"""Validated per-episode state shared by observation, reward, and termination terms."""

from __future__ import annotations

import torch


def cube_initial_height(env) -> torch.Tensor:
    """Return the reset-time cube height or fail loudly when reset state is invalid."""
    if not hasattr(env, "_so101_cube_initial_z"):
        raise RuntimeError("cube initial height is unavailable; reset_cube_pose did not run")
    initial_z = env._so101_cube_initial_z
    if initial_z.shape != (env.num_envs,):
        raise RuntimeError(
            f"cube initial height has shape {tuple(initial_z.shape)}, expected ({env.num_envs},)"
        )
    if not bool(torch.isfinite(initial_z).all().item()):
        raise RuntimeError("cube initial height contains NaN or Inf")
    return initial_z

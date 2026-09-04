"""Action term implementing bounded incremental joint-position targets."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.managers import ActionTerm
from isaaclab.utils import configclass


class JointPositionDeltaAction(JointAction):
    """Accumulate position deltas and hold the resulting target between policy steps."""

    def __init__(self, cfg: "JointPositionDeltaActionCfg", env) -> None:
        super().__init__(cfg, env)
        self._target = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._target[env_ids] = self._asset.data.joint_pos[env_ids][:, self._joint_ids]

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        self._target += self.processed_actions
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        self._target.clamp_(min=limits[..., 0], max=limits[..., 1])

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._target, joint_ids=self._joint_ids)


@configclass
class JointPositionDeltaActionCfg(JointActionCfg):
    class_type: type[ActionTerm] = JointPositionDeltaAction

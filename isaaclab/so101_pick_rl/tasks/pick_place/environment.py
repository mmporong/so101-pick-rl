# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved. SPDX-License-Identifier: BSD-3-Clause
# Adapted step loop; license retained in docs/licenses/ISAACLAB-BSD-3-Clause.txt.
"""Per-environment target and history allocation before observation managers load."""

import json
import torch

from ...task_contract import REPOSITORY_ROOT
from ...pick_place_state import PickPlaceState
from ..lift_cube.environment import SO101LiftCubeEnv
from . import mdp


class SO101PickPlaceEnv(SO101LiftCubeEnv):
    def load_managers(self):
        self.pick_place_spec = json.loads((REPOSITORY_ROOT / "common" / "pick_place_spec.json").read_text())
        self.target_pos_w = self.scene.env_origins.clone()
        self.pick_place_state = PickPlaceState(
            self.num_envs, self.device, self.pick_place_spec["task"]["success"], self.physics_dt)
        self.history_updates = 0
        self.policy_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_metrics = {name: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                                for name in ("reached", "contacted", "picked", "at_target", "released")}
        self.last_episode_metrics = {name: value.clone() for name, value in self.episode_metrics.items()}
        super().load_managers()

    def step(self, action):
        """Isaac Lab 2.1.1 ManagerBasedRLEnv.step (BSD-3-Clause), with substep history.

        The only added physics-loop operation is update_history after scene.update.
        Keep this adapter aligned when upgrading the pinned Isaac Lab dependency.
        """
        self.action_manager.process_action(action.to(self.device))
        self.policy_success.zero_()
        self.recorder_manager.record_pre_step()
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)
            self.policy_success |= mdp.update_history(self)
            self.history_updates += 1

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)
        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _reset_idx(self, env_ids):
        for name, value in self.episode_metrics.items():
            self.last_episode_metrics[name][env_ids] = value[env_ids]
        self.policy_success[env_ids] = False
        super()._reset_idx(env_ids)

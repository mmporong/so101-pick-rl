"""Batched Pick & Place episode history; torch-only for CPU and CUDA regression tests."""

from __future__ import annotations

import math
import torch


class PickPlaceState:
    def __init__(self, num_envs: int, device, success: dict, step_dt: float):
        self.success_cfg = success
        self.required_lift_steps = math.ceil(success["grasp_hold_seconds"] / step_dt)
        self.required_stable_steps = math.ceil(success["minimum_hold_seconds"] / step_dt)
        self.picked = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.carry_valid = torch.zeros_like(self.picked)
        self.released = torch.zeros_like(self.picked)
        self.previous_release_ready = torch.zeros_like(self.picked)
        self.pregrasp_opened = torch.zeros_like(self.picked)
        self.grasp_sequence_valid = torch.zeros_like(self.picked)
        self.lift_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.stable_steps = torch.zeros_like(self.lift_steps)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for value in (self.picked, self.carry_valid, self.released, self.previous_release_ready,
                      self.lift_steps, self.stable_steps, self.pregrasp_opened, self.grasp_sequence_valid):
            value[ids] = 0

    def update(self, *, lift_height_m, contact_forces_n, target_delta_m,
               linear_speed_m_s, angular_speed_rad_s, ee_distance_m, gripper_open_fraction):
        cfg = self.success_cfg
        finite = (torch.isfinite(lift_height_m) & torch.isfinite(contact_forces_n).all(dim=1)
                  & torch.isfinite(target_delta_m).all(dim=1) & torch.isfinite(linear_speed_m_s)
                  & torch.isfinite(angular_speed_rad_s) & torch.isfinite(ee_distance_m)
                  & torch.isfinite(gripper_open_fraction))
        contact = (contact_forces_n > cfg["contact_threshold_n"]).all(dim=1) & finite
        no_contact = (contact_forces_n < cfg["release_contact_threshold_n"]).all(dim=1) & finite
        near_cube = (ee_distance_m <= cfg["pregrasp_maximum_ee_distance_m"]) & finite
        # An opening observed before contact is necessary, but does not prove
        # collision geometry or mechanically valid enclosure of the object.
        new_grasp = (self.pregrasp_opened & contact & near_cube
                     & (gripper_open_fraction <= cfg["maximum_grasp_open_fraction"]))
        self.grasp_sequence_valid = (self.grasp_sequence_valid & contact) | new_grasp
        self.pregrasp_opened &= near_cube & ~new_grasp
        self.pregrasp_opened |= (no_contact & near_cube
                                & (gripper_open_fraction >= cfg["pregrasp_open_fraction"]))
        lifted = self.grasp_sequence_valid & (lift_height_m >= cfg["minimum_delta_z_m"])
        self.lift_steps = torch.where(lifted, self.lift_steps + 1, 0)
        new_pick = self.lift_steps == self.required_lift_steps
        self.picked |= new_pick
        self.carry_valid |= new_pick

        above_target = ((torch.linalg.vector_norm(target_delta_m[:, :2], dim=1)
                         <= cfg["placement_xy_tolerance_m"]) & finite)
        at_target = above_target & (target_delta_m[:, 2].abs() <= cfg["placement_z_tolerance_m"])
        gentle = ((linear_speed_m_s <= cfg["maximum_linear_speed_m_s"])
                  & (angular_speed_rad_s <= cfg["maximum_angular_speed_rad_s"]) & finite)
        # One-finger contact is allowed only during gentle placement at the target.
        # Otherwise a qualified lift followed by pushing could impersonate carrying.
        unsafe_partial_contact = ~contact & ~no_contact & (~at_target | ~gentle)
        # Allow lowering into the goal footprint before entering its resting Z band.
        dragging_outside_target = ~above_target & (lift_height_m < cfg["minimum_carry_clearance_m"])
        self.carry_valid &= ~unsafe_partial_contact & ~dragging_outside_target & finite
        # Require a gentle, at-target contact sample BEFORE separation as well.
        # A throw can collide and settle between two policy samples.
        controlled_release = no_contact & self.previous_release_ready & self.carry_valid & at_target & gentle
        self.released |= controlled_release
        self.released &= no_contact & at_target & gentle & finite
        self.carry_valid &= ~(no_contact & ~self.released)
        self.released &= self.carry_valid
        stable = (self.released & at_target & gentle & no_contact
                  & (ee_distance_m >= cfg["minimum_ee_clearance_m"])
                  & (gripper_open_fraction >= cfg["minimum_gripper_open_fraction"]))
        self.stable_steps = torch.where(stable, self.stable_steps + 1, 0)
        self.previous_release_ready.copy_(~no_contact & at_target & gentle & finite)
        return self.stable_steps >= self.required_stable_steps

    def observation(self):
        return torch.stack((self.picked.float(), self.carry_valid.float(), self.released.float(),
                            self.previous_release_ready.float(),
                            (self.lift_steps / self.required_lift_steps).clamp(max=1),
                            (self.stable_steps / self.required_stable_steps).clamp(max=1),
                            self.pregrasp_opened.float(), self.grasp_sequence_valid.float()), dim=1)

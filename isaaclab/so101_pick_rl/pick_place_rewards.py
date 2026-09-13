"""Torch-only phase-gated shaping; sequence flags are not geometry validation."""

import torch


def phase_potentials(*, state, success_cfg, reward_cfg, ee_to_cube_m, target_delta_m,
                  lift_height_m, gripper_open_fraction, linear_speed_m_s, angular_speed_rad_s):
    """Cumulative stage components; completed stages remain at one, not zero.

    These values are potentials, NOT per-step rewards. Runtime takes signed
    discounted differences so holding or revisiting a pose cannot farm reward.
    """
    cfg, scales = success_cfg, reward_cfg["distance_scales_m"]
    ee_distance_m = torch.linalg.vector_norm(ee_to_cube_m, dim=1)
    xy_distance_m = torch.linalg.vector_norm(target_delta_m[:, :2], dim=1)
    finite = (torch.isfinite(ee_to_cube_m).all(dim=1) & torch.isfinite(target_delta_m).all(dim=1)
              & torch.isfinite(lift_height_m) & torch.isfinite(gripper_open_fraction)
              & torch.isfinite(linear_speed_m_s) & torch.isfinite(angular_speed_rad_s))
    above_target = xy_distance_m <= cfg["placement_xy_tolerance_m"]
    at_target = above_target & (target_delta_m[:, 2].abs() <= cfg["placement_z_tolerance_m"])
    gentle = ((linear_speed_m_s <= cfg["maximum_linear_speed_m_s"])
              & (angular_speed_rad_s <= cfg["maximum_angular_speed_rad_s"]))
    approach = ~state.picked & ~state.grasp_sequence_valid
    grasp = ~state.picked & state.grasp_sequence_valid
    carry = state.carry_valid & ~state.released
    release_ready = carry & at_target & gentle
    opened = (gripper_open_fraction / cfg["pregrasp_open_fraction"]).clamp(0, 1)
    xy_alignment = 1 - torch.tanh(torch.linalg.vector_norm(ee_to_cube_m[:, :2], dim=1) / scales["alignment_xy"])
    z_alignment = 1 - torch.tanh((ee_to_cube_m[:, 2] + scales["pregrasp_vertical_gap"]).abs() / scales["alignment_z"])
    terms = {
        "reaching_cube": approach * opened * (1 - torch.tanh(ee_distance_m / scales["reach"])),
        # Position alignment only; orientation requires a validated grasp-frame calibration.
        "gripper_alignment": approach * opened * xy_alignment * z_alignment,
        "finger_contact": grasp.float(),
        "lift_progress": grasp * (lift_height_m / cfg["minimum_delta_z_m"]).clamp(0, 1),
        "transport": (carry & ~above_target) * (1 - torch.tanh(xy_distance_m / scales["transport"])),
        "placement": (carry & above_target & ~release_ready) * (1 - torch.tanh(
            torch.linalg.vector_norm(target_delta_m, dim=1) / scales["placement"])),
        # Unlike the previous released-only term, this teaches the opening transition.
        "gripper_opening": release_ready * (gripper_open_fraction / cfg["minimum_gripper_open_fraction"]).clamp(0, 1),
        "release_and_retreat": (state.released & at_target & gentle) * opened
                               * (ee_distance_m / cfg["minimum_ee_clearance_m"]).clamp(0, 1),
        "stable_placement": (state.stable_steps / state.required_stable_steps).clamp(0, 1),
    }
    # Carry and release preserve earlier accomplishments. Each new stage adds
    # potential instead of dropping the previous stage's shaping at its boundary.
    valid_history = state.carry_valid
    completed = {
        "reaching_cube": state.grasp_sequence_valid | valid_history,
        "gripper_alignment": state.grasp_sequence_valid | valid_history,
        "finger_contact": valid_history,
        "lift_progress": valid_history,
        "transport": valid_history & above_target,
        "placement": release_ready | state.released,
        "gripper_opening": state.released,
    }
    for name, mask in completed.items():
        terms[name] = torch.where(mask, 1.0, terms[name])
    return {name: torch.where(finite, value, 0.0) for name, value in terms.items()}


def potential_difference(before, after, *, gamma, step_dt, terminated):
    """RewardManager multiplies by dt; true terminals have zero next potential.

    Timeout truncations retain next potential for value-function bootstrapping.
    Neither relu nor absolute value is valid here: a backward transition must
    repay its earlier progress. Initial/reset potentials are recomputed at step start.
    """
    result = {}
    for name, previous in before.items():
        following = after[name]
        finite = torch.isfinite(previous) & torch.isfinite(following)
        following = torch.where(terminated, 0.0, following)
        result[name] = torch.where(finite, (gamma * following - previous) / step_dt, 0.0)
    return result

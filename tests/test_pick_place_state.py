from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
from copy import deepcopy
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_ROOT = ROOT / "isaaclab"
sys.path.insert(0, str(ISAACLAB_ROOT))

from so101_pick_rl.pick_place_state import PickPlaceState


VALIDATOR_PATH = ROOT / "scripts" / "validate_contract.py"
VALIDATOR_SPEC = importlib.util.spec_from_file_location("validate_contract", VALIDATOR_PATH)
assert VALIDATOR_SPEC is not None and VALIDATOR_SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(VALIDATOR_SPEC)
VALIDATOR_SPEC.loader.exec_module(VALIDATOR)


class PickPlaceStateTest(unittest.TestCase):
    def setUp(self) -> None:
        contract = json.loads((ROOT / "common" / "pick_place_spec.json").read_text(encoding="utf-8"))
        self.success = contract["task"]["success"]
        self.step_dt = 0.1
        self.device = os.environ.get("SO101_TEST_DEVICE", "cpu")
        self.state = PickPlaceState(1, self.device, self.success, self.step_dt)

    def update(
        self,
        *,
        lift: float = 0.0,
        contact: tuple[float, float] = (0.0, 0.0),
        target: tuple[float, float, float] = (1.0, 1.0, 1.0),
        linear_speed: float = 0.0,
        angular_speed: float = 0.0,
        clearance: float = 0.0,
        gripper_open: float = 0.0,
    ) -> bool:
        success = self.state.update(
            lift_height_m=torch.tensor([lift], device=self.device),
            contact_forces_n=torch.tensor([contact], device=self.device),
            target_delta_m=torch.tensor([target], device=self.device),
            linear_speed_m_s=torch.tensor([linear_speed], device=self.device),
            angular_speed_rad_s=torch.tensor([angular_speed], device=self.device),
            ee_distance_m=torch.tensor([clearance], device=self.device),
            gripper_open_fraction=torch.tensor([gripper_open], device=self.device),
        )
        return bool(success.item())

    def establish_pick(self) -> None:
        for _ in range(self.state.required_lift_steps):
            self.update(lift=self.success["minimum_delta_z_m"], contact=(1.0, 1.0))
        self.assertTrue(self.state.picked.item())
        self.update(lift=self.success["minimum_delta_z_m"], contact=(1.0, 1.0), target=(0.0, 0.0, 0.0))

    def controlled_release(self, **overrides: float) -> bool:
        values = {
            "target": (0.0, 0.0, 0.0),
            "clearance": self.success["minimum_ee_clearance_m"],
            "gripper_open": self.success["minimum_gripper_open_fraction"],
        }
        values.update(overrides)
        return self.update(**values)

    def test_full_pick_place_sequence_succeeds_after_stable_release(self) -> None:
        self.establish_pick()
        self.update(lift=self.success["minimum_delta_z_m"], contact=(1.0, 1.0), target=(0.0, 0.0, 0.0))
        result = self.controlled_release()
        for _ in range(self.state.required_stable_steps - 1):
            result = self.controlled_release()
        self.assertTrue(result)

    def test_success_waits_for_exact_stability_timing_boundary(self) -> None:
        self.establish_pick()
        self.update(lift=self.success["minimum_delta_z_m"], contact=(1.0, 1.0), target=(0.0, 0.0, 0.0))
        results = [self.controlled_release() for _ in range(self.state.required_stable_steps)]
        self.assertEqual([False] * (self.state.required_stable_steps - 1) + [True], results)

    def test_partial_reset_clears_only_selected_environment(self) -> None:
        state = PickPlaceState(2, "cpu", self.success, self.step_dt)
        for _ in range(state.required_lift_steps):
            state.update(
                lift_height_m=torch.tensor([0.08, 0.08]),
                contact_forces_n=torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
                target_delta_m=torch.zeros((2, 3)),
                linear_speed_m_s=torch.zeros(2),
                angular_speed_rad_s=torch.zeros(2),
                ee_distance_m=torch.zeros(2),
                gripper_open_fraction=torch.zeros(2),
            )
        state.reset(torch.tensor([0]))
        phase_state = state.observation()
        self.assertEqual([0.0] * 6, phase_state[0].tolist())
        self.assertEqual([1.0, 1.0, 0.0, 1.0, 1.0, 0.0], phase_state[1].tolist())

    def test_observation_includes_previous_release_ready_in_six_phase_values(self) -> None:
        self.update(lift=self.success["minimum_delta_z_m"], contact=(1.0, 1.0), target=(0.0, 0.0, 0.0))
        phase_state = self.state.observation()
        self.assertEqual((1, 6), tuple(phase_state.shape))
        self.assertEqual(1.0, phase_state[0, 3].item())

    def test_push_to_target_never_counts_as_pick_place(self) -> None:
        results = [self.controlled_release() for _ in range(self.state.required_stable_steps + 1)]
        self.assertFalse(any(results))

    def test_high_speed_release_cannot_succeed_after_cube_settles(self) -> None:
        self.establish_pick()
        self.update(lift=self.success["minimum_delta_z_m"], contact=(1.0, 1.0), target=(0.0, 0.0, 0.0))
        self.controlled_release(linear_speed=self.success["maximum_linear_speed_m_s"] + 0.01)
        results = [self.controlled_release() for _ in range(self.state.required_stable_steps + 1)]
        self.assertFalse(any(results))

    def test_partial_bilateral_loss_followed_by_fast_full_release_cannot_succeed(self) -> None:
        self.establish_pick()
        self.update(
            contact=(0.19, 1.0),
            target=(0.0, 0.0, 0.0),
            clearance=self.success["minimum_ee_clearance_m"],
            gripper_open=1.0,
        )
        self.controlled_release(linear_speed=self.success["maximum_linear_speed_m_s"] + 0.01)
        results = [self.controlled_release() for _ in range(self.state.required_stable_steps + 1)]
        self.assertFalse(any(results))

    def test_partial_bilateral_loss_followed_by_gentle_full_release_can_succeed(self) -> None:
        self.establish_pick()
        self.update(
            contact=(0.19, 1.0),
            target=(0.0, 0.0, 0.0),
            clearance=self.success["minimum_ee_clearance_m"],
            gripper_open=1.0,
        )
        results = [self.controlled_release() for _ in range(self.state.required_stable_steps)]
        self.assertTrue(results[-1])

    def test_losing_grip_before_target_invalidates_carry_history(self) -> None:
        self.establish_pick()
        self.update(lift=self.success["minimum_delta_z_m"], contact=(1.0, 1.0))
        self.update()
        self.controlled_release()
        self.assertFalse(self.state.carry_valid.item())

    def test_one_finger_push_after_pick_cannot_succeed(self) -> None:
        self.establish_pick()
        self.update(contact=(0.0, 1.0), target=(0.2, 0.0, 0.0))
        self.update(contact=(0.0, 1.0), target=(0.0, 0.0, 0.0))
        results = [self.controlled_release() for _ in range(self.state.required_stable_steps + 1)]
        self.assertFalse(any(results))
        self.assertFalse(self.state.carry_valid.item())

    def test_bilateral_drag_after_pick_cannot_succeed(self) -> None:
        self.establish_pick()
        self.update(lift=0.0, contact=(1.0, 1.0), target=(0.2, 0.0, 0.0))
        self.update(lift=0.0, contact=(1.0, 1.0), target=(0.0, 0.0, 0.0))
        results = [self.controlled_release() for _ in range(self.state.required_stable_steps + 1)]
        self.assertFalse(any(results))

    def test_continuous_lowering_above_goal_preserves_carry_into_rest_band(self) -> None:
        self.establish_pick()
        # Initial center is 22 mm and target center 20 mm. Traverse the previous
        # 30--32 mm dead band without needing an unphysical jump between samples.
        for cube_z in (0.040, 0.033, 0.032, 0.0315, 0.031, 0.0305, 0.030, 0.025, 0.020):
            self.update(lift=cube_z - 0.022, contact=(1.0, 1.0), target=(0.0, 0.0, 0.020 - cube_z))
            self.assertTrue(self.state.carry_valid.item(), msg=f"cube_z={cube_z}")
        results = [self.controlled_release() for _ in range(self.state.required_stable_steps)]
        self.assertTrue(results[-1])

    def test_throw_settling_between_samples_cannot_succeed(self) -> None:
        self.establish_pick()
        self.update(lift=0.08, contact=(1.0, 1.0), target=(0.2, 0.0, 0.0))
        results = [self.controlled_release() for _ in range(self.state.required_stable_steps + 1)]
        self.assertFalse(any(results))

    def test_post_release_bounce_cannot_requalify_without_new_pick(self) -> None:
        self.establish_pick()
        self.controlled_release()
        self.controlled_release(angular_speed=1.0)
        results = [self.controlled_release() for _ in range(self.state.required_stable_steps + 1)]
        self.assertFalse(any(results))

    def test_fast_substep_between_release_ready_policy_samples_is_rejected(self) -> None:
        self.state = PickPlaceState(1, self.device, self.success, 1 / 120)
        self.assertEqual(24, self.state.required_lift_steps)
        self.assertEqual(120, self.state.required_stable_steps)
        self.establish_pick()
        # Four 120 Hz substeps between two 30 Hz policy boundaries.
        self.controlled_release(linear_speed=0.5)
        for _ in range(3):
            self.controlled_release()
        self.assertFalse(self.state.carry_valid.item())
        results = [self.controlled_release() for _ in range(self.state.required_stable_steps + 1)]
        self.assertFalse(any(results))

    def test_fast_partial_contact_at_target_invalidates_carry(self) -> None:
        self.establish_pick()
        self.update(contact=(0.0, 1.0), target=(0.0, 0.0, 0.0), linear_speed=0.1)
        results = [self.controlled_release() for _ in range(self.state.required_stable_steps + 1)]
        self.assertFalse(any(results))

    def test_angular_toss_cannot_succeed_after_settling(self) -> None:
        self.establish_pick()
        self.controlled_release(angular_speed=self.success["maximum_angular_speed_rad_s"] + 0.01)
        results = [self.controlled_release() for _ in range(self.state.required_stable_steps + 1)]
        self.assertFalse(any(results))

    def test_release_above_target_cannot_succeed_after_settling(self) -> None:
        self.establish_pick()
        self.controlled_release(target=(0.0, 0.0, self.success["placement_z_tolerance_m"] + 0.001))
        results = [self.controlled_release() for _ in range(self.state.required_stable_steps + 1)]
        self.assertFalse(any(results))

    def test_contact_while_held_at_target_cannot_succeed(self) -> None:
        self.establish_pick()
        results = [
            self.update(
                lift=self.success["minimum_delta_z_m"],
                contact=(1.0, 1.0),
                target=(0.0, 0.0, 0.0),
                clearance=self.success["minimum_ee_clearance_m"],
                gripper_open=1.0,
            )
            for _ in range(self.state.required_stable_steps + 1)
        ]
        self.assertFalse(any(results))

    def test_clearance_threshold_is_inclusive(self) -> None:
        self.establish_pick()
        self.update(lift=self.success["minimum_delta_z_m"], contact=(1.0, 1.0), target=(0.0, 0.0, 0.0))
        results = [self.controlled_release(gripper_open=1.0) for _ in range(self.state.required_stable_steps)]
        self.assertTrue(results[-1])

    def test_clearance_below_threshold_prevents_success(self) -> None:
        self.establish_pick()
        self.update(lift=self.success["minimum_delta_z_m"], contact=(1.0, 1.0), target=(0.0, 0.0, 0.0))
        results = [
            self.controlled_release(clearance=self.success["minimum_ee_clearance_m"] - 0.001)
            for _ in range(self.state.required_stable_steps + 1)
        ]
        self.assertFalse(any(results))

    def test_gripper_open_threshold_is_inclusive(self) -> None:
        self.establish_pick()
        self.update(lift=self.success["minimum_delta_z_m"], contact=(1.0, 1.0), target=(0.0, 0.0, 0.0))
        results = [
            self.controlled_release(clearance=self.success["minimum_ee_clearance_m"] + 0.01)
            for _ in range(self.state.required_stable_steps)
        ]
        self.assertTrue(results[-1])

    def test_gripper_below_open_threshold_prevents_success(self) -> None:
        self.establish_pick()
        self.update(lift=self.success["minimum_delta_z_m"], contact=(1.0, 1.0), target=(0.0, 0.0, 0.0))
        results = [
            self.controlled_release(gripper_open=self.success["minimum_gripper_open_fraction"] - 0.001)
            for _ in range(self.state.required_stable_steps + 1)
        ]
        self.assertFalse(any(results))

    def test_nan_input_never_reports_success(self) -> None:
        self.establish_pick()
        self.update(lift=self.success["minimum_delta_z_m"], contact=(1.0, 1.0), target=(0.0, 0.0, 0.0))
        self.controlled_release()
        results = [self.controlled_release(linear_speed=float("nan")) for _ in range(self.state.required_stable_steps)]
        self.assertFalse(any(results))


class PickPlaceContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = json.loads((ROOT / "common" / "pick_place_spec.json").read_text(encoding="utf-8"))

    def test_repository_pick_place_contract_is_valid(self) -> None:
        self.assertEqual([], VALIDATOR.validate_pick_place_spec(self.contract))

    def test_non_finite_success_threshold_is_rejected(self) -> None:
        broken = deepcopy(self.contract)
        broken["task"]["success"]["minimum_ee_clearance_m"] = float("nan")
        errors = VALIDATOR.validate_pick_place_spec(broken)
        self.assertIn("success.minimum_ee_clearance_m must be finite and positive", errors)

    def test_non_positive_success_threshold_is_rejected(self) -> None:
        broken = deepcopy(self.contract)
        broken["task"]["success"]["maximum_linear_speed_m_s"] = 0.0
        errors = VALIDATOR.validate_pick_place_spec(broken)
        self.assertIn("success.maximum_linear_speed_m_s must be finite and positive", errors)

    def test_observation_dimension_must_equal_term_dimension_sum(self) -> None:
        broken = deepcopy(self.contract)
        broken["observation"]["dimension"] = 38
        errors = VALIDATOR.validate_pick_place_spec(broken)
        self.assertIn("observation.dimension must equal the sum of term_dimensions", errors)

    def test_episode_phase_state_must_have_six_dimensions(self) -> None:
        broken = deepcopy(self.contract)
        phase_index = broken["observation"]["terms"].index("episode_phase_state")
        broken["observation"]["term_dimensions"][phase_index] = 5
        broken["observation"]["dimension"] = 38
        errors = VALIDATOR.validate_pick_place_spec(broken)
        self.assertIn("observation.episode_phase_state must have dimension 6", errors)

    def test_required_history_flag_cannot_be_disabled(self) -> None:
        broken = deepcopy(self.contract)
        broken["task"]["success"]["requires_grasped_lift_history"] = False
        errors = VALIDATOR.validate_pick_place_spec(broken)
        self.assertIn("success.requires_grasped_lift_history must be true", errors)

    def test_required_controlled_release_flag_cannot_be_disabled(self) -> None:
        broken = deepcopy(self.contract)
        broken["task"]["success"]["requires_controlled_release_at_target"] = False
        errors = VALIDATOR.validate_pick_place_spec(broken)
        self.assertIn("success.requires_controlled_release_at_target must be true", errors)

    def test_terminal_bonus_dominates_dense_plateau_with_discount(self) -> None:
        reward = self.contract["reward"]
        maximum_dense_rate = sum(reward["positive_rates"].values())
        dt = 1 / self.contract["control"]["policy_hz"]
        gamma = reward["discount_gamma"]
        bonus = reward["terminal_success_bonus"]
        self.assertGreater(bonus, maximum_dense_rate * dt / (1 - gamma))
        self.assertGreater(bonus * (1 - gamma), maximum_dense_rate * dt)

    def test_invalid_reward_and_phase_contracts_are_rejected(self) -> None:
        mutations = (
            ("task", "revision", 1),
            ("reward", "discount_gamma", 1.0),
            ("reward", "terminal_success_bonus", 1.0),
            ("reward", "positive_rates", {"transport": float("nan")}),
            ("observation", "episode_phase_fields", ["old_contact_flag"] * 6),
        )
        for section, key, value in mutations:
            with self.subTest(key=key):
                broken = deepcopy(self.contract)
                broken[section][key] = value
                self.assertTrue(VALIDATOR.validate_pick_place_spec(broken))


if __name__ == "__main__":
    unittest.main()

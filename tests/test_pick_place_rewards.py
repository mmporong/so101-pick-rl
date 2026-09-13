import json
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaaclab"))
from so101_pick_rl.pick_place_rewards import phase_potentials, potential_difference
from so101_pick_rl.pick_place_state import PickPlaceState


class PhaseRewardTest(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads((ROOT / "common/pick_place_spec.json").read_text())
        self.state = PickPlaceState(1, "cpu", self.spec["task"]["success"], 1 / 120)

    def rewards(self, *, target=(0.1, 0.0, -0.08), ee=(0.0, 0.0, -0.04),
                opened=1.0, speed=0.0, lift=0.08):
        return phase_potentials(
            state=self.state, success_cfg=self.spec["task"]["success"], reward_cfg=self.spec["reward"],
            ee_to_cube_m=torch.tensor([ee]), target_delta_m=torch.tensor([target]),
            lift_height_m=torch.tensor([lift]), gripper_open_fraction=torch.tensor([opened]),
            linear_speed_m_s=torch.tensor([speed]), angular_speed_rad_s=torch.zeros(1))

    def carry(self):
        self.state.picked[:] = True
        self.state.carry_valid[:] = True
        self.state.grasp_sequence_valid[:] = True

    def test_closed_approach_has_no_positive_shaping(self):
        self.assertTrue(all(value.item() == 0 for value in self.rewards(opened=0.0).values()))

    def test_opening_helps_approach(self):
        self.assertGreater(self.rewards(opened=1.0)["reaching_cube"].item(),
                           self.rewards(opened=0.3)["reaching_cube"].item())

    def test_contact_without_sequence_is_not_rewarded(self):
        self.assertEqual(0, self.rewards()["finger_contact"].item())

    def test_pick_freezes_early_potentials_so_they_cannot_be_earned_again(self):
        self.carry()
        terms = self.rewards()
        for name in ("reaching_cube", "gripper_alignment", "finger_contact", "lift_progress"):
            self.assertEqual(1, terms[name].item(), name)
        self.assertGreater(terms["transport"].item(), 0)

    def test_transport_requires_valid_carry(self):
        self.carry()
        self.state.carry_valid[:] = False
        self.assertEqual(0, self.rewards()["transport"].item())

    def test_arriving_above_target_switches_from_transport_to_lowering(self):
        self.carry()
        terms = self.rewards(target=(0.0, 0.0, -0.08))
        self.assertEqual(1, terms["transport"].item())
        self.assertGreater(terms["placement"].item(), 0)
        self.assertEqual(0, terms["gripper_opening"].item())

    def test_gentle_target_pose_rewards_opening_before_release(self):
        self.carry()
        low = self.rewards(target=(0.0, 0.0, 0.0), opened=0.1)
        high = self.rewards(target=(0.0, 0.0, 0.0), opened=0.5)
        self.assertFalse(self.state.released.item())
        self.assertGreater(high["gripper_opening"].item(), low["gripper_opening"].item())
        self.assertEqual(1, high["placement"].item())
        self.assertEqual(0, high["release_and_retreat"].item())

    def test_fast_or_airborne_opening_is_not_rewarded(self):
        self.carry()
        self.assertEqual(0, self.rewards(target=(0.0, 0.0, 0.0), speed=0.2)["gripper_opening"].item())
        self.assertEqual(0, self.rewards(target=(0.0, 0.0, -0.1))["gripper_opening"].item())

    def test_released_phase_only_rewards_retreat_and_stability(self):
        self.carry()
        self.state.released[:] = True
        terms = self.rewards(target=(0.0, 0.0, 0.0), ee=(0.0, 0.0, -0.1))
        for name in ("transport", "placement", "gripper_opening"):
            self.assertEqual(1, terms[name].item())
        self.assertGreater(terms["release_and_retreat"].item(), 0)

    def test_nonfinite_inputs_fail_closed_and_batched_envs_are_independent(self):
        self.assertTrue(all(torch.isfinite(value).all() and value.item() == 0
                            for value in self.rewards(opened=float("nan")).values()))
        self.state = PickPlaceState(2, "cpu", self.spec["task"]["success"], 1 / 120)
        self.state.picked[1] = True
        terms = self.rewards()
        self.assertGreater(terms["reaching_cube"][0].item(), 0)
        self.assertEqual(0, terms["reaching_cube"][1].item())

    def difference(self, before, after, terminated=False):
        return potential_difference(before, after, gamma=0.99, step_dt=1 / 30,
                                    terminated=torch.tensor([terminated]))

    def test_holding_any_stage_cannot_farm_positive_reward(self):
        self.carry()
        value = self.rewards(target=(0.0, 0.0, 0.0), opened=0.5)
        self.assertTrue(all(term.item() <= 0 for term in self.difference(value, value).values()))

    def test_discounted_round_trip_cannot_farm_progress(self):
        self.carry()
        a = self.rewards(target=(0.1, 0.0, -0.08))
        b = self.rewards(target=(0.0, 0.0, -0.08))
        forward, backward = self.difference(a, b), self.difference(b, a)
        for name in a:
            self.assertLessEqual((forward[name] + 0.99 * backward[name]).item(), 1e-5)

    def test_opening_transition_is_positive_before_release(self):
        self.carry()
        before = self.rewards(target=(0.0, 0.0, 0.0), opened=0.1)
        after = self.rewards(target=(0.0, 0.0, 0.0), opened=0.5)
        self.assertGreater(self.difference(before, after)["gripper_opening"].item(), 0)

    def test_termination_and_timeout_use_different_potentials(self):
        before, after = {"x": torch.tensor([0.4])}, {"x": torch.tensor([0.8])}
        self.assertAlmostEqual(-12.0, self.difference(before, after, True)["x"].item(), places=5)
        self.assertGreater(self.difference(before, after, False)["x"].item(), 0)

    def test_carry_loss_repays_progress(self):
        self.carry()
        before = self.rewards()
        self.state.carry_valid[:] = False
        self.state.grasp_sequence_valid[:] = False
        after = self.rewards()
        self.assertLess(sum(term.item() for term in self.difference(before, after).values()), 0)

    def test_difference_rejects_nonfinite_and_keeps_batch_slots_independent(self):
        terms = potential_difference({"x": torch.tensor([0.2, 0.2, float("nan")])},
                                     {"x": torch.tensor([0.4, 0.4, 0.4])}, gamma=0.99, step_dt=1 / 30,
                                     terminated=torch.tensor([False, True, False]))["x"]
        self.assertGreater(terms[0].item(), 0)
        self.assertLess(terms[1].item(), 0)
        self.assertEqual(0, terms[2].item())


if __name__ == "__main__":
    unittest.main()

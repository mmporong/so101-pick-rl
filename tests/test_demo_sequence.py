import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaaclab"))
from so101_pick_rl.demo_sequence import DemoSequenceGate, force_sum
from so101_pick_rl.demo_place_controller import ControlledPlace, bounded_cartesian_step, midpoint_jacobian


class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT / "configs/evaluation/demo_box_replay_gate.json").read_text())
        self.geometry = json.loads((ROOT / "configs/isaaclab/demo_box_pad_geometry.json").read_text())
        self.gate = DemoSequenceGate(self.cfg, [0, 0, 0])

    def feature(self, **changes):
        f = dict(penetration_safe=True, wrist_safe=True, hand_box_clear=True, distance_m=.04, opposing_sides=False,
                 bilateral=False, no_contact=True, inside_pad_planes=True, opened=True,
                 cube_position_m=np.zeros(3), lift_m=0., above_box=False, at_rest=False,
                 gentle=True, supported=False, approach_world_z=-1., gripper_opening=False)
        f["support_force_n"] = .1
        f.update(changes)
        return f

    def establish_pick(self):
        self.gate.update(self.feature())
        for _ in range(12):
            self.gate.update(self.feature(opposing_sides=True, bilateral=True, no_contact=False,
                                          opened=False, lift_m=.09))
        self.assertTrue(self.gate.picked)

    def test_supported_place_open_retreat_then_hold_passes(self):
        self.establish_pick()
        self.gate.update(self.feature(bilateral=True, no_contact=False, opened=False,
                                      at_rest=True, above_box=True, supported=True))
        self.gate.update(self.feature(at_rest=True, above_box=True, supported=True, gripper_opening=True))
        for _ in range(59):
            self.assertFalse(self.gate.update(self.feature(at_rest=True, above_box=True, supported=True, distance_m=.10)))
        self.assertTrue(self.gate.update(self.feature(at_rest=True, above_box=True, supported=True, distance_m=.10)))
        self.assertTrue(self.gate.report()["normal_grasp_gate_pass"])

    def test_high_drop_then_settle_never_passes(self):
        self.establish_pick()
        self.gate.update(self.feature(above_box=True, lift_m=.09, gripper_opening=True))
        for _ in range(120):
            self.gate.update(self.feature(at_rest=True, above_box=True, supported=True, distance_m=.10))
        self.assertFalse(self.gate.report()["normal_grasp_gate_pass"])
        self.assertIn("unsupported_or_fast_release", self.gate.failures)

    def test_no_open_before_contact_cannot_pick(self):
        for _ in range(40):
            self.gate.update(self.feature(bilateral=True, no_contact=False, opposing_sides=True, lift_m=.09))
        self.assertFalse(self.gate.picked)

    def test_no_enclosure_or_missing_side_contact_cannot_pick(self):
        for key in ("inside_pad_planes", "opposing_sides"):
            self.setUp()
            self.gate.update(self.feature())
            for _ in range(40):
                f = self.feature(bilateral=True, no_contact=False, opposing_sides=True, lift_m=.09, opened=False)
                f[key] = False
                self.gate.update(f)
            self.assertFalse(self.gate.picked)

    def test_unsafe_carry_and_penetration_stay_failed(self):
        for update in (dict(penetration_safe=False), dict(wrist_safe=False), dict(hand_box_clear=False),
                       dict(inside_pad_planes=False), dict(approach_world_z=1.)):
            self.setUp()
            self.establish_pick()
            self.gate.update(self.feature(bilateral=True, no_contact=False, lift_m=.09, **update))
            self.assertTrue(self.gate.failures)

    def test_closed_retraction_not_controlled_open_release(self):
        self.establish_pick()
        self.gate.update(self.feature(bilateral=True, no_contact=False, above_box=True, at_rest=True, supported=True))
        self.gate.update(self.feature(above_box=True, at_rest=True, supported=True, gripper_opening=False))
        self.assertFalse(self.gate.released)

    def test_contact_gap_while_airborne_invalidates(self):
        self.establish_pick()
        self.gate.update(self.feature(lift_m=.09))
        self.assertIn("grasp_lost_before_supported_placement", self.gate.failures)

    def test_open_aperture_contact_never_arms_grasp(self):
        self.gate.update(self.feature())
        for _ in range(40):
            self.gate.update(self.feature(bilateral=True, no_contact=False, opposing_sides=True, lift_m=.09))
        self.assertFalse(self.gate.picked)

    def test_initial_position_must_be_finite_vector(self):
        for position in ([float("nan"), 0, 0], [0, 0], 0):
            with self.assertRaises(ValueError):
                DemoSequenceGate(self.cfg, position)

    def test_supported_but_fast_release_never_passes(self):
        self.establish_pick()
        self.gate.update(self.feature(bilateral=True, no_contact=False, at_rest=True, above_box=True,
                                      supported=True, gentle=False))
        self.gate.update(self.feature(at_rest=True, above_box=True, supported=True, gripper_opening=True))
        self.assertFalse(self.gate.released)
        self.assertIn("unsupported_or_fast_release", self.gate.failures)

    def test_bounce_after_success_invalidates_result(self):
        self.test_supported_place_open_retreat_then_hold_passes()
        self.gate.update(self.feature(at_rest=True, above_box=True, supported=True, gentle=False))
        self.assertFalse(self.gate.report()["normal_grasp_gate_pass"])

    def test_missing_force_pair_and_nonfinite_force_rejected(self):
        with self.assertRaises(KeyError):
            force_sum({}, "cube_box")
        with self.assertRaises(ValueError):
            force_sum({"cube_box": [{"normal_force_n": [float("nan")]}]}, "cube_box")

    def test_controller_does_not_replace_prefix_without_loaded_grasp(self):
        controller = ControlledPlace(self.geometry, self.cfg)
        target = np.arange(6.) / 10
        actual = controller.action(proposed_target_rad=target, previous_target_rad=np.zeros(6),
            q_rad=np.zeros(6), cube_position_m=np.zeros(3), box_position_m=np.zeros(3),
            feature=self.feature(), body_quaternions=np.tile([1, 0, 0, 0], (2, 1)),
            spatial_jacobians=np.zeros((2, 6, 6)), lower_rad=-np.ones(6), upper_rad=np.ones(6))
        np.testing.assert_array_equal(actual, target)
        self.assertFalse(controller.active)

    def test_bounded_ik_step_and_point_jacobian(self):
        jac = np.zeros((3, 5))
        jac[:3, :3] = np.eye(3)
        step = bounded_cartesian_step(jac, np.array([1., 0., 0.]), np.zeros((3, 5)), [1, 0, 0, 0], [1, 0, 0, 0])
        self.assertGreater(step[0], 0)
        self.assertLessEqual(np.abs(step).max(), .006)
        spatial = np.zeros((2, 6, 6))
        spatial[:, :3, :3] = np.eye(3)
        np.testing.assert_array_equal(midpoint_jacobian(np.tile([1,0,0,0], (2,1)), spatial, self.geometry), jac)

    def test_vertical_tilt_step_has_correct_sign_and_ignores_yaw(self):
        angular = np.zeros((3, 5))
        angular[:3, :3] = np.eye(3)
        tilt = .3
        step = bounded_cartesian_step(np.zeros((3, 5)), np.zeros(3), angular,
            [np.cos(tilt/2), np.sin(tilt/2), 0, 0], [1,0,0,0], vertical_axis=True)
        self.assertLess(step[0], 0)
        yaw_only = bounded_cartesian_step(np.zeros((3, 5)), np.zeros(3), angular,
            [np.cos(tilt/2), 0, 0, np.sin(tilt/2)], [1,0,0,0], vertical_axis=True)
        np.testing.assert_allclose(yaw_only, 0, atol=1e-12)

    def test_saturated_joint_is_resolved_using_available_joint(self):
        jac = np.zeros((3, 5))
        jac[0, :2] = 1
        lower, upper = np.full(5, -.006), np.full(5, .006)
        upper[0] = 0
        step = bounded_cartesian_step(jac, np.array([.0005,0,0]), np.zeros((3,5)),
            [1,0,0,0], [1,0,0,0], delta_lower_rad=lower, delta_upper_rad=upper)
        self.assertEqual(step[0], 0)
        self.assertGreater(step[1], .00049)

    def test_place_offset_cannot_escape_target_region(self):
        for offset in ([.1,0], [float("nan"),0], [0]):
            with self.assertRaises(ValueError):
                ControlledPlace(self.geometry, self.cfg, offset)

    def test_physics_substeps_preserve_hold_duration(self):
        self.cfg["physics_dt_s"] = 1/240
        gate = DemoSequenceGate(self.cfg, [0,0,0])
        self.assertEqual(gate.required_lift_steps, 48)
        self.assertEqual(gate.required_stable_steps, 240)

    def test_cube_leveling_does_not_freeze_at_tilted_first_support(self):
        controller = ControlledPlace(self.geometry, self.cfg, level_cube=True)
        controller.active = True
        controller.phase = "lower_closed"
        controller.target_rad = np.zeros(6)
        controller.previous_q_rad = np.zeros(6)
        controller.reference_quaternion = np.array([1,0,0,0])
        arguments = dict(proposed_target_rad=np.zeros(6), previous_target_rad=np.zeros(6),
            q_rad=np.zeros(6), cube_position_m=np.array([0,0,.019]), box_position_m=np.zeros(3),
            feature=self.feature(supported=True, at_rest=True, cube_half_height_m=.015),
            body_quaternions=np.tile([1,0,0,0], (2,1)), spatial_jacobians=np.zeros((2,6,6)),
            lower_rad=-np.ones(6), upper_rad=np.ones(6))
        controller.action(**arguments, cube_quaternion=[np.cos(.15),np.sin(.15),0,0])
        self.assertEqual(controller.phase, "lower_closed")
        controller.action(**arguments, cube_quaternion=[1,0,0,0])
        self.assertEqual(controller.phase, "support_closed")

    def test_retreat_requires_geometric_opening_not_maximum_joint_angle(self):
        controller = ControlledPlace(self.geometry, self.cfg)
        controller.active = True
        controller.phase = "open_supported"
        controller.target_rad = np.array([0,0,0,0,0,.4])
        controller.previous_q_rad = controller.target_rad.copy()
        args = dict(proposed_target_rad=controller.target_rad, previous_target_rad=controller.target_rad,
            q_rad=controller.target_rad, cube_position_m=np.zeros(3), box_position_m=np.zeros(3),
            feature=self.feature(supported=True, opened=True, midpoint_w_m=np.zeros(3)),
            body_quaternions=np.tile([1,0,0,0],(2,1)), spatial_jacobians=np.zeros((2,6,6)),
            lower_rad=-np.ones(6)*2, upper_rad=np.ones(6)*2)
        for _ in range(12):
            controller.action(**args)
        self.assertEqual(controller.phase, "retreat_open")
        self.assertLess(controller.target_rad[-1], 1.15)

    def test_support_loss_returns_to_lowering_without_rearming_high_grip_force(self):
        controller = ControlledPlace(self.geometry, self.cfg, minimum_support_force_n=.08)
        controller.active = True
        controller._transition("support_closed")
        controller.target_rad = np.zeros(6)
        result = controller.action(proposed_target_rad=np.zeros(6), previous_target_rad=np.zeros(6),
            q_rad=np.zeros(6), cube_position_m=np.zeros(3), box_position_m=np.zeros(3),
            feature=self.feature(supported=True, support_force_n=.03),
            body_quaternions=np.tile([1,0,0,0],(2,1)), spatial_jacobians=np.zeros((2,6,6)),
            lower_rad=-np.ones(6), upper_rad=np.ones(6))
        self.assertEqual(controller.phase, "lower_closed")
        self.assertTrue(controller.support_started)
        np.testing.assert_array_equal(result, np.zeros(6))


if __name__ == "__main__":
    unittest.main()

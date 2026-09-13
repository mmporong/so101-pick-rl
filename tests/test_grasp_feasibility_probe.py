from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "isaaclab" / "scripts" / "probe_grasp_feasibility.py"
SPEC = importlib.util.spec_from_file_location("probe_grasp_feasibility", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class GraspFeasibilityProbeTests(unittest.TestCase):
    def test_point_jacobian_accounts_for_rotational_offset(self):
        spatial = torch.zeros((6, 1), dtype=torch.float64)
        spatial[5, 0] = 1.0
        result = PROBE.point_linear_jacobian(
            spatial, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        )
        torch.testing.assert_close(result[:, 0], torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64))

    def test_damped_step_is_bounded_and_reduces_identity_error(self):
        jacobian = torch.eye(3, dtype=torch.float64)
        error = torch.tensor([1.0, -0.5, 0.1], dtype=torch.float64)
        step = PROBE.damped_position_step(
            jacobian, error, damping=0.01, gain=1.0, max_joint_step_rad=0.2
        )
        self.assertLessEqual(float(step.abs().max()), 0.2)
        self.assertLess(torch.linalg.vector_norm(error - jacobian @ step), torch.linalg.vector_norm(error))

    def test_posture_bias_requires_state(self):
        with self.assertRaises(ValueError):
            PROBE.damped_position_step(
                torch.eye(3), torch.ones(3), damping=0.1, gain=1.0,
                max_joint_step_rad=0.1, posture_gain=0.1,
            )

    def test_collision_offsets_reject_inverted_pair(self):
        with self.assertRaises(ValueError):
            PROBE.validate_collision_offsets(0.001, 0.002)
        PROBE.validate_collision_offsets(0.003, 0.0)

    def test_no_contact_remains_none(self):
        result = PROBE.contact_extrema({"gripper_cube": []})
        self.assertEqual(result["gripper_cube"]["contact_count"], 0)
        self.assertIsNone(result["gripper_cube"]["minimum_separation_m"])
        self.assertIsNone(result["gripper_cube"]["peak_normal_force_n"])

    def test_contact_extrema_preserve_negative_separation(self):
        rows = [{"separation_m": [-0.002, 0.001], "normal_force_n": [2.0, 1.0]}]
        result = PROBE.contact_extrema({"jaw_cube": rows})["jaw_cube"]
        self.assertEqual(result["minimum_separation_m"], -0.002)
        self.assertEqual(result["peak_normal_force_n"], 2.0)

    def accumulator_step(self, current, accumulated, delta):
        return PROBE.bounded_accumulator_action(
            current,
            accumulated,
            delta,
            0.05,
            0.10,
            torch.full_like(current, -1.0),
            torch.full_like(current, 1.0),
        )

    def test_accumulator_integrates_under_static_deflection_without_windup(self):
        current = torch.tensor([0.0])
        target = torch.tensor([0.0])
        saturation_count = 0
        for _ in range(20):
            raw, bounded, saturated = self.accumulator_step(
                current, target, torch.tensor([0.02])
            )
            target = target + 0.05 * raw
            saturation_count += int(saturated.sum())
            self.assertLessEqual(float((target - current).abs().max()), 0.100001)
            torch.testing.assert_close(target, bounded)
        torch.testing.assert_close(target, torch.tensor([0.10]))
        self.assertGreater(saturation_count, 0)

    def test_accumulator_reverses_from_tracking_bound(self):
        current = torch.tensor([0.0])
        target = torch.tensor([0.10])
        for _ in range(6):
            raw, _, _ = self.accumulator_step(current, target, torch.tensor([-0.02]))
            target = target + 0.05 * raw
        self.assertLess(float(target), 0.0)
        self.assertGreaterEqual(float(target), -0.100001)

    def test_accumulator_rejects_disjoint_tracking_and_soft_intervals(self):
        with self.assertRaisesRegex(ValueError, "does not intersect"):
            self.accumulator_step(torch.tensor([2.0]), torch.tensor([1.0]), torch.tensor([0.0]))

    def test_accumulator_requested_bound_is_distinct_from_rate_limited_target(self):
        current, previous = torch.tensor([0.0]), torch.tensor([0.5])
        raw, requested, _ = self.accumulator_step(current, previous, torch.tensor([0.0]))
        torch.testing.assert_close(requested, torch.tensor([0.10]))
        torch.testing.assert_close(previous + 0.05 * raw, torch.tensor([0.45]))

    def test_accumulator_clamps_soft_limit_before_tracking_bound(self):
        current = torch.tensor([0.04])
        target = torch.tensor([0.04])
        raw, bounded, saturated = PROBE.bounded_accumulator_action(
            current,
            target,
            torch.tensor([0.04]),
            0.05,
            0.10,
            torch.tensor([-0.05]),
            torch.tensor([0.05]),
        )
        torch.testing.assert_close(bounded, torch.tensor([0.05]))
        torch.testing.assert_close(raw, torch.tensor([0.2]))
        self.assertTrue(bool(saturated[0]))

    def test_damped_step_supports_position_plus_one_leveling_row(self):
        jacobian = torch.eye(4, 5, dtype=torch.float64)
        error = torch.tensor([0.1, -0.1, 0.05, -0.02], dtype=torch.float64)
        step = PROBE.damped_position_step(
            jacobian, error, damping=0.01, gain=0.5, max_joint_step_rad=0.1
        )
        self.assertEqual(tuple(step.shape), (5,))
        self.assertLess(torch.linalg.vector_norm(error - jacobian @ step), torch.linalg.vector_norm(error))

    def test_tip_offsets_are_loaded_with_source_hash(self):
        payload = {
            "mesh_derived_tip_offsets_m": {
                "gripper": [-0.01, 0.0, -0.1],
                "jaw": [-0.01, -0.08, 0.02],
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            offsets, digest = PROBE.load_tip_offsets(path)
        self.assertEqual(offsets["jaw"], [-0.01, -0.08, 0.02])
        self.assertEqual(len(digest), 64)

    def test_tip_offsets_reject_missing_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                PROBE.load_tip_offsets(path)

    def test_table_penetration_gate_accepts_valid_no_contact_sample(self):
        self.assertTrue(
            PROBE.pair_penetration_bounded({}, ("gripper_table", "jaw_table"), 0.001, sampling_valid=True)
        )

    def test_table_penetration_gate_boundary_and_deep_overlap(self):
        pairs = ("gripper_table", "jaw_table")
        self.assertTrue(
            PROBE.pair_penetration_bounded(
                {"gripper_table": -0.001}, pairs, 0.001, sampling_valid=True
            )
        )
        self.assertFalse(
            PROBE.pair_penetration_bounded(
                {"jaw_table": -0.00101}, pairs, 0.001, sampling_valid=True
            )
        )
        self.assertFalse(PROBE.pair_penetration_bounded({}, pairs, 0.001, sampling_valid=False))

    def test_midpoint_and_leveling_row_follow_measured_tip_kinematics(self):
        fixed = torch.tensor([0.0, -0.1, 0.02])
        moving = torch.tensor([0.0, 0.1, 0.04])
        fixed_jacobian = torch.zeros((3, 5))
        moving_jacobian = torch.ones((3, 5))
        midpoint, midpoint_jacobian, height_error, level_jacobian = (
            PROBE.fingertip_midpoint_kinematics(
                fixed, moving, fixed_jacobian, moving_jacobian
            )
        )
        torch.testing.assert_close(midpoint, torch.tensor([0.0, 0.0, 0.03]))
        torch.testing.assert_close(midpoint_jacobian, torch.full((3, 5), 0.5))
        torch.testing.assert_close(height_error, torch.tensor(0.02))
        torch.testing.assert_close(level_jacobian, torch.ones(5))

    def test_contact_latched_close_switches_from_coarse_to_fine_then_holds(self):
        coarse = PROBE.contact_latched_gripper_command(
            unilateral_contact_seen=False,
            latched_target_rad=None,
            accumulated_target_rad=0.7,
            coarse_close_step_rad=0.035,
            fine_close_step_rad=0.0035,
        )
        fine = PROBE.contact_latched_gripper_command(
            unilateral_contact_seen=True,
            latched_target_rad=None,
            accumulated_target_rad=0.665,
            coarse_close_step_rad=0.035,
            fine_close_step_rad=0.0035,
        )
        hold = PROBE.contact_latched_gripper_command(
            unilateral_contact_seen=True,
            latched_target_rad=0.6615,
            accumulated_target_rad=0.6615,
            coarse_close_step_rad=0.035,
            fine_close_step_rad=0.0035,
        )
        self.assertEqual(coarse, (-0.035, None, "coarse_close"))
        self.assertEqual(fine, (-0.0035, None, "fine_close_after_unilateral_contact"))
        self.assertEqual(hold, (0.0, 0.6615, "latched_hold"))

    def test_first_bilateral_contact_latches_exact_existing_target_once(self):
        unilateral, target, event = PROBE.update_contact_latch(
            unilateral_contact_seen=True,
            latched_target_rad=None,
            fixed_force_n=0.21,
            moving_force_n=0.22,
            contact_threshold_n=0.2,
            accumulated_target_rad=0.64125,
        )
        self.assertTrue(unilateral)
        self.assertEqual(target, 0.64125)
        self.assertEqual(event, "bilateral_target_latched")
        _, retained, second_event = PROBE.update_contact_latch(
            unilateral_contact_seen=unilateral,
            latched_target_rad=target,
            fixed_force_n=2.0,
            moving_force_n=2.0,
            contact_threshold_n=0.2,
            accumulated_target_rad=0.5,
        )
        self.assertEqual(retained, 0.64125)
        self.assertIsNone(second_event)

    def test_latched_contact_requires_both_fingers_above_threshold(self):
        self.assertTrue(PROBE.latched_contact_maintained(0.6, 0.21, 0.22, 0.2))
        self.assertFalse(PROBE.latched_contact_maintained(0.6, 0.19, 0.22, 0.2))
        self.assertFalse(PROBE.latched_contact_maintained(None, 1.0, 1.0, 0.2))

    def test_fine_close_step_must_not_exceed_coarse_step(self):
        with self.assertRaisesRegex(ValueError, "no larger than coarse"):
            PROBE.contact_latched_gripper_command(
                unilateral_contact_seen=True,
                latched_target_rad=None,
                accumulated_target_rad=0.0,
                coarse_close_step_rad=0.003,
                fine_close_step_rad=0.0035,
            )


if __name__ == "__main__":
    unittest.main()

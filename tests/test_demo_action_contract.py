import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab"))
from so101_pick_rl.demo_action_contract import (
    aligned_targets, audit_targets, denormalize_targets, normalize_targets, validate_source_timing,
)


class DemoActionContractTests(unittest.TestCase):
    def test_target_is_next_command_not_next_observation(self):
        q = np.arange(18).reshape(3, 6) / 100
        target = q + 0.4
        post = np.concatenate([q[1:], q[-1:]])
        result = aligned_targets(q, target, post)
        np.testing.assert_array_equal(result, target[1:])
        self.assertFalse(np.array_equal(result, q[1:]))
        result[0, 0] = 999
        self.assertNotEqual(target[1, 0], 999)

    def test_alignment_rejects_shifted_and_nonfinite_states(self):
        q = np.arange(18).reshape(3, 6) / 100
        for post in (q, np.full_like(q, np.nan)):
            with self.assertRaises(ValueError):
                aligned_targets(q, q, post)

    def test_normalization_roundtrip_asymmetric_limits(self):
        lower = np.arange(6) - 4
        upper = lower + np.arange(6) + 1
        target = np.stack([lower, upper, (lower + upper) / 2])
        normalized = normalize_targets(target, lower, upper)
        np.testing.assert_allclose(normalized, np.stack([-np.ones(6), np.ones(6), np.zeros(6)]))
        np.testing.assert_allclose(denormalize_targets(normalized, lower, upper), target)

    def test_no_silent_clipping(self):
        target = np.zeros((3, 6))
        target[1, 2] = 1.1
        before = target.copy()
        with self.assertRaises(ValueError):
            normalize_targets(target, -np.ones(6), np.ones(6))
        report = audit_targets(target, -np.ones(6), np.ones(6))
        self.assertEqual(report["out_of_bounds_frames"], 1)
        self.assertEqual(report["discontinuous_transition_indices"], [0, 1])
        self.assertFalse(report["label_quality_pass"])
        np.testing.assert_array_equal(target, before)

    def test_quality_pass_is_not_physics_success(self):
        report = audit_targets(np.zeros((2, 6)), -np.ones(6), np.ones(6))
        self.assertTrue(report["label_quality_pass"])
        self.assertFalse(report["physics_replay_validated"])
        self.assertFalse(report["real_robot_safe"])

    def test_bad_limits_and_shapes_fail(self):
        for value in (np.zeros(6), np.zeros((2, 5)), np.full((2, 6), np.inf)):
            with self.assertRaises(ValueError):
                normalize_targets(value, -np.ones(6), np.ones(6))
        with self.assertRaises(ValueError):
            normalize_targets(np.zeros((2, 6)), np.ones(6), np.ones(6))
        with self.assertRaises(ValueError):
            denormalize_targets(np.full((2, 6), 1.01), -np.ones(6), np.ones(6))

    def test_source_clock_not_legacy_30hz(self):
        validate_source_timing({"sim_args": {"dt": 1 / 60, "decimation": 1}})
        for cfg in ({"dt": 1 / 120, "decimation": 4}, {"dt": 1 / 60, "decimation": True},
                    {"dt": float("nan"), "decimation": 1}, {}):
            with self.assertRaises(ValueError):
                validate_source_timing({"sim_args": cfg})


if __name__ == "__main__":
    unittest.main()

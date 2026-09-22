import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab"))
from so101_pick_rl.demo_reset_bootstrap import (
    HOME_OPEN_COMMAND_RAD,
    controller_action,
    normalized_home_open_action,
    reset_bootstrap_metadata,
    validate_reset_bootstrap,
)


LOWER = [-1.92, -1.75, -1.75, -1.66, -2.80, -0.18]
UPPER = [1.92, 1.75, 1.58, 1.66, 2.80, 1.75]


class RecordingPolicy:
    def __init__(self):
        self.calls = []

    def __call__(self, value):
        self.calls.append(value.clone())
        return torch.full((len(value), 6), 0.25, dtype=value.dtype, device=value.device)


class DemoResetBootstrapTests(unittest.TestCase):
    def test_first_step_skips_policy_and_later_step_calls_it(self):
        observation = torch.zeros((2, 34), dtype=torch.float32)
        mean, std = torch.zeros(34), torch.ones(34)
        policy = RecordingPolicy()
        first = controller_action(
            "home-open-one-step", 0, observation, mean, std, policy, LOWER, UPPER
        )
        self.assertEqual(policy.calls, [])
        expected = normalized_home_open_action(
            LOWER, UPPER, 2, device=observation.device, dtype=observation.dtype
        )
        torch.testing.assert_close(first, expected)
        later = controller_action(
            "home-open-one-step", 1, observation, mean, std, policy, LOWER, UPPER
        )
        self.assertEqual(len(policy.calls), 1)
        torch.testing.assert_close(later, torch.full((2, 6), 0.25))

    def test_none_delegates_step_zero_without_changing_policy_input(self):
        observation = torch.arange(68, dtype=torch.float64).reshape(2, 34)
        mean = torch.ones(34, dtype=torch.float64)
        std = torch.full((34,), 2.0, dtype=torch.float64)
        policy = RecordingPolicy()
        result = controller_action("none", 0, observation, mean, std, policy, LOWER, UPPER)
        torch.testing.assert_close(policy.calls[0], (observation - mean) / std)
        self.assertEqual(result.dtype, torch.float64)

    def test_64_batch_preserves_device_dtype_shape_and_actual_command(self):
        action = normalized_home_open_action(
            LOWER, UPPER, 64, device=torch.device("cpu"), dtype=torch.float64
        )
        self.assertEqual(action.shape, (64, 6))
        self.assertEqual(action.device, torch.device("cpu"))
        self.assertEqual(action.dtype, torch.float64)
        lower, upper = torch.tensor(LOWER), torch.tensor(UPPER)
        actual = lower + (action[0].float() + 1.0) * (upper - lower) / 2.0
        torch.testing.assert_close(actual, torch.tensor(HOME_OPEN_COMMAND_RAD), atol=1e-6, rtol=0)

    def test_validation_and_bounds_fail_closed(self):
        for mode, controller in (("bad", "bc"), ("home-open-one-step", "source"), ("none", "bad")):
            with self.subTest(mode=mode, controller=controller), self.assertRaises(ValueError):
                validate_reset_bootstrap(mode, controller)
        with self.assertRaises(ValueError):
            normalized_home_open_action(
                LOWER, [1.92, 1.75, 1.58, 1.66, 2.80, 1.0], 1,
                device=torch.device("cpu"), dtype=torch.float32,
            )
        with self.assertRaises(ValueError):
            normalized_home_open_action(
                LOWER, UPPER, 0, device=torch.device("cpu"), dtype=torch.float32
            )
        with self.assertRaises(ValueError):
            normalized_home_open_action(
                LOWER, UPPER, 1, device=torch.device("cpu"), dtype=torch.int64
            )

    def test_report_metadata_is_explicit(self):
        self.assertEqual(
            reset_bootstrap_metadata("home-open-one-step"),
            {
                "mode": "home-open-one-step",
                "command_rad": list(HOME_OPEN_COMMAND_RAD),
                "steps": 1,
                "policy_weights_unchanged": True,
                "derived_controller": True,
            },
        )
        self.assertFalse(reset_bootstrap_metadata("none")["derived_controller"])


if __name__ == "__main__":
    unittest.main()

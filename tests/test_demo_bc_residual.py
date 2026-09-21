import io
import unittest

import numpy as np

from so101_pick_rl.demo_bc_residual import build_residual_policy


class DemoBCResidualTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError as exc:
            self.skipTest(f"torch unavailable: {exc}")
        self.torch = torch
        self.torch.manual_seed(0)
        self.mean = np.linspace(-0.4, 0.6, 34, dtype=np.float32)
        self.std = np.linspace(0.25, 1.75, 34, dtype=np.float32)
        self.lower = np.array([-2.0, -1.5, -0.7, -2.2, -3.0, -0.3], dtype=np.float32)
        self.upper = np.array([1.0, 2.5, 1.8, 0.8, 2.0, 1.4], dtype=np.float32)

    def _observation_for_previous_target(self, previous):
        observation = np.zeros((len(previous), 34), dtype=np.float32)
        observation[:, 12:18] = (
            np.asarray(previous, dtype=np.float32) - self.mean[12:18]
        ) / self.std[12:18]
        return self.torch.from_numpy(observation)

    def test_initial_output_holds_clipped_previous_target(self):
        previous = np.stack(
            [
                (self.lower + self.upper) / 2,
                self.lower - 0.5,
                self.upper + 0.5,
            ]
        )
        policy = build_residual_policy(self.mean, self.std, self.lower, self.upper)
        actual = policy(self._observation_for_previous_target(previous)).detach().numpy()
        expected = np.clip(2 * (previous - self.lower) / (self.upper - self.lower) - 1, -1, 1)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-6)

    def test_output_remains_in_normalized_action_range(self):
        policy = build_residual_policy(self.mean, self.std, self.lower, self.upper, scale=0.2)
        with self.torch.no_grad():
            policy.residual[-1].weight.fill_(100.0)
            policy.residual[-1].bias.fill_(100.0)
        observations = self.torch.randn(8, 34, dtype=self.torch.float32)
        output = policy(observations)
        self.assertTrue(self.torch.all(output <= 1.0))
        self.assertTrue(self.torch.all(output >= -1.0))

    def test_correction_is_applied_before_final_clamp(self):
        policy = build_residual_policy(self.mean, self.std, self.lower, self.upper)
        with self.torch.no_grad():
            policy.residual[-1].bias.fill_(-0.25)
        previous = self.upper + 0.25 * (self.upper - self.lower)
        output = policy(self._observation_for_previous_target(previous[None, :]))
        self.torch.testing.assert_close(output, self.torch.ones_like(output))

    def test_last_layer_receives_gradient_from_initialization(self):
        policy = build_residual_policy(self.mean, self.std, self.lower, self.upper)
        previous = np.tile((self.lower + self.upper) / 2, (4, 1))
        observations = self._observation_for_previous_target(previous)
        loss = (policy(observations) - 0.25).square().mean()
        loss.backward()
        last_layer = policy.residual[-1]
        self.assertIsNotNone(last_layer.weight.grad)
        self.assertGreater(float(last_layer.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(last_layer.bias.grad.abs().sum()), 0.0)

    def test_state_dict_round_trip_preserves_float32_buffers_and_output(self):
        policy = build_residual_policy(self.mean, self.std, self.lower, self.upper, scale=0.15)
        with self.torch.no_grad():
            policy.residual[-1].weight.fill_(0.01)
            policy.residual[-1].bias.copy_(self.torch.linspace(-0.1, 0.1, 6))
        observations = self.torch.randn(3, 34, dtype=self.torch.float32)
        expected = policy(observations)
        buffer = io.BytesIO()
        self.torch.save(policy.state_dict(), buffer)
        buffer.seek(0)

        restored = build_residual_policy(self.mean, self.std, self.lower, self.upper, scale=0.15)
        restored.load_state_dict(self.torch.load(buffer, weights_only=True))
        actual = restored(observations)
        self.torch.testing.assert_close(actual, expected)
        for value in restored.state_dict().values():
            self.assertEqual(value.dtype, self.torch.float32)

    def test_rejects_invalid_configuration_and_inputs(self):
        with self.assertRaises(ValueError):
            build_residual_policy(self.mean, np.zeros(34), self.lower, self.upper)
        with self.assertRaises(ValueError):
            build_residual_policy(self.mean, self.std, self.upper, self.lower)
        with self.assertRaises(ValueError):
            build_residual_policy(self.mean, self.std, self.lower, self.upper, scale=0)

        policy = build_residual_policy(self.mean, self.std, self.lower, self.upper)
        with self.assertRaises(ValueError):
            policy(self.torch.zeros(2, 33))
        invalid = self.torch.zeros(2, 34)
        invalid[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            policy(invalid)


if __name__ == "__main__":
    unittest.main()

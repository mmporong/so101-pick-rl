import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab/scripts"))
from evaluate_demo_bc import diagnostic_flags


class DemoBCRolloutTests(unittest.TestCase):
    def state(self):
        obs = torch.zeros((2, 34))
        obs[:, 5] = .5
        obs[:, 18:21] = torch.tensor([.58, -.35, .061])
        obs[:, 31:34] = torch.tensor([.58, -.35, .0455])
        return obs

    def test_source_release_uses_cube_box_and_measured_gripper_not_target(self):
        obs = self.state()
        obs[1, 5] = 0
        obs[1, 17] = 1.5
        lift, release = diagnostic_flags(obs, torch.tensor([.01, .061]))
        self.assertEqual(release.tolist(), [True, False])
        torch.testing.assert_close(lift, torch.tensor([.051, 0]))

    def test_rejects_fast_linear_angular_outside_or_high_drop(self):
        for channel, value in ((25, .031), (28, .501), (18, .7), (20, .2)):
            obs = self.state()
            obs[1, channel] = value
            _, release = diagnostic_flags(obs, torch.zeros(2))
            self.assertEqual(release.tolist(), [True, False])

    def test_rejects_nonfinite_and_shape(self):
        obs = self.state()
        obs[0, 25] = float("nan")
        with self.assertRaises(ValueError):
            diagnostic_flags(obs, torch.zeros(2))
        with self.assertRaises(ValueError):
            diagnostic_flags(self.state()[:, :30], torch.zeros(2))


if __name__ == "__main__":
    unittest.main()

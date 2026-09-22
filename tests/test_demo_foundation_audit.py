import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab"))
from so101_pick_rl.demo_foundation_audit import audit_episode, summarize


class DemoFoundationAuditTests(unittest.TestCase):
    def fixture(self):
        q = np.zeros((4, 6))
        q[2:, -1] = .3
        pre = np.concatenate((np.zeros((1, 6)), q[:-1]))
        target = np.zeros((4, 6))
        target[0, 0] = 2
        cube = np.tile([0, 0, .06, 1, 0, 0, 0], (4, 1)).astype(float)
        cube[1:, 2] = .16
        box = np.tile([0, 0, .041, 1, 0, 0, 0], (4, 1)).astype(float)
        vel = np.zeros((4, 6))
        vel[2, 0] = .04
        return [pre, target, q, cube, box, vel, .06]

    def test_opening_is_poststate_proxy_not_release_success(self):
        result = audit_episode(*self.fixture())
        self.assertTrue(result["first_transition_rejected_by_existing_continuity_filter"])
        proxy = result["opening_proxy"]
        self.assertEqual(proxy["poststate_index"], 2)
        self.assertAlmostEqual(proxy["bottom_clearance_m"], .1)
        self.assertAlmostEqual(proxy["linear_speed_m_s"], .04)
        self.assertFalse(proxy["contact_verified_release"])
        summary = summarize({"demo_0": result})
        self.assertEqual(summary["opening_above_1cm"], 1)
        self.assertIsNone(summary["full_supported_place_success_count"])

    def test_no_crossing_and_no_lift_are_not_successes(self):
        fixture = self.fixture()
        fixture[3][:, 2] = .06
        result = audit_episode(*fixture)
        self.assertIsNone(result["opening_proxy"])
        summary = summarize({"demo": result})
        self.assertEqual(summary["episodes_with_opening_proxy"], 0)
        self.assertIsNone(summary["opening_clearance_quantiles_m"])

    def test_tilted_cube_uses_projected_half_height(self):
        fixture = self.fixture()
        fixture[3][:, 3:] = [np.cos(np.pi/8), np.sin(np.pi/8), 0, 0]
        result = audit_episode(*fixture)
        self.assertAlmostEqual(result["opening_proxy"]["bottom_clearance_m"], .16-.045-.015*np.sqrt(2))

    def test_malformed_and_misaligned_data_fail_closed(self):
        for kind in ("alignment", "nan", "quaternion", "length"):
            fixture = self.fixture()
            if kind == "alignment":
                fixture[0][1, 0] = 1
            elif kind == "nan":
                fixture[5][0, 0] = np.nan
            elif kind == "quaternion":
                fixture[3][0, 3] = 2
            else:
                fixture[1] = fixture[1][:-1]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                audit_episode(*fixture)

    def test_source_center_window_is_not_rotated_volume_containment(self):
        fixture = self.fixture()
        fixture[3][:, 0] = .044
        fixture[3][:, 3:] = [np.cos(np.pi/8), 0, 0, np.sin(np.pi/8)]
        proxy = audit_episode(*fixture)["opening_proxy"]
        self.assertTrue(proxy["center_in_source_xy_window"])
        self.assertGreater(.044 + .015*np.sqrt(2), .06)
        self.assertFalse(proxy["supported_place_success_claimed"])

    def test_nonfinite_geometry_and_rotated_box_rejected(self):
        with self.assertRaises(ValueError):
            audit_episode(*self.fixture(), edge_m=float("inf"))
        fixture = self.fixture()
        fixture[4][:, 3:] = [np.cos(np.pi/8), 0, 0, np.sin(np.pi/8)]
        with self.assertRaises(ValueError):
            audit_episode(*fixture)


if __name__ == "__main__":
    unittest.main()

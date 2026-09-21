import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab/scripts"))
from audit_demo_bc import phase_flags


class PhaseAuditTests(unittest.TestCase):
    cfg = {"pregrasp_maximum_distance_m": .06, "minimum_lift_m": .08}

    def feature(self):
        return dict(distance_m=.02, bilateral=True, opposing_sides=True, inside_pad_planes=True,
                    penetration_safe=True, approach_world_z=-1., opened=False, lift_m=.1)

    def test_lift_alone_never_counts_as_valid_grasp(self):
        for key in ("bilateral", "opposing_sides", "inside_pad_planes", "penetration_safe"):
            f = self.feature()
            f[key] = False
            flags = phase_flags(f, self.cfg)
            self.assertFalse(flags["valid_grasp"])
            self.assertFalse(flags["valid_lift"])
        for key, value in (("approach_world_z", .1), ("distance_m", .1)):
            f = self.feature()
            f[key] = value
            self.assertFalse(phase_flags(f, self.cfg)["valid_lift"])

    def test_open_approach_is_not_contact_and_respects_lift_threshold(self):
        f = self.feature()
        f.update(opened=True, bilateral=False)
        flags = phase_flags(f, self.cfg)
        self.assertTrue(flags["near_open"])
        self.assertFalse(flags["valid_grasp"])
        f = self.feature()
        f["lift_m"] = .079
        self.assertTrue(phase_flags(f, self.cfg)["valid_grasp"])
        self.assertFalse(phase_flags(f, self.cfg)["valid_lift"])


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
import json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab"))
from so101_pick_rl.demo_pose_probe import opening_schedule, contact_summary, validate_candidate_names, gravity_compensated_target
from so101_pick_rl.demo_pose_probe import separating_axis_clearance
from so101_pick_rl.demo_pose_probe import upright_cube_projected_width
from so101_pick_rl.demo_pose_probe import validate_pad_geometry


class PoseProbeTests(unittest.TestCase):
    def test_pad_asset_and_axes_are_bound_to_search(self):
        geometry = json.loads((Path(__file__).resolve().parents[1] / "configs/isaaclab/demo_box_pad_geometry.json").read_text())
        validate_pad_geometry(geometry, geometry["valid_for"])
        with self.assertRaises(ValueError):
            validate_pad_geometry(geometry, "wrong-asset")
        geometry["inward_normals"] = [[0, 1, 0], [0, -1, 0]]
        with self.assertRaises(ValueError):
            validate_pad_geometry(geometry, geometry["valid_for"])

    def test_upright_cube_aperture_depends_on_finger_orientation(self):
        self.assertAlmostEqual(upright_cube_projected_width(np.eye(3), .03), .03)
        c = np.sqrt(.5)
        rotation = np.array([[c, -c, 0], [c, c, 0], [0, 0, 1]])
        width = upright_cube_projected_width(rotation, .03)
        self.assertAlmostEqual(width, .03 * np.sqrt(2))
        self.assertLess(.040, width + .005)

    def test_projection_witness_rejects_containment_despite_clear_vertices(self):
        vertices = np.array([[x, y, z] for x in (-2, 2) for y in (-2, 2) for z in (-2, 2)])
        # All mesh vertices are outside the wall, but the hull encloses it.
        self.assertLess(separating_axis_clearance(vertices, np.eye(3), np.eye(3), np.zeros(3), [[0, 0, 0]], [[1, 1, 1]])[0], 0)
        gap = separating_axis_clearance(vertices, np.eye(3), np.eye(3), [4, 0, 0], [[0, 0, 0]], [[1, 1, 1]])
        self.assertAlmostEqual(gap[0], 1.)

    def test_gravity_offset_changes_only_arm_command_not_reference(self):
        ref = np.zeros(6)
        target = gravity_compensated_target(ref, np.ones(6), np.full(6, 20), -np.ones(6), np.ones(6))
        np.testing.assert_allclose(target, [.05]*5 + [0])
        np.testing.assert_array_equal(ref, np.zeros(6))
        for stiffness in (np.zeros(6), np.ones(6)):
            with self.assertRaises(ValueError):
                gravity_compensated_target(ref, np.ones(6), stiffness, -np.ones(6), np.ones(6))
        with self.assertRaises(ValueError):
            gravity_compensated_target(np.full(6, .99), np.ones(6), np.full(6, 20), -np.ones(6), np.ones(6))

    def test_candidate_names_cannot_escape_output_or_overwrite_traces(self):
        validate_candidate_names([{"name": "geometry_07"}])
        for candidates in ([], [{"name": "../outside"}], [{"name": "a"}, {"name": "a"}], [{"name": None}]):
            with self.assertRaises(ValueError):
                validate_candidate_names(candidates)

    def test_ramp_holds_arm_and_has_no_gripper_jump(self):
        targets, phases = opening_schedule([.1, .2, .3, .4, .5])
        self.assertEqual(targets.shape, (570, 6))
        np.testing.assert_allclose(targets[:, :5], np.tile([.1, .2, .3, .4, .5], (570, 1)))
        self.assertLessEqual(np.diff(targets[:, 5]).max(), .00300000001)
        self.assertAlmostEqual(targets[-1, -1], 1.35)
        self.assertEqual(phases.count("opening"), 450)

    def test_bad_commands_rejected(self):
        for arm in ([0]*4, [float("nan")]*5):
            with self.assertRaises(ValueError):
                opening_schedule(arm)
        with self.assertRaises(ValueError):
            opening_schedule([0]*5, hold_steps=0)

    def test_contact_data_preserved_not_smoothed(self):
        self.assertIsNone(contact_summary([])["minimum_separation_m"])
        result = contact_summary([{"normal_force_n": [.1, .2], "separation_m": [-.002, .001]}])
        self.assertAlmostEqual(result["normal_force_n"], .3)
        self.assertEqual(result["minimum_separation_m"], -.002)
        with self.assertRaises(ValueError):
            contact_summary([{"normal_force_n": [float("nan")], "separation_m": [0]}])


if __name__ == "__main__":
    unittest.main()

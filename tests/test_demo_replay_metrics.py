import copy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab"))
from so101_pick_rl.demo_replay_metrics import loaded_opposing_sides, stable_release_candidate, summarize_contact_trace


def sample(step=1):
    row = {"step": step, "joint_pos": [0.] * 6,
           "cube_pose": [0., 0., .1, 1., 0., 0., 0.], "lift_m": .05,
           "contacts": {name: [] for name in ("gripper_cube", "jaw_cube", "gripper_table", "jaw_table")}}
    for pair, side in (("gripper_cube", -1), ("jaw_cube", 1)):
        row["contacts"][pair] = [{"normal_force_n": [1.], "points_w_m": [[side * .015, 0., .1]],
                                  "normals_w": [[side, 0., 0.]], "separation_m": [-.0001]}]
    return row


class ReplayMetricsTests(unittest.TestCase):
    def test_release_requires_position_open_and_low_speed(self):
        self.assertTrue(stable_release_candidate([0, 0, .02], .3, [0, 0, 0], [0, 0, 0]))
        for position, gripper, linear, angular in [
            ([.05, 0, .02], .3, [0, 0, 0], [0, 0, 0]),
            ([0, 0, .02], .26, [0, 0, 0], [0, 0, 0]),
            ([0, 0, .02], .3, [.031, 0, 0], [0, 0, 0]),
            ([0, 0, .02], .3, [0, 0, 0], [.51, 0, 0]),
        ]:
            self.assertFalse(stable_release_candidate(position, gripper, linear, angular))

    def test_nonfinite_release_rejected(self):
        with self.assertRaises(ValueError):
            stable_release_candidate([math.nan, 0, .02], .3, [0, 0, 0], [0, 0, 0])

    def test_diagnostics_never_certify_full_grasp_or_training(self):
        result = summarize_contact_trace([sample(1), sample(2)])
        self.assertTrue(result["diagnostic_checks_pass"])
        self.assertFalse(result["normal_grasp_gate_pass"])
        self.assertFalse(result["training_ready"])
        self.assertEqual(result["opposing_contact_steps"], [1, 2])
        self.assertAlmostEqual(result["longest_opposing_contact_seconds"], 2 / 60)

    def test_penetration_and_wrist_fail_without_changing_limits(self):
        record = sample()
        record["joint_pos"][3] = math.radians(76)
        record["contacts"]["jaw_table"] = [{"separation_m": [-.0011]}]
        result = summarize_contact_trace([record])
        self.assertFalse(result["checks"]["finger_table_penetration_bounded"])
        self.assertFalse(result["checks"]["wrist_flex_bounded"])

    def test_missing_pair_or_step_rejected(self):
        for record in [sample(2), sample()]:
            if record["step"] == 1:
                del record["contacts"]["jaw_table"]
            with self.assertRaises(ValueError):
                summarize_contact_trace([record])

    def test_unloaded_and_intermittent_contacts_not_continuous_grasp(self):
        middle = sample(2)
        middle["contacts"]["jaw_cube"][0]["normal_force_n"] = [0.]
        result = summarize_contact_trace([sample(1), middle, sample(3)])
        self.assertEqual(result["opposing_contact_steps"], [1, 3])
        self.assertAlmostEqual(result["longest_opposing_contact_seconds"], 1 / 60)

    def test_input_preserved(self):
        data = [sample()]
        original = copy.deepcopy(data)
        summarize_contact_trace(data)
        self.assertEqual(data, original)

    def test_edge_top_same_side_and_inward_normals_rejected(self):
        for variant in ("edge", "top", "same_side", "inward"):
            record = sample()
            jaw = record["contacts"]["jaw_cube"][0]
            if variant == "edge":
                jaw["points_w_m"][0][1] = .014
            elif variant == "top":
                jaw["points_w_m"] = [[0, 0, .115]]
                jaw["normals_w"] = [[0, 0, 1]]
            elif variant == "same_side":
                jaw["points_w_m"] = [[-.015, 0, .1]]
                jaw["normals_w"] = [[-1, 0, 0]]
            else:
                jaw["normals_w"] = [[-1, 0, 0]]
            self.assertFalse(loaded_opposing_sides(record["contacts"], record["cube_pose"], [.03]*3, .2))

    def test_rotated_translated_cube_contacts(self):
        record = sample()
        record["cube_pose"] = [1, 2, 3, math.sqrt(.5), 0, 0, math.sqrt(.5)]
        for name, side in (("gripper_cube", -1), ("jaw_cube", 1)):
            row = record["contacts"][name][0]
            row["points_w_m"] = [[1, 2 + side * .015, 3]]
            row["normals_w"] = [[0, side, 0]]
        self.assertTrue(loaded_opposing_sides(record["contacts"], record["cube_pose"], [.03]*3, .2))

    def test_bad_pose_and_malformed_unloaded_contact_rejected(self):
        record = sample()
        record["contacts"]["jaw_cube"][0]["normal_force_n"] = [0., 0.]
        with self.assertRaises(ValueError):
            loaded_opposing_sides(record["contacts"], record["cube_pose"], [.03]*3, .2)
        record = sample()
        record["cube_pose"][3] = 2.
        with self.assertRaises(ValueError):
            loaded_opposing_sides(record["contacts"], record["cube_pose"], [.03]*3, .2)


if __name__ == "__main__":
    unittest.main()

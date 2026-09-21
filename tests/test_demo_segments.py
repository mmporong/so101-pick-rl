import sys
import unittest
import json
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab"))
from so101_pick_rl.demo_segments import carry_ranges, state_action_pair
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab/scripts"))
from audit_demo_segments import audit, sha256, verify_trace_digest
from so101_pick_rl.demo_action_contract import validate_source_timing


class DemoSegmentTests(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(physics_dt_s=1/60, minimum_lift_hold_s=.2, minimum_lift_m=.08)
        self.good = dict(bilateral=True, opposing_sides=True, inside_pad_planes=True,
                         penetration_safe=True, hand_box_clear=True, approach_world_z=-.7, lift_m=.09)

    def test_duration_counts_transitions_not_states(self):
        self.assertEqual(carry_ranges([self.good]*12, self.cfg), [])
        self.assertEqual(carry_ranges([self.good]*13, self.cfg), [(0,12)])

    def test_gaps_never_join_and_release_is_excluded(self):
        for key in ("bilateral", "opposing_sides", "inside_pad_planes", "penetration_safe", "hand_box_clear"):
            bad = dict(self.good, **{key: False})
            self.assertEqual(carry_ranges([self.good]*12+[bad]+[self.good]*12, self.cfg), [])
            self.assertEqual(carry_ranges([self.good]*13+[bad]+[self.good]*13, self.cfg), [(0,12),(14,26)])

    def test_upward_and_low_lift_are_excluded(self):
        for changes in (dict(approach_world_z=.1), dict(lift_m=.01)):
            self.assertEqual(carry_ranges([dict(self.good, **changes)]*15, self.cfg), [])

    def test_malformed_timing_and_pose_fail_closed(self):
        with self.assertRaises(ValueError):
            carry_ranges([self.good], dict(self.cfg, physics_dt_s=0))
        with self.assertRaises(ValueError):
            carry_ranges([dict(self.good, approach_world_z=float("nan"))], self.cfg)

    def pair(self):
        previous = dict(step=1, phase="recorded_transition", control_boundary=True,
                        joint_pos=[0.]*6, executed_target=[.1]*6, cube_pose=[0,0,.1,1,0,0,0])
        pre = dict(joint_position_rad=previous["joint_pos"], joint_velocity_rad_s=[0.]*6,
                   previous_target_rad=previous["executed_target"], cube_pose_w=previous["cube_pose"],
                   cube_velocity_w=[0.]*6, box_position_w_m=[0.,0.,0.])
        return previous, dict(step=2, phase="recorded_transition", control_boundary=True,
                              pre_step_state=pre, executed_target=[.2]*6)

    def test_hold_controller_and_nonboundary_are_excluded(self):
        for changes in (dict(phase="additional_terminal_hold"), dict(phase="lower_closed"),
                        dict(control_boundary=False)):
            previous, current = self.pair()
            current.update(changes)
            with self.assertRaises(ValueError):
                state_action_pair(previous, current)

    def test_actual_pre_state_and_action_have_fixed_order(self):
        previous, current = self.pair()
        observation, action = state_action_pair(previous, current)
        self.assertEqual(observation.shape, (34,))
        np.testing.assert_allclose(observation[12:18], .1)
        np.testing.assert_allclose(action, .2)

    def test_modified_action_cannot_use_unrelated_old_observation(self):
        previous, current = self.pair()
        current["pre_step_state"]["joint_position_rad"] = [.5]*6
        with self.assertRaises(ValueError):
            state_action_pair(previous, current)

    def test_nonconsecutive_and_nonfinite_rejected(self):
        previous, current = self.pair()
        current["step"] = 4
        with self.assertRaises(ValueError):
            state_action_pair(previous, current)
        previous, current = self.pair()
        current["executed_target"][0] = float("nan")
        with self.assertRaises(ValueError):
            state_action_pair(previous, current)

    def test_incomplete_reports_cannot_approve_candidates(self):
        examples = [dict(status="running"),
                    dict(status="replay_complete_sequence_audited", physics_substeps=4, control_dt_s=1/60),
                    dict(status="replay_complete_sequence_audited", physics_substeps=1,
                         control_dt_s=1/60, code_sha256={})]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            for report in examples:
                path.write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaises(ValueError):
                    audit(path)

    def test_missing_source_timing_is_not_assumed_from_mimic(self):
        with self.assertRaisesRegex(ValueError, "missing source timing metadata"):
            validate_source_timing({"env_name": "", "type": 2})

    def test_trace_digest_binding_rejects_missing_and_modified_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text("original", encoding="utf-8")
            digest = sha256(path)
            self.assertEqual(verify_trace_digest(path, digest), digest)
            with self.assertRaisesRegex(ValueError, "missing"):
                verify_trace_digest(path, None)
            self.assertEqual(verify_trace_digest(path, None, allow_unbound=True), digest)
            path.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                verify_trace_digest(path, digest, allow_unbound=True)


if __name__ == "__main__":
    unittest.main()

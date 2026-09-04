from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_PROJECT_ROOT = REPOSITORY_ROOT / "isaaclab"
sys.path.insert(0, str(ISAACLAB_PROJECT_ROOT))

from so101_pick_rl import task_contract
from so101_pick_rl.validation import (
    count_reset_events,
    finalize_smoke_checks,
    summarize_reset_failures,
)

from so101_pick_rl.kit_log import bind_kit_log, summarize_kit_log


class IsaacLabContractAlignmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = json.loads((REPOSITORY_ROOT / "common" / "task_spec.json").read_text(encoding="utf-8"))
        self.runtime = json.loads(
            (REPOSITORY_ROOT / "configs" / "isaaclab" / "windows_runtime.json").read_text(encoding="utf-8")
        )

    def test_python_constants_are_loaded_from_common_contract(self) -> None:
        self.assertEqual(self.spec["task"]["id"], task_contract.TASK_ID)
        self.assertEqual(tuple(self.spec["robot"]["joint_order"]), task_contract.JOINT_ORDER)
        self.assertEqual(self.spec["control"]["physics_hz"], task_contract.PHYSICS_HZ)
        self.assertEqual(self.spec["control"]["policy_hz"], task_contract.POLICY_HZ)
        self.assertEqual(self.spec["control"]["action"]["clip_abs_rad"], task_contract.ACTION_SCALE_RAD)

    def test_runtime_profile_matches_common_contract(self) -> None:
        self.assertEqual(self.spec["task"]["id"], self.runtime["task"])
        self.assertEqual(self.spec["task"]["episode_seconds"], self.runtime["episode_seconds"])
        self.assertEqual(
            self.spec["control"],
            {
                "physics_hz": self.runtime["physics_hz"],
                "policy_hz": self.runtime["policy_hz"],
                "action": self.runtime["action"],
            },
        )
        self.assertEqual(
            self.spec["task"]["success"],
            {
                "reference": "cube_initial_z",
                **self.runtime["success"],
            },
        )

    def test_gate_profile_keeps_required_smoke_sizes(self) -> None:
        self.assertEqual(1000, self.runtime["gates"]["g2"]["physics_steps"])
        self.assertEqual(64, self.runtime["gates"]["g3"]["num_envs"])
        self.assertEqual(10000, self.runtime["gates"]["g3"]["physics_steps"])
        self.assertEqual(64, self.runtime["gates"]["g4"]["num_envs"])
        self.assertEqual(10, self.runtime["gates"]["g4"]["max_iterations"])

    def test_action_contract_is_enforced_in_environment_configuration(self) -> None:
        env_source = (
            ISAACLAB_PROJECT_ROOT / "so101_pick_rl" / "tasks" / "lift_cube" / "env_cfg.py"
        ).read_text(encoding="utf-8")
        observation_source = (
            ISAACLAB_PROJECT_ROOT / "so101_pick_rl" / "tasks" / "lift_cube" / "mdp" / "observations.py"
        ).read_text(encoding="utf-8")
        self.assertIn("clip={joint_name: (-ACTION_SCALE_RAD, ACTION_SCALE_RAD)", env_source)
        self.assertIn("get_term(action_name).processed_actions", observation_source)

    def test_g3_rejects_unsafe_termination_and_too_few_resets(self) -> None:
        checks = finalize_smoke_checks(
            {"finite_observations": True},
            {"non_finite": 1, "workspace_exit": 0},
            is_so101_task=True,
            is_g3_run=True,
            automatic_reset_count=99,
            reset_failure_count=0,
            resource_sample_count=1,
            simulator_log_path="kit.log",
            simulator_error_count=0,
        )
        self.assertFalse(checks["no_non_finite_termination"])
        self.assertFalse(checks["at_least_100_automatic_resets"])
        self.assertFalse(all(checks.values()))
        reset_failure_checks = finalize_smoke_checks(
            {"finite_observations": True},
            {"non_finite": 0, "workspace_exit": 0},
            is_so101_task=True,
            is_g3_run=True,
            automatic_reset_count=100,
            reset_failure_count=1,
            resource_sample_count=1,
            simulator_log_path="kit.log",
            simulator_error_count=0,
        )
        self.assertFalse(reset_failure_checks["reset_failure_count_zero"])

    def test_reset_events_are_counted_once_when_done_flags_overlap(self) -> None:
        self.assertEqual(
            3,
            count_reset_events(
                [True, False, True, False],
                [True, True, False, False],
            ),
        )
        with self.assertRaises(ValueError):
            count_reset_events([True], [False, True])

    def test_reset_failure_summary_counts_unique_failed_resets_and_reasons(self) -> None:
        summary = summarize_reset_failures(
            {
                "returned_observation_non_finite": [True, False, False],
                "episode_length_not_zero": [True, False, True],
                "robot_state_non_finite": [False, False, False],
            }
        )
        self.assertEqual(2, summary["failure_count"])
        self.assertEqual(
            {
                "returned_observation_non_finite": 1,
                "episode_length_not_zero": 2,
                "robot_state_non_finite": 0,
            },
            summary["reason_counts"],
        )

    def test_final_evidence_rejects_missing_resources_or_kit_log(self) -> None:
        checks = finalize_smoke_checks(
            {"finite_observations": True},
            {"non_finite": 0, "workspace_exit": 0},
            is_so101_task=True,
            is_g3_run=False,
            automatic_reset_count=0,
            reset_failure_count=0,
            resource_sample_count=0,
            simulator_log_path=None,
            simulator_error_count=None,
        )
        self.assertFalse(checks["resource_samples_present"])
        self.assertFalse(checks["simulator_log_captured"])
        self.assertFalse(checks["simulator_error_free"])

    def test_kit_log_is_bound_by_unique_command_token(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log_dir = Path(temporary)
            own_log = log_dir / "kit_own.log"
            other_log = log_dir / "kit_other.log"
            own_log.write_text("Cmd: kit.exe --output unique-report.json\n[Warning] sample\n", encoding="utf-8")
            other_log.write_text("Cmd: kit.exe --output other-report.json\n[Error] unrelated\n", encoding="utf-8")
            started_at = time.time() - 1.0
            binding = bind_kit_log(started_at, "unique-report.json", log_dir=log_dir)
            summary = summarize_kit_log(binding)
        self.assertEqual(str(own_log), binding["path"])
        self.assertEqual("command_line_token", binding["binding"])
        self.assertEqual(1, summary["warning_count"])
        self.assertEqual(0, summary["error_count"])


if __name__ == "__main__":
    unittest.main()

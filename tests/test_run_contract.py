from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_PROJECT_ROOT = REPOSITORY_ROOT / "isaaclab"
sys.path.insert(0, str(ISAACLAB_PROJECT_ROOT))

from so101_pick_rl import run_contract


class RunContractTest(unittest.TestCase):
    def test_pick_place_training_requires_normal_grasp_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "grasp_feasibility_report"):
            run_contract.load_grasp_training_gate(None, run_contract.PICK_PLACE_TASK_ID)
        self.assertIsNone(run_contract.load_grasp_training_gate(None, run_contract.LIFT_TASK_ID))

    def test_grasp_training_gate_rejects_failed_dirty_stale_or_overridden_evidence(self) -> None:
        names = ("opened_near_cube_before_grasp", "cube_not_pushed_before_close", "bilateral_contact",
                 "finger_cube_penetration_bounded", "finger_table_penetration_bounded", "cube_lifted",
                 "cube_reached_target_xy", "wrist_flex_bounded", "controlled_release_observed",
                 "full_task_success", "simulator_error_free")
        valid = {"schema": "so101_pick_rl.grasp_feasibility_probe.v1", "status": "feasible",
                 "contract_sha256": run_contract.contract_sha256(run_contract.PICK_PLACE_TASK_ID),
                 "git_dirty": False, "git_commit": "test-source", "classification": "scripted_diagnostic",
                 "gate_limits": dict(run_contract.GRASP_GATE_LIMITS),
                 "gates": dict.fromkeys(names, True), "collision_probe": {"source_override": False}}
        with tempfile.TemporaryDirectory() as temporary, patch.object(
                run_contract.subprocess, "check_output",
                side_effect=lambda command, **kwargs: "test-source\n" if "rev-parse" in command else ""):
            report = Path(temporary) / "probe.json"
            report.write_text(json.dumps(valid))
            gate = run_contract.load_grasp_training_gate(report, run_contract.PICK_PLACE_TASK_ID)
            assert gate is not None
            self.assertEqual("test-source", gate["source_commit"])
            for change in ({"status": "not_feasible_or_incomplete"}, {"git_dirty": True}, {"git_commit": "old"},
                           {"contract_sha256": "old"}, {"gates": {}}, {"collision_probe": {"source_override": True}}):
                report.write_text(json.dumps({**valid, **change}))
                with self.assertRaisesRegex(ValueError, "feasibility report"):
                    run_contract.load_grasp_training_gate(report, run_contract.PICK_PLACE_TASK_ID)
            for name, maximum in run_contract.GRASP_GATE_LIMITS.items():
                report.write_text(json.dumps({**valid, "gate_limits": {**valid["gate_limits"], name: maximum * 2}}))
                with self.assertRaisesRegex(ValueError, "relaxed"):
                    run_contract.load_grasp_training_gate(report, run_contract.PICK_PLACE_TASK_ID)
            for malformed in ([], {**valid, "gates": None}, {**valid, "gate_limits": None}):
                report.write_text(json.dumps(malformed))
                with self.assertRaises(ValueError):
                    run_contract.load_grasp_training_gate(report, run_contract.PICK_PLACE_TASK_ID)
            report.write_text(json.dumps(valid))
            with patch.object(run_contract.subprocess, "check_output", return_value="test-source\n"):
                with self.assertRaisesRegex(ValueError, "feasibility report"):
                    run_contract.load_grasp_training_gate(report, run_contract.PICK_PLACE_TASK_ID)

    def test_supported_tasks_resolve_to_distinct_contracts(self) -> None:
        self.assertEqual(
            REPOSITORY_ROOT / "common" / "task_spec.json",
            run_contract.spec_path(run_contract.LIFT_TASK_ID),
        )
        self.assertEqual(
            REPOSITORY_ROOT / "common" / "pick_place_spec.json",
            run_contract.spec_path(run_contract.PICK_PLACE_TASK_ID),
        )

    def test_unknown_task_is_rejected_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported task ID"):
            run_contract.spec_path("Cartpole-v1")

    def test_load_rejects_contract_declaring_another_task(self) -> None:
        with patch.object(run_contract, "spec_path", return_value=run_contract.spec_path(run_contract.LIFT_TASK_ID)):
            with self.assertRaisesRegex(ValueError, "task contract ID mismatch"):
                run_contract.load_spec(run_contract.PICK_PLACE_TASK_ID)

    def test_hash_uses_sorted_compact_unicode_json(self) -> None:
        task = run_contract.LIFT_TASK_ID
        spec = json.loads(run_contract.spec_path(task).read_text(encoding="utf-8"))
        canonical = json.dumps(
            spec,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), run_contract.contract_sha256(task))

    def test_run_binding_checks_dimensions(self) -> None:
        binding = run_contract.run_binding(run_contract.LIFT_TASK_ID, 22, 6)
        self.assertEqual(run_contract.LIFT_TASK_ID, binding["task"])
        self.assertEqual(22, binding["observation_dimension"])
        with self.assertRaisesRegex(ValueError, "observation dimension"):
            run_contract.run_binding(run_contract.LIFT_TASK_ID, 39, 6)
        with self.assertRaisesRegex(ValueError, "action dimension"):
            run_contract.run_binding(run_contract.LIFT_TASK_ID, 22, 5)

    def test_resume_binding_requires_exact_contract_identity(self) -> None:
        expected = run_contract.run_binding(run_contract.LIFT_TASK_ID, 22, 6)
        run_contract.validate_resume_binding(dict(expected), expected)
        incompatible = dict(expected, contract_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            run_contract.validate_resume_binding(incompatible, expected)

    def test_pick_place_resume_requires_sidecar_but_legacy_lift_can_resume_without_one(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "model_9.pt"
            lift_binding = run_contract.run_binding(run_contract.LIFT_TASK_ID, 22, 6)
            self.assertIsNone(run_contract.load_resume_binding(checkpoint, lift_binding))

            pick_place_binding = run_contract.run_binding(run_contract.PICK_PLACE_TASK_ID, 41, 6)
            with self.assertRaisesRegex(FileNotFoundError, "missing run contract sidecar"):
                run_contract.load_resume_binding(checkpoint, pick_place_binding)

            sidecar = checkpoint.parent / "run_contract.json"
            sidecar.write_text(json.dumps(pick_place_binding), encoding="utf-8")
            self.assertEqual(
                pick_place_binding,
                run_contract.load_resume_binding(checkpoint, pick_place_binding),
            )


if __name__ == "__main__":
    unittest.main()

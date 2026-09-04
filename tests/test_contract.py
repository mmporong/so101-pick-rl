from __future__ import annotations

import importlib.util
import json
import unittest
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "validate_contract.py"
SPEC = importlib.util.spec_from_file_location("validate_contract", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


class TaskContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = json.loads((ROOT / "common" / "task_spec.json").read_text(encoding="utf-8"))

    def test_repository_contract_is_valid(self) -> None:
        self.assertEqual([], VALIDATOR.validate_task_spec(self.contract))

    def test_action_dimension_must_match_joint_order(self) -> None:
        broken = deepcopy(self.contract)
        broken["control"]["action"]["dimension"] = 5
        errors = VALIDATOR.validate_task_spec(broken)
        self.assertIn("action.dimension must match joint_order length", errors)

    def test_policy_rate_must_divide_physics_rate(self) -> None:
        broken = deepcopy(self.contract)
        broken["control"]["policy_hz"] = 50
        errors = VALIDATOR.validate_task_spec(broken)
        self.assertIn("physics_hz must be a positive multiple of policy_hz", errors)

    def test_invalid_seed_type_is_reported_without_crashing(self) -> None:
        broken = deepcopy(self.contract)
        broken["evaluation"]["seeds"] = [0, []]
        errors = VALIDATOR.validate_task_spec(broken)
        self.assertIn("evaluation.seeds must contain non-negative integers", errors)

    def test_runtime_validated_status_requires_measurement_evidence(self) -> None:
        broken = deepcopy(self.contract)
        broken["status"] = "runtime_validated"
        errors = VALIDATOR.validate_task_spec(broken)
        self.assertIn("runtime_validated status requires runtime measurements", errors)

    def test_contract_digest_is_stable_for_key_order(self) -> None:
        reversed_contract = dict(reversed(list(self.contract.items())))
        self.assertEqual(
            VALIDATOR.canonical_sha256(self.contract),
            VALIDATOR.canonical_sha256(reversed_contract),
        )


if __name__ == "__main__":
    unittest.main()

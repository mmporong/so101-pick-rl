from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_PROJECT_ROOT = REPOSITORY_ROOT / "isaaclab"
sys.path.insert(0, str(ISAACLAB_PROJECT_ROOT))

from so101_pick_rl import task_contract


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


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab/scripts"))
from audit_parallel_demo_bc import fixed_episode_ids, selected_environment_indices


class ParallelAuditSelectionTests(unittest.TestCase):
    def test_fixed_population_and_selected_mapping(self):
        episodes = fixed_episode_ids()
        self.assertEqual(len(episodes), 64)
        self.assertEqual(len(set(episodes)), 64)
        self.assertEqual(selected_environment_indices([30, 412]), {30: 30, 412: 44})

    def test_selection_rejects_duplicates_empty_and_outside_population(self):
        for selected in ([], [30, 30], [32], [399], [432]):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                selected_environment_indices(selected)


if __name__ == "__main__":
    unittest.main()

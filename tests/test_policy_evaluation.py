from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "isaaclab"))

from so101_pick_rl.policy_evaluation import _phase_failure_masks, evaluate_policy


def phase_metrics(category: str) -> dict[str, bool]:
    order = ("reached", "contacted", "picked", "at_target", "released")
    achieved = {
        "not_reached": 0,
        "reached_not_grasped": 1,
        "grasped_not_lifted": 2,
        "lifted_not_transferred": 3,
        "transferred_not_released": 4,
        "released_unstable": 5,
    }.get(category, 0)
    return {name: index < achieved for index, name in enumerate(order)}


class FakeTerminationManager:
    active_terms = ("pick_place_success", "non_finite", "workspace_exit", "time_out")

    def __init__(self, num_envs: int):
        self.terminated = torch.zeros(num_envs, dtype=torch.bool)
        self.terms = {name: torch.zeros(num_envs, dtype=torch.bool) for name in self.active_terms}

    def get_term(self, name: str) -> torch.Tensor:
        return self.terms[name]


class FakeEnvironment:
    def __init__(self, scenarios):
        self.device = torch.device("cpu")
        self.num_envs = len(scenarios[0])
        self.max_episode_length = 4
        self.episode_length_buf = torch.ones(self.num_envs, dtype=torch.long)
        self.termination_manager = FakeTerminationManager(self.num_envs)
        self.last_episode_metrics = {
            name: torch.zeros(self.num_envs, dtype=torch.bool)
            for name in ("reached", "contacted", "picked", "at_target", "released")
        }
        self.scenarios = scenarios
        self.step_index = 0
        self.reset_seed = None

    def reset(self, seed: int):
        self.reset_seed = seed
        self.episode_length_buf.zero_()


class FakeWrappedEnvironment:
    def __init__(self, scenarios):
        self.unwrapped = FakeEnvironment(scenarios)

    def get_observations(self):
        return torch.zeros(self.unwrapped.num_envs, 3), {}

    def step(self, actions):
        env = self.unwrapped
        categories = env.scenarios[env.step_index]
        env.step_index += 1
        manager = env.termination_manager
        for value in manager.terms.values():
            value.zero_()
        manager.terminated.zero_()
        for env_id, category in enumerate(categories):
            for name, achieved in phase_metrics(category).items():
                env.last_episode_metrics[name][env_id] = achieved
            if category in manager.terms:
                manager.terms[category][env_id] = True
                manager.terminated[env_id] = True
            elif category == "success":
                manager.terms["pick_place_success"][env_id] = True
                manager.terminated[env_id] = True
            else:
                manager.terms["time_out"][env_id] = True
        dones = torch.ones(env.num_envs, dtype=torch.long)
        return torch.zeros_like(actions), torch.zeros(env.num_envs), dones, {}


class FakeRunner:
    def get_inference_policy(self, device):
        return lambda observations: torch.zeros(observations.shape[0], 3, device=device)


class PolicyEvaluationTest(unittest.TestCase):
    def test_highest_phase_wins_when_metrics_are_not_monotonic(self) -> None:
        metrics = {
            "reached": torch.tensor([False, False, True, True, False]),
            "contacted": torch.tensor([True, False, True, False, False]),
            "picked": torch.tensor([False, True, False, True, False]),
            "at_target": torch.tensor([False, False, True, False, False]),
            "released": torch.tensor([False, False, False, True, True]),
        }
        masks = _phase_failure_masks(metrics)
        assigned = torch.stack(list(masks.values())).sum(dim=0)
        self.assertTrue(torch.equal(torch.ones_like(assigned), assigned))
        self.assertTrue(masks["grasped_not_lifted"][0])
        self.assertTrue(masks["lifted_not_transferred"][1])
        self.assertTrue(masks["transferred_not_released"][2])
        self.assertTrue(masks["released_unstable"][3])
        self.assertTrue(masks["released_unstable"][4])

    def test_fixed_quotas_and_failure_classification_include_invalid_episodes(self) -> None:
        scenarios = [
            ("success", "not_reached", "reached_not_grasped"),
            ("grasped_not_lifted", "lifted_not_transferred", "transferred_not_released"),
            ("released_unstable", "workspace_exit", "non_finite"),
        ]
        wrapped = FakeWrappedEnvironment(scenarios)
        result = evaluate_policy(FakeRunner(), wrapped, episodes=9, seed=17)

        self.assertEqual(17, wrapped.unwrapped.reset_seed)
        self.assertEqual([3, 3, 3], result["per_environment_quota"])
        self.assertEqual([3, 3, 3], result["per_environment_completed"])
        self.assertEqual(9, result["completed_episodes"])
        self.assertEqual(1, result["successes"])
        self.assertEqual(8, result["failures"])
        self.assertEqual(
            {
                "not_reached": 1,
                "reached_not_grasped": 1,
                "grasped_not_lifted": 1,
                "lifted_not_transferred": 1,
                "transferred_not_released": 1,
                "released_unstable": 1,
                "non_finite": 1,
                "workspace_exit": 1,
            },
            result["failure_counts"],
        )
        self.assertAlmostEqual(1 / 9, result["success_rate"])
        self.assertFalse(result["policy_success_claimed"])

    def test_zero_episode_diagnostic_does_not_step_environment(self) -> None:
        wrapped = FakeWrappedEnvironment([("success",)])
        result = evaluate_policy(FakeRunner(), wrapped, episodes=0, seed=3)
        self.assertEqual(0, result["completed_episodes"])
        self.assertEqual("diagnostic", result["evaluation_scope"])
        self.assertEqual(0, wrapped.unwrapped.step_index)


if __name__ == "__main__":
    unittest.main()

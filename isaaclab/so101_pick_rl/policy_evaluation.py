"""Torch-only deterministic policy evaluation and Pick & Place failure accounting."""

from __future__ import annotations

from typing import Any

import torch


PHASE_FAILURES = (
    "not_reached",
    "reached_not_grasped",
    "grasped_not_lifted",
    "lifted_not_transferred",
    "transferred_not_released",
    "released_unstable",
)
INVALID_FAILURES = ("non_finite", "workspace_exit")
FAILURE_CLASSES = PHASE_FAILURES + INVALID_FAILURES


def _phase_failure_masks(metrics: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    reached = metrics["reached"].bool()
    contacted = metrics["contacted"].bool()
    picked = metrics["picked"].bool()
    at_target = metrics["at_target"].bool()
    released = metrics["released"].bool()
    released_unstable = released
    transferred_not_released = at_target & ~released_unstable
    lifted_not_transferred = picked & ~at_target & ~released_unstable
    grasped_not_lifted = contacted & ~picked & ~at_target & ~released_unstable
    reached_not_grasped = reached & ~contacted & ~picked & ~at_target & ~released_unstable
    any_progress = reached | contacted | picked | at_target | released
    return {
        "not_reached": ~any_progress,
        "reached_not_grasped": reached_not_grasped,
        "grasped_not_lifted": grasped_not_lifted,
        "lifted_not_transferred": lifted_not_transferred,
        "transferred_not_released": transferred_not_released,
        "released_unstable": released_unstable,
    }


def evaluate_policy(runner, wrapped_env, episodes: int, seed: int) -> dict[str, Any]:
    """Evaluate fixed per-environment episode quotas without randomized episode lengths."""
    if episodes < 0:
        raise ValueError("episodes must be non-negative")
    if episodes == 0:
        return {
            "evaluation_scope": "diagnostic",
            "seed": seed,
            "requested_episodes": 0,
            "completed_episodes": 0,
            "successes": 0,
            "failures": 0,
            "failure_counts": {name: 0 for name in FAILURE_CLASSES},
            "success_rate": None,
            "policy_success_claimed": False,
        }

    unwrapped = wrapped_env.unwrapped
    unwrapped.reset(seed=seed)
    if bool(torch.count_nonzero(unwrapped.episode_length_buf).item()):
        raise RuntimeError("evaluation reset did not clear randomized episode lengths")
    observations, _ = wrapped_env.get_observations()
    policy = runner.get_inference_policy(device=unwrapped.device)

    base_quota, remainder = divmod(episodes, unwrapped.num_envs)
    quotas = torch.full((unwrapped.num_envs,), base_quota, dtype=torch.long, device=unwrapped.device)
    quotas[:remainder] += 1
    completed = torch.zeros_like(quotas)
    successes = torch.zeros_like(quotas)
    failure_counts = {
        name: torch.zeros_like(quotas) for name in FAILURE_CLASSES
    }
    max_policy_steps = int(unwrapped.max_episode_length) * (int(quotas.max().item()) + 1)

    with torch.inference_mode():
        for policy_step in range(max_policy_steps):
            actions = policy(observations)
            observations, _, dones, _ = wrapped_env.step(actions)
            active = completed < quotas
            done = dones.bool() & active
            if not bool(done.any().item()):
                continue

            termination_manager = unwrapped.termination_manager
            non_finite = termination_manager.get_term("non_finite").bool() & done
            workspace_exit = termination_manager.get_term("workspace_exit").bool() & done & ~non_finite
            invalid = non_finite | workspace_exit
            success = (
                termination_manager.get_term("pick_place_success").bool()
                & termination_manager.terminated.bool()
                & done
                & ~invalid
            )
            ordinary_failure = done & ~success & ~invalid
            phase_masks = _phase_failure_masks(unwrapped.last_episode_metrics)

            successes += success.long()
            failure_counts["non_finite"] += non_finite.long()
            failure_counts["workspace_exit"] += workspace_exit.long()
            for name in PHASE_FAILURES:
                failure_counts[name] += (ordinary_failure & phase_masks[name]).long()
            completed += done.long()
            if bool((completed >= quotas).all().item()):
                break
        else:
            raise RuntimeError(
                "evaluation could not collect its fixed episode quota; "
                f"completed={int(completed.sum().item())}/{episodes}"
            )

    completed_count = int(completed.sum().item())
    success_count = int(successes.sum().item())
    flat_failures = {
        name: int(count.sum().item()) for name, count in failure_counts.items()
    }
    classified_failures = sum(flat_failures.values())
    if classified_failures != completed_count - success_count:
        raise RuntimeError(
            "episode failure classification is incomplete or overlapping: "
            f"classified={classified_failures}, expected={completed_count - success_count}"
        )
    return {
        "evaluation_scope": "diagnostic",
        "seed": seed,
        "requested_episodes": episodes,
        "completed_episodes": completed_count,
        "successes": success_count,
        "failures": classified_failures,
        "failure_counts": flat_failures,
        "attempted_episodes": completed_count,
        "policy_steps": policy_step + 1,
        "per_environment_quota": quotas.cpu().tolist(),
        "per_environment_completed": completed.cpu().tolist(),
        "per_environment_successes": successes.cpu().tolist(),
        "per_environment_failure_counts": {
            name: count.cpu().tolist() for name, count in failure_counts.items()
        },
        "success_rate": success_count / completed_count,
        "policy_success_claimed": False,
    }

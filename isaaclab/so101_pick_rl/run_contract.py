"""Select and hash the immutable task contract bound to an Isaac Lab run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LIFT_TASK_ID = "SO101-LiftCube-v0"
PICK_PLACE_TASK_ID = "SO101-PickPlace-v0"
_SPEC_FILENAMES = {
    LIFT_TASK_ID: "task_spec.json",
    PICK_PLACE_TASK_ID: "pick_place_spec.json",
}


def spec_path(task: str) -> Path:
    """Return the contract path for a supported task ID."""
    try:
        filename = _SPEC_FILENAMES[task]
    except KeyError as exc:
        supported = ", ".join(sorted(_SPEC_FILENAMES))
        raise ValueError(f"unsupported task ID {task!r}; expected one of: {supported}") from exc
    return REPOSITORY_ROOT / "common" / filename


def load_spec(task: str) -> dict[str, Any]:
    """Load a supported task contract and verify that it declares the selected ID."""
    path = spec_path(task)
    if not path.is_file():
        raise FileNotFoundError(f"task contract for {task!r} is missing: {path}")
    spec = json.loads(path.read_text(encoding="utf-8"))
    declared_task = spec.get("task", {}).get("id")
    if declared_task != task:
        raise ValueError(
            f"task contract ID mismatch for {path}: expected {task!r}, found {declared_task!r}"
        )
    return spec


def canonical_spec_bytes(task: str) -> bytes:
    """Serialize a task contract as sorted compact Unicode JSON encoded as UTF-8."""
    return json.dumps(
        load_spec(task),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def contract_sha256(task: str) -> str:
    """Return the SHA-256 of the task's canonical JSON representation."""
    return hashlib.sha256(canonical_spec_bytes(task)).hexdigest()


def run_binding(task: str, observation_dimension: int, action_dimension: int) -> dict[str, Any]:
    """Build and validate the contract identity stored beside a training run."""
    spec = load_spec(task)
    expected_observation_dimension = int(spec.get("observation", {}).get("dimension", 22))
    expected_action_dimension = int(spec["control"]["action"]["dimension"])
    if observation_dimension != expected_observation_dimension:
        raise ValueError(
            "runtime observation dimension does not match task contract: "
            f"expected {expected_observation_dimension}, found {observation_dimension}"
        )
    if action_dimension != expected_action_dimension:
        raise ValueError(
            "runtime action dimension does not match task contract: "
            f"expected {expected_action_dimension}, found {action_dimension}"
        )
    return {
        "schema": "so101_pick_rl.run_contract.v1",
        "task": task,
        "contract_sha256": contract_sha256(task),
        "observation_dimension": observation_dimension,
        "action_dimension": action_dimension,
    }


def validate_resume_binding(binding: dict[str, Any], expected: dict[str, Any]) -> None:
    """Reject a checkpoint sidecar that cannot belong to the current run contract."""
    fields = ("schema", "task", "contract_sha256", "observation_dimension", "action_dimension")
    mismatches = {
        field: {"expected": expected.get(field), "found": binding.get(field)}
        for field in fields
        if binding.get(field) != expected.get(field)
    }
    if mismatches:
        raise ValueError(f"resume run contract is incompatible: {mismatches}")


def load_resume_binding(checkpoint: Path, expected: dict[str, Any]) -> dict[str, Any] | None:
    """Load a checkpoint's sibling sidecar, requiring it for Pick & Place runs."""
    sidecar = checkpoint.parent / "run_contract.json"
    if not sidecar.is_file():
        if expected["task"] == PICK_PLACE_TASK_ID:
            raise FileNotFoundError(
                f"Pick & Place resume checkpoint is missing run contract sidecar: {sidecar}"
            )
        return None
    binding = json.loads(sidecar.read_text(encoding="utf-8"))
    validate_resume_binding(binding, expected)
    return binding

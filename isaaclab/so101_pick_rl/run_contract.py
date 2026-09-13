"""Select and hash the immutable task contract bound to an Isaac Lab run."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LIFT_TASK_ID = "SO101-LiftCube-v0"
PICK_PLACE_TASK_ID = "SO101-PickPlace-v0"
# Conservative diagnostic design limits, not calibrated real-robot tolerances.
GRASP_GATE_LIMITS = {
    "max_allowed_penetration_m": 0.001,
    "max_allowed_finger_table_penetration_m": 0.001,
    "max_preclose_cube_motion_m": 0.012,
    "max_abs_wrist_flex_deg": 75.0,
}
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


def load_grasp_training_gate(report_path: Path | None, task: str) -> dict[str, Any] | None:
    """Reject PickPlace training until a matching normal-grasp probe has passed.

    This is an evidence compatibility check, not authentication of measurements.
    Diagnostic collision overrides cannot authorize training the default geometry.
    """
    if task != PICK_PLACE_TASK_ID:
        return None
    if report_path is None:
        raise ValueError("Pick & Place requires --grasp_feasibility_report before training")
    data = report_path.read_bytes()
    report = json.loads(data)
    if not isinstance(report, dict):
        raise ValueError("normal-grasp feasibility report must be an object")
    limits = report.get("gate_limits", {})
    if not isinstance(limits, dict) or any(
        not isinstance(limits.get(name), (int, float))
        or isinstance(limits.get(name), bool)
        or not math.isfinite(limits[name])
        or not 0 <= limits[name] <= maximum
        for name, maximum in GRASP_GATE_LIMITS.items()
    ):
        raise ValueError("normal-grasp feasibility report has missing or relaxed diagnostic limits")
    if not isinstance(report.get("gates"), dict) or not isinstance(report.get("collision_probe"), dict):
        raise ValueError("normal-grasp feasibility report has malformed gates or collision metadata")
    current_commit = subprocess.check_output(["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"], text=True).strip()
    current_dirty = bool(subprocess.check_output(
        ["git", "-C", str(REPOSITORY_ROOT), "status", "--porcelain"], text=True).strip())
    required_gates = (
        "opened_near_cube_before_grasp", "cube_not_pushed_before_close", "bilateral_contact",
        "finger_cube_penetration_bounded", "finger_table_penetration_bounded", "cube_lifted",
        "cube_reached_target_xy", "wrist_flex_bounded", "controlled_release_observed",
        "full_task_success", "simulator_error_free",
    )
    if (report.get("schema") != "so101_pick_rl.grasp_feasibility_probe.v1"
            or report.get("status") != "feasible"
            or report.get("contract_sha256") != contract_sha256(task)
            or report.get("git_dirty") is not False or report.get("git_commit") != current_commit
            or current_dirty
            or any(report.get("gates", {}).get(name) is not True for name in required_gates)
            or report.get("collision_probe", {}).get("source_override") is not False):
        raise ValueError("normal-grasp feasibility report is failed, incomplete, overridden or contract-incompatible")
    return {"path": str(report_path.resolve()), "sha256": hashlib.sha256(data).hexdigest(),
            "classification": report["classification"], "source_commit": report.get("git_commit")}

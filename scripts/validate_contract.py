#!/usr/bin/env python3
"""Validate the cross-platform task contract without third-party packages."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TASK_SPEC = ROOT / "common" / "task_spec.json"
PICK_PLACE_SPEC = ROOT / "common" / "pick_place_spec.json"
SCHEMA_DIR = ROOT / "common" / "schemas"
EXPECTED_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_task_spec(spec: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    if spec.get("schema_version") != "0.1.0":
        errors.append("schema_version must be 0.1.0")

    status = spec.get("status")
    runtime_measurements_present = spec.get("provenance", {}).get("runtime_measurements_present")
    valid_statuses = {"design_default_requires_runtime_validation", "runtime_validated"}
    if status not in valid_statuses:
        errors.append(f"status must be one of {sorted(valid_statuses)}")
    if not isinstance(runtime_measurements_present, bool):
        errors.append("provenance.runtime_measurements_present must be boolean")
    elif status == "design_default_requires_runtime_validation" and runtime_measurements_present:
        errors.append("design status cannot claim runtime measurements")
    elif status == "runtime_validated" and not runtime_measurements_present:
        errors.append("runtime_validated status requires runtime measurements")

    task = spec.get("task", {})
    robot = spec.get("robot", {})
    control = spec.get("control", {})
    action = control.get("action", {})
    evaluation = spec.get("evaluation", {})

    joints = robot.get("joint_order")
    if joints != EXPECTED_JOINTS:
        errors.append(f"joint_order must be {EXPECTED_JOINTS}")
    if action.get("dimension") != len(EXPECTED_JOINTS):
        errors.append("action.dimension must match joint_order length")
    if action.get("type") != "joint_position_delta":
        errors.append("action.type must be joint_position_delta")
    if action.get("unit") != "radian":
        errors.append("action.unit must be radian")
    if not isinstance(action.get("clip_abs_rad"), (int, float)) or action.get("clip_abs_rad", 0) <= 0:
        errors.append("action.clip_abs_rad must be positive")

    physics_hz = control.get("physics_hz")
    policy_hz = control.get("policy_hz")
    if not isinstance(physics_hz, int) or not isinstance(policy_hz, int):
        errors.append("physics_hz and policy_hz must be integers")
    elif physics_hz <= 0 or policy_hz <= 0 or physics_hz % policy_hz != 0:
        errors.append("physics_hz must be a positive multiple of policy_hz")

    episode_seconds = task.get("episode_seconds")
    success = task.get("success", {})
    lift = success.get("minimum_delta_z_m")
    hold = success.get("minimum_hold_seconds")
    if not isinstance(episode_seconds, (int, float)) or episode_seconds <= 0:
        errors.append("task.episode_seconds must be positive")
    if not isinstance(lift, (int, float)) or lift <= 0:
        errors.append("success.minimum_delta_z_m must be positive")
    if not isinstance(hold, (int, float)) or hold <= 0:
        errors.append("success.minimum_hold_seconds must be positive")
    elif isinstance(episode_seconds, (int, float)) and hold >= episode_seconds:
        errors.append("success hold time must be shorter than an episode")

    seeds = evaluation.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(not isinstance(seed, int) or seed < 0 for seed in seeds):
        errors.append("evaluation.seeds must contain non-negative integers")
    elif len(seeds) != len(set(seeds)):
        errors.append("evaluation.seeds must be unique")
    episodes = evaluation.get("episodes_per_seed")
    if not isinstance(episodes, int) or episodes <= 0:
        errors.append("evaluation.episodes_per_seed must be a positive integer")
    ladder = evaluation.get("environment_scale_ladder")
    if not isinstance(ladder, list) or not ladder or ladder[0] != 1:
        errors.append("environment_scale_ladder must start with 1")
    elif any(not isinstance(v, int) or v <= 0 for v in ladder):
        errors.append("environment_scale_ladder must contain positive integers")
    elif ladder != sorted(set(ladder)):
        errors.append("environment_scale_ladder must contain increasing positive integers")

    if spec.get("observation", {}).get("real_robot_deployable") is not False:
        errors.append("v0.1.0 privileged-state observation must be marked non-deployable")
    return errors


def validate_pick_place_spec(spec: dict[str, Any]) -> list[str]:
    errors = validate_task_spec(spec)
    task = spec.get("task", {})
    success = task.get("success", {})
    observation = spec.get("observation", {})

    if task.get("id") != "SO101-PickPlace-v0":
        errors.append("task.id must be SO101-PickPlace-v0")
    if task.get("revision") != 2:
        errors.append("task.revision must be 2")

    positive_thresholds = (
        "minimum_delta_z_m",
        "minimum_carry_clearance_m",
        "grasp_hold_seconds",
        "minimum_hold_seconds",
        "placement_xy_tolerance_m",
        "placement_z_tolerance_m",
        "maximum_linear_speed_m_s",
        "maximum_angular_speed_rad_s",
        "minimum_ee_clearance_m",
        "minimum_gripper_open_fraction",
        "contact_threshold_n",
        "release_contact_threshold_n",
    )
    for name in positive_thresholds:
        value = success.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            errors.append(f"success.{name} must be finite and positive")

    open_fraction = success.get("minimum_gripper_open_fraction")
    if isinstance(open_fraction, (int, float)) and math.isfinite(open_fraction) and open_fraction > 1:
        errors.append("success.minimum_gripper_open_fraction must not exceed 1")
    contact_threshold = success.get("contact_threshold_n")
    release_threshold = success.get("release_contact_threshold_n")
    if (
        isinstance(contact_threshold, (int, float))
        and not isinstance(contact_threshold, bool)
        and isinstance(release_threshold, (int, float))
        and not isinstance(release_threshold, bool)
        and math.isfinite(contact_threshold)
        and math.isfinite(release_threshold)
        and release_threshold >= contact_threshold
    ):
        errors.append("success.release_contact_threshold_n must be below contact_threshold_n")

    for name in ("requires_grasped_lift_history", "requires_controlled_release_at_target"):
        if success.get(name) is not True:
            errors.append(f"success.{name} must be true")
    if success.get("history_sampling") != "every_physics_step":
        errors.append("success.history_sampling must be every_physics_step")
    if success.get("carry_clearance_scope") != "outside_target_xy_footprint":
        errors.append("success.carry_clearance_scope must be outside_target_xy_footprint")

    terms = observation.get("terms")
    term_dimensions = observation.get("term_dimensions")
    dimension = observation.get("dimension")
    if not isinstance(terms, list) or not isinstance(term_dimensions, list) or len(terms) != len(term_dimensions):
        errors.append("observation.terms and term_dimensions must be equal-length lists")
    elif any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in term_dimensions):
        errors.append("observation.term_dimensions must contain positive integers")
    else:
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension != sum(term_dimensions):
            errors.append("observation.dimension must equal the sum of term_dimensions")
        if "episode_phase_state" not in terms:
            errors.append("observation.terms must include episode_phase_state")
        elif term_dimensions[terms.index("episode_phase_state")] != 6:
            errors.append("observation.episode_phase_state must have dimension 6")
    if observation.get("episode_phase_fields") != [
        "picked", "carry_valid", "released", "previous_release_ready", "lift_hold_fraction", "stable_hold_fraction"
    ]:
        errors.append("observation.episode_phase_fields must match the six runtime state fields")
    reward = spec.get("reward", {})
    rates = reward.get("positive_rates", {})
    expected_rates = {"reaching_cube", "gripper_alignment", "finger_contact", "lift_progress",
                      "transport", "placement", "release_and_retreat", "stable_placement"}
    valid_rates = (isinstance(rates, dict) and set(rates) == expected_rates and all(
        not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v) and v >= 0
        for v in rates.values()))
    if not valid_rates:
        errors.append("reward.positive_rates must contain the runtime terms with finite non-negative rates")
    gamma = reward.get("discount_gamma")
    bonus = reward.get("terminal_success_bonus")
    valid_gamma = not isinstance(gamma, bool) and isinstance(gamma, (int, float)) and math.isfinite(gamma) and 0 < gamma < 1
    valid_bonus = not isinstance(bonus, bool) and isinstance(bonus, (int, float)) and math.isfinite(bonus) and bonus > 0
    if not valid_gamma:
        errors.append("reward.discount_gamma must be finite and between zero and one")
    if not valid_bonus:
        errors.append("reward.terminal_success_bonus must be finite and positive")
    policy_hz = spec.get("control", {}).get("policy_hz")
    if valid_rates and valid_gamma and valid_bonus and isinstance(policy_hz, int) and policy_hz > 0:
        if bonus <= sum(rates.values()) / policy_hz / (1 - gamma):
            errors.append("reward.terminal_success_bonus must dominate the discounted dense reward bound")
    return errors


def validate_schema_documents() -> list[str]:
    errors: list[str] = []
    required_names = {"run_manifest.schema.json", "evaluation_result.schema.json"}
    found = {path.name for path in SCHEMA_DIR.glob("*.json")}
    missing = required_names - found
    if missing:
        errors.append(f"missing schema documents: {sorted(missing)}")
    for path in sorted(SCHEMA_DIR.glob("*.json")):
        try:
            schema = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
            continue
        if schema.get("type") != "object" or not schema.get("required"):
            errors.append(f"{path.relative_to(ROOT)} must define object and required fields")
    return errors


def validate_markdown_links() -> list[str]:
    errors: list[str] = []
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for document in sorted(ROOT.rglob("*.md")):
        for target in link_pattern.findall(document.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "#")):
                continue
            relative_target = target.split("#", 1)[0]
            if not relative_target:
                continue
            resolved = (document.parent / relative_target).resolve()
            if not resolved.exists():
                errors.append(f"{document.relative_to(ROOT)} -> missing {relative_target}")
    return errors


def main() -> int:
    try:
        spec = json.loads(TASK_SPEC.read_text(encoding="utf-8"))
        pick_place_spec = json.loads(PICK_PLACE_SPEC.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"CONTRACT_FAIL: {exc}", file=sys.stderr)
        return 1

    errors = validate_task_spec(spec)
    errors.extend(validate_pick_place_spec(pick_place_spec))
    errors.extend(validate_schema_documents())
    errors.extend(validate_markdown_links())
    if errors:
        for error in errors:
            print(f"CONTRACT_FAIL: {error}", file=sys.stderr)
        return 1

    print("CONTRACT_PASS")
    print(f"contract_sha256={canonical_sha256(spec)}")
    print(f"pick_place_contract_sha256={canonical_sha256(pick_place_spec)}")
    print(f"joint_count={len(spec['robot']['joint_order'])}")
    print(
        "evaluation_episodes_per_condition="
        f"{len(spec['evaluation']['seeds']) * spec['evaluation']['episodes_per_seed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

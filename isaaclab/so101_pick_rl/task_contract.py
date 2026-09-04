"""Load the cross-simulator task contract as the Isaac implementation's source of truth."""

from __future__ import annotations

import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TASK_SPEC_PATH = REPOSITORY_ROOT / "common" / "task_spec.json"
TASK_SPEC = json.loads(TASK_SPEC_PATH.read_text(encoding="utf-8"))

TASK_ID = TASK_SPEC["task"]["id"]
JOINT_ORDER = tuple(TASK_SPEC["robot"]["joint_order"])
PHYSICS_HZ = int(TASK_SPEC["control"]["physics_hz"])
POLICY_HZ = int(TASK_SPEC["control"]["policy_hz"])
DECIMATION = PHYSICS_HZ // POLICY_HZ
EPISODE_SECONDS = float(TASK_SPEC["task"]["episode_seconds"])
ACTION_SCALE_RAD = float(TASK_SPEC["control"]["action"]["clip_abs_rad"])
LIFT_THRESHOLD_M = float(TASK_SPEC["task"]["success"]["minimum_delta_z_m"])
SUCCESS_HOLD_SECONDS = float(TASK_SPEC["task"]["success"]["minimum_hold_seconds"])

if PHYSICS_HZ % POLICY_HZ:
    raise ValueError("task contract physics_hz must be divisible by policy_hz")
if len(JOINT_ORDER) != int(TASK_SPEC["control"]["action"]["dimension"]):
    raise ValueError("task contract joint order and action dimension disagree")

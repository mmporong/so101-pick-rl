"""Behavior-cloning helpers for the DemoBox absolute-target baseline.

This module trains only an approach/grasp/carry-prefix initializer for the
original source task.  It deliberately excludes the first opening command
after an 8 cm lift and every later transition, so it is neither a supported
placement expert nor evidence of full-task policy success.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from so101_pick_rl.demo_action_contract import (
    JOINT_NAMES,
    aligned_targets,
    normalize_targets,
    validate_source_timing,
)


OBSERVATION_FIELDS = (
    ("q", 6),
    ("dq", 6),
    ("previous_target", 6),
    ("cube_pose", 7),
    ("cube_velocity", 6),
    ("box_position", 3),
)
OBSERVATION_DIMENSION = sum(size for _, size in OBSERVATION_FIELDS)
ACTION_DIMENSION = 6
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SOURCE_STOP = 16
LIFT_HEIGHT_METERS = 0.08
OPENING_DELTA_RAD = 0.03
MAXIMUM_TARGET_STEP_NORM_RAD = 1.0


@dataclass(frozen=True)
class EpisodeTransitions:
    observations: np.ndarray
    actions: np.ndarray
    source_index: int
    source_episode: str
    audit: dict[str, Any]


@dataclass(frozen=True)
class BCDataset:
    train_observations: np.ndarray
    train_actions: np.ndarray
    validation_observations: np.ndarray
    validation_actions: np.ndarray
    audit: dict[str, Any]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(path: str | Path, expected_sha256: str | None = None) -> tuple[dict[str, Any], str]:
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise ValueError(f"contract SHA-256 mismatch: expected {expected_sha256}, got {sha256}")
    contract = json.loads(raw.decode("utf-8"))
    if contract.get("schema") != "so101_pick_rl.demo_box_training_contract.v1":
        raise ValueError("unsupported DemoBox training contract schema")
    if contract.get("task_id") != "SO101-DemoBox-Abs-v0":
        raise ValueError("unexpected DemoBox task_id")
    observation = contract.get("observation", {})
    ordered = tuple((item.get("name"), item.get("dimension")) for item in observation.get("ordered_fields", ()))
    if observation.get("dimension") != OBSERVATION_DIMENSION or ordered != OBSERVATION_FIELDS:
        raise ValueError("contract observation layout does not match the 34D BC adapter")
    if (observation.get("position_frame") != "environment_local"
            or observation.get("quaternion_order") != "wxyz"
            or observation.get("normalization")
            != "per_dimension mean/std fitted on training split only, std below 1e-6 replaced by 1"):
        raise ValueError("contract observation frames or normalization do not match the BC adapter")
    action = contract.get("action", {})
    if (action.get("dimension") != ACTION_DIMENSION
            or action.get("type") != "normalized_absolute_joint_targets"
            or tuple(action.get("joint_names", ())) != JOINT_NAMES):
        raise ValueError("contract action layout does not match the absolute-target BC adapter")
    _joint_limits(contract)
    control = contract.get("control", {})
    if (control.get("decimation") != 1 or isinstance(control.get("decimation"), bool)
            or not np.isclose(control.get("dt", np.nan), 1 / 60, rtol=0, atol=1e-12)):
        raise ValueError("BC source contract must preserve 60 Hz with decimation 1")
    expected_dataset_sha = contract.get("dataset", {}).get("expected_sha256")
    if not isinstance(expected_dataset_sha, str) or len(expected_dataset_sha) != 64:
        raise ValueError("contract dataset.expected_sha256 must be a SHA-256 hex string")
    try:
        bytes.fromhex(expected_dataset_sha)
    except ValueError as exc:
        raise ValueError("contract dataset.expected_sha256 must be hexadecimal") from exc
    dataset = contract["dataset"]
    transition_filter = dataset.get("transition_filter", {})
    prefix_filter = dataset.get("prefix_filter", {})
    split = dataset.get("split", {})
    if (dataset.get("read_only") is not True
            or dataset.get("input_action_field") != "obs/joint_pos_target[t+1]"
            or transition_filter.get("reject_out_of_bounds") is not True
            or transition_filter.get("maximum_target_step_l2_rad") != MAXIMUM_TARGET_STEP_NORM_RAD
            or prefix_filter.get("first_lift_m") != LIFT_HEIGHT_METERS
            or prefix_filter.get("opening_delta_rad") != OPENING_DELTA_RAD
            or prefix_filter.get("exclude_first_open_after_lift_and_all_later") is not True
            or split.get("train_source_index") != [0, 15]
            or split.get("validation_source_index") != [16, 19]
            or split.get("unit") != "whole_source_shard"):
        raise ValueError("contract dataset filters or deterministic split do not match the BC adapter")
    policy = contract.get("policy", {})
    if (policy.get("architecture") != [34, 128, 128, 6]
            or policy.get("hidden_activation") != "relu"
            or policy.get("output_activation") != "tanh"
            or policy.get("loss") != "normalized_action_mse"):
        raise ValueError("contract policy does not match the fixed BC model")
    return contract, sha256


def _joint_limits(contract: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    action = contract.get("action", {})
    lower = np.asarray(action.get("lower_rad"), dtype=np.float64)
    upper = np.asarray(action.get("upper_rad"), dtype=np.float64)
    if (lower.shape != (6,) or upper.shape != (6,) or not np.isfinite(lower).all()
            or not np.isfinite(upper).all() or not (lower < upper).all()):
        raise ValueError("contract action limits must be finite ordered six-joint arrays")
    return lower, upper


def source_split(source_index: int) -> str:
    if isinstance(source_index, bool) or not isinstance(source_index, (int, np.integer)) or source_index < 0:
        raise ValueError("source_index must be a non-negative integer")
    return "train" if source_index < TRAIN_SOURCE_STOP else "validation"


def _finite_matrix(value: Any, rows: int, columns: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (rows, columns) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite ({rows}, {columns}) array")
    return result


def _initial_row(episode: Any, path: str, columns: int) -> np.ndarray:
    if path not in episode:
        raise ValueError(f"missing required initial-state dataset: {path}")
    value = np.asarray(episode[path], dtype=np.float64)
    if value.shape == (columns,):
        value = value.reshape(1, columns)
    if value.shape != (1, columns) or not np.isfinite(value).all():
        raise ValueError(f"{path} must contain one finite {columns}D row")
    return value[0]


def _prestate(episode: Any, state_path: str, initial_path: str, rows: int, columns: int) -> np.ndarray:
    state = _finite_matrix(episode[state_path], rows, columns, state_path)
    initial = _initial_row(episode, initial_path, columns)
    return np.concatenate((initial[None, :], state[:-1]), axis=0)


def _object_observation_or_prestate(
    episode: Any,
    observation_path: str,
    state_path: str,
    initial_path: str,
    rows: int,
    columns: int,
) -> np.ndarray:
    if observation_path in episode:
        return _finite_matrix(episode[observation_path], rows, columns, observation_path)
    return _prestate(episode, state_path, initial_path, rows, columns)


def episode_transitions(episode: Any, contract: Mapping[str, Any]) -> EpisodeTransitions:
    """Convert one successful numeric episode into masked BC transitions.

    Malformed state is rejected rather than imputed.  Target-bound and target-
    discontinuity violations reject individual transitions and are counted.
    """
    raw_source_index = episode.attrs.get("source_index")
    if (isinstance(raw_source_index, (bool, np.bool_))
            or not isinstance(raw_source_index, (int, np.integer))
            or raw_source_index < 0):
        raise ValueError("episode source_index must be a non-negative integer")
    source_index = int(raw_source_index)
    source_episode = str(episode.attrs.get("source_episode", "unknown"))
    if not bool(episode.attrs.get("success", False)):
        raise ValueError(f"episode is not explicitly successful: {source_episode}")

    q = np.asarray(episode["obs/joint_pos"], dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 6 or len(q) < 2 or not np.isfinite(q).all():
        raise ValueError("obs/joint_pos must be a finite (T, 6) array with T >= 2")
    rows = len(q)
    dq = _finite_matrix(episode["obs/joint_vel"], rows, 6, "obs/joint_vel")
    target = _finite_matrix(episode["obs/joint_pos_target"], rows, 6, "obs/joint_pos_target")
    post_q = _finite_matrix(
        episode["states/articulation/robot/joint_position"], rows, 6,
        "states/articulation/robot/joint_position",
    )
    labels = aligned_targets(q, target, post_q)

    cube_pose = _object_observation_or_prestate(
        episode, "obs/cube_pose", "states/rigid_object/cube/root_pose",
        "initial_state/rigid_object/cube/root_pose", rows, 7,
    )
    cube_velocity = _object_observation_or_prestate(
        episode, "obs/cube_velocity", "states/rigid_object/cube/root_velocity",
        "initial_state/rigid_object/cube/root_velocity", rows, 6,
    )
    if "obs/box_position" in episode:
        box_position = _finite_matrix(episode["obs/box_position"], rows, 3, "obs/box_position")
    else:
        box_pose = _prestate(
            episode, "states/rigid_object/box_target/root_pose",
            "initial_state/rigid_object/box_target/root_pose", rows, 7,
        )
        box_position = box_pose[:, :3]

    transition_observations = np.concatenate(
        (q[:-1], dq[:-1], target[:-1], cube_pose[:-1], cube_velocity[:-1], box_position[:-1]), axis=1,
    )
    if transition_observations.shape != (rows - 1, OBSERVATION_DIMENSION):
        raise AssertionError("internal BC observation layout mismatch")

    lower, upper = _joint_limits(contract)
    in_bounds = ((labels >= lower) & (labels <= upper)).all(axis=1)
    target_steps = np.linalg.norm(labels - target[:-1], axis=1)
    continuous = target_steps < MAXIMUM_TARGET_STEP_NORM_RAD

    lift = cube_pose[:-1, 2] - cube_pose[0, 2]
    opening = labels[:, -1] - target[:-1, -1]
    lift_candidates = np.flatnonzero(lift >= LIFT_HEIGHT_METERS)
    first_lift = int(lift_candidates[0]) if len(lift_candidates) else None
    release_candidates = (
        np.flatnonzero((np.arange(rows - 1) >= first_lift) & (opening > OPENING_DELTA_RAD))
        if first_lift is not None else np.empty(0, dtype=np.int64)
    )
    release_start = int(release_candidates[0]) if len(release_candidates) else None
    before_release = np.ones(rows - 1, dtype=bool)
    if release_start is not None:
        before_release[release_start:] = False

    accepted = in_bounds & continuous & before_release
    if not accepted.any():
        raise ValueError(f"episode has zero eligible BC transitions: {source_episode}")
    normalized_actions = normalize_targets(labels[accepted], lower, upper).astype(np.float32)
    observations = transition_observations[accepted].astype(np.float32)
    audit = {
        "source_index": source_index,
        "source_episode": source_episode,
        "candidate_transitions": rows - 1,
        "accepted_transitions": int(accepted.sum()),
        "rejected_out_of_bounds": int((~in_bounds).sum()),
        "rejected_discontinuous": int((~continuous).sum()),
        "excluded_release_or_later": int((~before_release).sum()),
        "first_lift_transition": first_lift,
        "release_start_transition": release_start,
        "maximum_target_step_norm_rad": float(target_steps.max()),
        "purpose": "original_source_task_approach_grasp_carry_prefix_initialization",
        "supported_placement_training_eligible": False,
        "full_task_success_label": False,
    }
    return EpisodeTransitions(observations, normalized_actions, source_index, source_episode, audit)


def load_bc_dataset(path: str | Path, contract: Mapping[str, Any]) -> BCDataset:
    """Load Mimic numeric data read-only with whole-source-shard splitting."""
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to load the Mimic numeric dataset") from exc

    path = Path(path).expanduser().resolve()
    train_obs: list[np.ndarray] = []
    train_actions: list[np.ndarray] = []
    validation_obs: list[np.ndarray] = []
    validation_actions: list[np.ndarray] = []
    episodes: list[dict[str, Any]] = []
    source_splits: dict[int, str] = {}
    timing_validated_sources: list[int] = []
    with h5py.File(path, "r") as dataset:
        if dataset.attrs.get("schema") != "so101_pick_rl.mimic_numeric_export.v1":
            raise ValueError("unsupported Mimic numeric dataset schema")
        if "data" not in dataset or not dataset["data"]:
            raise ValueError("Mimic numeric dataset contains no episodes")
        if "source_metadata" not in dataset or not dataset["source_metadata"]:
            raise ValueError("Mimic numeric dataset contains no source timing metadata")
        for name in sorted(dataset["source_metadata"].keys()):
            source = dataset[f"source_metadata/{name}"]
            if "data_attrs" not in source or "env_args" not in source["data_attrs"].attrs:
                raise ValueError(f"source timing metadata is missing env_args: {name}")
            env_args = json.loads(source["data_attrs"].attrs["env_args"])
            validate_source_timing(env_args)
            try:
                timing_validated_sources.append(int(name.removeprefix("source_")))
            except ValueError as exc:
                raise ValueError(f"invalid source metadata name: {name}") from exc
        for name in sorted(dataset["data"].keys()):
            converted = episode_transitions(dataset[f"data/{name}"], contract)
            split = source_split(converted.source_index)
            previous = source_splits.setdefault(converted.source_index, split)
            if previous != split:
                raise AssertionError("a source shard crossed deterministic split boundaries")
            if split == "train":
                train_obs.append(converted.observations)
                train_actions.append(converted.actions)
            else:
                validation_obs.append(converted.observations)
                validation_actions.append(converted.actions)
            episodes.append(dict(converted.audit, split=split, output_episode=name))
    if not train_obs or not validation_obs:
        raise ValueError("dataset must contain both train source shards <16 and validation shards >=16")
    train_sources = sorted(index for index, split in source_splits.items() if split == "train")
    validation_sources = sorted(index for index, split in source_splits.items() if split == "validation")
    if set(train_sources) & set(validation_sources):
        raise AssertionError("source shard leakage across train and validation splits")
    return BCDataset(
        np.concatenate(train_obs), np.concatenate(train_actions),
        np.concatenate(validation_obs), np.concatenate(validation_actions),
        {
            "episodes": episodes,
            "episode_count": len(episodes),
            "train_source_indices": train_sources,
            "validation_source_indices": validation_sources,
            "source_timing_validated_indices": timing_validated_sources,
            "train_transitions": int(sum(len(value) for value in train_obs)),
            "validation_transitions": int(sum(len(value) for value in validation_obs)),
            "lineage_independence_known": False,
            "independent_heldout_success_claimed": False,
            "evaluation_scope": "loss_only_not_policy_success",
        },
    )


def fit_observation_normalizer(train_observations: Any) -> tuple[np.ndarray, np.ndarray]:
    observations = np.asarray(train_observations, dtype=np.float64)
    if (observations.ndim != 2 or observations.shape[1] != OBSERVATION_DIMENSION
            or len(observations) == 0 or not np.isfinite(observations).all()):
        raise ValueError("training observations must be a non-empty finite (N, 34) array")
    mean = observations.mean(axis=0)
    std = observations.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_observations(observations: Any, mean: Any, std: Any) -> np.ndarray:
    observations = np.asarray(observations, dtype=np.float32)
    mean, std = np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)
    if (observations.ndim != 2 or observations.shape[1] != OBSERVATION_DIMENSION
            or mean.shape != (OBSERVATION_DIMENSION,) or std.shape != (OBSERVATION_DIMENSION,)
            or not np.isfinite(observations).all() or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or (std <= 0).any()):
        raise ValueError("invalid observation normalization inputs")
    return ((observations - mean) / std).astype(np.float32)


def verify_smoke_reports(paths: list[str | Path], contract_sha256: str) -> list[dict[str, Any]]:
    if len(paths) != 2:
        raise ValueError("exactly two smoke reports are required")
    reports = []
    by_envs: dict[int, dict[str, Any]] = {}
    for path in paths:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        checks = report.get("checks")
        if (report.get("status") != "passed" or report.get("task_contract_sha256") != contract_sha256
                or not isinstance(checks, dict) or not checks or not all(value is True for value in checks.values())):
            raise ValueError(f"smoke report did not pass its bound contract: {path}")
        num_envs = report.get("num_envs")
        if num_envs not in (1, 64) or num_envs in by_envs:
            raise ValueError("smoke reports must contain one unique 1-env and 64-env result")
        code_sha256 = report.get("code_sha256")
        required_code = {
            "common/demo_box_spec.json": contract_sha256,
            "isaaclab/scripts/smoke_demo_box.py": file_sha256(REPOSITORY_ROOT / "isaaclab/scripts/smoke_demo_box.py"),
            "isaaclab/so101_pick_rl/demo_box_runtime.py": file_sha256(
                REPOSITORY_ROOT / "isaaclab/so101_pick_rl/demo_box_runtime.py"
            ),
            "isaaclab/so101_pick_rl/demo_bc.py": file_sha256(
                REPOSITORY_ROOT / "isaaclab/so101_pick_rl/demo_bc.py"
            ),
            "isaaclab/so101_pick_rl/demo_action_contract.py": file_sha256(
                REPOSITORY_ROOT / "isaaclab/so101_pick_rl/demo_action_contract.py"
            ),
        }
        if not isinstance(code_sha256, dict) or any(
            code_sha256.get(name) != digest for name, digest in required_code.items()
        ):
            raise ValueError("smoke report code hashes do not match the current BC/runtime contract code")
        by_envs[num_envs] = report
        reports.append(report)
    if set(by_envs) != {1, 64}:
        raise ValueError("smoke reports must cover exactly 1 and 64 environments")
    return reports


def build_policy():
    """Construct the fixed 34→128→128→6 tanh model lazily."""
    import torch

    return torch.nn.Sequential(
        torch.nn.Linear(OBSERVATION_DIMENSION, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, ACTION_DIMENSION),
        torch.nn.Tanh(),
    )

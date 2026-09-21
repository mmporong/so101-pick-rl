import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab"))
from so101_pick_rl.demo_action_contract import JOINT_NAMES
from so101_pick_rl.demo_bc import (
    OBSERVATION_DIMENSION,
    build_policy,
    episode_transitions,
    file_sha256,
    fit_observation_normalizer,
    load_bc_dataset,
    load_contract,
    normalize_observations,
    source_split,
    verify_smoke_reports,
)


def contract():
    return {
        "action": {
            "joint_names": list(JOINT_NAMES),
            "lower_rad": [-2.0] * 6,
            "upper_rad": [2.0] * 6,
        },
    }


def add_episode(data, name, source_index, *, release=True, bad_alignment=False):
    rows = 5
    episode = data.create_group(name)
    episode.attrs["success"] = True
    episode.attrs["source_index"] = source_index
    episode.attrs["source_episode"] = f"source_{name}"
    obs = episode.create_group("obs")
    q = np.arange(rows * 6, dtype=np.float32).reshape(rows, 6) / 100
    obs.create_dataset("joint_pos", data=q)
    obs.create_dataset("joint_vel", data=np.full((rows, 6), 0.02, dtype=np.float32))
    target = np.zeros((rows, 6), dtype=np.float32)
    if release:
        target[3:, -1] = 0.1
    obs.create_dataset("joint_pos_target", data=target)

    states = episode.create_group("states")
    robot = states.create_group("articulation").create_group("robot")
    post_q = np.concatenate((q[1:], q[-1:]))
    if bad_alignment:
        post_q[0, 0] += 0.1
    robot.create_dataset("joint_position", data=post_q)
    robot.create_dataset("joint_velocity", data=np.zeros((rows, 6), dtype=np.float32))
    objects = states.create_group("rigid_object")
    cube = objects.create_group("cube")
    cube_pose = np.tile(np.array([0.3, -0.4, 0.05, 1, 0, 0, 0], dtype=np.float32), (rows, 1))
    cube_pose[1:, 2] = 0.14
    cube.create_dataset("root_pose", data=cube_pose)
    cube.create_dataset("root_velocity", data=np.zeros((rows, 6), dtype=np.float32))
    box = objects.create_group("box_target")
    box.create_dataset(
        "root_pose", data=np.tile(np.array([0.58, -0.35, 0.045, 1, 0, 0, 0]), (rows, 1)),
    )
    box.create_dataset("root_velocity", data=np.zeros((rows, 6), dtype=np.float32))

    initial = episode.create_group("initial_state")
    initial_robot = initial.create_group("articulation").create_group("robot")
    initial_robot.create_dataset("joint_position", data=q[:1])
    initial_robot.create_dataset("joint_velocity", data=np.zeros((1, 6), dtype=np.float32))
    initial_objects = initial.create_group("rigid_object")
    initial_cube = initial_objects.create_group("cube")
    initial_cube.create_dataset("root_pose", data=cube_pose[:1])
    initial_cube.create_dataset("root_velocity", data=np.zeros((1, 6), dtype=np.float32))
    initial_box = initial_objects.create_group("box_target")
    initial_box.create_dataset(
        "root_pose", data=np.array([[0.58, -0.35, 0.045, 1, 0, 0, 0]], dtype=np.float32),
    )
    initial_box.create_dataset("root_velocity", data=np.zeros((1, 6), dtype=np.float32))
    return episode


def add_source_metadata(output, *source_indices):
    metadata = output.require_group("source_metadata")
    for source_index in source_indices:
        source = metadata.create_group(f"source_{source_index:04d}")
        attrs = source.create_group("data_attrs").attrs
        attrs["env_args"] = json.dumps({"sim_args": {"dt": 1 / 60, "decimation": 1}})


class DemoBCTests(unittest.TestCase):
    def test_residual_contract_factory_and_parent_physics(self):
        import torch
        root = Path(__file__).resolve().parents[1]
        base, _ = load_contract(root / "common/demo_box_spec.json")
        variant, _ = load_contract(root / "common/demo_box_residual_spec.json")
        for key in ("action", "observation", "control", "scene", "dataset", "success"):
            self.assertEqual(base[key], variant[key])
        model = build_policy(variant, np.zeros(34), np.ones(34))
        obs = torch.zeros((2, 34))
        result = model(obs)
        lower = torch.tensor(variant["action"]["lower_rad"])
        upper = torch.tensor(variant["action"]["upper_rad"])
        expected = (2 * (obs[:, 12:18] - lower) / (upper - lower) - 1).clamp(-1, 1)
        torch.testing.assert_close(result, expected)

    def test_repository_contract_is_strictly_bound_by_raw_file_hash(self):
        path = Path(__file__).resolve().parents[1] / "common" / "demo_box_spec.json"
        expected = file_sha256(path)
        loaded, actual = load_contract(path, expected)
        self.assertEqual(actual, expected)
        self.assertEqual(loaded["task_id"], "SO101-DemoBox-Abs-v0")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            load_contract(path, "0" * 64)

    def test_alignment_uses_next_target_and_reconstructs_prestate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.h5"
            with h5py.File(path, "w") as output:
                episode = add_episode(output.create_group("data"), "demo_0", 0, release=False)
                converted = episode_transitions(episode, contract())
            self.assertEqual(converted.observations.shape, (4, 34))
            self.assertEqual(converted.actions.shape, (4, 6))
            np.testing.assert_allclose(converted.observations[0, 18:25], [0.3, -0.4, 0.05, 1, 0, 0, 0])
            np.testing.assert_allclose(converted.observations[1, 18:25], [0.3, -0.4, 0.05, 1, 0, 0, 0])
            np.testing.assert_allclose(converted.observations[2, 18:25], [0.3, -0.4, 0.14, 1, 0, 0, 0])
            np.testing.assert_allclose(converted.observations[0, 31:34], [0.58, -0.35, 0.045])
            np.testing.assert_allclose(converted.actions, np.zeros((4, 6)))

    def test_release_and_later_transitions_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.h5"
            with h5py.File(path, "w") as output:
                episode = add_episode(output.create_group("data"), "demo_0", 0, release=True)
                converted = episode_transitions(episode, contract())
            self.assertEqual(converted.audit["release_start_transition"], 2)
            self.assertEqual(converted.audit["excluded_release_or_later"], 2)
            self.assertEqual(converted.audit["accepted_transitions"], 2)
            self.assertFalse(converted.audit["supported_placement_training_eligible"])
            self.assertFalse(converted.audit["full_task_success_label"])

    def test_release_filter_latches_lift_after_cube_is_lowered(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.h5"
            with h5py.File(path, "w") as output:
                episode = add_episode(output.create_group("data"), "demo_0", 0, release=True)
                cube_pose = episode["states/rigid_object/cube/root_pose"][:]
                cube_pose[2:, 2] = 0.05
                del episode["states/rigid_object/cube/root_pose"]
                episode["states/rigid_object/cube"].create_dataset("root_pose", data=cube_pose)
                target = episode["obs/joint_pos_target"][:]
                target[3, -1] = 0.0
                del episode["obs/joint_pos_target"]
                episode["obs"].create_dataset("joint_pos_target", data=target)
                converted = episode_transitions(episode, contract())
            self.assertEqual(converted.audit["first_lift_transition"], 2)
            self.assertEqual(converted.audit["release_start_transition"], 3)
            self.assertEqual(converted.audit["accepted_transitions"], 3)

    def test_out_of_bounds_and_one_radian_jump_are_rejected_and_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.h5"
            with h5py.File(path, "w") as output:
                episode = add_episode(output.create_group("data"), "demo_0", 0, release=False)
                target = episode["obs/joint_pos_target"][:]
                target[1, 0] = 1.0
                target[2, 0] = 3.0
                del episode["obs/joint_pos_target"]
                episode["obs"].create_dataset("joint_pos_target", data=target)
                converted = episode_transitions(episode, contract())
            self.assertEqual(converted.audit["rejected_out_of_bounds"], 1)
            self.assertGreaterEqual(converted.audit["rejected_discontinuous"], 2)
            self.assertEqual(converted.audit["accepted_transitions"], 1)

    def test_malformed_alignment_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.h5"
            with h5py.File(path, "w") as output:
                episode = add_episode(
                    output.create_group("data"), "demo_0", 0, release=False, bad_alignment=True,
                )
                with self.assertRaisesRegex(ValueError, "pre-step"):
                    episode_transitions(episode, contract())

    def test_whole_source_shard_split_has_no_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.h5"
            with h5py.File(path, "w") as output:
                output.attrs["schema"] = "so101_pick_rl.mimic_numeric_export.v1"
                data = output.create_group("data")
                add_source_metadata(output, 0, 15, 16, 19)
                add_episode(data, "demo_0", 0, release=False)
                add_episode(data, "demo_1", 15, release=False)
                add_episode(data, "demo_2", 16, release=False)
                add_episode(data, "demo_3", 19, release=False)
            dataset = load_bc_dataset(path, contract())
            self.assertEqual(dataset.audit["train_source_indices"], [0, 15])
            self.assertEqual(dataset.audit["validation_source_indices"], [16, 19])
            self.assertEqual(dataset.audit["train_transitions"], 8)
            self.assertEqual(dataset.audit["validation_transitions"], 8)
            self.assertFalse(dataset.audit["lineage_independence_known"])
            self.assertFalse(dataset.audit["independent_heldout_success_claimed"])
            self.assertEqual(source_split(15), "train")
            self.assertEqual(source_split(16), "validation")

    def test_normalizer_is_fit_only_from_supplied_training_rows(self):
        train = np.vstack((np.zeros((1, 34)), np.full((1, 34), 2))).astype(np.float32)
        validation = np.full((1, 34), 101, dtype=np.float32)
        mean, std = fit_observation_normalizer(train)
        np.testing.assert_array_equal(mean, np.ones(34))
        np.testing.assert_array_equal(std, np.ones(34))
        np.testing.assert_array_equal(normalize_observations(train, mean, std), np.vstack((-np.ones(34), np.ones(34))))
        np.testing.assert_array_equal(normalize_observations(validation, mean, std), np.full((1, 34), 100))
        _, constant_std = fit_observation_normalizer(np.zeros((2, 34), dtype=np.float32))
        np.testing.assert_array_equal(constant_std, np.ones(34, dtype=np.float32))

    def test_smoke_pair_requires_bound_one_and_64_environment_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(__file__).resolve().parents[1]
            contract_sha = file_sha256(root / "common/demo_box_spec.json")
            code_sha256 = {
                relative: file_sha256(root / relative)
                for relative in (
                    "common/demo_box_spec.json",
                    "isaaclab/scripts/smoke_demo_box.py",
                    "isaaclab/so101_pick_rl/demo_box_runtime.py",
                    "isaaclab/so101_pick_rl/demo_bc.py",
                    "isaaclab/so101_pick_rl/demo_action_contract.py",
                )
            }
            paths = []
            for num_envs in (1, 64):
                path = Path(directory) / f"smoke{num_envs}.json"
                path.write_text(json.dumps({
                    "status": "passed",
                    "num_envs": num_envs,
                    "task_contract_sha256": contract_sha,
                    "code_sha256": code_sha256,
                    "checks": {"finite": True, "shape": True},
                }), encoding="utf-8")
                paths.append(path)
            reports = verify_smoke_reports(paths, contract_sha)
            self.assertEqual({report["num_envs"] for report in reports}, {1, 64})
            bad = json.loads(paths[1].read_text(encoding="utf-8"))
            bad["checks"]["finite"] = False
            paths[1].write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_smoke_reports(paths, contract_sha)

    def test_policy_shape_and_tanh_range(self):
        import torch

        policy = build_policy()
        output = policy(torch.zeros((3, OBSERVATION_DIMENSION)))
        self.assertEqual(tuple(output.shape), (3, 6))
        self.assertTrue(bool((output.abs() <= 1).all()))


if __name__ == "__main__":
    unittest.main()

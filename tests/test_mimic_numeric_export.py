"""Tests for the successful Mimic numeric HDF5 exporter."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab" / "scripts"))
import export_mimic_numeric as exporter
from export_mimic_numeric import export_numeric, resolve_inputs


def write_source(path: Path, *, success=True, alignment_error=0.0, finite=True) -> None:
    observed = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]], dtype=np.float32)
    state = np.array([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]], dtype=np.float32)
    observed[1, 0] += alignment_error
    target = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    actions = target.copy()
    if not finite:
        actions[0, 0] = np.nan
    with h5py.File(path, "w") as output:
        output.attrs["sim_dt"] = 1.0 / 120.0
        output.attrs["decimation"] = 4
        data = output.create_group("data")
        data.attrs["total"] = 3
        data.attrs["env_args"] = '{"sim":{"dt":0.008333333333333333},"decimation":4}'
        episode = data.create_group("demo_7")
        if success is not None:
            episode.attrs["success"] = success
        episode.attrs["num_samples"] = 3
        episode.create_dataset("actions", data=actions, chunks=(10, 2), maxshape=(None, 2))
        initial = episode.create_group("initial_state")
        initial.create_dataset("robot", data=np.array([1.0, 2.0]))
        obs = episode.create_group("obs")
        obs.create_dataset("joint_pos", data=observed)
        obs.create_dataset("joint_pos_target", data=target)
        obs.create_dataset("front_rgb", data=np.full((3, 4, 5, 3), 17, dtype=np.uint8))
        states = episode.create_group("states")
        articulation = states.create_group("articulation")
        robot = articulation.create_group("robot")
        robot.create_dataset("joint_position", data=state)


class MimicNumericExportTests(unittest.TestCase):
    def test_exports_numeric_data_manifest_and_rgb_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "shard_000.hdf5"
            output = root / "numeric.hdf5"
            write_source(source)

            manifest = export_numeric([root], output)

            self.assertEqual(manifest["episode_count"], 1)
            self.assertTrue(manifest["episodes"][0]["alignment"]["passed"])
            self.assertEqual(len(manifest["sources"][0]["rgb_datasets"]), 1)
            self.assertEqual(len(manifest["sources"][0]["numeric_content_sha256"]), 64)
            with h5py.File(output, "r") as exported:
                demo = exported["data/demo_0"]
                np.testing.assert_array_equal(demo["actions"][:], np.array([
                    [0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32))
                self.assertNotIn("front_rgb", demo["obs"])
                self.assertEqual(demo.attrs["source_episode"], "demo_7")
                self.assertAlmostEqual(exported["source_metadata/source_0000"].attrs["sim_dt"], 1 / 120)
                self.assertEqual(exported["source_metadata/source_0000"].attrs["decimation"], 4)
                self.assertIn("sim", exported["source_metadata/source_0000/data_attrs"].attrs["env_args"])
                embedded = json.loads(exported["manifest/json"][()].decode("utf-8"))
                self.assertEqual(embedded["rgb_policy"], "datasets_not_decoded")
                self.assertFalse(embedded["training_ready"])
                self.assertFalse(embedded["physics_replay_validated"])
                self.assertFalse(embedded["full_file_bytes_read_for_sha256"])
                self.assertEqual(embedded["source_timing_validation"], "pending_training_adapter")

    def test_directory_excludes_failed_shards_and_orders_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_source(root / "shard_002.hdf5")
            write_source(root / "shard_001_failed.hdf5", success=False)
            write_source(root / "shard_000.hdf5")
            resolved = resolve_inputs([root])
            self.assertEqual([path.name for path in resolved], ["shard_000.hdf5", "shard_002.hdf5"])

    def test_rejects_missing_or_failed_success_attribute(self):
        for success in (None, False):
            with self.subTest(success=success), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source.hdf5"
                write_source(source, success=success)
                with self.assertRaisesRegex(ValueError, "success|failed episode"):
                    export_numeric([source], root / "numeric.hdf5")

    def test_option_skips_only_explicit_failed_episode_and_records_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "mixed.hdf5"
            output = root / "numeric.hdf5"
            write_source(source)
            with h5py.File(source, "a") as dataset:
                dataset.copy("data/demo_7", "data/demo_8")
                dataset["data/demo_8"].attrs["success"] = False

            manifest = export_numeric([source], output, skip_failed_episodes=True)

            self.assertEqual(manifest["episode_count"], 1)
            self.assertEqual(manifest["skipped_episodes"], [{
                "source_index": 0,
                "source_episode": "demo_8",
                "reason": "success=false",
            }])

    def test_skip_option_does_not_hide_missing_success_or_zero_success_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing.hdf5"
            write_source(missing, success=None)
            with self.assertRaisesRegex(ValueError, "no explicit success"):
                export_numeric([missing], root / "missing-out.hdf5", skip_failed_episodes=True)

            failed = root / "all_unsuccessful.hdf5"
            write_source(failed, success=False)
            with self.assertRaisesRegex(ValueError, "zero successful episodes"):
                export_numeric([failed], root / "failed-out.hdf5", skip_failed_episodes=True)

    def test_rejects_shape_sample_count_and_empty_source_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target_mismatch = root / "target-mismatch.hdf5"
            write_source(target_mismatch)
            with h5py.File(target_mismatch, "a") as dataset:
                del dataset["data/demo_7/obs/joint_pos_target"]
                dataset["data/demo_7/obs"].create_dataset(
                    "joint_pos_target", data=np.zeros((2, 2), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "shapes differ"):
                export_numeric([target_mismatch], root / "target-out.hdf5")

            samples_mismatch = root / "samples-mismatch.hdf5"
            write_source(samples_mismatch)
            with h5py.File(samples_mismatch, "a") as dataset:
                dataset["data/demo_7"].attrs["num_samples"] = 2
            with self.assertRaisesRegex(ValueError, "does not match num_samples"):
                export_numeric([samples_mismatch], root / "samples-out.hdf5")

            with h5py.File(samples_mismatch, "a") as dataset:
                dataset["data/demo_7/actions"].resize((2, 2))
            with self.assertRaisesRegex(ValueError, "observation time axis does not match"):
                export_numeric([samples_mismatch], root / "time-out.hdf5")

            empty = root / "empty.hdf5"
            with h5py.File(empty, "w") as dataset:
                dataset.create_group("data")
            with self.assertRaisesRegex(ValueError, "contains no episodes"):
                export_numeric([empty], root / "empty-out.hdf5")

    def test_rejects_alignment_error_and_nonfinite_numeric_data(self):
        cases = (({"alignment_error": 1e-3}, "alignment failed"), ({"finite": False}, "non-finite"))
        for options, message in cases:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source.hdf5"
                write_source(source, **options)
                output = root / "numeric.hdf5"
                with self.assertRaisesRegex(ValueError, message):
                    export_numeric([source], output)
                self.assertFalse(output.exists())

    def test_refuses_overwrite_and_same_input_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.hdf5"
            output = root / "numeric.hdf5"
            write_source(source)
            output.touch()
            with self.assertRaises(FileExistsError):
                export_numeric([source], output)
            with self.assertRaisesRegex(ValueError, "must differ"):
                export_numeric([source], source)

    def test_publish_race_never_overwrites_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.hdf5"
            output = root / "numeric.hdf5"
            write_source(source)

            def win_race(_temporary, destination):
                Path(destination).write_bytes(b"winner")
                raise FileExistsError("simulated publication race")

            with mock.patch.object(exporter.os, "link", side_effect=win_race):
                with self.assertRaises(FileExistsError):
                    export_numeric([source], output)
            self.assertEqual(output.read_bytes(), b"winner")
            self.assertEqual(list(root.glob("*.partial")), [])


if __name__ == "__main__":
    unittest.main()

"""Alignment and encoder validation without Isaac Sim or a GPU."""

import sys
import tempfile
import unittest
from unittest.mock import Mock
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab"))
from so101_pick_rl.capture_data import aligned_arrays, sha256_file, VideoWriter


class CaptureDataTests(unittest.TestCase):
    def test_alignment_keeps_terminal_state(self):
        arrays = aligned_arrays(
            [{"obs": np.array([1, 2])}, {"obs": np.array([3, 4])}],
            [{"action": np.array([0.1]), "done": True}], 30,
        )
        np.testing.assert_array_equal(arrays["obs"][-1], [3, 4])
        self.assertEqual(arrays["action"].shape, (1, 1))
        self.assertEqual(arrays["state_time_seconds"][-1], 1 / 30)
        self.assertEqual(arrays["transition_time_seconds"].tolist(), [0])

    def test_reset_state_cannot_replace_missing_terminal_state(self):
        with self.assertRaises(ValueError):
            aligned_arrays([{"obs": 1}], [{"action": 0}], 30)

    def test_empty_capture_rejected(self):
        with self.assertRaises(ValueError):
            aligned_arrays([], [], 30)

    def test_invalid_fps_rejected(self):
        with self.assertRaises(ValueError):
            aligned_arrays([{"obs": 1}, {"obs": 2}], [{"action": 0}], 0)

    def test_nonfinite_rejected(self):
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            aligned_arrays([{"obs": 1}, {"obs": np.nan}], [{"action": 0}], 30)

    def test_field_collision_rejected(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            aligned_arrays([{"obs": 1}, {"obs": 2}], [{"obs": 3}], 30)

    def test_encoder_refuses_existing_video(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.mp4"
            path.touch()
            with self.assertRaises(FileExistsError):
                VideoWriter(path, 64, 64, 30)

    def test_hash_is_content_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty"
            path.touch()
            self.assertEqual(sha256_file(path),
                             "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")

    def test_close_reaps_encoder_when_pipe_is_broken(self):
        writer = VideoWriter.__new__(VideoWriter)
        writer.closed = False
        writer.process = Mock()
        writer.process.stdin.close.side_effect = BrokenPipeError("encoder stopped")
        writer.process.wait.return_value = 1
        writer.log = Mock()
        with self.assertRaisesRegex(RuntimeError, "exited with 1"):
            writer.close()
        writer.process.wait.assert_called_once_with(timeout=30)
        writer.log.close.assert_called_once()
        writer.close()
        writer.process.wait.assert_called_once()


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from isaaclab.so101_pick_rl.parallel_video import (
    ParallelTrainingVideo,
    camera_views,
    interpolated_view,
    overlay_training_label,
)


class FakeWriter:
    def __init__(self, path, width, height, fps):
        self.path = path
        self.shape = (height, width, 3)
        self.frames = 0
        self.closed = False
        path.write_bytes(b"fake-video")

    def write(self, pixels):
        assert pixels.shape == self.shape and pixels.dtype == np.uint8
        self.frames += 1

    def close(self):
        self.closed = True


class FakeSim:
    def __init__(self):
        self.views = []
        self.renders = 0

    def set_camera_view(self, eye, lookat):
        self.views.append((eye, lookat))

    def render(self):
        self.renders += 1


class FakeEnv:
    def __init__(self, width=64, height=64):
        self.sim = FakeSim()
        self.frame = np.arange(width * height * 3, dtype=np.uint8).reshape(height, width, 3)
        self.render_calls = 0

    def render(self):
        self.render_calls += 1
        return self.frame.copy()


class ParallelVideoTests(unittest.TestCase):
    def test_camera_starts_full_then_smoothly_reaches_detail(self):
        origins = np.array([(x, y, 0) for x in range(32) for y in range(32)])
        overview, detail = camera_views(origins)
        self.assertEqual(overview.name, "full_1024_env_grid")
        self.assertEqual(detail.name, "detail_64_env_region")
        first, alpha0 = interpolated_view(overview, detail, 1, 30)
        middle, alpha_mid = interpolated_view(overview, detail, 181, 30)
        last, alpha1 = interpolated_view(overview, detail, 211, 30)
        self.assertEqual(first, overview)
        self.assertEqual(alpha0, 0.0)
        self.assertEqual(middle.name, "zoom_transition")
        self.assertGreater(alpha_mid, 0.0)
        self.assertLess(alpha_mid, 1.0)
        self.assertEqual(last, detail)
        self.assertEqual(alpha1, 1.0)

    def test_overlay_keeps_shape_and_changes_header(self):
        frame = np.full((64, 128, 3), 120, dtype=np.uint8)
        result = overlay_training_label(frame, 1024, 300)
        self.assertEqual(result.shape, frame.shape)
        self.assertEqual(result.dtype, np.uint8)
        self.assertFalse(np.array_equal(result[:38], frame[:38]))
        np.testing.assert_array_equal(result[55:], frame[55:])

    def test_records_one_post_step_frame_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "capture"
            origins = np.array([(x, y, 0) for x in range(2) for y in range(2)])
            capture = ParallelTrainingVideo(
                output, 64, 64, 30, 4, 2, 300, 1, origins, writer_factory=FakeWriter,
                provenance={"git": {"commit": "test-commit", "dirty": False}},
            )
            env = FakeEnv()
            capture.warm_up(env, render_count=3)
            self.assertEqual(capture.frames, 0)
            self.assertEqual(env.sim.renders, 3)
            self.assertEqual(len(env.sim.views), 1)
            capture.capture_post_step(env)
            capture.capture_post_step(env)
            capture.close(completed=True)
            manifest = json.loads((output / "manifest.json").read_text())
            rows = [json.loads(line) for line in (output / "frames.jsonl").read_text().splitlines()]
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["frames"], 2)
            self.assertEqual(manifest["provenance"]["git"]["commit"], "test-commit")
            self.assertEqual([row["global_iteration"] for row in rows], [300, 300])
            self.assertEqual([row["policy_step"] for row in rows], [1, 2])
            self.assertEqual(env.sim.renders, 5)
            self.assertTrue(capture.writer.closed)

    def test_rejects_existing_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileExistsError):
                ParallelTrainingVideo(
                    Path(temporary), 64, 64, 30, 1, 1, 0, 1, np.zeros((1, 3)),
                    writer_factory=FakeWriter,
                )


if __name__ == "__main__":
    unittest.main()

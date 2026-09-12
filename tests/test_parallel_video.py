import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from isaaclab.so101_pick_rl.parallel_video import (
    ParallelTrainingVideo,
    camera_views,
    interpolated_view,
    overlay_training_label,
    select_focus_region,
    cinematic_view,
    cinematic_interpolated_view,
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
    def test_focus_prefers_measured_success_in_compact_sixteen(self):
        origins = np.array([(x, y, 0) for x in range(8) for y in range(8)])
        scores = np.zeros(64)
        scores[27:30] = 1000
        scores[36] = 1000000
        ids = select_focus_region(origins, scores)
        self.assertEqual(len(ids), 16)
        self.assertIn(36, ids)
        self.assertTrue((np.ptp(origins[ids, :2], axis=0) <= 3).all())

    def test_cinematic_angle_and_continuous_dolly(self):
        origins = np.array([(x, y, 0) for x in range(32) for y in range(32)])
        wide = cinematic_view("wide", origins, overview=True)
        detail = cinematic_view("detail", origins[:4].copy())
        offset = np.array(wide.eye) - wide.lookat
        elevation = np.degrees(np.arctan2(offset[2], np.linalg.norm(offset[:2])))
        self.assertGreater(elevation, 30)
        self.assertLess(elevation, 45)
        self.assertAlmostEqual(elevation, 37.0, delta=1.0)
        first, _ = cinematic_interpolated_view(wide, detail, 1, 30)
        start, _ = cinematic_interpolated_view(wide, detail, 121, 30)
        end, alpha = cinematic_interpolated_view(wide, detail, 421, 30)
        np.testing.assert_allclose(first.eye, wide.eye)
        np.testing.assert_allclose(start.eye, wide.eye)
        np.testing.assert_allclose(end.eye, detail.eye)
        self.assertEqual(alpha, 1)
        self.assertLess(np.linalg.norm(np.array(end.eye) - end.lookat), np.linalg.norm(offset))

    def test_edge_hero_still_selects_full_four_by_four_block(self):
        origins = np.array([(x, y, 0) for x in range(8) for y in range(8)])
        scores = np.zeros(64)
        scores[0] = 1000000
        ids = select_focus_region(origins, scores)
        self.assertIn(0, ids)
        self.assertEqual(len(ids), 16)
        np.testing.assert_array_equal(np.ptp(origins[ids, :2], axis=0), [3, 3])

    def test_cinematic_selection_records_real_lift_not_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            origins = np.array([(x, y, 0) for x in range(4) for y in range(4)])
            capture = ParallelTrainingVideo(Path(temporary) / "capture", 64, 64, 1,
                16, 4, 340, 1, origins, writer_factory=FakeWriter, style="cinematic")
            env = FakeEnv()
            env.pick_place_state = SimpleNamespace(picked=np.ones(16, bool),
                carry_valid=np.ones(16, bool), released=np.zeros(16, bool))
            env.termination_manager = SimpleNamespace(get_term=lambda name: np.zeros(16, bool))
            for _ in range(4):
                capture.capture_post_step(env)
            capture.close(completed=True)
            assert capture.focus is not None
            self.assertEqual(capture.focus["qualified_lift_seen_count"], 16)
            self.assertEqual(capture.focus["full_task_success_seen_count"], 0)
            self.assertEqual(capture.focus["selected_at_policy_step"], 4)
            self.assertEqual(len(capture.focus["environment_ids"]), 16)
            components, weights = capture.focus["components"], capture.focus["score_weights"]
            expected = (np.array(components["carry_steps"]) * weights["carry_step"]
                        + np.array(components["released_steps"]) * weights["released_step"]
                        + np.array(components["qualified_lift_seen"]) * weights["qualified_lift_seen"]
                        + np.array(components["full_success_seen"]) * weights["full_success_seen"])
            np.testing.assert_array_equal(capture.focus["scores"], expected)

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

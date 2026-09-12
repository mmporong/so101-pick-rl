"""Portfolio video capture for a live vectorized PPO training run."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .capture_data import VideoWriter, sha256_file

FOCUS_SCORE_WEIGHTS = {"carry_step": 1, "released_step": 10, "qualified_lift_seen": 1000,
                       "full_success_seen": 1000000}


@dataclass(frozen=True)
class CameraView:
    name: str
    eye: tuple[float, float, float]
    lookat: tuple[float, float, float]


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _view_for_origins(name: str, origins: np.ndarray) -> CameraView:
    xy = origins[:, :2]
    lower, upper = xy.min(axis=0), xy.max(axis=0)
    center = (lower + upper) / 2.0
    diagonal = max(float(np.linalg.norm(upper - lower)), 1.0)
    return CameraView(
        name=name,
        eye=(float(center[0]), float(center[1] - 0.35 * diagonal), 1.40 * diagonal),
        lookat=(float(center[0]), float(center[1]), 0.10),
    )


def camera_views(environment_origins) -> tuple[CameraView, CameraView]:
    """Frame every environment, then the roughly 8x8 region nearest the grid center."""
    origins = _as_numpy(environment_origins)
    if origins.ndim != 2 or origins.shape[0] < 1 or origins.shape[1] < 2:
        raise ValueError("environment_origins must have shape [num_envs, >=2]")
    center = (origins[:, :2].min(axis=0) + origins[:, :2].max(axis=0)) / 2.0
    count = min(64, origins.shape[0])
    nearest = np.argsort(np.linalg.norm(origins[:, :2] - center, axis=1))[:count]
    return (
        _view_for_origins(f"full_{origins.shape[0]}_env_grid", origins),
        _view_for_origins(f"detail_{count}_env_region", origins[nearest]),
    )


def select_focus_region(origins: np.ndarray, scores: np.ndarray, count: int = 16) -> list[int]:
    """Select a compact region using measured progress, never synthesizing success."""
    if scores.shape != (len(origins),) or not np.isfinite(scores).all():
        raise ValueError("Require one finite score per environment")
    count = min(count, len(origins))
    if count < 1:
        raise ValueError("Focus count must be positive")
    # Enumerate complete 4x4 blocks; edge-centered nearest neighbors can form diamonds.
    xs, ys = (np.unique(origins[:, axis]) for axis in (0, 1))
    candidates = []
    if count == 16 and len(xs) >= 4 and len(ys) >= 4:
        for ix in range(len(xs) - 3):
            for iy in range(len(ys) - 3):
                ids = np.flatnonzero((origins[:, 0] >= xs[ix]) & (origins[:, 0] <= xs[ix + 3])
                                    & (origins[:, 1] >= ys[iy]) & (origins[:, 1] <= ys[iy + 3]))
                if len(ids) == count:
                    candidates.append(ids)
    if not candidates:
        for origin in origins:
            delta = np.abs(origins[:, :2] - origin[:2])
            candidates.append(np.lexsort((np.arange(len(origins)), delta.sum(axis=1), delta.max(axis=1)))[:count])
    best_ids, best_key = None, None
    grid_center = origins[:, :2].mean(axis=0)
    for ids in candidates:
        key = (float(scores[ids].sum()), float(scores[ids].max()),
               -float(np.linalg.norm(origins[ids, :2].mean(axis=0) - grid_center)))
        if best_key is None or key > best_key:
            best_ids, best_key = ids, key
    if best_ids is None:
        raise ValueError("No valid focus region")
    return [int(index) for index in best_ids]


def cinematic_view(name: str, origins: np.ndarray, overview: bool = False) -> CameraView:
    center = (origins[:, :2].min(axis=0) + origins[:, :2].max(axis=0)) / 2
    center = np.array([center[0], center[1] + 0.15, 0.15])
    diagonal_m = max(float(np.linalg.norm(np.ptp(origins[:, :2], axis=0))), 1.0)
    distance_m = max(4.0, diagonal_m * 1.70) if overview else max(1.5, diagonal_m * 0.75)
    direction = np.array([0.38, 0.70, 0.60])
    direction /= np.linalg.norm(direction)
    return CameraView(name, tuple(center + direction * distance_m), tuple(center))


def cinematic_interpolated_view(overview: CameraView, detail: CameraView,
                               policy_step: int, fps: int) -> tuple[CameraView, float]:
    """Four-second establishing shot, ten-second eased logarithmic dolly, slow orbit."""
    elapsed_s = (policy_step - 1) / fps
    t = np.clip((elapsed_s - 4.0) / 10.0, 0.0, 1.0)
    alpha = float(t * t * t * (10.0 + t * (-15.0 + 6.0 * t)))
    first_target, last_target = np.array(overview.lookat), np.array(detail.lookat)
    target = first_target + alpha * (last_target - first_target)
    first_offset = np.array(overview.eye) - first_target
    last_offset = np.array(detail.eye) - last_target
    first_distance, last_distance = np.linalg.norm(first_offset), np.linalg.norm(last_offset)
    distance_m = float(np.exp((1 - alpha) * np.log(first_distance) + alpha * np.log(last_distance)))
    direction = (1 - alpha) * first_offset / first_distance + alpha * last_offset / last_distance
    direction /= np.linalg.norm(direction)
    # A small camera-only arc adds parallax after arrival without chasing the policy.
    orbit_rad = 0.10 * np.clip((elapsed_s - 14.0) / 18.0, 0.0, 1.0)
    c, s = np.cos(orbit_rad), np.sin(orbit_rad)
    direction[:2] = (c * direction[0] - s * direction[1], s * direction[0] + c * direction[1])
    name = overview.name if alpha == 0 else detail.name if alpha == 1 else "cinematic_dolly"
    return CameraView(name, tuple(target + direction * distance_m), tuple(target)), alpha


def interpolated_view(
    overview: CameraView,
    detail: CameraView,
    policy_step: int,
    fps: int,
    overview_seconds: float = 5.0,
    transition_seconds: float = 2.0,
) -> tuple[CameraView, float]:
    """Hold the full grid for five simulation seconds, then smoothly zoom."""
    if policy_step < 1 or fps <= 0:
        raise ValueError("policy_step and fps must be positive")
    elapsed = (policy_step - 1) / fps
    if elapsed <= overview_seconds:
        alpha = 0.0
    else:
        alpha = min(1.0, (elapsed - overview_seconds) / transition_seconds)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)

    def blend(first, second):
        return tuple(float(a + alpha * (b - a)) for a, b in zip(first, second))

    name = overview.name if alpha == 0.0 else detail.name if alpha == 1.0 else "zoom_transition"
    return CameraView(name=name, eye=blend(overview.eye, detail.eye), lookat=blend(overview.lookat, detail.lookat)), alpha


def overlay_training_label(frame: np.ndarray, num_envs: int, iteration: int, focus: dict | None = None) -> np.ndarray:
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected a uint8 RGB frame")
    image = Image.fromarray(frame, mode="RGB").convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font_size = max(12, frame.shape[0] // 36)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    bar_height = min(frame.shape[0], max(54, 2 * font_size + 26))
    draw.rectangle((0, 0, image.width, bar_height), fill=(0, 0, 0, 185))
    draw.text((12, 7), f"LIVE PPO TRAINING | {num_envs} parallel envs | iter {iteration}", fill="white", font=font)
    description = "Baseline continuation | diagnostic capture, not a success evaluation"
    if focus is not None:
        description = (f"Selected {len(focus['environment_ids'])}-env view | "
                       f"lifts {focus['qualified_lift_seen_count']}, full successes {focus['full_task_success_seen_count']} "
                       "observed by 4s | not an evaluation")
    draw.text(
        (12, 13 + font_size), description,
        fill=(255, 205, 80, 255),
        font=font,
    )
    return np.asarray(Image.alpha_composite(image, layer).convert("RGB"))


class ParallelTrainingVideo:
    """Capture one post-step RGB frame for every policy step during PPO learning."""

    def __init__(
        self,
        output_dir: Path,
        width: int,
        height: int,
        fps: int,
        num_envs: int,
        num_steps_per_env: int,
        start_iteration: int,
        max_iterations: int,
        environment_origins,
        writer_factory: Callable[..., VideoWriter] = VideoWriter,
        provenance: dict | None = None,
        style: str = "overview",
    ):
        if output_dir.exists():
            raise FileExistsError(output_dir)
        if width < 64 or height < 64 or width % 2 or height % 2:
            raise ValueError("Video dimensions must be even and at least 64")
        if min(fps, num_envs, num_steps_per_env, max_iterations) <= 0:
            raise ValueError("Capture and training dimensions must be positive")
        output_dir.mkdir(parents=True)
        self.output_dir = output_dir
        self.video_path = output_dir / "parallel_training.mp4"
        self.metadata_path = output_dir / "frames.jsonl"
        self.manifest_path = output_dir / "manifest.json"
        self.width, self.height, self.fps = width, height, fps
        self.num_envs = num_envs
        self.num_steps_per_env = num_steps_per_env
        self.start_iteration = start_iteration
        self.provenance = provenance or {}
        if style not in ("overview", "cinematic"):
            raise ValueError("Unknown camera style")
        self.style = style
        self.expected_frames = num_steps_per_env * max_iterations
        origins = _as_numpy(environment_origins)
        if origins.shape[0] != num_envs:
            raise ValueError(f"Expected {num_envs} environment origins, got {origins.shape[0]}")
        self.overview, self.detail = camera_views(origins)
        self.origins = origins
        self.focus = None
        self.carry_steps = np.zeros(num_envs, dtype=np.int64)
        self.released_steps = np.zeros(num_envs, dtype=np.int64)
        self.picked_seen = np.zeros(num_envs, dtype=bool)
        self.success_seen = np.zeros(num_envs, dtype=bool)
        self.lens = None
        if style == "cinematic":
            self.overview = cinematic_view(f"oblique_full_{num_envs}_env_grid", origins, overview=True)
            self.detail = self.overview
        self.writer = writer_factory(self.video_path, width, height, fps)
        self.metadata = self.metadata_path.open("x", encoding="utf-8", newline="\n")
        self.frames = 0
        self.closed = False
        self._write_manifest("recording")

    def _manifest(self, status: str, error: str | None = None) -> dict:
        payload = {
            "schema": "so101_pick_rl.parallel_training_video.v1",
            "status": status,
            "purpose": "portfolio_evidence_of_live_ppo_training_not_expert_or_transition_dataset",
            "provenance": self.provenance,
            "camera_style": self.style,
            "focus_selection": self.focus,
            "lens": self.lens if self.style == "cinematic" else "runtime_default",
            "video_clock": "simulation_time_only; optimizer wall-clock pauses are omitted",
            "capture_timing": "post-policy-step after automatic partial resets; terminal states are not guaranteed",
            "fps": self.fps,
            "resolution": [self.width, self.height],
            "num_envs": self.num_envs,
            "start_iteration": self.start_iteration,
            "num_steps_per_env": self.num_steps_per_env,
            "expected_frames": self.expected_frames,
            "frames": self.frames,
            "simulation_duration_seconds": self.frames / self.fps,
            "camera_views": [self.overview.__dict__, self.detail.__dict__],
            "files": {},
        }
        if error:
            payload["error"] = error
        if status == "completed":
            payload["files"] = {
                "video": {"path": str(self.video_path), "sha256": sha256_file(self.video_path)},
                "frame_metadata": {"path": str(self.metadata_path), "sha256": sha256_file(self.metadata_path)},
            }
        return payload

    def _write_manifest(self, status: str, error: str | None = None):
        self.manifest_path.write_text(
            json.dumps(self._manifest(status, error), indent=2) + "\n", encoding="utf-8"
        )

    def warm_up(self, raw_env, render_count: int = 12):
        """Initialize the RGB annotator without advancing physics or recording frames."""
        camera = None
        if self.style == "cinematic":
            from pxr import Usd, UsdGeom
            camera = UsdGeom.Camera(raw_env.sim.stage.GetPrimAtPath(raw_env.cfg.viewer.cam_prim_path))
            # The viewport camera already has stronger session-layer lens opinions.
            with Usd.EditContext(raw_env.sim.stage, raw_env.sim.stage.GetSessionLayer()):
                camera.GetFocalLengthAttr().Set(35.0)
                camera.GetHorizontalApertureAttr().Set(36.0)
                camera.GetVerticalApertureAttr().Set(36.0 * self.height / self.width)
        raw_env.sim.set_camera_view(self.overview.eye, self.overview.lookat)
        raw_env.render()
        for _ in range(render_count):
            raw_env.sim.render()
        if self.style == "cinematic":
            assert camera is not None
            self.lens = {"focal_length_mm": camera.GetFocalLengthAttr().Get(),
                         "horizontal_aperture_mm": camera.GetHorizontalApertureAttr().Get(),
                         "vertical_aperture_mm": camera.GetVerticalApertureAttr().Get()}
            if abs(self.lens["focal_length_mm"] - 35.0) > 0.01:
                raise RuntimeError(f"Viewport overrode cinematic lens: {self.lens}")
            # Framing-only still: unchanged initial state, no policy or physics step.
            ids = select_focus_region(self.origins, np.zeros(self.num_envs))
            framing = cinematic_view("framing_only", self.origins[ids])
            raw_env.sim.set_camera_view(framing.eye, framing.lookat)
            for _ in range(4):
                raw_env.sim.render()
            Image.fromarray(raw_env.render()).save(self.output_dir / "framing_only_no_policy.png")
            raw_env.sim.set_camera_view(self.overview.eye, self.overview.lookat)
            for _ in range(4):
                raw_env.sim.render()
            self._write_manifest("recording")

    def capture_post_step(self, raw_env):
        if self.closed:
            raise RuntimeError("Capture is already closed")
        policy_step = self.frames + 1
        if self.style == "cinematic" and self.focus is None:
            state = raw_env.pick_place_state
            picked = _as_numpy(state.picked).astype(bool)
            carry = _as_numpy(state.carry_valid).astype(bool)
            released = _as_numpy(state.released).astype(bool)
            success = _as_numpy(raw_env.termination_manager.get_term("pick_place_success")).astype(bool)
            self.picked_seen |= picked
            self.success_seen |= success
            self.carry_steps += carry
            self.released_steps += released
            if policy_step >= 4 * self.fps:
                weights = FOCUS_SCORE_WEIGHTS
                scores = (weights["carry_step"] * self.carry_steps + weights["released_step"] * self.released_steps
                          + weights["qualified_lift_seen"] * self.picked_seen
                          + weights["full_success_seen"] * self.success_seen)
                ids = select_focus_region(self.origins, scores)
                hero = int(ids[int(np.argmax(scores[ids]))])
                self.detail = cinematic_view("selected_16_env_oblique_detail", self.origins[ids])
                self.focus = {
                    "selected_at_policy_step": policy_step, "environment_ids": ids,
                    "hero_environment_id": hero, "scores": scores[ids].tolist(),
                    "score_weights": weights,
                    "components": {"carry_steps": self.carry_steps[ids].tolist(),
                                   "released_steps": self.released_steps[ids].tolist(),
                                   "qualified_lift_seen": self.picked_seen[ids].tolist(),
                                   "full_success_seen": self.success_seen[ids].tolist()},
                    "qualified_lift_seen_count": int(self.picked_seen[ids].sum()),
                    "full_task_success_seen_count": int(self.success_seen[ids].sum()),
                    "criterion": "compact 16-env region maximizing weighted score sum (carry/release steps and qualified lift/full success history) in first 4 simulation seconds; selection bias, not evaluation",
                }
                self._write_manifest("recording")
        view, zoom_alpha = (cinematic_interpolated_view if self.style == "cinematic" else interpolated_view)(
            self.overview, self.detail, policy_step, self.fps)
        raw_env.sim.set_camera_view(view.eye, view.lookat)
        raw_env.sim.render()
        pixels = raw_env.render()
        if pixels is None:
            raise RuntimeError("Renderer returned no frame")
        pixels = np.asarray(pixels)
        if pixels.shape != (self.height, self.width, 3) or pixels.dtype != np.uint8:
            raise RuntimeError(f"Unexpected RGB frame: shape={pixels.shape}, dtype={pixels.dtype}")
        if not np.isfinite(pixels).all() or float(pixels.std()) < 1.0:
            raise RuntimeError("Renderer returned a blank or non-finite frame")
        iteration = self.start_iteration + (policy_step - 1) // self.num_steps_per_env
        self.writer.write(overlay_training_label(pixels, self.num_envs, iteration, self.focus))
        if self.style == "cinematic" and self.frames in (0, 14 * self.fps, 24 * self.fps):
            Image.fromarray(pixels).save(self.output_dir / f"preview_{self.frames:04d}.png")
        row = {
            "frame_index": self.frames,
            "policy_step": policy_step,
            "simulation_time_seconds": policy_step / self.fps,
            "global_iteration": iteration,
            "camera_view": view.name,
            "zoom_alpha": zoom_alpha,
            "camera_eye": view.eye,
            "camera_lookat": view.lookat,
            "num_envs_training": self.num_envs,
        }
        self.metadata.write(json.dumps(row, separators=(",", ":")) + "\n")
        self.metadata.flush()
        self.frames += 1

    def close(self, completed: bool, error: str | None = None):
        if self.closed:
            return
        self.closed = True
        close_error = None
        try:
            self.writer.close()
        except BaseException as exc:  # preserve encoder evidence in the manifest
            close_error = f"{type(exc).__name__}: {exc}"
        finally:
            self.metadata.close()
        status = "completed" if completed and self.frames == self.expected_frames and close_error is None else "failed"
        detail = error or close_error
        if completed and self.frames != self.expected_frames:
            detail = detail or f"Expected {self.expected_frames} frames, captured {self.frames}"
        self._write_manifest(status, detail)
        if close_error:
            raise RuntimeError(close_error)
        if completed and self.frames != self.expected_frames:
            raise RuntimeError(detail)

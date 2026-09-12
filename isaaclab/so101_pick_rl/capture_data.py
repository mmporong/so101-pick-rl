"""Aligned diagnostic trajectories; no simulator imports or training operations."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aligned_arrays(states: list[dict], transitions: list[dict], fps: int) -> dict:
    """Frame/state i precedes action i; terminal state is retained before reset."""
    if fps <= 0 or not transitions or len(states) != len(transitions) + 1:
        raise ValueError("Require positive fps and T+1 states for T>0 transitions")
    arrays = {key: np.stack([row[key] for row in states]) for key in states[0]}
    arrays.update({key: np.stack([row[key] for row in transitions]) for key in transitions[0]})
    if len(arrays) != len(states[0]) + len(transitions[0]):
        raise ValueError("State and transition field names must be distinct")
    for key, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite trajectory field: {key}")
    arrays["state_time_seconds"] = np.arange(len(states), dtype=np.float64) / fps
    arrays["transition_time_seconds"] = np.arange(len(transitions), dtype=np.float64) / fps
    return arrays


class VideoWriter:
    """Stream real RGB frames to a local encoder without retaining them in RAM."""

    def __init__(self, path: Path, width: int, height: int, fps: int):
        if path.exists():
            raise FileExistsError(path)
        encoder = shutil.which("ffmpeg")
        if encoder is None:
            raise RuntimeError("ffmpeg is required for MP4 capture")
        self.shape = (height, width, 3)
        self.frames = 0
        self.closed = False
        self.log = path.with_suffix(".encoder.log").open("xb")
        try:
            self.process = subprocess.Popen(
                [encoder, "-nostdin", "-n", "-loglevel", "error", "-f", "rawvideo",
                 "-pix_fmt", "rgb24", "-s:v", f"{width}x{height}", "-r", str(fps),
                 "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast",
                 "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except BaseException:
            self.log.close()
            raise

    def write(self, pixels):
        if self.closed or pixels.dtype != np.uint8 or pixels.shape != self.shape:
            raise ValueError("Expected open writer and uint8 RGB frame of configured size")
        self.process.stdin.write(np.ascontiguousarray(pixels).tobytes())
        self.frames += 1

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            pipe_error = None
            try:
                self.process.stdin.close()
            except OSError as exc:
                pipe_error = exc
            try:
                code = self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
                raise RuntimeError("Capture encoder did not finish") from None
            if code:
                raise RuntimeError(f"Capture encoder exited with {code}")
            if pipe_error is not None:
                raise RuntimeError(f"Capture encoder pipe failed: {pipe_error}") from pipe_error
        finally:
            self.log.close()

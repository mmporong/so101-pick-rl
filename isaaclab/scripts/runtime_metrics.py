"""Low-overhead resource sampling shared by Windows runtime scripts."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil


def query_gpu() -> dict[str, float | str]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,utilization.memory",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
    fields = [field.strip() for field in completed.stdout.splitlines()[0].split(",")]
    return {
        "name": fields[0],
        "driver_version": fields[1],
        "memory_total_mib": float(fields[2]),
        "memory_used_mib": float(fields[3]),
        "utilization_gpu_percent": float(fields[4]),
        "utilization_memory_percent": float(fields[5]),
    }


def enforce_resource_guard(
    max_gpu_util_percent: float,
    minimum_free_vram_mib: float,
    max_cpu_util_percent: float,
) -> dict[str, float | str]:
    sample = query_gpu()
    system_cpu_percent = psutil.cpu_percent(interval=0.25)
    free_vram = float(sample["memory_total_mib"]) - float(sample["memory_used_mib"])
    if float(sample["utilization_gpu_percent"]) > max_gpu_util_percent:
        raise RuntimeError(
            f"RESOURCE_GUARD: baseline GPU utilization {sample['utilization_gpu_percent']}% exceeds "
            f"{max_gpu_util_percent}%"
        )
    if free_vram < minimum_free_vram_mib:
        raise RuntimeError(
            f"RESOURCE_GUARD: free VRAM {free_vram:.0f} MiB is below {minimum_free_vram_mib:.0f} MiB"
        )
    if system_cpu_percent > max_cpu_util_percent:
        raise RuntimeError(
            f"RESOURCE_GUARD: baseline system CPU utilization {system_cpu_percent:.1f}% exceeds "
            f"{max_cpu_util_percent:.1f}%"
        )
    sample["memory_free_mib"] = free_vram
    sample["system_cpu_percent"] = system_cpu_percent
    return sample


@dataclass
class ResourceSampler:
    interval_seconds: float = 1.0
    samples: list[dict[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = psutil.Process(os.getpid())
        self.error_count = 0
        self.last_error: str | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        psutil.cpu_percent(interval=None)
        self._process.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._run, name="resource-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 2))
        if not self.samples:
            return {"sample_count": 0, "error_count": self.error_count, "last_error": self.last_error}
        return {
            "sample_count": len(self.samples),
            "error_count": self.error_count,
            "last_error": self.last_error,
            "peak_vram_mib": max(sample["memory_used_mib"] for sample in self.samples),
            "mean_gpu_utilization_percent": sum(sample["gpu_utilization_percent"] for sample in self.samples)
            / len(self.samples),
            "peak_gpu_utilization_percent": max(sample["gpu_utilization_percent"] for sample in self.samples),
            "mean_system_cpu_percent": sum(sample["system_cpu_percent"] for sample in self.samples) / len(self.samples),
            "peak_system_cpu_percent": max(sample["system_cpu_percent"] for sample in self.samples),
            "mean_process_cpu_percent": sum(sample["process_cpu_percent"] for sample in self.samples) / len(self.samples),
            "peak_process_cpu_percent": max(sample["process_cpu_percent"] for sample in self.samples),
            "peak_process_rss_mib": max(sample["process_rss_mib"] for sample in self.samples),
        }

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                gpu = query_gpu()
                self.samples.append(
                    {
                        "timestamp": time.time(),
                        "memory_used_mib": float(gpu["memory_used_mib"]),
                        "gpu_utilization_percent": float(gpu["utilization_gpu_percent"]),
                        "system_cpu_percent": psutil.cpu_percent(interval=None),
                        "process_cpu_percent": self._process.cpu_percent(interval=None),
                        "process_rss_mib": self._process.memory_info().rss / (1024 * 1024),
                    }
                )
            except (OSError, subprocess.SubprocessError, psutil.Error, ValueError) as exc:
                self.error_count += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
                continue


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)

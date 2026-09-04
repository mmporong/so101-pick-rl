#!/usr/bin/env python3
"""Collect the pinned Windows Isaac stack without launching the simulator."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from runtime_metrics import query_gpu, write_json


ISAAC_SIM_ROOT = Path(r"E:\IsaacSim\isaac-sim-4.5.0")
ISAAC_LAB_ROOT = Path.home() / "IsaacLab"


def run(*command: str, cwd: Path | None = None) -> str:
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True).stdout.rstrip()


def main() -> int:
    output = REPOSITORY_ROOT / "reports" / "windows" / "w0_environment.json"
    contract_output = run(sys.executable, str(REPOSITORY_ROOT / "scripts" / "validate_contract.py"))
    contract_sha = next(
        line.split("=", 1)[1] for line in contract_output.splitlines() if line.startswith("contract_sha256=")
    )
    repository_status = run("git", "status", "--porcelain", cwd=REPOSITORY_ROOT)
    windows = json.loads(
        run(
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber | ConvertTo-Json -Compress",
        )
    )
    payload = {
        "schema": "so101_pick_rl.windows_environment.v1",
        "status": "passed",
        "classification": "runtime_validated",
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "os": {
            "platform": platform.platform(),
            "caption": windows["Caption"],
            "version": windows["Version"],
            "build_number": windows["BuildNumber"],
            "windows_version": platform.win32_ver(),
            "processor": platform.processor(),
            "physical_cpu_cores": psutil.cpu_count(logical=False),
            "logical_cpu_cores": psutil.cpu_count(logical=True),
            "memory_total_mib": psutil.virtual_memory().total / (1024 * 1024),
        },
        "gpu": query_gpu(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "pytorch": {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "isaac_sim": {
            "root": str(ISAAC_SIM_ROOT),
            "version": (ISAAC_SIM_ROOT / "VERSION").read_text(encoding="utf-8").strip(),
        },
        "isaac_lab": {
            "root": str(ISAAC_LAB_ROOT),
            "commit": run("git", "rev-parse", "HEAD", cwd=ISAAC_LAB_ROOT),
            "describe": run("git", "describe", "--tags", "--always", "--dirty", cwd=ISAAC_LAB_ROOT),
            "status": run("git", "status", "--short", "--branch", cwd=ISAAC_LAB_ROOT),
        },
        "rsl_rl": {
            "version": importlib.metadata.version("rsl-rl-lib"),
        },
        "repository": {
            "commit": run("git", "rev-parse", "HEAD", cwd=REPOSITORY_ROOT),
            "branch": run("git", "branch", "--show-current", cwd=REPOSITORY_ROOT),
            "dirty": bool(repository_status),
            "dirty_paths": repository_status.splitlines(),
        },
        "contract": {
            "status": "passed",
            "sha256": contract_sha,
        },
    }
    write_json(output, payload)
    print(f"ENVIRONMENT_REPORT={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

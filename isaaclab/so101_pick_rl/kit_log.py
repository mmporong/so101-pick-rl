"""Bind and summarize Isaac Kit logs without importing the simulator or resource libraries."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def bind_kit_log(
    started_at_epoch: float,
    command_token: str,
    log_dir: Path | None = None,
) -> dict[str, Any]:
    """Bind this invocation to the Kit log whose command line contains a unique token."""
    isaac_root = os.environ.get("ISAAC_PATH")
    if log_dir is None:
        if not isaac_root:
            return {"path": None, "binding": None, "reason": "ISAAC_PATH is unset"}
        log_dir = Path(isaac_root) / "kit" / "logs" / "Kit" / "Isaac-Sim" / "4.5"
    candidates = [
        path for path in log_dir.glob("kit_*.log") if path.stat().st_mtime >= started_at_epoch - 2.0
    ]
    matching: list[Path] = []
    for path in candidates:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            header = stream.read(64 * 1024)
        if command_token.casefold() in header.casefold():
            matching.append(path)
    if len(matching) != 1:
        return {
            "path": None,
            "binding": None,
            "reason": f"expected one Kit log containing token {command_token!r}, found {len(matching)}",
        }
    return {
        "path": str(matching[0]),
        "binding": "command_line_token",
        "command_token": command_token,
        "process_id": os.getpid(),
    }


def summarize_kit_log(binding: dict[str, Any]) -> dict[str, Any]:
    """Summarize warning/error lines from an invocation-bound Kit log."""
    if not binding.get("path"):
        return {
            **binding,
            "warning_count": None,
            "error_count": None,
            "warning_samples": [],
            "error_samples": [],
        }
    log_path = Path(binding["path"])
    if not log_path.is_file():
        return {
            **binding,
            "warning_count": None,
            "error_count": None,
            "warning_samples": [],
            "error_samples": [],
            "reason": "bound Kit log no longer exists",
        }
    warning_lines: list[str] = []
    error_lines: list[str] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        normalized = line.strip()
        if "[Warning]" in normalized:
            warning_lines.append(normalized)
        if "[Error]" in normalized or "[Fatal]" in normalized:
            error_lines.append(normalized)
    return {
        **binding,
        "warning_count": len(warning_lines),
        "error_count": len(error_lines),
        "warning_samples": list(dict.fromkeys(warning_lines))[:20],
        "error_samples": list(dict.fromkeys(error_lines))[:20],
    }

"""Diagnostic first-step controller for demonstration-reset rollouts."""

from __future__ import annotations

import numpy as np


RESET_BOOTSTRAP_MODES = ("none", "home-open-one-step")
HOME_OPEN_COMMAND_RAD = (0.0, 0.0, 0.0, 0.0, 0.0, 1.35)


def validate_reset_bootstrap(mode: str, controller: str = "bc") -> None:
    """Fail closed on unknown modes and source-controller composition."""
    if mode not in RESET_BOOTSTRAP_MODES:
        raise ValueError(f"unknown reset bootstrap mode: {mode!r}")
    if controller not in ("bc", "source"):
        raise ValueError(f"unknown controller: {controller!r}")
    if controller == "source" and mode != "none":
        raise ValueError("reset bootstrap is only defined for the BC controller")


def reset_bootstrap_metadata(mode: str, controller: str = "bc") -> dict:
    """Return explicit report provenance for the derived controller."""
    validate_reset_bootstrap(mode, controller)
    enabled = mode == "home-open-one-step"
    return {
        "mode": mode,
        "command_rad": list(HOME_OPEN_COMMAND_RAD) if enabled else None,
        "steps": 1 if enabled else 0,
        "policy_weights_unchanged": True,
        "derived_controller": enabled,
    }


def normalized_home_open_action(lower, upper, batch_size: int, *, device, dtype):
    """Build the fixed in-range absolute target as a normalized Torch batch."""
    import torch

    lower_np = np.asarray(lower, dtype=np.float64)
    upper_np = np.asarray(upper, dtype=np.float64)
    command = np.asarray(HOME_OPEN_COMMAND_RAD, dtype=np.float64)
    if (lower_np.shape != (6,) or upper_np.shape != (6,)
            or not np.isfinite(lower_np).all() or not np.isfinite(upper_np).all()
            or not (lower_np < upper_np).all()):
        raise ValueError("joint limits must be finite ordered six-joint bounds")
    if ((command < lower_np) | (command > upper_np)).any():
        raise ValueError("home-open bootstrap command is outside contracted joint limits")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not getattr(dtype, "is_floating_point", False):
        raise ValueError("bootstrap action dtype must be floating point")
    normalized = 2.0 * (command - lower_np) / (upper_np - lower_np) - 1.0
    if not np.isfinite(normalized).all() or (np.abs(normalized) > 1.0).any():
        raise ValueError("invalid normalized bootstrap command")
    row = torch.as_tensor(normalized, device=device, dtype=dtype)
    return row.unsqueeze(0).expand(batch_size, -1).clone()


def controller_action(
    mode: str,
    step: int,
    observation,
    mean,
    std,
    policy,
    lower,
    upper,
):
    """Use the fixed command at step zero, then delegate unchanged to the policy."""
    validate_reset_bootstrap(mode)
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer")
    if observation.ndim != 2 or observation.shape[1] != 34:
        raise ValueError("observation must have shape (N, 34)")
    if mode == "home-open-one-step" and step == 0:
        return normalized_home_open_action(
            lower,
            upper,
            len(observation),
            device=observation.device,
            dtype=observation.dtype,
        )
    return policy((observation - mean) / std)

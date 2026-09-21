"""Residual behavior-cloning policy for the DemoBox task.

The policy starts from the previous absolute joint target contained in the
normalized observation and learns a bounded normalized-action correction.
"""

from __future__ import annotations

from typing import Any


OBSERVATION_DIMENSION = 34
ACTION_DIMENSION = 6
PREVIOUS_TARGET_SLICE = slice(12, 18)


def build_residual_policy(
    mean: Any,
    std: Any,
    lower: Any,
    upper: Any,
    scale: float = 1.0,
):
    """Build a zero-initialized residual policy over the previous target.

    Inputs to ``forward`` are normalized 34D observations.  The previous
    target is recovered in radians, mapped through the configured joint
    bounds to normalized action space, then corrected by ``scale * MLP(x)``.
    """
    import torch

    mean_tensor = torch.as_tensor(mean, dtype=torch.float32)
    std_tensor = torch.as_tensor(std, dtype=torch.float32)
    lower_tensor = torch.as_tensor(lower, dtype=torch.float32)
    upper_tensor = torch.as_tensor(upper, dtype=torch.float32)

    if mean_tensor.shape != (OBSERVATION_DIMENSION,):
        raise ValueError("mean must be a 34D vector")
    if std_tensor.shape != (OBSERVATION_DIMENSION,):
        raise ValueError("std must be a 34D vector")
    if lower_tensor.shape != (ACTION_DIMENSION,) or upper_tensor.shape != (ACTION_DIMENSION,):
        raise ValueError("lower and upper must be 6D vectors")
    if not torch.isfinite(mean_tensor).all() or not torch.isfinite(std_tensor).all():
        raise ValueError("mean and std must be finite")
    if not torch.all(std_tensor > 0):
        raise ValueError("std must be strictly positive")
    if not torch.isfinite(lower_tensor).all() or not torch.isfinite(upper_tensor).all():
        raise ValueError("joint bounds must be finite")
    if not torch.all(lower_tensor < upper_tensor):
        raise ValueError("joint bounds must be strictly ordered")
    if isinstance(scale, bool):
        raise ValueError("scale must be a finite positive number")
    try:
        scale_value = float(scale)
    except (TypeError, ValueError) as exc:
        raise ValueError("scale must be a finite positive number") from exc
    if not torch.isfinite(torch.tensor(scale_value)) or scale_value <= 0:
        raise ValueError("scale must be a finite positive number")

    class ResidualPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("observation_mean", mean_tensor.clone())
            self.register_buffer("observation_std", std_tensor.clone())
            self.register_buffer("action_lower", lower_tensor.clone())
            self.register_buffer("action_upper", upper_tensor.clone())
            self.register_buffer("residual_scale", torch.tensor(scale_value, dtype=torch.float32))
            self.residual = torch.nn.Sequential(
                torch.nn.Linear(OBSERVATION_DIMENSION, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, ACTION_DIMENSION),
            )
            torch.nn.init.zeros_(self.residual[-1].weight)
            torch.nn.init.zeros_(self.residual[-1].bias)

        def forward(self, observations):
            if observations.ndim < 1 or observations.shape[-1] != OBSERVATION_DIMENSION:
                raise ValueError("observations must have final dimension 34")
            if not torch.is_floating_point(observations):
                raise ValueError("observations must be floating point")
            if not torch.isfinite(observations).all():
                raise ValueError("observations must be finite")

            previous_normalized = observations[..., PREVIOUS_TARGET_SLICE]
            previous_raw = (
                previous_normalized * self.observation_std[PREVIOUS_TARGET_SLICE]
                + self.observation_mean[PREVIOUS_TARGET_SLICE]
            )
            base = 2.0 * (previous_raw - self.action_lower) / (
                self.action_upper - self.action_lower
            ) - 1.0
            correction = self.residual_scale * self.residual(observations)
            return torch.clamp(base + correction, -1.0, 1.0)

    return ResidualPolicy()

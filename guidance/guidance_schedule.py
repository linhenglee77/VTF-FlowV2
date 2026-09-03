"""Configurable scalar schedules for inference-time Flow guidance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


GuidanceScheduleName = Literal["constant", "early-strong", "late-strong"]


@dataclass(frozen=True)
class GuidanceScheduleConfig:
    """Guidance schedule name and positive power-law exponent."""

    name: GuidanceScheduleName = "late-strong"
    gamma: float = 1.0

    def __post_init__(self) -> None:
        if self.name not in {"constant", "early-strong", "late-strong"}:
            raise ValueError(
                "name must be 'constant', 'early-strong', or 'late-strong'"
            )
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive")


def guidance_weight(
    time: torch.Tensor | float,
    strength: float,
    schedule: GuidanceScheduleConfig,
) -> torch.Tensor:
    """Return ``eta(t)`` while preserving tensor device and dtype."""

    if strength < 0.0:
        raise ValueError("guidance strength must be non-negative")
    value = torch.as_tensor(time)
    if not torch.isfinite(value).all() or bool(((value < 0.0) | (value > 1.0)).any()):
        raise ValueError("integration time must lie in [0,1]")
    if schedule.name == "constant":
        response = torch.ones_like(value)
    elif schedule.name == "early-strong":
        response = (1.0 - value).pow(schedule.gamma)
    else:
        response = value.pow(schedule.gamma)
    return response * strength


__all__ = ["GuidanceScheduleConfig", "GuidanceScheduleName", "guidance_weight"]

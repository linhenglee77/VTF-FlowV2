"""Constant-velocity trajectory baseline using the shared planner interface."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from TerraFlow.interfaces import BasePlanner, SceneBatch, TrajectoryBatch


@dataclass(frozen=True)
class ConstantVelocityConfig:
    """Hyperparameters for ego-history velocity estimation and extrapolation."""

    horizon: int = 30
    planning_dt_s: float = 0.5
    history_dt_s: float = 0.1
    velocity_window: int = 5
    trajectory_dimensions: int = 3
    stationary_fallback: bool = True

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.planning_dt_s <= 0.0 or self.history_dt_s <= 0.0:
            raise ValueError("planning_dt_s and history_dt_s must be positive")
        if self.velocity_window <= 0:
            raise ValueError("velocity_window must be positive")
        if self.trajectory_dimensions < 2:
            raise ValueError("trajectory_dimensions must be at least two")


class ConstantVelocityPlanner(BasePlanner):
    """Estimate recent ego velocity and extrapolate metric local waypoints.

    ``ego_history`` is interpreted as ``[B,T,F]`` with position in its first
    ``D`` features and monotonically ordered from oldest to newest. The output
    is relative to the current ego pose, so time zero is always the origin and
    the first returned point is at ``planning_dt_s``. A single history state
    cannot identify velocity; the configurable fallback makes that case an
    explicit stationary model, which is useful for legacy caches that only
    retain the current pose.
    """

    def __init__(self, config: ConstantVelocityConfig | None = None) -> None:
        super().__init__()
        self.config = config or ConstantVelocityConfig()

    def _estimate_velocity(self, history: torch.Tensor, dimensions: int) -> torch.Tensor:
        if history.ndim != 3:
            raise ValueError("ego_history must have shape [B,T,F]")
        if history.shape[-1] < dimensions:
            raise ValueError(
                f"ego_history has {history.shape[-1]} features but {dimensions} are required"
            )
        positions = history[..., :dimensions]
        if not positions.is_floating_point() or not torch.isfinite(positions).all():
            raise ValueError("ego_history positions must be finite floating-point values")
        if positions.shape[1] < 2:
            if not self.config.stationary_fallback:
                raise ValueError(
                    "at least two ego-history positions are required to estimate velocity"
                )
            return torch.zeros(
                positions.shape[0], dimensions, dtype=positions.dtype, device=positions.device
            )
        deltas = torch.diff(positions, dim=1) / self.config.history_dt_s
        window = min(self.config.velocity_window, deltas.shape[1])
        return deltas[:, -window:].mean(dim=1)

    def forward(self, scene: SceneBatch) -> TrajectoryBatch:
        """Return one constant-velocity candidate shaped ``[B,1,H,D]``."""

        scene = scene.as_batch()
        dimensions = min(self.config.trajectory_dimensions, scene.gt_future.shape[-1])
        velocity = self._estimate_velocity(scene.ego_history, dimensions)
        times = torch.arange(
            1,
            self.config.horizon + 1,
            dtype=velocity.dtype,
            device=velocity.device,
        ) * self.config.planning_dt_s
        trajectories = velocity[:, None, :] * times[None, :, None]
        return TrajectoryBatch(trajectories=trajectories.unsqueeze(1))


# Concise alias retained for config files and experiment tables.
ConstantVelocity = ConstantVelocityPlanner


__all__ = ["ConstantVelocity", "ConstantVelocityConfig", "ConstantVelocityPlanner"]

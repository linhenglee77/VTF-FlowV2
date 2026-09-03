"""Deterministic neural trajectory-regression baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

from TerraFlow.interfaces import BasePlanner, SceneBatch, TrajectoryBatch
from TerraFlow.models.scene_encoder import RegressionSceneEncoder


LossName = Literal["l1", "smooth_l1"]


@dataclass(frozen=True)
class RegressionPlannerConfig:
    """Architecture and trajectory-shape hyperparameters."""

    horizon: int = 30
    trajectory_dim: int = 3
    feature_dim: int = 128
    decoder_hidden_dim: int = 256
    history_features: int = 3
    terrain_channels: int = 3
    dropout: float = 0.0
    metric_scales: tuple[float, float, float] = (24.0, 12.0, 3.0)
    goal_anchor: bool = True

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.trajectory_dim != 3:
            raise ValueError("version-one regression planner requires trajectory_dim=3")
        if self.feature_dim < 32 or self.decoder_hidden_dim < 32:
            raise ValueError("feature dimensions must be at least 32")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")


class RegressionPlanner(BasePlanner):
    """Encode a scene once and decode one deterministic metric xyz trajectory.

    :meth:`predict_trajectory` exposes the requested raw ``[B,H,3]`` tensor.
    :meth:`forward` preserves VTF-Flow's public planner contract by wrapping it
    as a one-candidate :class:`TrajectoryBatch` of shape ``[B,1,H,3]``.
    """

    def __init__(self, config: RegressionPlannerConfig | None = None) -> None:
        super().__init__()
        self.config = config or RegressionPlannerConfig()
        self.encoder = RegressionSceneEncoder(
            feature_dim=self.config.feature_dim,
            history_features=self.config.history_features,
            terrain_channels=self.config.terrain_channels,
            metric_scales=self.config.metric_scales,
            dropout=self.config.dropout,
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.config.feature_dim, self.config.decoder_hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.decoder_hidden_dim, self.config.decoder_hidden_dim),
            nn.SiLU(),
            nn.Linear(
                self.config.decoder_hidden_dim,
                self.config.horizon * self.config.trajectory_dim,
            ),
        )
        # The initial prediction is a straight interpolation to the supplied
        # goal; learning then models a fully unconstrained residual trajectory.
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def predict_trajectory(self, scene: SceneBatch) -> torch.Tensor:
        """Predict deterministic ego-centric xyz with shape ``[B,H,3]``."""

        scene = scene.as_batch()
        feature = self.encoder(scene.ego_history, scene.goal, scene.terrain_map)
        residual = self.decoder(feature).reshape(
            scene.batch_size, self.config.horizon, self.config.trajectory_dim
        )
        if not self.config.goal_anchor:
            return residual
        alpha = torch.linspace(
            1.0 / self.config.horizon,
            1.0,
            self.config.horizon,
            device=residual.device,
            dtype=residual.dtype,
        ).view(1, self.config.horizon, 1)
        return residual + alpha * scene.goal[:, None, :3]

    def forward(self, scene: SceneBatch) -> TrajectoryBatch:
        """Return the deterministic trajectory under the shared planner API."""

        return TrajectoryBatch(trajectories=self.predict_trajectory(scene).unsqueeze(1))

    def trajectory_loss(
        self,
        scene: SceneBatch,
        loss_name: LossName = "smooth_l1",
        beta: float = 1.0,
    ) -> torch.Tensor:
        """Compute mean metric-coordinate L1 or Smooth-L1 trajectory loss."""

        scene = scene.as_batch()
        prediction = self.predict_trajectory(scene)
        if prediction.shape != scene.gt_future[..., :3].shape:
            raise ValueError(
                "prediction and gt_future shapes differ; check configured horizon"
            )
        if loss_name == "l1":
            return F.l1_loss(prediction, scene.gt_future[..., :3])
        if loss_name == "smooth_l1":
            if beta <= 0.0:
                raise ValueError("Smooth-L1 beta must be positive")
            return F.smooth_l1_loss(prediction, scene.gt_future[..., :3], beta=beta)
        raise ValueError(f"unsupported trajectory loss: {loss_name}")


DeterministicRegressionPlanner = RegressionPlanner


__all__ = [
    "DeterministicRegressionPlanner",
    "LossName",
    "RegressionPlanner",
    "RegressionPlannerConfig",
]

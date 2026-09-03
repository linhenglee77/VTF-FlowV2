"""Lightweight scene encoder shared by trajectory planners."""

from __future__ import annotations

import torch
from torch import nn


class SceneEncoder(nn.Module):
    """Encode a three-channel terrain BEV and normalized local goal."""

    def __init__(self, feature_dim: int = 192):
        super().__init__()
        self.bev = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2),
            nn.GroupNorm(6, 24),
            nn.SiLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1),
            nn.GroupNorm(12, 96),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(96 * 4 * 4, feature_dim),
            nn.SiLU(),
        )
        self.goal = nn.Sequential(
            nn.Linear(3, feature_dim), nn.SiLU(), nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, terrain_map: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.bev(terrain_map) + self.goal(goal)


class RegressionSceneEncoder(nn.Module):
    """Lightweight deterministic encoder for history, goal, and terrain BEV.

    Ego history may have any temporal length and is represented by the mean and
    last per-state MLP embeddings. Metric position-like inputs are normalized by
    ``metric_scales`` before encoding. This class is intentionally separate
    from :class:`SceneEncoder` so existing Flow Matching checkpoints retain an
    unchanged parameter layout.
    """

    def __init__(
        self,
        feature_dim: int = 128,
        history_features: int = 3,
        terrain_channels: int = 3,
        metric_scales: tuple[float, float, float] = (24.0, 12.0, 3.0),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if feature_dim < 32:
            raise ValueError("feature_dim must be at least 32")
        if history_features < 3:
            raise ValueError("history_features must include at least xyz")
        if terrain_channels <= 0:
            raise ValueError("terrain_channels must be positive")
        if len(metric_scales) != 3 or min(metric_scales) <= 0.0:
            raise ValueError("metric_scales must contain three positive values")
        self.history_features = history_features
        self.terrain_channels = terrain_channels
        self.register_buffer(
            "metric_scales", torch.tensor(metric_scales, dtype=torch.float32)
        )
        self.terrain_encoder = nn.Sequential(
            nn.Conv2d(terrain_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(48 * 4 * 4, feature_dim),
            nn.SiLU(),
        )
        history_dim = max(feature_dim // 2, 32)
        self.history_state = nn.Sequential(
            nn.Linear(history_features, history_dim),
            nn.SiLU(),
            nn.Linear(history_dim, history_dim),
            nn.SiLU(),
        )
        self.history_fusion = nn.Sequential(
            nn.Linear(history_dim * 2, history_dim), nn.SiLU()
        )
        self.goal_encoder = nn.Sequential(
            nn.Linear(3, history_dim), nn.SiLU(), nn.Linear(history_dim, history_dim), nn.SiLU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim + history_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
        )

    def _normalize_history(self, ego_history: torch.Tensor) -> torch.Tensor:
        if ego_history.ndim != 3:
            raise ValueError("ego_history must have shape [B,T,F]")
        if ego_history.shape[1] < 1 or ego_history.shape[-1] < self.history_features:
            raise ValueError(
                f"ego_history must contain at least one state with {self.history_features} features"
            )
        history = ego_history[..., : self.history_features].clone()
        history[..., :3] = history[..., :3] / self.metric_scales.to(history)
        return history

    def forward(
        self,
        ego_history: torch.Tensor,
        goal: torch.Tensor,
        terrain_map: torch.Tensor,
    ) -> torch.Tensor:
        """Return one fused scene feature vector per batch element."""

        if goal.ndim != 2 or goal.shape[-1] < 3:
            raise ValueError("goal must have shape [B,D] with D >= 3")
        if terrain_map.ndim != 4 or terrain_map.shape[1] != self.terrain_channels:
            raise ValueError(
                f"terrain_map must have shape [B,{self.terrain_channels},H,W]"
            )
        if not all(
            value.shape[0] == goal.shape[0] for value in (ego_history, terrain_map)
        ):
            raise ValueError("history, goal, and terrain batch sizes must match")
        if not all(
            torch.isfinite(value).all() for value in (ego_history, goal, terrain_map)
        ):
            raise ValueError("scene encoder inputs must be finite")
        state_features = self.history_state(self._normalize_history(ego_history))
        history_feature = self.history_fusion(
            torch.cat((state_features.mean(dim=1), state_features[:, -1]), dim=-1)
        )
        normalized_goal = goal[:, :3] / self.metric_scales.to(goal)
        goal_feature = self.goal_encoder(normalized_goal)
        terrain_feature = self.terrain_encoder(terrain_map)
        return self.fusion(
            torch.cat((terrain_feature, history_feature, goal_feature), dim=-1)
        )


__all__ = ["RegressionSceneEncoder", "SceneEncoder"]

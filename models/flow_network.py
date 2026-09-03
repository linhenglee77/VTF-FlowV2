"""Minimal conditional trajectory Flow Matching velocity network."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from TerraFlow.models.scene_encoder import RegressionSceneEncoder


def sinusoidal_time_embedding(time: torch.Tensor, dimension: int) -> torch.Tensor:
    """Encode scalar times ``[B]`` with deterministic sinusoidal features."""

    if time.ndim != 1:
        raise ValueError("time must have shape [B]")
    half = dimension // 2
    frequency = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=time.device, dtype=time.dtype)
        / max(half - 1, 1)
    )
    angles = time[:, None] * frequency[None]
    embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
    return F.pad(embedding, (0, dimension - embedding.shape[-1]))


def linear_flow_matching_sample(
    clean: torch.Tensor,
    base: torch.Tensor | None = None,
    time: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct the exact linear conditional Flow Matching training tuple.

    Returns ``(x_t, u_t, x0, t)`` where ``x_t=(1-t)x0+t*x1`` and
    ``u_t=x1-x0``. ``clean`` is the metric GT trajectory ``x1 [B,H,3]``.
    """

    if clean.ndim != 3 or clean.shape[-1] != 3:
        raise ValueError("clean trajectory must have shape [B,H,3]")
    if not clean.is_floating_point() or not torch.isfinite(clean).all():
        raise ValueError("clean trajectory must be finite floating point")
    if base is None:
        base = torch.randn_like(clean)
    if base.shape != clean.shape or not torch.isfinite(base).all():
        raise ValueError("base must be finite and have the same shape as clean")
    if time is None:
        time = torch.rand(clean.shape[0], dtype=clean.dtype, device=clean.device)
    if time.shape != (clean.shape[0],) or not torch.isfinite(time).all():
        raise ValueError("time must be finite and have shape [B]")
    if bool(((time < 0.0) | (time > 1.0)).any()):
        raise ValueError("time values must lie in [0,1]")
    weight = time[:, None, None]
    state = (1.0 - weight) * base + weight * clean
    target_velocity = clean - base
    return state, target_velocity, base, time


def estimate_clean_trajectory(
    state: torch.Tensor,
    time: torch.Tensor,
    predicted_velocity: torch.Tensor,
) -> torch.Tensor:
    """Decode a clean endpoint estimate from a linear Flow Matching state.

    For ``x_t=(1-t)x0+t*x1`` and a velocity estimate ``v_theta``, the
    mathematically consistent one-step endpoint estimate is
    ``x1_hat=x_t+(1-t)*v_theta``.  This operation intentionally remains in the
    autograd graph so endpoint losses can train the velocity field without
    changing the Flow Matching target or objective.
    """

    if state.ndim != 3 or state.shape[-1] != 3:
        raise ValueError("state must have shape [B,H,3]")
    if predicted_velocity.shape != state.shape:
        raise ValueError("predicted_velocity must have the same shape as state")
    if time.shape != (state.shape[0],):
        raise ValueError("time must have shape [B]")
    if not torch.isfinite(state).all() or not torch.isfinite(predicted_velocity).all():
        raise ValueError("state and predicted_velocity must be finite")
    return state + (1.0 - time[:, None, None]) * predicted_velocity


class ConditionalTrajectoryFlow(nn.Module):
    """Predict metric trajectory velocity ``v_theta(x_t,t,condition)``.

    The condition reuses the deterministic regression planner's history, goal,
    and BEV encoder concept. A compact trajectory-level MLP is intentional for
    this first Flow Matching baseline.
    """

    def __init__(
        self,
        trajectory_points: int = 30,
        hidden_dim: int = 256,
        layers: int = 3,
        dropout: float = 0.0,
        history_features: int = 3,
        terrain_channels: int = 3,
        metric_scales: tuple[float, float, float] = (24.0, 12.0, 3.0),
        minimum_remaining_time: float = 0.05,
    ) -> None:
        super().__init__()
        if trajectory_points <= 0:
            raise ValueError("trajectory_points must be positive")
        if hidden_dim < 32:
            raise ValueError("hidden_dim must be at least 32")
        if layers < 1:
            raise ValueError("layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if not 0.0 < minimum_remaining_time <= 0.25:
            raise ValueError("minimum_remaining_time must be in (0,0.25]")
        self.trajectory_points = trajectory_points
        self.hidden_dim = hidden_dim
        self.minimum_remaining_time = minimum_remaining_time
        self.scene_encoder = RegressionSceneEncoder(
            feature_dim=hidden_dim,
            history_features=history_features,
            terrain_channels=terrain_channels,
            metric_scales=metric_scales,
            dropout=dropout,
        )
        self.condition_dim = hidden_dim + 3
        self.condition_projection = nn.Sequential(
            nn.Linear(self.condition_dim, hidden_dim), nn.SiLU()
        )
        self.state_projection = nn.Sequential(
            nn.Linear(trajectory_points * 3, hidden_dim), nn.SiLU()
        )
        self.time_projection = nn.Sequential(
            nn.Linear(64, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        blocks: list[nn.Module] = []
        for _ in range(layers):
            blocks.extend(
                (
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                )
            )
        self.velocity_mlp = nn.Sequential(*blocks)
        self.output = nn.Linear(hidden_dim, trajectory_points * 3)
        self.clean_residual = nn.Linear(hidden_dim, trajectory_points * 3)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.clean_residual.weight)
        nn.init.zeros_(self.clean_residual.bias)
        alpha = torch.linspace(
            1.0 / trajectory_points, 1.0, trajectory_points
        ).view(1, trajectory_points, 1)
        self.register_buffer("goal_alpha", alpha)

    def encode_condition(
        self,
        ego_history: torch.Tensor,
        goal: torch.Tensor,
        terrain_map: torch.Tensor,
    ) -> torch.Tensor:
        """Encode scene context once for reuse across ODE steps and samples."""

        scene_feature = self.scene_encoder(ego_history, goal, terrain_map)
        # Retaining metric goal coordinates lets the velocity field start from
        # a useful straight-to-goal transport without post-hoc endpoint edits.
        return torch.cat((scene_feature, goal[:, :3]), dim=-1)

    def forward(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity with the same ``[B,H,3]`` shape as ``state``."""

        if state.ndim != 3 or state.shape[1:] != (self.trajectory_points, 3):
            raise ValueError(
                f"state must have shape [B,{self.trajectory_points},3]"
            )
        if time.shape != (state.shape[0],):
            raise ValueError("time must have shape [B]")
        if condition.shape != (state.shape[0], self.condition_dim):
            raise ValueError(f"condition must have shape [B,{self.condition_dim}]")
        condition_feature = self.condition_projection(condition)
        hidden = (
            self.state_projection(state.flatten(start_dim=1))
            + self.time_projection(sinusoidal_time_embedding(time, 64))
            + condition_feature
        )
        learned_clean = (
            self.goal_alpha.to(state) * condition[:, None, -3:]
            + self.clean_residual(condition_feature).reshape(
                state.shape[0], self.trajectory_points, 3
            )
        )
        # Clipping only regularizes the velocity parameterization near t=1;
        # x_t, target u_t, and the unweighted MSE objective remain exact.
        remaining_time = (1.0 - time).clamp_min(
            self.minimum_remaining_time
        )[:, None, None]
        linear_transport = (learned_clean - state) / remaining_time
        correction = self.output(self.velocity_mlp(hidden)).reshape(
            state.shape[0], self.trajectory_points, 3
        )
        return linear_transport + correction

    def flow_matching_loss(
        self,
        clean: torch.Tensor,
        ego_history: torch.Tensor,
        goal: torch.Tensor,
        terrain_map: torch.Tensor,
        *,
        base: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute exactly ``mean(||v_theta(x_t,t,c) - (x1-x0)||^2)``."""

        state, target_velocity, sampled_base, sampled_time = linear_flow_matching_sample(
            clean, base=base, time=time
        )
        condition = self.encode_condition(ego_history, goal, terrain_map)
        predicted_velocity = self(state, sampled_time, condition)
        loss = F.mse_loss(predicted_velocity, target_velocity)
        return loss, {
            "flow_matching_loss": loss.detach(),
            "predicted_velocity": predicted_velocity.detach(),
            "target_velocity": target_velocity.detach(),
            "x_t": state.detach(),
            "x0": sampled_base.detach(),
            "t": sampled_time.detach(),
        }

    def flow_matching_training_terms(
        self,
        clean: torch.Tensor,
        ego_history: torch.Tensor,
        goal: torch.Tensor,
        terrain_map: torch.Tensor,
        *,
        base: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return differentiable FM terms for optional endpoint regularizers.

        ``flow_matching_loss`` remains exactly the same unweighted mean-square
        velocity error.  This companion API only exposes its intermediates
        without detaching them, including the decoded ``estimated_clean``.
        """

        state, target_velocity, sampled_base, sampled_time = linear_flow_matching_sample(
            clean, base=base, time=time
        )
        condition = self.encode_condition(ego_history, goal, terrain_map)
        predicted_velocity = self(state, sampled_time, condition)
        loss = F.mse_loss(predicted_velocity, target_velocity)
        return {
            "flow_matching_loss": loss,
            "predicted_velocity": predicted_velocity,
            "target_velocity": target_velocity,
            "x_t": state,
            "x0": sampled_base,
            "t": sampled_time,
            "estimated_clean": estimate_clean_trajectory(
                state, sampled_time, predicted_velocity
            ),
        }


TrajectoryFlowNetwork = ConditionalTrajectoryFlow


__all__ = [
    "ConditionalTrajectoryFlow",
    "TrajectoryFlowNetwork",
    "linear_flow_matching_sample",
    "estimate_clean_trajectory",
    "sinusoidal_time_embedding",
]

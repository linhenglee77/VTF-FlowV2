"""Frozen Transformer Flow architecture used by the five 4201--4205 runs."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from TerraFlow.models.scene_encoder import SceneEncoder


def time_embedding(time: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    frequency = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=time.device, dtype=time.dtype)
        / max(half - 1, 1)
    )
    angle = time[:, None] * frequency[None]
    result = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)
    return F.pad(result, (0, dimension - result.shape[-1]))


class LegacyConditionalTrajectoryFlow(nn.Module):
    """Checkpoint-stable point-token Transformer velocity field."""

    def __init__(self, trajectory_points=30, hidden_dim=192, layers=4, dropout=0.05):
        super().__init__()
        if hidden_dim % 8:
            raise ValueError("hidden_dim must be divisible by 8")
        self.trajectory_points = trajectory_points
        self.hidden_dim = hidden_dim
        self.scene_encoder = SceneEncoder(hidden_dim)
        self.input_projection = nn.Linear(3, hidden_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(64, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.position = nn.Parameter(torch.zeros(1, trajectory_points, hidden_dim))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 3,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=layers)
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 3))

    def encode_condition(self, terrain_map: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.scene_encoder(terrain_map, goal)

    def forward(self, state, time, terrain_map=None, goal=None, condition=None):
        if condition is None:
            if terrain_map is None or goal is None:
                raise ValueError("terrain_map and goal are required when condition is absent")
            condition = self.encode_condition(terrain_map, goal)
        time_feature = self.time_projection(time_embedding(time, 64))
        hidden = (
            self.input_projection(state) + self.position
            + condition[:, None, :] + time_feature[:, None, :]
        )
        return self.output(self.temporal(hidden))

    def flow_matching_loss(self, clean, terrain_map, goal):
        batch = len(clean)
        base = torch.randn_like(clean)
        time = torch.rand(batch, device=clean.device, dtype=clean.dtype)
        state = (1.0 - time[:, None, None]) * base + time[:, None, None] * clean
        target_velocity = clean - base
        predicted_velocity = self(state, time, terrain_map, goal)
        loss = F.mse_loss(predicted_velocity, target_velocity)
        endpoint_loss = F.smooth_l1_loss(
            state[:, -1] + (1.0 - time[:, None]) * predicted_velocity[:, -1],
            clean[:, -1],
        )
        return loss + 0.05 * endpoint_loss, {
            "flow_loss": loss.detach(), "endpoint_loss": endpoint_loss.detach()
        }

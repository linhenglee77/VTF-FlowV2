"""Learned, differentiable and vehicle-conditioned terrain feasibility field."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from TerraFlow.interfaces import BaseTerrainField
from TerraFlow.terrain.feasibility_field import AnalyticTerrainField, TerrainFieldConfig


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        groups = max(1, min(8, output_channels // 4))
        while output_channels % groups:
            groups -= 1
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class FeasibilityFieldNet(nn.Module):
    """Small U-Net predicting base risk and speed sensitivity on the BEV grid."""

    def __init__(self, input_channels: int = 3, width: int = 24):
        super().__init__()
        if input_channels not in (2, 3):
            raise ValueError("input_channels must be 2 (geometry-only) or 3")
        self.input_channels = input_channels
        self.enc1 = ConvBlock(input_channels, width)
        self.enc2 = ConvBlock(width, width * 2)
        self.enc3 = ConvBlock(width * 2, width * 4)
        self.bottleneck = ConvBlock(width * 4, width * 6)
        self.dec3 = ConvBlock(width * 6 + width * 4, width * 4)
        self.dec2 = ConvBlock(width * 4 + width * 2, width * 2)
        self.dec1 = ConvBlock(width * 2 + width, width)
        self.output = nn.Conv2d(width, 2, 1)

    def forward(self, terrain_map: torch.Tensor) -> dict[str, torch.Tensor]:
        if terrain_map.shape[1] == 3 and self.input_channels == 2:
            terrain_map = terrain_map[:, 1:3]
        if terrain_map.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, got {terrain_map.shape[1]}"
            )
        e1 = self.enc1(terrain_map)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        hidden = self.bottleneck(F.avg_pool2d(e3, 2))
        d3 = self.dec3(torch.cat([F.interpolate(hidden, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], dim=1))
        output = self.output(d1)
        return {
            "base_logit": output[:, 0:1],
            "speed_sensitivity_logit": output[:, 1:2],
        }


@dataclass(frozen=True)
class LearnedFieldConfig:
    forward_m: float = 24.0
    lateral_m: float = 12.0
    speed_reference_mps: float = 3.0
    maximum_speed_addition: float = 0.35
    cost_temperature: float = 3.0
    vehicle_physics_enabled: bool = False
    height_range_m: float = 4.5
    vehicle_width_m: float = 1.8
    maximum_grade_deg: float = 20.0
    maximum_cross_slope_deg: float = 15.0
    base_weight: float = 1.0
    speed_weight: float = 0.35
    occupancy_weight: float = 0.35
    longitudinal_slope_weight: float = 0.25
    cross_slope_weight: float = 0.30


class LearnedTerrainField(BaseTerrainField):
    """Bilinearly query learned grid predictions at continuous trajectory states."""

    def __init__(
        self,
        terrain_map: torch.Tensor,
        model: FeasibilityFieldNet,
        config: LearnedFieldConfig | None = None,
        prediction: dict[str, torch.Tensor] | None = None,
    ):
        if terrain_map.ndim == 3:
            terrain_map = terrain_map.unsqueeze(0)
        self.terrain_map = terrain_map
        self.model = model
        self.config = config or LearnedFieldConfig()
        prediction = model(terrain_map) if prediction is None else prediction
        self.base_cost_map = torch.sigmoid(prediction["base_logit"])
        self.speed_sensitivity_map = torch.sigmoid(prediction["speed_sensitivity_logit"])
        height = terrain_map[:, 2:3] * self.config.height_range_m
        cell_m = self.config.forward_m / terrain_map.shape[-2]
        kernel_x = torch.tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
            device=terrain_map.device,
            dtype=terrain_map.dtype,
        ).unsqueeze(0) / (8.0 * cell_m)
        # BEV rows encode forward x and columns encode lateral y.
        self.dz_dx_map = F.conv2d(height, kernel_x.transpose(-1, -2), padding=1)
        self.dz_dy_map = F.conv2d(height, kernel_x, padding=1)
        footprint_cells = max(1, int(round(self.config.vehicle_width_m / cell_m)))
        if footprint_cells % 2 == 0:
            footprint_cells += 1
        self.clearance_map = F.max_pool2d(
            terrain_map[:, 1:2], footprint_cells, stride=1, padding=footprint_cells // 2
        )

    def repeat_interleave(self, repeats: int) -> "LearnedTerrainField":
        prediction = {
            "base_logit": torch.logit(
                self.base_cost_map.clamp(1e-5, 1 - 1e-5)
            ).repeat_interleave(repeats, dim=0),
            "speed_sensitivity_logit": torch.logit(
                self.speed_sensitivity_map.clamp(1e-5, 1 - 1e-5)
            ).repeat_interleave(repeats, dim=0),
        }
        return LearnedTerrainField(
            self.terrain_map.repeat_interleave(repeats, dim=0),
            self.model,
            self.config,
            prediction=prediction,
        )

    def _sample(self, value_map: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        original_shape = xy.shape[:-1]
        grid = torch.stack(
            [
                xy[..., 1] / self.config.lateral_m,
                xy[..., 0] / (self.config.forward_m / 2.0) - 1.0,
            ],
            dim=-1,
        ).reshape(xy.shape[0], -1, 1, 2)
        values = F.grid_sample(
            value_map, grid, mode="bilinear", padding_mode="border", align_corners=False
        )[:, 0, :, 0]
        return values.reshape(original_shape)

    def component_costs(self, xy: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "learned_base": self._sample(self.base_cost_map, xy),
            "speed_sensitivity": self._sample(self.speed_sensitivity_map, xy),
            "occupancy": self._sample(self.terrain_map[:, 1:2], xy),
            "clearance": self._sample(self.clearance_map, xy),
            "dz_dx": self._sample(self.dz_dx_map, xy),
            "dz_dy": self._sample(self.dz_dy_map, xy),
        }

    def cost(
        self, xyz: torch.Tensor, vehicle_state: dict[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        components = self.component_costs(xyz[..., :2])
        cost = components["learned_base"]
        if vehicle_state is not None and "speed" in vehicle_state:
            speed_ratio = (
                vehicle_state["speed"] / self.config.speed_reference_mps
            ).clamp(0.0, 1.5)
            speed_cost = speed_ratio * components["speed_sensitivity"]
            if not self.config.vehicle_physics_enabled:
                cost = cost + self.config.maximum_speed_addition * speed_cost
        else:
            speed_cost = torch.zeros_like(cost)
        if not self.config.vehicle_physics_enabled:
            return cost.clamp(0.0, 1.0)

        heading = (
            vehicle_state.get("heading", torch.zeros_like(cost))
            if vehicle_state is not None
            else torch.zeros_like(cost)
        )
        cos_heading, sin_heading = torch.cos(heading), torch.sin(heading)
        longitudinal_grade = torch.abs(
            components["dz_dx"] * cos_heading + components["dz_dy"] * sin_heading
        )
        cross_grade = torch.abs(
            -components["dz_dx"] * sin_heading + components["dz_dy"] * cos_heading
        )
        longitudinal_risk = (
            longitudinal_grade / math.tan(math.radians(self.config.maximum_grade_deg))
        ).clamp(0.0, 1.0)
        cross_slope_risk = (
            cross_grade / math.tan(math.radians(self.config.maximum_cross_slope_deg))
        ).clamp(0.0, 1.0)
        numerator = (
            self.config.base_weight * components["learned_base"]
            + self.config.speed_weight * speed_cost
            + self.config.occupancy_weight * components["clearance"]
            + self.config.longitudinal_slope_weight * longitudinal_risk
            + self.config.cross_slope_weight * cross_slope_risk
        )
        denominator = (
            self.config.base_weight
            + self.config.speed_weight
            + self.config.occupancy_weight
            + self.config.longitudinal_slope_weight
            + self.config.cross_slope_weight
        )
        return (numerator / denominator).clamp(0.0, 1.0)

    def query(
        self, xyz: torch.Tensor, vehicle_state: dict[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        return torch.exp(-self.config.cost_temperature * self.cost(xyz, vehicle_state))


@torch.no_grad()
def analytic_speed_sensitivity_target(terrain_map: torch.Tensor) -> torch.Tensor:
    """Physics-informed auxiliary target; semantic risk remains the main label."""

    analytic = AnalyticTerrainField(terrain_map, TerrainFieldConfig())
    return (
        0.65 * analytic.components["slope"]
        + 0.35 * analytic.components["clearance"]
    ).clamp(0.0, 1.0)

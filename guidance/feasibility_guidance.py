"""Normalized feasibility gradients for trajectory Flow ODE integration."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from TerraFlow.interfaces import BaseTerrainField
from TerraFlow.terrain.feasibility_field import AnalyticTerrainField, TerrainFieldConfig


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Differentiably wrap angles to ``[-pi, pi]``."""

    return torch.atan2(torch.sin(angle), torch.cos(angle))


def trajectory_dynamics_terms(
    metric_path: torch.Tensor,
    wheelbase_m: float = 2.2,
    sample_dt_s: float = 0.5,
    max_steering_deg: float = 30.0,
    max_steering_rate_deg_s: float = 45.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the legacy curvature and steering-rate guidance terms.

    This local implementation keeps the legacy guidance module self-contained;
    earlier development versions imported it from a sibling repository.
    """

    origin = torch.zeros_like(metric_path[:, :1, :2])
    xy = torch.cat([origin, metric_path[..., :2]], dim=1)
    segments = torch.diff(xy, dim=1)
    segment_length = torch.linalg.norm(segments, dim=-1).clamp_min(0.05)
    heading = torch.atan2(segments[..., 1], segments[..., 0])
    heading_change = _wrap_angle(torch.diff(heading, dim=1))
    arc_length = 0.5 * (segment_length[:, 1:] + segment_length[:, :-1])
    curvature = heading_change / arc_length.clamp_min(0.05)
    steering = torch.atan(wheelbase_m * curvature)
    initial_steering = torch.zeros_like(steering[:, :1])
    steering_rate = torch.diff(
        torch.cat([initial_steering, steering], dim=1), dim=1
    ) / sample_dt_s

    curvature_limit = math.tan(math.radians(max_steering_deg)) / wheelbase_m
    steering_rate_limit = math.radians(max_steering_rate_deg_s)
    curvature_cost = (
        0.10 * curvature.square()
        + 4.0 * F.relu(curvature.abs() - curvature_limit).square()
    ).mean(dim=1)
    steering_rate_cost = (
        0.10 * steering_rate.square()
        + 2.0 * F.relu(steering_rate.abs() - steering_rate_limit).square()
    ).mean(dim=1)
    return curvature_cost, steering_rate_cost, curvature, steering_rate


@dataclass(frozen=True)
class FlowGuidanceConfig:
    enabled: bool = True
    strength: float = 0.12
    schedule: str = "late"
    terrain_weight: float = 1.0
    occupancy_weight: float = 0.0
    smoothness_weight: float = 0.03
    curvature_weight: float = 0.20
    steering_rate_weight: float = 0.10
    boundary_weight: float = 0.20
    progress_weight: float = 0.0
    path_efficiency_weight: float = 0.0
    initial_heading_weight: float = 0.0
    gradient_clip: float = 4.0
    vehicle_conditioned: bool = True
    planning_dt_s: float = 0.5
    normalize_objective_terms: bool = False
    smoothness_reference_m: float = 0.75
    boundary_reference_m: float = 2.0

    def eta(self, time: float) -> float:
        if not self.enabled:
            return 0.0
        if self.schedule == "constant":
            return self.strength
        if self.schedule == "sine":
            return self.strength * float(torch.sin(torch.tensor(torch.pi * time)).square())
        if self.schedule == "late":
            return self.strength * time * time
        raise ValueError(f"Unknown guidance schedule: {self.schedule}")


def trajectory_vehicle_state(
    path: torch.Tensor,
    planning_dt_s: float,
    initial_vehicle_state: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Infer per-waypoint speed and heading, anchored by the current ego state."""

    origin = torch.zeros_like(path[:, :1, :2])
    delta = torch.diff(torch.cat([origin, path[..., :2]], dim=1), dim=1)
    speed = torch.linalg.norm(delta, dim=-1) / planning_dt_s
    heading = torch.atan2(delta[..., 1], delta[..., 0])
    if initial_vehicle_state is not None:
        if "speed" in initial_vehicle_state:
            initial_speed = initial_vehicle_state["speed"].reshape(-1, 1).to(speed)
            speed = torch.cat([initial_speed, speed[:, 1:]], dim=1)
        if "heading" in initial_vehicle_state:
            initial_heading = initial_vehicle_state["heading"].reshape(-1, 1).to(heading)
            heading = torch.cat([initial_heading, heading[:, 1:]], dim=1)
    return {"speed": speed, "heading": heading}


def feasibility_gradient(
    state: torch.Tensor,
    metric_path_fn,
    terrain_map: torch.Tensor,
    guidance: FlowGuidanceConfig,
    terrain_config: TerrainFieldConfig | None = None,
    field: BaseTerrainField | None = None,
    initial_vehicle_state: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Differentiate terrain and kinematic costs with respect to Flow state."""
    with torch.enable_grad():
        guided_state = state.detach().requires_grad_(True)
        path = metric_path_fn(guided_state)
        if field is None:
            field = AnalyticTerrainField(terrain_map, terrain_config)
        vehicle_state = (
            trajectory_vehicle_state(path, guidance.planning_dt_s, initial_vehicle_state)
            if guidance.vehicle_conditioned
            else None
        )
        terrain_point_cost = field.cost(path, vehicle_state)
        components = field.component_costs(path[..., :2])
        origin = torch.zeros_like(path[:, :1, :2])
        boundary = (
            F.relu(-path[..., 0])
            + F.relu(path[..., 0] - field.config.forward_m)
            + F.relu(path[..., 1].abs() - field.config.lateral_m)
        )
        second = torch.linalg.norm(torch.diff(path[..., :2], n=2, dim=1), dim=-1).mean(dim=1)
        curvature, steering_rate, _, _ = trajectory_dynamics_terms(
            path, sample_dt_s=guidance.planning_dt_s
        )
        delta = torch.diff(torch.cat([origin, path[..., :2]], dim=1), dim=1)
        forward_regression = F.relu(-delta[..., 0]).mean(dim=1)
        path_length = torch.linalg.norm(delta, dim=-1).sum(dim=1)
        direct_distance = torch.linalg.norm(path[:, -1, :2], dim=-1).clamp_min(0.5)
        path_efficiency = F.relu(path_length / direct_distance - 1.0)
        initial_heading = torch.atan2(delta[:, 0, 1], delta[:, 0, 0]).abs()
        if guidance.normalize_objective_terms:
            second_objective = (second / guidance.smoothness_reference_m).clamp(0.0, 1.0)
            curvature_objective = curvature.clamp(0.0, 1.0)
            steering_rate_objective = steering_rate.clamp(0.0, 1.0)
            boundary_objective = (boundary.mean(dim=1) / guidance.boundary_reference_m).clamp(0.0, 1.0)
            progress_objective = (forward_regression / 0.5).clamp(0.0, 1.0)
            path_efficiency_objective = path_efficiency.clamp(0.0, 1.0)
            initial_heading_objective = (initial_heading / (torch.pi / 2.0)).clamp(0.0, 1.0)
        else:
            second_objective = second
            curvature_objective = curvature
            steering_rate_objective = steering_rate
            boundary_objective = boundary.mean(dim=1)
            progress_objective = forward_regression
            path_efficiency_objective = path_efficiency
            initial_heading_objective = initial_heading
        per_path = (
            guidance.terrain_weight * terrain_point_cost.mean(dim=1)
            + guidance.occupancy_weight * components["occupancy"].mean(dim=1)
            + guidance.smoothness_weight * second_objective
            + guidance.curvature_weight * curvature_objective
            + guidance.steering_rate_weight * steering_rate_objective
            + guidance.boundary_weight * boundary_objective
            + guidance.progress_weight * progress_objective
            + guidance.path_efficiency_weight * path_efficiency_objective
            + guidance.initial_heading_weight * initial_heading_objective
        )
        gradient = torch.autograd.grad(per_path.sum(), guided_state)[0]
        rms = gradient.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        normalized = (gradient / rms).clamp(-guidance.gradient_clip, guidance.gradient_clip)
    return normalized.detach(), {
        "total_cost": per_path.detach(),
        "terrain_cost": terrain_point_cost.mean(dim=1).detach(),
        "smoothness_cost": second.detach(),
        "curvature_cost": curvature.detach(),
        "steering_rate_cost": steering_rate.detach(),
        "forward_regression_cost": progress_objective.detach(),
        "path_efficiency_cost": path_efficiency_objective.detach(),
        "initial_heading_cost": initial_heading_objective.detach(),
    }

"""Feasibility metric interfaces with explicit per-metric availability."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional

import torch
from torch import Tensor

from TerraFlow.terrain.feasibility_field import AnalyticTerrainField


MetricState = Dict[str, object]


@dataclass(frozen=True)
class FeasibilityMetricConfig:
    """Thresholds used by currently computable feasibility metrics."""

    planning_dt_s: float = 0.5
    occupancy_threshold: float = 0.5
    nontraversable_threshold: float = 0.5
    normalized_slope_threshold: float = 1.0
    curvature_limit_per_m: float = math.tan(math.radians(30.0)) / 2.2
    lateral_acceleration_limit_mps2: float = 4.0

    def __post_init__(self) -> None:
        if self.planning_dt_s <= 0.0:
            raise ValueError("planning_dt_s must be positive")
        if self.curvature_limit_per_m <= 0.0:
            raise ValueError("curvature_limit_per_m must be positive")
        if self.lateral_acceleration_limit_mps2 <= 0.0:
            raise ValueError("lateral_acceleration_limit_mps2 must be positive")


TERRAIN_METRIC_NAMES = (
    "occupancy_violation_rate",
    "traversability_violation_rate",
    "slope_violation_rate",
    "mean_terrain_cost",
    "minimum_obstacle_clearance",
    "elevation_consistency_error",
    "curvature_violation_rate",
    "lateral_acceleration_violation_rate",
)


def available_metric(values: Tensor, unit: str, description: str) -> MetricState:
    """Create an available metric state."""

    return {
        "status": "available",
        "values": values,
        "unit": unit,
        "description": description,
    }


def unavailable_metric(
    reason: str, required_inputs: List[str], unit: str
) -> MetricState:
    """Create a machine-readable unavailable state without a fake number."""

    return {
        "status": "unavailable",
        "reason": reason,
        "required_inputs": required_inputs,
        "unit": unit,
        "values": None,
    }


def _validate_trajectories(trajectories: Tensor) -> None:
    if trajectories.ndim != 4 or trajectories.shape[-1] < 2:
        raise ValueError("trajectories must have shape [B, K, H, D] with D >= 2")
    if not trajectories.is_floating_point() or not torch.isfinite(trajectories).all():
        raise ValueError("trajectories must be a finite floating-point tensor")


def _kinematic_metrics(
    trajectories: Tensor, config: FeasibilityMetricConfig
) -> Dict[str, MetricState]:
    batch_candidates = trajectories.shape[:2]
    if trajectories.shape[2] < 3:
        reason = "at least three trajectory points are required for curvature"
        return {
            "curvature_violation_rate": unavailable_metric(
                reason, ["trajectories with H >= 3"], "fraction"
            ),
            "lateral_acceleration_violation_rate": unavailable_metric(
                reason, ["trajectories with H >= 3", "planning_dt_s"], "fraction"
            ),
        }

    xy = trajectories[..., :2]
    delta = torch.diff(xy, dim=2)
    segment_length = torch.linalg.vector_norm(delta, dim=-1)
    heading = torch.atan2(delta[..., 1], delta[..., 0])
    heading_delta = torch.diff(heading, dim=2)
    heading_delta = torch.atan2(torch.sin(heading_delta), torch.cos(heading_delta))
    arc_length = 0.5 * (segment_length[..., :-1] + segment_length[..., 1:])
    curvature = heading_delta / arc_length.clamp_min(1e-6)
    curvature = torch.where(arc_length <= 1e-6, torch.zeros_like(curvature), curvature)
    curvature_rate = (
        curvature.abs() > config.curvature_limit_per_m
    ).to(xy.dtype).mean(dim=-1)

    segment_speed = segment_length / config.planning_dt_s
    centered_speed = 0.5 * (segment_speed[..., :-1] + segment_speed[..., 1:])
    lateral_acceleration = centered_speed.square() * curvature.abs()
    lateral_rate = (
        lateral_acceleration > config.lateral_acceleration_limit_mps2
    ).to(xy.dtype).mean(dim=-1)
    assert curvature_rate.shape == batch_candidates
    return {
        "curvature_violation_rate": available_metric(
            curvature_rate,
            "fraction",
            "fraction of interior waypoints exceeding configured curvature",
        ),
        "lateral_acceleration_violation_rate": available_metric(
            lateral_rate,
            "fraction",
            "fraction of interior waypoints exceeding configured lateral acceleration",
        ),
    }


def feasibility_metrics(
    trajectories: Tensor,
    terrain_map: Optional[Tensor] = None,
    planning_dt_s: float = 0.5,
    config: Optional[FeasibilityMetricConfig] = None,
) -> Dict[str, MetricState]:
    """Return per-candidate feasibility metrics or unavailable states.

    Available numeric values always have shape ``[B,K]``. A generic tensor is
    not silently treated as every possible terrain input: clearance and
    elevation error remain unavailable until dedicated inputs exist.
    """

    _validate_trajectories(trajectories)
    cfg = config or FeasibilityMetricConfig(planning_dt_s=planning_dt_s)
    states: Dict[str, MetricState] = {}
    terrain_names = (
        "occupancy_violation_rate",
        "traversability_violation_rate",
        "slope_violation_rate",
        "mean_terrain_cost",
    )
    if terrain_map is None:
        for name in terrain_names:
            states[name] = unavailable_metric(
                "terrain_map is unavailable",
                ["terrain_map [B,3,H_map,W_map] with documented VTF-Flow channels"],
                "fraction" if name.endswith("rate") else "normalized_cost",
            )
    else:
        try:
            if terrain_map.ndim == 3:
                terrain_map = terrain_map.unsqueeze(0)
            if terrain_map.shape[0] != trajectories.shape[0]:
                raise ValueError("terrain map batch size does not match trajectories")
            batch, candidates, horizon, dimension = trajectories.shape
            if dimension >= 3:
                flat = trajectories[..., :3].reshape(batch * candidates, horizon, 3)
            else:
                zeros = torch.zeros_like(trajectories[..., :1])
                flat = torch.cat((trajectories[..., :2], zeros), dim=-1).reshape(
                    batch * candidates, horizon, 3
                )
            repeated_map = terrain_map.repeat_interleave(candidates, dim=0)
            field = AnalyticTerrainField(repeated_map)
            components = field.component_costs(flat[..., :2])
            terrain_cost = field.cost(flat)
            shape = (batch, candidates)
            states["occupancy_violation_rate"] = available_metric(
                (components["occupancy"] >= cfg.occupancy_threshold)
                .to(trajectories.dtype)
                .mean(dim=-1)
                .reshape(shape),
                "fraction",
                "fraction of waypoints above occupancy threshold",
            )
            states["traversability_violation_rate"] = available_metric(
                (components["nontraversable"] >= cfg.nontraversable_threshold)
                .to(trajectories.dtype)
                .mean(dim=-1)
                .reshape(shape),
                "fraction",
                "fraction of waypoints above non-traversability threshold",
            )
            states["slope_violation_rate"] = available_metric(
                (components["slope"] >= cfg.normalized_slope_threshold)
                .to(trajectories.dtype)
                .mean(dim=-1)
                .reshape(shape),
                "fraction",
                "fraction of waypoints above normalized slope threshold",
            )
            states["mean_terrain_cost"] = available_metric(
                terrain_cost.mean(dim=-1).reshape(shape),
                "normalized_cost",
                "mean analytic terrain cost along each candidate",
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            for name in terrain_names:
                states[name] = unavailable_metric(
                    f"terrain inputs are incompatible: {error}",
                    ["terrain_map [B,3,H_map,W_map] with documented VTF-Flow channels"],
                    "fraction" if name.endswith("rate") else "normalized_cost",
                )

    states["minimum_obstacle_clearance"] = unavailable_metric(
        "metric requires a metric-distance obstacle geometry or distance field",
        ["obstacle_distance_field", "map_resolution_and_origin"],
        "m",
    )
    states["elevation_consistency_error"] = unavailable_metric(
        "metric requires a calibrated elevation field sampled in the trajectory frame",
        ["elevation_field", "map_resolution_and_origin", "verified_frame_transform"],
        "m",
    )
    states.update(_kinematic_metrics(trajectories, cfg))
    return states

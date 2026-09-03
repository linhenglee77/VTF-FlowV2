"""Independent component and candidate-availability metrics for planning claims.

The metrics deliberately keep the raw BEV components separate from the weighted
TVK training objective.  A demonstration envelope can be fitted on validation
GT component distributions and then frozen before evaluating test candidates.
It is a relative surrogate constraint envelope, not a calibrated safety label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

from TerraFlow.terrain.feasibility_field import AnalyticTerrainField, TerrainFieldConfig
from TerraFlow.terrain.trajectory_kinematics import (
    TrajectoryKinematicConfig,
    trajectory_kinematic_cost,
)


UPPER_ENVELOPE_METRICS = (
    "occupancy_exposure_rate",
    "nontraversable_exposure_rate",
    "slope_exposure_rate",
    "roughness_mean",
    "curvature_violation_rate",
    "lateral_acceleration_violation_rate",
)
LOWER_ENVELOPE_METRICS = ("clearance_q05_m",)


@dataclass(frozen=True)
class PlanningClaimMetricConfig:
    """Configuration for independently decomposed planning diagnostics."""

    forward_m: float = 24.0
    lateral_m: float = 12.0
    occupancy_threshold: float = 0.5
    traversability_threshold: float = 0.5
    normalized_slope_threshold: float = 1.0
    planning_dt_s: float = 0.5
    goal_tolerance_m: float = 0.25
    envelope_upper_quantile: float = 0.95
    envelope_lower_quantile: float = 0.05

    def __post_init__(self) -> None:
        if self.forward_m <= 0.0 or self.lateral_m <= 0.0:
            raise ValueError("BEV metric extents must be positive")
        if self.planning_dt_s <= 0.0 or self.goal_tolerance_m <= 0.0:
            raise ValueError("time step and goal tolerance must be positive")
        if not 0.0 <= self.occupancy_threshold <= 1.0:
            raise ValueError("occupancy_threshold must lie in [0,1]")
        if not 0.0 <= self.traversability_threshold <= 1.0:
            raise ValueError("traversability_threshold must lie in [0,1]")
        if not 0.0 < self.envelope_lower_quantile < self.envelope_upper_quantile < 1.0:
            raise ValueError("envelope quantiles must be ordered inside (0,1)")


def _validate_maps(terrain_map: torch.Tensor) -> torch.Tensor:
    if terrain_map.ndim == 3:
        terrain_map = terrain_map.unsqueeze(0)
    if terrain_map.ndim != 4 or terrain_map.shape[1] != 3:
        raise ValueError("terrain_map must have shape [B,3,H,W]")
    if not torch.isfinite(terrain_map).all():
        raise ValueError("terrain_map must contain finite values")
    return terrain_map


def derive_independent_component_maps(
    terrain_map: torch.Tensor,
    terrain_config: TerrainFieldConfig,
    metric_config: PlanningClaimMetricConfig,
) -> dict[str, torch.Tensor]:
    """Derive unweighted BEV components and a metric clearance proxy.

    Clearance is computed with an Euclidean distance transform of the raw
    obstacle-density mask.  It is independent of the weighted TVK combination,
    but remains a BEV proxy rather than a calibrated vehicle-body clearance.
    """

    terrain_map = _validate_maps(terrain_map)
    field = AnalyticTerrainField(terrain_map, terrain_config)
    occupancy = terrain_map[:, 1:2].clamp(0.0, 1.0)
    traversability = terrain_map[:, 0:1].clamp(0.0, 1.0)
    rows, columns = terrain_map.shape[-2:]
    forward_resolution = metric_config.forward_m / rows
    lateral_resolution = 2.0 * metric_config.lateral_m / columns
    clearance_arrays = []
    for sample in occupancy[:, 0].detach().cpu().numpy():
        occupied = sample >= metric_config.occupancy_threshold
        clearance_arrays.append(
            distance_transform_edt(
                ~occupied,
                sampling=(forward_resolution, lateral_resolution),
            ).astype(np.float32)
        )
    clearance = torch.from_numpy(np.stack(clearance_arrays))[:, None].to(
        device=terrain_map.device,
        dtype=terrain_map.dtype,
    )
    return {
        "occupancy": occupancy,
        "nontraversable": 1.0 - traversability,
        "slope": field.components["slope"],
        "roughness": field.components["roughness"],
        "clearance_m": clearance,
    }


def sample_component_maps(
    component_maps: Mapping[str, torch.Tensor],
    trajectories: torch.Tensor,
    metric_config: PlanningClaimMetricConfig,
) -> dict[str, torch.Tensor]:
    """Bilinearly sample component maps along ``[B,K,H,D]`` trajectories."""

    if trajectories.ndim != 4 or trajectories.shape[-1] < 2:
        raise ValueError("trajectories must have shape [B,K,H,D>=2]")
    if not torch.isfinite(trajectories).all():
        raise ValueError("trajectories must contain finite values")
    batch, candidates, horizon = trajectories.shape[:3]
    xy = trajectories[..., :2]
    grid = torch.stack(
        (
            xy[..., 1] / metric_config.lateral_m,
            xy[..., 0] / (metric_config.forward_m / 2.0) - 1.0,
        ),
        dim=-1,
    ).reshape(batch, candidates * horizon, 1, 2)
    sampled: dict[str, torch.Tensor] = {}
    for name, values in component_maps.items():
        if values.ndim != 4 or values.shape[0] != batch or values.shape[1] != 1:
            raise ValueError(f"component {name} must have shape [B,1,H,W]")
        result = F.grid_sample(
            values,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )[:, 0, :, 0]
        sampled[name] = result.reshape(batch, candidates, horizon)
    return sampled


def candidate_claim_metrics(
    trajectories: torch.Tensor,
    goals: torch.Tensor,
    component_maps: Mapping[str, torch.Tensor],
    metric_config: PlanningClaimMetricConfig,
    kinematic_config: TrajectoryKinematicConfig,
) -> dict[str, torch.Tensor]:
    """Return unweighted terrain and kinematic metrics with shape ``[B,K]``."""

    if goals.ndim == 1:
        goals = goals.unsqueeze(0)
    if goals.ndim != 2 or goals.shape[0] != trajectories.shape[0]:
        raise ValueError("goals must have shape [B,D]")
    sampled = sample_component_maps(component_maps, trajectories, metric_config)
    occupancy = sampled["occupancy"] >= metric_config.occupancy_threshold
    nontraversable = (
        sampled["nontraversable"] >= 1.0 - metric_config.traversability_threshold
    )
    slope = sampled["slope"] >= metric_config.normalized_slope_threshold
    clearance = sampled["clearance_m"]
    flat = trajectories.reshape(-1, trajectories.shape[-2], trajectories.shape[-1])
    kinematics = trajectory_kinematic_cost(
        flat,
        metric_config.planning_dt_s,
        kinematic_config,
    )
    batch, candidates, horizon = trajectories.shape[:3]
    curvature_violation = kinematics["curvature_violation"].reshape(
        batch, candidates, horizon
    )
    lateral_violation = kinematics["lateral_acceleration_violation"].reshape(
        batch, candidates, horizon
    )
    second = torch.linalg.vector_norm(torch.diff(trajectories, n=2, dim=2), dim=-1)
    return {
        "goal_error_m": torch.linalg.vector_norm(
            trajectories[:, :, -1] - goals[:, None, : trajectories.shape[-1]], dim=-1
        ),
        "occupancy_exposure_rate": occupancy.float().mean(dim=2),
        "nontraversable_exposure_rate": nontraversable.float().mean(dim=2),
        "slope_exposure_rate": slope.float().mean(dim=2),
        "roughness_mean": sampled["roughness"].mean(dim=2),
        "clearance_min_m": clearance.min(dim=2).values,
        "clearance_q05_m": torch.quantile(clearance, 0.05, dim=2),
        "curvature_violation_rate": curvature_violation.float().mean(dim=2),
        "lateral_acceleration_violation_rate": lateral_violation.float().mean(dim=2),
        "smoothness_m": second.mean(dim=2),
    }


def fit_demonstration_envelope(
    validation_metrics: Mapping[str, torch.Tensor],
    metric_config: PlanningClaimMetricConfig,
) -> dict[str, float]:
    """Fit component-wise limits from validation GT only."""

    envelope = {"goal_error_m": float(metric_config.goal_tolerance_m)}
    for name in UPPER_ENVELOPE_METRICS:
        values = validation_metrics[name].reshape(-1)
        if not torch.isfinite(values).all():
            raise ValueError(f"non-finite validation values for {name}")
        envelope[name] = float(
            torch.quantile(values, metric_config.envelope_upper_quantile)
        )
    for name in LOWER_ENVELOPE_METRICS:
        values = validation_metrics[name].reshape(-1)
        if not torch.isfinite(values).all():
            raise ValueError(f"non-finite validation values for {name}")
        envelope[name] = float(
            torch.quantile(values, metric_config.envelope_lower_quantile)
        )
    return envelope


def compliance_mask(
    candidate_metrics: Mapping[str, torch.Tensor],
    envelope: Mapping[str, float],
) -> torch.Tensor:
    """Return candidates inside the frozen validation demonstration envelope."""

    result = candidate_metrics["goal_error_m"] <= float(envelope["goal_error_m"])
    for name in UPPER_ENVELOPE_METRICS:
        result = result & (candidate_metrics[name] <= float(envelope[name]) + 1e-8)
    for name in LOWER_ENVELOPE_METRICS:
        result = result & (candidate_metrics[name] >= float(envelope[name]) - 1e-8)
    return result


def compliant_diversity(
    trajectories: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-scene compliant diversity and availability of two candidates."""

    if mask.shape != trajectories.shape[:2]:
        raise ValueError("mask must match [B,K]")
    values = torch.full(
        (trajectories.shape[0],),
        float("nan"),
        dtype=trajectories.dtype,
        device=trajectories.device,
    )
    has_pair = mask.sum(dim=1) >= 2
    for index in torch.nonzero(has_pair, as_tuple=False).flatten().tolist():
        selected = trajectories[index, mask[index]]
        pairwise = torch.linalg.vector_norm(
            selected[:, None] - selected[None, :], dim=-1
        ).mean(dim=-1)
        upper = torch.triu_indices(
            selected.shape[0], selected.shape[0], offset=1, device=selected.device
        )
        values[index] = pairwise[upper[0], upper[1]].mean()
    return values, has_pair


__all__ = [
    "PlanningClaimMetricConfig",
    "UPPER_ENVELOPE_METRICS",
    "LOWER_ENVELOPE_METRICS",
    "candidate_claim_metrics",
    "compliance_mask",
    "compliant_diversity",
    "derive_independent_component_maps",
    "fit_demonstration_envelope",
    "sample_component_maps",
]

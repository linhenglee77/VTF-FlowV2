"""Clean-trajectory-space terrain gradients for Flow Matching inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from TerraFlow.guidance.guidance_schedule import (
    GuidanceScheduleConfig,
    guidance_weight,
)
from TerraFlow.guidance.structure_preserving_guidance import (
    EndpointProjection,
    flow_gradient_cosine_similarity,
    smooth_trajectory_gradient,
)
from TerraFlow.models.flow_network import (
    ConditionalTrajectoryFlow,
    estimate_clean_trajectory,
)
from TerraFlow.terrain.feasibility_field import AnalyticTerrainField, TerrainFieldConfig
from TerraFlow.terrain.trajectory_kinematics import (
    TrajectoryKinematicConfig,
    trajectory_kinematic_cost,
)
from TerraFlow.terrain.vehicle_conditioned_field import (
    BatchedVehicleConditionedTerrainField,
    VehicleConditionedFieldConfig,
    trajectory_motion_state,
)


GradientNormalization = Literal["none", "l2", "rms"]
GuidanceFieldType = Literal["terrain", "vehicle"]


@dataclass(frozen=True)
class FeasibilityFlowGuidanceConfig:
    """Inference-only guidance controls; no training objective is changed."""

    enabled: bool = True
    strength: float = 0.1
    schedule: str = "late-strong"
    gamma: float = 1.0
    gradient_normalization: GradientNormalization = "rms"
    maximum_gradient_norm: float = 4.0
    minimum_gradient_norm: float = 1e-7
    field_type: GuidanceFieldType = "vehicle"
    planning_dt_s: float = 0.5
    curvature_weight: float = 0.0
    lateral_acceleration_weight: float = 0.0
    maximum_curvature_per_m: float = 0.35
    maximum_lateral_acceleration_mps2: float = 2.5
    curvature_softness_per_m: float = 0.05
    lateral_acceleration_softness_mps2: float = 0.5
    minimum_curvature_displacement_m: float = 0.1
    curvature_reliability_softness_m: float = 0.02
    save_clean_estimate_history: bool = False
    smoothing_kernel: str = "none"
    endpoint_projection: EndpointProjection = "none"
    trust_region_rho: float | None = None
    trust_region_scope: str = "trajectory"
    adaptive_trigger_enabled: bool = False
    trigger_alpha: float = 10.0
    trigger_reference_cost: float = 0.5

    def __post_init__(self) -> None:
        if self.strength < 0.0:
            raise ValueError("strength must be non-negative")
        GuidanceScheduleConfig(self.schedule, self.gamma)  # type: ignore[arg-type]
        if self.gradient_normalization not in {"none", "l2", "rms"}:
            raise ValueError("gradient_normalization must be 'none', 'l2', or 'rms'")
        if self.maximum_gradient_norm <= 0.0:
            raise ValueError("maximum_gradient_norm must be positive")
        if self.minimum_gradient_norm < 0.0:
            raise ValueError("minimum_gradient_norm must be non-negative")
        if self.field_type not in {"terrain", "vehicle"}:
            raise ValueError("field_type must be 'terrain' or 'vehicle'")
        if self.planning_dt_s <= 0.0:
            raise ValueError("planning_dt_s must be positive")
        TrajectoryKinematicConfig(
            curvature_weight=self.curvature_weight,
            lateral_acceleration_weight=self.lateral_acceleration_weight,
            maximum_curvature_per_m=self.maximum_curvature_per_m,
            maximum_lateral_acceleration_mps2=self.maximum_lateral_acceleration_mps2,
            curvature_softness_per_m=self.curvature_softness_per_m,
            lateral_acceleration_softness_mps2=self.lateral_acceleration_softness_mps2,
            minimum_curvature_displacement_m=self.minimum_curvature_displacement_m,
            curvature_reliability_softness_m=self.curvature_reliability_softness_m,
        )
        if self.smoothing_kernel not in {"none", "kernel_3", "kernel_5"}:
            raise ValueError("invalid smoothing_kernel")
        if self.endpoint_projection not in {"none", "terminal", "affine"}:
            raise ValueError("invalid endpoint_projection")
        if self.trust_region_rho is not None and self.trust_region_rho <= 0.0:
            raise ValueError("trust_region_rho must be positive or None")
        if self.trust_region_scope not in {"trajectory", "waypoint"}:
            raise ValueError("invalid trust_region_scope")
        if self.trigger_alpha <= 0.0:
            raise ValueError("trigger_alpha must be positive")

    @property
    def schedule_config(self) -> GuidanceScheduleConfig:
        """Return validated schedule parameters."""

        return GuidanceScheduleConfig(self.schedule, self.gamma)  # type: ignore[arg-type]

    @property
    def kinematic_config(self) -> TrajectoryKinematicConfig:
        """Return the validated trajectory-level kinematic objective."""

        return TrajectoryKinematicConfig(
            curvature_weight=self.curvature_weight,
            lateral_acceleration_weight=self.lateral_acceleration_weight,
            maximum_curvature_per_m=self.maximum_curvature_per_m,
            maximum_lateral_acceleration_mps2=self.maximum_lateral_acceleration_mps2,
            curvature_softness_per_m=self.curvature_softness_per_m,
            lateral_acceleration_softness_mps2=self.lateral_acceleration_softness_mps2,
            minimum_curvature_displacement_m=self.minimum_curvature_displacement_m,
            curvature_reliability_softness_m=self.curvature_reliability_softness_m,
        )


def normalize_and_clip_gradient(
    gradient: torch.Tensor,
    normalization: GradientNormalization,
    maximum_norm: float,
    minimum_norm: float = 0.0,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Normalize per trajectory and clip its flattened L2 norm."""

    if gradient.ndim != 3:
        raise ValueError("gradient must have shape [N,H,D]")
    if maximum_norm <= 0.0 or epsilon <= 0.0 or minimum_norm < 0.0:
        raise ValueError("norm bounds must be valid and epsilon positive")
    raw_norm = torch.linalg.vector_norm(gradient.flatten(start_dim=1), dim=1)
    active = raw_norm >= minimum_norm
    if normalization == "none":
        normalized = gradient
    elif normalization == "l2":
        normalized = gradient / raw_norm.clamp_min(epsilon)[:, None, None]
    elif normalization == "rms":
        rms = gradient.square().mean(dim=(1, 2)).sqrt()
        normalized = gradient / rms.clamp_min(epsilon)[:, None, None]
    else:
        raise ValueError("unknown gradient normalization")
    preclip_norm = torch.linalg.vector_norm(normalized.flatten(start_dim=1), dim=1)
    scale = (maximum_norm / preclip_norm.clamp_min(epsilon)).clamp(max=1.0)
    clipped = normalized * scale[:, None, None] * active[:, None, None]
    clipped_norm = torch.linalg.vector_norm(clipped.flatten(start_dim=1), dim=1)
    return clipped, {
        "raw_gradient_norm": raw_norm,
        "normalized_gradient_norm": preclip_norm,
        "guidance_gradient_norm": clipped_norm,
        "gradient_clip_scale": scale,
        "zero_gradient": ~active,
    }


def clean_feasibility_gradient(
    model: ConditionalTrajectoryFlow,
    state: torch.Tensor,
    time: torch.Tensor,
    condition: torch.Tensor,
    terrain_map: torch.Tensor,
    config: FeasibilityFlowGuidanceConfig,
    terrain_config: TerrainFieldConfig | None = None,
    vehicle_config: VehicleConditionedFieldConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Differentiate vehicle/terrain cost through ``x -> v -> x1_hat -> J``.

    The returned velocity and gradient are detached only *after*
    ``torch.autograd.grad`` has propagated through the clean estimate. Model
    parameters are not updated and no graph is retained across Euler steps.
    """

    if state.ndim != 3 or state.shape[-1] != 3:
        raise ValueError("state must have shape [N,H,3]")
    with torch.enable_grad():
        differentiable_state = state.detach().requires_grad_(True)
        velocity = model(differentiable_state, time, condition)
        estimated_clean = estimate_clean_trajectory(
            differentiable_state, time, velocity
        )
        terrain_field = AnalyticTerrainField(terrain_map, terrain_config)
        terrain_components = terrain_field.component_costs(estimated_clean[..., :2])
        terrain_cost = terrain_field.cost(estimated_clean)
        if config.field_type == "vehicle":
            vehicle_cfg = vehicle_config or VehicleConditionedFieldConfig()
            vehicle_state = trajectory_motion_state(
                estimated_clean, config.planning_dt_s, vehicle_cfg
            )
            field = BatchedVehicleConditionedTerrainField(terrain_field, vehicle_cfg)
            point_cost = field.cost(estimated_clean, vehicle_state)
            feasibility = field.query(estimated_clean, vehicle_state)
        else:
            point_cost = terrain_cost
            feasibility = terrain_field.query(estimated_clean)
        field_cost = point_cost.mean(dim=1)
        kinematic = trajectory_kinematic_cost(
            estimated_clean,
            config.planning_dt_s,
            config.kinematic_config,
        )
        kinematic_cost = kinematic["trajectory_kinematic_cost"]
        per_trajectory_cost = field_cost + kinematic_cost
        unified_feasibility = feasibility * torch.exp(
            -kinematic["pointwise_kinematic_cost"]
        )
        gradient = torch.autograd.grad(
            per_trajectory_cost.sum(), differentiable_state,
            create_graph=False, retain_graph=False,
        )[0]
        smoothed_gradient = smooth_trajectory_gradient(
            gradient, config.smoothing_kernel  # type: ignore[arg-type]
        )
        processed, gradient_diagnostics = normalize_and_clip_gradient(
            smoothed_gradient,
            config.gradient_normalization,
            config.maximum_gradient_norm,
            config.minimum_gradient_norm,
        )
    diagnostics = {
        **{name: value.detach() for name, value in gradient_diagnostics.items()},
        "guidance_cost": per_trajectory_cost.detach(),
        "vehicle_cost": field_cost.detach(),
        "kinematic_cost": kinematic_cost.detach(),
        "curvature_cost": kinematic["weighted_curvature_cost"].mean(dim=1).detach(),
        "lateral_acceleration_cost": (
            kinematic["weighted_lateral_acceleration_cost"].mean(dim=1).detach()
        ),
        "terrain_cost": terrain_cost.mean(dim=1).detach(),
        "mean_feasibility": unified_feasibility.mean(dim=1).detach(),
        "mean_field_feasibility": feasibility.mean(dim=1).detach(),
        "occupancy_cost": terrain_components["occupancy"].mean(dim=1).detach(),
        "slope_cost": terrain_components["slope"].mean(dim=1).detach(),
        "roughness_cost": terrain_components["roughness"].mean(dim=1).detach(),
        "clearance_cost": terrain_components["clearance"].mean(dim=1).detach(),
        "mean_absolute_curvature_per_m": (
            kinematic["absolute_curvature_per_m"].mean(dim=1).detach()
        ),
        "maximum_absolute_curvature_per_m": (
            kinematic["absolute_curvature_per_m"].max(dim=1).values.detach()
        ),
        "curvature_violation_rate": (
            kinematic["curvature_violation"].float().mean(dim=1).detach()
        ),
        "mean_lateral_acceleration_mps2": (
            kinematic["lateral_acceleration_mps2"].mean(dim=1).detach()
        ),
        "maximum_lateral_acceleration_mps2": (
            kinematic["lateral_acceleration_mps2"].max(dim=1).values.detach()
        ),
        "lateral_acceleration_violation_rate": (
            kinematic["lateral_acceleration_violation"].float().mean(dim=1).detach()
        ),
        "gradient_nonfinite": (~torch.isfinite(gradient)).flatten(start_dim=1).any(dim=1),
        "clean_displacement_norm": torch.linalg.vector_norm(
            (estimated_clean - differentiable_state).flatten(start_dim=1), dim=1
        ).detach(),
        "smoothed_gradient_norm": torch.linalg.vector_norm(
            smoothed_gradient.flatten(start_dim=1), dim=1
        ).detach(),
        "flow_gradient_cosine_similarity": flow_gradient_cosine_similarity(
            velocity, gradient
        ).detach(),
    }
    if estimated_clean.shape[1] >= 3:
        second = torch.linalg.vector_norm(
            torch.diff(estimated_clean, n=2, dim=1), dim=-1
        )
        diagnostics["clean_smoothness"] = second.mean(dim=1).detach()
        diagnostics["clean_max_second_difference"] = second.max(dim=1).values.detach()
    else:
        diagnostics["clean_smoothness"] = torch.zeros_like(per_trajectory_cost)
        diagnostics["clean_max_second_difference"] = torch.zeros_like(per_trajectory_cost)
    if config.save_clean_estimate_history:
        diagnostics["clean_estimate"] = estimated_clean.detach()
    return velocity.detach(), processed.detach(), diagnostics


def scheduled_guidance_strength(
    time: torch.Tensor,
    config: FeasibilityFlowGuidanceConfig,
) -> torch.Tensor:
    """Evaluate the configured eta schedule, including the enabled switch."""

    if not config.enabled:
        return torch.zeros_like(time)
    return guidance_weight(time, config.strength, config.schedule_config)


__all__ = [
    "FeasibilityFlowGuidanceConfig",
    "clean_feasibility_gradient",
    "normalize_and_clip_gradient",
    "scheduled_guidance_strength",
]

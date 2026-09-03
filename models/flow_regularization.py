"""Optional differentiable endpoint regularizers for Flow Matching training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from TerraFlow.interfaces import SceneBatch
from TerraFlow.models.flow_network import ConditionalTrajectoryFlow
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


RegularizationMode = Literal["none", "terrain", "vehicle"]


@dataclass(frozen=True)
class FlowRegularizationConfig:
    """Weights and temporal assumptions for optional Flow endpoint losses."""

    mode: RegularizationMode = "none"
    lambda_feasibility: float = 0.0
    lambda_smoothness: float = 0.0
    planning_dt_s: float = 0.5
    curvature_weight: float = 0.0
    lateral_acceleration_weight: float = 0.0
    maximum_curvature_per_m: float = 0.35
    maximum_lateral_acceleration_mps2: float = 2.5
    curvature_softness_per_m: float = 0.05
    lateral_acceleration_softness_mps2: float = 0.5
    minimum_curvature_displacement_m: float = 0.1
    curvature_reliability_softness_m: float = 0.02

    def __post_init__(self) -> None:
        if self.mode not in {"none", "terrain", "vehicle"}:
            raise ValueError("mode must be 'none', 'terrain', or 'vehicle'")
        if self.lambda_feasibility < 0.0 or self.lambda_smoothness < 0.0:
            raise ValueError("regularization weights must be non-negative")
        if self.planning_dt_s <= 0.0:
            raise ValueError("planning_dt_s must be positive")
        if self.mode == "none" and self.lambda_feasibility != 0.0:
            raise ValueError("lambda_feasibility must be zero when mode='none'")
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

    @property
    def kinematic_config(self) -> TrajectoryKinematicConfig:
        """Return trajectory-level terms used by the unified objective."""

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


def trajectory_smoothness_loss(trajectories: torch.Tensor) -> torch.Tensor:
    """Mean L2 magnitude of second waypoint differences, kept separate from F."""

    if trajectories.ndim != 3 or trajectories.shape[-1] != 3:
        raise ValueError("trajectories must have shape [B,H,3]")
    if trajectories.shape[1] < 3:
        return trajectories.sum() * 0.0
    second_difference = torch.diff(trajectories, n=2, dim=1)
    return torch.linalg.vector_norm(second_difference, dim=-1).mean()


def regularized_flow_matching_loss(
    model: ConditionalTrajectoryFlow,
    scene: SceneBatch,
    config: FlowRegularizationConfig,
    *,
    terrain_config: TerrainFieldConfig | None = None,
    vehicle_config: VehicleConditionedFieldConfig | None = None,
    base: torch.Tensor | None = None,
    time: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute unchanged FM loss plus optional clean-endpoint regularizers.

    Feasibility is evaluated on ``x1_hat=x_t+(1-t)v_theta``.  The returned
    ``flow_matching_loss`` is exactly the original unweighted velocity MSE;
    it is never replaced by a trajectory reconstruction loss.
    """

    clean = scene.gt_future[..., :3]
    terms = model.flow_matching_training_terms(
        clean,
        scene.ego_history,
        scene.goal,
        scene.terrain_map,
        base=base,
        time=time,
    )
    estimated_clean = terms["estimated_clean"]
    zero = estimated_clean.sum() * 0.0
    feasibility_loss = zero
    mean_feasibility = torch.ones((), dtype=clean.dtype, device=clean.device)
    mean_field_cost = zero
    mean_kinematic_cost = zero
    mean_curvature_cost = zero
    mean_lateral_acceleration_cost = zero

    if config.mode != "none" and config.lambda_feasibility > 0.0:
        if scene.terrain_map is None:
            raise ValueError("terrain_map is required for feasibility regularization")
        terrain_field = AnalyticTerrainField(scene.terrain_map, terrain_config)
        if config.mode == "terrain":
            field = terrain_field
            vehicle_state = None
        else:
            vehicle_cfg = vehicle_config or VehicleConditionedFieldConfig()
            field = BatchedVehicleConditionedTerrainField(terrain_field, vehicle_cfg)
            vehicle_state = trajectory_motion_state(
                estimated_clean,
                planning_dt_s=config.planning_dt_s,
                config=vehicle_cfg,
            )
        feasibility = field.query(estimated_clean, vehicle_state)
        field_cost = field.cost(estimated_clean, vehicle_state)
        kinematic = trajectory_kinematic_cost(
            estimated_clean,
            config.planning_dt_s,
            config.kinematic_config,
        )
        unified_feasibility = feasibility * torch.exp(
            -kinematic["pointwise_kinematic_cost"]
        )
        feasibility_loss = (1.0 - unified_feasibility).mean()
        mean_feasibility = unified_feasibility.mean()
        mean_field_cost = field_cost.mean()
        mean_kinematic_cost = kinematic["trajectory_kinematic_cost"].mean()
        mean_curvature_cost = kinematic["weighted_curvature_cost"].mean()
        mean_lateral_acceleration_cost = (
            kinematic["weighted_lateral_acceleration_cost"].mean()
        )

    smoothness_loss = (
        trajectory_smoothness_loss(estimated_clean)
        if config.lambda_smoothness > 0.0
        else zero
    )
    total = (
        terms["flow_matching_loss"]
        + config.lambda_feasibility * feasibility_loss
        + config.lambda_smoothness * smoothness_loss
    )
    terms.update(
        {
            "feasibility_loss": feasibility_loss,
            "smoothness_loss": smoothness_loss,
            "mean_feasibility": mean_feasibility,
            "mean_field_cost": mean_field_cost,
            "mean_kinematic_cost": mean_kinematic_cost,
            "mean_curvature_cost": mean_curvature_cost,
            "mean_lateral_acceleration_cost": mean_lateral_acceleration_cost,
            "total_loss": total,
        }
    )
    return total, terms


__all__ = [
    "FlowRegularizationConfig",
    "RegularizationMode",
    "regularized_flow_matching_loss",
    "trajectory_smoothness_loss",
]

"""Differentiable trajectory-level kinematic feasibility terms.

The terrain field remains a point-query interface. Curvature and lateral
acceleration require neighbouring waypoints, so they are evaluated here as
trajectory-level costs and combined with the terrain objective by the training
and inference guidance modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TrajectoryKinematicConfig:
    """Configurable soft limits for planar trajectory kinematics.

    The limits are nominal planning hyperparameters rather than calibrated
    limits of the RELLIS-3D data-collection vehicle. Zero weights disable the
    corresponding term exactly, preserving all legacy experiments.
    """

    curvature_weight: float = 0.0
    lateral_acceleration_weight: float = 0.0
    maximum_curvature_per_m: float = 0.35
    maximum_lateral_acceleration_mps2: float = 2.5
    curvature_softness_per_m: float = 0.05
    lateral_acceleration_softness_mps2: float = 0.5
    minimum_curvature_displacement_m: float = 0.1
    curvature_reliability_softness_m: float = 0.02
    numerical_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.curvature_weight < 0.0 or self.lateral_acceleration_weight < 0.0:
            raise ValueError("kinematic cost weights must be non-negative")
        positive = (
            self.maximum_curvature_per_m,
            self.maximum_lateral_acceleration_mps2,
            self.curvature_softness_per_m,
            self.lateral_acceleration_softness_mps2,
            self.minimum_curvature_displacement_m,
            self.curvature_reliability_softness_m,
            self.numerical_epsilon,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("kinematic limits, softness values and epsilon must be positive")

    @property
    def enabled(self) -> bool:
        """Whether at least one kinematic cost contributes to the objective."""

        return self.curvature_weight > 0.0 or self.lateral_acceleration_weight > 0.0


def _validate_trajectories(trajectories: torch.Tensor, planning_dt_s: float) -> None:
    if trajectories.ndim < 3 or trajectories.shape[-1] < 2:
        raise ValueError("trajectories must have shape [..., H, D] with D >= 2")
    if trajectories.shape[-2] < 1:
        raise ValueError("trajectories must contain at least one future waypoint")
    if planning_dt_s <= 0.0:
        raise ValueError("planning_dt_s must be positive")
    if not torch.isfinite(trajectories).all():
        raise ValueError("trajectories must contain only finite values")


def trajectory_kinematic_quantities(
    trajectories: torch.Tensor,
    planning_dt_s: float,
    *,
    numerical_epsilon: float = 1e-6,
    minimum_curvature_displacement_m: float = 0.1,
    curvature_reliability_softness_m: float = 0.02,
) -> dict[str, torch.Tensor]:
    """Return waypoint-aligned speed, curvature and lateral acceleration.

    The fixed current ego origin is prepended to the future waypoints. Signed
    curvature uses the circumcircle expression for consecutive planar
    segments. The last valid curvature is repeated at the final waypoint so
    every returned tensor has shape [..., H].
    """

    _validate_trajectories(trajectories, planning_dt_s)
    if min(
        numerical_epsilon,
        minimum_curvature_displacement_m,
        curvature_reliability_softness_m,
    ) <= 0.0:
        raise ValueError("curvature reliability and numerical values must be positive")
    xy = trajectories[..., :2]
    origin = torch.zeros_like(xy[..., :1, :])
    path = torch.cat((origin, xy), dim=-2)
    segments = torch.diff(path, dim=-2)
    segment_lengths = torch.sqrt(
        segments.square().sum(dim=-1) + numerical_epsilon**2
    )
    speed = segment_lengths / planning_dt_s

    horizon = trajectories.shape[-2]
    if horizon == 1:
        zeros = torch.zeros_like(speed)
        return {
            "speed_mps": speed,
            "signed_curvature_per_m": zeros,
            "absolute_curvature_per_m": zeros,
            "lateral_acceleration_mps2": zeros,
            "curvature_reliability": zeros,
        }

    incoming = segments[..., :-1, :]
    outgoing = segments[..., 1:, :]
    chord = incoming + outgoing
    incoming_norm = torch.sqrt(
        incoming.square().sum(dim=-1) + numerical_epsilon**2
    )
    outgoing_norm = torch.sqrt(
        outgoing.square().sum(dim=-1) + numerical_epsilon**2
    )
    chord_norm = torch.sqrt(chord.square().sum(dim=-1) + numerical_epsilon**2)
    cross = incoming[..., 0] * outgoing[..., 1] - incoming[..., 1] * outgoing[..., 0]
    raw_signed_valid = 2.0 * cross / (
        incoming_norm * outgoing_norm * chord_norm + numerical_epsilon
    )
    support_length = torch.minimum(incoming_norm, outgoing_norm)
    # Squaring the smooth gate strongly suppresses ill-conditioned curvature
    # estimates from near-stationary pose jitter while preserving differentiable
    # gradients and approaching one for normally spaced planning waypoints.
    curvature_reliability_valid = torch.sigmoid(
        (support_length - minimum_curvature_displacement_m)
        / curvature_reliability_softness_m
    ).square()
    signed_valid = raw_signed_valid * curvature_reliability_valid
    signed_curvature = torch.cat((signed_valid, signed_valid[..., -1:]), dim=-1)
    curvature_reliability = torch.cat(
        (curvature_reliability_valid, curvature_reliability_valid[..., -1:]),
        dim=-1,
    )
    absolute_curvature = signed_curvature.abs()
    central_speed = 0.5 * (speed[..., :-1] + speed[..., 1:])
    central_speed = torch.cat((central_speed, speed[..., -1:]), dim=-1)
    lateral_acceleration = central_speed.square() * absolute_curvature
    return {
        "speed_mps": speed,
        "signed_curvature_per_m": signed_curvature,
        "absolute_curvature_per_m": absolute_curvature,
        "lateral_acceleration_mps2": lateral_acceleration,
        "curvature_reliability": curvature_reliability,
    }


def _normalized_soft_excess(
    value: torch.Tensor,
    limit: float,
    softness: float,
) -> torch.Tensor:
    """Smoothly penalize positive limit exceedance and equal zero at value zero."""

    scaled = (value - limit) / softness
    baseline = F.softplus(value.new_tensor(-limit / softness))
    excess = softness * (F.softplus(scaled) - baseline)
    return excess.clamp_min(0.0) / limit


def trajectory_kinematic_cost(
    trajectories: torch.Tensor,
    planning_dt_s: float,
    config: TrajectoryKinematicConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Return differentiable pointwise and per-trajectory kinematic costs."""

    cfg = config or TrajectoryKinematicConfig()
    quantities = trajectory_kinematic_quantities(
        trajectories,
        planning_dt_s,
        numerical_epsilon=cfg.numerical_epsilon,
        minimum_curvature_displacement_m=cfg.minimum_curvature_displacement_m,
        curvature_reliability_softness_m=cfg.curvature_reliability_softness_m,
    )
    curvature_penalty = _normalized_soft_excess(
        quantities["absolute_curvature_per_m"],
        cfg.maximum_curvature_per_m,
        cfg.curvature_softness_per_m,
    )
    lateral_penalty = _normalized_soft_excess(
        quantities["lateral_acceleration_mps2"],
        cfg.maximum_lateral_acceleration_mps2,
        cfg.lateral_acceleration_softness_mps2,
    )
    weighted_curvature = cfg.curvature_weight * curvature_penalty
    weighted_lateral = cfg.lateral_acceleration_weight * lateral_penalty
    pointwise = weighted_curvature + weighted_lateral
    return {
        **quantities,
        "curvature_penalty": curvature_penalty,
        "lateral_acceleration_penalty": lateral_penalty,
        "weighted_curvature_cost": weighted_curvature,
        "weighted_lateral_acceleration_cost": weighted_lateral,
        "pointwise_kinematic_cost": pointwise,
        "trajectory_kinematic_cost": pointwise.mean(dim=-1),
        "curvature_violation": (
            quantities["absolute_curvature_per_m"] > cfg.maximum_curvature_per_m
        ),
        "lateral_acceleration_violation": (
            quantities["lateral_acceleration_mps2"]
            > cfg.maximum_lateral_acceleration_mps2
        ),
    }


__all__ = [
    "TrajectoryKinematicConfig",
    "trajectory_kinematic_cost",
    "trajectory_kinematic_quantities",
]

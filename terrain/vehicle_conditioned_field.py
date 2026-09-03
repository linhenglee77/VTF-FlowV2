"""Differentiable vehicle-motion conditioning for continuous terrain fields."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F

from TerraFlow.interfaces import BaseTerrainField
from TerraFlow.terrain.feasibility_field import AnalyticTerrainField, ContinuousTerrainField


@dataclass(frozen=True)
class VehicleConditionedFieldConfig:
    """Continuous speed response and clearance requirement parameters.

    The additional motion cost is zero at zero speed, so this field reduces to
    its terrain-only base field for a stationary vehicle. Heading only affects
    the look-ahead clearance term and is blended by a reliability value in
    ``[0,1]``.
    """

    speed_reference_mps: float = 3.0
    maximum_speed_ratio: float = 2.0
    speed_response_exponent: float = 1.5
    slope_speed_gain: float = 1.0
    roughness_speed_gain: float = 0.8
    clearance_speed_gain: float = 1.0
    base_clearance_requirement_m: float = 0.75
    speed_clearance_gain_m: float = 1.25
    clearance_softness_m: float = 0.25
    heading_lookahead_time_s: float = 0.5
    heading_min_displacement_m: float = 0.15
    heading_reliability_softness_m: float = 0.05

    def __post_init__(self) -> None:
        positive = (
            self.speed_reference_mps,
            self.maximum_speed_ratio,
            self.speed_response_exponent,
            self.clearance_softness_m,
            self.heading_reliability_softness_m,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("vehicle response scales and exponents must be positive")
        nonnegative = (
            self.slope_speed_gain,
            self.roughness_speed_gain,
            self.clearance_speed_gain,
            self.base_clearance_requirement_m,
            self.speed_clearance_gain_m,
            self.heading_lookahead_time_s,
            self.heading_min_displacement_m,
        )
        if any(value < 0.0 for value in nonnegative):
            raise ValueError("vehicle cost gains and distances must be non-negative")


@dataclass(frozen=True)
class TrajectoryGradientSmoothingConfig:
    """Temporal Gaussian smoothing applied to waypoint guidance gradients."""

    sigma_waypoints: float = 0.0
    truncate_sigma: float = 3.0

    def __post_init__(self) -> None:
        if self.sigma_waypoints < 0.0:
            raise ValueError("sigma_waypoints must be non-negative")
        if self.truncate_sigma <= 0.0:
            raise ValueError("truncate_sigma must be positive")


def load_vehicle_conditioned_config(path: str | Path) -> VehicleConditionedFieldConfig:
    """Load the ``vehicle_conditioning`` section of a terrain JSON config."""

    with Path(path).expanduser().open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    section = raw.get("vehicle_conditioning", {})
    if not isinstance(section, Mapping):
        raise TypeError("vehicle_conditioning must be a JSON object")
    return VehicleConditionedFieldConfig(**dict(section))


def load_gradient_smoothing_config(path: str | Path) -> TrajectoryGradientSmoothingConfig:
    """Load reusable trajectory-gradient smoothing hyperparameters."""

    with Path(path).expanduser().open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    section = raw.get("guidance", {})
    if not isinstance(section, Mapping):
        raise TypeError("guidance must be a JSON object")
    return TrajectoryGradientSmoothingConfig(
        sigma_waypoints=float(
            section.get("trajectory_gradient_smoothing_sigma_waypoints", 0.0)
        ),
        truncate_sigma=float(
            section.get("trajectory_gradient_smoothing_truncate_sigma", 3.0)
        ),
    )


def smooth_trajectory_gradient(
    gradient: torch.Tensor,
    config: TrajectoryGradientSmoothingConfig,
) -> torch.Tensor:
    """Smooth ``[..., H, D]`` gradients along H without mixing coordinates."""

    if gradient.ndim < 2:
        raise ValueError("gradient must have shape [..., H, D]")
    if config.sigma_waypoints == 0.0 or gradient.shape[-2] < 2:
        return gradient
    radius = max(
        1,
        int(torch.ceil(torch.tensor(
            config.sigma_waypoints * config.truncate_sigma
        )).item()),
    )
    offsets = torch.arange(
        -radius, radius + 1, device=gradient.device, dtype=gradient.dtype
    )
    kernel = torch.exp(-0.5 * (offsets / config.sigma_waypoints).square())
    kernel = kernel / kernel.sum()
    horizon = gradient.shape[-2]
    dimensions = gradient.shape[-1]
    leading = gradient.shape[:-2]
    flattened = gradient.reshape(-1, horizon, dimensions).transpose(1, 2)
    padded = F.pad(flattened, (radius, radius), mode="replicate")
    smoothed = F.conv1d(
        padded,
        kernel.reshape(1, 1, -1).expand(dimensions, 1, -1),
        groups=dimensions,
    )
    return smoothed.transpose(1, 2).reshape(*leading, horizon, dimensions)


def _broadcast_state(
    value: torch.Tensor | float,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    state = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if not torch.isfinite(state).all():
        raise ValueError(f"vehicle state '{name}' must be finite")
    try:
        return torch.broadcast_to(state, reference.shape)
    except RuntimeError:
        while state.ndim < reference.ndim:
            state = state.unsqueeze(-1)
        try:
            return torch.broadcast_to(state, reference.shape)
        except RuntimeError as error:
            raise ValueError(
                f"vehicle state '{name}' with shape {tuple(state.shape)} cannot "
                f"broadcast to query shape {tuple(reference.shape)}"
            ) from error


def trajectory_motion_state(
    trajectories: torch.Tensor,
    planning_dt_s: float,
    config: VehicleConditionedFieldConfig | None = None,
    initial_vehicle_state: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Derive waypoint speed, tangent heading and soft heading reliability.

    Args:
        trajectories: Tensor shaped ``[..., H, D]`` with ``D >= 2``.
        planning_dt_s: Time between future waypoints.
        config: Reliability transition parameters.
        initial_vehicle_state: Optional current ``speed`` and reliable
            ``heading`` used for the first predicted waypoint.

    Heading reliability is a sigmoid of segment displacement, avoiding a hard
    branch at the minimum reliable displacement. Stationary segments therefore
    contribute almost no heading-based look-ahead penalty.
    """

    if trajectories.ndim < 2 or trajectories.shape[-1] < 2:
        raise ValueError("trajectories must have shape [..., H, D] with D >= 2")
    if planning_dt_s <= 0.0:
        raise ValueError("planning_dt_s must be positive")
    cfg = config or VehicleConditionedFieldConfig()
    xy = trajectories[..., :2]
    origin = torch.zeros_like(xy[..., :1, :])
    delta = torch.diff(torch.cat((origin, xy), dim=-2), dim=-2)
    # The small quadratic floor keeps derivatives finite for repeated
    # waypoints. It changes reported displacement by at most 1e-6 m while
    # avoiding the undefined derivative of ||delta|| at delta == 0.
    numerical_epsilon_m = 1e-6
    displacement = torch.sqrt(
        delta.square().sum(dim=-1) + numerical_epsilon_m**2
    )
    speed = displacement / planning_dt_s
    # atan2(0, 0) has undefined gradients. A tiny positive x reference gives a
    # finite zero heading on stationary segments, whose reliability is already
    # driven close to zero by the displacement response below.
    heading = torch.atan2(delta[..., 1], delta[..., 0] + numerical_epsilon_m)
    reliability = torch.sigmoid(
        (displacement - cfg.heading_min_displacement_m)
        / cfg.heading_reliability_softness_m
    )
    if initial_vehicle_state is not None:
        target = speed[..., 0]
        if "speed" in initial_vehicle_state:
            initial_speed = _broadcast_state(initial_vehicle_state["speed"], target, "speed")
            speed = torch.cat((initial_speed.unsqueeze(-1), speed[..., 1:]), dim=-1)
        if "heading" in initial_vehicle_state:
            initial_heading = _broadcast_state(
                initial_vehicle_state["heading"], target, "heading"
            )
            heading = torch.cat((initial_heading.unsqueeze(-1), heading[..., 1:]), dim=-1)
            reliability = torch.cat(
                (torch.ones_like(initial_heading).unsqueeze(-1), reliability[..., 1:]),
                dim=-1,
            )
    return {
        "speed": speed,
        "heading": heading,
        "heading_reliability": reliability,
    }


class BinaryTraversabilityField(BaseTerrainField):
    """Thresholded geometry/semantic traversability comparison baseline."""

    def __init__(
        self,
        terrain_field: ContinuousTerrainField,
        occupancy_threshold: float = 0.5,
        semantic_cost_threshold: float = 0.5,
    ) -> None:
        self.terrain_field = terrain_field
        self.occupancy_threshold = float(occupancy_threshold)
        self.semantic_cost_threshold = float(semantic_cost_threshold)

    def query(
        self,
        xyz_or_xy: torch.Tensor,
        vehicle_state: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        del vehicle_state
        components = self.terrain_field.raw_component_costs(xyz_or_xy[..., :2])
        return (
            (components["occupancy"] < self.occupancy_threshold)
            & (components["semantic"] < self.semantic_cost_threshold)
            & (components["unknown"] < 0.5)
        ).to(dtype=xyz_or_xy.dtype)

    def cost(
        self,
        xyz_or_xy: torch.Tensor,
        vehicle_state: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        return 1.0 - self.query(xyz_or_xy, vehicle_state)


class VehicleConditionedTerrainField(BaseTerrainField):
    """Add continuous speed and reliable-heading penalties to a terrain field.

    For speed response ``r(v) = clamp(v/v_ref, 0, r_max)^p`` the additional
    point cost is

    ``gain_slope*r*C_slope + gain_rough*r*C_rough +``
    ``gain_clear*r*sigmoid((d_required(v)-d_effective)/softness)``.

    ``d_effective`` continuously blends current clearance with the minimum of
    current and heading look-ahead clearance using ``heading_reliability``.
    No pitch or roll state is introduced.
    """

    def __init__(
        self,
        terrain_field: ContinuousTerrainField,
        config: VehicleConditionedFieldConfig | None = None,
    ) -> None:
        self.terrain_field = terrain_field
        self.features = terrain_field.features
        self.config = config or VehicleConditionedFieldConfig()
        self._clearance_map = terrain_field._spatially_smooth(
            terrain_field._map(self.features.clearance_m)
        )

    def _vehicle_components(
        self,
        xyz_or_xy: torch.Tensor,
        vehicle_state: dict[str, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor]:
        xy = xyz_or_xy[..., :2]
        reference = xy[..., 0]
        if vehicle_state is None or "speed" not in vehicle_state:
            speed = torch.zeros_like(reference)
        else:
            speed = _broadcast_state(vehicle_state["speed"], reference, "speed")
            if bool((speed < 0.0).any()):
                raise ValueError("vehicle speed must be non-negative")
        speed_ratio = (speed / self.config.speed_reference_mps).clamp(
            0.0, self.config.maximum_speed_ratio
        )
        response = speed_ratio.pow(self.config.speed_response_exponent)
        base_components = self.terrain_field.component_costs(xy)

        current_clearance = self.terrain_field._sample(
            self._clearance_map, xy, padding_mode="border"
        )
        if vehicle_state is None or "heading" not in vehicle_state:
            heading = torch.zeros_like(reference)
            reliability = torch.zeros_like(reference)
        else:
            heading = _broadcast_state(vehicle_state["heading"], reference, "heading")
            reliability_value = vehicle_state.get("heading_reliability", 1.0)
            reliability = _broadcast_state(
                reliability_value, reference, "heading_reliability"
            ).clamp(0.0, 1.0)
        lookahead_distance = speed * self.config.heading_lookahead_time_s
        lookahead_xy = xy + torch.stack(
            (lookahead_distance * torch.cos(heading), lookahead_distance * torch.sin(heading)),
            dim=-1,
        )
        ahead_clearance = self.terrain_field._sample(
            self._clearance_map, lookahead_xy, padding_mode="border"
        )
        effective_clearance = (
            (1.0 - reliability) * current_clearance
            + reliability * torch.minimum(current_clearance, ahead_clearance)
        )
        required_clearance = (
            self.config.base_clearance_requirement_m
            + self.config.speed_clearance_gain_m * response
        )
        clearance_barrier = torch.sigmoid(
            (required_clearance - effective_clearance)
            / self.config.clearance_softness_m
        )
        slope_addition = self.config.slope_speed_gain * response * base_components["slope"]
        roughness_addition = (
            self.config.roughness_speed_gain * response * base_components["roughness"]
        )
        clearance_addition = (
            self.config.clearance_speed_gain * response * clearance_barrier
        )
        return {
            "speed_response": response,
            "effective_clearance_m": effective_clearance,
            "required_clearance_m": required_clearance,
            "clearance_barrier": clearance_barrier,
            "slope_speed_addition": slope_addition,
            "roughness_speed_addition": roughness_addition,
            "clearance_speed_addition": clearance_addition,
            "vehicle_additional_cost": (
                slope_addition + roughness_addition + clearance_addition
            ),
        }

    def component_costs(
        self,
        xyz_or_xy: torch.Tensor,
        vehicle_state: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return terrain components plus motion-conditioned additions."""

        components = self.terrain_field.component_costs(xyz_or_xy[..., :2])
        components.update(self._vehicle_components(xyz_or_xy, vehicle_state))
        return components

    def cost(
        self,
        xyz_or_xy: torch.Tensor,
        vehicle_state: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        vehicle = self._vehicle_components(xyz_or_xy, vehicle_state)
        return self.terrain_field.cost(xyz_or_xy) + vehicle["vehicle_additional_cost"]

    def query(
        self,
        xyz_or_xy: torch.Tensor,
        vehicle_state: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        vehicle = self._vehicle_components(xyz_or_xy, vehicle_state)
        terrain_feasibility = self.terrain_field.query(xyz_or_xy)
        return (
            terrain_feasibility * torch.exp(-vehicle["vehicle_additional_cost"])
        ).clamp(0.0, 1.0)


class BatchedVehicleConditionedTerrainField(BaseTerrainField):
    """Differentiable vehicle-conditioned wrapper for batched BEV fields.

    The cached training BEV contains occupancy proximity rather than a metric
    Euclidean clearance map.  Consequently the clearance component below is a
    documented *proximity-cost proxy*: heading look-ahead blends the current
    and forward sampled clearance costs, and speed increases its contribution.
    Metric clearance evaluation must continue to use
    :class:`VehicleConditionedTerrainField` built from raw LiDAR geometry.
    """

    def __init__(
        self,
        terrain_field: AnalyticTerrainField,
        config: VehicleConditionedFieldConfig | None = None,
    ) -> None:
        self.terrain_field = terrain_field
        self.config = config or VehicleConditionedFieldConfig()

    def _vehicle_components(
        self,
        xyz_or_xy: torch.Tensor,
        vehicle_state: dict[str, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor]:
        xy = xyz_or_xy[..., :2]
        reference = xy[..., 0]
        if vehicle_state is None or "speed" not in vehicle_state:
            speed = torch.zeros_like(reference)
        else:
            speed = _broadcast_state(vehicle_state["speed"], reference, "speed")
            if bool((speed < 0.0).any()):
                raise ValueError("vehicle speed must be non-negative")
        response = (
            speed.div(self.config.speed_reference_mps)
            .clamp(0.0, self.config.maximum_speed_ratio)
            .pow(self.config.speed_response_exponent)
        )
        base = self.terrain_field.component_costs(xy)
        current_clearance_cost = base["clearance"]
        if vehicle_state is None or "heading" not in vehicle_state:
            heading = torch.zeros_like(reference)
            reliability = torch.zeros_like(reference)
        else:
            heading = _broadcast_state(vehicle_state["heading"], reference, "heading")
            reliability = _broadcast_state(
                vehicle_state.get("heading_reliability", 1.0),
                reference,
                "heading_reliability",
            ).clamp(0.0, 1.0)
        lookahead_distance = speed * self.config.heading_lookahead_time_s
        lookahead_xy = xy + torch.stack(
            (
                lookahead_distance * torch.cos(heading),
                lookahead_distance * torch.sin(heading),
            ),
            dim=-1,
        )
        ahead_clearance_cost = self.terrain_field._sample(
            self.terrain_field.components["clearance"], lookahead_xy
        )
        effective_clearance_cost = (
            (1.0 - reliability) * current_clearance_cost
            + reliability * torch.maximum(current_clearance_cost, ahead_clearance_cost)
        )
        slope_addition = self.config.slope_speed_gain * response * base["slope"]
        roughness_addition = (
            self.config.roughness_speed_gain * response * base["roughness"]
        )
        clearance_addition = (
            self.config.clearance_speed_gain * response * effective_clearance_cost
        )
        return {
            "speed_response": response,
            "clearance_proximity_cost": effective_clearance_cost,
            "slope_speed_addition": slope_addition,
            "roughness_speed_addition": roughness_addition,
            "clearance_speed_addition": clearance_addition,
            "vehicle_additional_cost": (
                slope_addition + roughness_addition + clearance_addition
            ),
        }

    def component_costs(
        self,
        xyz_or_xy: torch.Tensor,
        vehicle_state: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return analytic terrain components and speed-conditioned additions."""

        components = self.terrain_field.component_costs(xyz_or_xy[..., :2])
        components.update(self._vehicle_components(xyz_or_xy, vehicle_state))
        return components

    def cost(
        self,
        xyz_or_xy: torch.Tensor,
        vehicle_state: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Return terrain cost plus the unnormalized vehicle motion addition."""

        vehicle = self._vehicle_components(xyz_or_xy, vehicle_state)
        return self.terrain_field.cost(xyz_or_xy) + vehicle["vehicle_additional_cost"]

    def query(
        self,
        xyz_or_xy: torch.Tensor,
        vehicle_state: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Return normalized feasibility in ``[0,1]`` for batched trajectories."""

        vehicle = self._vehicle_components(xyz_or_xy, vehicle_state)
        return (
            self.terrain_field.query(xyz_or_xy)
            * torch.exp(-vehicle["vehicle_additional_cost"])
        ).clamp(0.0, 1.0)

"""Differentiable 2.5D terrain and vehicle-conditioned feasibility field."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from TerraFlow.interfaces import BaseTerrainField
from TerraFlow.terrain.terrain_features import (
    TerrainFeatureConfig,
    TerrainFeatures,
    TerrainGridSpec,
)


@dataclass(frozen=True)
class SemanticClassPolicy:
    """Documented initial cost assigned to one exact semantic label ID."""

    name: str
    cost: float
    role: str
    rationale: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.cost <= 1.0:
            raise ValueError("semantic costs must be in [0, 1]")
        if self.role not in {"ground", "obstacle", "unknown", "ignore"}:
            raise ValueError(f"unsupported semantic role: {self.role}")
        if not self.rationale.strip():
            raise ValueError("every semantic cost requires a rationale")


@dataclass(frozen=True)
class ContinuousTerrainFieldConfig:
    """Weights and physical reference values for the continuous cost field."""

    occupancy_weight: float = 3.0
    slope_weight: float = 1.2
    roughness_weight: float = 0.8
    semantic_weight: float = 1.0
    clearance_weight: float = 1.0
    unknown_weight: float = 0.8
    slope_reference_deg: float = 25.0
    roughness_reference_m: float = 0.2
    clearance_reference_m: float = 2.0
    unknown_semantic_cost: float = 0.55
    spatial_smoothing_sigma_m: float = 0.0
    spatial_smoothing_truncate_sigma: float = 3.0
    semantic_classes: Mapping[int, SemanticClassPolicy] | None = None

    def __post_init__(self) -> None:
        weights = (
            self.occupancy_weight,
            self.slope_weight,
            self.roughness_weight,
            self.semantic_weight,
            self.clearance_weight,
            self.unknown_weight,
        )
        if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
            raise ValueError("cost weights must be non-negative with a positive sum")
        if (
            self.slope_reference_deg <= 0.0
            or self.roughness_reference_m <= 0.0
            or self.clearance_reference_m <= 0.0
        ):
            raise ValueError("cost reference values must be positive")
        if not 0.0 <= self.unknown_semantic_cost <= 1.0:
            raise ValueError("unknown_semantic_cost must be in [0, 1]")
        if self.spatial_smoothing_sigma_m < 0.0:
            raise ValueError("spatial_smoothing_sigma_m must be non-negative")
        if self.spatial_smoothing_truncate_sigma <= 0.0:
            raise ValueError("spatial_smoothing_truncate_sigma must be positive")


@dataclass(frozen=True)
class TerrainFieldDefinition:
    """Configuration bundle shared by feature extraction and field queries."""

    feature: TerrainFeatureConfig
    cost: ContinuousTerrainFieldConfig
    semantic_provenance: str
    label_encoding: str


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"terrain-field section '{name}' must be an object")
    return value


def load_terrain_field_config(path: str | Path) -> TerrainFieldDefinition:
    """Load the documented semantic policy and numeric field hyperparameters."""

    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = _require_mapping(json.load(stream), "root")
    grid_raw = dict(_require_mapping(raw.get("grid", {}), "grid"))
    feature_raw = dict(_require_mapping(raw.get("features", {}), "features"))
    cost_raw = dict(_require_mapping(raw.get("cost", {}), "cost"))
    semantic_raw = _require_mapping(raw.get("semantic_policy", {}), "semantic_policy")
    class_raw = _require_mapping(semantic_raw.get("classes", {}), "semantic_policy.classes")
    policies: dict[int, SemanticClassPolicy] = {}
    for label_text, value in class_raw.items():
        entry = _require_mapping(value, f"semantic class {label_text}")
        policies[int(label_text)] = SemanticClassPolicy(
            name=str(entry["name"]),
            cost=float(entry["cost"]),
            role=str(entry["role"]),
            rationale=str(entry["rationale"]),
        )
    grid = TerrainGridSpec(**grid_raw)
    feature = TerrainFeatureConfig(
        grid=grid,
        semantic_obstacle_ids=tuple(
            sorted(label for label, policy in policies.items() if policy.role == "obstacle")
        ),
        ignored_semantic_ids=tuple(
            sorted(label for label, policy in policies.items() if policy.role == "ignore")
        ),
        **feature_raw,
    )
    cost = ContinuousTerrainFieldConfig(semantic_classes=policies, **cost_raw)
    provenance = str(semantic_raw.get("provenance", "")).strip()
    encoding = str(semantic_raw.get("label_encoding", "")).strip()
    if not provenance or not encoding:
        raise ValueError("semantic_policy must document provenance and label_encoding")
    return TerrainFieldDefinition(feature, cost, provenance, encoding)


class ContinuousTerrainField(BaseTerrainField):
    """Continuous metric terrain feasibility field built from LiDAR features.

    The discrete component maps are normalized to ``[0, 1]`` and combined as

    ``C = w_occ*C_occ + w_slope*C_slope + w_rough*C_rough +``
    ``w_sem*C_sem + w_clear*C_clearance + w_unknown*C_unknown``.

    The returned feasibility is ``exp(-C)`` and is therefore bounded in
    ``[0, 1]`` with larger values indicating more feasible terrain. Queries
    bilinearly sample the final field at cell-centred metric coordinates.
    """

    def __init__(
        self,
        features: TerrainFeatures,
        config: ContinuousTerrainFieldConfig | None = None,
    ) -> None:
        self.features = features
        self.config = config or ContinuousTerrainFieldConfig()
        self.raw_components = self._build_components(features)
        self.components = {
            name: self._spatially_smooth(values)
            for name, values in self.raw_components.items()
        }
        cfg = self.config
        self.cost_map = (
            cfg.occupancy_weight * self.components["occupancy"]
            + cfg.slope_weight * self.components["slope"]
            + cfg.roughness_weight * self.components["roughness"]
            + cfg.semantic_weight * self.components["semantic"]
            + cfg.clearance_weight * self.components["clearance"]
            + cfg.unknown_weight * self.components["unknown"]
        ).clamp_min(0.0)
        self.feasibility_map = torch.exp(-self.cost_map).clamp(0.0, 1.0)

    def _spatially_smooth(self, value_map: torch.Tensor) -> torch.Tensor:
        """Apply a normalized separable Gaussian in metric grid space."""

        sigma_m = self.config.spatial_smoothing_sigma_m
        if sigma_m == 0.0:
            return value_map
        sigma_cells = sigma_m / self.features.grid.resolution_m
        radius = max(
            1,
            int(torch.ceil(torch.tensor(
                sigma_cells * self.config.spatial_smoothing_truncate_sigma
            )).item()),
        )
        offsets = torch.arange(
            -radius, radius + 1, device=value_map.device, dtype=value_map.dtype
        )
        kernel = torch.exp(-0.5 * (offsets / sigma_cells).square())
        kernel = kernel / kernel.sum()
        horizontal = kernel.reshape(1, 1, 1, -1)
        vertical = kernel.reshape(1, 1, -1, 1)
        padded = F.pad(value_map, (radius, radius, 0, 0), mode="replicate")
        smoothed = F.conv2d(padded, horizontal)
        padded = F.pad(smoothed, (0, 0, radius, radius), mode="replicate")
        return F.conv2d(padded, vertical)

    @staticmethod
    def _map(value: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return value.to(dtype=dtype).unsqueeze(0).unsqueeze(0)

    def _build_components(self, features: TerrainFeatures) -> dict[str, torch.Tensor]:
        cfg = self.config
        occupancy = self._map(features.occupancy).clamp(0.0, 1.0)
        slope = self._map(torch.nan_to_num(features.slope_deg, nan=0.0))
        slope = (slope / cfg.slope_reference_deg).clamp(0.0, 1.0)
        roughness = self._map(torch.nan_to_num(features.roughness_m, nan=0.0))
        roughness = (roughness / cfg.roughness_reference_m).clamp(0.0, 1.0)
        clearance = self._map(features.clearance_m)
        clearance = (1.0 - clearance / cfg.clearance_reference_m).clamp(0.0, 1.0)
        semantic_ids = features.semantic_class.to(dtype=torch.long)
        semantic_cost = torch.full_like(
            features.elevation_m, cfg.unknown_semantic_cost, dtype=torch.float32
        )
        for label, policy in (cfg.semantic_classes or {}).items():
            semantic_cost = torch.where(
                semantic_ids == int(label),
                torch.as_tensor(policy.cost, dtype=torch.float32, device=semantic_ids.device),
                semantic_cost,
            )
        unknown = (~features.geometry_valid).to(dtype=torch.float32)
        return {
            "occupancy": occupancy,
            "slope": slope,
            "roughness": roughness,
            "semantic": self._map(semantic_cost).clamp(0.0, 1.0),
            "clearance": clearance,
            "unknown": self._map(unknown),
        }

    def _metric_grid(self, xy: torch.Tensor) -> torch.Tensor:
        grid = self.features.grid
        x_index = (xy[..., 0] - (grid.x_min_m + 0.5 * grid.resolution_m)) / grid.resolution_m
        y_index = (xy[..., 1] - (grid.y_min_m + 0.5 * grid.resolution_m)) / grid.resolution_m
        x_normalized = (
            torch.zeros_like(x_index)
            if grid.width == 1
            else 2.0 * x_index / (grid.width - 1) - 1.0
        )
        y_normalized = (
            torch.zeros_like(y_index)
            if grid.height == 1
            else 2.0 * y_index / (grid.height - 1) - 1.0
        )
        return torch.stack((x_normalized, y_normalized), dim=-1)

    def _sample(
        self, value_map: torch.Tensor, xy: torch.Tensor, padding_mode: str = "zeros"
    ) -> torch.Tensor:
        if xy.shape[-1] not in (2, 3):
            raise ValueError("query coordinates must have trailing dimension 2 or 3")
        coordinates = xy[..., :2].to(device=value_map.device, dtype=value_map.dtype)
        original_shape = coordinates.shape[:-1]
        sample_grid = self._metric_grid(coordinates).reshape(1, -1, 1, 2)
        sampled = F.grid_sample(
            value_map,
            sample_grid,
            mode="bilinear",
            padding_mode=padding_mode,
            align_corners=True,
        )[0, 0, :, 0]
        return sampled.reshape(original_shape)

    def component_costs(self, xy: torch.Tensor) -> dict[str, torch.Tensor]:
        """Bilinearly query every spatially smoothed component cost."""

        return {
            name: self._sample(values, xy, padding_mode="border")
            for name, values in self.components.items()
        }

    def raw_component_costs(self, xy: torch.Tensor) -> dict[str, torch.Tensor]:
        """Query unsmoothed components for discrete violation accounting."""

        return {
            name: self._sample(values, xy, padding_mode="border")
            for name, values in self.raw_components.items()
        }

    def cost(
        self, xy: torch.Tensor, vehicle_state: dict[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        """Bilinearly query continuous terrain cost; vehicle state is reserved."""

        del vehicle_state
        return self._sample(self.cost_map, xy, padding_mode="border")

    def query(
        self, xy: torch.Tensor, vehicle_state: dict[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        """Return bilinearly interpolated feasibility for ``[...,2|3]`` points."""

        del vehicle_state
        return self._sample(self.feasibility_map, xy, padding_mode="zeros").clamp(0.0, 1.0)


@dataclass(frozen=True)
class TerrainFieldConfig:
    """Documented analytic cost weights and physical scaling."""

    forward_m: float = 24.0
    lateral_m: float = 12.0
    height_range_m: float = 4.5
    slope_reference: float = 0.35
    roughness_reference_m: float = 0.25
    occupancy_weight: float = 2.0
    traversability_weight: float = 1.2
    slope_weight: float = 0.8
    roughness_weight: float = 0.6
    clearance_weight: float = 0.8
    speed_reference_mps: float = 3.0
    cost_temperature: float = 3.0


class AnalyticTerrainField(BaseTerrainField):
    """Build and query continuous costs from RELLIS-3D BEV channels.

    Input channels are traversable fraction, obstacle density and normalized
    mean height. Unknown cells remain conservatively expensive through the
    traversability term. All query operations use bilinear ``grid_sample`` so
    gradients propagate to trajectory coordinates.
    """

    def __init__(self, terrain_map: torch.Tensor, config: TerrainFieldConfig | None = None):
        if terrain_map.ndim == 3:
            terrain_map = terrain_map.unsqueeze(0)
        if terrain_map.ndim != 4 or terrain_map.shape[1] != 3:
            raise ValueError("terrain_map must have shape [B, 3, H, W]")
        self.terrain_map = terrain_map
        self.config = config or TerrainFieldConfig()
        self.components = self._build_components(terrain_map)

    def _build_components(self, terrain_map: torch.Tensor) -> dict[str, torch.Tensor]:
        cfg = self.config
        traversable = terrain_map[:, 0:1].clamp(0.0, 1.0)
        occupancy = terrain_map[:, 1:2].clamp(0.0, 1.0)
        height = terrain_map[:, 2:3] * cfg.height_range_m
        cell_m = cfg.forward_m / terrain_map.shape[-2]
        kernel_x = torch.tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
            device=terrain_map.device, dtype=terrain_map.dtype,
        ).unsqueeze(0) / (8.0 * cell_m)
        kernel_y = kernel_x.transpose(-1, -2)
        dz_dx = F.conv2d(height, kernel_x, padding=1)
        dz_dy = F.conv2d(height, kernel_y, padding=1)
        slope = torch.sqrt(dz_dx.square() + dz_dy.square() + 1e-8)
        mean = F.avg_pool2d(height, kernel_size=5, stride=1, padding=2)
        mean_square = F.avg_pool2d(height.square(), kernel_size=5, stride=1, padding=2)
        roughness = torch.sqrt((mean_square - mean.square()).clamp_min(0.0) + 1e-8)
        clearance_cost = F.max_pool2d(occupancy, kernel_size=9, stride=1, padding=4)
        return {
            "occupancy": occupancy,
            "nontraversable": 1.0 - traversable,
            "slope": (slope / cfg.slope_reference).clamp(0.0, 1.0),
            "roughness": (roughness / cfg.roughness_reference_m).clamp(0.0, 1.0),
            "clearance": clearance_cost,
        }

    def _sample(self, value_map: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        original_shape = xy.shape[:-1]
        if xy.shape[0] != value_map.shape[0]:
            if value_map.shape[0] == 1:
                value_map = value_map.expand(xy.shape[0], -1, -1, -1)
            else:
                raise ValueError("xy and terrain batch dimensions differ")
        grid = torch.stack(
            [
                xy[..., 1] / self.config.lateral_m,
                xy[..., 0] / (self.config.forward_m / 2.0) - 1.0,
            ],
            dim=-1,
        ).reshape(xy.shape[0], -1, 1, 2)
        sampled = F.grid_sample(
            value_map, grid, mode="bilinear", padding_mode="border", align_corners=False
        )[:, 0, :, 0]
        return sampled.reshape(original_shape)

    def component_costs(self, xy: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: self._sample(values, xy) for name, values in self.components.items()}

    def cost(
        self, xyz: torch.Tensor, vehicle_state: dict[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        components = self.component_costs(xyz[..., :2])
        cfg = self.config
        slope_multiplier = 1.0
        roughness_multiplier = 1.0
        clearance_multiplier = 1.0
        if vehicle_state is not None and "speed" in vehicle_state:
            speed_ratio = (vehicle_state["speed"] / cfg.speed_reference_mps).clamp_min(0.0)
            slope_multiplier = 1.0 + speed_ratio
            roughness_multiplier = 1.0 + 0.5 * speed_ratio
            clearance_multiplier = 1.0 + 0.25 * speed_ratio
        numerator = (
            cfg.occupancy_weight * components["occupancy"]
            + cfg.traversability_weight * components["nontraversable"]
            + cfg.slope_weight * slope_multiplier * components["slope"]
            + cfg.roughness_weight * roughness_multiplier * components["roughness"]
            + cfg.clearance_weight * clearance_multiplier * components["clearance"]
        )
        denominator = (
            cfg.occupancy_weight + cfg.traversability_weight + cfg.slope_weight
            + cfg.roughness_weight + cfg.clearance_weight
        )
        return (numerator / denominator).clamp(0.0, 1.0)

    def query(
        self, xyz: torch.Tensor, vehicle_state: dict[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        return torch.exp(-self.config.cost_temperature * self.cost(xyz, vehicle_state))

"""Geometry and semantic features for a local ego-centric 2.5D grid.

This module only assigns axis meaning supplied by the caller. Raw RELLIS-3D
sensor coordinates must be transformed with an explicitly provided
``T_ego_sensor`` before they are described as ego-centric.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class TerrainGridSpec:
    """Metric bounds and resolution of a cell-centred ``[y, x]`` grid."""

    x_min_m: float = 0.0
    x_max_m: float = 24.0
    y_min_m: float = -12.0
    y_max_m: float = 12.0
    resolution_m: float = 0.25

    def __post_init__(self) -> None:
        if self.x_max_m <= self.x_min_m or self.y_max_m <= self.y_min_m:
            raise ValueError("grid maximum bounds must exceed minimum bounds")
        if self.resolution_m <= 0.0:
            raise ValueError("grid resolution must be positive")
        for span in (self.x_max_m - self.x_min_m, self.y_max_m - self.y_min_m):
            cells = span / self.resolution_m
            if not math.isclose(cells, round(cells), rel_tol=0.0, abs_tol=1e-6):
                raise ValueError("grid spans must be integer multiples of resolution_m")

    @property
    def width(self) -> int:
        """Number of columns along local ``x``."""

        return int(round((self.x_max_m - self.x_min_m) / self.resolution_m))

    @property
    def height(self) -> int:
        """Number of rows along local ``y``."""

        return int(round((self.y_max_m - self.y_min_m) / self.resolution_m))

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Matplotlib-compatible ``(xmin, xmax, ymin, ymax)`` bounds."""

        return self.x_min_m, self.x_max_m, self.y_min_m, self.y_max_m


@dataclass(frozen=True)
class TerrainFeatureConfig:
    """Quality thresholds used to aggregate points into grid features."""

    grid: TerrainGridSpec = TerrainGridSpec()
    minimum_points_per_cell: int = 3
    elevation_percentile: float = 10.0
    obstacle_height_threshold_m: float = 0.5
    minimum_obstacle_points: int = 2
    semantic_obstacle_min_fraction: float = 0.1
    maximum_clearance_m: float = 6.0
    semantic_obstacle_ids: tuple[int, ...] = ()
    ignored_semantic_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.minimum_points_per_cell <= 0 or self.minimum_obstacle_points <= 0:
            raise ValueError("point-count thresholds must be positive")
        if not 0.0 <= self.elevation_percentile <= 100.0:
            raise ValueError("elevation_percentile must be in [0, 100]")
        if self.obstacle_height_threshold_m <= 0.0 or self.maximum_clearance_m <= 0.0:
            raise ValueError("metric feature thresholds must be positive")
        if not 0.0 <= self.semantic_obstacle_min_fraction <= 1.0:
            raise ValueError("semantic_obstacle_min_fraction must be in [0, 1]")


@dataclass
class TerrainFeatures:
    """Aligned metric feature maps with shape ``[H_y, W_x]``.

    Unknown elevation, slope and roughness values are stored as NaN and are
    accompanied by explicit validity masks. Consumers must not interpret
    filled values used internally for finite differences as measurements.
    """

    grid: TerrainGridSpec
    elevation_m: torch.Tensor
    slope_deg: torch.Tensor
    roughness_m: torch.Tensor
    semantic_class: torch.Tensor
    occupancy: torch.Tensor
    clearance_m: torch.Tensor
    point_count: torch.Tensor
    geometry_valid: torch.Tensor
    slope_valid: torch.Tensor
    semantic_valid: torch.Tensor

    def __post_init__(self) -> None:
        expected = (self.grid.height, self.grid.width)
        fields = (
            self.elevation_m,
            self.slope_deg,
            self.roughness_m,
            self.semantic_class,
            self.occupancy,
            self.clearance_m,
            self.point_count,
            self.geometry_valid,
            self.slope_valid,
            self.semantic_valid,
        )
        if any(tuple(value.shape) != expected for value in fields):
            raise ValueError(f"every terrain feature must have shape {expected}")

    def to(self, *args: object, **kwargs: object) -> "TerrainFeatures":
        """Move all tensors while preserving integer and boolean dtypes."""

        def move(value: torch.Tensor) -> torch.Tensor:
            if value.is_floating_point():
                return value.to(*args, **kwargs)
            device = kwargs.get("device", args[0] if args else None)
            return value.to(device=device) if device is not None else value

        return TerrainFeatures(
            grid=self.grid,
            elevation_m=move(self.elevation_m),
            slope_deg=move(self.slope_deg),
            roughness_m=move(self.roughness_m),
            semantic_class=move(self.semantic_class),
            occupancy=move(self.occupancy),
            clearance_m=move(self.clearance_m),
            point_count=move(self.point_count),
            geometry_valid=move(self.geometry_valid),
            slope_valid=move(self.slope_valid),
            semantic_valid=move(self.semantic_valid),
        )


def transform_points(points_xyz: np.ndarray, target_from_source: np.ndarray) -> np.ndarray:
    """Apply an explicit homogeneous transform to point coordinates.

    ``target_from_source`` is defined solely by the caller: multiplying a
    homogeneous source point produces coordinates in the target frame. No
    RELLIS-3D transform direction or axis convention is inferred here.
    """

    points = np.asarray(points_xyz, dtype=np.float64)
    transform = np.asarray(target_from_source, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape [N, 3]")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("target_from_source must be a finite 4x4 matrix")
    if not np.allclose(transform[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-6):
        raise ValueError("homogeneous transform must end with [0, 0, 0, 1]")
    homogeneous = np.concatenate(
        (points, np.ones((len(points), 1), dtype=np.float64)), axis=1
    )
    return (homogeneous @ transform.T)[:, :3].astype(np.float32)


def _nearest_valid_fill(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill missing cells from their nearest 8-connected valid source."""

    if not valid.any():
        return np.zeros_like(values, dtype=np.float32)
    height, width = values.shape
    distance = np.full((height, width), np.inf, dtype=np.float64)
    filled = np.asarray(values, dtype=np.float32).copy()
    queue: list[tuple[float, int, int, float]] = []
    for row, col in np.argwhere(valid):
        distance[row, col] = 0.0
        heapq.heappush(queue, (0.0, int(row), int(col), float(values[row, col])))
    neighbours = (
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
    )
    while queue:
        current, row, col, source_value = heapq.heappop(queue)
        if current > distance[row, col]:
            continue
        filled[row, col] = source_value
        for drow, dcol, step in neighbours:
            next_row, next_col = row + drow, col + dcol
            if 0 <= next_row < height and 0 <= next_col < width:
                candidate = current + step
                if candidate < distance[next_row, next_col]:
                    distance[next_row, next_col] = candidate
                    heapq.heappush(
                        queue, (candidate, next_row, next_col, source_value)
                    )
    return filled


def obstacle_clearance(
    occupancy: np.ndarray, resolution_m: float, maximum_clearance_m: float
) -> np.ndarray:
    """Compute an 8-connected Euclidean-distance approximation to obstacles."""

    occupied = np.asarray(occupancy, dtype=bool)
    height, width = occupied.shape
    distance = np.full((height, width), np.inf, dtype=np.float64)
    queue: list[tuple[float, int, int]] = []
    for row, col in np.argwhere(occupied):
        distance[row, col] = 0.0
        heapq.heappush(queue, (0.0, int(row), int(col)))
    if not queue:
        return np.full((height, width), maximum_clearance_m, dtype=np.float32)
    neighbours = (
        (-1, 0, resolution_m), (1, 0, resolution_m),
        (0, -1, resolution_m), (0, 1, resolution_m),
        (-1, -1, resolution_m * math.sqrt(2.0)),
        (-1, 1, resolution_m * math.sqrt(2.0)),
        (1, -1, resolution_m * math.sqrt(2.0)),
        (1, 1, resolution_m * math.sqrt(2.0)),
    )
    while queue:
        current, row, col = heapq.heappop(queue)
        if current > distance[row, col] or current >= maximum_clearance_m:
            continue
        for drow, dcol, step in neighbours:
            next_row, next_col = row + drow, col + dcol
            candidate = current + step
            if (
                0 <= next_row < height
                and 0 <= next_col < width
                and candidate < distance[next_row, next_col]
            ):
                distance[next_row, next_col] = candidate
                heapq.heappush(queue, (candidate, next_row, next_col))
    return np.minimum(distance, maximum_clearance_m).astype(np.float32)


def _neighbour_support(valid: np.ndarray) -> np.ndarray:
    padded = np.pad(valid.astype(np.int16), 1)
    support = np.zeros_like(valid, dtype=np.int16)
    for row_offset in range(3):
        for col_offset in range(3):
            support += padded[
                row_offset : row_offset + valid.shape[0],
                col_offset : col_offset + valid.shape[1],
            ]
    return support


def build_terrain_features(
    points_xyz: np.ndarray | torch.Tensor,
    semantic_labels: np.ndarray | torch.Tensor | None = None,
    config: TerrainFeatureConfig | None = None,
) -> TerrainFeatures:
    """Aggregate local-frame LiDAR geometry and optional point labels.

    Elevation is a configurable low percentile, roughness is the within-cell
    height standard deviation, slope is the magnitude of the elevation
    gradient in degrees, and occupancy is the union of geometry and configured
    obstacle semantics. The semantic class is the per-cell exact-ID mode after
    ignored labels are removed.
    """

    cfg = config or TerrainFeatureConfig()
    points = np.asarray(
        points_xyz.detach().cpu().numpy() if isinstance(points_xyz, torch.Tensor) else points_xyz,
        dtype=np.float32,
    )
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points_xyz must have shape [N, >=3]")
    points = points[:, :3]
    labels: np.ndarray | None = None
    if semantic_labels is not None:
        labels = np.asarray(
            semantic_labels.detach().cpu().numpy()
            if isinstance(semantic_labels, torch.Tensor)
            else semantic_labels
        ).reshape(-1)
        if len(labels) != len(points):
            raise ValueError("semantic_labels must contain one value per point")
        labels = labels.astype(np.int64, copy=False)

    grid = cfg.grid
    finite = np.isfinite(points).all(axis=1)
    inside = (
        finite
        & (points[:, 0] >= grid.x_min_m)
        & (points[:, 0] < grid.x_max_m)
        & (points[:, 1] >= grid.y_min_m)
        & (points[:, 1] < grid.y_max_m)
    )
    points = points[inside]
    if labels is not None:
        labels = labels[inside]

    shape = (grid.height, grid.width)
    elevation = np.full(shape, np.nan, dtype=np.float32)
    roughness = np.full(shape, np.nan, dtype=np.float32)
    semantic = np.full(shape, -1, dtype=np.int64)
    point_count = np.zeros(shape, dtype=np.int32)
    geometry_obstacle = np.zeros(shape, dtype=bool)
    semantic_obstacle = np.zeros(shape, dtype=bool)

    if len(points):
        cols = np.floor((points[:, 0] - grid.x_min_m) / grid.resolution_m).astype(np.int64)
        rows = np.floor((points[:, 1] - grid.y_min_m) / grid.resolution_m).astype(np.int64)
        # Float32 subtraction/division can round a value that passed the
        # half-open metric bound to exactly ``width`` or ``height``. Apply the
        # authoritative integer-grid bounds as a second conservative filter.
        index_inside = (
            (cols >= 0)
            & (cols < grid.width)
            & (rows >= 0)
            & (rows < grid.height)
        )
        if not np.all(index_inside):
            points = points[index_inside]
            cols = cols[index_inside]
            rows = rows[index_inside]
            if labels is not None:
                labels = labels[index_inside]
        linear = rows * grid.width + cols
        order = np.argsort(linear, kind="stable")
        sorted_linear = linear[order]
        if len(order):
            boundaries = np.flatnonzero(np.diff(sorted_linear)) + 1
            starts = np.concatenate(([0], boundaries))
            ends = np.concatenate((boundaries, [len(order)]))
        else:
            starts = ends = np.empty(0, dtype=np.int64)
        obstacle_ids = set(cfg.semantic_obstacle_ids)
        ignored_ids = set(cfg.ignored_semantic_ids)
        for start, end in zip(starts, ends):
            indices = order[start:end]
            cell = int(sorted_linear[start])
            row, col = divmod(cell, grid.width)
            heights = points[indices, 2]
            count = len(indices)
            point_count[row, col] = count
            base_height = float(np.percentile(heights, cfg.elevation_percentile))
            elevation[row, col] = base_height
            roughness[row, col] = float(np.std(heights, ddof=0))
            elevated_count = int(
                np.count_nonzero(heights > base_height + cfg.obstacle_height_threshold_m)
            )
            geometry_obstacle[row, col] = elevated_count >= cfg.minimum_obstacle_points
            if labels is not None:
                cell_labels = labels[indices]
                kept = np.asarray(
                    [value for value in cell_labels if int(value) not in ignored_ids],
                    dtype=np.int64,
                )
                if len(kept):
                    values, counts = np.unique(kept, return_counts=True)
                    semantic[row, col] = int(values[np.argmax(counts)])
                    obstacle_fraction = float(
                        np.count_nonzero(np.isin(kept, tuple(obstacle_ids))) / len(kept)
                    ) if obstacle_ids else 0.0
                    semantic_obstacle[row, col] = (
                        obstacle_fraction >= cfg.semantic_obstacle_min_fraction
                    )

    geometry_valid = point_count >= cfg.minimum_points_per_cell
    semantic_valid = semantic >= 0
    filled_elevation = _nearest_valid_fill(elevation, geometry_valid)
    dz_dy, dz_dx = np.gradient(filled_elevation, grid.resolution_m, edge_order=1)
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy))).astype(np.float32)
    slope_valid = geometry_valid & (_neighbour_support(geometry_valid) >= 3)
    slope[~slope_valid] = np.nan
    occupancy = (geometry_obstacle | semantic_obstacle).astype(np.float32)
    clearance = obstacle_clearance(
        occupancy > 0.5, grid.resolution_m, cfg.maximum_clearance_m
    )

    return TerrainFeatures(
        grid=grid,
        elevation_m=torch.from_numpy(elevation),
        slope_deg=torch.from_numpy(slope),
        roughness_m=torch.from_numpy(roughness),
        semantic_class=torch.from_numpy(semantic),
        occupancy=torch.from_numpy(occupancy),
        clearance_m=torch.from_numpy(clearance),
        point_count=torch.from_numpy(point_count.astype(np.int64)),
        geometry_valid=torch.from_numpy(geometry_valid),
        slope_valid=torch.from_numpy(slope_valid),
        semantic_valid=torch.from_numpy(semantic_valid),
    )


def stack_feature_names() -> Iterable[str]:
    """Return stable serialized feature names for audit tooling."""

    return (
        "elevation_m", "slope_deg", "roughness_m", "semantic_class",
        "occupancy", "clearance_m", "point_count", "geometry_valid",
        "slope_valid", "semantic_valid",
    )

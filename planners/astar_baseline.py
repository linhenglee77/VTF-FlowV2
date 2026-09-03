"""A* baseline for VTF-Flow local 2D/2.5D terrain grids."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable

import torch

from TerraFlow.interfaces import BasePlanner, SceneBatch, TrajectoryBatch
from TerraFlow.planners.local_path_baseline import resample_polyline


class AStarPlanningError(RuntimeError):
    """Raised when the supplied map and goal do not admit a valid A* path."""


@dataclass(frozen=True)
class AStarConfig:
    """Grid geometry, obstacle rules, and A* cost weights."""

    horizon: int = 30
    forward_extent_m: float = 24.0
    lateral_extent_m: float = 12.0
    traversability_channel: int = 0
    occupancy_channel: int = 1
    elevation_channel: int = 2
    occupancy_threshold: float = 0.5
    minimum_traversability: float = 0.05
    forbid_occupied: bool = True
    forbid_nontraversable: bool = True
    occupancy_cost_weight: float = 4.0
    nontraversable_cost_weight: float = 2.0
    allow_diagonal: bool = True
    allow_forbidden_start: bool = True
    height_min_m: float = -2.5
    height_range_m: float = 4.5

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.forward_extent_m <= 0.0 or self.lateral_extent_m <= 0.0:
            raise ValueError("map extents must be positive")
        if self.occupancy_cost_weight < 0.0 or self.nontraversable_cost_weight < 0.0:
            raise ValueError("cost weights must be non-negative")
        if self.height_range_m <= 0.0:
            raise ValueError("height_range_m must be positive")


GridIndex = tuple[int, int]


class AStarTerrainPlanner(BasePlanner):
    """Plan from the ego origin to an ego-frame goal on a terrain BEV grid.

    VTF-Flow map rows cover forward ``x in [0, forward_extent_m]`` and columns
    cover lateral ``y in [-lateral_extent_m, lateral_extent_m]``. The default
    channel semantics match the cached RELLIS BEV: traversability, occupancy,
    and height normalized from ``[-2.5, 2.0]`` metres.
    """

    def __init__(self, config: AStarConfig | None = None) -> None:
        super().__init__()
        self.config = config or AStarConfig()

    def _validate_channels(self, grid: torch.Tensor) -> None:
        required = max(
            self.config.traversability_channel,
            self.config.occupancy_channel,
            self.config.elevation_channel,
        )
        if grid.ndim != 3 or grid.shape[0] <= required:
            raise AStarPlanningError(
                f"terrain_map must have at least {required + 1} channels [C,H,W]"
            )
        if not grid.is_floating_point() or not torch.isfinite(grid).all():
            raise AStarPlanningError("terrain_map must be finite floating-point values")

    def _xy_to_index(self, xy: torch.Tensor, rows: int, columns: int) -> GridIndex:
        x, y = float(xy[0]), float(xy[1])
        cfg = self.config
        tolerance = 1e-6
        if (
            x < -tolerance
            or x > cfg.forward_extent_m + tolerance
            or y < -cfg.lateral_extent_m - tolerance
            or y > cfg.lateral_extent_m + tolerance
        ):
            raise AStarPlanningError(
                f"goal ({x:.3f}, {y:.3f}) m is outside the configured local map"
            )
        row = min(rows - 1, max(0, int(math.floor(x / cfg.forward_extent_m * rows))))
        column = min(
            columns - 1,
            max(
                0,
                int(
                    math.floor(
                        (y + cfg.lateral_extent_m)
                        / (2.0 * cfg.lateral_extent_m)
                        * columns
                    )
                ),
            ),
        )
        return row, column

    def _index_to_xy(self, index: GridIndex, rows: int, columns: int) -> tuple[float, float]:
        row, column = index
        x = (row + 0.5) * self.config.forward_extent_m / rows
        y = (
            (column + 0.5) * 2.0 * self.config.lateral_extent_m / columns
            - self.config.lateral_extent_m
        )
        return x, y

    def _neighbors(self, node: GridIndex, rows: int, columns: int) -> Iterable[tuple[GridIndex, float]]:
        moves = ((1, 0), (-1, 0), (0, 1), (0, -1))
        if self.config.allow_diagonal:
            moves += ((1, 1), (1, -1), (-1, 1), (-1, -1))
        for dr, dc in moves:
            neighbor = (node[0] + dr, node[1] + dc)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < columns:
                yield neighbor, math.sqrt(2.0) if dr and dc else 1.0

    def _is_forbidden(self, grid: torch.Tensor, node: GridIndex) -> bool:
        row, column = node
        occupied = float(grid[self.config.occupancy_channel, row, column])
        traversable = float(grid[self.config.traversability_channel, row, column])
        return (
            self.config.forbid_occupied and occupied >= self.config.occupancy_threshold
        ) or (
            self.config.forbid_nontraversable
            and traversable < self.config.minimum_traversability
        )

    def _astar(self, grid: torch.Tensor, start: GridIndex, goal: GridIndex) -> tuple[list[GridIndex], float]:
        if start == goal:
            return [start], 0.0
        if self._is_forbidden(grid, goal):
            raise AStarPlanningError("goal cell is occupied or non-traversable")
        rows, columns = grid.shape[-2:]

        def heuristic(node: GridIndex) -> float:
            dr, dc = goal[0] - node[0], goal[1] - node[1]
            return math.hypot(dr, dc)

        frontier: list[tuple[float, int, GridIndex]] = [(heuristic(start), 0, start)]
        serial = 0
        cost_so_far = {start: 0.0}
        parent: dict[GridIndex, GridIndex] = {}
        while frontier:
            _, _, current = heapq.heappop(frontier)
            if current == goal:
                path = [current]
                while current != start:
                    current = parent[current]
                    path.append(current)
                path.reverse()
                return path, cost_so_far[goal]
            for neighbor, move_cost in self._neighbors(current, rows, columns):
                if neighbor != start and self._is_forbidden(grid, neighbor):
                    continue
                row, column = neighbor
                occupancy = float(grid[self.config.occupancy_channel, row, column])
                nontraversable = 1.0 - float(
                    grid[self.config.traversability_channel, row, column]
                )
                terrain_multiplier = (
                    1.0
                    + self.config.occupancy_cost_weight * max(occupancy, 0.0)
                    + self.config.nontraversable_cost_weight * max(nontraversable, 0.0)
                )
                new_cost = cost_so_far[current] + move_cost * terrain_multiplier
                if new_cost < cost_so_far.get(neighbor, math.inf):
                    cost_so_far[neighbor] = new_cost
                    parent[neighbor] = current
                    serial += 1
                    heapq.heappush(
                        frontier,
                        (new_cost + heuristic(neighbor), serial, neighbor),
                    )
        raise AStarPlanningError("no path connects the ego origin to the goal")

    def _metric_path(
        self,
        grid: torch.Tensor,
        indices: list[GridIndex],
        goal: torch.Tensor,
        dimensions: int,
    ) -> torch.Tensor:
        rows, columns = grid.shape[-2:]
        values = []
        for index in indices:
            x, y = self._index_to_xy(index, rows, columns)
            height_normalized = float(grid[self.config.elevation_channel, index[0], index[1]])
            z = self.config.height_min_m + height_normalized * self.config.height_range_m
            values.append((x, y, z))
        device, dtype = goal.device, goal.dtype
        path = torch.tensor(values, dtype=dtype, device=device)
        # Use the exact continuous ego origin and requested goal around the grid path.
        path[0] = 0.0
        exact_goal = torch.zeros(3, dtype=dtype, device=device)
        exact_goal[: min(3, goal.numel())] = goal[:3]
        if torch.linalg.vector_norm(path[-1, :2] - exact_goal[:2]) > 1e-6:
            path = torch.cat((path, exact_goal[None]), dim=0)
        else:
            path[-1] = exact_goal
        if path.shape[0] == 1 or bool(torch.linalg.vector_norm(path[-1] - path[0]) <= 1e-7):
            sampled = path[-1:].expand(self.config.horizon, -1).clone()
        else:
            sampled = resample_polyline(path, self.config.horizon)
        if dimensions <= 3:
            return sampled[:, :dimensions]
        return torch.cat(
            (
                sampled,
                torch.zeros(
                    sampled.shape[0], dimensions - 3, dtype=dtype, device=device
                ),
            ),
            dim=-1,
        )

    def forward(self, scene: SceneBatch) -> TrajectoryBatch:
        """Return one A* candidate shaped ``[B,1,H,D]``."""

        scene = scene.as_batch()
        if scene.goal.shape[-1] < 2:
            raise AStarPlanningError("goal must provide at least ego-frame x and y")
        paths, scores = [], []
        dimensions = scene.gt_future.shape[-1]
        for batch_index in range(scene.batch_size):
            grid = scene.terrain_map[batch_index]
            self._validate_channels(grid)
            rows, columns = grid.shape[-2:]
            start = self._xy_to_index(grid.new_zeros(2), rows, columns)
            if self._is_forbidden(grid, start) and not self.config.allow_forbidden_start:
                raise AStarPlanningError(f"scene {batch_index} ego start cell is forbidden")
            goal = self._xy_to_index(scene.goal[batch_index, :2], rows, columns)
            try:
                indices, score = self._astar(grid.detach().cpu(), start, goal)
            except AStarPlanningError as error:
                raise AStarPlanningError(f"scene {batch_index}: {error}") from error
            paths.append(self._metric_path(grid, indices, scene.goal[batch_index], dimensions))
            scores.append(score)
        score_tensor = torch.tensor(
            scores, dtype=scene.gt_future.dtype, device=scene.gt_future.device
        ).unsqueeze(1)
        return TrajectoryBatch(
            trajectories=torch.stack(paths, dim=0).unsqueeze(1), scores=score_tensor
        )


AStarBaseline = AStarTerrainPlanner


__all__ = ["AStarBaseline", "AStarConfig", "AStarPlanningError", "AStarTerrainPlanner"]

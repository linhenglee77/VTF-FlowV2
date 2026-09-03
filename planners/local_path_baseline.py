"""Smooth local-route interpolation baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from TerraFlow.interfaces import BasePlanner, SceneBatch, TrajectoryBatch


class LocalPathUnavailableError(RuntimeError):
    """Raised when a scene does not provide a usable local path or route."""


@dataclass(frozen=True)
class LocalPathConfig:
    """Configuration for metadata route discovery and spline sampling."""

    horizon: int = 30
    metadata_keys: tuple[str, ...] = (
        "local_path",
        "future_route",
        "local_route",
        "route",
    )
    spline_samples_per_segment: int = 16
    prepend_ego_origin: bool = True
    origin_tolerance_m: float = 1e-4

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.spline_samples_per_segment < 2:
            raise ValueError("spline_samples_per_segment must be at least two")
        if not self.metadata_keys:
            raise ValueError("metadata_keys cannot be empty")


def _remove_consecutive_duplicates(path: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    if path.shape[0] <= 1:
        return path
    keep = torch.ones(path.shape[0], dtype=torch.bool, device=path.device)
    keep[1:] = torch.linalg.vector_norm(torch.diff(path, dim=0), dim=-1) > epsilon
    return path[keep]


def resample_polyline(
    path: torch.Tensor,
    points: int,
    *,
    include_start: bool = False,
) -> torch.Tensor:
    """Arc-length resample a finite ``[N,D]`` polyline.

    Planning trajectories exclude time zero by default, hence target distances
    start at ``1 / points`` of the path rather than duplicating the ego origin.
    """

    if path.ndim != 2 or path.shape[-1] < 2:
        raise ValueError("path must have shape [N,D] with D >= 2")
    if points <= 0:
        raise ValueError("points must be positive")
    if not path.is_floating_point() or not torch.isfinite(path).all():
        raise ValueError("path must be a finite floating-point tensor")
    path = _remove_consecutive_duplicates(path)
    if path.shape[0] < 2:
        raise ValueError("path must contain at least two distinct waypoints")
    lengths = torch.linalg.vector_norm(torch.diff(path, dim=0), dim=-1)
    cumulative = torch.cat((lengths.new_zeros(1), lengths.cumsum(dim=0)))
    if include_start:
        targets = torch.linspace(
            0.0, float(cumulative[-1]), points, dtype=path.dtype, device=path.device
        )
    else:
        targets = torch.arange(
            1, points + 1, dtype=path.dtype, device=path.device
        ) * (cumulative[-1] / points)
    right = torch.searchsorted(cumulative, targets, right=True).clamp(1, len(cumulative) - 1)
    left = right - 1
    alpha = (targets - cumulative[left]) / (cumulative[right] - cumulative[left]).clamp_min(
        torch.finfo(path.dtype).eps
    )
    return path[left] + alpha[:, None] * (path[right] - path[left])


def _catmull_rom(path: torch.Tensor, samples_per_segment: int) -> torch.Tensor:
    """Densely interpolate a path with an endpoint-preserving cubic spline."""

    if path.shape[0] < 3:
        return path
    chunks = []
    for index in range(path.shape[0] - 1):
        p0 = path[max(index - 1, 0)]
        p1 = path[index]
        p2 = path[index + 1]
        p3 = path[min(index + 2, path.shape[0] - 1)]
        t = torch.arange(
            samples_per_segment, dtype=path.dtype, device=path.device
        ) / samples_per_segment
        t = t[:, None]
        values = 0.5 * (
            2.0 * p1
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t.square()
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t.pow(3)
        )
        chunks.append(values)
    chunks.append(path[-1:])
    return torch.cat(chunks, dim=0)


class LocalPathPlanner(BasePlanner):
    """Fit a smooth spline through an explicitly supplied ego-frame route."""

    def __init__(self, config: LocalPathConfig | None = None) -> None:
        super().__init__()
        self.config = config or LocalPathConfig()

    def _metadata_items(self, metadata: Any, batch_size: int) -> list[Mapping[str, Any]]:
        if isinstance(metadata, Mapping):
            return [metadata] if batch_size == 1 else [metadata] * batch_size
        if isinstance(metadata, Sequence) and not isinstance(metadata, (str, bytes)):
            if len(metadata) != batch_size or not all(isinstance(item, Mapping) for item in metadata):
                raise LocalPathUnavailableError(
                    "batched metadata must contain one mapping per scene"
                )
            return list(metadata)
        raise LocalPathUnavailableError("scene metadata is not a route-bearing mapping")

    def _route_value(self, item: Mapping[str, Any], batch_index: int) -> Any:
        for key in self.config.metadata_keys:
            if key in item and item[key] is not None:
                return item[key]
        raise LocalPathUnavailableError(
            f"scene {batch_index} has no local route; checked keys {self.config.metadata_keys}"
        )

    def forward(self, scene: SceneBatch) -> TrajectoryBatch:
        """Return one spline candidate or fail explicitly when route data is absent."""

        scene = scene.as_batch()
        items = self._metadata_items(scene.metadata, scene.batch_size)
        dimensions = scene.gt_future.shape[-1]
        output = []
        for batch_index, item in enumerate(items):
            value = self._route_value(item, batch_index)
            try:
                route = torch.as_tensor(
                    value, dtype=scene.gt_future.dtype, device=scene.gt_future.device
                )
            except (TypeError, ValueError) as error:
                raise LocalPathUnavailableError(
                    f"scene {batch_index} route cannot be converted to a tensor: {error}"
                ) from error
            # A collator may retain route data as one tensor inside a shared
            # metadata mapping instead of a list of per-scene mappings.
            if route.ndim == 3:
                if route.shape[0] != scene.batch_size:
                    raise LocalPathUnavailableError(
                        "batched route tensor must have shape [B,N,D]"
                    )
                route = route[batch_index]
            if route.ndim != 2 or route.shape[0] < 2 or route.shape[1] < 2:
                raise LocalPathUnavailableError(
                    f"scene {batch_index} route must have shape [N,D], N >= 2, D >= 2"
                )
            if not torch.isfinite(route).all():
                raise LocalPathUnavailableError(f"scene {batch_index} route contains non-finite values")
            if route.shape[1] < dimensions:
                route = torch.cat(
                    (
                        route,
                        torch.zeros(
                            route.shape[0],
                            dimensions - route.shape[1],
                            dtype=route.dtype,
                            device=route.device,
                        ),
                    ),
                    dim=-1,
                )
            else:
                route = route[:, :dimensions]
            if self.config.prepend_ego_origin and bool(
                torch.linalg.vector_norm(route[0]).item() > self.config.origin_tolerance_m
            ):
                route = torch.cat((torch.zeros_like(route[:1]), route), dim=0)
            try:
                smooth = _catmull_rom(route, self.config.spline_samples_per_segment)
                output.append(resample_polyline(smooth, self.config.horizon))
            except ValueError as error:
                raise LocalPathUnavailableError(
                    f"scene {batch_index} route is unusable: {error}"
                ) from error
        return TrajectoryBatch(trajectories=torch.stack(output, dim=0).unsqueeze(1))


LocalPathBaseline = LocalPathPlanner


__all__ = [
    "LocalPathBaseline",
    "LocalPathConfig",
    "LocalPathPlanner",
    "LocalPathUnavailableError",
    "resample_polyline",
]

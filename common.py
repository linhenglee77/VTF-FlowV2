"""Shared typed containers used at VTF-Flow module boundaries."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
from torch import Tensor


@dataclass
class SceneBatch:
    """A batch of observations and supervision in the current ego frame.

    The container intentionally does not impose sensor-specific spatial shapes.
    This lets loaders represent fixed-size tensors, padded tensors, or sparse
    structures while keeping a consistent interface for planners.

    Attributes:
        ego_history: Historical ego states, normally shaped ``[B, T_hist, D_e]``.
        gt_future: Ground-truth future xyz positions, shaped ``[B, H, 3]``.
        goal: Goal position or goal state, normally shaped ``[B, D_g]``.
        point_cloud: Batched point-cloud features, normally ``[B, N, C_p]``.
        semantic_labels: Semantic annotations aligned with the point cloud or map.
        terrain_map: A raster, voxel, or learned terrain representation.
        metadata: Non-tensor batch information such as sequence and frame IDs.
    """

    ego_history: Tensor
    gt_future: Tensor
    goal: Tensor
    point_cloud: Tensor
    semantic_labels: Tensor
    terrain_map: Tensor
    metadata: Mapping[str, Any]

    @property
    def batch_size(self) -> int:
        """Return the leading batch dimension from ground-truth trajectories."""

        if self.gt_future.ndim < 1:
            raise ValueError("gt_future must have a leading batch dimension")
        return int(self.gt_future.shape[0])

    def to(self, *args: Any, **kwargs: Any) -> "SceneBatch":
        """Return a copy with all tensor fields moved via :meth:`Tensor.to`.

        Metadata is retained unchanged because it may contain strings and other
        non-tensor identifiers.
        """

        return SceneBatch(
            ego_history=self.ego_history.to(*args, **kwargs),
            gt_future=self.gt_future.to(*args, **kwargs),
            goal=self.goal.to(*args, **kwargs),
            point_cloud=self.point_cloud.to(*args, **kwargs),
            semantic_labels=self.semantic_labels.to(*args, **kwargs),
            terrain_map=self.terrain_map.to(*args, **kwargs),
            metadata=self.metadata,
        )


@dataclass
class TrajectoryBatch:
    """Candidate future trajectories and optional planner scores.

    Attributes:
        trajectories: Candidate trajectories shaped ``[B, K, H, D]``. Version
            one uses ``D = 3`` for ``(x, y, z)``.
        scores: Optional candidate scores shaped ``[B, K]``. A score's direction
            (higher-is-better or lower-is-better) is planner-specific and must be
            documented by the producing planner.
    """

    trajectories: Tensor
    scores: Optional[Tensor] = None

    def __post_init__(self) -> None:
        """Validate the stable batch-level shape contract."""

        if not isinstance(self.trajectories, torch.Tensor):
            raise TypeError("trajectories must be a torch.Tensor")
        if self.trajectories.ndim != 4:
            raise ValueError(
                "trajectories must have shape [B, K, H, D]; "
                f"received {tuple(self.trajectories.shape)}"
            )
        if self.scores is not None:
            if not isinstance(self.scores, torch.Tensor):
                raise TypeError("scores must be a torch.Tensor or None")
            expected = tuple(self.trajectories.shape[:2])
            if self.scores.ndim != 2 or tuple(self.scores.shape) != expected:
                raise ValueError(
                    f"scores must have shape [B, K] = {expected}; "
                    f"received {tuple(self.scores.shape)}"
                )

    @property
    def batch_size(self) -> int:
        """Number of scenes represented by the batch."""

        return int(self.trajectories.shape[0])

    @property
    def num_candidates(self) -> int:
        """Number of candidate trajectories per scene."""

        return int(self.trajectories.shape[1])

    def to(self, *args: Any, **kwargs: Any) -> "TrajectoryBatch":
        """Return a copy with tensor fields moved via :meth:`Tensor.to`."""

        return TrajectoryBatch(
            trajectories=self.trajectories.to(*args, **kwargs),
            scores=None if self.scores is None else self.scores.to(*args, **kwargs),
        )

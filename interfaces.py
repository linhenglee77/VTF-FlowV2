"""Stable public interfaces shared by VTF-Flow datasets, planners and metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn


@dataclass
class SceneBatch:
    """One scene or a batch of scenes in ego-centric coordinates.

    Tensor fields may be unbatched when returned by ``dataset[index]``. Calling
    :meth:`as_batch` adds the leading batch dimension without changing metadata.
    """

    ego_history: torch.Tensor
    gt_future: torch.Tensor
    goal: torch.Tensor
    point_cloud: torch.Tensor | None
    semantic_labels: torch.Tensor | None
    terrain_map: torch.Tensor
    metadata: Any
    vehicle_state: dict[str, torch.Tensor] | None = None

    @property
    def batch_size(self) -> int:
        """Return the explicit batch size, or one for an unbatched scene."""

        return int(self.goal.shape[0]) if self.goal.ndim > 1 else 1

    def as_batch(self) -> "SceneBatch":
        if self.goal.ndim > 1:
            return self
        return SceneBatch(
            ego_history=self.ego_history.unsqueeze(0),
            gt_future=self.gt_future.unsqueeze(0),
            goal=self.goal.unsqueeze(0),
            point_cloud=None if self.point_cloud is None else self.point_cloud.unsqueeze(0),
            semantic_labels=(
                None if self.semantic_labels is None else self.semantic_labels.unsqueeze(0)
            ),
            terrain_map=self.terrain_map.unsqueeze(0),
            metadata=[self.metadata],
            vehicle_state=(
                None
                if self.vehicle_state is None
                else {key: value.unsqueeze(0) for key, value in self.vehicle_state.items()}
            ),
        )

    def to(self, *args: Any, **kwargs: Any) -> "SceneBatch":
        """Move or convert tensor fields using standard ``Tensor.to`` arguments."""

        def move(value):
            return None if value is None else value.to(*args, **kwargs)

        return replace(
            self,
            ego_history=move(self.ego_history),
            gt_future=move(self.gt_future),
            goal=move(self.goal),
            point_cloud=move(self.point_cloud),
            semantic_labels=move(self.semantic_labels),
            terrain_map=move(self.terrain_map),
            vehicle_state=(
                None
                if self.vehicle_state is None
                else {key: move(value) for key, value in self.vehicle_state.items()}
            ),
        )


@dataclass
class TrajectoryBatch:
    """Multimodal trajectory prediction with shape ``[B, K, H, D]``."""

    trajectories: torch.Tensor
    scores: torch.Tensor | None = None
    integration_history: torch.Tensor | None = None
    diagnostics: dict[str, torch.Tensor] | None = None

    def __post_init__(self) -> None:
        if self.trajectories.ndim != 4:
            raise ValueError("trajectories must have shape [B, K, H, D]")
        if self.scores is not None and self.scores.shape != self.trajectories.shape[:2]:
            raise ValueError("scores must have shape [B, K]")

    @property
    def batch_size(self) -> int:
        """Number of scenes in the prediction batch."""

        return int(self.trajectories.shape[0])

    @property
    def num_candidates(self) -> int:
        """Number of candidate trajectories per scene."""

        return int(self.trajectories.shape[1])


class BasePlanner(nn.Module, ABC):
    """Common planner interface: ``prediction = planner(scene)``."""

    @abstractmethod
    def forward(self, scene: SceneBatch) -> TrajectoryBatch:
        raise NotImplementedError


class BaseTerrainField(ABC):
    """Continuous terrain field interface."""

    @abstractmethod
    def query(
        self, xyz: torch.Tensor, vehicle_state: dict[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        raise NotImplementedError


class Evaluator(ABC):
    """Common evaluator interface."""

    @abstractmethod
    def __call__(self, prediction: TrajectoryBatch, scene: SceneBatch) -> dict[str, Any]:
        raise NotImplementedError

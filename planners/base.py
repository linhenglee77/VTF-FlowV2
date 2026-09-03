"""Stable planner interface."""

from abc import ABC, abstractmethod

from torch import nn

from ..common import SceneBatch, TrajectoryBatch


class BasePlanner(nn.Module, ABC):
    """Base class for all VTF-Flow trajectory planners.

    Subclasses implement :meth:`forward`; standard PyTorch call semantics then
    provide the public ``prediction = planner(scene)`` interface.
    """

    @abstractmethod
    def forward(self, scene: SceneBatch) -> TrajectoryBatch:
        """Generate candidate future trajectories for a scene batch.

        Args:
            scene: Batched observations and goals in ego-centric coordinates.

        Returns:
            Candidate trajectories shaped ``[B, K, H, D]`` and optional scores.
        """

        raise NotImplementedError

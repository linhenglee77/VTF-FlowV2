"""Stable terrain-field interface."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from torch import Tensor, nn


class BaseTerrainField(nn.Module, ABC):
    """Base class for terrain cost or feasibility fields.

    Implementations may return a cost (lower is better) or a feasibility value
    (higher is better), but must document that convention explicitly.
    """

    @abstractmethod
    def query(self, xyz: Tensor, vehicle_state: Optional[Any] = None) -> Tensor:
        """Evaluate terrain at arbitrary ego-frame xyz positions.

        Args:
            xyz: Query positions with trailing coordinate dimension ``3``.
            vehicle_state: Optional vehicle parameters or dynamic state used to
                condition traversability.

        Returns:
            Cost or feasibility values with implementation-defined feature axes.
        """

        raise NotImplementedError


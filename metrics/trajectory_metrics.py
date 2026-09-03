"""Vectorized metrics for multimodal trajectory predictions."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor


CoordinateSelection = Optional[Union[int, Sequence[int]]]


def _validate_inputs(prediction: Tensor, ground_truth: Tensor) -> None:
    if prediction.ndim != 4:
        raise ValueError(
            f"prediction must have shape [B, K, H, D], got {tuple(prediction.shape)}"
        )
    if ground_truth.ndim != 3:
        raise ValueError(
            f"ground_truth must have shape [B, H, D], got {tuple(ground_truth.shape)}"
        )
    if prediction.shape[0] != ground_truth.shape[0]:
        raise ValueError("prediction and ground_truth batch sizes differ")
    if prediction.shape[2] != ground_truth.shape[1]:
        raise ValueError("prediction and ground_truth horizons differ")
    if prediction.shape[1] < 1 or prediction.shape[2] < 1:
        raise ValueError("K and H must both be at least one")
    if not prediction.is_floating_point() or not ground_truth.is_floating_point():
        raise TypeError("prediction and ground_truth must be floating-point tensors")
    if not torch.isfinite(prediction).all() or not torch.isfinite(ground_truth).all():
        raise ValueError("prediction and ground_truth must contain only finite values")


def _coordinate_indices(
    prediction: Tensor, ground_truth: Tensor, coordinates: CoordinateSelection
) -> Tuple[int, ...]:
    shared_dimension = min(prediction.shape[-1], ground_truth.shape[-1])
    if coordinates is None:
        if shared_dimension < 1:
            raise ValueError("trajectory coordinate dimension must be positive")
        return tuple(range(shared_dimension))
    if isinstance(coordinates, int):
        if coordinates < 1 or coordinates > shared_dimension:
            raise ValueError(
                f"coordinates={coordinates} exceeds shared dimension {shared_dimension}"
            )
        return tuple(range(coordinates))
    indices = tuple(int(index) for index in coordinates)
    if not indices or min(indices) < 0 or max(indices) >= shared_dimension:
        raise ValueError("coordinate indices are empty or outside the shared dimension")
    return indices


def displacement_errors(
    prediction: Tensor,
    ground_truth: Tensor,
    coordinates: CoordinateSelection = None,
) -> Tensor:
    """Return waypoint Euclidean errors with shape ``[B, K, H]``.

    By default all coordinate channels shared by prediction and ground truth are
    used. Pass ``coordinates=2`` for an explicitly planar benchmark.
    """

    _validate_inputs(prediction, ground_truth)
    indices = _coordinate_indices(prediction, ground_truth, coordinates)
    pred = prediction[..., list(indices)]
    target = ground_truth[:, None, :, list(indices)]
    return torch.linalg.vector_norm(pred - target, dim=-1)


def average_displacement_error(
    prediction: Tensor,
    ground_truth: Tensor,
    coordinates: CoordinateSelection = None,
) -> Tensor:
    """Return ADE for every candidate with shape ``[B, K]``."""

    return displacement_errors(prediction, ground_truth, coordinates).mean(dim=-1)


def final_displacement_error(
    prediction: Tensor,
    ground_truth: Tensor,
    coordinates: CoordinateSelection = None,
) -> Tensor:
    """Return FDE for every candidate with shape ``[B, K]``."""

    return displacement_errors(prediction, ground_truth, coordinates)[..., -1]


def trajectory_diversity(
    prediction: Tensor, coordinates: CoordinateSelection = None
) -> Tensor:
    """Mean pairwise candidate displacement, averaged over time, per scene.

    Returns shape ``[B]``. Diversity is defined as zero for ``K = 1``.
    """

    if prediction.ndim != 4:
        raise ValueError("prediction must have shape [B, K, H, D]")
    if not prediction.is_floating_point() or not torch.isfinite(prediction).all():
        raise ValueError("prediction must be a finite floating-point tensor")
    indices = _coordinate_indices(prediction, prediction, coordinates)
    batch_size, candidate_count = prediction.shape[:2]
    if candidate_count == 1:
        return prediction.new_zeros(batch_size)
    values = prediction[..., list(indices)]
    pairwise = torch.linalg.vector_norm(
        values[:, :, None] - values[:, None, :], dim=-1
    ).mean(dim=-1)
    upper = torch.triu(
        torch.ones(
            candidate_count,
            candidate_count,
            dtype=torch.bool,
            device=prediction.device,
        ),
        diagonal=1,
    )
    return pairwise[:, upper].mean(dim=-1)


def trajectory_path_length(
    prediction: Tensor, coordinates: CoordinateSelection = None
) -> Tensor:
    """Sum distances between adjacent predicted waypoints, shape ``[B, K]``.

    The current origin is not part of ``[B,K,H,D]`` and therefore the
    origin-to-first-waypoint segment is not included.
    """

    if prediction.ndim != 4:
        raise ValueError("prediction must have shape [B, K, H, D]")
    indices = _coordinate_indices(prediction, prediction, coordinates)
    if prediction.shape[2] < 2:
        return prediction.new_zeros(prediction.shape[:2])
    segments = torch.diff(prediction[..., list(indices)], dim=2)
    return torch.linalg.vector_norm(segments, dim=-1).sum(dim=-1)


def trajectory_smoothness(
    prediction: Tensor, coordinates: CoordinateSelection = None
) -> Tensor:
    """Mean norm of second finite differences, shape ``[B, K]``.

    A straight constant-velocity waypoint sequence has zero smoothness penalty.
    Horizons shorter than three points return zero because no second difference
    exists.
    """

    if prediction.ndim != 4:
        raise ValueError("prediction must have shape [B, K, H, D]")
    indices = _coordinate_indices(prediction, prediction, coordinates)
    if prediction.shape[2] < 3:
        return prediction.new_zeros(prediction.shape[:2])
    second = torch.diff(prediction[..., list(indices)], n=2, dim=2)
    return torch.linalg.vector_norm(second, dim=-1).mean(dim=-1)


def trajectory_metrics(
    prediction: Tensor,
    ground_truth: Tensor,
    coordinates: CoordinateSelection = None,
) -> Dict[str, Tensor]:
    """Compute all initial trajectory metrics.

    Candidate-level outputs have shape ``[B,K]``. ``ADE_m`` and ``FDE_m`` use
    candidate zero and are standard ADE/FDE when ``K=1``. Oracle best-of-K
    outputs use ``minADE@K_m`` and ``minFDE@K_m``; aliases without ``@K`` are
    retained for compatibility with existing VTF-Flow reports.
    """

    errors = displacement_errors(prediction, ground_truth, coordinates)
    ade = errors.mean(dim=-1)
    fde = errors[..., -1]
    diversity = trajectory_diversity(prediction, coordinates)
    path_length = trajectory_path_length(prediction, coordinates)
    smoothness = trajectory_smoothness(prediction, coordinates)
    min_ade = ade.min(dim=1).values
    min_fde = fde.min(dim=1).values
    return {
        "ADE": ade[:, 0],
        "FDE": fde[:, 0],
        "minADE@K": min_ade,
        "minFDE@K": min_fde,
        "trajectory_diversity": diversity,
        "path_length": path_length,
        "smoothness": smoothness,
        "ADE_by_candidate_m": ade,
        "FDE_by_candidate_m": fde,
        "ADE_m": ade[:, 0],
        "FDE_m": fde[:, 0],
        "minADE@K_m": min_ade,
        "minFDE@K_m": min_fde,
        "minADE_m": min_ade,
        "minFDE_m": min_fde,
        "diversity_m": diversity,
        "path_length_by_candidate_m": path_length,
        "smoothness_by_candidate_m": smoothness,
    }

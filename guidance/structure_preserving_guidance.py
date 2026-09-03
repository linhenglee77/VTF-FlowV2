"""Structure-preserving operators for inference-time feasibility correction."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


SmoothingKernel = Literal["none", "kernel_3", "kernel_5"]
TrustRegionScope = Literal["trajectory", "waypoint"]
EndpointProjection = Literal["none", "terminal", "affine"]


def smooth_trajectory_gradient(
    gradient: torch.Tensor,
    kernel: SmoothingKernel,
) -> torch.Tensor:
    """Smooth only along H in ``[N,H,D]`` using replicate padding."""

    if gradient.ndim != 3:
        raise ValueError("gradient must have shape [N,H,D]")
    if kernel == "none":
        return gradient
    coefficients = {
        "kernel_3": (0.25, 0.50, 0.25),
        "kernel_5": (0.0625, 0.25, 0.375, 0.25, 0.0625),
    }
    if kernel not in coefficients:
        raise ValueError("kernel must be 'none', 'kernel_3', or 'kernel_5'")
    values = torch.tensor(
        coefficients[kernel], device=gradient.device, dtype=gradient.dtype
    )
    radius = len(coefficients[kernel]) // 2
    channels = gradient.shape[-1]
    transposed = gradient.transpose(1, 2)
    padded = F.pad(transposed, (radius, radius), mode="replicate")
    weight = values.reshape(1, 1, -1).expand(channels, 1, -1)
    return F.conv1d(padded, weight, groups=channels).transpose(1, 2)


def project_goal_anchored_correction(
    correction: torch.Tensor,
    mode: EndpointProjection = "none",
) -> torch.Tensor:
    """Prevent feasibility guidance from moving the goal-anchored endpoint.

    ``terminal`` is the orthogonal projection onto the constraint that the
    final waypoint correction is zero. ``affine`` removes a linearly
    increasing copy of the terminal correction from the complete trajectory;
    it also fixes the endpoint while avoiding an abrupt terminal-only change.
    The operation uses no ground-truth intermediate waypoint.
    """

    if correction.ndim != 3:
        raise ValueError("correction must have shape [N,H,D]")
    if mode not in {"none", "terminal", "affine"}:
        raise ValueError("mode must be 'none', 'terminal', or 'affine'")
    if mode == "none":
        return correction
    if mode == "terminal":
        projected = correction.clone()
        projected[:, -1] = 0.0
        return projected
    horizon = correction.shape[1]
    fraction = torch.arange(
        1, horizon + 1, device=correction.device, dtype=correction.dtype
    ).reshape(1, horizon, 1) / float(horizon)
    return correction - fraction * correction[:, -1:, :]


def apply_relative_trust_region(
    correction: torch.Tensor,
    flow_velocity: torch.Tensor,
    rho: float | None,
    scope: TrustRegionScope = "trajectory",
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Bound correction norm by ``rho * ||v_theta||`` per sample/waypoint."""

    if correction.shape != flow_velocity.shape or correction.ndim != 3:
        raise ValueError("correction and flow_velocity must share shape [N,H,D]")
    if rho is not None and rho <= 0.0:
        raise ValueError("rho must be positive or None")
    if scope not in {"trajectory", "waypoint"}:
        raise ValueError("scope must be 'trajectory' or 'waypoint'")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if scope == "trajectory":
        dimensions = (1, 2)
        correction_norm = torch.linalg.vector_norm(
            correction.flatten(start_dim=1), dim=1
        )
        flow_norm = torch.linalg.vector_norm(
            flow_velocity.flatten(start_dim=1), dim=1
        )
        if rho is None:
            scale = torch.ones_like(correction_norm)
        else:
            scale = (rho * flow_norm / (correction_norm + epsilon)).clamp(max=1.0)
        applied = correction * scale[:, None, None]
        applied_norm = torch.linalg.vector_norm(applied.flatten(start_dim=1), dim=1)
    else:
        dimensions = (2,)
        correction_norm = torch.linalg.vector_norm(correction, dim=2)
        flow_norm = torch.linalg.vector_norm(flow_velocity, dim=2)
        if rho is None:
            scale = torch.ones_like(correction_norm)
        else:
            scale = (rho * flow_norm / (correction_norm + epsilon)).clamp(max=1.0)
        applied = correction * scale[..., None]
        applied_norm = torch.linalg.vector_norm(applied, dim=2)
    del dimensions
    ratio = applied_norm / (flow_norm + epsilon)
    return applied, {
        "trust_region_scale": scale,
        "pretrust_correction_norm": correction_norm,
        "applied_correction_norm": applied_norm,
        "flow_velocity_norm": flow_norm,
        "correction_flow_ratio": ratio,
    }


def adaptive_feasibility_trigger(
    cost: torch.Tensor,
    alpha: float,
    reference_cost: float,
) -> torch.Tensor:
    """Return a soft activation reference, not a calibrated safety decision."""

    if alpha <= 0.0:
        raise ValueError("trigger alpha must be positive")
    if not torch.isfinite(cost).all() or not torch.isfinite(
        torch.tensor(reference_cost)
    ):
        raise ValueError("cost and reference_cost must be finite")
    return torch.sigmoid(alpha * (cost - reference_cost))


def flow_gradient_cosine_similarity(
    flow_velocity: torch.Tensor,
    cost_gradient: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Cosine similarity in flattened trajectory space for each sample."""

    if flow_velocity.shape != cost_gradient.shape or flow_velocity.ndim != 3:
        raise ValueError("inputs must share shape [N,H,D]")
    flow = flow_velocity.flatten(start_dim=1)
    gradient = cost_gradient.flatten(start_dim=1)
    numerator = (flow * gradient).sum(dim=1)
    denominator = (
        torch.linalg.vector_norm(flow, dim=1)
        * torch.linalg.vector_norm(gradient, dim=1)
    ).clamp_min(epsilon)
    return numerator / denominator


def paired_correction_metrics(
    guided: torch.Tensor,
    unguided: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return required per-scene paired structure correction metrics.

    Inputs are ``[B,K,H,D]``. Candidate-level quantities are averaged over K,
    while ``maximum_waypoint_correction_m`` retains the maximum over K and H.
    """

    if guided.shape != unguided.shape or guided.ndim != 4:
        raise ValueError("guided and unguided must share shape [B,K,H,D]")
    displacement = torch.linalg.vector_norm(guided - unguided, dim=-1)
    guided_second = torch.linalg.vector_norm(torch.diff(guided, n=2, dim=2), dim=-1)
    base_second = torch.linalg.vector_norm(torch.diff(unguided, n=2, dim=2), dim=-1)
    if guided.shape[2] < 3:
        guided_smoothness = guided.new_zeros(guided.shape[:2])
        base_smoothness = guided.new_zeros(guided.shape[:2])
        guided_max_second = guided.new_zeros(guided.shape[:2])
    else:
        guided_smoothness = guided_second.mean(dim=2)
        base_smoothness = base_second.mean(dim=2)
        guided_max_second = guided_second.max(dim=2).values
    return {
        "mean_waypoint_correction_m": displacement.mean(dim=(1, 2)),
        "maximum_waypoint_correction_m": displacement.amax(dim=(1, 2)),
        "mean_trajectory_max_correction_m": displacement.max(dim=2).values.mean(dim=1),
        "endpoint_correction_m": displacement[:, :, -1].mean(dim=1),
        "second_difference_change_m": (
            guided_smoothness - base_smoothness
        ).mean(dim=1),
        "maximum_local_second_difference_m": guided_max_second.mean(dim=1),
    }


__all__ = [
    "EndpointProjection",
    "SmoothingKernel",
    "TrustRegionScope",
    "adaptive_feasibility_trigger",
    "apply_relative_trust_region",
    "flow_gradient_cosine_similarity",
    "paired_correction_metrics",
    "project_goal_anchored_correction",
    "smooth_trajectory_gradient",
]

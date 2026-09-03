"""Unified trajectory and feasibility evaluator for VTF-Flow planners."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import Tensor

from TerraFlow.interfaces import BasePlanner, Evaluator, SceneBatch, TrajectoryBatch
from TerraFlow.metrics.feasibility_metrics import (
    FeasibilityMetricConfig,
    feasibility_metrics,
)
from TerraFlow.metrics.trajectory_metrics import trajectory_metrics


LatencyInput = Optional[Union[float, Tensor]]


@dataclass(frozen=True)
class EvaluatorConfig:
    """Metric dimensions, candidate selection, and feasibility thresholds."""

    coordinate_dimensions: Optional[int] = None
    scores_higher_is_better: bool = False
    feasibility: FeasibilityMetricConfig = FeasibilityMetricConfig()


def _latency_tensor(
    latency_ms: LatencyInput,
    batch_size: int,
    reference: Tensor,
) -> Tuple[Optional[Tensor], Dict[str, object]]:
    if latency_ms is None:
        return None, {
            "status": "unavailable",
            "reason": "inference latency was not recorded",
            "required_inputs": ["inference_latency_ms"],
            "unit": "ms/sample",
        }
    values = torch.as_tensor(latency_ms, dtype=reference.dtype, device=reference.device)
    if values.ndim == 0:
        values = values.repeat(batch_size)
    if values.ndim != 1 or values.shape[0] != batch_size:
        raise ValueError("inference_latency_ms must be a scalar or have shape [B]")
    if not torch.isfinite(values).all() or bool((values < 0).any()):
        raise ValueError("inference_latency_ms must be finite and non-negative")
    return values, {"status": "available", "unit": "ms/sample"}


class TerraFlowEvaluator(Evaluator):
    """Evaluate oracle best-of-K and score-selected deployable trajectories."""

    def __init__(self, config: Optional[EvaluatorConfig] = None) -> None:
        self.config = config or EvaluatorConfig()

    def evaluate_tensors(
        self,
        trajectories: Tensor,
        ground_truth: Tensor,
        scores: Optional[Tensor] = None,
        terrain_map: Optional[Tensor] = None,
        inference_latency_ms: LatencyInput = None,
    ) -> Dict[str, Any]:
        """Evaluate tensors directly, including optional terrain and latency.

        Args:
            trajectories: Predictions shaped ``[B,K,H,D]``.
            ground_truth: Targets shaped ``[B,H,D]``.
            scores: Optional candidate scores shaped ``[B,K]``.
            terrain_map: Optional documented VTF-Flow BEV map.
            inference_latency_ms: Scalar or per-scene ``[B]`` milliseconds.
        """

        metrics = trajectory_metrics(
            trajectories,
            ground_truth,
            coordinates=self.config.coordinate_dimensions,
        )
        batch_size, candidates = trajectories.shape[:2]
        if scores is not None:
            if scores.shape != (batch_size, candidates):
                raise ValueError("scores must have shape [B,K]")
            if not torch.isfinite(scores).all():
                raise ValueError("scores must contain only finite values")
            selected_index = (
                scores.argmax(dim=1)
                if self.config.scores_higher_is_better
                else scores.argmin(dim=1)
            )
            selection_policy = (
                "maximum saved score"
                if self.config.scores_higher_is_better
                else "minimum saved score"
            )
        else:
            selected_index = torch.zeros(
                batch_size, dtype=torch.long, device=trajectories.device
            )
            selection_policy = (
                "only candidate" if candidates == 1 else "candidate zero (scores unavailable)"
            )
        batch_index = torch.arange(batch_size, device=trajectories.device)

        result: Dict[str, Any] = {
            "ADE_m": metrics["ADE_by_candidate_m"][batch_index, selected_index],
            "FDE_m": metrics["FDE_by_candidate_m"][batch_index, selected_index],
            "minADE@K_m": metrics["minADE@K_m"],
            "minFDE@K_m": metrics["minFDE@K_m"],
            "minADE_m": metrics["minADE@K_m"],
            "minFDE_m": metrics["minFDE@K_m"],
            "diversity_m": metrics["diversity_m"],
            "path_length_m": metrics["path_length_by_candidate_m"][
                batch_index, selected_index
            ],
            "smoothness_m": metrics["smoothness_by_candidate_m"][
                batch_index, selected_index
            ],
            "selected_ADE_m": metrics["ADE_by_candidate_m"][
                batch_index, selected_index
            ],
            "selected_FDE_m": metrics["FDE_by_candidate_m"][
                batch_index, selected_index
            ],
            "selected_path_length_m": metrics["path_length_by_candidate_m"][
                batch_index, selected_index
            ],
            "selected_smoothness_m": metrics["smoothness_by_candidate_m"][
                batch_index, selected_index
            ],
            "ADE_by_candidate_m": metrics["ADE_by_candidate_m"],
            "FDE_by_candidate_m": metrics["FDE_by_candidate_m"],
            "path_length_by_candidate_m": metrics["path_length_by_candidate_m"],
            "smoothness_by_candidate_m": metrics["smoothness_by_candidate_m"],
            "selected_index": selected_index,
            "selection_policy": selection_policy,
        }

        feasibility = feasibility_metrics(
            trajectories,
            terrain_map=terrain_map,
            config=self.config.feasibility,
        )
        availability: Dict[str, Dict[str, object]] = {}
        for name, state in feasibility.items():
            public_state = {
                key: value for key, value in state.items() if key != "values"
            }
            availability[name] = public_state
            values = state.get("values")
            if state.get("status") == "available" and isinstance(values, Tensor):
                result[f"selected_{name}"] = values[batch_index, selected_index]
        result["feasibility"] = availability

        latency, latency_state = _latency_tensor(
            inference_latency_ms, batch_size, trajectories
        )
        result["inference_latency"] = latency_state
        if latency is not None:
            result["inference_latency_ms"] = latency
        return result

    def __call__(
        self,
        prediction: TrajectoryBatch,
        scene: SceneBatch,
        inference_latency_ms: LatencyInput = None,
    ) -> Dict[str, Any]:
        """Evaluate ``prediction`` against ``scene`` using the public interface."""

        if hasattr(scene, "as_batch"):
            scene = scene.as_batch()
        ground_truth = scene.gt_future
        if ground_truth.ndim == 2:
            ground_truth = ground_truth.unsqueeze(0)
        terrain_map = scene.terrain_map
        if terrain_map is not None and terrain_map.ndim == 3:
            terrain_map = terrain_map.unsqueeze(0)
        return self.evaluate_tensors(
            trajectories=prediction.trajectories,
            ground_truth=ground_truth,
            scores=prediction.scores,
            terrain_map=terrain_map,
            inference_latency_ms=inference_latency_ms,
        )


def timed_planner_call(
    planner: BasePlanner,
    scene: SceneBatch,
) -> Tuple[TrajectoryBatch, float]:
    """Run one planner call and return prediction plus milliseconds per scene.

    CUDA is synchronized immediately before and after timing so asynchronous
    kernels are included. This helper intentionally performs one call and does
    not add hidden warm-up inference.
    """

    device = scene.gt_future.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    prediction = planner(scene)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if hasattr(scene, "batch_size"):
        batch_size = int(scene.batch_size)
    else:
        batch_size = int(scene.gt_future.shape[0]) if scene.gt_future.ndim >= 3 else 1
    return prediction, elapsed_ms / max(batch_size, 1)

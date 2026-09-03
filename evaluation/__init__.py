"""Unified VTF-Flow evaluator."""

from typing import Union

from torch import Tensor

from .evaluator import EvaluatorConfig, TerraFlowEvaluator, timed_planner_call

MetricValue = Union[float, int, Tensor]

__all__ = [
    "EvaluatorConfig",
    "MetricValue",
    "TerraFlowEvaluator",
    "timed_planner_call",
]

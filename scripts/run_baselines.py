"""Run VTF-Flow trajectory baselines on a small cached RELLIS-3D subset.

Example:
    python scripts/run_baselines.py --cache-root ../data/RELLIS3D/trajectory_cache_h150_s5
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Mapping

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_PARENT = PROJECT_ROOT.parent
if str(WORKSPACE_PARENT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_PARENT))

from TerraFlow.datasets.rellis3d import Rellis3DSceneDataset  # noqa: E402
from TerraFlow.evaluation import EvaluatorConfig, TerraFlowEvaluator, timed_planner_call  # noqa: E402
from TerraFlow.metrics import FeasibilityMetricConfig  # noqa: E402
from TerraFlow.planners import (  # noqa: E402
    AStarConfig,
    AStarPlanningError,
    AStarTerrainPlanner,
    ConstantVelocityConfig,
    ConstantVelocityPlanner,
    LocalPathConfig,
    LocalPathPlanner,
    LocalPathUnavailableError,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "baselines" / "results.csv"
PLANNER_FAILURES = (AStarPlanningError, LocalPathUnavailableError, ValueError)


def _scalar(value: Any) -> float | str:
    """Convert a scalar tensor/value to a CSV-safe primitive."""

    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return ""
        return float(value.detach().cpu().item())
    if isinstance(value, (float, int)):
        return float(value)
    return ""


def _scene_identity(metadata: Any, dataset_index: int) -> dict[str, Any]:
    item: Mapping[str, Any] = metadata if isinstance(metadata, Mapping) else {}
    return {
        "dataset_index": dataset_index,
        "sequence_id": item.get("sequence", item.get("sequence_id", "")),
        "frame_id": item.get("frame_id", item.get("frame_index", "")),
    }


def _metric_columns(result: Mapping[str, Any]) -> dict[str, float | str]:
    columns: dict[str, float | str] = {
        "ADE_m": _scalar(result.get("ADE_m")),
        "FDE_m": _scalar(result.get("FDE_m")),
        "path_length_m": _scalar(result.get("path_length_m")),
        "smoothness_m": _scalar(result.get("smoothness_m")),
        "inference_latency_ms": _scalar(result.get("inference_latency_ms")),
    }
    feasibility = result.get("feasibility", {})
    if isinstance(feasibility, Mapping):
        for name, state in feasibility.items():
            if isinstance(state, Mapping) and state.get("status") == "available":
                columns[name] = _scalar(result.get(f"selected_{name}"))
    return columns


def build_planners(horizon: int, planning_dt_s: float) -> dict[str, torch.nn.Module]:
    """Construct config-driven baselines sharing the dataset trajectory horizon."""

    return {
        "constant_velocity": ConstantVelocityPlanner(
            ConstantVelocityConfig(
                horizon=horizon,
                planning_dt_s=planning_dt_s,
                # The legacy cache has one current state. This is deliberately
                # represented as a stationary constant-velocity fallback.
                stationary_fallback=True,
            )
        ),
        "local_path_spline": LocalPathPlanner(LocalPathConfig(horizon=horizon)),
        "astar_terrain": AStarTerrainPlanner(
            AStarConfig(
                horizon=horizon,
                # Zero traversability also denotes unobserved cells in this
                # sparse cache. Keep them costly but do not conflate unknown
                # with a verified forbidden semantic class in the debug run.
                forbid_nontraversable=False,
            )
        ),
    }


def evaluate_debug_subset(
    cache_root: Path,
    *,
    split: str = "test",
    sample_count: int = 16,
    seed: int = 0,
    planning_dt_s: float = 0.5,
) -> list[dict[str, Any]]:
    """Evaluate all baseline attempts and return one row per planner and scene."""

    dataset = Rellis3DSceneDataset(cache_root, split)
    if len(dataset) == 0:
        raise ValueError(f"split '{split}' is empty under {cache_root}")
    count = min(sample_count, len(dataset))
    indices = sorted(random.Random(seed).sample(range(len(dataset)), count))
    first_scene = dataset[indices[0]]
    horizon = int(first_scene.gt_future.shape[-2])
    planners = build_planners(horizon, planning_dt_s)
    evaluator = TerraFlowEvaluator(
        EvaluatorConfig(
            feasibility=FeasibilityMetricConfig(planning_dt_s=planning_dt_s)
        )
    )
    rows: list[dict[str, Any]] = []
    for index in indices:
        scene = dataset[index]
        identity = _scene_identity(scene.metadata, index)
        for planner_name, planner in planners.items():
            row: dict[str, Any] = {
                **identity,
                "planner": planner_name,
                "status": "available",
                "reason": "",
            }
            try:
                prediction, latency_ms = timed_planner_call(planner, scene)
                result = evaluator(prediction, scene, inference_latency_ms=latency_ms)
                row.update(_metric_columns(result))
            except PLANNER_FAILURES as error:
                row.update({"status": "unavailable", "reason": str(error)})
            rows.append(row)
    return rows


def write_results(rows: Iterable[Mapping[str, Any]], output_path: Path) -> None:
    """Write heterogeneous metric rows to a stable union-column CSV."""

    materialized = [dict(row) for row in rows]
    preferred = [
        "dataset_index",
        "sequence_id",
        "frame_id",
        "planner",
        "status",
        "reason",
        "ADE_m",
        "FDE_m",
        "path_length_m",
        "smoothness_m",
        "inference_latency_ms",
    ]
    discovered = sorted({key for row in materialized for key in row} - set(preferred))
    fields = preferred + discovered
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def print_summary(rows: Iterable[Mapping[str, Any]], output_path: Path) -> None:
    """Print per-planner availability and mean core metrics."""

    materialized = list(rows)
    names = sorted({str(row["planner"]) for row in materialized})
    print("VTF-Flow baseline debug evaluation")
    for name in names:
        planner_rows = [row for row in materialized if row["planner"] == name]
        available = [row for row in planner_rows if row["status"] == "available"]
        parts = [f"{name}: {len(available)}/{len(planner_rows)} available"]
        for metric in ("ADE_m", "FDE_m", "path_length_m", "smoothness_m"):
            values = [float(row[metric]) for row in available if row.get(metric) != ""]
            if values:
                parts.append(f"mean {metric}={sum(values) / len(values):.4f}")
        print(" | ".join(parts))
        if not available and planner_rows:
            print(f"  reason: {planner_rows[0].get('reason', 'unavailable')}")
    print(f"CSV: {output_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=Path,
        required=True,
        help="RELLIS trajectory cache root containing split subdirectories",
    )
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.5, help="trajectory interval in seconds")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if args.dt <= 0.0:
        parser.error("--dt must be positive")
    return args


def main() -> int:
    args = parse_args()
    rows = evaluate_debug_subset(
        args.cache_root,
        split=args.split,
        sample_count=args.samples,
        seed=args.seed,
        planning_dt_s=args.dt,
    )
    write_results(rows, args.output)
    print_summary(rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

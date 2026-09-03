"""Evaluate saved VTF-Flow trajectory predictions and export JSON/CSV.

Supported containers are ``.npz``, ``.npy``, ``.pt``, ``.pth``, and ``.json``.
The prediction tensor must resolve to ``[B,K,H,D]`` (or ``[B,H,D]`` for K=1),
and ground truth must resolve to ``[B,H,D]``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_PARENT = PROJECT_ROOT.parent
if str(WORKSPACE_PARENT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_PARENT))

from TerraFlow.evaluation import EvaluatorConfig, TerraFlowEvaluator  # noqa: E402


PREDICTION_KEYS = ("predictions", "trajectories", "candidates", "prediction")
GROUND_TRUTH_KEYS = ("gt_future", "ground_truth", "gt", "targets", "target")
SCORE_KEYS = ("scores", "candidate_scores")
TERRAIN_KEYS = ("terrain_map", "bev", "terrain")
LATENCY_KEYS = ("inference_latency_ms", "latency_ms", "planning_time_ms")


def _load_container(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {key: np.asarray(archive[key]) for key in archive.files}
    if suffix == ".npy":
        return {"array": np.load(path, allow_pickle=False)}
    if suffix in {".pt", ".pth"}:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        return dict(loaded) if isinstance(loaded, Mapping) else {"array": loaded}
    if suffix == ".json":
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return dict(loaded) if isinstance(loaded, Mapping) else {"array": loaded}
    raise ValueError(f"unsupported saved prediction format: {path.suffix}")


def _normalized_key(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")


def _find_value(
    container: Mapping[str, Any],
    aliases: Sequence[str],
    explicit_key: Optional[str] = None,
    required: bool = False,
) -> Any:
    if explicit_key is not None:
        if explicit_key not in container:
            raise KeyError(
                f"key '{explicit_key}' not found; available keys: {sorted(container)}"
            )
        return container[explicit_key]
    normalized = {_normalized_key(key): key for key in container}
    for alias in aliases:
        key = normalized.get(_normalized_key(alias))
        if key is not None:
            return container[key]
    if len(container) == 1 and "array" in container:
        return container["array"]
    if required:
        raise KeyError(
            f"none of keys {list(aliases)} found; available keys: {sorted(container)}"
        )
    return None


def _as_float_tensor(value: Any, name: str) -> Tensor:
    tensor = value.detach().cpu() if isinstance(value, Tensor) else torch.as_tensor(value)
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.float32)
    else:
        tensor = tensor.to(torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")
    return tensor


def load_evaluation_inputs(
    prediction_path: Path,
    ground_truth_path: Optional[Path] = None,
    prediction_key: Optional[str] = None,
    ground_truth_key: Optional[str] = None,
    scores_key: Optional[str] = None,
    terrain_key: Optional[str] = None,
    latency_key: Optional[str] = None,
) -> Dict[str, Optional[Tensor]]:
    """Load and normalize tensors from one or two saved containers."""

    prediction_container = _load_container(prediction_path)
    prediction_value = _find_value(
        prediction_container, PREDICTION_KEYS, prediction_key, required=True
    )
    predictions = _as_float_tensor(prediction_value, "predictions")
    if predictions.ndim == 3:
        predictions = predictions.unsqueeze(1)
    if predictions.ndim != 4:
        raise ValueError("saved predictions must have shape [B,K,H,D] or [B,H,D]")

    if ground_truth_path is None:
        ground_truth_container = prediction_container
    else:
        ground_truth_container = _load_container(ground_truth_path)
    ground_truth_value = _find_value(
        ground_truth_container,
        GROUND_TRUTH_KEYS,
        ground_truth_key,
        required=True,
    )
    ground_truth = _as_float_tensor(ground_truth_value, "ground_truth")
    if ground_truth.ndim != 3:
        raise ValueError("saved ground truth must have shape [B,H,D]")

    scores_value = _find_value(prediction_container, SCORE_KEYS, scores_key)
    terrain_value = _find_value(prediction_container, TERRAIN_KEYS, terrain_key)
    latency_value = _find_value(prediction_container, LATENCY_KEYS, latency_key)
    scores = None if scores_value is None else _as_float_tensor(scores_value, "scores")
    terrain = None if terrain_value is None else _as_float_tensor(terrain_value, "terrain_map")
    latency = None if latency_value is None else _as_float_tensor(latency_value, "latency")
    return {
        "predictions": predictions,
        "ground_truth": ground_truth,
        "scores": scores,
        "terrain_map": terrain,
        "inference_latency_ms": latency,
    }


def _aggregate(values: Tensor) -> Dict[str, float]:
    values = values.detach().cpu().to(torch.float64).reshape(-1)
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _per_scene_rows(results: Mapping[str, Any], batch_size: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = [{"scene_index": index} for index in range(batch_size)]
    excluded = {
        "ADE_by_candidate_m",
        "FDE_by_candidate_m",
        "path_length_by_candidate_m",
        "smoothness_by_candidate_m",
    }
    for name, value in results.items():
        if name in excluded:
            continue
        if isinstance(value, Tensor) and value.ndim == 1 and value.shape[0] == batch_size:
            converted = value.detach().cpu().tolist()
            for row, item in zip(rows, converted):
                row[name] = int(item) if name == "selected_index" else float(item)
    feasibility = results["feasibility"]
    for name, state in feasibility.items():
        for row in rows:
            row[f"{name}_status"] = state["status"]
            if state["status"] == "unavailable":
                row[f"{name}_reason"] = state["reason"]
    latency_state = results["inference_latency"]
    for row in rows:
        row["inference_latency_status"] = latency_state["status"]
        if latency_state["status"] == "unavailable":
            row["inference_latency_reason"] = latency_state["reason"]
    return rows


def _summary_document(
    results: Mapping[str, Any],
    source: Path,
    predictions: Tensor,
    coordinate_dimensions: Optional[int],
) -> Dict[str, Any]:
    aggregate: Dict[str, Dict[str, float]] = {}
    excluded = {
        "selected_index",
        "ADE_by_candidate_m",
        "FDE_by_candidate_m",
        "path_length_by_candidate_m",
        "smoothness_by_candidate_m",
    }
    for name, value in results.items():
        if name not in excluded and isinstance(value, Tensor) and value.ndim == 1:
            aggregate[name] = _aggregate(value)

    feasibility_summary: Dict[str, Any] = {}
    for name, state in results["feasibility"].items():
        item = dict(state)
        selected_key = f"selected_{name}"
        if item["status"] == "available" and selected_key in results:
            item["aggregate"] = _aggregate(results[selected_key])
        feasibility_summary[name] = item
    return {
        "schema_version": 1,
        "source": str(source.resolve()),
        "prediction_shape": list(predictions.shape),
        "batch_size": int(predictions.shape[0]),
        "candidates": int(predictions.shape[1]),
        "horizon": int(predictions.shape[2]),
        "coordinate_dimensions": (
            int(coordinate_dimensions)
            if coordinate_dimensions is not None
            else int(predictions.shape[-1])
        ),
        "selection_policy": results["selection_policy"],
        "metrics": aggregate,
        "feasibility_metrics": feasibility_summary,
        "inference_latency": results["inference_latency"],
    }


def _terminal_summary(summary: Mapping[str, Any]) -> str:
    lines = [
        "VTF-Flow trajectory evaluation",
        (
            f"  samples={summary['batch_size']}  K={summary['candidates']}  "
            f"H={summary['horizon']}  D_eval={summary['coordinate_dimensions']}"
        ),
        f"  selection={summary['selection_policy']}",
    ]
    preferred = (
        "ADE_m",
        "FDE_m",
        "minADE@K_m",
        "minFDE@K_m",
        "diversity_m",
        "path_length_m",
        "smoothness_m",
        "inference_latency_ms",
    )
    metrics = summary["metrics"]
    for name in preferred:
        if name in metrics:
            unit = " ms/sample" if name == "inference_latency_ms" else ""
            lines.append(
                f"  {name}: mean={metrics[name]['mean']:.6g}, "
                f"median={metrics[name]['median']:.6g}{unit}"
            )
    lines.append("  feasibility availability:")
    for name, state in summary["feasibility_metrics"].items():
        if state["status"] == "available":
            lines.append(f"    {name}: available")
        else:
            lines.append(f"    {name}: unavailable — {state['reason']}")
    if summary["inference_latency"]["status"] == "unavailable":
        lines.append("  inference latency: unavailable — not recorded")
    return "\n".join(lines)


def evaluate_saved_predictions(
    prediction_path: Path,
    output_dir: Path,
    ground_truth_path: Optional[Path] = None,
    prediction_key: Optional[str] = None,
    ground_truth_key: Optional[str] = None,
    scores_key: Optional[str] = None,
    terrain_key: Optional[str] = None,
    latency_key: Optional[str] = None,
    latency_ms_override: Optional[float] = None,
    coordinate_dimensions: Optional[int] = None,
    scores_higher_is_better: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:
    """Evaluate a saved bundle and write JSON plus per-scene CSV."""

    inputs = load_evaluation_inputs(
        prediction_path,
        ground_truth_path=ground_truth_path,
        prediction_key=prediction_key,
        ground_truth_key=ground_truth_key,
        scores_key=scores_key,
        terrain_key=terrain_key,
        latency_key=latency_key,
    )
    predictions = inputs["predictions"]
    ground_truth = inputs["ground_truth"]
    assert predictions is not None and ground_truth is not None
    latency: Any = (
        latency_ms_override
        if latency_ms_override is not None
        else inputs["inference_latency_ms"]
    )
    evaluator = TerraFlowEvaluator(
        EvaluatorConfig(
            coordinate_dimensions=coordinate_dimensions,
            scores_higher_is_better=scores_higher_is_better,
        )
    )
    results = evaluator.evaluate_tensors(
        trajectories=predictions,
        ground_truth=ground_truth,
        scores=inputs["scores"],
        terrain_map=inputs["terrain_map"],
        inference_latency_ms=latency,
    )
    rows = _per_scene_rows(results, predictions.shape[0])
    summary = _summary_document(
        results, prediction_path, predictions, coordinate_dimensions
    )
    terminal = _terminal_summary(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "per_scene_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary, rows, terminal


def build_argument_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate saved [B,K,H,D] VTF-Flow trajectory predictions."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prediction-key")
    parser.add_argument("--ground-truth-key")
    parser.add_argument("--scores-key")
    parser.add_argument("--terrain-key")
    parser.add_argument("--latency-key")
    parser.add_argument("--latency-ms", type=float)
    parser.add_argument(
        "--coordinate-dimensions",
        type=int,
        choices=(2, 3),
        help="Coordinates used for distance metrics; default uses all saved dimensions.",
    )
    parser.add_argument("--scores-higher-is-better", action="store_true")
    return parser


def main() -> None:
    """Run saved-prediction evaluation and print a human-readable summary."""

    args = build_argument_parser().parse_args()
    _, _, terminal = evaluate_saved_predictions(
        prediction_path=args.predictions,
        output_dir=args.output_dir,
        ground_truth_path=args.ground_truth,
        prediction_key=args.prediction_key,
        ground_truth_key=args.ground_truth_key,
        scores_key=args.scores_key,
        terrain_key=args.terrain_key,
        latency_key=args.latency_key,
        latency_ms_override=args.latency_ms,
        coordinate_dimensions=args.coordinate_dimensions,
        scores_higher_is_better=args.scores_higher_is_better,
    )
    print(terminal)
    print(f"JSON: {(args.output_dir / 'evaluation_summary.json').resolve()}")
    print(f"CSV: {(args.output_dir / 'per_scene_metrics.csv').resolve()}")


if __name__ == "__main__":
    main()

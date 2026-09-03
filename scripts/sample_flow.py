"""Sample/evaluate Flow Matching and plot it against regression trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


TERRAFLOW_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TERRAFLOW_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from TerraFlow.datasets.rellis3d import Rellis3DSceneDataset, collate_scenes  # noqa: E402
from TerraFlow.evaluation import TerraFlowEvaluator, timed_planner_call  # noqa: E402
from TerraFlow.models.flow_network import ConditionalTrajectoryFlow  # noqa: E402
from TerraFlow.planners.flow_planner import FlowPlanner, FlowPlannerConfig  # noqa: E402
from TerraFlow.planners.regression_planner import (  # noqa: E402
    RegressionPlanner,
    RegressionPlannerConfig,
)


DEFAULT_OUTPUT = TERRAFLOW_ROOT / "outputs" / "flow_sampling"
METRIC_NAMES = (
    "ADE_m",
    "FDE_m",
    "minADE@K_m",
    "minFDE@K_m",
    "diversity_m",
    "smoothness_m",
)


def load_flow(path: Path, device: torch.device) -> tuple[ConditionalTrajectoryFlow, dict[str, Any]]:
    """Load a self-describing Flow Matching checkpoint."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(checkpoint["model_config"])
    if "metric_scales" in config:
        config["metric_scales"] = tuple(config["metric_scales"])
    model = ConditionalTrajectoryFlow(**config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def load_regression(path: Path, device: torch.device) -> RegressionPlanner:
    """Load the deterministic baseline used for paired plots."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(checkpoint["model_config"])
    if "metric_scales" in config:
        config["metric_scales"] = tuple(config["metric_scales"])
    model = RegressionPlanner(RegressionPlannerConfig(**config)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def choose_indices(
    dataset: Rellis3DSceneDataset,
    sample_count: int,
    seed: int,
    sequence: str | None,
) -> list[int]:
    """Choose a reproducible subset, optionally restricted to one sequence."""

    candidates = list(range(len(dataset)))
    if sequence is not None:
        normalized = str(sequence).zfill(5)
        candidates = [
            index
            for index, metadata in enumerate(dataset.manifest)
            if str(metadata.get("sequence", "")).zfill(5) == normalized
        ]
    if not candidates:
        raise ValueError("no dataset samples match the requested sequence")
    return sorted(
        random.Random(seed).sample(candidates, min(sample_count, len(candidates)))
    )


@torch.no_grad()
def evaluate_step_count(
    model: ConditionalTrajectoryFlow,
    dataset: Rellis3DSceneDataset,
    indices: Sequence[int],
    device: torch.device,
    candidates: int,
    steps: int,
    batch_size: int,
    seed: int,
) -> dict[str, float]:
    """Evaluate a fixed Euler step count with paired random seeds."""

    planner = FlowPlanner(
        model, FlowPlannerConfig(candidates=candidates, integration_steps=steps)
    ).to(device)
    evaluator = TerraFlowEvaluator()
    totals = {name: 0.0 for name in METRIC_NAMES}
    latency_total, count = 0.0, 0
    # Exclude one-time CUDA/library initialization from the step comparison.
    warm_indices = indices[: min(batch_size, len(indices))]
    warm_scene = collate_scenes([dataset[index] for index in warm_indices]).to(device)
    torch.manual_seed(seed - 1)
    planner(warm_scene)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    for batch_start in range(0, len(indices), batch_size):
        batch_indices = indices[batch_start : batch_start + batch_size]
        scene = collate_scenes([dataset[index] for index in batch_indices]).to(device)
        torch.manual_seed(seed + batch_start)
        prediction, latency_ms = timed_planner_call(planner, scene)
        result = evaluator(prediction, scene, inference_latency_ms=latency_ms)
        for name in METRIC_NAMES:
            totals[name] += float(result[name].sum().detach().cpu())
        latency_total += latency_ms * scene.batch_size
        count += scene.batch_size
    report = {name: value / count for name, value in totals.items()}
    report["latency_ms_per_sample"] = latency_total / count
    return report


@torch.no_grad()
def save_comparison_plots(
    flow: ConditionalTrajectoryFlow,
    regression: RegressionPlanner,
    dataset: Rellis3DSceneDataset,
    indices: Sequence[int],
    output_dir: Path,
    device: torch.device,
    candidates: int,
    steps: int,
    seed: int,
) -> list[str]:
    """Plot Regression, all Flow samples, oracle Flow sample, and GT."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    planner = FlowPlanner(
        flow, FlowPlannerConfig(candidates=candidates, integration_steps=steps)
    ).to(device)
    for plot_index, dataset_index in enumerate(indices):
        scene = dataset[dataset_index]
        device_scene = scene.to(device)
        torch.manual_seed(seed + dataset_index)
        flow_paths = planner(device_scene).trajectories[0].cpu()
        regression_path = regression(device_scene).trajectories[0, 0].cpu()
        target = scene.gt_future.cpu()
        ade = torch.linalg.vector_norm(flow_paths - target[None], dim=-1).mean(dim=-1)
        best = int(ade.argmin())
        figure, (axis_xy, axis_z) = plt.subplots(
            1, 2, figsize=(11.0, 4.8), constrained_layout=True
        )
        axis_xy.scatter([0.0], [0.0], marker="*", s=100, color="black", label="ego origin")
        for candidate in flow_paths:
            axis_xy.plot(candidate[:, 0], candidate[:, 1], color="#93c5fd", alpha=0.24)
        axis_xy.plot(target[:, 0], target[:, 1], color="black", lw=2.4, label="GT")
        axis_xy.plot(
            regression_path[:, 0], regression_path[:, 1],
            color="#16a34a", lw=2.0, label="Regression",
        )
        axis_xy.plot(
            flow_paths[best, :, 0], flow_paths[best, :, 1],
            color="#dc2626", lw=2.0, label="Flow best-of-K",
        )
        axis_xy.set_xlabel("ego x (m)")
        axis_xy.set_ylabel("ego y (m)")
        axis_xy.set_aspect("equal", adjustable="datalim")
        axis_xy.grid(alpha=0.25)
        axis_xy.legend(fontsize=8)
        waypoint = np.arange(1, len(target) + 1)
        axis_z.plot(waypoint, target[:, 2], color="black", label="GT z")
        axis_z.plot(waypoint, regression_path[:, 2], color="#16a34a", label="Regression z")
        axis_z.plot(waypoint, flow_paths[best, :, 2], color="#dc2626", label="Flow z")
        axis_z.set_xlabel("future waypoint")
        axis_z.set_ylabel("ego z (m)")
        axis_z.grid(alpha=0.25)
        axis_z.legend(fontsize=8)
        metadata: Mapping[str, Any] = scene.metadata if isinstance(scene.metadata, Mapping) else {}
        figure.suptitle(
            f"Regression vs Flow | sequence {metadata.get('sequence', '?')} "
            f"frame {metadata.get('frame_id', '?')} | {steps} Euler steps"
        )
        path = output_dir / f"comparison_{plot_index:02d}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(str(path.resolve()))
    return paths


def parse_steps(value: str) -> tuple[int, ...]:
    """Parse a comma-separated subset of the supported Euler counts."""

    steps = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    for step in steps:
        FlowPlannerConfig(integration_steps=step)
    if not steps:
        raise ValueError("at least one integration step count is required")
    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--regression-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--sequence", help="optional complete sequence ID filter")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--steps", default="4,8,16,32")
    parser.add_argument("--plot-examples", type=int, default=8)
    parser.add_argument("--comparison-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7301)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples <= 0 or args.batch_size <= 0 or args.candidates <= 0:
        raise ValueError("samples, batch size, and candidates must be positive")
    steps = parse_steps(args.steps)
    if args.comparison_steps not in steps:
        raise ValueError("comparison steps must be included in --steps")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Rellis3DSceneDataset(args.cache_root, args.split)
    indices = choose_indices(dataset, args.samples, args.seed, args.sequence)
    flow, checkpoint = load_flow(args.checkpoint, device)
    regression = load_regression(args.regression_checkpoint, device)
    rows = []
    for step in steps:
        metrics = evaluate_step_count(
            flow,
            dataset,
            indices,
            device,
            args.candidates,
            step,
            args.batch_size,
            args.seed,
        )
        row = {"integration_steps": step, "candidates": args.candidates, **metrics}
        rows.append(row)
        print(json.dumps(row), flush=True)
    csv_path = args.output_dir / "ode_step_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_indices = indices[: min(args.plot_examples, len(indices))]
    plots = save_comparison_plots(
        flow,
        regression,
        dataset,
        plot_indices,
        args.output_dir / "regression_vs_flow",
        device,
        args.candidates,
        args.comparison_steps,
        args.seed,
    )
    report = {
        "status": "complete",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "split": args.split,
        "sequence_filter": args.sequence,
        "sample_count": len(indices),
        "indices": indices,
        "flow_candidates": args.candidates,
        "ode_step_results": rows,
        "plots": plots,
        "selection_note": "ADE/FDE use candidate zero; min metrics are oracle best-of-K",
        "guidance": "none",
    }
    (args.output_dir / "sampling_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

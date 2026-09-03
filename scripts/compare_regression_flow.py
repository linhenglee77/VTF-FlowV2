"""Paired sequence-held-out comparison of Regression and Flow Matching."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys
import time
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


METRICS = (
    "ADE_m",
    "FDE_m",
    "minADE@K_m",
    "minFDE@K_m",
    "diversity_m",
    "smoothness_m",
)


def load_regression(path: Path, device: torch.device) -> RegressionPlanner:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    values = dict(checkpoint["model_config"])
    values["metric_scales"] = tuple(values["metric_scales"])
    model = RegressionPlanner(RegressionPlannerConfig(**values)).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def load_flow(path: Path, device: torch.device) -> ConditionalTrajectoryFlow:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    values = dict(checkpoint["model_config"])
    values["metric_scales"] = tuple(values["metric_scales"])
    model = ConditionalTrajectoryFlow(**values).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def validation_indices(dataset: Rellis3DSceneDataset, sequence: str) -> list[int]:
    normalized = str(sequence).zfill(5)
    indices = [
        index
        for index, metadata in enumerate(dataset.manifest)
        if str(metadata.get("sequence", "")).zfill(5) == normalized
    ]
    if not indices:
        raise ValueError(f"sequence {normalized} is absent from the selected split")
    return indices


@torch.inference_mode()
def warm_up(planner, scene, device: torch.device) -> None:
    planner(scene)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def metric_rows(
    method: str,
    candidates: int,
    result: Mapping[str, Any],
    latency_ms: float,
    indices: Sequence[int],
    metadata: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for local_index, dataset_index in enumerate(indices):
        row: dict[str, Any] = {
            "dataset_index": dataset_index,
            "sequence": metadata[local_index].get("sequence", ""),
            "frame_id": metadata[local_index].get("frame_id", ""),
            "method": method,
            "K": candidates,
            "latency_ms_per_sample": latency_ms,
        }
        for name in METRICS:
            row[name] = float(result[name][local_index].detach().cpu())
        row["oracle_ADE_gain_m"] = row["ADE_m"] - row["minADE@K_m"]
        row["oracle_FDE_gain_m"] = row["FDE_m"] - row["minFDE@K_m"]
        rows.append(row)
    return rows


@torch.inference_mode()
def run_paired_evaluation(
    regression: RegressionPlanner,
    flow: ConditionalTrajectoryFlow,
    dataset: Rellis3DSceneDataset,
    indices: Sequence[int],
    batch_size: int,
    candidate_counts: Sequence[int],
    integration_steps: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Evaluate all methods on identical scenes and paired first-candidate noise."""

    evaluator = TerraFlowEvaluator()
    regression_planner = regression
    flow_planners = {
        candidates: FlowPlanner(
            flow,
            FlowPlannerConfig(
                candidates=candidates, integration_steps=integration_steps
            ),
        ).to(device)
        for candidates in candidate_counts
    }
    warm_indices = indices[: min(batch_size, len(indices))]
    warm_scene = collate_scenes([dataset[index] for index in warm_indices]).to(device)
    warm_up(regression_planner, warm_scene, device)
    for planner in flow_planners.values():
        warm_up(planner, warm_scene, device)

    rows: list[dict[str, Any]] = []
    for batch_start in range(0, len(indices), batch_size):
        batch_indices = list(indices[batch_start : batch_start + batch_size])
        scenes = [dataset[index] for index in batch_indices]
        scene = collate_scenes(scenes).to(device)
        prediction, latency = timed_planner_call(regression_planner, scene)
        result = evaluator(prediction, scene, inference_latency_ms=latency)
        rows.extend(
            metric_rows(
                "Regression",
                1,
                result,
                latency,
                batch_indices,
                [item.metadata for item in scenes],
            )
        )
        for candidates, planner in flow_planners.items():
            torch.manual_seed(seed + batch_start)
            prediction, latency = timed_planner_call(planner, scene)
            result = evaluator(prediction, scene, inference_latency_ms=latency)
            rows.extend(
                metric_rows(
                    "Flow Matching",
                    candidates,
                    result,
                    latency,
                    batch_indices,
                    [item.metadata for item in scenes],
                )
            )
        print(
            f"evaluated {min(batch_start + len(batch_indices), len(indices))}/{len(indices)}",
            flush=True,
        )
    return rows


def summarize(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    groups = [("Regression", 1)] + [
        ("Flow Matching", candidates) for candidates in (1, 3, 6, 10)
    ]
    for method, candidates in groups:
        subset = [
            row for row in rows if row["method"] == method and row["K"] == candidates
        ]
        if not subset:
            continue
        summary: dict[str, Any] = {
            "method": method,
            "K": candidates,
            "scenes": len(subset),
        }
        for name in METRICS + (
            "latency_ms_per_sample",
            "oracle_ADE_gain_m",
            "oracle_FDE_gain_m",
        ):
            summary[name] = float(np.mean([float(row[name]) for row in subset]))
        result.append(summary)
    return result


def paired_analysis(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    regression = {
        int(row["dataset_index"]): row
        for row in rows
        if row["method"] == "Regression"
    }
    flow_one = {
        int(row["dataset_index"]): row
        for row in rows
        if row["method"] == "Flow Matching" and row["K"] == 1
    }
    flow_ten = {
        int(row["dataset_index"]): row
        for row in rows
        if row["method"] == "Flow Matching" and row["K"] == 10
    }
    common = sorted(set(regression) & set(flow_one) & set(flow_ten))
    gains = np.asarray(
        [float(flow_one[index]["ADE_m"]) - float(flow_ten[index]["minADE@K_m"]) for index in common]
    )
    diversity = np.asarray([float(flow_ten[index]["diversity_m"]) for index in common])
    regression_advantage = np.asarray(
        [
            float(regression[index]["ADE_m"])
            - float(flow_ten[index]["minADE@K_m"])
            for index in common
        ]
    )
    correlation = float(np.corrcoef(diversity, gains)[0, 1])
    return {
        "mean_oracle_ADE_gain_K10_vs_K1_m": float(gains.mean()),
        "median_oracle_ADE_gain_K10_vs_K1_m": float(np.median(gains)),
        "fraction_oracle_ADE_gain_gt_0_05m": float(np.mean(gains > 0.05)),
        "fraction_minADE10_better_than_regression": float(
            np.mean(regression_advantage > 0.0)
        ),
        "mean_regression_ADE_minus_flow_minADE10_m": float(
            regression_advantage.mean()
        ),
        "diversity_oracle_gain_correlation": correlation,
    }


def select_figure_indices(
    rows: Sequence[Mapping[str, Any]], count: int, seed: int
) -> list[int]:
    flow_ten = [
        row for row in rows if row["method"] == "Flow Matching" and row["K"] == 10
    ]
    ranked_gain = sorted(
        flow_ten, key=lambda row: float(row["oracle_ADE_gain_m"]), reverse=True
    )
    ranked_diversity = sorted(
        flow_ten, key=lambda row: float(row["diversity_m"]), reverse=True
    )
    selected: list[int] = []
    for row in ranked_gain[: max(1, count // 3)] + ranked_diversity[: max(1, count // 3)]:
        index = int(row["dataset_index"])
        if index not in selected:
            selected.append(index)
    remaining = [int(row["dataset_index"]) for row in flow_ten if int(row["dataset_index"]) not in selected]
    random.Random(seed).shuffle(remaining)
    selected.extend(remaining[: max(0, count - len(selected))])
    return selected[:count]


def _plot_limits(paths: torch.Tensor) -> tuple[tuple[float, float], tuple[float, float]]:
    x_values = paths[..., 0].flatten().numpy()
    y_values = paths[..., 1].flatten().numpy()
    x_min, x_max = min(0.0, float(x_values.min())), float(x_values.max())
    y_min, y_max = float(y_values.min()), float(y_values.max())
    x_padding = max(1.0, 0.08 * (x_max - x_min + 1e-3))
    y_padding = max(1.0, 0.12 * (y_max - y_min + 1e-3))
    return (
        (max(0.0, x_min - x_padding), min(24.0, x_max + x_padding)),
        (max(-12.0, y_min - y_padding), min(12.0, y_max + y_padding)),
    )


@torch.inference_mode()
def save_qualitative_figures(
    regression: RegressionPlanner,
    flow: ConditionalTrajectoryFlow,
    dataset: Rellis3DSceneDataset,
    indices: Sequence[int],
    output_dir: Path,
    device: torch.device,
    integration_steps: int,
    seed: int,
) -> list[str]:
    """Export image-plate + trajectory figures as PNG, SVG, and PDF."""

    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir / "source_data"
    source_dir.mkdir(exist_ok=True)
    planner = FlowPlanner(
        flow, FlowPlannerConfig(candidates=10, integration_steps=integration_steps)
    ).to(device)
    exported = []
    extent = (0.0, 24.0, -12.0, 12.0)
    for figure_index, dataset_index in enumerate(indices):
        scene = dataset[dataset_index]
        device_scene = scene.to(device)
        regression_path = regression(device_scene).trajectories[0, 0].cpu()
        torch.manual_seed(seed + dataset_index)
        flow_paths = planner(device_scene).trajectories[0].cpu()
        target = scene.gt_future.cpu()
        ade = torch.linalg.vector_norm(flow_paths - target[None], dim=-1).mean(dim=-1)
        best = int(ade.argmin())
        all_paths = torch.cat((target[None], regression_path[None], flow_paths), dim=0)
        xlim, ylim = _plot_limits(all_paths)
        terrain = scene.terrain_map.cpu()
        traversability = terrain[0].T.numpy()
        occupancy = terrain[1].T.numpy()
        elevation = terrain[2].T.numpy() * 4.5 - 2.5
        observed = (terrain[0] + terrain[1] + terrain[2]).T.numpy() > 0.0
        elevation = np.ma.masked_where(~observed, elevation)

        figure = plt.figure(figsize=(7.1, 4.15), constrained_layout=True)
        grid = figure.add_gridspec(3, 3, width_ratios=(1.4, 1.4, 1.0))
        main = figure.add_subplot(grid[:, :2])
        context_axes = [figure.add_subplot(grid[row, 2]) for row in range(3)]
        main.imshow(
            traversability,
            origin="lower",
            extent=extent,
            cmap="Greys",
            vmin=0.0,
            vmax=1.0,
            alpha=0.38,
            rasterized=True,
        )
        for candidate in flow_paths:
            main.plot(candidate[:, 0], candidate[:, 1], color="#78a6d1", alpha=0.32, lw=0.9)
        gt_line, = main.plot(
            target[:, 0], target[:, 1], color="#111827", lw=2.0, label="GT", zorder=5
        )
        regression_line, = main.plot(
            regression_path[:, 0], regression_path[:, 1],
            color="#2f855a", lw=1.7, label="Regression", zorder=4,
        )
        flow_line, = main.plot(
            flow_paths[best, :, 0], flow_paths[best, :, 1],
            color="#c2413b", lw=1.7, label="Flow best-of-10", zorder=4,
        )
        main.scatter([0.0], [0.0], marker="*", s=55, color="black", zorder=6)
        main.scatter([target[-1, 0]], [target[-1, 1]], marker="x", s=32, color="black", zorder=6)
        main.set(xlabel="Ego x (m)", ylabel="Ego y (m)", xlim=xlim, ylim=ylim)
        main.set_aspect("equal", adjustable="box")
        main.set_title("Terrain-conditioned future trajectories", loc="left", fontweight="bold")

        channel_specs = (
            (traversability, "Traversability", "Greens", 0.0, 1.0),
            (occupancy, "Occupancy", "magma", 0.0, 1.0),
            (elevation, "Elevation (m)", "terrain", -2.5, 2.0),
        )
        for axis, (values, title, cmap, low, high) in zip(context_axes, channel_specs):
            image = axis.imshow(
                values,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=low,
                vmax=high,
                rasterized=True,
            )
            axis.set(xlim=xlim, ylim=ylim, title=title)
            axis.set_xticks([])
            axis.set_yticks([])
            figure.colorbar(image, ax=axis, fraction=0.045, pad=0.02)
        metadata = scene.metadata if isinstance(scene.metadata, Mapping) else {}
        figure.suptitle(
            f"Sequence {metadata.get('sequence', '?')} | frame {metadata.get('frame_id', '?')} | "
            f"10 Flow samples",
            fontsize=8,
        )
        figure.legend(
            [gt_line, regression_line, flow_line],
            ["GT", "Regression", "Flow best-of-10"],
            loc="upper center",
            bbox_to_anchor=(0.42, 0.94),
            ncol=3,
            fontsize=6.5,
        )
        stem = output_dir / f"scene_{figure_index:02d}"
        figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
        figure.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
        figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
        figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(figure)
        np.savez_compressed(
            source_dir / f"scene_{figure_index:02d}.npz",
            dataset_index=dataset_index,
            terrain_map=terrain.numpy(),
            gt=target.numpy(),
            regression=regression_path.numpy(),
            flow=flow_paths.numpy(),
            flow_ADE=ade.numpy(),
        )
        exported.append(str(stem.with_suffix(".png").resolve()))
    return exported


def write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    summaries: Sequence[Mapping[str, Any]],
    analysis: Mapping[str, float],
    figure_paths: Sequence[str],
    scenes: int,
) -> None:
    header = (
        "| Planner | K | ADE (m) | FDE (m) | minADE@K (m) | minFDE@K (m) | "
        "Diversity (m) | Smoothness (m) | Latency (ms/sample) |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    table = [header]
    for row in summaries:
        table.append(
            f"| {row['method']} | {row['K']} | {row['ADE_m']:.4f} | "
            f"{row['FDE_m']:.4f} | {row['minADE@K_m']:.4f} | "
            f"{row['minFDE@K_m']:.4f} | {row['diversity_m']:.4f} | "
            f"{row['smoothness_m']:.4f} | {row['latency_ms_per_sample']:.4f} |"
        )
    relative_figures = [Path(value).relative_to(path.parent).as_posix() for value in figure_paths]
    figure_lines = "\n".join(
        f"![Qualitative scene {index + 1}]({value})"
        for index, value in enumerate(relative_figures)
    )
    text = f"""# Regression vs Flow Matching

## Experimental design

- Validation set: all {scenes} cached scenes from complete held-out sequence `00004`.
- Training sequences: `00000`, `00002`, and `00003`; sequence overlap is zero.
- Both methods use the same 192-dimensional `RegressionSceneEncoder` architecture for ego history, goal, and 3-channel terrain BEV context.
- Both use seed 4201, batch size 128, and 40 full-data epochs. Their trajectory decoders and training objectives necessarily differ.
- Flow uses 16 Euler steps. For K=1/3/6/10, the first candidate uses paired initial Gaussian noise per scene.
- ADE/FDE refer to candidate zero. `minADE@K` and `minFDE@K` are oracle metrics and do not describe a deployable selection rule.

## Results

{chr(10).join(table)}

## Multimodality analysis

- Increasing Flow from K=1 to K=10 reduced mean oracle ADE by **{analysis['mean_oracle_ADE_gain_K10_vs_K1_m']:.4f} m** (median {analysis['median_oracle_ADE_gain_K10_vs_K1_m']:.4f} m).
- **{analysis['fraction_oracle_ADE_gain_gt_0_05m'] * 100:.1f}%** of scenes gained more than 0.05 m in oracle ADE at K=10.
- Flow minADE@10 was lower than deterministic Regression ADE in **{analysis['fraction_minADE10_better_than_regression'] * 100:.1f}%** of paired scenes. The mean `Regression ADE - Flow minADE@10` was **{analysis['mean_regression_ADE_minus_flow_minADE10_m']:.4f} m**.
- Correlation between Flow diversity and K=10 oracle ADE gain was **{analysis['diversity_oracle_gain_correlation']:.3f}**.

These measurements support only **limited oracle value from multiple samples**. A positive best-of-K gain shows that some scenes benefit from drawing alternatives, but the dataset provides only one recorded future per scene. Therefore, diversity cannot by itself establish multiple valid behavioral modes. Some visually distinct samples may be sampling error rather than feasible alternative routes; terrain feasibility and counterfactual-route supervision were not used. Flow should not be described as better than Regression without a deployable candidate selector and validation that alternative samples are terrain-feasible.

Regression has an architectural goal endpoint anchor, whereas Flow has no post-hoc endpoint anchoring. Its zero FDE is therefore not a like-for-like learned endpoint comparison and should be interpreted separately from ADE.

## Qualitative figures

Each figure shows the local traversability, occupancy, and elevation context; the recorded GT; deterministic Regression; all ten Flow samples; and the oracle best-of-10 sample for visualization only.

{figure_lines}

## Reproducibility and limitations

- Metrics are means over all held-out scenes; raw per-scene values are stored beside this report.
- Latency was measured after warm-up on the available CUDA device and includes scene encoding plus planning.
- Only one training seed and one future per scene were evaluated.
- No terrain-feasibility guidance or learned candidate-ranking model was used.
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--regression-checkpoint", type=Path, required=True)
    parser.add_argument("--flow-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--validation-sequence", default="00004")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--integration-steps", type=int, default=16)
    parser.add_argument("--figure-count", type=int, default=9)
    parser.add_argument("--seed", type=int, default=9201)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Rellis3DSceneDataset(args.cache_root, args.source_split)
    indices = validation_indices(dataset, args.validation_sequence)
    regression = load_regression(args.regression_checkpoint, device)
    flow = load_flow(args.flow_checkpoint, device)
    rows = run_paired_evaluation(
        regression,
        flow,
        dataset,
        indices,
        args.batch_size,
        (1, 3, 6, 10),
        args.integration_steps,
        args.seed,
        device,
    )
    summaries = summarize(rows)
    analysis = paired_analysis(rows)
    write_summary_csv(args.output_dir / "regression_vs_flow.csv", summaries)
    write_summary_csv(args.output_dir / "regression_vs_flow_per_scene.csv", rows)
    figure_indices = select_figure_indices(rows, args.figure_count, args.seed)
    figures = save_qualitative_figures(
        regression,
        flow,
        dataset,
        figure_indices,
        args.output_dir / "figures",
        device,
        args.integration_steps,
        args.seed,
    )
    write_markdown(
        args.output_dir / "regression_vs_flow.md",
        summaries,
        analysis,
        figures,
        len(indices),
    )
    metadata = {
        "status": "complete",
        "device": str(device),
        "validation_sequence": str(args.validation_sequence).zfill(5),
        "scenes": len(indices),
        "candidate_counts": [1, 3, 6, 10],
        "integration_steps": args.integration_steps,
        "analysis": analysis,
        "figure_indices": figure_indices,
        "figures": figures,
    }
    (args.output_dir / "regression_vs_flow_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({"summary": summaries, "analysis": analysis}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Plots for guided Flow strength, schedules, evolution, and paired cases."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import textwrap
from typing import Any, Mapping, Sequence

import numpy as np


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV into dictionaries."""

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_eta_tradeoffs(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    """Save eight eta curves, a 3x3 summary, and a fidelity-cost Pareto plot."""

    plt = _pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: float(row["eta"]))
    eta = np.asarray([float(row["eta"]) for row in ordered])
    panels = (
        ("minADE@K_m", "minADE@K (m)"),
        ("minFDE@K_m", "minFDE@K (m)"),
        ("terrain_violation_rate", "terrain violation"),
        ("mean_terrain_cost", "mean terrain cost"),
        ("slope_violation_rate", "slope violation"),
        ("smoothness_m", "smoothness (m)"),
        ("diversity_m", "diversity (m)"),
        ("latency_ms_per_scene", "latency (ms/scene)"),
    )
    figure, axes = plt.subplots(3, 3, figsize=(12, 9), constrained_layout=True)
    for axis, (metric, label) in zip(axes.flat, panels):
        values = np.asarray([float(row[metric]) for row in ordered])
        axis.plot(eta, values, marker="o", color="#2563eb")
        axis.set_xlabel("eta")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        single, single_axis = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
        single_axis.plot(eta, values, marker="o", color="#2563eb")
        single_axis.set_xlabel("eta")
        single_axis.set_ylabel(label)
        single_axis.grid(alpha=0.25)
        single.savefig(output_dir / f"eta_vs_{metric}.png", dpi=180)
        plt.close(single)
    pareto_axis = axes.flat[-1]
    fidelity = np.asarray([float(row["minADE@K_m"]) for row in ordered])
    cost = np.asarray([float(row["mean_vehicle_conditioned_cost"]) for row in ordered])
    scatter = pareto_axis.scatter(fidelity, cost, c=eta, cmap="viridis", s=55)
    for x_value, y_value, eta_value in zip(fidelity, cost, eta):
        pareto_axis.annotate(f"{eta_value:g}", (x_value, y_value), xytext=(4, 3), textcoords="offset points")
    pareto_axis.set_xlabel("minADE@K (m)")
    pareto_axis.set_ylabel("vehicle-conditioned cost")
    pareto_axis.grid(alpha=0.25)
    figure.colorbar(scatter, ax=pareto_axis, label="eta")
    figure.savefig(output_dir / "eta_tradeoff_summary.png", dpi=190)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.5, 4.2), constrained_layout=True)
    scatter = axis.scatter(fidelity, cost, c=eta, cmap="viridis", s=70)
    for x_value, y_value, eta_value in zip(fidelity, cost, eta):
        axis.annotate(f"eta={eta_value:g}", (x_value, y_value), xytext=(5, 4), textcoords="offset points")
    axis.set_xlabel("trajectory fidelity: minADE@K (m), lower better")
    axis.set_ylabel("vehicle-conditioned cost, lower better")
    axis.grid(alpha=0.25)
    figure.colorbar(scatter, ax=axis, label="eta")
    figure.savefig(output_dir / "pareto_fidelity_vs_feasibility.png", dpi=190)
    plt.close(figure)


def plot_guidance_evolution(
    rows: Sequence[Mapping[str, Any]], output_dir: Path
) -> None:
    """Plot aggregate clean-estimate feasibility cost over Euler steps."""

    plt = _pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    labels = sorted({str(row["run"]) for row in rows})
    for label in labels:
        selected = sorted(
            (row for row in rows if str(row["run"]) == label),
            key=lambda row: int(row["step"]),
        )
        if not selected:
            continue
        axis.plot(
            [int(row["step"]) for row in selected],
            [float(row["guidance_cost_mean"]) for row in selected],
            marker="o", ms=2.5, lw=1.3, label=label,
        )
    axis.set_xlabel("Euler integration step")
    axis.set_ylabel("clean-estimate vehicle feasibility cost")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    figure.savefig(output_dir / "feasibility_cost_vs_flow_step.png", dpi=190)
    plt.close(figure)


def plot_schedule_comparison(
    rows: Sequence[Mapping[str, Any]], output_dir: Path
) -> None:
    """Compare constant, early-strong, and late-strong end metrics."""

    plt = _pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = (
        ("minADE@K_m", "minADE@K"),
        ("mean_vehicle_conditioned_cost", "vehicle cost"),
        ("terrain_violation_rate", "terrain violation"),
        ("latency_ms_per_scene", "latency"),
    )
    figure, axes = plt.subplots(1, 4, figsize=(12, 3.5), constrained_layout=True)
    names = [str(row["schedule"]) for row in rows]
    for axis, (metric, title) in zip(axes, metrics):
        axis.bar(names, [float(row[metric]) for row in rows], color="#0f766e")
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(output_dir / "schedule_comparison.png", dpi=190)
    plt.close(figure)


def plot_qualitative_case(
    feasibility_map: np.ndarray,
    extent: tuple[float, float, float, float],
    ground_truth: np.ndarray,
    unguided: np.ndarray,
    guided: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """Overlay paired candidate sets on a terrain feasibility map."""

    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(7.0, 7.0), constrained_layout=True)
    image = axis.imshow(
        feasibility_map, origin="lower", extent=extent, aspect="auto",
        cmap="viridis", vmin=0.0, vmax=1.0,
    )
    for index, trajectory in enumerate(unguided):
        axis.plot(trajectory[:, 1], trajectory[:, 0], color="#3b82f6", alpha=0.25, lw=1.0,
                  label="unguided Flow" if index == 0 else None)
    for index, trajectory in enumerate(guided):
        axis.plot(trajectory[:, 1], trajectory[:, 0], color="#f97316", alpha=0.36, lw=1.1,
                  label="guided Flow" if index == 0 else None)
    axis.plot(ground_truth[:, 1], ground_truth[:, 0], color="white", lw=2.5, label="GT")
    axis.scatter([0.0], [0.0], marker="*", s=100, color="red", edgecolor="white", label="ego")
    axis.set_xlabel("ego lateral y (m)")
    axis.set_ylabel("ego forward x (m)")
    axis.set_title(textwrap.fill(title, width=72), fontsize=10, loc="left")
    axis.legend(loc="upper right")
    figure.colorbar(image, ax=axis, label="terrain-only feasibility (context)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=190)
    plt.close(figure)


def regenerate_saved_qualitative_cases(input_dir: Path, cache_root: Path) -> None:
    """Re-render objectively selected cases from saved paired predictions."""

    import torch

    from TerraFlow.scripts.train_regression import CombinedSceneDataset
    from TerraFlow.terrain.feasibility_field import AnalyticTerrainField, TerrainFieldConfig

    baseline = np.load(input_dir / "flow_base" / "predictions.npz")
    guided = np.load(input_dir / "flow_guided" / "predictions.npz")
    effective = json.loads(
        (input_dir / "flow_base" / "config_effective.json").read_text(encoding="utf-8")
    )
    terrain_config = TerrainFieldConfig(**effective["terrain_field"])
    source = CombinedSceneDataset(cache_root, tuple(effective["data"]["source_splits"]))
    cases = json.loads(
        (input_dir / "qualitative_cases" / "case_selection.json").read_text(encoding="utf-8")
    )
    for label, record in cases.items():
        position = int(record["validation_position"])
        dataset_index = int(record["dataset_index"])
        scene = source[dataset_index].as_batch()
        field = AnalyticTerrainField(scene.terrain_map, terrain_config)
        map_h, map_w = scene.terrain_map.shape[-2:]
        x = torch.linspace(0.0, terrain_config.forward_m, map_h)
        y = torch.linspace(-terrain_config.lateral_m, terrain_config.lateral_m, map_w)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        query = torch.stack((xx, yy), dim=-1).reshape(1, -1, 2)
        feasibility = field.query(query).reshape(map_h, map_w).detach().cpu().numpy()
        plot_qualitative_case(
            feasibility,
            (-terrain_config.lateral_m, terrain_config.lateral_m, 0.0, terrain_config.forward_m),
            baseline["ground_truth"][position],
            baseline["trajectories"][position],
            guided["trajectories"][position],
            (
                f"{label}: index={dataset_index}, delta vehicle cost="
                f"{float(record['vehicle_cost_delta']):+.4f}, delta minADE="
                f"{float(record['minADE_delta_m']):+.4f} m"
            ),
            input_dir / "qualitative_cases" / f"{label}_index_{dataset_index}.png",
        )


__all__ = [
    "plot_eta_tradeoffs",
    "plot_guidance_evolution",
    "plot_schedule_comparison",
    "plot_qualitative_case",
    "regenerate_saved_qualitative_cases",
    "read_csv",
]

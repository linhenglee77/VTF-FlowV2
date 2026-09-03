"""Visual diagnostics for structure-preserving feasibility guidance."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_flow_step_structure(
    rows: Sequence[Mapping[str, Any]], output_dir: Path
) -> None:
    """Plot the five required mechanism curves for all main methods."""

    plt = _pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    panels = (
        ("guidance_cost_mean", "clean feasibility cost"),
        ("applied_correction_norm_mean", "applied correction norm"),
        ("correction_flow_ratio_mean", "correction / Flow norm"),
        ("clean_smoothness_mean", "clean-estimate smoothness"),
        ("flow_gradient_cosine_similarity_mean", "cos(Flow, cost gradient)"),
    )
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    figure, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    for axis, (metric, ylabel) in zip(axes.flat, panels):
        for method in methods:
            selected = sorted(
                (row for row in rows if str(row["method"]) == method),
                key=lambda row: int(row["step"]),
            )
            axis.plot(
                [int(row["step"]) for row in selected],
                [float(row[metric]) for row in selected],
                marker="o", ms=2.5, lw=1.3, label=method,
            )
        axis.set_xlabel("Euler step")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=7)
    axes.flat[-1].axis("off")
    figure.savefig(output_dir / "flow_step_structure_diagnostics.png", dpi=190)
    plt.close(figure)
    for metric, ylabel in panels:
        figure, axis = plt.subplots(figsize=(6.3, 4.1), constrained_layout=True)
        for method in methods:
            selected = sorted(
                (row for row in rows if str(row["method"]) == method),
                key=lambda row: int(row["step"]),
            )
            axis.plot(
                [int(row["step"]) for row in selected],
                [float(row[metric]) for row in selected],
                marker="o", ms=2.5, label=method,
            )
        axis.set_xlabel("Euler step")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
        figure.savefig(output_dir / f"step_{metric}.png", dpi=180)
        plt.close(figure)


def plot_main_comparison(
    rows: Sequence[Mapping[str, Any]], output_dir: Path
) -> None:
    """Show feasibility, fidelity, structure, diversity, and latency together."""

    plt = _pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = (
        ("minADE@K_m", "minADE@K (m)"),
        ("mean_vehicle_conditioned_cost", "vehicle cost"),
        ("terrain_violation_rate", "terrain violation"),
        ("smoothness_m", "smoothness (m)"),
        ("mean_waypoint_correction_m", "mean correction (m)"),
        ("latency_ms_per_scene", "latency (ms/scene)"),
    )
    labels = [str(row["method"]) for row in rows]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(2, 3, figsize=(14, 7.2), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, metrics):
        axis.bar(x, [float(row[metric]) for row in rows], color="#0f766e")
        axis.set_xticks(x, labels, rotation=28, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(output_dir / "main_structure_preserving_comparison.png", dpi=190)
    plt.close(figure)


def plot_multi_method_case(
    feasibility_map: np.ndarray,
    extent: tuple[float, float, float, float],
    ground_truth: np.ndarray,
    trajectories: Mapping[str, np.ndarray],
    output_path: Path,
    title: str,
    show_corrections: bool = True,
) -> None:
    """Overlay paired candidate sets and candidate-zero correction vectors."""

    plt = _pyplot()
    colors = {
        "Unguided": "#2563eb",
        "Raw": "#dc2626",
        "Smoothed": "#d97706",
        "Trust region": "#7c3aed",
        "Smooth + trust": "#059669",
        "Adaptive": "#0891b2",
    }
    figure, axis = plt.subplots(figsize=(7.6, 7.2), constrained_layout=True)
    image = axis.imshow(
        feasibility_map, origin="lower", extent=extent, aspect="auto",
        cmap="viridis", vmin=0.0, vmax=1.0,
    )
    base = trajectories["Unguided"]
    for method, candidates in trajectories.items():
        color = colors.get(method, "gray")
        for index, trajectory in enumerate(candidates):
            axis.plot(
                trajectory[:, 1], trajectory[:, 0], color=color,
                alpha=0.22 if method != "Unguided" else 0.16,
                lw=1.0, label=method if index == 0 else None,
            )
        if show_corrections and method != "Unguided":
            start = base[0, ::3]
            delta = candidates[0, ::3] - start
            axis.quiver(
                start[:, 1], start[:, 0], delta[:, 1], delta[:, 0],
                color=color, alpha=0.6, angles="xy", scale_units="xy", scale=1.0,
                width=0.003,
            )
    axis.plot(ground_truth[:, 1], ground_truth[:, 0], color="white", lw=2.7, label="GT")
    axis.scatter([0.0], [0.0], marker="*", s=95, color="red", edgecolor="white")
    axis.set_xlabel("ego lateral y (m)")
    axis.set_ylabel("ego forward x (m)")
    axis.set_title(title, fontsize=10, loc="left")
    axis.legend(fontsize=7, ncol=2)
    figure.colorbar(image, ax=axis, label="terrain-only feasibility context")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=190)
    plt.close(figure)


__all__ = [
    "plot_flow_step_structure",
    "plot_main_comparison",
    "plot_multi_method_case",
]

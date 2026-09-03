"""Create data-grounded modules for the VTF-Flow method framework figure.

Every module uses the same verified RELLIS-3D scene. Terrain panels use the
exact planner BEV, trajectory panels use frozen predictions, and motion-context
panels are reconstructed from the index-aligned raw pose stream.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
import torch


TERRAFLOW_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TERRAFLOW_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from TerraFlow.visualization.plot_method_bev_panels import (  # noqa: E402
    DEFAULT_CACHE_ROOT,
    DEFAULT_FIELD_CONFIG,
    _load_scene,
)
from TerraFlow.datasets.trajectory_builder import (  # noqa: E402
    load_rellis_sequence,
    relative_future_translations,
    rellis3d_os1_to_planning_ego,
)


DEFAULT_OUTPUT_DIR = (
    TERRAFLOW_ROOT
    / "outputs"
    / "final_experiments"
    / "figures"
    / "method_framework"
    / "modules"
)
DEFAULT_SEQUENCE = "00004"
DEFAULT_FRAME = 812
DEFAULT_RELLIS_ROOT = WORKSPACE_ROOT / "data" / "RELLIS3D" / "processed" / "Rellis-3D"
DEFAULT_RAW_FIELD = (
    TERRAFLOW_ROOT / "outputs" / "terrain_fields" / "00004_000812_verified.npz"
)
DEFAULT_PREDICTIONS = (
    TERRAFLOW_ROOT
    / "outputs"
    / "final_experiments"
    / "main_primary_seed0_D"
    / "predictions.npz"
)


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.titlesize": 7.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.6,
            "legend.frameon": False,
        }
    )


def _save_transparent(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", transparent=True)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", transparent=True)
    fig.savefig(
        base.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        transparent=True,
    )
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)


def _rounded_card(
    fig: plt.Figure,
    *,
    edgecolor: str,
    linewidth: float,
) -> None:
    card = FancyBboxPatch(
        (0.015, 0.025),
        0.97,
        0.95,
        boxstyle="round,pad=0.008,rounding_size=0.025",
        transform=fig.transFigure,
        facecolor="white",
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=-10,
        clip_on=False,
    )
    fig.patches.append(card)


def _component_cmap(name: str, colors: Sequence[str]) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, list(colors))


def plot_planner_used_terrain_field(scene: dict, output_dir: Path) -> Path:
    """Render the compact four-component planner terrain-field module."""

    maps = (
        scene["components"]["nontraversable"],
        scene["components"]["slope"],
        scene["components"]["roughness"],
        scene["components"]["clearance"],
    )
    titles = (
        "non-traversability",
        "slope",
        "roughness",
        "obstacle proximity",
    )
    cmaps = (
        _component_cmap("nontraversability_blue", ("#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5")),
        _component_cmap("slope_yellow_green", ("#FFFFE5", "#D9F0A3", "#78C679", "#238443")),
        _component_cmap("roughness_purple", ("#FCFBFD", "#DADAEB", "#9E9AC8", "#6A51A3")),
        _component_cmap("proximity_orange_red", ("#FFF5EB", "#FDD0A2", "#F16913", "#A63603")),
    )
    upper_labels = ("high", "high", "high", "near")
    lower_labels = ("low", "low", "low", "far")

    fig = plt.figure(figsize=(6.25, 2.05), facecolor="none")
    _rounded_card(fig, edgecolor="#66798A", linewidth=0.85)
    fig.text(
        0.5,
        0.91,
        "Planner-used terrain field",
        ha="center",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color="#263746",
    )

    heat_width = 0.135
    heat_height = 0.61
    heat_y = 0.16
    starts = (0.055, 0.285, 0.515, 0.745)
    for values, title, cmap, upper, lower, start in zip(
        maps, titles, cmaps, upper_labels, lower_labels, starts
    ):
        axis = fig.add_axes((start, heat_y, heat_width, heat_height))
        image = axis.imshow(
            values.T,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("#B7C0C8")
            spine.set_linewidth(0.5)
        fig.text(
            start + heat_width / 2.0,
            0.805,
            title,
            ha="center",
            va="bottom",
            fontsize=6.8,
            color="#303A43",
        )

        cbar_axis = fig.add_axes((start + heat_width + 0.012, heat_y, 0.012, heat_height))
        colorbar = fig.colorbar(image, cax=cbar_axis)
        colorbar.set_ticks([])
        colorbar.outline.set_linewidth(0.45)
        colorbar.outline.set_edgecolor("#8B969F")
        fig.text(
            start + heat_width + 0.031,
            heat_y + heat_height,
            upper,
            ha="left",
            va="top",
            fontsize=5.7,
            color="#46515A",
        )
        fig.text(
            start + heat_width + 0.031,
            heat_y,
            lower,
            ha="left",
            va="bottom",
            fontsize=5.7,
            color="#46515A",
        )

    fig.text(
        0.5,
        0.075,
        f"RELLIS-3D {scene['sequence']} · frame {scene['frame']:06d}",
        ha="center",
        va="center",
        fontsize=5.8,
        color="#6A7580",
    )
    base = output_dir / "module_planner_used_terrain_field"
    _save_transparent(fig, base)
    return base


def _load_vtf_flow_candidates(
    predictions_path: Path,
    scene: dict,
) -> np.ndarray:
    archive = np.load(predictions_path, allow_pickle=False)
    scene_id = f"{scene['sequence']}:{scene['frame']}:{scene['split']}"
    lookup = {str(value): index for index, value in enumerate(archive["scene_ids"])}
    if scene_id not in lookup:
        raise KeyError(f"{scene_id} is missing from {predictions_path}")
    index = lookup[scene_id]
    candidates = np.array(archive["trajectories"][index], dtype=np.float32, copy=True)
    prediction_gt = np.array(archive["ground_truth"][index], dtype=np.float32, copy=True)
    archive.close()
    if prediction_gt.shape != scene["ground_truth"].shape:
        raise ValueError("prediction and cache GT shapes differ")
    maximum_difference = float(np.max(np.abs(prediction_gt - scene["ground_truth"])))
    if maximum_difference > 1e-5:
        raise ValueError(
            f"prediction scene does not match cache GT (max difference {maximum_difference})"
        )
    return candidates


def _select_diverse_candidates(candidates: np.ndarray, count: int = 6) -> np.ndarray:
    """Select a deterministic visible subset without using GT or performance."""

    count = min(int(count), int(candidates.shape[0]))
    xy = candidates[..., :2]
    distances = np.linalg.norm(
        xy[:, None, :, :] - xy[None, :, :, :],
        axis=-1,
    ).mean(axis=-1)
    selected = [int(np.argmin(distances.mean(axis=1)))]
    while len(selected) < count:
        distance_to_selected = distances[:, selected].min(axis=1)
        distance_to_selected[selected] = -np.inf
        selected.append(int(np.argmax(distance_to_selected)))
    return np.asarray(selected, dtype=np.int64)


def _with_origin(trajectory: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (np.zeros((1, trajectory.shape[-1]), dtype=trajectory.dtype), trajectory),
        axis=0,
    )


def _load_raw_pose_context(
    scene: dict,
    rellis_root: Path,
    cache_root: Path,
    *,
    history_seconds: float = 1.5,
    history_dt_seconds: float = 0.5,
) -> dict:
    """Reconstruct observed history and the configured goal from raw poses."""

    sequence = load_rellis_sequence(rellis_root / scene["sequence"])
    current_index = int(scene["frame"])
    if history_seconds <= 0.0 or history_dt_seconds <= 0.0:
        raise ValueError("history duration and interval must be positive")
    history_steps = int(round(history_seconds / history_dt_seconds))
    if not np.isclose(history_steps * history_dt_seconds, history_seconds):
        raise ValueError("history_seconds must be divisible by history_dt_seconds")

    current_time = sequence.timestamps[current_index]
    offsets = torch.arange(
        -history_steps,
        1,
        dtype=sequence.timestamps.dtype,
    ) * history_dt_seconds
    target_times = current_time + offsets
    time_errors = torch.abs(sequence.timestamps[:, None] - target_times[None, :])
    history_indices = torch.argmin(time_errors, dim=0)
    history_xyz, current_origin = relative_future_translations(
        sequence.poses[current_index],
        sequence.poses[history_indices],
        rellis3d_os1_to_planning_ego,
    )
    if float(torch.linalg.vector_norm(current_origin)) > 1e-6:
        raise ValueError("raw pose transform does not map the current pose to the origin")

    dataset_config = json.loads(
        (cache_root / "dataset_config.json").read_text(encoding="utf-8")
    )
    horizon_frames = int(dataset_config["horizon_frames"])
    goal_index = current_index + horizon_frames
    if goal_index >= sequence.poses.shape[0]:
        raise IndexError("configured goal horizon extends beyond the pose sequence")
    raw_goal, _ = relative_future_translations(
        sequence.poses[current_index],
        sequence.poses[goal_index : goal_index + 1],
        rellis3d_os1_to_planning_ego,
    )
    raw_goal = raw_goal[0]
    cached_goal = torch.from_numpy(np.asarray(scene["goal"], dtype=np.float64))
    goal_error = float(torch.linalg.vector_norm(raw_goal - cached_goal))
    if goal_error > 1e-4:
        raise ValueError(f"raw-pose goal differs from cached goal by {goal_error:.6g} m")

    return {
        "history_xyz": history_xyz.detach().cpu().numpy().astype(np.float32),
        "history_offsets_s": offsets.detach().cpu().numpy().astype(np.float32),
        "history_frame_indices": history_indices.detach().cpu().numpy().astype(np.int64),
        "goal_xyz": raw_goal.detach().cpu().numpy().astype(np.float32),
        "goal_frame_index": goal_index,
        "goal_horizon_seconds": float(
            sequence.timestamps[goal_index] - sequence.timestamps[current_index]
        ),
        "cached_goal_error_m": goal_error,
    }


def _module_axis_arrows(axis: plt.Axes, x_end: float, y_end: float) -> None:
    """Draw compact local-coordinate arrows without numerical ticks."""

    style = dict(arrowstyle="-|>", mutation_scale=8, lw=0.75, color="#697680")
    axis.add_patch(FancyArrowPatch((0.0, 0.0), (x_end, 0.0), **style, zorder=2))
    axis.add_patch(FancyArrowPatch((0.0, 0.0), (0.0, y_end), **style, zorder=2))


def plot_ego_history_module(
    scene: dict,
    pose_context: dict,
    output_dir: Path,
) -> Path:
    """Render observed ego history reconstructed from the raw pose stream."""

    history = pose_context["history_xyz"]
    offsets = pose_context["history_offsets_s"]
    x_values = history[:, 0]
    y_values = history[:, 1]
    x_min = float(x_values.min()) - 0.28
    x_max = 0.28
    span = x_max - x_min
    y_center = float(0.5 * (y_values.min() + y_values.max()))
    y_min = y_center - 0.5 * span
    y_max = y_center + 0.5 * span

    fig = plt.figure(figsize=(3.25, 2.35), facecolor="none")
    _rounded_card(fig, edgecolor="#4C83B6", linewidth=1.0)
    fig.text(
        0.5, 0.88, r"Ego history $H_t$", ha="center", va="center",
        fontsize=9.1, fontweight="bold", color="#285A85",
    )
    axis = fig.add_axes((0.14, 0.19, 0.72, 0.58))
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.set_aspect("equal")
    axis.axis("off")
    _module_axis_arrows(axis, x_max - 0.07, y_max - 0.07)
    axis.text(x_max - 0.04, -0.06, "x (m)", ha="right", va="top", fontsize=6.2, color="#4E5963")
    axis.text(-0.04, y_max - 0.03, "y (m)", ha="right", va="top", fontsize=6.2, color="#4E5963")

    axis.plot(x_values, y_values, color="#3977B8", lw=1.55, zorder=4)
    colors = ("#B9D2E8", "#8BB6D8", "#5D98C9", "#275F9A")
    for index, (x_value, y_value, offset) in enumerate(zip(x_values, y_values, offsets)):
        color = colors[min(index, len(colors) - 1)]
        axis.scatter(
            [x_value], [y_value], s=28, facecolor=color,
            edgecolor="white", linewidth=0.65, zorder=5,
        )
        label = "t = 0" if np.isclose(offset, 0.0) else f"{offset:.1f} s"
        axis.text(
            x_value, y_value + 0.12, label, ha="center", va="bottom",
            fontsize=5.8, color="#2F4F68", zorder=6,
        )
    axis.scatter([0.0], [0.0], s=10, color="#173A59", zorder=7)
    fig.text(
        0.5, 0.085,
        f"raw poses · RELLIS-3D {scene['sequence']}:{scene['frame']:06d}",
        ha="center", va="center", fontsize=5.7, color="#687580",
    )
    base = output_dir / "module_ego_history"
    _save_transparent(fig, base)
    return base


def plot_goal_point_module(
    scene: dict,
    pose_context: dict,
    output_dir: Path,
) -> Path:
    """Render the raw-pose future endpoint used as the local planning goal."""

    goal = pose_context["goal_xyz"]
    fig = plt.figure(figsize=(3.25, 2.35), facecolor="none")
    _rounded_card(fig, edgecolor="#4C83B6", linewidth=1.0)
    fig.text(
        0.5, 0.88, r"Goal point $g_t$", ha="center", va="center",
        fontsize=9.1, fontweight="bold", color="#285A85",
    )
    axis = fig.add_axes((0.14, 0.19, 0.72, 0.58))
    axis.set_xlim(0.0, 24.0)
    axis.set_ylim(-12.0, 12.0)
    axis.set_aspect("equal")
    axis.axis("off")
    _module_axis_arrows(axis, 23.2, 11.2)
    axis.text(23.1, -0.75, "x (m)", ha="right", va="top", fontsize=6.2, color="#4E5963")
    axis.text(-0.55, 11.2, "y (m)", ha="right", va="top", fontsize=6.2, color="#4E5963")
    axis.plot(
        [0.0, goal[0]], [0.0, goal[1]], color="#8BA9C2",
        lw=0.9, linestyle=(0, (2.0, 2.0)), zorder=3,
    )
    axis.scatter([0.0], [0.0], s=20, color="#263746", zorder=5)
    axis.text(0.7, 0.55, "ego", fontsize=5.8, color="#263746", zorder=6)
    axis.scatter(
        [goal[0]], [goal[1]], s=100, facecolor="#3775BA",
        edgecolor="white", linewidth=1.0, zorder=7,
    )
    axis.scatter([goal[0]], [goal[1]], s=25, facecolor="white", edgecolor="none", zorder=8)
    axis.text(
        goal[0] + 0.7, goal[1] + 0.45, "goal",
        fontsize=6.4, color="#285A85", zorder=8,
    )
    fig.text(
        0.5, 0.085,
        f"raw pose endpoint at +{pose_context['goal_horizon_seconds']:.1f} s · "
        f"({goal[0]:.2f}, {goal[1]:.2f}) m",
        ha="center", va="center", fontsize=5.7, color="#687580",
    )
    base = output_dir / "module_goal_point"
    _save_transparent(fig, base)
    return base


def plot_candidate_trajectories(
    scene: dict,
    candidates: np.ndarray,
    output_dir: Path,
) -> Path:
    """Render actual frozen VTF-Flow samples on their planner obstacle field."""

    fig = plt.figure(figsize=(4.45, 4.05), facecolor="none")
    _rounded_card(fig, edgecolor="#4C83B6", linewidth=1.15)
    fig.text(
        0.5,
        0.91,
        "K candidate trajectories",
        ha="center",
        va="center",
        fontsize=9.4,
        fontweight="bold",
        color="#285A85",
    )
    axis = fig.add_axes((0.14, 0.17, 0.72, 0.67))
    axis.set_xlim(0.0, 24.0)
    axis.set_ylim(-12.0, 12.0)
    axis.set_aspect("equal")
    axis.axis("off")

    arrow_style = dict(arrowstyle="-|>", mutation_scale=8.5, lw=0.8, color="#6D7882")
    axis.add_patch(FancyArrowPatch((0.0, 0.0), (20.0, 0.0), **arrow_style, zorder=2))
    axis.add_patch(FancyArrowPatch((0.0, 0.0), (0.0, 11.6), **arrow_style, zorder=2))
    axis.text(20.1, -0.35, "x (m)", ha="right", va="top", fontsize=6.7, color="#4E5963")
    axis.text(-0.35, 11.65, "y (m)", ha="right", va="top", fontsize=6.7, color="#4E5963")

    proximity = scene["components"]["clearance"]
    occupancy = scene["bev"][1]
    cell_x = (np.arange(proximity.shape[0]) + 0.5) * 24.0 / proximity.shape[0]
    cell_y = -12.0 + (np.arange(proximity.shape[1]) + 0.5) * 24.0 / proximity.shape[1]
    proximity_cmap = _component_cmap(
        "matched_proximity_gray",
        ("#FFFFFF", "#F4F6F8", "#E5E9ED", "#CCD3D9", "#AEB8C2"),
    )
    axis.imshow(
        proximity.T,
        extent=(0.0, 24.0, -12.0, 12.0),
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        cmap=proximity_cmap,
        vmin=0.0,
        vmax=1.0,
        alpha=0.82,
        zorder=0.8,
    )
    if float(np.nanmax(occupancy)) >= 0.25:
        axis.contour(
            cell_x,
            cell_y,
            occupancy.T,
            levels=[0.25],
            colors=["#B5BDC5"],
            linewidths=0.45,
            alpha=0.62,
            zorder=1.7,
        )
    axis.text(
        23.4,
        -11.2,
        "matched proximity field",
        ha="right",
        va="bottom",
        fontsize=5.5,
        color="#707B85",
        zorder=3,
    )

    selected = _select_diverse_candidates(candidates, count=6)
    colors = ("#355C7D", "#2A9D8F", "#74A65A", "#E6A044", "#8C6BB1", "#C96B7E")
    for candidate_index, color in zip(selected, colors):
        path = _with_origin(candidates[candidate_index])
        axis.plot(
            path[:, 0],
            path[:, 1],
            color=color,
            lw=1.65,
            solid_capstyle="round",
            alpha=0.96,
            zorder=5,
        )

    ego = Circle(
        (0.0, 0.0),
        radius=0.28,
        facecolor="white",
        edgecolor="#263746",
        linewidth=1.0,
        zorder=8,
    )
    axis.add_patch(ego)
    axis.scatter([0.0], [0.0], s=10, color="#263746", zorder=9)
    axis.text(0.35, -0.45, "ego origin", ha="left", va="top", fontsize=6.5, color="#263746")

    goal = (float(scene["goal"][0]), float(scene["goal"][1]))
    axis.scatter(
        [goal[0]],
        [goal[1]],
        s=68,
        facecolor="#3775BA",
        edgecolor="white",
        linewidth=1.0,
        zorder=9,
    )
    axis.scatter(
        [goal[0]],
        [goal[1]],
        s=17,
        facecolor="white",
        edgecolor="none",
        zorder=10,
    )
    axis.text(goal[0] + 0.48, goal[1] + 0.22, "goal", fontsize=6.6, color="#285A85")

    direction = FancyArrowPatch(
        (1.0, -1.55),
        (3.0, -1.55),
        arrowstyle="-|>",
        mutation_scale=9,
        lw=1.0,
        color="#4C83B6",
        zorder=7,
    )
    axis.add_patch(direction)
    axis.text(2.02, -2.05, "forward", ha="center", va="top", fontsize=5.8, color="#4C83B6")
    fig.text(
        0.5,
        0.085,
        f"VTF-Flow samples (6 of K={candidates.shape[0]}) · "
        f"RELLIS-3D {scene['sequence']}:{scene['frame']:06d}",
        ha="center",
        va="center",
        fontsize=6.7,
        color="#5A6670",
    )

    base = output_dir / "module_k_candidate_trajectories"
    _save_transparent(fig, base)
    return base


def _write_metadata(
    scene: dict,
    candidates: np.ndarray,
    pose_context: dict,
    predictions_path: Path,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "method_framework_modules_metadata.json"
    payload = {
        "planner_used_terrain_field": {
            "source": "verified planner BEV-derived components",
            "scene": f"{scene['sequence']}:{scene['frame']}",
            "components": [
                "nontraversable",
                "normalized slope",
                "normalized roughness",
                "occupancy-proximity proxy",
            ],
            "range": [0.0, 1.0],
            "display_note": (
                "Full component arrays are shown with a vertically compact display aspect; "
                "the module is explanatory rather than a metric spatial map."
            ),
        },
        "k_candidate_trajectories": {
            "source": "frozen VTF-Flow predictions with matched planner BEV",
            "prediction_archive": str(predictions_path),
            "scene": f"{scene['sequence']}:{scene['frame']}:{scene['split']}",
            "candidate_count_available": int(candidates.shape[0]),
            "candidate_count_shown": min(6, int(candidates.shape[0])),
            "selected_candidate_indices": _select_diverse_candidates(
                candidates, count=6
            ).tolist(),
            "selection_rule": (
                "Deterministic farthest-first subset in pairwise mean trajectory distance; "
                "selection does not use GT error or terrain score."
            ),
            "note": (
                "Trajectories are actual frozen model samples. The grey background is the "
                "complete matched planner occupancy-proximity field, internal contours come "
                "from matched obstacle density, and the goal is the matched cached scene goal."
            ),
            "display_extent_m": {
                "forward": [0.0, 24.0],
                "lateral": [-12.0, 12.0],
            },
        },
        "ego_history": {
            "source": "index-aligned raw RELLIS-3D poses",
            "scene": f"{scene['sequence']}:{scene['frame']}",
            "frame_indices": pose_context["history_frame_indices"].tolist(),
            "time_offsets_s": pose_context["history_offsets_s"].tolist(),
            "xyz_m": pose_context["history_xyz"].tolist(),
            "coordinate_frame": "current planning-ego (x forward, y left, z up)",
        },
        "goal_point": {
            "source": "raw RELLIS-3D pose at the configured future horizon",
            "scene": f"{scene['sequence']}:{scene['frame']}",
            "goal_frame_index": int(pose_context["goal_frame_index"]),
            "horizon_seconds": float(pose_context["goal_horizon_seconds"]),
            "xyz_m": pose_context["goal_xyz"].tolist(),
            "cached_goal_agreement_error_m": float(
                pose_context["cached_goal_error_m"]
            ),
            "coordinate_frame": "current planning-ego (x forward, y left, z up)",
        },
        "export": {
            "transparent_outside_rounded_card": True,
            "formats": ["svg", "pdf", "png"],
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render compact VTF-Flow method-framework modules."
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--rellis-root", type=Path, default=DEFAULT_RELLIS_ROOT)
    parser.add_argument("--raw-field", type=Path, default=DEFAULT_RAW_FIELD)
    parser.add_argument("--field-config", type=Path, default=DEFAULT_FIELD_CONFIG)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE)
    parser.add_argument("--frame", type=int, default=DEFAULT_FRAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_matplotlib()
    scene = _load_scene(
        cache_root=args.cache_root,
        raw_field_path=args.raw_field,
        field_config_path=args.field_config,
        sequence=args.sequence,
        frame=args.frame,
    )
    candidates = _load_vtf_flow_candidates(args.predictions, scene)
    pose_context = _load_raw_pose_context(scene, args.rellis_root, args.cache_root)
    terrain_base = plot_planner_used_terrain_field(scene, args.output_dir)
    candidates_base = plot_candidate_trajectories(scene, candidates, args.output_dir)
    history_base = plot_ego_history_module(scene, pose_context, args.output_dir)
    goal_base = plot_goal_point_module(scene, pose_context, args.output_dir)
    metadata = _write_metadata(
        scene, candidates, pose_context, args.predictions, args.output_dir
    )
    print(f"terrain module: {terrain_base}")
    print(f"candidate module: {candidates_base}")
    print(f"ego-history module: {history_base}")
    print(f"goal module: {goal_base}")
    print(f"metadata: {metadata}")


if __name__ == "__main__":
    main()

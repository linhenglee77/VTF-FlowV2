"""Render real-data benchmark and terrain-mechanism figures for VTF-Flow.

The script uses the frozen predictions from the unified H=10, 5 s benchmark.
It creates two complementary figure families in a new output directory:

1. six-method benchmark small multiples on the same raw LiDAR-derived terrain
   potential; and
2. terrain-mechanism figures that retain only Flow, VTF-Flow and GT overlays so
   that the measured terrain attributes remain readable.

Generative methods show every candidate faintly and emphasize the minimum-ADE
candidate only as an explicitly retrospective best-of-K diagnostic.  The
selection uses GT and is not presented as an online trajectory selector.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch


TERRAFLOW_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TERRAFLOW_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from TerraFlow.scripts.build_terrain_field import build_archive  # noqa: E402
from TerraFlow.terrain.feasibility_field import (  # noqa: E402
    ContinuousTerrainField,
    load_terrain_field_config,
)
from TerraFlow.terrain.trajectory_kinematics import (  # noqa: E402
    TrajectoryKinematicConfig,
)
from TerraFlow.terrain.vehicle_conditioned_field import (  # noqa: E402
    VehicleConditionedTerrainField,
    load_vehicle_conditioned_config,
)
from TerraFlow.visualization.plot_raw_terrain_trajectory import (  # noqa: E402
    _features_from_archive,
    _sample_metrics,
)


DEFAULT_BENCHMARK_ROOT = TERRAFLOW_ROOT / "outputs" / "unified_h10_benchmark"
DEFAULT_FIELD_CONFIG = TERRAFLOW_ROOT / "configs" / "rellis3d_terrain_field.json"
DEFAULT_SENSOR_TRANSFORM = TERRAFLOW_ROOT / "configs" / "rellis3d_os1_to_planning_ego.json"
DEFAULT_TVK_CONFIG = TERRAFLOW_ROOT / "configs" / "final_tvk_validation.json"

METHOD_ORDER = ("CV", "ASTAR", "REG", "FLOW", "VT", "VTF")
DISPLAY_NAMES = {
    "CV": "Constant velocity",
    "ASTAR": "A* terrain planner",
    "REG": "Deterministic regression",
    "FLOW": "Flow Matching",
    "VT": "VTF-Flow w/o kinematic feasibility",
    "VTF": "VTF-Flow",
}
METHOD_COLORS = {
    "CV": "#8A919B",
    "ASTAR": "#7A6F9B",
    "REG": "#2F6B9A",
    "FLOW": "#56B4E9",
    "VT": "#4C956C",
    "VTF": "#E69F00",
}
GT_COLOR = "#15191F"
ORIGIN_COLOR = "#D83A34"
GOAL_COLOR = "#2155A6"
TERRAIN_COST_VMAX = 5.5
FINAL_WIDTH_IN = 7.2


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.titlesize": 7.4,
            "axes.labelsize": 7.0,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.fontsize": 6.4,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _save_all(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", transparent=True)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", transparent=True)
    fig.savefig(base.with_suffix(".png"), dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def _load_unified_predictions(
    benchmark_root: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    trajectories: dict[str, np.ndarray] = {}
    ground_truth: np.ndarray | None = None
    scene_ids: list[str] | None = None
    for method in METHOD_ORDER:
        run = benchmark_root / "runs" / f"{method}_seed0"
        prediction_path = run / "predictions.npz"
        metrics_path = run / "scene_level_metrics.csv"
        if not prediction_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(f"missing unified benchmark run for {method}: {run}")
        with np.load(prediction_path, allow_pickle=False) as archive:
            trajectories[method] = archive["trajectories"].copy()
            current_gt = archive["ground_truth"].copy()
        current_ids = pd.read_csv(metrics_path)["scene_id"].astype(str).tolist()
        if ground_truth is None:
            ground_truth = current_gt
            scene_ids = current_ids
        elif not np.array_equal(ground_truth, current_gt) or current_ids != scene_ids:
            raise ValueError(f"{method} predictions are not aligned with the shared benchmark")
    assert ground_truth is not None and scene_ids is not None
    if ground_truth.shape != (1909, 10, 3):
        raise ValueError(f"expected unified [1909,10,3] GT, got {ground_truth.shape}")
    return trajectories, ground_truth, scene_ids


def _minimum_ade_candidate(candidates: np.ndarray, ground_truth: np.ndarray) -> int:
    if candidates.ndim != 3 or candidates.shape[1:] != ground_truth.shape:
        raise ValueError("candidate and ground-truth trajectory shapes are incompatible")
    ade = np.linalg.norm(candidates - ground_truth[None, ...], axis=-1).mean(axis=-1)
    return int(np.argmin(ade))


def _trajectory_geometry(ground_truth: np.ndarray) -> dict[str, np.ndarray]:
    origins = np.zeros((len(ground_truth), 1, 3), dtype=ground_truth.dtype)
    full = np.concatenate((origins, ground_truth), axis=1)
    segments = np.diff(full, axis=1)
    return {
        "path_length_m": np.linalg.norm(segments, axis=-1).sum(axis=-1),
        "curvature_proxy": np.linalg.norm(
            np.diff(full[:, :, :2], n=2, axis=1), axis=-1
        ).sum(axis=-1),
        "final_x_m": ground_truth[:, -1, 0],
        "final_y_m": ground_truth[:, -1, 1],
        "maximum_abs_y_m": np.abs(ground_truth[:, :, 1]).max(axis=-1),
        "minimum_x_m": ground_truth[:, :, 0].min(axis=-1),
        "maximum_x_m": ground_truth[:, :, 0].max(axis=-1),
    }


def _select_mechanism_scenes(
    ground_truth: np.ndarray,
    scene_ids: list[str],
    *,
    minimum_frame_separation: int,
) -> list[dict[str, Any]]:
    """Select six H=10 scenes from GT geometry without using method outcomes."""

    geometry = _trajectory_geometry(ground_truth)
    valid = (
        (geometry["minimum_x_m"] >= -0.05)
        & (geometry["maximum_x_m"] <= 23.5)
        & (geometry["maximum_abs_y_m"] <= 11.5)
        & (geometry["path_length_m"] >= 0.25)
        & (geometry["final_x_m"] >= 2.0)
    )
    median_length = float(np.median(geometry["path_length_m"][valid]))
    score_rules = (
        ("long_path", geometry["path_length_m"]),
        ("left_turn", geometry["final_y_m"]),
        ("right_turn", -geometry["final_y_m"]),
        ("high_curvature", geometry["curvature_proxy"]),
        (
            "typical_straight",
            -(
                np.abs(geometry["path_length_m"] - median_length)
                + 2.0 * np.abs(geometry["final_y_m"])
            ),
        ),
        ("short_path", -geometry["path_length_m"]),
    )
    selected: list[dict[str, Any]] = []
    for category, score in score_rules:
        for index in np.argsort(score)[::-1]:
            if not bool(valid[index]):
                continue
            sequence, frame_text, _ = str(scene_ids[index]).split(":")
            frame = int(frame_text)
            if any(
                sequence == row["sequence"]
                and abs(frame - int(row["frame_id"])) < minimum_frame_separation
                for row in selected
            ):
                continue
            selected.append(
                {
                    "category": category,
                    "scene_id": str(scene_ids[index]),
                    "sequence": sequence,
                    "frame_id": frame,
                    "prediction_index": int(index),
                    "path_length_m": float(geometry["path_length_m"][index]),
                    "final_x_m": float(geometry["final_x_m"][index]),
                    "final_y_m": float(geometry["final_y_m"][index]),
                    "curvature_proxy": float(geometry["curvature_proxy"][index]),
                }
            )
            break
    if len(selected) != len(score_rules):
        raise RuntimeError("could not select all six separated H=10 mechanism scenes")
    return selected


def _load_benchmark_scenes(
    benchmark_root: Path,
    scene_lookup: Mapping[str, int],
) -> list[dict[str, Any]]:
    path = (
        benchmark_root
        / "figure_source_data"
        / "selected_advantage_scenes"
        / "selection_manifest.json"
    )
    manifest = _read_json(path)
    scenes: list[dict[str, Any]] = []
    for raw in manifest["selected_scenes"]:
        record = dict(raw)
        scene_id = str(record["scene_id"])
        if scene_id not in scene_lookup:
            raise KeyError(f"selected benchmark scene is absent from frozen predictions: {scene_id}")
        record["prediction_index"] = int(scene_lookup[scene_id])
        scenes.append(record)
    return scenes


def _build_field_if_needed(
    data_root: Path,
    field_dir: Path,
    record: Mapping[str, Any],
    field_config: Path,
    sensor_transform: Path,
    *,
    rebuild: bool,
) -> Path:
    sequence = str(record["sequence"])
    frame = int(record["frame_id"])
    field_path = field_dir / f"{sequence}_{frame:06d}_verified.npz"
    if rebuild or not field_path.is_file():
        build_archive(
            argparse.Namespace(
                data_root=data_root,
                sequence=sequence,
                frame=frame,
                sensor="ouster",
                config=field_config,
                output=field_path,
                sensor_to_ego=sensor_transform,
                allow_unverified_identity=False,
                geometry_only=False,
            )
        )
    return field_path


def _load_field_context(
    field_path: Path,
    field_config: Path,
) -> tuple[dict[str, np.ndarray], Any, ContinuousTerrainField, VehicleConditionedTerrainField, set[int]]:
    with np.load(field_path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    if str(data["coordinate_status"]) != "ego_from_explicit_T_ego_sensor":
        raise ValueError(f"field does not use the verified explicit transform: {field_path}")
    definition = load_terrain_field_config(field_config)
    features = _features_from_archive(data)
    terrain_field = ContinuousTerrainField(features, definition.cost)
    rebuilt = terrain_field.cost_map[0, 0].detach().cpu().numpy()
    if not np.allclose(rebuilt, data["terrain_cost"], atol=1e-6):
        raise ValueError(f"serialized and rebuilt terrain potential disagree: {field_path}")
    vehicle_field = VehicleConditionedTerrainField(
        terrain_field, load_vehicle_conditioned_config(field_config)
    )
    policy = json.loads(str(data["semantic_policy_json"]))
    obstacle_ids = {
        int(label) for label, entry in policy.items() if entry["role"] == "obstacle"
    }
    return data, features, terrain_field, vehicle_field, obstacle_ids


def _with_origin(trajectory: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (np.zeros((1, trajectory.shape[-1]), dtype=trajectory.dtype), trajectory), axis=0
    )


def _draw_path(
    axis: plt.Axes,
    trajectory: np.ndarray,
    color: str,
    *,
    linewidth: float,
    linestyle: str = "-",
    alpha: float = 1.0,
    zorder: int = 6,
    halo: bool = True,
) -> None:
    path = _with_origin(trajectory)
    (line,) = axis.plot(
        path[:, 0],
        path[:, 1],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder,
        solid_capstyle="round",
    )
    if halo:
        line.set_path_effects(
            [
                path_effects.Stroke(linewidth=linewidth + 1.0, foreground="white", alpha=0.9),
                path_effects.Normal(),
            ]
        )


def _draw_origin_goal(axis: plt.Axes, ground_truth: np.ndarray) -> None:
    axis.scatter(
        [0.0],
        [0.0],
        marker="*",
        s=42,
        facecolor=ORIGIN_COLOR,
        edgecolor="white",
        linewidth=0.6,
        clip_on=False,
        zorder=12,
    )
    goal = ground_truth[-1]
    axis.scatter(
        [goal[0]],
        [goal[1]],
        marker="o",
        s=30,
        facecolor="white",
        edgecolor=GOAL_COLOR,
        linewidth=1.0,
        zorder=11,
    )
    axis.scatter(
        [goal[0]], [goal[1]], marker="+", s=34, color=GOAL_COLOR, linewidth=0.9, zorder=12
    )


def _format_axis(axis: plt.Axes, extent: tuple[float, float, float, float]) -> None:
    axis.set_xlim(extent[0], extent[1])
    axis.set_ylim(extent[2], extent[3])
    axis.set_aspect("equal")
    axis.grid(False)


def _planning_view_extent(
    full_extent: tuple[float, float, float, float],
    paths: Iterable[np.ndarray],
) -> tuple[float, float, float, float]:
    """Return a shared crop around the H=10 planning envelope.

    The complete terrain grid is retained in source data.  Cropping only changes
    the displayed viewport so that 5 s trajectory differences remain legible.
    """

    xy = np.concatenate(
        [np.zeros((1, 2), dtype=np.float32)]
        + [np.asarray(path)[..., :2].reshape(-1, 2) for path in paths],
        axis=0,
    )
    finite = np.isfinite(xy).all(axis=1)
    xy = xy[finite]
    if not len(xy):
        return full_extent
    x_min = max(float(full_extent[0]), min(0.0, float(np.min(xy[:, 0])) - 0.5))
    x_max = min(float(full_extent[1]), max(8.0, float(np.max(xy[:, 0])) + 1.5))
    y_min = max(float(full_extent[2]), min(-4.0, float(np.min(xy[:, 1])) - 2.0))
    y_max = min(float(full_extent[3]), max(4.0, float(np.max(xy[:, 1])) + 2.0))
    if y_max - y_min < 8.0:
        center = 0.5 * (y_min + y_max)
        y_min = max(float(full_extent[2]), center - 4.0)
        y_max = min(float(full_extent[3]), center + 4.0)
    return x_min, x_max, y_min, y_max


def _selected_trajectories_and_metrics(
    prediction_index: int,
    trajectories: Mapping[str, np.ndarray],
    ground_truth: np.ndarray,
    terrain_field: ContinuousTerrainField,
    vehicle_field: VehicleConditionedTerrainField,
    features: Any,
    field_config: Path,
    obstacle_ids: set[int],
    kinematic_config: TrajectoryKinematicConfig,
) -> tuple[dict[str, int], dict[str, np.ndarray], list[dict[str, Any]]]:
    gt = ground_truth[prediction_index]
    selected: dict[str, int] = {}
    selected_paths: dict[str, np.ndarray] = {}
    metrics: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        candidates = trajectories[method][prediction_index]
        choice = _minimum_ade_candidate(candidates, gt)
        selected[method] = choice
        selected_paths[method] = candidates[choice]
        row = _sample_metrics(
            DISPLAY_NAMES[method],
            candidates[choice],
            terrain_field,
            vehicle_field,
            features,
            field_config,
            obstacle_ids,
            0.5,
            kinematic_config,
        )
        row["candidate_index"] = choice
        row["selection"] = "oracle minimum ADE"
        row["ADE_m"] = float(
            np.linalg.norm(candidates[choice] - gt, axis=-1).mean()
        )
        metrics.append(row)
    gt_row = _sample_metrics(
        "GT",
        gt,
        terrain_field,
        vehicle_field,
        features,
        field_config,
        obstacle_ids,
        0.5,
        kinematic_config,
    )
    gt_row["candidate_index"] = 0
    gt_row["selection"] = "recorded future trajectory"
    gt_row["ADE_m"] = 0.0
    metrics.append(gt_row)
    return selected, selected_paths, metrics


def _write_scene_source_data(
    source_root: Path,
    record: Mapping[str, Any],
    data: Mapping[str, np.ndarray],
    trajectories: Mapping[str, np.ndarray],
    ground_truth: np.ndarray,
    selected: Mapping[str, int],
    metrics: list[dict[str, Any]],
) -> dict[str, str]:
    sequence = str(record["sequence"])
    frame = int(record["frame_id"])
    scene_dir = source_root / f"{sequence}_{frame:06d}"
    scene_dir.mkdir(parents=True, exist_ok=True)
    grid_path = scene_dir / "terrain_grid.csv"
    resolution = float(data["resolution_m"])
    x_min = float(data["x_min_m"])
    y_min = float(data["y_min_m"])
    names = (
        "terrain_cost",
        "feasibility",
        "elevation_m",
        "slope_deg",
        "roughness_m",
        "semantic_class",
        "occupancy",
        "clearance_m",
        "geometry_valid",
        "slope_valid",
        "semantic_valid",
    )
    with grid_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row", "col", "x_m", "y_m", *names],
        )
        writer.writeheader()
        height, width = data["terrain_cost"].shape
        for row in range(height):
            for col in range(width):
                payload: dict[str, Any] = {
                    "row": row,
                    "col": col,
                    "x_m": x_min + (col + 0.5) * resolution,
                    "y_m": y_min + (row + 0.5) * resolution,
                }
                payload.update({name: data[name][row, col] for name in names})
                writer.writerow(payload)

    trajectory_path = scene_dir / "trajectories.csv"
    with trajectory_path.open("w", newline="", encoding="utf-8") as handle:
        fields = (
            "method",
            "candidate",
            "selected_oracle_minade",
            "waypoint",
            "time_s",
            "x_m",
            "y_m",
            "z_m",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in METHOD_ORDER:
            for candidate_index, candidate in enumerate(trajectories[method]):
                for waypoint, point in enumerate(candidate):
                    writer.writerow(
                        {
                            "method": DISPLAY_NAMES[method],
                            "candidate": candidate_index,
                            "selected_oracle_minade": int(
                                candidate_index == selected[method]
                            ),
                            "waypoint": waypoint + 1,
                            "time_s": (waypoint + 1) * 0.5,
                            "x_m": point[0],
                            "y_m": point[1],
                            "z_m": point[2],
                        }
                    )
        for waypoint, point in enumerate(ground_truth):
            writer.writerow(
                {
                    "method": "GT",
                    "candidate": 0,
                    "selected_oracle_minade": 1,
                    "waypoint": waypoint + 1,
                    "time_s": (waypoint + 1) * 0.5,
                    "x_m": point[0],
                    "y_m": point[1],
                    "z_m": point[2],
                }
            )
    metrics_path = scene_dir / "selected_trajectory_metrics.csv"
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    metadata_path = scene_dir / "scene_metadata.json"
    metadata = {
        "scene_id": record["scene_id"],
        "sequence": sequence,
        "frame": frame,
        "selection_category": record["category"],
        "prediction_index": int(record["prediction_index"]),
        "trajectory_protocol": {"H": 10, "dt_s": 0.5, "horizon_s": 5.0},
        "candidate_selection": (
            "minimum ADE using GT; retrospective best-of-K visualization only, "
            "not an online selector"
        ),
        "coordinate_convention": "planning ego: x forward, y left, z up",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "terrain_grid": str(grid_path.resolve()),
        "trajectories": str(trajectory_path.resolve()),
        "selected_metrics": str(metrics_path.resolve()),
        "metadata": str(metadata_path.resolve()),
    }


def _benchmark_figure(
    output_dir: Path,
    record: Mapping[str, Any],
    data: Mapping[str, np.ndarray],
    trajectories: Mapping[str, np.ndarray],
    ground_truth: np.ndarray,
    selected: Mapping[str, int],
    metrics: list[dict[str, Any]],
) -> Path:
    sequence = str(record["sequence"])
    frame = int(record["frame_id"])
    extent = (
        float(data["x_min_m"]),
        float(data["x_max_m"]),
        float(data["y_min_m"]),
        float(data["y_max_m"]),
    )
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(FINAL_WIDTH_IN, 5.1),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    metric_lookup = {str(row["method"]): row for row in metrics}
    images = []
    gt = ground_truth
    terrain_cmap = mpl.colormaps["magma"].copy()
    terrain_cmap.set_bad("white")
    terrain_cost = np.asarray(data["terrain_cost"], dtype=np.float32)
    view_extent = _planning_view_extent(
        extent,
        [ground_truth]
        + [candidate for method in METHOD_ORDER for candidate in trajectories[method]],
    )
    for panel_index, (axis, method) in enumerate(zip(axes.flat, METHOD_ORDER)):
        image = axis.imshow(
            terrain_cost,
            origin="lower",
            extent=extent,
            cmap=terrain_cmap,
            vmin=0.0,
            vmax=TERRAIN_COST_VMAX,
            interpolation="bilinear",
            aspect="equal",
            rasterized=True,
        )
        images.append(image)
        candidates = trajectories[method]
        choice = int(selected[method])
        for candidate_index, candidate in enumerate(candidates):
            is_selected = candidate_index == choice
            _draw_path(
                axis,
                candidate,
                METHOD_COLORS[method],
                linewidth=1.65 if is_selected else 0.55,
                alpha=1.0 if is_selected else 0.16,
                zorder=7 if is_selected else 4,
                halo=is_selected,
            )
        _draw_path(axis, gt, GT_COLOR, linewidth=1.35, zorder=9)
        _draw_origin_goal(axis, gt)
        _format_axis(axis, view_extent)
        letter = chr(ord("a") + panel_index)
        axis.set_title(
            f"{letter}  {DISPLAY_NAMES[method]}  (K={len(candidates)})",
            loc="left",
            fontweight="bold" if method == "VTF" else "normal",
            color="#8A4B00" if method == "VTF" else "#20252B",
        )
        row = metric_lookup[DISPLAY_NAMES[method]]
        axis.text(
            0.025,
            0.025,
            f"oracle ADE {row['ADE_m']:.3f} m\nraw-field TVK {row['mean_unified_tvk_cost']:.3f}",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=5.5,
            color="#20252B",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "alpha": 0.86,
                "edgecolor": "#CFD5DC",
                "linewidth": 0.45,
            },
            zorder=12,
        )
    for axis in axes[-1, :]:
        axis.set_xlabel("Ego-forward x (m)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Ego-left y (m)")
    handles = [
        Line2D([0], [0], color=GT_COLOR, lw=1.8, label="Recorded GT"),
        Line2D([0], [0], color="#64748B", lw=0.7, alpha=0.35, label="All candidates"),
        Line2D([0], [0], color="#334155", lw=1.8, label="Highlighted: oracle minADE"),
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor=ORIGIN_COLOR,
            markeredgecolor="white",
            markersize=7,
            label="Ego origin",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor=GOAL_COLOR,
            markersize=5,
            label="5 s goal",
        ),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.025))
    colorbar = fig.colorbar(images[0], ax=axes, fraction=0.025, pad=0.018, shrink=0.90)
    colorbar.set_label("Static terrain potential (lower → more feasible)")
    fig.suptitle(
        f"Unified benchmark trajectories on raw RELLIS-3D terrain | "
        f"sequence {sequence}, frame {frame:06d}",
        y=1.075,
        fontsize=8.0,
    )
    fig.text(
        0.5,
        -0.01,
        "All panels share the same field and axes. Oracle minADE uses GT only for retrospective visualization; "
        "raw-field TVK values are relative diagnostics, not calibrated safety probabilities.",
        ha="center",
        va="bottom",
        fontsize=5.5,
    )
    base = output_dir / "benchmark_comparison" / f"benchmark_{sequence}_{frame:06d}"
    _save_all(fig, base)
    return base


def _masked(values: np.ndarray, mask: np.ndarray | None) -> np.ma.MaskedArray | np.ndarray:
    return np.ma.array(values, mask=mask) if mask is not None else values


def _mechanism_figure(
    output_dir: Path,
    record: Mapping[str, Any],
    data: Mapping[str, np.ndarray],
    selected_paths: Mapping[str, np.ndarray],
    ground_truth: np.ndarray,
    metrics: list[dict[str, Any]],
) -> Path:
    sequence = str(record["sequence"])
    frame = int(record["frame_id"])
    extent = (
        float(data["x_min_m"]),
        float(data["x_max_m"]),
        float(data["y_min_m"]),
        float(data["y_max_m"]),
    )
    geometry_mask = ~np.asarray(data["geometry_valid"], dtype=bool)
    slope_mask = ~np.asarray(data["slope_valid"], dtype=bool)
    semantic_mask = ~np.asarray(data["semantic_valid"], dtype=bool)
    policy = json.loads(str(data["semantic_policy_json"]))
    semantic_values = np.asarray(data["semantic_class"], dtype=np.int64)
    encountered = sorted(int(value) for value in np.unique(semantic_values[~semantic_mask]))
    semantic_display = np.full_like(semantic_values, np.nan, dtype=np.float32)
    for index, label_id in enumerate(encountered):
        semantic_display[semantic_values == label_id] = index
    semantic_cmap = ListedColormap(
        mpl.colormaps["tab10"](np.linspace(0.0, 1.0, max(len(encountered), 1)))
    )
    semantic_norm = BoundaryNorm(np.arange(-0.5, len(encountered) + 0.5), semantic_cmap.N)

    fig = plt.figure(figsize=(FINAL_WIDTH_IN, 5.2), layout="constrained")
    layout = fig.add_gridspec(3, 4, width_ratios=(1.42, 1.42, 1.0, 1.0))
    hero = fig.add_subplot(layout[:, :2])
    supports = [fig.add_subplot(layout[row, col]) for row in range(3) for col in range(2, 4)]
    terrain_cmap = mpl.colormaps["magma"].copy()
    terrain_cmap.set_bad("white")
    hero_image = hero.imshow(
        np.asarray(data["terrain_cost"], dtype=np.float32),
        origin="lower",
        extent=extent,
        cmap=terrain_cmap,
        vmin=0.0,
        vmax=TERRAIN_COST_VMAX,
        interpolation="bilinear",
        aspect="equal",
        rasterized=True,
    )
    view_extent = _planning_view_extent(
        extent,
        [ground_truth, selected_paths["FLOW"], selected_paths["VTF"]],
    )
    _draw_path(
        hero,
        selected_paths["FLOW"],
        METHOD_COLORS["FLOW"],
        linewidth=1.3,
        linestyle="--",
        zorder=8,
    )
    _draw_path(hero, selected_paths["VTF"], METHOD_COLORS["VTF"], linewidth=1.8, zorder=9)
    _draw_path(hero, ground_truth, GT_COLOR, linewidth=1.35, zorder=10)
    _draw_origin_goal(hero, ground_truth)
    _format_axis(hero, view_extent)
    hero.set_xlabel("Ego-forward x (m)")
    hero.set_ylabel("Ego-left y (m)")
    hero.set_title("a  Static terrain-feasibility potential", loc="left", fontweight="bold")
    hero.legend(
        handles=[
            Line2D([0], [0], color=METHOD_COLORS["FLOW"], lw=1.5, ls="--", label="Flow (oracle minADE)"),
            Line2D([0], [0], color=METHOD_COLORS["VTF"], lw=1.8, label="VTF-Flow (oracle minADE)"),
            Line2D([0], [0], color=GT_COLOR, lw=1.5, label="Recorded GT"),
        ],
        loc="upper left",
        frameon=True,
        facecolor="white",
        framealpha=0.86,
        edgecolor="#CBD5E1",
    )
    hero_bar = fig.colorbar(hero_image, ax=hero, fraction=0.035, pad=0.02)
    hero_bar.set_label("Terrain potential (lower → more feasible)")
    metric_lookup = {str(row["method"]): row for row in metrics}
    hero.text(
        0.98,
        0.02,
        "Raw-field TVK potential\n"
        f"Flow {metric_lookup[DISPLAY_NAMES['FLOW']]['mean_unified_tvk_cost']:.3f}  |  "
        f"VTF-Flow {metric_lookup[DISPLAY_NAMES['VTF']]['mean_unified_tvk_cost']:.3f}\n"
        f"GT {metric_lookup['GT']['mean_unified_tvk_cost']:.3f}",
        transform=hero.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.5,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "alpha": 0.88,
            "edgecolor": "#CBD5E1",
            "linewidth": 0.45,
        },
        zorder=12,
    )

    valid_elevation = np.asarray(data["elevation_m"])[~geometry_mask]
    elevation_limits = (
        float(np.nanpercentile(valid_elevation, 2.0)),
        float(np.nanpercentile(valid_elevation, 98.0)),
    ) if valid_elevation.size else (None, None)
    valid_roughness = np.asarray(data["roughness_m"])[~geometry_mask]
    roughness_vmax = float(np.nanpercentile(valid_roughness, 99.0)) if valid_roughness.size else None
    component_specs: tuple[tuple[str, Any, Any, str, np.ndarray | None, float | None, float | None], ...] = (
        ("b  Elevation", data["elevation_m"], "terrain", "m", geometry_mask, *elevation_limits),
        ("c  Local slope", data["slope_deg"], "magma", "degrees", slope_mask, 0.0, 65.0),
        ("d  Height roughness", data["roughness_m"], "cividis", "m (SD)", geometry_mask, 0.0, roughness_vmax),
        ("semantic", semantic_display, semantic_cmap, "", semantic_mask, None, None),
        ("f  Obstacle occupancy", data["occupancy"], "gray_r", "occupied", None, 0.0, 1.0),
        ("g  Obstacle clearance", data["clearance_m"], "Blues", "m", None, 0.0, 4.0),
    )
    for axis, spec in zip(supports, component_specs):
        title, values, cmap, label, mask, vmin, vmax = spec
        shown = _masked(np.asarray(values), mask)
        if title == "semantic":
            image = axis.imshow(
                shown,
                origin="lower",
                extent=extent,
                cmap=semantic_cmap,
                norm=semantic_norm,
                interpolation="nearest",
                aspect="equal",
                rasterized=True,
            )
            axis.set_title("e  Dominant semantic class", loc="left", fontweight="bold")
            colorbar = fig.colorbar(
                image, ax=axis, fraction=0.046, pad=0.03, ticks=np.arange(len(encountered))
            )
            colorbar.ax.set_yticklabels(
                [policy.get(str(value), {}).get("name", str(value)) for value in encountered]
            )
        else:
            image = axis.imshow(
                shown,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                aspect="equal",
                rasterized=True,
            )
            axis.set_title(title, loc="left", fontweight="bold")
            colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
            colorbar.set_label(label, fontsize=5.5, labelpad=2)
        colorbar.ax.tick_params(labelsize=5.2, width=0.5, length=2)
        _draw_path(axis, ground_truth, GT_COLOR, linewidth=1.0, linestyle="--", zorder=8)
        _draw_path(axis, selected_paths["VTF"], METHOD_COLORS["VTF"], linewidth=1.45, zorder=9)
        axis.scatter(
            [0.0], [0.0], marker="*", s=25, facecolor=ORIGIN_COLOR,
            edgecolor="white", linewidth=0.5, clip_on=False, zorder=10,
        )
        _format_axis(axis, view_extent)
    for index, axis in enumerate(supports):
        row, col = divmod(index, 2)
        if row == 2:
            axis.set_xlabel("forward x (m)", fontsize=5.5)
        else:
            axis.set_xticklabels([])
        if col == 0:
            axis.set_ylabel("lateral y (m)", fontsize=5.5)
        else:
            axis.set_yticklabels([])
    fig.suptitle(
        f"Raw RELLIS-3D terrain attributes and unified H=10 trajectories | "
        f"sequence {sequence}, frame {frame:06d}",
        fontsize=8.0,
    )
    fig.text(
        0.5,
        -0.01,
        "Support panels overlay VTF-Flow (orange) and recorded GT (black). White cells are unobserved. "
        "Potentials are relative diagnostics, not calibrated safety probabilities or thresholds.",
        ha="center",
        va="bottom",
        fontsize=5.5,
    )
    base = output_dir / "terrain_mechanism" / f"mechanism_{sequence}_{frame:06d}"
    _save_all(fig, base)
    return base


def _write_text_assets(output_root: Path) -> None:
    caption = """Benchmark comparison figures. Each six-panel figure uses the same raw LiDAR-derived static terrain potential and coordinate limits for all methods. Deterministic methods provide one path; generative methods show all eight frozen candidates and emphasize the minimum-ADE member as a retrospective best-of-K diagnostic only. Recorded GT, ego origin and the common 5 s goal are shown in every panel. Raw-field TVK potentials are relative model diagnostics rather than calibrated safety probabilities.\n\nTerrain-mechanism figures. The hero panel compares Flow Matching, VTF-Flow and the recorded GT on the same static terrain potential. Supporting panels decompose the measured elevation, local slope, height roughness, semantic class, obstacle occupancy and obstacle clearance. Only VTF-Flow and GT are retained in the supporting panels to keep the raw terrain evidence visible.\n"""
    (output_root / "figure_captions.txt").write_text(caption, encoding="utf-8")
    qa = """# Unified raw-terrain scene figures — QA contract

## Figure contract

- Core conclusion: frozen unified-benchmark planners produce visibly different paths on the same measured terrain, while the VTF-Flow path can be interpreted against decomposed raw terrain attributes.
- Archetypes: quantitative small-multiple grid for benchmark comparison; asymmetric mixed-modality figure for terrain mechanism.
- Backend: Python/matplotlib only.
- Final width: 182.9 mm; editable SVG/PDF; 450 dpi PNG; LZW-compressed 600 dpi TIFF.
- Data protocol: held-out RELLIS-3D sequence 00004, 1,909-scene ordering, H=10, dt=0.5 s, 5 s horizon, frozen seed-0 predictions.

## Evidence and selection

- Benchmark scenes use the predeclared cross-seed selection manifest from all 1,909 test scenes.
- Mechanism scenes are selected deterministically from H=10 GT geometry only; method outcomes are not used.
- No trajectory, map cell or candidate is removed. All K=8 candidates are retained in generative benchmark panels.
- The emphasized candidate is selected with GT minimum ADE and is explicitly labeled retrospective/oracle; it is not an online selector.
- Figures use a common per-scene viewport around the complete H=10 planning envelope; the full 96x96 terrain grid remains in each source-data CSV.

## Image and numerical integrity

- Raw terrain archives are rebuilt from synchronized Ouster point clouds and point-wise semantic labels using the verified planning-ego transform.
- White map cells denote unavailable measurements; they are not recolored as feasible terrain.
- Static terrain-potential color limits and coordinate limits are shared across all six methods within each benchmark scene.
- Potential and violation values are relative diagnostics, not calibrated safety probabilities or formal safety guarantees.

## Panel audit checklist

| Figure family | Unique claim | Axes/scales | Selection disclosed | Collision check | Status |
|---|---|---|---|---|---|
| Benchmark comparison | six methods on the identical measured field | shared | yes | inspect rendered files | pending visual QA |
| Terrain mechanism hero | Flow/VTF/GT spatial relation to static potential | fixed field extent | yes | inspect rendered files | pending visual QA |
| Terrain attributes b–g | physical/semantic factors underlying the field | per-attribute units | not applicable | inspect colorbars and labels | pending visual QA |
"""
    (output_root / "figure_QA.md").write_text(qa, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--field-config", type=Path, default=DEFAULT_FIELD_CONFIG)
    parser.add_argument("--sensor-to-ego", type=Path, default=DEFAULT_SENSOR_TRANSFORM)
    parser.add_argument("--tvk-config", type=Path, default=DEFAULT_TVK_CONFIG)
    parser.add_argument("--minimum-frame-separation", type=int, default=100)
    parser.add_argument("--rebuild-fields", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _configure_plotting()
    benchmark_root = args.benchmark_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else benchmark_root / "figures" / "raw_terrain_scenes_unified_h10"
    )
    source_root = output_root / "source_data"
    field_dir = source_root / "field_archives"
    output_root.mkdir(parents=True, exist_ok=True)
    trajectories, ground_truth, scene_ids = _load_unified_predictions(benchmark_root)
    scene_lookup = {scene_id: index for index, scene_id in enumerate(scene_ids)}
    benchmark_scenes = _load_benchmark_scenes(benchmark_root, scene_lookup)
    mechanism_scenes = _select_mechanism_scenes(
        ground_truth,
        scene_ids,
        minimum_frame_separation=args.minimum_frame_separation,
    )
    tvk_definition = _read_json(args.tvk_config.resolve())
    kinematic_config = TrajectoryKinematicConfig(**tvk_definition["kinematic"])

    records_by_scene: dict[str, dict[str, Any]] = {}
    for family, records in (
        ("benchmark_comparison", benchmark_scenes),
        ("terrain_mechanism", mechanism_scenes),
    ):
        for record in records:
            scene_id = str(record["scene_id"])
            shared = records_by_scene.setdefault(scene_id, dict(record))
            shared.setdefault("figure_families", []).append(family)
            shared.setdefault("selection_categories", {})[family] = record["category"]

    manifest_records: list[dict[str, Any]] = []
    for scene_id, record in records_by_scene.items():
        prediction_index = int(record["prediction_index"])
        field_path = _build_field_if_needed(
            args.data_root.resolve(),
            field_dir,
            record,
            args.field_config.resolve(),
            args.sensor_to_ego.resolve(),
            rebuild=args.rebuild_fields,
        )
        data, features, terrain_field, vehicle_field, obstacle_ids = _load_field_context(
            field_path, args.field_config.resolve()
        )
        candidates_for_scene = {
            method: trajectories[method][prediction_index] for method in METHOD_ORDER
        }
        gt = ground_truth[prediction_index]
        selected, selected_paths, metrics = _selected_trajectories_and_metrics(
            prediction_index,
            trajectories,
            ground_truth,
            terrain_field,
            vehicle_field,
            features,
            args.field_config.resolve(),
            obstacle_ids,
            kinematic_config,
        )
        source_files = _write_scene_source_data(
            source_root,
            record,
            data,
            candidates_for_scene,
            gt,
            selected,
            metrics,
        )
        figure_files: dict[str, str] = {}
        if "benchmark_comparison" in record["figure_families"]:
            base = _benchmark_figure(
                output_root,
                record,
                data,
                candidates_for_scene,
                gt,
                selected,
                metrics,
            )
            figure_files["benchmark_comparison"] = str(base.resolve())
        if "terrain_mechanism" in record["figure_families"]:
            base = _mechanism_figure(
                output_root,
                record,
                data,
                selected_paths,
                gt,
                metrics,
            )
            figure_files["terrain_mechanism"] = str(base.resolve())
        manifest_records.append(
            {
                **record,
                "field_archive": str(field_path.resolve()),
                "source_files": source_files,
                "figure_bases": figure_files,
                "selected_candidates": selected,
                "selected_metrics": metrics,
            }
        )

    manifest = {
        "status": "complete",
        "protocol": {
            "test_scenes": 1909,
            "H": 10,
            "dt_s": 0.5,
            "horizon_s": 5.0,
            "prediction_seed": 0,
            "generative_candidates": 8,
        },
        "method_order": [DISPLAY_NAMES[method] for method in METHOD_ORDER],
        "benchmark_scene_rule": (
            "predeclared cross-seed advantage categories from the unified benchmark"
        ),
        "mechanism_scene_rule": (
            "six deterministic categories selected from H=10 GT geometry only; "
            "trajectories stay inside the forward BEV and reach at least x=2 m"
        ),
        "candidate_display_rule": (
            "all candidates retained; GT-minimum-ADE member emphasized for retrospective "
            "best-of-K visualization only"
        ),
        "coordinate_convention": "planning ego: x forward, y left, z up",
        "terrain_background": "raw LiDAR/semantic-derived static terrain potential C_T",
        "records": manifest_records,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_text_assets(output_root)
    print(
        json.dumps(
            {
                "status": "complete",
                "output_root": str(output_root),
                "benchmark_figures": len(benchmark_scenes),
                "mechanism_figures": len(mechanism_scenes),
                "unique_scenes": len(records_by_scene),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

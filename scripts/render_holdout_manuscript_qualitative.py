"""Render held-out real-terrain figures for the VTF-Flow manuscript.

The script uses the frozen seed-0 predictions from the strict sequence-holdout
benchmark.  It never selects a candidate with ground truth: Flow Matching and
VTF-Flow always display candidate 0.  One camera-visible scene is selected from
each held-out sequence using only recorded path length and geometric projection
visibility.  The terrain-decomposition hero is then selected from those three
scenes using static-potential heterogeneity, without using method errors or
method gains.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image
import torch


TERRAFLOW_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TERRAFLOW_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from TerraFlow.scripts.build_terrain_field import build_archive  # noqa: E402
from TerraFlow.scripts.render_camera_trajectory_results import (  # noqa: E402
    DEFAULT_CAMERA_INTRINSICS,
    DEFAULT_EXTRINSIC_VARIANT,
    DEFAULT_SENSOR_TRANSFORM,
    _camera_image,
    _draw_projected,
    _load_calibration,
    _project,
)
from TerraFlow.scripts.run_unified_h10_benchmark import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_DATA,
    CombinedSceneDataset,
    H10PlanningDataset,
)
from TerraFlow.terrain.feasibility_field import (  # noqa: E402
    AnalyticTerrainField,
    TerrainFieldConfig,
)


BENCHMARK_ROOT = TERRAFLOW_ROOT / "outputs" / "sequence_holdout_full_benchmark"
ROBUSTNESS_ROOT = TERRAFLOW_ROOT / "outputs" / "sequence_holdout_robustness"
DEFAULT_OUTPUT = BENCHMARK_ROOT / "figures" / "heldout_real_data"
DEFAULT_RAW_FIELD_CONFIG = TERRAFLOW_ROOT / "configs" / "rellis3d_terrain_field.json"
SEQUENCES = ("00000", "00001", "00002")
REPLICATE_DIR = "seed_0"
FLOW_COLOR = "#6B7280"
VTF_COLOR = "#008C86"
GT_COLOR = "#161B22"
ORIGIN_COLOR = "#D13C3C"
GOAL_COLOR = "#2F5DA8"
FULL_WIDTH_IN = 183.0 / 25.4


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.2,
            "axes.titlesize": 7.8,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.4,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _save_all(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def _load_predictions(sequence: str, method: str) -> dict[str, np.ndarray]:
    path = (
        ROBUSTNESS_ROOT
        / "runs"
        / f"holdout_{sequence}"
        / REPLICATE_DIR
        / method
        / "predictions.npz"
    )
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def _load_metrics(sequence: str, method: str) -> pd.DataFrame:
    path = (
        BENCHMARK_ROOT
        / "runs"
        / f"holdout_{sequence}"
        / REPLICATE_DIR
        / method
        / "scene_level_metrics.csv"
    )
    return pd.read_csv(path)


def _trajectory_length(trajectories: np.ndarray) -> np.ndarray:
    origin = np.zeros((len(trajectories), 1, 3), dtype=trajectories.dtype)
    full = np.concatenate((origin, trajectories), axis=1)
    return np.linalg.norm(np.diff(full, axis=1), axis=-1).sum(axis=-1)


def _visibility(trajectory: np.ndarray, calibration: Mapping[str, Any]) -> float:
    _, visible, _ = _project(trajectory, calibration)
    return float(np.mean(visible))


def _choose_camera_scene(
    sequence: str,
    flow: Mapping[str, np.ndarray],
    vtf: Mapping[str, np.ndarray],
    metrics: pd.DataFrame,
    calibration: Mapping[str, Any],
    data_root: Path,
) -> dict[str, Any]:
    """Choose a camera-visible long scene without using method outcomes."""

    ground_truth = np.asarray(vtf["ground_truth"])
    if not np.array_equal(ground_truth, flow["ground_truth"]):
        raise ValueError(f"paired archives disagree on GT for held-out sequence {sequence}")
    if len(metrics) != len(ground_truth):
        raise ValueError(f"metrics and predictions disagree for held-out sequence {sequence}")
    lengths = _trajectory_length(ground_truth)
    long_threshold = float(np.quantile(lengths, 0.75))
    candidates: list[dict[str, Any]] = []
    for archive_index in np.flatnonzero(lengths >= long_threshold):
        frame_id = int(metrics.iloc[int(archive_index)]["frame_id"])
        try:
            image_path = _camera_image(data_root, sequence, frame_id)
        except FileNotFoundError:
            continue
        fractions = {
            "GT": _visibility(ground_truth[archive_index], calibration),
            "Flow": _visibility(flow["trajectories"][archive_index, 0], calibration),
            "VTF-Flow": _visibility(vtf["trajectories"][archive_index, 0], calibration),
        }
        minimum_visibility = min(fractions.values())
        candidates.append(
            {
                "sequence": sequence,
                "frame_id": frame_id,
                "archive_index": int(archive_index),
                "dataset_index": int(metrics.iloc[int(archive_index)]["dataset_index"]),
                "path_length_m": float(lengths[archive_index]),
                "minimum_visibility": minimum_visibility,
                "visibility": fractions,
                "image_path": str(image_path.resolve()),
            }
        )
    if not candidates:
        raise RuntimeError(f"no long camera-synchronised scenes for sequence {sequence}")
    accepted: list[dict[str, Any]] = []
    threshold_used = 0.0
    for threshold in (0.8, 0.7, 0.6, 0.5, 0.4):
        accepted = [row for row in candidates if row["minimum_visibility"] >= threshold]
        if accepted:
            threshold_used = threshold
            break
    if not accepted:
        raise RuntimeError(f"no sufficiently visible long scene for sequence {sequence}")
    length_values = np.asarray([row["path_length_m"] for row in accepted], dtype=np.float64)
    span = max(float(length_values.max() - length_values.min()), 1e-8)
    for row in accepted:
        relative_length = (float(row["path_length_m"]) - float(length_values.min())) / span
        row["selection_score"] = float(row["minimum_visibility"] + 0.08 * relative_length)
        row["visibility_threshold_used"] = threshold_used
    return sorted(
        accepted,
        key=lambda row: (-float(row["selection_score"]), int(row["frame_id"])),
    )[0]


def _final_terrain_config() -> TerrainFieldConfig:
    path = (
        ROBUSTNESS_ROOT
        / "checkpoints"
        / "holdout_00000"
        / REPLICATE_DIR
        / "flow_tvk"
        / "effective_config.json"
    )
    return TerrainFieldConfig(**_read_json(path)["terrain_field"])


def _planner_static_potential(
    terrain_map: torch.Tensor,
    config: TerrainFieldConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    field = AnalyticTerrainField(terrain_map.unsqueeze(0), config)
    components = field.components
    numerator = (
        config.occupancy_weight * components["occupancy"]
        + config.traversability_weight * components["nontraversable"]
        + config.slope_weight * components["slope"]
        + config.roughness_weight * components["roughness"]
        + config.clearance_weight * components["clearance"]
    )
    denominator = (
        config.occupancy_weight
        + config.traversability_weight
        + config.slope_weight
        + config.roughness_weight
        + config.clearance_weight
    )
    potential = (numerator / denominator).clamp(0.0, 1.0)[0, 0].cpu().numpy()
    component_arrays = {
        name: value[0, 0].detach().cpu().numpy() for name, value in components.items()
    }
    return potential, component_arrays


def _build_raw_archive(
    record: Mapping[str, Any],
    data_root: Path,
    output_root: Path,
    raw_field_config: Path,
    sensor_transform: Path,
    rebuild: bool,
) -> Path:
    archive = (
        output_root
        / "source_data"
        / "field_archives"
        / f"{record['sequence']}_{int(record['frame_id']):06d}_verified.npz"
    )
    if rebuild or not archive.is_file():
        archive.parent.mkdir(parents=True, exist_ok=True)
        build_archive(
            argparse.Namespace(
                data_root=data_root,
                sequence=str(record["sequence"]),
                frame=int(record["frame_id"]),
                sensor="ouster",
                config=raw_field_config,
                output=archive,
                sensor_to_ego=sensor_transform,
                allow_unverified_identity=False,
                geometry_only=False,
            )
        )
    return archive


def _load_raw_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]).copy() for name in archive.files}
    if str(data["coordinate_status"]) != "ego_from_explicit_T_ego_sensor":
        raise ValueError(f"raw field is not in the verified planning-ego frame: {path}")
    return data


def _with_origin(trajectory: np.ndarray) -> np.ndarray:
    return np.concatenate((np.zeros((1, 3), dtype=trajectory.dtype), trajectory), axis=0)


def _draw_bev_path(
    axis: plt.Axes,
    trajectory: np.ndarray,
    color: str,
    *,
    linewidth: float,
    linestyle: str = "-",
    zorder: int = 7,
    alpha: float = 1.0,
) -> None:
    path = _with_origin(trajectory)
    (line,) = axis.plot(
        path[:, 0],
        path[:, 1],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        solid_capstyle="round",
        zorder=zorder,
    )
    line.set_path_effects(
        [path_effects.Stroke(linewidth=linewidth + 0.9, foreground="white", alpha=0.92), path_effects.Normal()]
    )


def _draw_origin_goal(axis: plt.Axes, ground_truth: np.ndarray) -> None:
    axis.scatter(
        0.0,
        0.0,
        marker="*",
        s=38,
        facecolor=ORIGIN_COLOR,
        edgecolor="white",
        linewidth=0.6,
        clip_on=False,
        zorder=12,
    )
    axis.scatter(
        ground_truth[-1, 0],
        ground_truth[-1, 1],
        marker="P",
        s=28,
        facecolor=GOAL_COLOR,
        edgecolor="white",
        linewidth=0.6,
        zorder=12,
    )


def _planning_view(paths: Iterable[np.ndarray]) -> tuple[float, float, float, float]:
    points = np.concatenate(
        [np.zeros((1, 2), dtype=np.float32)]
        + [np.asarray(path)[..., :2].reshape(-1, 2) for path in paths],
        axis=0,
    )
    x_max = min(24.0, max(8.0, float(np.nanmax(points[:, 0])) + 1.3))
    y_min = max(-12.0, min(-4.0, float(np.nanmin(points[:, 1])) - 1.5))
    y_max = min(12.0, max(4.0, float(np.nanmax(points[:, 1])) + 1.5))
    return 0.0, x_max, y_min, y_max


def _mask(values: np.ndarray, invalid: np.ndarray | None) -> np.ndarray | np.ma.MaskedArray:
    return np.ma.array(values, mask=invalid) if invalid is not None else values


def _terrain_decomposition_figure(
    record: Mapping[str, Any],
    raw: Mapping[str, np.ndarray],
    potential: np.ndarray,
    flow_trajectory: np.ndarray,
    vtf_trajectory: np.ndarray,
    ground_truth: np.ndarray,
    output_root: Path,
    stem_name: str = "heldout_terrain_attribute_decomposition",
) -> Path:
    extent = (0.0, 24.0, -12.0, 12.0)
    view = _planning_view((ground_truth, flow_trajectory, vtf_trajectory))
    geometry_mask = ~np.asarray(raw["geometry_valid"], dtype=bool)
    slope_mask = ~np.asarray(raw["slope_valid"], dtype=bool)
    semantic_mask = ~np.asarray(raw["semantic_valid"], dtype=bool)
    policy = json.loads(str(raw["semantic_policy_json"]))
    semantic_values = np.asarray(raw["semantic_class"], dtype=np.int64)
    encountered = sorted(int(value) for value in np.unique(semantic_values[~semantic_mask]))
    semantic_display = np.full_like(semantic_values, np.nan, dtype=np.float32)
    for index, label_id in enumerate(encountered):
        semantic_display[semantic_values == label_id] = index
    semantic_cmap = ListedColormap(
        mpl.colormaps["tab20"](np.linspace(0.0, 1.0, max(len(encountered), 1)))
    )
    semantic_norm = BoundaryNorm(np.arange(-0.5, len(encountered) + 0.5), semantic_cmap.N)

    fig = plt.figure(figsize=(FULL_WIDTH_IN, 5.15), layout="constrained")
    layout = fig.add_gridspec(3, 4, width_ratios=(1.42, 1.42, 1.0, 1.0))
    hero = fig.add_subplot(layout[:, :2])
    supports = [fig.add_subplot(layout[row, col]) for row in range(3) for col in range(2, 4)]
    terrain_cmap = mpl.colormaps["magma"].copy()
    terrain_cmap.set_bad("white")
    hero_image = hero.imshow(
        potential.T,
        origin="lower",
        extent=extent,
        cmap=terrain_cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="bilinear",
        aspect="equal",
        rasterized=True,
    )
    _draw_bev_path(hero, flow_trajectory, FLOW_COLOR, linewidth=1.35, linestyle="--", zorder=8)
    _draw_bev_path(hero, vtf_trajectory, VTF_COLOR, linewidth=1.85, zorder=9)
    _draw_bev_path(hero, ground_truth, GT_COLOR, linewidth=1.4, zorder=10)
    _draw_origin_goal(hero, ground_truth)
    hero.set_xlim(view[0], view[1])
    hero.set_ylim(view[2], view[3])
    hero.set_aspect("equal")
    hero.set_xlabel("Ego-forward x (m)")
    hero.set_ylabel("Ego-left y (m)")
    hero.set_title("a  Planner-used static terrain potential", loc="left", fontweight="bold")
    hero.legend(
        handles=[
            Line2D([0], [0], color=FLOW_COLOR, lw=1.5, ls="--", label="Flow Matching, candidate 0"),
            Line2D([0], [0], color=VTF_COLOR, lw=1.8, label="VTF-Flow, candidate 0"),
            Line2D([0], [0], color=GT_COLOR, lw=1.5, label="Recorded GT"),
        ],
        loc="upper left",
        frameon=True,
        facecolor="white",
        framealpha=0.88,
        edgecolor="#CBD5E1",
    )
    hero_bar = fig.colorbar(hero_image, ax=hero, fraction=0.035, pad=0.02)
    hero_bar.set_label("Static potential C_T (higher = less feasible)")

    valid_elevation = np.asarray(raw["elevation_m"])[~geometry_mask]
    elevation_limits = (
        float(np.nanpercentile(valid_elevation, 2.0)),
        float(np.nanpercentile(valid_elevation, 98.0)),
    ) if valid_elevation.size else (None, None)
    valid_roughness = np.asarray(raw["roughness_m"])[~geometry_mask]
    roughness_vmax = float(np.nanpercentile(valid_roughness, 99.0)) if valid_roughness.size else None
    specs: Sequence[tuple[str, Any, Any, str, np.ndarray | None, float | None, float | None]] = (
        ("b  Elevation", raw["elevation_m"], "terrain", "m", geometry_mask, *elevation_limits),
        ("c  Local slope", raw["slope_deg"], "magma", "degrees", slope_mask, 0.0, 65.0),
        ("d  Height roughness", raw["roughness_m"], "cividis", "m (SD)", geometry_mask, 0.0, roughness_vmax),
        ("semantic", semantic_display, semantic_cmap, "", semantic_mask, None, None),
        ("f  Obstacle occupancy", raw["occupancy"], "gray_r", "occupied", None, 0.0, 1.0),
        ("g  Obstacle clearance", raw["clearance_m"], "Blues", "m", None, 0.0, 4.0),
    )
    for axis, spec in zip(supports, specs):
        title, values, cmap, unit, invalid, vmin, vmax = spec
        shown = _mask(np.asarray(values), invalid)
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
            bar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03, ticks=np.arange(len(encountered)))
            bar.ax.set_yticklabels([policy.get(str(value), {}).get("name", str(value)) for value in encountered])
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
            bar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
            bar.set_label(unit, fontsize=5.5, labelpad=2)
        bar.ax.tick_params(labelsize=5.2, width=0.5, length=2)
        _draw_bev_path(axis, vtf_trajectory, VTF_COLOR, linewidth=1.35, zorder=8)
        _draw_bev_path(axis, ground_truth, GT_COLOR, linewidth=1.0, linestyle="--", zorder=9)
        axis.scatter(0.0, 0.0, marker="*", s=23, facecolor=ORIGIN_COLOR, edgecolor="white", linewidth=0.5, zorder=10)
        axis.set_xlim(view[0], view[1])
        axis.set_ylim(view[2], view[3])
        axis.set_aspect("equal")
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
        f"Held-out RELLIS-3D terrain attributes | sequence {record['sequence']}, "
        f"frame {int(record['frame_id']):06d}",
        fontsize=8.2,
    )
    fig.text(
        0.5,
        -0.005,
        "Candidate 0 is fixed before GT comparison. White cells are unobserved; semantic labels support offline BEV construction and are not a separate planner channel.",
        ha="center",
        va="bottom",
        fontsize=5.5,
    )
    stem = output_root / stem_name
    _save_all(fig, stem)
    return stem


def _terrain_cross_sequence_figure(
    materials: Sequence[Mapping[str, Any]],
    raw_by_sequence: Mapping[str, Mapping[str, np.ndarray]],
    output_root: Path,
) -> Path:
    """Render a compact held-out sequence grid with shared component scales."""
    extent = (0.0, 24.0, -12.0, 12.0)
    all_paths: list[np.ndarray] = []
    for material in materials:
        all_paths.extend((material["gt"], material["flow"], material["vtf"]))
    view = _planning_view(all_paths)

    elevation_samples: list[np.ndarray] = []
    roughness_samples: list[np.ndarray] = []
    for material in materials:
        raw = raw_by_sequence[str(material["record"]["sequence"])]
        valid_geometry = np.asarray(raw["geometry_valid"], dtype=bool)
        elevation_samples.append(np.asarray(raw["elevation_m"])[valid_geometry])
        roughness_samples.append(np.asarray(raw["roughness_m"])[valid_geometry])
    elevation_values = np.concatenate([values for values in elevation_samples if values.size])
    roughness_values = np.concatenate([values for values in roughness_samples if values.size])
    elevation_limits = (
        float(np.nanpercentile(elevation_values, 2.0)),
        float(np.nanpercentile(elevation_values, 98.0)),
    )
    roughness_vmax = float(np.nanpercentile(roughness_values, 99.0))

    column_specs = (
        ("Static potential C_T", "potential", "magma", None, 0.0, 1.0, "C_T"),
        ("Elevation", "elevation_m", "terrain", "geometry", *elevation_limits, "m"),
        ("Local slope", "slope_deg", "magma", "slope", 0.0, 65.0, "degrees"),
        ("Roughness", "roughness_m", "cividis", "geometry", 0.0, roughness_vmax, "m (SD)"),
        ("Occupancy", "occupancy", "gray_r", None, 0.0, 1.0, "occupied"),
        ("Clearance", "clearance_m", "Blues", None, 0.0, 4.0, "m"),
    )
    fig, axes = plt.subplots(
        len(materials),
        len(column_specs),
        figsize=(FULL_WIDTH_IN, 4.55),
        sharex=True,
        sharey=True,
        layout="constrained",
        squeeze=False,
    )
    column_images: list[Any] = []
    for row, material in enumerate(materials):
        record = material["record"]
        raw = raw_by_sequence[str(record["sequence"])]
        for col, (title, field, cmap_name, mask_name, vmin, vmax, _unit) in enumerate(column_specs):
            axis = axes[row, col]
            if field == "potential":
                values = np.asarray(material["potential"])
                invalid = None
            else:
                values = np.asarray(raw[field])
                if mask_name == "geometry":
                    invalid = ~np.asarray(raw["geometry_valid"], dtype=bool)
                elif mask_name == "slope":
                    invalid = ~np.asarray(raw["slope_valid"], dtype=bool)
                else:
                    invalid = None
            cmap = mpl.colormaps[cmap_name].copy()
            cmap.set_bad("white")
            shown = _mask(values, invalid)
            image = axis.imshow(
                shown.T if field == "potential" else shown,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="bilinear" if field == "potential" else "nearest",
                aspect="equal",
                rasterized=True,
            )
            if row == 0:
                axis.set_title(f"{chr(97 + col)}  {title}", loc="left", fontweight="bold", fontsize=6.0)
                column_images.append(image)
            if field == "potential":
                _draw_bev_path(axis, material["flow"], FLOW_COLOR, linewidth=1.0, linestyle="--", zorder=7)
                _draw_bev_path(axis, material["vtf"], VTF_COLOR, linewidth=1.25, zorder=8)
                _draw_bev_path(axis, material["gt"], GT_COLOR, linewidth=1.05, zorder=9)
                _draw_origin_goal(axis, material["gt"])
            else:
                _draw_bev_path(axis, material["vtf"], VTF_COLOR, linewidth=1.0, zorder=8)
                _draw_bev_path(axis, material["gt"], GT_COLOR, linewidth=0.85, linestyle="--", zorder=9)
                axis.scatter(
                    0.0,
                    0.0,
                    marker="*",
                    s=16,
                    facecolor=ORIGIN_COLOR,
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=10,
                )
            axis.set_xlim(view[0], view[1])
            axis.set_ylim(view[2], view[3])
            axis.set_aspect("equal")
            axis.tick_params(labelsize=5.2, width=0.5, length=2)
            if col == 0:
                axis.set_ylabel(
                    f"held-out {record['sequence']}\nEgo-left y (m)",
                    fontsize=5.5,
                )
            if row == len(materials) - 1:
                axis.set_xlabel("Ego-forward x (m)", fontsize=5.5)

    for col, image in enumerate(column_images):
        bar = fig.colorbar(
            image,
            ax=axes[:, col],
            orientation="horizontal",
            fraction=0.050,
            pad=0.055,
            shrink=0.88,
        )
        bar.set_label(column_specs[col][-1], fontsize=5.2, labelpad=1)
        bar.ax.tick_params(labelsize=5.0, width=0.5, length=2, pad=1)
    fig.legend(
        handles=[
            Line2D([0], [0], color=FLOW_COLOR, lw=1.2, ls="--", label="Flow Matching, candidate 0"),
            Line2D([0], [0], color=VTF_COLOR, lw=1.4, label="VTF-Flow, candidate 0"),
            Line2D([0], [0], color=GT_COLOR, lw=1.2, label="Recorded GT"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=3,
        frameon=False,
        fontsize=5.6,
    )
    fig.text(
        0.5,
        -0.004,
        "Fixed candidate 0 is shown for both generative planners. Shared axes and component scales support cross-sequence comparison; white cells are unobserved.",
        ha="center",
        va="bottom",
        fontsize=5.3,
    )
    stem = output_root / "heldout_terrain_attribute_cross_sequence"
    _save_all(fig, stem)
    return stem


def _draw_camera_panel(
    axis: plt.Axes,
    image: np.ndarray,
    flow_trajectory: np.ndarray,
    vtf_trajectory: np.ndarray,
    ground_truth: np.ndarray,
    calibration: Mapping[str, Any],
) -> dict[str, float]:
    axis.imshow(image)
    visibility: dict[str, float] = {}
    for name, trajectory, color, style, width, zorder in (
        ("Flow", flow_trajectory, FLOW_COLOR, "--", 1.35, 6),
        ("VTF-Flow", vtf_trajectory, VTF_COLOR, "-", 1.8, 7),
        ("GT", ground_truth, GT_COLOR, "-", 1.55, 8),
    ):
        pixels, visible, _ = _project(trajectory, calibration)
        _draw_projected(
            axis,
            pixels,
            visible,
            color=color,
            linewidth=width,
            linestyle=style,
            alpha=0.98,
            zorder=zorder,
        )
        visibility[name] = float(np.mean(visible))
    axis.set_xlim(0, image.shape[1])
    axis.set_ylim(image.shape[0], 0)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    return visibility


def _camera_bev_figure(
    materials: Sequence[Mapping[str, Any]],
    output_root: Path,
    stem_name: str = "heldout_camera_bev_cross_view",
) -> Path:
    figure_height = 2.95 if len(materials) == 1 else 6.65
    fig = plt.figure(figsize=(FULL_WIDTH_IN, figure_height), layout="constrained")
    grid = fig.add_gridspec(len(materials) + 1, 2, height_ratios=(0.13,) + (1.0,) * len(materials), width_ratios=(1.55, 1.0))
    legend_axis = fig.add_subplot(grid[0, :])
    legend_axis.axis("off")
    legend_axis.legend(
        handles=[
            Line2D([0], [0], color=GT_COLOR, lw=1.6, label="Recorded GT"),
            Line2D([0], [0], color=FLOW_COLOR, lw=1.5, ls="--", label="Flow Matching, candidate 0"),
            Line2D([0], [0], color=VTF_COLOR, lw=1.8, label="VTF-Flow, candidate 0"),
            Line2D([0], [0], marker="*", color="none", markerfacecolor=ORIGIN_COLOR, markeredgecolor="white", markersize=7, label="Ego origin"),
            Line2D([0], [0], marker="P", color="none", markerfacecolor=GOAL_COLOR, markeredgecolor="white", markersize=6, label="Common 5 s goal"),
        ],
        loc="center",
        ncol=5,
        frameon=False,
    )
    bev_images = []
    for row_index, material in enumerate(materials, start=1):
        record = material["record"]
        camera_axis = fig.add_subplot(grid[row_index, 0])
        _draw_camera_panel(
            camera_axis,
            material["image"],
            material["flow"],
            material["vtf"],
            material["gt"],
            material["calibration"],
        )
        camera_axis.set_title(
            f"{chr(96 + 2 * row_index - 1)}  Camera context | held-out {record['sequence']}, frame {int(record['frame_id']):06d}",
            loc="left",
            fontweight="bold",
        )
        bev_axis = fig.add_subplot(grid[row_index, 1])
        image = bev_axis.imshow(
            material["potential"].T,
            origin="lower",
            extent=(0.0, 24.0, -12.0, 12.0),
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            interpolation="bilinear",
            aspect="equal",
            rasterized=True,
        )
        bev_images.append(image)
        _draw_bev_path(bev_axis, material["flow"], FLOW_COLOR, linewidth=1.3, linestyle="--", zorder=7)
        _draw_bev_path(bev_axis, material["vtf"], VTF_COLOR, linewidth=1.75, zorder=8)
        _draw_bev_path(bev_axis, material["gt"], GT_COLOR, linewidth=1.4, zorder=9)
        _draw_origin_goal(bev_axis, material["gt"])
        view = _planning_view((material["flow"], material["vtf"], material["gt"]))
        bev_axis.set_xlim(view[0], view[1])
        bev_axis.set_ylim(view[2], view[3])
        bev_axis.set_xlabel("Ego-forward x (m)")
        bev_axis.set_ylabel("Ego-left y (m)")
        bev_axis.set_title(
            f"{chr(96 + 2 * row_index)}  Static terrain potential $C_T(x,y)$",
            loc="left",
            fontweight="bold",
        )
    bar = fig.colorbar(bev_images[0], ax=fig.axes[1:], fraction=0.020, pad=0.035, shrink=0.82)
    bar.set_label("Static potential C_T (higher = less feasible)")
    fig.text(
        0.5,
        -0.002,
        "Camera images provide qualitative context only; they are not planner inputs. Projections are geometric and do not model occlusion.",
        ha="center",
        va="bottom",
        fontsize=5.5,
    )
    stem = output_root / stem_name
    _save_all(fig, stem)
    return stem


def _write_source_data(
    records: Sequence[Mapping[str, Any]],
    materials: Sequence[Mapping[str, Any]],
    raw_by_sequence: Mapping[str, Mapping[str, np.ndarray]],
    output_root: Path,
) -> None:
    source_root = output_root / "source_data"
    source_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        rows.append(
            {
                "sequence": record["sequence"],
                "frame_id": record["frame_id"],
                "dataset_index": record["dataset_index"],
                "archive_index": record["archive_index"],
                "recorded_path_length_m": record["path_length_m"],
                "minimum_projected_visibility": record["minimum_visibility"],
                "visibility_threshold_used": record["visibility_threshold_used"],
                "selection_score": record["selection_score"],
            }
        )
    pd.DataFrame(rows).to_csv(source_root / "selected_scenes.csv", index=False)

    projection_rows = []
    trajectory_rows = []
    for material in materials:
        record = material["record"]
        for method, trajectory in (
            ("Flow Matching", material["flow"]),
            ("VTF-Flow", material["vtf"]),
            ("Recorded GT", material["gt"]),
        ):
            pixels, visible, depth = _project(trajectory, material["calibration"])
            for step, point in enumerate(trajectory):
                common = {
                    "sequence": record["sequence"],
                    "frame_id": record["frame_id"],
                    "method": method,
                    "candidate": 0,
                    "waypoint": step + 1,
                    "time_s": 0.5 * (step + 1),
                    "x_m": float(point[0]),
                    "y_m": float(point[1]),
                    "z_m": float(point[2]),
                }
                trajectory_rows.append(common)
                projection_rows.append(
                    {
                        **common,
                        "pixel_u": float(pixels[step, 0]),
                        "pixel_v": float(pixels[step, 1]),
                        "camera_depth_m": float(depth[step]),
                        "visible_in_image": int(visible[step]),
                    }
                )
    pd.DataFrame(trajectory_rows).to_csv(source_root / "trajectory_points.csv", index=False)
    pd.DataFrame(projection_rows).to_csv(source_root / "camera_projection_points.csv", index=False)

    grid_rows = []
    names = (
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
    records_by_sequence = {str(record["sequence"]): record for record in records}
    for sequence in SEQUENCES:
        raw = raw_by_sequence[sequence]
        raw_record = records_by_sequence[sequence]
        resolution = float(raw["resolution_m"])
        x_min = float(raw["x_min_m"])
        y_min = float(raw["y_min_m"])
        height, width = raw["elevation_m"].shape
        scene_rows = []
        for row in range(height):
            for col in range(width):
                payload = {
                    "sequence": raw_record["sequence"],
                    "frame_id": raw_record["frame_id"],
                    "row": row,
                    "col": col,
                    "x_m": x_min + (col + 0.5) * resolution,
                    "y_m": y_min + (row + 0.5) * resolution,
                }
                payload.update({name: raw[name][row, col] for name in names})
                scene_rows.append(payload)
        grid_rows.extend(scene_rows)
        pd.DataFrame(scene_rows).to_csv(
            source_root / f"raw_terrain_grid_{sequence}_{int(raw_record['frame_id']):06d}.csv",
            index=False,
        )
    pd.DataFrame(grid_rows).to_csv(source_root / "raw_terrain_grid.csv", index=False)


def _write_manuscript_assets(
    records: Sequence[Mapping[str, Any]],
    terrain_record: Mapping[str, Any],
    terrain_stem: Path,
    terrain_stems: Mapping[str, Path],
    terrain_cross_sequence_stem: Path,
    camera_stem: Path,
    camera_stems: Mapping[str, Path],
    output_root: Path,
) -> None:
    manifest = {
        "status": "complete",
        "protocol": {
            "held_out_sequences": list(SEQUENCES),
            "seed": 0,
            "H": 10,
            "dt_s": 0.5,
            "horizon_s": 5.0,
            "displayed_candidate": 0,
        },
        "scene_selection": (
            "one scene per held-out sequence selected from the upper quartile of recorded path length "
            "using geometric camera visibility; no method error, GT distance, or method gain is used"
        ),
        "terrain_hero_selection": (
            "largest planner-static-potential standard deviation among the three preselected camera-visible scenes"
        ),
        "terrain_record": dict(terrain_record),
        "camera_records": [dict(record) for record in records],
        "figures": {
            "terrain_decomposition": str(terrain_stem.resolve()),
            "terrain_decompositions_by_sequence": {
                sequence: str(stem.resolve()) for sequence, stem in terrain_stems.items()
            },
            "terrain_cross_sequence": str(terrain_cross_sequence_stem.resolve()),
            "camera_bev_cross_view": str(camera_stem.resolve()),
            "camera_bev_by_sequence": {
                sequence: str(stem.resolve()) for sequence, stem in camera_stems.items()
            },
        },
        "interpretation_boundary": (
            "qualitative held-out-scene grounding; not an online safety certificate, collision test, "
            "or replacement for the complete sequence-level benchmark"
        ),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    contract = """# Held-out manuscript figures -- figure contract

## Core conclusions

- Terrain decomposition: the exact planner-used static potential is spatially grounded in measured elevation, slope, roughness, semantic evidence, occupancy, and clearance in all three held-out sequences.
- Camera--BEV cross-view: candidate-0 trajectories can be interpreted consistently in the real camera context and the aligned planning-ego potential across all three held-out sequences.

## Evidence and integrity

- Archetype: asymmetric mixed-modality figures.
- Backend: Python/matplotlib only.
- Final width: 183 mm; editable SVG/PDF; 450 dpi PNG; 600 dpi LZW TIFF.
- Candidate rule: fixed candidate 0 for both generative methods; no GT-oracle selection.
- Scene rule: recorded path length and geometric camera visibility only; no method error or gain.
- Camera images: RGB conversion only; no crop-specific contrast manipulation.
- Projection: geometric calibration only; occlusion is not modelled.
- White terrain cells: unobserved measurements, not feasible terrain.
- Potential values: relative planning diagnostics, not calibrated safety probabilities.
"""
    (output_root / "figure_contract.md").write_text(contract, encoding="utf-8")
    captions = f"""Figs. 5--7 | Held-out terrain-attribute decomposition of the planner-used static potential. Figures 5, 6, and 7 correspond to sequences 00000, 00001, and 00002 and frames {int(records[0]['frame_id']):06d}, {int(records[1]['frame_id']):06d}, and {int(records[2]['frame_id']):06d}, respectively. In each figure, a shows the exact static potential C_T with fixed candidate 0 from Flow Matching and VTF-Flow and the recorded GT; b--g show elevation, local slope, height roughness, dominant semantic class, obstacle occupancy, and obstacle clearance in the same planning-ego frame. No candidate is selected using GT. White cells are unobserved. Semantic labels support offline BEV construction and are not supplied as a separate planner channel. Potential values are relative diagnostics rather than calibrated safety probabilities.

Figs. 8--10 | Camera--BEV correspondence in the three held-out sequences. Figures 8, 9, and 10 correspond to the same sequence--frame pairs as Figs. 5--7. In each figure, a shows the synchronized RELLIS-3D camera frame and b shows the exact planner-used static terrain potential in the aligned planning-ego frame. Curves show fixed candidate 0 for Flow Matching and VTF-Flow and the recorded GT; no candidate is selected using GT. Camera images are used only for qualitative environmental context and are not planner inputs. The geometric projection does not model occlusion. Source data are provided with the figures.
"""
    (output_root / "figure_captions.txt").write_text(captions, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-field-config", type=Path, default=DEFAULT_RAW_FIELD_CONFIG)
    parser.add_argument("--sensor-transform", type=Path, default=DEFAULT_SENSOR_TRANSFORM)
    parser.add_argument("--camera-intrinsics", type=Path, default=DEFAULT_CAMERA_INTRINSICS)
    parser.add_argument("--extrinsic-variant", default=DEFAULT_EXTRINSIC_VARIANT)
    parser.add_argument("--rebuild-fields", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _configure_plotting()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    source = CombinedSceneDataset(args.cache_root.resolve(), ("train", "val", "test"))
    dataset = H10PlanningDataset(
        source,
        data_root / "processed" / "Rellis-3D",
        horizon=10,
        history_steps=6,
    )
    terrain_config = _final_terrain_config()
    selected_records: list[dict[str, Any]] = []
    materials: list[dict[str, Any]] = []
    raw_by_sequence: dict[str, dict[str, np.ndarray]] = {}

    for sequence in SEQUENCES:
        flow = _load_predictions(sequence, "FLOW")
        vtf = _load_predictions(sequence, "VTF_V2")
        metrics = _load_metrics(sequence, "VTF_V2").reset_index(drop=True)
        calibration = _load_calibration(
            data_root,
            sequence,
            args.sensor_transform.resolve(),
            args.camera_intrinsics.resolve(),
            args.extrinsic_variant,
        )
        record = _choose_camera_scene(sequence, flow, vtf, metrics, calibration, data_root)
        scene = dataset[int(record["dataset_index"])]
        gt = np.asarray(vtf["ground_truth"][int(record["archive_index"])]).copy()
        if not np.allclose(scene.gt_future.numpy(), gt, atol=1e-6):
            raise ValueError(f"dataset and archive GT disagree for {sequence}")
        potential, _ = _planner_static_potential(scene.terrain_map.float(), terrain_config)
        record["static_potential_std"] = float(np.std(potential))
        selected_records.append(record)
        image = np.asarray(Image.open(record["image_path"]).convert("RGB"))
        materials.append(
            {
                "record": record,
                "image": image,
                "flow": np.asarray(flow["trajectories"][int(record["archive_index"]), 0]).copy(),
                "vtf": np.asarray(vtf["trajectories"][int(record["archive_index"]), 0]).copy(),
                "gt": gt,
                "potential": potential,
                "calibration": calibration,
            }
        )
        field_path = _build_raw_archive(
            record,
            data_root,
            output_root,
            args.raw_field_config.resolve(),
            args.sensor_transform.resolve(),
            bool(args.rebuild_fields),
        )
        raw_by_sequence[sequence] = _load_raw_archive(field_path)

    terrain_material = sorted(
        materials,
        key=lambda item: (-float(item["record"]["static_potential_std"]), str(item["record"]["sequence"])),
    )[0]
    terrain_record = terrain_material["record"]
    terrain_raw = raw_by_sequence[str(terrain_record["sequence"])]
    terrain_stems: dict[str, Path] = {}
    for material in materials:
        record = material["record"]
        sequence = str(record["sequence"])
        terrain_stems[sequence] = _terrain_decomposition_figure(
            record,
            raw_by_sequence[sequence],
            material["potential"],
            material["flow"],
            material["vtf"],
            material["gt"],
            output_root,
            stem_name=(
                f"heldout_terrain_attribute_decomposition_{sequence}_"
                f"{int(record['frame_id']):06d}"
            ),
        )
    terrain_stem = _terrain_decomposition_figure(
        terrain_record,
        terrain_raw,
        terrain_material["potential"],
        terrain_material["flow"],
        terrain_material["vtf"],
        terrain_material["gt"],
        output_root,
    )
    terrain_cross_sequence_stem = _terrain_cross_sequence_figure(
        materials,
        raw_by_sequence,
        output_root,
    )
    camera_stem = _camera_bev_figure(materials, output_root)
    camera_stems: dict[str, Path] = {}
    for material in materials:
        record = material["record"]
        sequence = str(record["sequence"])
        camera_stems[sequence] = _camera_bev_figure(
            [material],
            output_root,
            stem_name=(
                f"heldout_camera_bev_cross_view_{sequence}_"
                f"{int(record['frame_id']):06d}"
            ),
        )
    _write_source_data(selected_records, materials, raw_by_sequence, output_root)
    _write_manuscript_assets(
        selected_records,
        terrain_record,
        terrain_stem,
        terrain_stems,
        terrain_cross_sequence_stem,
        camera_stem,
        camera_stems,
        output_root,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "terrain_figure": str(terrain_stem),
                "terrain_figures_by_sequence": {
                    sequence: str(stem) for sequence, stem in terrain_stems.items()
                },
                "terrain_cross_sequence_figure": str(terrain_cross_sequence_stem),
                "camera_bev_figure": str(camera_stem),
                "camera_bev_figures_by_sequence": {
                    sequence: str(stem) for sequence, stem in camera_stems.items()
                },
                "terrain_scene": f"{terrain_record['sequence']}:{int(terrain_record['frame_id']):06d}",
                "camera_scenes": [
                    f"{record['sequence']}:{int(record['frame_id']):06d}" for record in selected_records
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

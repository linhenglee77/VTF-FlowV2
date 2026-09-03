"""Overlay frozen VTF-Flow trajectories on raw LiDAR terrain attributes.

This figure complements cached-BEV method visualizations with a spatial audit
against the raw RELLIS-3D geometry and point-wise semantic labels.  It does not
convert a representative scene into an absolute safety claim: unobserved cells
remain explicit and feasibility is reported only as a relative score.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import torch


TERRAFLOW_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TERRAFLOW_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from TerraFlow.terrain.feasibility_field import (  # noqa: E402
    ContinuousTerrainField,
    load_terrain_field_config,
)
from TerraFlow.terrain.terrain_features import TerrainFeatures, TerrainGridSpec  # noqa: E402
from TerraFlow.terrain.trajectory_kinematics import (  # noqa: E402
    TrajectoryKinematicConfig,
    trajectory_kinematic_cost,
)
from TerraFlow.terrain.vehicle_conditioned_field import (  # noqa: E402
    VehicleConditionedTerrainField,
    load_vehicle_conditioned_config,
    trajectory_motion_state,
)


DEFAULT_OUTPUT_ROOT = TERRAFLOW_ROOT / "outputs" / "final_experiments_tvk_final"
DEFAULT_FIELD = TERRAFLOW_ROOT / "outputs" / "terrain_fields" / "00004_001875_verified.npz"
DEFAULT_FIELD_CONFIG = TERRAFLOW_ROOT / "configs" / "rellis3d_terrain_field.json"
DEFAULT_STYLE = TERRAFLOW_ROOT / "configs" / "final_figure_style.json"
DEFAULT_TVK_CONFIG = TERRAFLOW_ROOT / "configs" / "final_tvk_validation.json"
DEFAULT_CASE = "largest_terrain_violation_improvement"
FINAL_WIDTH_IN = 7.2

FLOW_COLOR = "#56B4E9"
FULL_COLOR = "#E69F00"
GT_COLOR = "#111827"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _configure(style: Mapping[str, Any]) -> None:
    font_size = float(style["font_size_pt"])
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size + 0.5,
            "legend.fontsize": max(5.5, font_size - 1.0),
            "xtick.labelsize": max(5.5, font_size - 0.5),
            "ytick.labelsize": max(5.5, font_size - 0.5),
            "axes.linewidth": 0.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _features_from_archive(data: Mapping[str, np.ndarray]) -> TerrainFeatures:
    grid = TerrainGridSpec(
        x_min_m=float(data["x_min_m"]),
        x_max_m=float(data["x_max_m"]),
        y_min_m=float(data["y_min_m"]),
        y_max_m=float(data["y_max_m"]),
        resolution_m=float(data["resolution_m"]),
    )
    return TerrainFeatures(
        grid=grid,
        elevation_m=torch.from_numpy(np.array(data["elevation_m"], copy=True)),
        slope_deg=torch.from_numpy(np.array(data["slope_deg"], copy=True)),
        roughness_m=torch.from_numpy(np.array(data["roughness_m"], copy=True)),
        semantic_class=torch.from_numpy(np.array(data["semantic_class"], copy=True)),
        occupancy=torch.from_numpy(np.array(data["occupancy"], copy=True)),
        clearance_m=torch.from_numpy(np.array(data["clearance_m"], copy=True)),
        point_count=torch.from_numpy(np.array(data["point_count"], copy=True)),
        geometry_valid=torch.from_numpy(np.array(data["geometry_valid"], copy=True)),
        slope_valid=torch.from_numpy(np.array(data["slope_valid"], copy=True)),
        semantic_valid=torch.from_numpy(np.array(data["semantic_valid"], copy=True)),
    )


def _load_predictions(output_root: Path, scene_id: str) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, int]]:
    full_path = output_root / "main_primary_seed0_VTF" / "predictions.npz"
    if not full_path.is_file():
        full_path = output_root / "main_primary_seed0_D" / "predictions.npz"
    paths = {
        "Flow": output_root / "main_primary_seed0_A" / "predictions.npz",
        "VTF-Flow": full_path,
    }
    trajectories: dict[str, np.ndarray] = {}
    selected: dict[str, int] = {}
    ground_truth: np.ndarray | None = None
    for method, path in paths.items():
        with np.load(path, allow_pickle=False) as archive:
            lookup = {str(value): index for index, value in enumerate(archive["scene_ids"])}
            if scene_id not in lookup:
                raise KeyError(f"scene {scene_id} missing from {method} predictions")
            position = lookup[scene_id]
            current_gt = archive["ground_truth"][position].copy()
            candidates = archive["trajectories"][position].copy()
        if ground_truth is None:
            ground_truth = current_gt
        elif not np.allclose(ground_truth, current_gt, atol=1e-6):
            raise ValueError("Flow and VTF-Flow archives disagree on ground truth")
        ade = np.linalg.norm(candidates - current_gt[None, ...], axis=-1).mean(axis=-1)
        selected[method] = int(np.argmin(ade))
        trajectories[method] = candidates
    assert ground_truth is not None
    return trajectories, ground_truth, selected


def _with_origin(trajectory: np.ndarray) -> np.ndarray:
    return np.concatenate((np.zeros((1, trajectory.shape[-1]), dtype=trajectory.dtype), trajectory), axis=0)


def _overlay(
    axis: plt.Axes,
    trajectory: np.ndarray,
    color: str,
    *,
    linewidth: float,
    linestyle: str = "-",
    label: str | None = None,
    zorder: int = 8,
) -> None:
    points = _with_origin(trajectory)
    (line,) = axis.plot(
        points[:, 0], points[:, 1], color=color, lw=linewidth, ls=linestyle,
        label=label, zorder=zorder, solid_capstyle="round",
    )
    line.set_path_effects(
        [path_effects.Stroke(linewidth=linewidth + 1.2, foreground="white"), path_effects.Normal()]
    )


def _origin(axis: plt.Axes) -> None:
    axis.scatter(
        [0.0], [0.0], marker="*", s=32, facecolor="#D62728", edgecolor="white",
        linewidth=0.6, clip_on=False, zorder=10,
    )


def _format_axis(axis: plt.Axes, extent: tuple[float, float, float, float], *, label: bool) -> None:
    axis.set_xlim(extent[0], extent[1])
    axis.set_ylim(extent[2], extent[3])
    axis.set_aspect("equal")
    if label:
        axis.set_xlabel("Ego forward x (m)")
        axis.set_ylabel("Ego lateral y (m)")
    axis.grid(False)


def _sample_metrics(
    name: str,
    trajectory: np.ndarray,
    terrain_field: ContinuousTerrainField,
    vehicle_field: VehicleConditionedTerrainField,
    features: TerrainFeatures,
    vehicle_config_path: Path,
    obstacle_ids: set[int],
    planning_dt_s: float,
    kinematic_config: TrajectoryKinematicConfig,
) -> dict[str, Any]:
    points = torch.from_numpy(trajectory[:, :2]).float().unsqueeze(0)
    states = trajectory_motion_state(
        torch.from_numpy(trajectory).float().unsqueeze(0),
        planning_dt_s=planning_dt_s,
        config=load_vehicle_conditioned_config(vehicle_config_path),
    )
    with torch.no_grad():
        terrain_feasibility = terrain_field.query(points)[0]
        vehicle_feasibility = vehicle_field.query(points, states)[0]
        raw = terrain_field.raw_component_costs(points)
        observed_map = terrain_field._map(features.geometry_valid.float())
        observed = terrain_field._sample(observed_map, points, padding_mode="border")[0] >= 0.5
        clearance_map = terrain_field._map(features.clearance_m)
        clearance = terrain_field._sample(clearance_map, points, padding_mode="border")[0]
        kinematic = trajectory_kinematic_cost(
            torch.from_numpy(trajectory).float().unsqueeze(0),
            planning_dt_s,
            kinematic_config,
        )
        mean_vehicle_cost = float(vehicle_field.cost(points, states)[0].mean())
        mean_kinematic_cost = float(kinematic["trajectory_kinematic_cost"][0])

    grid = features.grid
    x_index = np.floor((trajectory[:, 0] - grid.x_min_m) / grid.resolution_m).astype(int)
    y_index = np.floor((trajectory[:, 1] - grid.y_min_m) / grid.resolution_m).astype(int)
    x_index = np.clip(x_index, 0, grid.width - 1)
    y_index = np.clip(y_index, 0, grid.height - 1)
    semantic_ids = features.semantic_class[y_index, x_index].cpu().numpy()
    semantic_valid = features.semantic_valid[y_index, x_index].cpu().numpy().astype(bool)
    semantic_obstacle = semantic_valid & np.isin(semantic_ids, sorted(obstacle_ids))
    return {
        "method": name,
        "mean_terrain_feasibility": float(terrain_feasibility.mean()),
        "minimum_terrain_feasibility": float(terrain_feasibility.min()),
        "mean_vehicle_feasibility": float(vehicle_feasibility.mean()),
        "mean_vehicle_conditioned_cost": mean_vehicle_cost,
        "mean_kinematic_cost": mean_kinematic_cost,
        "mean_unified_tvk_cost": mean_vehicle_cost + mean_kinematic_cost,
        "curvature_violation_rate": float(
            kinematic["curvature_violation"][0].float().mean()
        ),
        "lateral_acceleration_violation_rate": float(
            kinematic["lateral_acceleration_violation"][0].float().mean()
        ),
        "observed_waypoint_rate": float(observed.float().mean()),
        "occupancy_violation_rate": float((raw["occupancy"][0] >= 0.5).float().mean()),
        "slope_violation_rate": float((raw["slope"][0] >= 1.0).float().mean()),
        "semantic_obstacle_rate": float(np.mean(semantic_obstacle)),
        "minimum_clearance_m": float(clearance.min()),
    }


def _write_source_data(
    output_root: Path,
    scene_id: str,
    data: Mapping[str, np.ndarray],
    trajectories: Mapping[str, np.ndarray],
    ground_truth: np.ndarray,
    selected: Mapping[str, int],
    metrics: list[dict[str, Any]],
) -> tuple[Path, Path]:
    source_dir = output_root / "figure_source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    sequence, frame_text, _ = scene_id.split(":")
    stem = f"raw_terrain_trajectory_{sequence}_{int(frame_text):06d}"
    map_path = source_dir / f"{stem}.csv"
    map_names = (
        "feasibility", "elevation_m", "slope_deg", "roughness_m",
        "semantic_class", "occupancy", "clearance_m", "geometry_valid",
        "slope_valid", "semantic_valid",
    )
    resolution = float(data["resolution_m"])
    x_min = float(data["x_min_m"])
    y_min = float(data["y_min_m"])
    with map_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "record_type", "scene_id", "map_name", "row", "col", "x_m", "y_m",
            "value", "method", "candidate", "waypoint", "z_m", "selected",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for map_name in map_names:
            values = data[map_name]
            for row in range(values.shape[0]):
                for col in range(values.shape[1]):
                    writer.writerow(
                        {
                            "record_type": "map", "scene_id": scene_id,
                            "map_name": map_name, "row": row, "col": col,
                            "x_m": x_min + (col + 0.5) * resolution,
                            "y_m": y_min + (row + 0.5) * resolution,
                            "value": values[row, col], "method": "", "candidate": "",
                            "waypoint": "", "z_m": "", "selected": "",
                        }
                    )
        all_trajectories = dict(trajectories)
        all_trajectories["GT"] = ground_truth[None, ...]
        for method, candidates in all_trajectories.items():
            for candidate, trajectory in enumerate(candidates):
                for waypoint, point in enumerate(trajectory):
                    writer.writerow(
                        {
                            "record_type": "trajectory", "scene_id": scene_id,
                            "map_name": "", "row": "", "col": "", "x_m": point[0],
                            "y_m": point[1], "value": "", "method": method,
                            "candidate": candidate, "waypoint": waypoint, "z_m": point[2],
                            "selected": int(
                                method == "GT" or candidate == selected.get(method, -1)
                            ),
                        }
                    )
    metrics_path = source_dir / f"{stem}_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    return map_path, metrics_path


def _add_component_overlay(
    axis: plt.Axes,
    full: np.ndarray,
    ground_truth: np.ndarray,
) -> None:
    _overlay(axis, ground_truth, GT_COLOR, linewidth=1.0, linestyle="--")
    _overlay(axis, full, FULL_COLOR, linewidth=1.45)
    _origin(axis)


def render(
    field_path: Path,
    output_root: Path,
    config_path: Path,
    case_name: str,
    style: Mapping[str, Any],
    scene_id_override: str | None = None,
    tvk_config_path: Path = DEFAULT_TVK_CONFIG,
) -> tuple[Path, tuple[Path, Path], list[dict[str, Any]]]:
    if not np.isclose(float(style["figure_width_in"]), FINAL_WIDTH_IN):
        raise ValueError(f"publication figure width must be {FINAL_WIDTH_IN} inches")
    if scene_id_override is None:
        selection = _read_json(output_root / "qualitative_case_selection.json")
        if case_name not in selection:
            raise KeyError(f"unknown qualitative case: {case_name}")
        scene_id = str(selection[case_name]["scene_id"])
    else:
        scene_id = str(scene_id_override)
    with np.load(field_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    expected_frame = int(scene_id.split(":")[1])
    if str(data["sequence"]) != scene_id.split(":")[0] or int(data["frame"]) != expected_frame:
        raise ValueError("raw terrain archive and selected prediction scene do not match")
    if str(data["coordinate_status"]) != "ego_from_explicit_T_ego_sensor":
        raise ValueError("raw terrain archive does not use an explicit verified transform")

    trajectories, ground_truth, selected = _load_predictions(output_root, scene_id)
    definition = load_terrain_field_config(config_path)
    features = _features_from_archive(data)
    terrain_field = ContinuousTerrainField(features, definition.cost)
    archive_feasibility = np.asarray(data["feasibility"], dtype=np.float32)
    rebuilt_feasibility = terrain_field.feasibility_map[0, 0].detach().cpu().numpy()
    if not np.allclose(archive_feasibility, rebuilt_feasibility, atol=1e-6):
        raise ValueError("rebuilt field does not reproduce serialized feasibility")
    vehicle_config = load_vehicle_conditioned_config(config_path)
    vehicle_field = VehicleConditionedTerrainField(terrain_field, vehicle_config)
    policy = json.loads(str(data["semantic_policy_json"]))
    obstacle_ids = {int(label) for label, entry in policy.items() if entry["role"] == "obstacle"}
    planning_dt_s = 0.5
    tvk_definition = _read_json(tvk_config_path)
    kinematic_config = TrajectoryKinematicConfig(**tvk_definition["kinematic"])

    selected_paths = {
        method: candidates[selected[method]] for method, candidates in trajectories.items()
    }
    metrics = [
        _sample_metrics(
            method, trajectory, terrain_field, vehicle_field, features,
            config_path, obstacle_ids, planning_dt_s,
            kinematic_config,
        )
        for method, trajectory in (
            ("Flow", selected_paths["Flow"]),
            ("VTF-Flow", selected_paths["VTF-Flow"]),
            ("GT", ground_truth),
        )
    ]
    source_paths = _write_source_data(
        output_root, scene_id, data, trajectories, ground_truth, selected, metrics
    )

    extent = features.grid.extent
    geometry_mask = ~np.asarray(data["geometry_valid"], dtype=bool)
    slope_mask = ~np.asarray(data["slope_valid"], dtype=bool)
    semantic_mask = ~np.asarray(data["semantic_valid"], dtype=bool)
    semantic_values = np.asarray(data["semantic_class"], dtype=np.int64)
    encountered = sorted(int(value) for value in np.unique(semantic_values[~semantic_mask]))
    semantic_display = np.full_like(semantic_values, np.nan, dtype=np.float32)
    for index, label_id in enumerate(encountered):
        semantic_display[semantic_values == label_id] = index
    semantic_cmap = ListedColormap(
        plt.get_cmap("tab10")(np.linspace(0.0, 1.0, max(len(encountered), 1)))
    )
    semantic_norm = BoundaryNorm(
        np.arange(-0.5, len(encountered) + 0.5), semantic_cmap.N
    )

    fig = plt.figure(figsize=(7.2, 5.2), constrained_layout=True)
    layout = fig.add_gridspec(
        3, 4, width_ratios=(1.42, 1.42, 1.0, 1.0),
    )
    hero = fig.add_subplot(layout[:, :2])
    supports = [fig.add_subplot(layout[row, col]) for row in range(3) for col in range(2, 4)]

    hero_image = hero.imshow(
        data["feasibility"], origin="lower", extent=extent, cmap="viridis",
        vmin=0.0, vmax=1.0, interpolation="nearest", aspect="equal", rasterized=True,
    )
    _overlay(
        hero, selected_paths["Flow"], FLOW_COLOR, linewidth=1.25,
        linestyle="--", label="Flow (minADE)",
    )
    _overlay(
        hero, selected_paths["VTF-Flow"], FULL_COLOR,
        linewidth=1.75, label="VTF-Flow (minADE)",
    )
    _overlay(hero, ground_truth, GT_COLOR, linewidth=1.25, label="GT")
    _origin(hero)
    hero.set_title("a  Continuous terrain feasibility", loc="left", fontweight="bold")
    _format_axis(hero, extent, label=True)
    hero.legend(loc="upper left")
    metric_lookup = {str(row["method"]): row for row in metrics}
    hero.text(
        0.98,
        0.02,
        "Unified TVK cost\n"
        f"Flow {metric_lookup['Flow']['mean_unified_tvk_cost']:.3f}  |  "
        f"VTF-Flow {metric_lookup['VTF-Flow']['mean_unified_tvk_cost']:.3f}\n"
        f"GT {metric_lookup['GT']['mean_unified_tvk_cost']:.3f}",
        transform=hero.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.5,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "alpha": 0.88,
              "edgecolor": "#CBD5E1", "linewidth": 0.5},
        zorder=12,
    )
    hero_bar = fig.colorbar(hero_image, ax=hero, fraction=0.035, pad=0.02)
    hero_bar.set_label("Relative F(x, y)")

    component_specs = (
        ("b  Elevation", data["elevation_m"], "terrain", "m", geometry_mask, None, None),
        ("c  Local slope", data["slope_deg"], "magma", "degrees", slope_mask, 0.0, 65.0),
        (
            "d  Height roughness", data["roughness_m"], "cividis", "m (SD)",
            geometry_mask, 0.0, None,
        ),
        ("semantic", semantic_display, semantic_cmap, "", semantic_mask, None, None),
        ("f  Obstacle occupancy", data["occupancy"], "gray_r", "occupied", None, 0.0, 1.0),
        ("g  Obstacle clearance", data["clearance_m"], "Blues", "m", None, 0.0, None),
    )
    for axis, spec in zip(supports, component_specs):
        title, values, cmap, label, mask, vmin, vmax = spec
        if title == "semantic":
            image = axis.imshow(
                np.ma.array(values, mask=mask), origin="lower", extent=extent,
                cmap=semantic_cmap, norm=semantic_norm, interpolation="nearest",
                aspect="equal", rasterized=True,
            )
            axis.set_title("e  Dominant semantic class", loc="left", fontweight="bold")
            colorbar = fig.colorbar(
                image, ax=axis, fraction=0.046, pad=0.03,
                ticks=np.arange(len(encountered)),
            )
            colorbar.ax.set_yticklabels(
                [policy.get(str(value), {}).get("name", str(value)) for value in encountered]
            )
        else:
            shown = np.ma.array(values, mask=mask) if mask is not None else values
            image = axis.imshow(
                shown, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
                interpolation="nearest", aspect="equal", rasterized=True,
            )
            axis.set_title(title, loc="left", fontweight="bold")
            colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
            colorbar.set_label(label, fontsize=5.5, labelpad=2)
        colorbar.ax.tick_params(labelsize=5.5, width=0.5, length=2)
        _add_component_overlay(
            axis, selected_paths["VTF-Flow"], ground_truth
        )
        _format_axis(axis, extent, label=False)
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

    sequence, frame_text, _ = scene_id.split(":")
    fig.suptitle(
        "Raw RELLIS-3D terrain attributes with unified-TVK trajectories | "
        f"sequence {sequence}, frame {int(frame_text):06d} | verified planning ego transform",
        fontsize=float(style["font_size_pt"]) + 0.5,
    )
    fig.text(
        0.5, -0.008,
        "Support panels overlay VTF-Flow (orange) and GT (black). White cells are unobserved. "
        "F is relative, not a calibrated safety probability or threshold.",
        ha="center", va="bottom", fontsize=5.5,
    )
    base = (
        output_root / "figures" / "raw_terrain_scenes"
        / f"figure_raw_terrain_{sequence}_{int(frame_text):06d}"
    )
    _save(fig, base)
    return base, source_paths, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", type=Path, default=DEFAULT_FIELD)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--field-config", type=Path, default=DEFAULT_FIELD_CONFIG)
    parser.add_argument("--style", type=Path, default=DEFAULT_STYLE)
    parser.add_argument("--tvk-config", type=Path, default=DEFAULT_TVK_CONFIG)
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument(
        "--scene-id",
        help="Optional exact scene ID (for example 00004:1172:train), overriding --case.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    style = _read_json(args.style)
    _configure(style)
    figure, sources, metrics = render(
        args.field.resolve(), args.output_root.resolve(), args.field_config.resolve(),
        args.case, style, args.scene_id, args.tvk_config.resolve(),
    )
    if args.scene_id is None:
        alias = args.output_root.resolve() / "figures" / "figure_I_raw_terrain_trajectory"
        for suffix in (".svg", ".pdf", ".tiff", ".png"):
            shutil.copy2(figure.with_suffix(suffix), alias.with_suffix(suffix))
    print(
        json.dumps(
            {
                "status": "complete",
                "figure": str(figure),
                "source_data": [str(path) for path in sources],
                "metrics": metrics,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

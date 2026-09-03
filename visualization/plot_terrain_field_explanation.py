"""Draw publication-ready terrain-field explanation figures.

The figures are reconstructed from the frozen final-experiment predictions and
the exact three-channel BEV consumed by the final planner.  A vehicle-conditioned
map is necessarily a state slice; its speed and heading are deterministically
derived from the representative scene's ground-truth trajectory and recorded in
the source-data metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch


TERRAFLOW_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TERRAFLOW_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from TerraFlow.scripts.train_regression import CombinedSceneDataset  # noqa: E402
from TerraFlow.terrain.feasibility_field import (  # noqa: E402
    AnalyticTerrainField,
    TerrainFieldConfig,
)
from TerraFlow.terrain.vehicle_conditioned_field import (  # noqa: E402
    BatchedVehicleConditionedTerrainField,
    VehicleConditionedFieldConfig,
    trajectory_motion_state,
)


DEFAULT_OUTPUT_ROOT = TERRAFLOW_ROOT / "outputs" / "final_experiments"
DEFAULT_CACHE_ROOT = WORKSPACE_ROOT / "data" / "RELLIS3D" / "trajectory_cache_h150_s5"
DEFAULT_STYLE = TERRAFLOW_ROOT / "configs" / "final_figure_style.json"
DEFAULT_FIELD_CONFIG = TERRAFLOW_ROOT / "configs" / "rellis3d_flow_feasibility.json"
DEFAULT_CASE = "best_joint_improvement"
FINAL_WIDTH_IN = 7.2

FLOW_COLOR = "#56B4E9"
FULL_COLOR = "#E69F00"
GT_COLOR = "#D62728"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _configure_matplotlib(style: Mapping[str, Any]) -> None:
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
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
        }
    )


def _save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _cell_centres(config: TerrainFieldConfig, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    x = (np.arange(height, dtype=np.float32) + 0.5) * config.forward_m / height
    y = -config.lateral_m + (np.arange(width, dtype=np.float32) + 0.5) * (
        2.0 * config.lateral_m / width
    )
    return np.meshgrid(x, y, indexing="ij")


def _representative_state(
    ground_truth: torch.Tensor,
    planning_dt_s: float,
    config: VehicleConditionedFieldConfig,
) -> tuple[float, float]:
    state = trajectory_motion_state(
        ground_truth.unsqueeze(0), planning_dt_s=planning_dt_s, config=config
    )
    speed = state["speed"][0]
    heading = state["heading"][0]
    reliability = state["heading_reliability"][0]
    reliable = reliability >= 0.5
    if bool(reliable.any()):
        selected_heading = heading[reliable]
        mean_heading = torch.atan2(
            torch.sin(selected_heading).mean(), torch.cos(selected_heading).mean()
        )
    else:
        mean_heading = torch.zeros((), dtype=ground_truth.dtype)
    positive_speed = speed[speed > 1e-6]
    median_speed = positive_speed.median() if positive_speed.numel() else speed.new_zeros(())
    return float(median_speed), float(mean_heading)


def _selected_candidate(trajectories: np.ndarray, ground_truth: np.ndarray) -> int:
    ade = np.linalg.norm(trajectories - ground_truth[None, ...], axis=-1).mean(axis=-1)
    return int(np.argmin(ade))


def _with_origin(trajectory: np.ndarray) -> np.ndarray:
    origin = np.zeros((1, trajectory.shape[-1]), dtype=trajectory.dtype)
    return np.concatenate((origin, trajectory), axis=0)


def _draw_trajectory(
    axis: plt.Axes,
    trajectory: np.ndarray,
    *,
    color: str,
    linewidth: float,
    alpha: float = 1.0,
    zorder: int = 4,
    outline: bool = False,
) -> Line2D:
    points = _with_origin(trajectory)
    (line,) = axis.plot(
        points[:, 1], points[:, 0], color=color, lw=linewidth,
        alpha=alpha, solid_capstyle="round", zorder=zorder,
    )
    if outline:
        line.set_path_effects(
            [path_effects.Stroke(linewidth=linewidth + 1.25, foreground="white"), path_effects.Normal()]
        )
    return line


def _draw_origin(axis: plt.Axes) -> None:
    axis.scatter(
        [0.0], [0.0], marker="*", s=34, facecolor="white", edgecolor="black",
        linewidth=0.7, clip_on=False, zorder=8,
    )


def _format_map_axis(axis: plt.Axes, config: TerrainFieldConfig, *, labels: bool = True) -> None:
    axis.set_xlim(-config.lateral_m, config.lateral_m)
    axis.set_ylim(0.0, config.forward_m)
    axis.set_aspect("equal")
    if labels:
        axis.set_xlabel("Ego lateral y (m)")
        axis.set_ylabel("Ego forward x (m)")
    axis.grid(False)


def _imshow(
    axis: plt.Axes,
    values: np.ndarray,
    config: TerrainFieldConfig,
    *,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> mpl.image.AxesImage:
    return axis.imshow(
        values,
        extent=(-config.lateral_m, config.lateral_m, 0.0, config.forward_m),
        origin="lower",
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        rasterized=True,
    )


def _load_case(
    output_root: Path,
    cache_root: Path,
    case_name: str,
    field_config_path: Path,
) -> dict[str, Any]:
    selection = _read_json(output_root / "qualitative_case_selection.json")
    if case_name not in selection or case_name == "selection_rules":
        raise KeyError(f"unknown deterministic case: {case_name}")
    record = selection[case_name]
    scene_id = str(record["scene_id"])
    dataset_index = int(record["dataset_index"])

    predictions: dict[str, np.lib.npyio.NpzFile] = {
        "Flow": np.load(output_root / "main_primary_seed0_A" / "predictions.npz"),
        "Full": np.load(output_root / "main_primary_seed0_D" / "predictions.npz"),
    }
    positions: dict[str, int] = {}
    for method, archive in predictions.items():
        lookup = {str(value): index for index, value in enumerate(archive["scene_ids"])}
        if scene_id not in lookup:
            raise ValueError(f"scene {scene_id} missing from {method} predictions")
        positions[method] = lookup[scene_id]

    source = CombinedSceneDataset(cache_root, ("train", "val", "test"))
    scene = source[dataset_index].as_batch()
    raw_config = _read_json(field_config_path)
    terrain_config = TerrainFieldConfig(**raw_config["terrain_field"])
    vehicle_config = VehicleConditionedFieldConfig(**raw_config["vehicle_conditioning"])
    planning_dt_s = float(raw_config["regularization"]["planning_dt_s"])

    terrain_field = AnalyticTerrainField(scene.terrain_map, terrain_config)
    vehicle_field = BatchedVehicleConditionedTerrainField(terrain_field, vehicle_config)
    map_height, map_width = scene.terrain_map.shape[-2:]
    xx, yy = _cell_centres(terrain_config, map_height, map_width)
    query = torch.from_numpy(np.stack((xx, yy), axis=-1).reshape(1, -1, 2)).float()
    ground_truth = predictions["Flow"]["ground_truth"][positions["Flow"]]
    speed_mps, heading_rad = _representative_state(
        torch.from_numpy(ground_truth).float(), planning_dt_s, vehicle_config
    )
    state = {
        "speed": torch.full(query.shape[:-1], speed_mps),
        "heading": torch.full(query.shape[:-1], heading_rad),
        "heading_reliability": torch.ones(query.shape[:-1]),
    }
    with torch.no_grad():
        terrain_feasibility = terrain_field.query(query).reshape(map_height, map_width)
        vehicle_feasibility = vehicle_field.query(query, state).reshape(map_height, map_width)
        base_components = {
            name: value[0, 0].detach().cpu().numpy()
            for name, value in terrain_field.components.items()
        }
        queried_vehicle = vehicle_field.component_costs(query, state)
        vehicle_addition = queried_vehicle["vehicle_additional_cost"].reshape(
            map_height, map_width
        )
        terrain_cost = terrain_field.cost(query).reshape(map_height, map_width)

    denominator = (
        terrain_config.occupancy_weight
        + terrain_config.traversability_weight
        + terrain_config.slope_weight
        + terrain_config.roughness_weight
        + terrain_config.clearance_weight
    )
    contribution_weights = {
        "occupancy_contribution": terrain_config.occupancy_weight / denominator,
        "nontraversable_contribution": terrain_config.traversability_weight / denominator,
        "slope_contribution": terrain_config.slope_weight / denominator,
        "roughness_contribution": terrain_config.roughness_weight / denominator,
        "clearance_contribution": terrain_config.clearance_weight / denominator,
    }
    maps = {
        "encoded_mean_height_m": (
            scene.terrain_map[0, 2].detach().cpu().numpy() * terrain_config.height_range_m
        ),
        "occupancy_contribution": (
            base_components["occupancy"] * contribution_weights["occupancy_contribution"]
        ),
        "nontraversable_contribution": (
            base_components["nontraversable"]
            * contribution_weights["nontraversable_contribution"]
        ),
        "slope_contribution": (
            base_components["slope"] * contribution_weights["slope_contribution"]
        ),
        "roughness_contribution": (
            base_components["roughness"] * contribution_weights["roughness_contribution"]
        ),
        "clearance_contribution": (
            base_components["clearance"] * contribution_weights["clearance_contribution"]
        ),
        "terrain_cost": terrain_cost.detach().cpu().numpy(),
        "vehicle_additional_cost": vehicle_addition.detach().cpu().numpy(),
        "terrain_feasibility": terrain_feasibility.detach().cpu().numpy(),
        "vehicle_feasibility": vehicle_feasibility.detach().cpu().numpy(),
    }
    method_trajectories = {
        method: archive["trajectories"][positions[method]].copy()
        for method, archive in predictions.items()
    }
    selected = {
        method: _selected_candidate(trajectories, ground_truth)
        for method, trajectories in method_trajectories.items()
    }
    for archive in predictions.values():
        archive.close()
    return {
        "record": record,
        "scene_id": scene_id,
        "dataset_index": dataset_index,
        "terrain_config": terrain_config,
        "vehicle_config": vehicle_config,
        "planning_dt_s": planning_dt_s,
        "speed_mps": speed_mps,
        "heading_rad": heading_rad,
        "xx": xx,
        "yy": yy,
        "maps": maps,
        "ground_truth": ground_truth,
        "trajectories": method_trajectories,
        "selected": selected,
    }


def _write_source_data(case: Mapping[str, Any], source_dir: Path, case_name: str) -> tuple[Path, Path]:
    source_dir.mkdir(parents=True, exist_ok=True)
    csv_path = source_dir / f"terrain_field_{case_name}.csv"
    fieldnames = [
        "record_type", "scene_id", "dataset_index", "map_name", "grid_row", "grid_col",
        "x_m", "y_m", "value", "method", "candidate", "waypoint", "z_m", "selected",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        xx = np.asarray(case["xx"])
        yy = np.asarray(case["yy"])
        for map_name, values in case["maps"].items():
            for row in range(values.shape[0]):
                for col in range(values.shape[1]):
                    writer.writerow(
                        {
                            "record_type": "map", "scene_id": case["scene_id"],
                            "dataset_index": case["dataset_index"], "map_name": map_name,
                            "grid_row": row, "grid_col": col, "x_m": float(xx[row, col]),
                            "y_m": float(yy[row, col]), "value": float(values[row, col]),
                            "method": "", "candidate": "", "waypoint": "", "z_m": "",
                            "selected": "",
                        }
                    )
        for method, trajectories in case["trajectories"].items():
            for candidate, trajectory in enumerate(trajectories):
                for waypoint, point in enumerate(trajectory):
                    writer.writerow(
                        {
                            "record_type": "trajectory", "scene_id": case["scene_id"],
                            "dataset_index": case["dataset_index"], "map_name": "",
                            "grid_row": "", "grid_col": "", "x_m": float(point[0]),
                            "y_m": float(point[1]), "value": "", "method": method,
                            "candidate": candidate, "waypoint": waypoint,
                            "z_m": float(point[2]),
                            "selected": int(candidate == case["selected"][method]),
                        }
                    )
        for waypoint, point in enumerate(case["ground_truth"]):
            writer.writerow(
                {
                    "record_type": "trajectory", "scene_id": case["scene_id"],
                    "dataset_index": case["dataset_index"], "map_name": "",
                    "grid_row": "", "grid_col": "", "x_m": float(point[0]),
                    "y_m": float(point[1]), "value": "", "method": "GT",
                    "candidate": -1, "waypoint": waypoint, "z_m": float(point[2]),
                    "selected": 1,
                }
            )
    metadata_path = source_dir / f"terrain_field_{case_name}_metadata.json"
    metadata = {
        "scene_id": case["scene_id"],
        "dataset_index": case["dataset_index"],
        "selection_rule": "deterministic best_joint_improvement from frozen paired analysis",
        "coordinate_convention": "planning ego: x forward, y left, z up; current ego at origin",
        "vehicle_field_slice": {
            "speed_mps": case["speed_mps"],
            "heading_rad": case["heading_rad"],
            "heading_reliability": 1.0,
            "derivation": "median positive GT waypoint speed and circular-mean reliable GT heading",
        },
        "planner_field_inputs": [
            "traversable_fraction", "obstacle_density", "encoded_mean_height"
        ],
        "semantic_note": (
            "The frozen final-planner cache has no semantic-ID channel. Semantic class labels "
            "are therefore not plotted or fabricated."
        ),
        "clearance_note": (
            "The cached planner field uses occupancy-proximity cost, not metric Euclidean clearance."
        ),
        "field_note": (
            "F is a relative continuous feasibility score, not a calibrated safety probability or threshold."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return csv_path, metadata_path


def _draw_selected_pair(axis: plt.Axes, case: Mapping[str, Any]) -> None:
    for method, color in (("Flow", FLOW_COLOR), ("Full", FULL_COLOR)):
        selected = int(case["selected"][method])
        _draw_trajectory(
            axis, case["trajectories"][method][selected], color=color,
            linewidth=1.7, outline=True, zorder=6,
        )
    _draw_trajectory(
        axis, case["ground_truth"], color=GT_COLOR, linewidth=1.7,
        outline=True, zorder=7,
    )
    _draw_origin(axis)


def plot_main_figure(case: Mapping[str, Any], figures_dir: Path, style: Mapping[str, Any]) -> Path:
    config = case["terrain_config"]
    if not np.isclose(float(style["figure_width_in"]), FINAL_WIDTH_IN):
        raise ValueError(f"publication figure width must be {FINAL_WIDTH_IN} inches")
    fig, axes = plt.subplots(
        1, 3, figsize=(7.2, 2.75),
        sharex=True, sharey=True, constrained_layout=True,
    )
    maps = case["maps"]
    titles = (
        "a  Terrain-only feasibility",
        "b  Vehicle-conditioned slice",
        "c  Candidate trajectory response",
    )
    backgrounds = (
        maps["terrain_feasibility"], maps["vehicle_feasibility"], maps["vehicle_feasibility"]
    )
    images = []
    for axis, title, background in zip(axes, titles, backgrounds):
        images.append(_imshow(axis, background, config, cmap="viridis", vmin=0.0, vmax=1.0))
        axis.set_title(title, loc="left", fontweight="bold")
        _format_map_axis(axis, config, labels=False)
        _draw_origin(axis)
    axes[0].set_ylabel("Ego forward x (m)")
    for axis in axes:
        axis.set_xlabel("Ego lateral y (m)")

    flow_selected = int(case["selected"]["Flow"])
    full_selected = int(case["selected"]["Full"])
    _draw_trajectory(
        axes[0], case["trajectories"]["Flow"][flow_selected], color=FLOW_COLOR,
        linewidth=1.7, outline=True,
    )
    _draw_trajectory(
        axes[1], case["trajectories"]["Full"][full_selected], color=FULL_COLOR,
        linewidth=1.7, outline=True,
    )
    for method, color in (("Flow", FLOW_COLOR), ("Full", FULL_COLOR)):
        for candidate, trajectory in enumerate(case["trajectories"][method]):
            _draw_trajectory(
                axes[2], trajectory, color=color,
                linewidth=1.15 if candidate == case["selected"][method] else 0.55,
                alpha=0.95 if candidate == case["selected"][method] else 0.35,
                zorder=5 if candidate == case["selected"][method] else 3,
                outline=candidate == case["selected"][method],
            )
    for axis in axes:
        _draw_trajectory(
            axis, case["ground_truth"], color=GT_COLOR, linewidth=1.55,
            outline=True, zorder=7,
        )
        _draw_origin(axis)
    legend = [
        Line2D([0], [0], color=FLOW_COLOR, lw=1.7, label="Flow"),
        Line2D([0], [0], color=FULL_COLOR, lw=1.7, label="VTF-Flow"),
        Line2D([0], [0], color=GT_COLOR, lw=1.7, label="GT"),
    ]
    axes[2].legend(handles=legend, loc="upper right", handlelength=2.0)
    colorbar = fig.colorbar(images[-1], ax=axes, fraction=0.028, pad=0.02)
    colorbar.set_label("Relative feasibility F(x, y)")
    heading_deg = np.degrees(float(case["heading_rad"]))
    fig.suptitle(
        f"RELLIS-3D {case['scene_id']} | vehicle-field slice: "
        f"v={case['speed_mps']:.2f} m/s, ψ={heading_deg:.1f}°",
        fontsize=float(style["font_size_pt"]) + 0.5,
    )
    base = figures_dir / "figure_H_terrain_conditioning"
    _save_figure(fig, base)
    return base


def _small_colorbar(fig: plt.Figure, image: mpl.image.AxesImage, axis: plt.Axes, label: str) -> None:
    colorbar = fig.colorbar(image, ax=axis, fraction=0.047, pad=0.025)
    colorbar.ax.tick_params(labelsize=5.5, width=0.5, length=2)
    colorbar.set_label(label, fontsize=5.5, labelpad=2)


def plot_decomposition_figure(
    case: Mapping[str, Any], figures_dir: Path, style: Mapping[str, Any]
) -> Path:
    config = case["terrain_config"]
    if not np.isclose(float(style["figure_width_in"]), FINAL_WIDTH_IN):
        raise ValueError(f"publication figure width must be {FINAL_WIDTH_IN} inches")
    maps = case["maps"]
    fig = plt.figure(figsize=(7.2, 5.05), constrained_layout=True)
    grid = fig.add_gridspec(3, 5, width_ratios=(1.35, 1.35, 1.0, 1.0, 1.0))
    hero = fig.add_subplot(grid[:, :2])
    image = _imshow(hero, maps["vehicle_feasibility"], config, cmap="viridis", vmin=0.0, vmax=1.0)
    _draw_selected_pair(hero, case)
    hero.set_title("a  Vehicle-conditioned terrain field", loc="left", fontweight="bold")
    _format_map_axis(hero, config)
    legend = [
        Line2D([0], [0], color=FLOW_COLOR, lw=1.7, label="Flow (minADE)"),
        Line2D([0], [0], color=FULL_COLOR, lw=1.7, label="VTF-Flow (minADE)"),
        Line2D([0], [0], color=GT_COLOR, lw=1.7, label="GT"),
    ]
    hero.legend(handles=legend, loc="upper right")
    _small_colorbar(fig, image, hero, "Relative F")

    contribution_names = (
        "occupancy_contribution", "nontraversable_contribution", "slope_contribution",
        "roughness_contribution", "clearance_contribution",
    )
    common_max = max(float(np.nanmax(maps[name])) for name in contribution_names)
    common_max = max(common_max, 1e-6)
    panels: Sequence[tuple[str, str, str, float | None, float | None, str]] = (
        ("encoded_mean_height_m", "b  Encoded mean-height input", "terrain", None, None, "m"),
        ("nontraversable_contribution", "c  Non-traversability contribution", "magma", 0.0, common_max, "cost"),
        ("occupancy_contribution", "d  Occupancy contribution", "magma", 0.0, common_max, "cost"),
        ("slope_contribution", "e  Slope contribution", "magma", 0.0, common_max, "cost"),
        ("roughness_contribution", "f  Roughness contribution", "magma", 0.0, common_max, "cost"),
        ("clearance_contribution", "g  Clearance-proximity contribution", "magma", 0.0, common_max, "cost"),
        ("terrain_cost", "h  Terrain-only cost", "magma", 0.0, 1.0, "cost"),
        ("vehicle_additional_cost", "i  Motion-state cost addition", "magma", 0.0, None, "cost"),
        ("vehicle_feasibility", "j  Final vehicle-conditioned F", "viridis", 0.0, 1.0, "relative F"),
    )
    for index, (name, title, cmap, vmin, vmax, colorbar_label) in enumerate(panels):
        row, col = divmod(index, 3)
        axis = fig.add_subplot(grid[row, col + 2])
        panel_image = _imshow(axis, maps[name], config, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title, loc="left", fontweight="bold", fontsize=6.0)
        _format_map_axis(axis, config, labels=False)
        axis.set_xticks((-10, 0, 10) if row == 2 else ())
        axis.set_yticks((0, 12, 24) if col == 0 else ())
        if row == 2:
            axis.set_xlabel("lateral y (m)", fontsize=5.5)
        if col == 0:
            axis.set_ylabel("forward x (m)", fontsize=5.5)
        _draw_origin(axis)
        _small_colorbar(fig, panel_image, axis, colorbar_label)
    heading_deg = np.degrees(float(case["heading_rad"]))
    fig.suptitle(
        f"Terrain-field decomposition | RELLIS-3D {case['scene_id']} | "
        f"planning ego: x forward, y left | v={case['speed_mps']:.2f} m/s, ψ={heading_deg:.1f}°",
        fontsize=float(style["font_size_pt"]) + 0.5,
    )
    fig.text(
        0.5, -0.012,
        "Final planner cache: traversability, obstacle density, encoded mean height. "
        "No semantic-ID channel is displayed; clearance is an occupancy-proximity proxy. "
        "F is relative, not a safety threshold.",
        ha="center", va="bottom", fontsize=5.5,
    )
    base = figures_dir / "figure_S1_terrain_field_decomposition"
    _save_figure(fig, base)
    return base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--style", type=Path, default=DEFAULT_STYLE)
    parser.add_argument("--field-config", type=Path, default=DEFAULT_FIELD_CONFIG)
    parser.add_argument("--case", default=DEFAULT_CASE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    style = _read_json(args.style)
    _configure_matplotlib(style)
    case = _load_case(args.output_root, args.cache_root, args.case, args.field_config)
    source_paths = _write_source_data(
        case, args.output_root / "figure_source_data", args.case
    )
    figures_dir = args.output_root / "figures"
    main_base = plot_main_figure(case, figures_dir, style)
    supplement_base = plot_decomposition_figure(case, figures_dir, style)
    result = {
        "status": "complete",
        "case": args.case,
        "scene_id": case["scene_id"],
        "main_figure": str(main_base),
        "decomposition_figure": str(supplement_base),
        "source_data": [str(path) for path in source_paths],
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

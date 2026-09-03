"""Render manuscript panels for the planner-used BEV and raw diagnostics.

The main inset uses the exact three-channel BEV consumed by VTF-Flow:
traversable fraction, obstacle density, and normalized mean height.  The
terrain-cost panel is reconstructed with the frozen analytic field.  The
supplementary diagnostic panel adds raw LiDAR slope, roughness, and semantic
class, plus the planner's occupancy-proximity proxy.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import torch


TERRAFLOW_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TERRAFLOW_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from TerraFlow.terrain.feasibility_field import (  # noqa: E402
    AnalyticTerrainField,
    TerrainFieldConfig,
)


DEFAULT_CACHE_ROOT = WORKSPACE_ROOT / "data" / "RELLIS3D" / "trajectory_cache_h150_s5"
DEFAULT_FIELD_CONFIG = TERRAFLOW_ROOT / "configs" / "rellis3d_flow_feasibility.json"
DEFAULT_RAW_FIELD = TERRAFLOW_ROOT / "outputs" / "terrain_fields" / "00004_000812_verified.npz"
DEFAULT_OUTPUT_DIR = (
    TERRAFLOW_ROOT / "outputs" / "final_experiments" / "figures" / "method_framework"
)
DEFAULT_SEQUENCE = "00004"
DEFAULT_FRAME = 812
FORWARD_M = 24.0
LATERAL_M = 12.0
MAIN_CMAP = LinearSegmentedColormap.from_list(
    "vtf_bev",
    [
        "#F8FAFC",
        "#DCEAF0",
        "#B7D6D8",
        "#78B7B0",
        "#3E8788",
        "#234F6D",
        "#162A46",
    ],
)
GT_COLOR = "#252525"
ORIGIN_COLOR = "#D62728"
OCCUPANCY_CONTOUR_COLOR = "#00BFC4"


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.5,
            "axes.labelsize": 6.5,
            "axes.titlesize": 7.4,
            "legend.fontsize": 5.8,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def _find_cache_row(manifest_path: Path, sequence: str, frame: int) -> int:
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            if (
                str(record["sequence"]).zfill(5) == str(sequence).zfill(5)
                and int(record["frame_index"]) == int(frame)
            ):
                return int(record["row"])
    raise KeyError(f"scene {sequence}:{frame} was not found in {manifest_path}")


def _load_scene(
    *,
    cache_root: Path,
    raw_field_path: Path,
    field_config_path: Path,
    sequence: str,
    frame: int,
) -> dict[str, Any]:
    split = "train"
    row = _find_cache_row(cache_root / split / "manifest.csv", sequence, frame)
    bev_archive = np.load(cache_root / split / "bev.npy", mmap_mode="r")
    goal_archive = np.load(cache_root / split / "goal.npy", mmap_mode="r")
    trajectory_archive = np.load(cache_root / split / "trajectory.npy", mmap_mode="r")
    bev = np.array(bev_archive[row], dtype=np.float32, copy=True) / 255.0
    goal = np.array(goal_archive[row], dtype=np.float32, copy=True)
    ground_truth = np.array(trajectory_archive[row], dtype=np.float32, copy=True)

    raw_config = json.loads(field_config_path.read_text(encoding="utf-8"))
    field_config = TerrainFieldConfig(**raw_config["terrain_field"])
    if not np.isclose(field_config.forward_m, FORWARD_M):
        raise ValueError("the plotting extent must match the frozen 24 m forward field")
    if not np.isclose(field_config.lateral_m, LATERAL_M):
        raise ValueError("the plotting extent must match the frozen +/-12 m lateral field")

    terrain = torch.from_numpy(bev).unsqueeze(0)
    field = AnalyticTerrainField(terrain, field_config)
    components = {
        name: value[0, 0].detach().cpu().numpy()
        for name, value in field.components.items()
    }
    denominator = (
        field_config.occupancy_weight
        + field_config.traversability_weight
        + field_config.slope_weight
        + field_config.roughness_weight
        + field_config.clearance_weight
    )
    terrain_cost = (
        field_config.occupancy_weight * components["occupancy"]
        + field_config.traversability_weight * components["nontraversable"]
        + field_config.slope_weight * components["slope"]
        + field_config.roughness_weight * components["roughness"]
        + field_config.clearance_weight * components["clearance"]
    ) / denominator
    terrain_cost = np.clip(terrain_cost, 0.0, 1.0)

    raw = np.load(raw_field_path, allow_pickle=False)
    raw_sequence = str(raw["sequence"].item()).zfill(5)
    raw_frame = int(raw["frame"].item())
    if raw_sequence != str(sequence).zfill(5) or raw_frame != int(frame):
        raise ValueError(
            f"raw terrain archive is {raw_sequence}:{raw_frame}, expected {sequence}:{frame}"
        )
    coordinate_status = str(raw["coordinate_status"].item())
    if "T_ego_sensor" not in coordinate_status:
        raise ValueError("raw diagnostic archive lacks an explicit verified ego transform")
    semantic_policy = json.loads(str(raw["semantic_policy_json"].item()))

    return {
        "sequence": str(sequence).zfill(5),
        "frame": int(frame),
        "split": split,
        "row": row,
        "bev": bev,
        "goal": goal,
        "ground_truth": ground_truth,
        "terrain_cost": terrain_cost,
        "components": components,
        "raw": {key: np.array(raw[key], copy=True) for key in raw.files},
        "semantic_policy": semantic_policy,
        "field_config": field_config,
        "coordinate_status": coordinate_status,
    }


def _format_axis(axis: plt.Axes, *, show_x: bool = True, show_y: bool = True) -> None:
    axis.set_xlim(0.0, FORWARD_M)
    axis.set_ylim(-LATERAL_M, LATERAL_M)
    axis.set_aspect("equal")
    axis.set_xticks([0, 12, 24])
    axis.set_yticks([-12, 0, 12])
    axis.grid(False)
    if not show_x:
        axis.tick_params(labelbottom=False)
    if not show_y:
        axis.tick_params(labelleft=False)


def _imshow(
    axis: plt.Axes,
    values: np.ndarray,
    *,
    cmap: str | mpl.colors.Colormap,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: mpl.colors.Normalize | None = None,
) -> mpl.image.AxesImage:
    return axis.imshow(
        np.asarray(values).T,
        extent=(0.0, FORWARD_M, -LATERAL_M, LATERAL_M),
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        rasterized=True,
    )


def _trajectory_with_origin(trajectory: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (np.zeros((1, trajectory.shape[-1]), dtype=trajectory.dtype), trajectory),
        axis=0,
    )


def _draw_gt(axis: plt.Axes, trajectory: np.ndarray, *, label: bool = False) -> None:
    points = _trajectory_with_origin(trajectory)
    (line,) = axis.plot(
        points[:, 0],
        points[:, 1],
        color=GT_COLOR,
        lw=1.45,
        solid_capstyle="round",
        zorder=7,
        label="GT trajectory" if label else None,
    )
    line.set_path_effects(
        [
            path_effects.Stroke(linewidth=2.65, foreground="#FFFFFF"),
            path_effects.Normal(),
        ]
    )
    origin = axis.scatter(
        [0.0],
        [0.0],
        marker="*",
        s=35,
        facecolor=ORIGIN_COLOR,
        edgecolor="white",
        linewidth=0.65,
        zorder=9,
        clip_on=False,
        label="Ego origin" if label else None,
    )
    origin.set_path_effects(
        [path_effects.Stroke(linewidth=1.6, foreground="#111111"), path_effects.Normal()]
    )


def _small_colorbar(
    fig: plt.Figure,
    image: mpl.image.AxesImage,
    axis: plt.Axes,
    label: str,
    *,
    ticks: list[float] | None = None,
) -> None:
    colorbar = fig.colorbar(image, ax=axis, fraction=0.045, pad=0.025)
    colorbar.set_label(label, labelpad=2.0)
    colorbar.ax.tick_params(width=0.5, length=2)
    if ticks is not None:
        colorbar.set_ticks(ticks)


def plot_main_inset(scene: dict[str, Any], output_dir: Path) -> Path:
    maps = (
        scene["bev"][0],
        scene["bev"][1],
        scene["bev"][2],
        scene["terrain_cost"],
    )
    titles = (
        r"a  Traversable fraction $T$",
        r"b  Obstacle density $O$",
        r"c  Normalized mean height $\bar{Z}$",
        r"d  Derived terrain cost $C_T$",
    )
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(7.2, 1.92),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    images = []
    for index, (axis, values, title) in enumerate(zip(axes, maps, titles)):
        image = _imshow(axis, values, cmap=MAIN_CMAP, vmin=0.0, vmax=1.0)
        images.append(image)
        axis.set_title(title, loc="left", fontweight="bold", pad=3)
        _format_axis(axis, show_y=index == 0)
    _draw_gt(axes[-1], scene["ground_truth"], label=True)
    axes[-1].legend(loc="upper right", handlelength=1.5, borderaxespad=0.35)
    fig.supxlabel("Ego-forward x (m)", y=-0.005)
    fig.supylabel("Ego-left y (m)", x=-0.005)
    colorbar = fig.colorbar(
        images[-1],
        ax=axes,
        location="right",
        fraction=0.024,
        pad=0.018,
        ticks=[0.0, 0.5, 1.0],
    )
    colorbar.set_label("Normalized value (low → high)", labelpad=2.0)
    fig.suptitle(
        f"Planner-used BEV | RELLIS-3D {scene['sequence']}, frame {scene['frame']:06d}",
        fontsize=7.5,
        y=1.035,
    )
    base = output_dir / "figure_method_bev_inset"
    _save_figure(fig, base)
    return base


def _semantic_display(
    semantic: np.ndarray,
    valid: np.ndarray,
    policy: dict[str, Any],
) -> tuple[np.ma.MaskedArray, ListedColormap, BoundaryNorm, list[Patch]]:
    present_ids = [
        int(value)
        for value in np.unique(semantic[valid])
        if int(value) >= 0
    ]
    preferred_colors = {
        "void": "#9E9E9E",
        "dirt": "#B07D4F",
        "grass": "#55A868",
        "tree": "#1B7837",
        "person": "#D62728",
        "bush": "#8C9A3C",
        "mud": "#8C6D31",
        "rubble": "#8172B2",
    }
    fallback = plt.get_cmap("tab20")
    encoded = np.full(semantic.shape, np.nan, dtype=np.float32)
    colors: list[str | tuple[float, float, float, float]] = []
    labels: list[str] = []
    for display_index, class_id in enumerate(present_ids):
        encoded[(semantic == class_id) & valid] = float(display_index)
        entry = policy.get(str(class_id), {"name": f"class {class_id}"})
        name = str(entry["name"])
        colors.append(preferred_colors.get(name, fallback(display_index % 20)))
        labels.append(name.capitalize())
    cmap = ListedColormap(colors)
    cmap.set_bad("white")
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5), len(colors))
    handles = [
        Patch(facecolor=color, edgecolor="none", label=label)
        for color, label in zip(colors, labels)
    ]
    return np.ma.masked_invalid(encoded), cmap, norm, handles


def plot_supplementary_diagnostics(scene: dict[str, Any], output_dir: Path) -> Path:
    raw = scene["raw"]
    slope = np.ma.masked_where(~raw["slope_valid"].astype(bool), raw["slope_deg"])
    roughness = np.ma.masked_where(
        ~raw["geometry_valid"].astype(bool), raw["roughness_m"]
    )
    semantic, semantic_cmap, semantic_norm, semantic_handles = _semantic_display(
        raw["semantic_class"],
        raw["semantic_valid"].astype(bool),
        scene["semantic_policy"],
    )
    proximity = scene["components"]["clearance"]
    occupancy = scene["bev"][1]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(5.35, 4.55),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    slope_image = _imshow(axes[0, 0], slope, cmap="inferno", vmin=0.0, vmax=60.0)
    roughness_image = _imshow(
        axes[0, 1], roughness, cmap="cividis", vmin=0.0, vmax=0.5
    )
    _imshow(
        axes[1, 0],
        semantic,
        cmap=semantic_cmap,
        norm=semantic_norm,
    )
    proximity_image = _imshow(
        axes[1, 1], proximity, cmap="magma", vmin=0.0, vmax=1.0
    )

    panel_titles = (
        r"a  Raw local slope",
        r"b  Raw height roughness",
        r"c  Dominant semantic class",
        r"d  Occupancy / proximity proxy $C_d$",
    )
    for index, (axis, title) in enumerate(zip(axes.flat, panel_titles)):
        axis.set_title(title, loc="left", fontweight="bold", pad=3)
        _format_axis(
            axis,
            show_x=index >= 2,
            show_y=index % 2 == 0,
        )
        _draw_gt(axis, scene["ground_truth"], label=False)

    cell_x = (np.arange(occupancy.shape[0]) + 0.5) * FORWARD_M / occupancy.shape[0]
    cell_y = (
        -LATERAL_M
        + (np.arange(occupancy.shape[1]) + 0.5)
        * (2.0 * LATERAL_M / occupancy.shape[1])
    )
    occupancy_level = 0.25
    if float(np.nanmax(occupancy)) >= occupancy_level:
        axes[1, 1].contour(
            cell_x,
            cell_y,
            occupancy.T,
            levels=[occupancy_level],
            colors=[OCCUPANCY_CONTOUR_COLOR],
            linewidths=0.8,
            zorder=6,
        )

    _small_colorbar(fig, slope_image, axes[0, 0], "Slope (degrees)", ticks=[0, 30, 60])
    _small_colorbar(
        fig,
        roughness_image,
        axes[0, 1],
        "Height SD (m)",
        ticks=[0.0, 0.25, 0.5],
    )
    axes[1, 0].legend(
        handles=semantic_handles,
        loc="upper right",
        borderaxespad=0.3,
        handlelength=1.0,
        handletextpad=0.35,
    )
    _small_colorbar(
        fig,
        proximity_image,
        axes[1, 1],
        "Proximity cost",
        ticks=[0.0, 0.5, 1.0],
    )
    axes[1, 1].legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=OCCUPANCY_CONTOUR_COLOR,
                lw=1.0,
                label=r"$O=0.25$ contour",
            )
        ],
        loc="upper right",
        borderaxespad=0.3,
    )
    fig.supxlabel("Ego-forward x (m)", y=0.005)
    fig.supylabel("Ego-left y (m)", x=0.005)
    fig.suptitle(
        f"Terrain diagnostics | RELLIS-3D {scene['sequence']}, frame {scene['frame']:06d}",
        fontsize=7.5,
        y=1.015,
    )
    base = output_dir / "figure_supplementary_bev_diagnostics"
    _save_figure(fig, base)
    return base


def _write_source_data(scene: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    data_path = source_dir / "method_bev_scene_00004_000812.npz"
    raw = scene["raw"]
    np.savez_compressed(
        data_path,
        traversable_fraction=scene["bev"][0],
        obstacle_density=scene["bev"][1],
        normalized_mean_height=scene["bev"][2],
        terrain_cost=scene["terrain_cost"],
        planner_slope_proxy=scene["components"]["slope"],
        planner_roughness_proxy=scene["components"]["roughness"],
        planner_proximity_proxy=scene["components"]["clearance"],
        raw_slope_deg=raw["slope_deg"],
        raw_roughness_m=raw["roughness_m"],
        raw_semantic_class=raw["semantic_class"],
        raw_occupancy=raw["occupancy"],
        raw_geometry_valid=raw["geometry_valid"],
        raw_slope_valid=raw["slope_valid"],
        raw_semantic_valid=raw["semantic_valid"],
        ground_truth=scene["ground_truth"],
    )
    metadata_path = source_dir / "method_bev_scene_00004_000812_metadata.json"
    metadata = {
        "scene": {
            "sequence": scene["sequence"],
            "frame": scene["frame"],
            "split": scene["split"],
            "cache_row": scene["row"],
        },
        "selection_rationale": (
            "Representative right-turn scene selected without model-performance criteria: "
            "complete GT within the planning window, broad LiDAR support, and visible terrain boundaries."
        ),
        "coordinate_convention": (
            "planning ego: x forward, y left, z up; current ego at origin"
        ),
        "extent_m": {
            "forward": [0.0, FORWARD_M],
            "lateral": [-LATERAL_M, LATERAL_M],
        },
        "planner_input_channels": [
            "traversable_fraction",
            "obstacle_density",
            "normalized_mean_height",
        ],
        "terrain_cost": (
            "C_T=(2.0*C_o+1.2*C_nt+0.8*C_s+0.6*C_r+0.8*C_d)/5.4, clipped to [0,1]"
        ),
        "semantic_note": (
            "Dominant semantic class comes from the verified raw LiDAR diagnostic archive "
            "and is not an input channel of the frozen final planner."
        ),
        "proximity_note": (
            "C_d is a 9x9 max-pool occupancy-proximity proxy from the planner BEV, "
            "not metric Euclidean obstacle clearance."
        ),
        "trajectory_note": (
            "GT is the future logged vehicle trajectory derived from poses; it is not an "
            "optimality or safety certificate."
        ),
        "coordinate_status": scene["coordinate_status"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return data_path, metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render planner-used BEV and supplementary terrain diagnostic panels."
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--raw-field", type=Path, default=DEFAULT_RAW_FIELD)
    parser.add_argument("--field-config", type=Path, default=DEFAULT_FIELD_CONFIG)
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
    main_base = plot_main_inset(scene, args.output_dir)
    supplementary_base = plot_supplementary_diagnostics(scene, args.output_dir)
    data_path, metadata_path = _write_source_data(scene, args.output_dir)
    print(f"main figure: {main_base}")
    print(f"supplementary figure: {supplementary_base}")
    print(f"source data: {data_path}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()

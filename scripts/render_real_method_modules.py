"""Render data-grounded method-framework modules for VTF-Flow.

The first module shows the five normalized components of the exact
planner-used static terrain field.  The second combines all frozen VTF-Flow
candidates on the corresponding static terrain potential with a calibrated
camera projection of the lowest-trajectory-potential candidate.  Candidate
selection never uses ground-truth error.
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
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image
import torch


TERRAFLOW_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TERRAFLOW_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from TerraFlow.scripts.plot_unified_h10_qualitative import (  # noqa: E402
    _load_predictions,
    _terrain_cost_map,
)
from TerraFlow.scripts.render_camera_bev_tvk_interpretation import (  # noqa: E402
    _pointwise_tvk_cost,
)
from TerraFlow.scripts.render_camera_trajectory_results import (  # noqa: E402
    DEFAULT_CAMERA_INTRINSICS,
    DEFAULT_EXTRINSIC_VARIANT,
    DEFAULT_SENSOR_TRANSFORM,
    _camera_image,
    _load_calibration,
    _project,
)
from TerraFlow.scripts.train_regression import CombinedSceneDataset  # noqa: E402
from TerraFlow.terrain.feasibility_field import (  # noqa: E402
    AnalyticTerrainField,
    TerrainFieldConfig,
)
from TerraFlow.terrain.trajectory_kinematics import (  # noqa: E402
    TrajectoryKinematicConfig,
)
from TerraFlow.terrain.vehicle_conditioned_field import (  # noqa: E402
    VehicleConditionedFieldConfig,
)


DEFAULT_DATA_ROOT = WORKSPACE_ROOT / "data" / "RELLIS3D"
DEFAULT_CACHE_ROOT = DEFAULT_DATA_ROOT / "trajectory_cache_h150_s5"
DEFAULT_BENCHMARK = TERRAFLOW_ROOT / "outputs" / "unified_h10_benchmark"
DEFAULT_OUTPUT = DEFAULT_BENCHMARK / "figures" / "method_framework_real"

COMPONENT_COLORS: Mapping[str, Sequence[str]] = {
    "occupancy": ("#FFF7EC", "#FDD49E", "#FC8D59", "#B30000"),
    "nontraversable": ("#F7FBFF", "#C6DBEF", "#6BAED6", "#08519C"),
    "slope": ("#FFFFE5", "#D9F0A3", "#78C679", "#238443"),
    "roughness": ("#FCFBFD", "#DADAEB", "#9E9AC8", "#6A51A3"),
    "clearance": ("#FFF5EB", "#FDD0A2", "#F16913", "#A63603"),
}
COMPONENT_TITLES = {
    "occupancy": "Occupancy\n$C_o$",
    "nontraversable": "Non-traversability\n$C_{nt}$",
    "slope": "Slope\n$C_s$",
    "roughness": "Roughness\n$C_r$",
    "clearance": "Obstacle proximity\n$C_d$",
}
CANDIDATE_COLORS = (
    "#0072B2",
    "#009E73",
    "#E69F00",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#6F4E7C",
    "#7A7A7A",
)
SELECTED_COLOR = "#C15B92"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.2,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.2,
            "axes.linewidth": 0.7,
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _component_cmap(name: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        f"vtf_{name}", list(COMPONENT_COLORS[name])
    )


def _rounded_card(fig: plt.Figure, edgecolor: str) -> None:
    card = FancyBboxPatch(
        (0.012, 0.025),
        0.976,
        0.95,
        boxstyle="round,pad=0.006,rounding_size=0.024",
        transform=fig.transFigure,
        facecolor="white",
        edgecolor=edgecolor,
        linewidth=0.85,
        zorder=-20,
        clip_on=False,
    )
    fig.patches.append(card)


def _save_all(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", transparent=True)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", transparent=True)
    fig.savefig(
        base.with_suffix(".png"), dpi=600, bbox_inches="tight", transparent=True
    )
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def _load_material(
    *,
    data_root: Path,
    cache_root: Path,
    benchmark_root: Path,
    category: str,
    sensor_transform: Path,
    camera_intrinsics: Path,
    extrinsic_variant: str,
) -> dict[str, Any]:
    protocol = _read_json(benchmark_root / "effective_protocol.json")
    selection = _read_json(
        benchmark_root
        / "figure_source_data"
        / "selected_advantage_scenes"
        / "selection_manifest.json"
    )
    matching = [
        record
        for record in selection["selected_scenes"]
        if str(record["category"]) == category
    ]
    if len(matching) != 1:
        raise ValueError(f"expected one scene for category {category!r}, got {len(matching)}")
    record = matching[0]

    effective = _read_json(
        benchmark_root
        / "checkpoints"
        / "seed_0"
        / "flow_tvk"
        / "effective_config.json"
    )
    terrain_config = TerrainFieldConfig(**effective["terrain_field"])
    vehicle_config = VehicleConditionedFieldConfig(**effective["vehicle_conditioning"])
    kinematic_config = TrajectoryKinematicConfig(**protocol["kinematic"])
    planning_dt_s = float(protocol["trajectory"]["planning_dt_s"])

    dataset = CombinedSceneDataset(
        cache_root, tuple(protocol["protocol"]["source_splits"])
    )
    scene = dataset[int(record["dataset_index"])]
    sequence = str(record["sequence"])
    frame = int(record["frame_id"])
    if str(scene.metadata["sequence"]) != sequence or int(
        scene.metadata["frame_id"]
    ) != frame:
        raise ValueError("selected record and cached scene are not index-aligned")

    trajectories, ground_truth, scene_ids = _load_predictions(benchmark_root)
    scene_lookup = {scene_id: index for index, scene_id in enumerate(scene_ids)}
    prediction_index = int(scene_lookup[str(record["scene_id"])])
    candidates = trajectories["VTF"][prediction_index]
    if candidates.ndim != 3 or candidates.shape[-1] != 3:
        raise ValueError(f"unexpected candidate tensor shape {candidates.shape}")

    terrain_map = scene.terrain_map.float()
    field = AnalyticTerrainField(terrain_map.unsqueeze(0), terrain_config)
    components = {
        name: values[0, 0].detach().cpu().numpy()
        for name, values in field.components.items()
    }
    cost_map, extent = _terrain_cost_map(terrain_map, terrain_config, samples=192)
    if not np.all(np.isfinite(cost_map)):
        raise ValueError("static terrain cost contains non-finite values")

    pointwise = [
        _pointwise_tvk_cost(
            candidate,
            terrain_map,
            terrain_config,
            vehicle_config,
            kinematic_config,
            planning_dt_s,
        )
        for candidate in candidates
    ]
    candidate_potentials = np.asarray(
        [float(values["tvk_cost"].mean()) for values in pointwise], dtype=np.float64
    )
    selected_index = int(np.argmin(candidate_potentials))

    calibration = _load_calibration(
        data_root,
        sequence,
        sensor_transform,
        camera_intrinsics,
        extrinsic_variant,
    )
    image_path = _camera_image(data_root, sequence, frame)
    image = np.asarray(Image.open(image_path).convert("RGB"))
    pixels, visible, depth = _project(candidates[selected_index], calibration)
    if int(np.count_nonzero(visible)) < 3:
        raise RuntimeError("lowest-potential candidate has fewer than three visible points")

    return {
        "record": record,
        "protocol": protocol,
        "terrain_config": terrain_config,
        "vehicle_config": vehicle_config,
        "kinematic_config": kinematic_config,
        "planning_dt_s": planning_dt_s,
        "scene": scene,
        "terrain_map": terrain_map,
        "components": components,
        "cost_map": cost_map,
        "extent": extent,
        "candidates": candidates,
        "ground_truth": ground_truth[prediction_index],
        "pointwise": pointwise,
        "candidate_potentials": candidate_potentials,
        "selected_index": selected_index,
        "calibration": calibration,
        "image_path": image_path,
        "image": image,
        "selected_pixels": pixels,
        "selected_visible": visible,
        "selected_depth": depth,
    }


def _render_components(material: Mapping[str, Any], output_dir: Path) -> Path:
    keys = ("occupancy", "nontraversable", "slope", "roughness", "clearance")
    fig = plt.figure(figsize=(7.2, 2.05), facecolor="none")
    _rounded_card(fig, "#3D745B")
    fig.text(
        0.5,
        0.925,
        "Level 1 — Static terrain-feasibility potential field $C_T(x,y)$",
        ha="center",
        va="center",
        fontsize=9.0,
        fontweight="bold",
        color="#245B43",
    )
    starts = (0.035, 0.225, 0.415, 0.605, 0.795)
    heat_width = 0.115
    heat_y = 0.185
    heat_height = 0.56
    for key, start in zip(keys, starts):
        axis = fig.add_axes((start, heat_y, heat_width, heat_height))
        image = axis.imshow(
            material["components"][key].T,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=_component_cmap(key),
            vmin=0.0,
            vmax=1.0,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#9BA8A1")
            spine.set_linewidth(0.5)
        axis.set_title(COMPONENT_TITLES[key], pad=3.0, fontsize=7.4)
        color_axis = fig.add_axes((start + heat_width + 0.009, heat_y, 0.009, heat_height))
        colorbar = fig.colorbar(image, cax=color_axis)
        colorbar.set_ticks([0.0, 1.0])
        colorbar.set_ticklabels(["low", "high"])
        colorbar.ax.tick_params(labelsize=5.5, length=1.8, pad=1.3)
        colorbar.outline.set_linewidth(0.45)
    record = material["record"]
    fig.text(
        0.5,
        0.085,
        "All components normalized to [0, 1]; higher values indicate larger local potential",
        ha="center",
        va="center",
        fontsize=6.3,
        color="#52635B",
    )
    fig.text(
        0.985,
        0.055,
        f"RELLIS-3D {record['sequence']} · frame {int(record['frame_id']):06d}",
        ha="right",
        va="center",
        fontsize=5.6,
        color="#728078",
    )
    base = output_dir / "module_real_static_terrain_components"
    _save_all(fig, base)
    return base


def _draw_bev_candidates(axis: plt.Axes, material: Mapping[str, Any]) -> mpl.image.AxesImage:
    image = axis.imshow(
        material["cost_map"],
        origin="lower",
        extent=material["extent"],
        aspect="equal",
        interpolation="bilinear",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    selected = int(material["selected_index"])
    for candidate_index, candidate in enumerate(material["candidates"]):
        is_selected = candidate_index == selected
        color = SELECTED_COLOR if is_selected else "#7A7A7A"
        width = 2.05 if is_selected else 0.9
        alpha = 1.0 if is_selected else 0.72
        line, = axis.plot(
            candidate[:, 0], candidate[:, 1], color=color, linewidth=width,
            alpha=alpha, linestyle="-" if is_selected else "--",
            solid_capstyle="round", zorder=6 if is_selected else 5,
        )
        line.set_path_effects(
            [path_effects.Stroke(linewidth=width + 0.9, foreground="white"), path_effects.Normal()]
        )
        axis.scatter(
            [candidate[-1, 0]], [candidate[-1, 1]], s=8 if not is_selected else 18,
            color=color, edgecolor="white", linewidth=0.35, zorder=6,
        )
    axis.scatter(
        [0.0], [0.0], marker="*", s=48, facecolor="white", edgecolor="#1F2D3A",
        linewidth=0.8, zorder=8,
    )
    candidate_xy = material["candidates"][..., :2].reshape(-1, 2)
    x_max = min(material["extent"][1], max(7.0, float(candidate_xy[:, 0].max()) + 0.7))
    y_center = 0.5 * float(candidate_xy[:, 1].min() + candidate_xy[:, 1].max())
    y_half = max(2.6, 0.5 * float(np.ptp(candidate_xy[:, 1])) + 0.7)
    axis.set_xlim(0.0, x_max)
    axis.set_ylim(
        max(material["extent"][2], y_center - y_half),
        min(material["extent"][3], y_center + y_half),
    )
    axis.set_xlabel("Ego-forward $x$ (m)")
    axis.set_ylabel("Ego-left $y$ (m)")
    axis.grid(color="white", alpha=0.16, linewidth=0.45)
    axis.set_title(
        f"a  Static $C_T(x,y)$ with $K={len(material['candidates'])}$ guided candidates (trajectory ROI)",
        loc="left", fontweight="bold", pad=5,
    )
    return image


def _visible_segments(
    pixels: np.ndarray, visible: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    segments = []
    indices = []
    for index in range(len(pixels) - 1):
        if bool(visible[index] and visible[index + 1]):
            segments.append(np.stack((pixels[index], pixels[index + 1]), axis=0))
            indices.append(index)
    if not segments:
        raise RuntimeError("no contiguous visible trajectory segments")
    return np.asarray(segments), np.asarray(indices, dtype=np.int64)


def _draw_camera_tvk(
    axis: plt.Axes,
    material: Mapping[str, Any],
    normalization: Normalize,
) -> LineCollection:
    image = material["image"]
    pixels = material["selected_pixels"]
    visible = material["selected_visible"]
    values = material["pointwise"][material["selected_index"]]["tvk_cost"]
    segments, indices = _visible_segments(pixels, visible)

    axis.imshow(image)
    underlay = LineCollection(
        segments, colors="white", linewidths=4.5, capstyle="round", zorder=5
    )
    axis.add_collection(underlay)
    collection = LineCollection(
        segments,
        cmap="viridis",
        norm=normalization,
        linewidths=2.7,
        capstyle="round",
        zorder=6,
    )
    collection.set_array(0.5 * (values[indices] + values[indices + 1]))
    axis.add_collection(collection)
    visible_indices = np.flatnonzero(visible)
    axis.scatter(
        pixels[visible_indices, 0],
        pixels[visible_indices, 1],
        c=values[visible_indices],
        cmap="viridis",
        norm=normalization,
        s=23,
        edgecolor="white",
        linewidth=0.75,
        zorder=7,
    )

    visible_pixels = pixels[visible]
    height, width = image.shape[:2]
    u_min = max(0.0, float(visible_pixels[:, 0].min()) - 300.0)
    u_max = min(float(width), float(visible_pixels[:, 0].max()) + 300.0)
    v_min = max(0.0, float(visible_pixels[:, 1].min()) - 230.0)
    v_max = min(float(height), float(visible_pixels[:, 1].max()) + 85.0)
    if u_max - u_min < 620.0:
        center = 0.5 * (u_min + u_max)
        u_min = max(0.0, center - 310.0)
        u_max = min(float(width), center + 310.0)
    axis.set_xlim(u_min, u_max)
    axis.set_ylim(v_max, v_min)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_title(
        "b  Pointwise $C_{TVK,i}$ along the lowest-$J_{TVK}$ candidate",
        loc="left", fontweight="bold", pad=5,
    )
    return collection


def _render_candidates_and_camera(
    material: Mapping[str, Any], output_dir: Path
) -> Path:
    selected_values = material["pointwise"][material["selected_index"]]["tvk_cost"]
    vmax = max(0.1, float(np.quantile(selected_values, 0.95)))
    normalization = Normalize(vmin=0.0, vmax=vmax, clip=True)

    fig = plt.figure(figsize=(7.2, 2.95), facecolor="none")
    _rounded_card(fig, "#39758A")
    grid = fig.add_gridspec(
        1, 2, left=0.075, right=0.905, bottom=0.18, top=0.80,
        width_ratios=(1.08, 1.48), wspace=0.32,
    )
    bev_axis = fig.add_subplot(grid[0, 0])
    camera_axis = fig.add_subplot(grid[0, 1])
    terrain_image = _draw_bev_candidates(bev_axis, material)
    tvk_collection = _draw_camera_tvk(camera_axis, material, normalization)

    terrain_cax = fig.add_axes((0.405, 0.27, 0.012, 0.42))
    terrain_cbar = fig.colorbar(terrain_image, cax=terrain_cax, orientation="vertical")
    terrain_cbar.set_ticks([0.0, 1.0])
    terrain_cbar.set_ticklabels(["low\nmore feasible", "high\nless feasible"])
    terrain_cbar.set_label("Static terrain potential $C_T$")
    terrain_cbar.ax.tick_params(labelsize=5.4, pad=1.0)

    tvk_cax = fig.add_axes((0.92, 0.27, 0.012, 0.42))
    tvk_cbar = fig.colorbar(tvk_collection, cax=tvk_cax, orientation="vertical")
    tvk_cbar.set_ticks([0.0, vmax])
    tvk_cbar.set_ticklabels(["low\nmore feasible", "high\nless feasible"])
    tvk_cbar.set_label("Pointwise unified potential $C_{TVK,i}$")
    tvk_cbar.ax.tick_params(labelsize=5.4, pad=1.0)

    selected = int(material["selected_index"])
    handles = [
        Line2D(
            [0], [0], color="#7A7A7A", lw=1.1, linestyle="--",
            label="Frozen VTF-Flow candidates",
        ),
        Line2D(
            [0], [0], color=SELECTED_COLOR, lw=2.0,
            label=f"Lowest $J_{{TVK}}$ candidate (index {selected})",
        ),
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.94),
        ncol=2, fontsize=7.2,
    )
    record = material["record"]
    fig.text(
        0.5,
        0.045,
        f"RELLIS-3D {record['sequence']}, frame {int(record['frame_id']):06d} · "
        "calibrated qualitative projection; camera-space occlusion is not modelled",
        ha="center",
        va="center",
        fontsize=5.8,
        color="#62727A",
    )
    base = output_dir / "module_real_candidates_camera_tvk"
    _save_all(fig, base)
    return base


def _write_source_data(material: Mapping[str, Any], output_dir: Path) -> Path:
    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        source_dir / "static_terrain_components.npz",
        **material["components"],
        static_terrain_cost=material["cost_map"],
    )
    rows = []
    for candidate_index, (trajectory, values) in enumerate(
        zip(material["candidates"], material["pointwise"])
    ):
        for step, xyz in enumerate(trajectory):
            rows.append(
                {
                    "candidate_index": candidate_index,
                    "selected_lowest_j_tvk": candidate_index == material["selected_index"],
                    "step": step,
                    "x_m": float(xyz[0]),
                    "y_m": float(xyz[1]),
                    "z_m": float(xyz[2]),
                    "terrain_vehicle_cost": float(values["terrain_vehicle_cost"][step]),
                    "kinematic_cost": float(values["kinematic_cost"][step]),
                    "tvk_cost": float(values["tvk_cost"][step]),
                    "speed_mps": float(values["speed_mps"][step]),
                    "curvature_per_m": float(values["curvature_per_m"][step]),
                    "lateral_acceleration_mps2": float(
                        values["lateral_acceleration_mps2"][step]
                    ),
                }
            )
    csv_path = source_dir / "candidate_trajectories_and_pointwise_tvk.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    record = material["record"]
    manifest = {
        "core_conclusion": (
            "The five static terrain-potential components come from the exact planner-used "
            "RELLIS-3D BEV, while complete TVK potential is candidate-dependent and is "
            "therefore encoded along frozen VTF-Flow trajectories."
        ),
        "scene_id": record["scene_id"],
        "sequence": record["sequence"],
        "frame_id": int(record["frame_id"]),
        "scene_category": record["category"],
        "prediction_source": "unified H=10 benchmark, VTF seed 0, K=8",
        "candidate_selection": "minimum mean pointwise TVK potential; no GT error used",
        "selected_candidate_index": int(material["selected_index"]),
        "candidate_mean_j_tvk": material["candidate_potentials"].tolist(),
        "selected_camera_visible_fraction": float(
            np.mean(material["selected_visible"])
        ),
        "static_components": [
            "occupancy",
            "nontraversable",
            "normalized slope",
            "normalized roughness",
            "occupancy-proximity proxy",
        ],
        "camera_projection": (
            "released calibration, positive-depth and image-bound filters; no occlusion model"
        ),
        "camera_processing": "RGB conversion only; no brightness/contrast/gamma adjustment",
        "population_boundary": "quantitative conclusions use all 1909 protocol test scenes",
        "source_data": [
            str((source_dir / "static_terrain_components.npz").resolve()),
            str(csv_path.resolve()),
        ],
    }
    manifest_path = source_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--category", choices=("terrain", "balanced"), default="balanced")
    parser.add_argument("--sensor-transform", type=Path, default=DEFAULT_SENSOR_TRANSFORM)
    parser.add_argument("--camera-intrinsics", type=Path, default=DEFAULT_CAMERA_INTRINSICS)
    parser.add_argument("--extrinsic-variant", default=DEFAULT_EXTRINSIC_VARIANT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _configure()
    material = _load_material(
        data_root=args.data_root.resolve(),
        cache_root=args.cache_root.resolve(),
        benchmark_root=args.benchmark_root.resolve(),
        category=args.category,
        sensor_transform=args.sensor_transform.resolve(),
        camera_intrinsics=args.camera_intrinsics.resolve(),
        extrinsic_variant=args.extrinsic_variant,
    )
    components_base = _render_components(material, args.output_dir.resolve())
    candidate_base = _render_candidates_and_camera(material, args.output_dir.resolve())
    manifest = _write_source_data(material, args.output_dir.resolve())
    summary = {
        "components": str(components_base.resolve()),
        "candidates_camera": str(candidate_base.resolve()),
        "manifest": str(manifest.resolve()),
        "selected_candidate_index": int(material["selected_index"]),
        "candidate_mean_j_tvk": material["candidate_potentials"].tolist(),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

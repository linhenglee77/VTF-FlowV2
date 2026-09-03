"""Render camera/BEV interpretation plates for the VTF-Flow feasibility field.

The synchronized camera frame provides environmental context, the planner's
static terrain cost is shown without perspective distortion in BEV, and the
complete candidate-dependent TVK objective is encoded pointwise along the
VTF-Flow trajectory instead of being misrepresented as a fixed image field.
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
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image
import torch


TERRAFLOW_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TERRAFLOW_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from TerraFlow.scripts.plot_unified_h10_qualitative import (  # noqa: E402
    _best_candidate,
    _load_predictions,
    _terrain_cost_map,
)
from TerraFlow.scripts.render_camera_trajectory_results import (  # noqa: E402
    DEFAULT_CAMERA_INTRINSICS,
    DEFAULT_EXTRINSIC_VARIANT,
    DEFAULT_SENSOR_TRANSFORM,
    _camera_image,
    _draw_projected,
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
    trajectory_kinematic_cost,
)
from TerraFlow.terrain.vehicle_conditioned_field import (  # noqa: E402
    BatchedVehicleConditionedTerrainField,
    VehicleConditionedFieldConfig,
    trajectory_motion_state,
)


DEFAULT_DATA_ROOT = WORKSPACE_ROOT / "data" / "RELLIS3D"
DEFAULT_CACHE_ROOT = DEFAULT_DATA_ROOT / "trajectory_cache_h150_s5"
DEFAULT_BENCHMARK = TERRAFLOW_ROOT / "outputs" / "unified_h10_benchmark"

METHODS = ("FLOW", "VT", "VTF")
DISPLAY_NAMES = {
    "FLOW": "Flow baseline",
    "VT": "VTF-Flow w/o kinematic terms",
    "VTF": "VTF-Flow (ours)",
}
COLORS = {"FLOW": "#7C8796", "VT": "#3478A8", "VTF": "#C84D58"}
CATEGORY_NAMES = {
    "terrain": "Terrain-violation reduction",
    "balanced": "Balanced terrain–kinematic gain",
    "long_visible": "Long trajectory with high camera visibility",
}
STATIC_CMAP = mpl.colormaps["magma"]
TVK_CMAP = mpl.colormaps["viridis"]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _record_for_scene(metrics_path: Path, scene_id: str) -> dict[str, Any]:
    """Build a rendering record for one protocol scene without changing selection files."""

    with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        row = next((item for item in rows if str(item["scene_id"]) == scene_id), None)
    if row is None:
        raise ValueError(f"scene_id is not present in the benchmark metrics: {scene_id}")
    return {
        "category": "long_visible",
        "selection_metric": "GT/VTF path length and calibrated projection visibility",
        "scene_id": scene_id,
        "sequence": str(int(float(row["FLOW_sequence"]))).zfill(5),
        "frame_id": int(float(row["FLOW_frame_id"])),
        "dataset_index": int(float(row["FLOW_dataset_index"])),
    }


def _configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.2,
            "axes.linewidth": 0.7,
            "legend.fontsize": 6.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _save_all(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(
        base.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
        facecolor="white", pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def _clean_camera_axis(axis: plt.Axes, image: np.ndarray) -> None:
    axis.set_xlim(0, image.shape[1])
    axis.set_ylim(image.shape[0], 0)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _set_trajectory_roi(
    axis: plt.Axes,
    item: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> None:
    """Magnify the calibrated trajectory footprint without altering pixels."""

    visible_pixels = []
    for path in list(item["paths"].values()) + [item["ground_truth"]]:
        pixels, visible, _ = _project(path, calibration)
        if bool(np.any(visible)):
            visible_pixels.append(pixels[visible])
    if not visible_pixels:
        return
    pixels = np.concatenate(visible_pixels, axis=0)
    image_height, image_width = item["image"].shape[:2]
    u_min = max(0.0, float(pixels[:, 0].min()) - 240.0)
    u_max = min(float(image_width), float(pixels[:, 0].max()) + 240.0)
    v_min = max(0.0, float(pixels[:, 1].min()) - 190.0)
    v_max = min(float(image_height), float(pixels[:, 1].max()) + 35.0)
    axis.set_xlim(u_min, u_max)
    axis.set_ylim(v_max, v_min)


def _draw_camera_path(
    axis: plt.Axes,
    trajectory: np.ndarray,
    calibration: Mapping[str, Any],
    *,
    color: str,
    linewidth: float,
    linestyle: str = "-",
    zorder: float = 6,
) -> float:
    pixels, visible, _ = _project(trajectory, calibration)
    _draw_projected(
        axis, pixels, visible, color="white", linewidth=linewidth + 1.25,
        linestyle=linestyle, alpha=0.9, zorder=zorder - 0.1,
    )
    _draw_projected(
        axis, pixels, visible, color=color, linewidth=linewidth,
        linestyle=linestyle, alpha=1.0, zorder=zorder,
    )
    return float(np.mean(visible))


def _pointwise_tvk_cost(
    trajectory: np.ndarray,
    terrain_map: torch.Tensor,
    terrain_config: TerrainFieldConfig,
    vehicle_config: VehicleConditionedFieldConfig,
    kinematic_config: TrajectoryKinematicConfig,
    planning_dt_s: float,
) -> dict[str, np.ndarray]:
    path = torch.as_tensor(trajectory, dtype=torch.float32).unsqueeze(0)
    terrain_field = AnalyticTerrainField(terrain_map.float().unsqueeze(0), terrain_config)
    vehicle_field = BatchedVehicleConditionedTerrainField(terrain_field, vehicle_config)
    motion = trajectory_motion_state(path, planning_dt_s, vehicle_config)
    with torch.no_grad():
        vehicle = vehicle_field.cost(path, motion)[0]
        kinematic = trajectory_kinematic_cost(path, planning_dt_s, kinematic_config)
        kinematic_point = kinematic["pointwise_kinematic_cost"][0]
    return {
        "terrain_vehicle_cost": vehicle.cpu().numpy(),
        "kinematic_cost": kinematic_point.cpu().numpy(),
        "tvk_cost": (vehicle + kinematic_point).cpu().numpy(),
        "speed_mps": motion["speed"][0].cpu().numpy(),
        "curvature_per_m": kinematic["absolute_curvature_per_m"][0].cpu().numpy(),
        "lateral_acceleration_mps2": kinematic["lateral_acceleration_mps2"][0].cpu().numpy(),
    }


def _draw_bev(
    axis: plt.Axes,
    item: Mapping[str, Any],
) -> mpl.image.AxesImage:
    image = axis.imshow(
        item["cost_map"], origin="lower", extent=item["extent"], cmap=STATIC_CMAP,
        vmin=0.0, vmax=1.0, interpolation="bilinear", aspect="equal",
    )
    for method, style, width in (
        ("FLOW", "--", 1.15), ("VT", "-.", 1.25), ("VTF", "-", 1.75),
    ):
        path = item["paths"][method]
        line, = axis.plot(
            path[:, 0], path[:, 1], color=COLORS[method], linestyle=style,
            linewidth=width, zorder=7,
        )
        line.set_path_effects(
            [path_effects.Stroke(linewidth=width + 0.9, foreground="white"), path_effects.Normal()]
        )
    gt = item["ground_truth"]
    line, = axis.plot(gt[:, 0], gt[:, 1], color="#171A1F", linewidth=1.65, zorder=8)
    line.set_path_effects(
        [path_effects.Stroke(linewidth=2.55, foreground="white"), path_effects.Normal()]
    )
    axis.scatter(
        0.0, 0.0, marker="*", s=48, color="#D62F2F", edgecolor="white",
        linewidth=0.7, zorder=10,
    )
    axis.set_xlim(item["extent"][0], item["extent"][1])
    axis.set_ylim(item["extent"][2], item["extent"][3])
    axis.set_xlabel("Ego-forward x (m)")
    axis.set_ylabel("Ego-left y (m)")
    axis.grid(color="white", linewidth=0.3, alpha=0.25)
    return image


def _draw_tvk_colored_trajectory(
    axis: plt.Axes,
    item: Mapping[str, Any],
    calibration: Mapping[str, Any],
    normalization: Normalize,
) -> None:
    axis.imshow(item["image"])
    for method, style, width in (("FLOW", "--", 1.2), ("VT", "-.", 1.25)):
        _draw_camera_path(
            axis, item["paths"][method], calibration, color=COLORS[method],
            linewidth=width, linestyle=style, zorder=6,
        )
    _draw_camera_path(
        axis, item["ground_truth"], calibration, color="#171A1F",
        linewidth=1.45, zorder=7,
    )
    path = item["paths"]["VTF"]
    pixels, visible, _ = _project(path, calibration)
    costs = item["pointwise"]["VTF"]["tvk_cost"]
    valid_segments = visible[:-1] & visible[1:]
    segments = np.stack((pixels[:-1], pixels[1:]), axis=1)[valid_segments]
    segment_cost = (0.5 * (costs[:-1] + costs[1:]))[valid_segments]
    if segments.size:
        outline = LineCollection(
            segments, colors="white", linewidths=4.2, zorder=8,
            capstyle="round", joinstyle="round",
        )
        axis.add_collection(outline)
        collection = LineCollection(
            segments, cmap=TVK_CMAP, norm=normalization, linewidths=2.7,
            zorder=9, capstyle="round", joinstyle="round",
        )
        collection.set_array(segment_cost)
        axis.add_collection(collection)
    if bool(np.any(visible)):
        axis.scatter(
            pixels[visible, 0], pixels[visible, 1], c=costs[visible], cmap=TVK_CMAP,
            norm=normalization, s=14, edgecolor="white", linewidth=0.45, zorder=10,
        )
    _clean_camera_axis(axis, item["image"])
    _set_trajectory_roi(axis, item, calibration)


def _method_legend() -> list[Line2D]:
    return [
        Line2D([0], [0], color="#171A1F", lw=1.7, label="GT trajectory"),
        Line2D([0], [0], color=COLORS["FLOW"], lw=1.5, ls="--", label=DISPLAY_NAMES["FLOW"]),
        Line2D([0], [0], color=COLORS["VT"], lw=1.5, ls="-.", label=DISPLAY_NAMES["VT"]),
        Line2D([0], [0], color=COLORS["VTF"], lw=1.8, label=DISPLAY_NAMES["VTF"]),
    ]


def _render_scene(
    item: Mapping[str, Any],
    calibration: Mapping[str, Any],
    tvk_normalization: Normalize,
    output_base: Path,
) -> None:
    fig = plt.figure(figsize=(7.2, 4.35))
    grid = fig.add_gridspec(
        2, 3,
        left=0.055, right=0.985, bottom=0.14, top=0.86,
        width_ratios=(1.12, 1.12, 1.18), hspace=0.64, wspace=0.34,
    )
    raw_axis = fig.add_subplot(grid[:, :2])
    bev_axis = fig.add_subplot(grid[0, 2])
    tvk_axis = fig.add_subplot(grid[1, 2])
    fig.legend(
        handles=_method_legend(), loc="upper center", bbox_to_anchor=(0.53, 0.985),
        ncol=4, frameon=False, columnspacing=1.2, handlelength=2.0,
    )
    raw_axis.imshow(item["image"])
    _clean_camera_axis(raw_axis, item["image"])
    terrain_image = _draw_bev(bev_axis, item)
    bev_axis.set_xlabel("")
    _draw_tvk_colored_trajectory(tvk_axis, item, calibration, tvk_normalization)
    record = item["record"]
    vtf_visibility = item["projection_visibility"]["VTF"]
    raw_axis.set_title(
        f"a  Raw camera context\nsequence {record['sequence']}, frame {int(record['frame_id']):06d}",
        loc="left", fontweight="bold",
    )
    bev_axis.set_title("b  Planner BEV: static terrain cost C_T", loc="left", fontweight="bold")
    tvk_axis.set_title(
        "c  Pointwise TVK cost along VTF-Flow\n"
        f"camera-visible subset: {vtf_visibility['visible_count']}/"
        f"{vtf_visibility['total_count']} waypoints; full trajectory in b",
        loc="left", fontweight="bold",
    )
    terrain_cbar = fig.colorbar(
        terrain_image, ax=bev_axis, orientation="horizontal", fraction=0.050, pad=0.15,
    )
    terrain_cbar.set_label("Static terrain cost C_T (low → high)")
    tvk_scalar = mpl.cm.ScalarMappable(norm=tvk_normalization, cmap=TVK_CMAP)
    tvk_cbar = fig.colorbar(
        tvk_scalar, ax=tvk_axis, orientation="horizontal", fraction=0.050, pad=0.15,
    )
    tvk_cbar.set_label("Pointwise TVK objective (display clipped at pooled q95)")
    fig.text(
        0.012, 0.015,
        f"Scene role: {CATEGORY_NAMES[record['category']]}. Camera is contextual; trajectory projection does not model occlusion.",
        ha="left", va="bottom", fontsize=6.2, color="#4B5563",
    )
    _save_all(fig, output_base)


def _render_comparison(
    materials: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
    tvk_normalization: Normalize,
    output_base: Path,
) -> None:
    fig, axes = plt.subplots(
        len(materials), 3, squeeze=False,
        figsize=(7.2, 2.15 * len(materials) + 0.75),
        gridspec_kw={"width_ratios": (1.45, 1.0, 1.0)},
    )
    fig.subplots_adjust(left=0.045, right=0.995, bottom=0.16, top=0.86, hspace=0.52, wspace=0.24)
    fig.legend(
        handles=_method_legend(), loc="upper center", bbox_to_anchor=(0.52, 0.985),
        ncol=4, frameon=False, columnspacing=1.0, handlelength=1.8,
    )
    terrain_image = None
    for row_index, item in enumerate(materials):
        raw_axis, bev_axis, tvk_axis = axes[row_index]
        raw_axis.imshow(item["image"])
        _clean_camera_axis(raw_axis, item["image"])
        terrain_image = _draw_bev(bev_axis, item)
        bev_axis.set_ylabel("")
        _draw_tvk_colored_trajectory(tvk_axis, item, calibration, tvk_normalization)
        vtf_visibility = item["projection_visibility"]["VTF"]
        offset = 3 * row_index
        record = item["record"]
        raw_axis.set_title(
            f"{chr(97 + offset)}  Camera | frame {int(record['frame_id']):06d}\n"
            f"{CATEGORY_NAMES[record['category']].replace('Terrain-violation reduction', 'Terrain gain').replace('Balanced terrain–kinematic gain', 'Terrain–kinematic gain')}",
            loc="left", fontweight="bold",
        )
        bev_axis.set_title(f"{chr(98 + offset)}  BEV terrain cost C_T", loc="left", fontweight="bold")
        tvk_axis.set_title(
            f"{chr(99 + offset)}  Pointwise TVK on ours | "
            f"visible {vtf_visibility['visible_count']}/{vtf_visibility['total_count']}",
            loc="left", fontweight="bold",
        )
    assert terrain_image is not None
    terrain_axis = fig.add_axes([0.475, 0.050, 0.17, 0.018])
    terrain_cbar = fig.colorbar(terrain_image, cax=terrain_axis, orientation="horizontal")
    terrain_cbar.set_label("Static terrain cost C_T")
    tvk_axis_bar = fig.add_axes([0.795, 0.050, 0.17, 0.018])
    tvk_cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=tvk_normalization, cmap=TVK_CMAP),
        cax=tvk_axis_bar, orientation="horizontal",
    )
    tvk_cbar.set_label("Pointwise TVK objective")
    _save_all(fig, output_base)


def _build_materials(
    records: Sequence[Mapping[str, Any]],
    dataset: CombinedSceneDataset,
    trajectories: Mapping[str, np.ndarray],
    ground_truth: np.ndarray,
    scene_lookup: Mapping[str, int],
    terrain_config: TerrainFieldConfig,
    vehicle_config: VehicleConditionedFieldConfig,
    kinematic_config: TrajectoryKinematicConfig,
    planning_dt_s: float,
    calibration: Mapping[str, Any],
    data_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    materials: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for record in records:
        prediction_index = int(scene_lookup[str(record["scene_id"])])
        scene = dataset[int(record["dataset_index"])]
        terrain_map = scene.terrain_map.float()
        cost_map, extent = _terrain_cost_map(terrain_map, terrain_config, samples=160)
        image_path = _camera_image(
            data_root, str(record["sequence"]), int(record["frame_id"])
        )
        image = np.asarray(Image.open(image_path).convert("RGB"))
        per_method = {method: trajectories[method][prediction_index] for method in METHODS}
        gt = ground_truth[prediction_index]
        selected = {method: _best_candidate(per_method[method], gt) for method in METHODS}
        paths = {method: per_method[method][selected[method]] for method in METHODS}
        projection_visibility = {}
        for method, path in {**paths, "GT": gt}.items():
            _, visible, _ = _project(path, calibration)
            projection_visibility[method] = {
                "visible_mask": visible,
                "visible_count": int(np.sum(visible)),
                "total_count": int(visible.size),
                "visible_fraction": float(np.mean(visible)),
            }
        pointwise = {
            method: _pointwise_tvk_cost(
                path, terrain_map, terrain_config, vehicle_config, kinematic_config,
                planning_dt_s,
            )
            for method, path in {**paths, "GT": gt}.items()
        }
        for method, path in {**paths, "GT": gt}.items():
            values = pointwise[method]
            visible = projection_visibility[method]["visible_mask"]
            for step, xyz in enumerate(path):
                source_rows.append(
                    {
                        "scene_id": record["scene_id"],
                        "sequence": record["sequence"],
                        "frame_id": record["frame_id"],
                        "method": DISPLAY_NAMES.get(method, "GT trajectory"),
                        "selected_candidate": selected.get(method, -1),
                        "step": step,
                        "x_m": float(xyz[0]),
                        "y_m": float(xyz[1]),
                        "z_m": float(xyz[2]),
                        "terrain_vehicle_cost": float(values["terrain_vehicle_cost"][step]),
                        "kinematic_cost": float(values["kinematic_cost"][step]),
                        "tvk_cost": float(values["tvk_cost"][step]),
                        "speed_mps": float(values["speed_mps"][step]),
                        "curvature_per_m": float(values["curvature_per_m"][step]),
                        "lateral_acceleration_mps2": float(values["lateral_acceleration_mps2"][step]),
                        "camera_visible": bool(visible[step]),
                    }
                )
        materials.append(
            {
                "record": record,
                "image": image,
                "image_path": image_path,
                "terrain_map": terrain_map,
                "cost_map": cost_map,
                "extent": extent,
                "paths": paths,
                "ground_truth": gt,
                "selected": selected,
                "pointwise": pointwise,
                "projection_visibility": projection_visibility,
            }
        )
    return materials, source_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--sensor-transform", type=Path, default=DEFAULT_SENSOR_TRANSFORM)
    parser.add_argument("--camera-intrinsics", type=Path, default=DEFAULT_CAMERA_INTRINSICS)
    parser.add_argument("--extrinsic-variant", default=DEFAULT_EXTRINSIC_VARIANT)
    parser.add_argument(
        "--scene-id",
        action="append",
        default=None,
        help=(
            "Optional protocol scene ID, e.g. 00004:867:train. Repeat the option "
            "to render a multi-scene long/visible comparison."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _configure()
    benchmark_root = args.benchmark_root.resolve()
    data_root = args.data_root.resolve()
    protocol = _read_json(benchmark_root / "effective_protocol.json")
    selection = _read_json(
        benchmark_root / "figure_source_data" / "selected_advantage_scenes"
        / "selection_manifest.json"
    )
    metrics_path = (
        benchmark_root / "figure_source_data" / "selected_advantage_scenes"
        / "all_scene_selection_metrics.csv"
    )
    if args.scene_id is not None:
        records = [_record_for_scene(metrics_path, str(scene_id)) for scene_id in args.scene_id]
    else:
        records = [
            record for record in selection["selected_scenes"]
            if record["category"] in ("terrain", "balanced")
        ]
        records.sort(key=lambda row: ("terrain", "balanced").index(row["category"]))
    effective = _read_json(
        benchmark_root / "checkpoints" / "seed_0" / "flow_tvk" / "effective_config.json"
    )
    terrain_config = TerrainFieldConfig(**effective["terrain_field"])
    vehicle_config = VehicleConditionedFieldConfig(**effective["vehicle_conditioning"])
    kinematic_config = TrajectoryKinematicConfig(**protocol["kinematic"])
    planning_dt_s = float(protocol["trajectory"]["planning_dt_s"])
    dataset = CombinedSceneDataset(
        args.cache_root.resolve(), tuple(protocol["protocol"]["source_splits"])
    )
    trajectories, ground_truth, scene_ids = _load_predictions(benchmark_root)
    scene_lookup = {scene_id: index for index, scene_id in enumerate(scene_ids)}
    sequence = str(records[0]["sequence"])
    calibration = _load_calibration(
        data_root, sequence, args.sensor_transform.resolve(),
        args.camera_intrinsics.resolve(), args.extrinsic_variant,
    )
    materials, source_rows = _build_materials(
        records, dataset, trajectories, ground_truth, scene_lookup,
        terrain_config, vehicle_config, kinematic_config, planning_dt_s,
        calibration, data_root,
    )
    displayed_tvk = np.concatenate(
        [item["pointwise"]["VTF"]["tvk_cost"] for item in materials]
    )
    tvk_vmax = max(0.1, float(np.quantile(displayed_tvk, 0.95)))
    tvk_normalization = Normalize(vmin=0.0, vmax=tvk_vmax, clip=True)
    figures_dir = benchmark_root / "figures"
    bases = []
    for item in materials:
        category = str(item["record"]["category"])
        if args.scene_id is None:
            output_tag = category
        else:
            output_tag = (
                f"long_visible_{item['record']['sequence']}_"
                f"{int(item['record']['frame_id']):06d}"
            )
        base = figures_dir / f"figure_camera_bev_tvk_{output_tag}"
        _render_scene(item, calibration, tvk_normalization, base)
        bases.append(base)
    comparison_base = None
    if len(materials) > 1:
        comparison_name = (
            "figure_camera_bev_tvk_comparison"
            if args.scene_id is None
            else "figure_camera_bev_tvk_long_visible_comparison"
        )
        comparison_base = figures_dir / comparison_name
        _render_comparison(materials, calibration, tvk_normalization, comparison_base)

    source_tag = (
        "camera_bev_tvk_interpretation"
        if args.scene_id is None
        else "camera_bev_tvk_long_visible"
    )
    source_dir = benchmark_root / "figure_source_data" / source_tag
    source_dir.mkdir(parents=True, exist_ok=True)
    with (source_dir / "pointwise_tvk_cost.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)
    manifest = {
        "core_conclusion": (
            "The synchronized camera frame provides scene context, the planner-used static terrain cost "
            "is spatially interpretable in BEV, and complete TVK cost is candidate-dependent and displayed along trajectories."
        ),
        "static_field": "C_T(x,y) queried from the exact planner-used cached BEV",
        "complete_pointwise_objective": (
            "vehicle-conditioned terrain cost + weighted curvature and lateral-acceleration soft excess"
        ),
        "tvk_display_clip": {"rule": "pooled VTF-Flow q95", "value": tvk_vmax},
        "candidate_selection": "frozen seed-0 minimum-ADE candidate for qualitative diagnosis only",
        "trajectory_projection": "released calibration, positive-depth filter, no camera-space occlusion model",
        "camera_panel_scope": (
            "panel c shows only positive-depth in-image waypoints; panel b shows all H=10 waypoints"
        ),
        "camera_processing": "RGB conversion only; no brightness/contrast/gamma enhancement",
        "population_boundary": "quantitative conclusions use all 1909 protocol test scenes",
        "scenes": [
            {
                "scene_id": item["record"]["scene_id"],
                "category": item["record"]["category"],
                "projection_visibility": {
                    method: {
                        "visible_count": values["visible_count"],
                        "total_count": values["total_count"],
                        "visible_fraction": values["visible_fraction"],
                    }
                    for method, values in item["projection_visibility"].items()
                },
            }
            for item in materials
        ],
        "figures": [str(base.resolve()) for base in bases]
        + ([str(comparison_base.resolve())] if comparison_base is not None else []),
    }
    (source_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    caption = (
        "Cross-view interpretation of VTF-Flow feasibility guidance. The synchronized raw camera "
        "frame provides environmental context, while the exact planner-used static terrain cost "
        "C_T is shown in the undistorted ego-centric BEV. The complete candidate-dependent TVK "
        "objective is evaluated over all H=10 future waypoints. The BEV panel displays the complete "
        "5 s trajectory, whereas the calibrated camera ROI displays only positive-depth waypoints "
        "inside the image and reports the visible count. Trajectory projection does not model "
        "occlusion, and these examples complement "
        "rather than replace the aggregate evaluation over 1,909 test scenes."
    )
    caption_name = (
        "figure_camera_bev_tvk_caption.txt"
        if args.scene_id is None
        else "figure_camera_bev_tvk_long_visible_caption.txt"
    )
    (figures_dir / caption_name).write_text(
        caption + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

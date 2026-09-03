"""Render calibrated camera--BEV pairs for unified H=10 advantage scenes.

Camera overlays are geometric projections for qualitative environmental
context.  They do not model occlusion and are not used by the planner or by
the quantitative feasibility evaluation.
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
from PIL import Image
import pandas as pd
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
from TerraFlow.terrain.feasibility_field import TerrainFieldConfig  # noqa: E402


DEFAULT_DATA_ROOT = WORKSPACE_ROOT / "data" / "RELLIS3D"
DEFAULT_CACHE_ROOT = DEFAULT_DATA_ROOT / "trajectory_cache_h150_s5"
DEFAULT_BENCHMARK = TERRAFLOW_ROOT / "outputs" / "unified_h10_benchmark"

METHODS = ("FLOW", "VT", "VTF")
DISPLAY_NAMES = {
    "FLOW": "Flow baseline",
    "VT": "VTF-Flow w/o kinematic terms",
    "VTF": "VTF-Flow (ours)",
}
COLORS = {
    "FLOW": "#7C8796",
    "VT": "#3478A8",
    "VTF": "#C84D58",
}
CATEGORY_NAMES = {
    "terrain": "Terrain-violation reduction",
    "kinematic": "Kinematic-feasibility correction",
    "smoothness": "Trajectory-coherence improvement",
    "balanced": "Balanced terrain–kinematic gain",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.2,
            "axes.linewidth": 0.7,
            "legend.fontsize": 6.6,
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


def _load_scene_arrays(
    benchmark_root: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, int]]:
    trajectories, ground_truth, scene_ids = _load_predictions(benchmark_root)
    return trajectories, ground_truth, {
        scene_id: index for index, scene_id in enumerate(scene_ids)
    }


def _selected_for_scene(
    trajectories: Mapping[str, np.ndarray],
    ground_truth: np.ndarray,
    scene_index: int,
) -> dict[str, int]:
    return {
        method: _best_candidate(trajectories[method][scene_index], ground_truth[scene_index])
        for method in METHODS
    }


def _draw_camera(
    axis: plt.Axes,
    image: np.ndarray,
    trajectories: Mapping[str, np.ndarray],
    ground_truth: np.ndarray,
    selected: Mapping[str, int],
    calibration: Mapping[str, Any],
) -> dict[str, float]:
    axis.imshow(image)
    for candidate_index, trajectory in enumerate(trajectories["VTF"]):
        if candidate_index == selected["VTF"]:
            continue
        pixels, visible, _ = _project(trajectory, calibration)
        _draw_projected(
            axis, pixels, visible, color=COLORS["VTF"], linewidth=0.55,
            alpha=0.14, zorder=3,
        )
    visibility: dict[str, float] = {}
    for method, style, width in (
        ("FLOW", "--", 1.25),
        ("VT", "-.", 1.35),
        ("VTF", "-", 1.75),
    ):
        trajectory = trajectories[method][selected[method]]
        pixels, visible, _ = _project(trajectory, calibration)
        _draw_projected(
            axis, pixels, visible, color=COLORS[method], linewidth=width,
            linestyle=style, alpha=0.98, zorder=6,
        )
        visibility[method] = float(np.mean(visible))
    pixels, visible, _ = _project(ground_truth, calibration)
    _draw_projected(
        axis, pixels, visible, color="#171A1F", linewidth=1.55,
        linestyle="-", alpha=1.0, zorder=7,
    )
    visibility["GT"] = float(np.mean(visible))
    axis.set_xlim(0, image.shape[1])
    axis.set_ylim(image.shape[0], 0)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    return visibility


def _draw_bev(
    axis: plt.Axes,
    cost_map: np.ndarray,
    extent: tuple[float, float, float, float],
    trajectories: Mapping[str, np.ndarray],
    ground_truth: np.ndarray,
    selected: Mapping[str, int],
    record: Mapping[str, Any],
) -> mpl.image.AxesImage:
    image = axis.imshow(
        cost_map, origin="lower", extent=extent, cmap="YlOrBr",
        vmin=0.0, vmax=1.0, interpolation="bilinear", aspect="equal",
    )
    paths: list[np.ndarray] = [ground_truth]
    for method, style, width in (
        ("FLOW", "--", 1.15),
        ("VT", "-.", 1.25),
        ("VTF", "-", 1.65),
    ):
        trajectory = trajectories[method][selected[method]]
        paths.append(trajectory)
        line, = axis.plot(
            trajectory[:, 0], trajectory[:, 1], color=COLORS[method],
            linewidth=width, linestyle=style, zorder=6,
        )
        line.set_path_effects(
            [path_effects.Stroke(linewidth=width + 0.8, foreground="white"), path_effects.Normal()]
        )
    line, = axis.plot(
        ground_truth[:, 0], ground_truth[:, 1], color="#171A1F",
        linewidth=1.55, zorder=7,
    )
    line.set_path_effects(
        [path_effects.Stroke(linewidth=2.35, foreground="white"), path_effects.Normal()]
    )
    axis.scatter(
        0.0, 0.0, marker="*", s=42, color="#D62F2F", edgecolor="white",
        linewidth=0.6, zorder=9,
    )
    axis.scatter(
        ground_truth[-1, 0], ground_truth[-1, 1], marker="o", s=34,
        facecolor="white", edgecolor="#2155A6", linewidth=1.1, zorder=9,
    )
    xy = np.concatenate(
        [np.zeros((1, 2))] + [path[..., :2].reshape(-1, 2) for path in paths]
    )
    x_max = min(24.0, max(3.5, float(xy[:, 0].max()) + 0.55))
    y_min = max(-12.0, float(xy[:, 1].min()) - 0.55)
    y_max = min(12.0, float(xy[:, 1].max()) + 0.55)
    if y_max - y_min < 2.4:
        center = 0.5 * (y_min + y_max)
        y_min, y_max = center - 1.2, center + 1.2
    axis.set_xlim(0.0, x_max)
    axis.set_ylim(y_min, y_max)
    axis.set_xlabel("Ego-forward x (m)")
    axis.set_ylabel("Ego-left y (m)")
    axis.grid(color="white", linewidth=0.35, alpha=0.35)
    axis.text(
        0.02, 0.02,
        (
            f"TVK cost reduction {float(record['cross_seed_delta_tvk_vs_flow']):.3f}\n"
            f"terrain {100.0 * float(record['cross_seed_delta_terrain_violation_vs_flow']):.1f} pp; "
            f"curvature {100.0 * float(record['cross_seed_delta_curvature_violation_vs_flow']):.1f} pp"
        ),
        transform=axis.transAxes, ha="left", va="bottom", fontsize=5.8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.84, "edgecolor": "none"},
        zorder=10,
    )
    return image


def _legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color="#171A1F", lw=1.8, label="GT trajectory"),
        Line2D([0], [0], color=COLORS["FLOW"], lw=1.5, ls="--", label=DISPLAY_NAMES["FLOW"]),
        Line2D([0], [0], color=COLORS["VT"], lw=1.5, ls="-.", label=DISPLAY_NAMES["VT"]),
        Line2D([0], [0], color=COLORS["VTF"], lw=1.8, label=DISPLAY_NAMES["VTF"]),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#D62F2F", markeredgecolor="white", markersize=7, label="Ego origin"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#2155A6", markersize=5.5, label="5 s goal"),
    ]


def _write_projection_rows(
    output_path: Path,
    records: Sequence[Mapping[str, Any]],
    trajectories: Mapping[str, np.ndarray],
    ground_truth: np.ndarray,
    scene_lookup: Mapping[str, int],
    calibration: Mapping[str, Any],
    data_root: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for record in records:
        scene_id = str(record["scene_id"])
        index = int(scene_lookup[scene_id])
        selected = _selected_for_scene(trajectories, ground_truth, index)
        image_path = _camera_image(data_root, str(record["sequence"]), int(record["frame_id"]))
        collections = {
            **{method: trajectories[method][index] for method in METHODS},
            "GT": ground_truth[index][None],
        }
        for method, candidates in collections.items():
            for candidate_index, trajectory in enumerate(candidates):
                pixels, visible, depth = _project(trajectory, calibration)
                for step, point in enumerate(trajectory):
                    rows.append(
                        {
                            "scene_id": scene_id,
                            "image": str(image_path.resolve()),
                            "method": DISPLAY_NAMES.get(method, "GT trajectory"),
                            "candidate": candidate_index,
                            "is_minade_candidate": int(
                                method == "GT" or candidate_index == selected.get(method, -1)
                            ),
                            "step": step,
                            "ego_x_m": float(point[0]),
                            "ego_y_m": float(point[1]),
                            "ego_z_m": float(point[2]),
                            "pixel_u": float(pixels[step, 0]),
                            "pixel_v": float(pixels[step, 1]),
                            "camera_depth_m": float(depth[step]),
                            "visible_in_image": int(visible[step]),
                        }
                    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _scene_materials(
    records: Sequence[Mapping[str, Any]],
    dataset: CombinedSceneDataset,
    terrain_config: TerrainFieldConfig,
    trajectories: Mapping[str, np.ndarray],
    ground_truth: np.ndarray,
    scene_lookup: Mapping[str, int],
    calibration: Mapping[str, Any],
    data_root: Path,
) -> list[dict[str, Any]]:
    materials = []
    for record in records:
        index = int(scene_lookup[str(record["scene_id"])])
        image_path = _camera_image(data_root, str(record["sequence"]), int(record["frame_id"]))
        image = np.asarray(Image.open(image_path).convert("RGB"))
        scene = dataset[int(record["dataset_index"])]
        cost_map, extent = _terrain_cost_map(scene.terrain_map.float(), terrain_config)
        per_method = {method: trajectories[method][index] for method in METHODS}
        selected = _selected_for_scene(trajectories, ground_truth, index)
        visibility_axis = plt.figure().subplots()
        visibility = _draw_camera(
            visibility_axis, image, per_method, ground_truth[index], selected, calibration
        )
        plt.close(visibility_axis.figure)
        materials.append(
            {
                "record": record,
                "image": image,
                "cost_map": cost_map,
                "extent": extent,
                "trajectories": per_method,
                "ground_truth": ground_truth[index],
                "selected": selected,
                "visibility": visibility,
                "image_path": image_path,
            }
        )
    return materials


def render_hero(
    materials: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
    output_root: Path,
) -> Path:
    chosen = [
        next(item for item in materials if item["record"]["category"] == category)
        for category in ("terrain", "balanced")
    ]
    fig = plt.figure(figsize=(7.2, 4.75), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=(0.13, 1.0, 0.78))
    legend_axis = fig.add_subplot(grid[0, :])
    legend_axis.axis("off")
    legend_axis.legend(handles=_legend_handles(), loc="center", ncol=3, frameon=False)
    bev_images = []
    for column, item in enumerate(chosen):
        record = item["record"]
        camera_axis = fig.add_subplot(grid[1, column])
        _draw_camera(
            camera_axis, item["image"], item["trajectories"],
            item["ground_truth"], item["selected"], calibration,
        )
        camera_axis.set_title(
            f"{chr(97 + column)}  {CATEGORY_NAMES[record['category']]}\n"
            f"sequence {record['sequence']}, frame {int(record['frame_id']):06d}",
            loc="left", fontweight="bold",
        )
        bev_axis = fig.add_subplot(grid[2, column])
        bev_images.append(
            _draw_bev(
                bev_axis, item["cost_map"], item["extent"], item["trajectories"],
                item["ground_truth"], item["selected"], record,
            )
        )
        bev_axis.set_title(
            f"{chr(99 + column)}  Aligned planner terrain cost",
            loc="left", fontweight="bold",
        )
    colorbar = fig.colorbar(bev_images[0], ax=fig.axes[1:], fraction=0.024, pad=0.018, shrink=0.72)
    colorbar.set_label("Derived terrain cost (low to high)")
    base = output_root / "figures" / "figure_camera_bev_advantage_hero"
    _save_all(fig, base)
    return base


def render_full(
    materials: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
    output_root: Path,
) -> Path:
    fig = plt.figure(
        figsize=(7.2, 0.9 + 2.05 * len(materials)), layout="constrained"
    )
    grid = fig.add_gridspec(
        len(materials) + 1, 2,
        height_ratios=(0.12,) + (1.0,) * len(materials),
        width_ratios=(1.65, 1.0),
    )
    legend_axis = fig.add_subplot(grid[0, :])
    legend_axis.axis("off")
    legend_axis.legend(handles=_legend_handles(), loc="center", ncol=3, frameon=False)
    bev_images = []
    visibility_records = []
    for row_index, item in enumerate(materials, start=1):
        record = item["record"]
        camera_axis = fig.add_subplot(grid[row_index, 0])
        visibility = _draw_camera(
            camera_axis, item["image"], item["trajectories"],
            item["ground_truth"], item["selected"], calibration,
        )
        camera_axis.set_title(
            f"{chr(96 + 2 * row_index - 1)}  {CATEGORY_NAMES[record['category']]} | "
            f"frame {int(record['frame_id']):06d}",
            loc="left", fontweight="bold",
        )
        bev_axis = fig.add_subplot(grid[row_index, 1])
        bev_images.append(
            _draw_bev(
                bev_axis, item["cost_map"], item["extent"], item["trajectories"],
                item["ground_truth"], item["selected"], record,
            )
        )
        bev_axis.set_title(
            f"{chr(96 + 2 * row_index)}  Aligned terrain cost",
            loc="left", fontweight="bold",
        )
        visibility_records.append(
            {
                "scene_id": record["scene_id"],
                **{f"visible_fraction_{key}": value for key, value in visibility.items()},
            }
        )
    colorbar = fig.colorbar(bev_images[0], ax=fig.axes[1:], fraction=0.022, pad=0.018, shrink=0.80)
    colorbar.set_label("Derived terrain cost (low to high)")
    base = output_root / "figures" / "figure_camera_bev_advantage_full"
    _save_all(fig, base)
    pd.DataFrame(visibility_records).to_csv(
        output_root / "figure_source_data" / "selected_advantage_scenes"
        / "camera_projection_visibility.csv",
        index=False,
    )
    return base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--sensor-transform", type=Path, default=DEFAULT_SENSOR_TRANSFORM)
    parser.add_argument("--camera-intrinsics", type=Path, default=DEFAULT_CAMERA_INTRINSICS)
    parser.add_argument("--extrinsic-variant", default=DEFAULT_EXTRINSIC_VARIANT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _configure()
    benchmark_root = args.benchmark_root.resolve()
    data_root = args.data_root.resolve()
    selection = _read_json(
        benchmark_root / "figure_source_data" / "selected_advantage_scenes"
        / "selection_manifest.json"
    )
    records = list(selection["selected_scenes"])
    effective_protocol = _read_json(benchmark_root / "effective_protocol.json")
    effective_model = _read_json(
        benchmark_root / "checkpoints" / "seed_0" / "flow_tvk" / "effective_config.json"
    )
    terrain_config = TerrainFieldConfig(**effective_model["terrain_field"])
    dataset = CombinedSceneDataset(
        args.cache_root.resolve(),
        tuple(effective_protocol["protocol"]["source_splits"]),
    )
    trajectories, ground_truth, scene_lookup = _load_scene_arrays(benchmark_root)
    sequences = sorted({str(record["sequence"]) for record in records})
    if len(sequences) != 1:
        raise ValueError("the selected benchmark figure currently expects one test sequence")
    calibration = _load_calibration(
        data_root, sequences[0], args.sensor_transform.resolve(),
        args.camera_intrinsics.resolve(), args.extrinsic_variant,
    )
    materials = _scene_materials(
        records, dataset, terrain_config, trajectories, ground_truth,
        scene_lookup, calibration, data_root,
    )
    minimum_visible_fraction = 0.4
    camera_materials = [
        item for item in materials
        if min(float(value) for value in item["visibility"].values())
        >= minimum_visible_fraction
    ]
    if len(camera_materials) < 2:
        raise RuntimeError("fewer than two selected scenes pass camera visibility QA")
    included_scene_ids = {
        str(item["record"]["scene_id"]) for item in camera_materials
    }
    hero = render_hero(camera_materials, calibration, benchmark_root)
    full = render_full(camera_materials, calibration, benchmark_root)
    source_dir = benchmark_root / "figure_source_data" / "selected_advantage_scenes"
    _write_projection_rows(
        source_dir / "camera_projection_coordinates.csv", records,
        trajectories, ground_truth, scene_lookup, calibration, data_root,
    )
    manifest = {
        "purpose": "qualitative environmental context paired with the exact planner BEV",
        "camera_images": "released synchronized RELLIS-3D frames; RGB conversion only",
        "projection": {
            "geometric_only": True,
            "occlusion_modelled": False,
            "ground_display_offset": "calibrated LiDAR-to-base height",
            "candidate_emphasis": "offline minimum-ADE candidate for visualization only",
            "minimum_visible_waypoint_fraction": minimum_visible_fraction,
        },
        "calibration": calibration,
        "hero_figure": str(hero.resolve()),
        "full_figure": str(full.resolve()),
        "scene_visibility": [
            {
                "scene_id": item["record"]["scene_id"],
                "included_in_camera_composite": (
                    str(item["record"]["scene_id"]) in included_scene_ids
                ),
                "exclusion_reason": (
                    "" if str(item["record"]["scene_id"]) in included_scene_ids
                    else "at least one displayed trajectory has fewer than 40% visible projected waypoints"
                ),
                **item["visibility"],
            }
            for item in materials
        ],
    }
    (source_dir / "camera_bev_manifest.json").write_text(
        json.dumps(
            manifest, indent=2,
            default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value,
        ),
        encoding="utf-8",
    )
    caption = (
        "Calibrated camera context and aligned planner terrain cost for representative "
        "camera-visible held-out scenes. Curves are geometric projections of frozen seed-0 trajectories; "
        "the camera images are used only for environmental interpretation. Projection "
        "does not model occlusion, and best-of-K highlighting uses GT only for offline "
        "visualization. One BEV-selected scene was excluded from the camera composite "
        "because its projected waypoints were outside the image. Quantitative claims "
        "are based on all 1909 test scenes."
    )
    (benchmark_root / "figures" / "figure_camera_bev_advantage_caption.txt").write_text(
        caption + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "hero": str(hero), "full": str(full),
        "selected_scenes": len(records), "camera_visible_scenes": len(camera_materials),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

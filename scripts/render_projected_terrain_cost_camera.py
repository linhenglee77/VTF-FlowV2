"""Project the planner-used terrain-cost field into synchronized camera views.

The overlay is a calibrated 2.5D projection of the static spatial terrain
component C_T(x, y).  It is not the complete trajectory-dependent TVK
objective, and image-space occlusion is not modelled.
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
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image


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
COLORS = {"FLOW": "#7C8796", "VT": "#3478A8", "VTF": "#C84D58"}
CATEGORY_NAMES = {
    "terrain": "Terrain-violation reduction",
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


def _project_surface_points(
    points: np.ndarray,
    calibration: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project terrain points without applying the trajectory ground shift."""

    adjusted = np.asarray(points, dtype=np.float64).copy()
    adjusted[:, 2] += float(calibration["lidar_height_m"])
    return _project(adjusted, calibration)


def _projected_field(
    terrain_map: np.ndarray,
    cost_yx: np.ndarray,
    terrain_config: TerrainFieldConfig,
    calibration: Mapping[str, Any],
    image_shape: tuple[int, int, int],
) -> tuple[list[np.ndarray], np.ndarray, list[dict[str, Any]]]:
    """Build depth-sorted projected polygons from observed planner BEV cells."""

    _, nx, ny = terrain_map.shape
    if cost_yx.shape != (ny, nx):
        raise ValueError("cost map orientation does not match planner BEV")
    x_edges = np.linspace(0.0, terrain_config.forward_m, nx + 1)
    y_edges = np.linspace(-terrain_config.lateral_m, terrain_config.lateral_m, ny + 1)
    normalized_height = terrain_map[2]
    observed = normalized_height > 0.0
    image_height, image_width = image_shape[:2]
    records: list[tuple[float, np.ndarray, float, dict[str, Any]]] = []
    for ix in range(nx):
        for iy in range(ny):
            if not bool(observed[ix, iy]):
                continue
            # Exact inverse of the cache encoding:
            # height_norm = clip((mean_height + 2.5) / 4.5, 0, 1).
            height_m = float(normalized_height[ix, iy] * 4.5 - 2.5)
            corners = np.asarray(
                [
                    [x_edges[ix], y_edges[iy], height_m],
                    [x_edges[ix + 1], y_edges[iy], height_m],
                    [x_edges[ix + 1], y_edges[iy + 1], height_m],
                    [x_edges[ix], y_edges[iy + 1], height_m],
                ],
                dtype=np.float64,
            )
            pixels, _, depth = _project_surface_points(corners, calibration)
            center = pixels.mean(axis=0)
            span = np.ptp(pixels, axis=0)
            depth_valid = bool(np.all(depth > 0.5))
            near_image = bool(
                (-0.2 * image_width <= center[0] <= 1.2 * image_width)
                and (-0.2 * image_height <= center[1] <= 1.2 * image_height)
            )
            stable_polygon = bool(np.all(np.isfinite(pixels)) and np.max(span) < 420.0)
            projected = depth_valid and near_image and stable_polygon
            cost = float(cost_yx[iy, ix])
            row = {
                "grid_x_index": ix,
                "grid_y_index": iy,
                "center_x_m": float(0.5 * (x_edges[ix] + x_edges[ix + 1])),
                "center_y_m": float(0.5 * (y_edges[iy] + y_edges[iy + 1])),
                "mean_height_m": height_m,
                "terrain_cost": cost,
                "mean_camera_depth_m": float(np.mean(depth)),
                "projected": int(projected),
                "pixel_u_center": float(center[0]),
                "pixel_v_center": float(center[1]),
            }
            if projected:
                records.append((float(np.mean(depth)), pixels, cost, row))
            else:
                row["polygon_rank"] = -1
                records.append((float("nan"), np.empty((0, 2)), float("nan"), row))

    drawable = [item for item in records if item[3]["projected"] == 1 and item[1].size]
    drawable.sort(key=lambda item: item[0], reverse=True)
    polygons = [item[1] for item in drawable]
    costs = np.asarray([item[2] for item in drawable], dtype=np.float64)
    for rank, item in enumerate(drawable):
        item[3]["polygon_rank"] = rank
    unique_rows: dict[tuple[int, int], dict[str, Any]] = {}
    for item in records:
        row = item[3]
        unique_rows[(int(row["grid_x_index"]), int(row["grid_y_index"]))] = row
    return polygons, costs, list(unique_rows.values())


def _draw_field_overlay(
    axis: plt.Axes,
    image: np.ndarray,
    polygons: Sequence[np.ndarray],
    costs: np.ndarray,
    *,
    show_trajectories: bool,
    trajectories: Mapping[str, np.ndarray],
    ground_truth: np.ndarray,
    selected: Mapping[str, int],
    calibration: Mapping[str, Any],
) -> None:
    axis.imshow(image)
    cmap = mpl.colormaps["magma"]
    colors = cmap(np.clip(costs, 0.0, 1.0))
    # Preserve the photograph in low-cost regions while making costly terrain
    # visually salient.  Alpha only controls display; hue retains the cost value.
    colors[:, 3] = 0.02 + 0.58 * np.power(np.clip(costs, 0.0, 1.0), 1.35)
    collection = PolyCollection(
        polygons, facecolors=colors, edgecolors="none", antialiased=False,
        rasterized=True, zorder=2,
    )
    axis.add_collection(collection)
    if show_trajectories:
        for method, style, width in (
            ("FLOW", "--", 1.25),
            ("VT", "-.", 1.35),
            ("VTF", "-", 1.8),
        ):
            pixels, visible, _ = _project(
                trajectories[method][selected[method]], calibration
            )
            _draw_projected(
                axis, pixels, visible, color=COLORS[method], linewidth=width,
                linestyle=style, alpha=1.0, zorder=6,
            )
        pixels, visible, _ = _project(ground_truth, calibration)
        _draw_projected(
            axis, pixels, visible, color="#171A1F", linewidth=1.55,
            alpha=1.0, zorder=7,
        )
    axis.set_xlim(0, image.shape[1])
    axis.set_ylim(image.shape[0], 0)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color="#171A1F", lw=1.8, label="GT trajectory"),
        Line2D([0], [0], color=COLORS["FLOW"], lw=1.5, ls="--", label=DISPLAY_NAMES["FLOW"]),
        Line2D([0], [0], color=COLORS["VT"], lw=1.5, ls="-.", label=DISPLAY_NAMES["VT"]),
        Line2D([0], [0], color=COLORS["VTF"], lw=1.8, label=DISPLAY_NAMES["VTF"]),
    ]


def _render_grid(
    materials: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
    output_base: Path,
) -> None:
    rows = len(materials)
    fig, axes = plt.subplots(
        rows, 3, squeeze=False, figsize=(7.2, 0.55 + 1.82 * rows)
    )
    fig.subplots_adjust(
        left=0.008, right=0.995, bottom=0.17 if rows == 1 else 0.11,
        top=0.82 if rows == 1 else 0.88, hspace=0.48, wspace=0.075,
    )
    fig.legend(
        handles=_legend_handles(), loc="upper center", bbox_to_anchor=(0.5, 0.985),
        ncol=4, frameon=False, columnspacing=1.4, handlelength=2.0,
    )
    for row_index, item in enumerate(materials):
        record = item["record"]
        raw_axis = axes[row_index, 0]
        raw_axis.imshow(item["image"])
        raw_axis.set_xticks([])
        raw_axis.set_yticks([])
        for spine in raw_axis.spines.values():
            spine.set_visible(False)
        field_axis = axes[row_index, 1]
        _draw_field_overlay(
            field_axis, item["image"], item["polygons"], item["costs"],
            show_trajectories=False, trajectories=item["trajectories"],
            ground_truth=item["ground_truth"], selected=item["selected"],
            calibration=calibration,
        )
        trajectory_axis = axes[row_index, 2]
        _draw_field_overlay(
            trajectory_axis, item["image"], item["polygons"], item["costs"],
            show_trajectories=True, trajectories=item["trajectories"],
            ground_truth=item["ground_truth"], selected=item["selected"],
            calibration=calibration,
        )
        letter_offset = 3 * row_index
        raw_axis.set_title(
            f"{chr(97 + letter_offset)}  Camera context | frame {int(record['frame_id']):06d}\n"
            f"{CATEGORY_NAMES[record['category']]}",
            loc="left", fontweight="bold",
        )
        field_axis.set_title(
            f"{chr(98 + letter_offset)}  Projected static terrain cost C_T",
            loc="left", fontweight="bold",
        )
        trajectory_axis.set_title(
            f"{chr(99 + letter_offset)}  Field-guided trajectory outcome",
            loc="left", fontweight="bold",
        )
    scalar = mpl.cm.ScalarMappable(norm=Normalize(0.0, 1.0), cmap="magma")
    colorbar_axis = fig.add_axes(
        [0.385, 0.055 if rows == 1 else 0.025, 0.23, 0.025 if rows == 1 else 0.015]
    )
    colorbar = fig.colorbar(scalar, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("Projected static terrain cost C_T (low → high)")
    _save_all(fig, output_base)


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
    protocol = _read_json(benchmark_root / "effective_protocol.json")
    selection = _read_json(
        benchmark_root / "figure_source_data" / "selected_advantage_scenes"
        / "selection_manifest.json"
    )
    records = [
        record for record in selection["selected_scenes"]
        if record["category"] in CATEGORY_NAMES
    ]
    records.sort(key=lambda row: ("terrain", "balanced").index(row["category"]))
    model_config = _read_json(
        benchmark_root / "checkpoints" / "seed_0" / "flow_tvk" / "effective_config.json"
    )
    terrain_config = TerrainFieldConfig(**model_config["terrain_field"])
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
    materials: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    for record in records:
        index = int(scene_lookup[str(record["scene_id"])])
        scene = dataset[int(record["dataset_index"])]
        terrain = scene.terrain_map.float().numpy()
        cost_yx, _ = _terrain_cost_map(scene.terrain_map.float(), terrain_config, samples=64)
        image_path = _camera_image(
            data_root, str(record["sequence"]), int(record["frame_id"])
        )
        image = np.asarray(Image.open(image_path).convert("RGB"))
        polygons, costs, rows = _projected_field(
            terrain, cost_yx, terrain_config, calibration, image.shape
        )
        for row in rows:
            row.update(
                {
                    "scene_id": record["scene_id"],
                    "sequence": record["sequence"],
                    "frame_id": record["frame_id"],
                    "image": str(image_path.resolve()),
                }
            )
        cell_rows.extend(rows)
        per_method = {method: trajectories[method][index] for method in METHODS}
        selected = {
            method: _best_candidate(per_method[method], ground_truth[index])
            for method in METHODS
        }
        materials.append(
            {
                "record": record,
                "image": image,
                "polygons": polygons,
                "costs": costs,
                "trajectories": per_method,
                "ground_truth": ground_truth[index],
                "selected": selected,
                "projected_cells": len(polygons),
                "observed_cells": sum(int(row["mean_height_m"] > -2.5) for row in rows),
            }
        )
    hero_base = benchmark_root / "figures" / "figure_projected_terrain_cost_hero"
    comparison_base = benchmark_root / "figures" / "figure_projected_terrain_cost_comparison"
    _render_grid(materials[:1], calibration, hero_base)
    _render_grid(materials, calibration, comparison_base)

    source_dir = benchmark_root / "figure_source_data" / "projected_terrain_cost"
    source_dir.mkdir(parents=True, exist_ok=True)
    with (source_dir / "projected_cells.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cell_rows[0]))
        writer.writeheader()
        writer.writerows(cell_rows)
    manifest = {
        "field_displayed": "static planner terrain cost C_T(x,y)",
        "not_displayed_as_global_image": (
            "vehicle-conditioned and kinematic TVK additions; these depend on the candidate motion state"
        ),
        "height_decoding": "mean_height_m = normalized_mean_height * 4.5 - 2.5",
        "observed_cell_rule": "normalized mean height > 0",
        "projection": {
            "calibrated_2p5d": True,
            "occlusion_modelled": False,
            "depth_sorting": "far to near",
            "maximum_projected_cell_span_px": 420,
        },
        "camera_processing": "RGB conversion only; no crop-specific enhancement",
        "calibration": calibration,
        "scenes": [
            {
                "scene_id": item["record"]["scene_id"],
                "observed_cells": item["observed_cells"],
                "projected_cells": item["projected_cells"],
            }
            for item in materials
        ],
        "hero_figure": str(hero_base.resolve()),
        "comparison_figure": str(comparison_base.resolve()),
    }
    (source_dir / "projection_manifest.json").write_text(
        json.dumps(
            manifest, indent=2,
            default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value,
        ),
        encoding="utf-8",
    )
    caption = (
        "Camera-space visualization of the planner-used static terrain cost. Observed "
        "2.5D BEV cells are projected with released calibration and colored by C_T. "
        "The right panels overlay frozen best-of-K trajectories for offline interpretation. "
        "The complete TVK objective additionally contains candidate-dependent speed, heading, "
        "curvature, and lateral-acceleration terms and therefore is not a fixed image-space field. "
        "Occlusion is not modelled; quantitative evidence uses all 1909 test scenes."
    )
    (benchmark_root / "figures" / "figure_projected_terrain_cost_caption.txt").write_text(
        caption + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "hero": str(hero_base), "comparison": str(comparison_base),
        "scenes": manifest["scenes"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

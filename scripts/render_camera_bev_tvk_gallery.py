"""Render a temporally diverse camera–BEV–TVK scene gallery for selection.

Scenes are selected from the held-out test sequence using only temporal spacing,
camera/trajectory visibility, path length, and recorded-GT geometry. Planner
outcomes are not used for gallery selection.
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
import numpy as np
from PIL import Image


TERRAFLOW_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TERRAFLOW_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from TerraFlow.scripts.render_camera_bev_tvk_interpretation import (
    CATEGORY_NAMES,
    DEFAULT_BENCHMARK,
    DEFAULT_CACHE_ROOT,
    DEFAULT_CAMERA_INTRINSICS,
    DEFAULT_DATA_ROOT,
    DEFAULT_EXTRINSIC_VARIANT,
    DEFAULT_SENSOR_TRANSFORM,
    _best_candidate,
    _build_materials,
    _configure,
    _load_calibration,
    _load_predictions,
    _project,
    _read_json,
    _render_scene,
)
from TerraFlow.scripts.render_camera_trajectory_results import _camera_image
from TerraFlow.scripts.train_regression import CombinedSceneDataset
from TerraFlow.terrain.feasibility_field import TerrainFieldConfig
from TerraFlow.terrain.trajectory_kinematics import TrajectoryKinematicConfig
from TerraFlow.terrain.vehicle_conditioned_field import VehicleConditionedFieldConfig


CATEGORY_NAMES.update(
    {
        "gallery_straight": "Geometry gallery: near-straight",
        "gallery_left": "Geometry gallery: left bend",
        "gallery_right": "Geometry gallery: right bend",
    }
)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.0,
    }
)


def _geometry(path: np.ndarray) -> dict[str, float | str]:
    xy = np.asarray(path, dtype=np.float64)[:, :2]
    increments = np.diff(np.vstack((np.zeros((1, 2)), xy)), axis=0)
    lengths = np.linalg.norm(increments, axis=1)
    valid = lengths > 1e-6
    headings = np.unwrap(np.arctan2(increments[valid, 1], increments[valid, 0]))
    total_turn_deg = (
        float(np.degrees(np.abs(np.diff(headings)).sum())) if headings.size > 1 else 0.0
    )
    path_length_m = float(lengths.sum())
    end_lateral_m = float(xy[-1, 1])
    if abs(end_lateral_m) < 0.25 and total_turn_deg < 12.0:
        category = "gallery_straight"
        label = "near-straight"
    elif end_lateral_m >= 0.0:
        category = "gallery_left"
        label = "left bend"
    else:
        category = "gallery_right"
        label = "right bend"
    return {
        "path_length_m": path_length_m,
        "end_lateral_m": end_lateral_m,
        "total_turn_deg": total_turn_deg,
        "category": category,
        "geometry_label": label,
    }


def _temporally_spaced(
    candidates: Sequence[dict[str, Any]], count: int, minimum_separation: int
) -> list[dict[str, Any]]:
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} eligible scenes for requested count={count}")
    ordered = sorted(candidates, key=lambda row: int(row["frame_id"]))
    targets = np.linspace(
        float(ordered[0]["frame_id"]), float(ordered[-1]["frame_id"]), count
    )
    selected: list[dict[str, Any]] = []
    remaining = list(ordered)
    for target in targets:
        eligible = [
            row
            for row in remaining
            if all(
                abs(int(row["frame_id"]) - int(chosen["frame_id"]))
                >= minimum_separation
                for chosen in selected
            )
        ]
        if not eligible:
            eligible = remaining
        chosen = min(eligible, key=lambda row: abs(float(row["frame_id"]) - target))
        selected.append(chosen)
        remaining.remove(chosen)
    return sorted(selected, key=lambda row: int(row["frame_id"]))


def _select_records(
    metrics_path: Path,
    trajectories: Mapping[str, np.ndarray],
    ground_truth: np.ndarray,
    scene_lookup: Mapping[str, int],
    calibration: Mapping[str, Any],
    data_root: Path,
    count: int,
    minimum_visibility: float,
    minimum_path_length_m: float,
    minimum_frame_separation: int,
) -> list[dict[str, Any]]:
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    candidates: list[dict[str, Any]] = []
    for row in rows:
        scene_id = str(row["scene_id"])
        if scene_id not in scene_lookup:
            continue
        prediction_index = int(scene_lookup[scene_id])
        frame_id = int(float(row["FLOW_frame_id"]))
        sequence = str(int(float(row["FLOW_sequence"]))).zfill(5)
        gt = ground_truth[prediction_index]
        geometry = _geometry(gt)
        if float(geometry["path_length_m"]) < minimum_path_length_m:
            continue
        try:
            _camera_image(data_root, sequence, frame_id)
        except FileNotFoundError:
            continue
        _, gt_visible, _ = _project(gt, calibration)
        vtf_candidates = trajectories["VTF"][prediction_index]
        vtf_index = _best_candidate(vtf_candidates, gt)
        _, vtf_visible, _ = _project(vtf_candidates[vtf_index], calibration)
        visibility = float(min(np.mean(gt_visible), np.mean(vtf_visible)))
        if visibility < minimum_visibility:
            continue
        candidates.append(
            {
                "scene_id": scene_id,
                "sequence": sequence,
                "frame_id": frame_id,
                "dataset_index": int(float(row["FLOW_dataset_index"])),
                "category": geometry["category"],
                "geometry_label": geometry["geometry_label"],
                "path_length_m": geometry["path_length_m"],
                "end_lateral_m": geometry["end_lateral_m"],
                "total_turn_deg": geometry["total_turn_deg"],
                "minimum_projection_visibility": visibility,
            }
        )
    return _temporally_spaced(candidates, count, minimum_frame_separation)


def _contact_sheet(png_paths: Sequence[Path], records: Sequence[Mapping[str, Any]], path: Path) -> None:
    columns = 3
    rows = int(np.ceil(len(png_paths) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(13.5, 3.0 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for index, (png_path, record) in enumerate(zip(png_paths, records)):
        axis = axes.flat[index]
        axis.imshow(np.asarray(Image.open(png_path).convert("RGB")))
        axis.set_title(
            f"{index + 1:02d} | frame {int(record['frame_id']):06d} | {record['geometry_label']} | "
            f"y_H={float(record['end_lateral_m']):+.2f} m",
            loc="left", fontsize=8.0, fontweight="bold",
        )
        axis.axis("off")
    fig.suptitle(
        "Camera–BEV–TVK gallery | held-out sequence 00004 | geometry/time selection",
        fontsize=11.0, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.98), h_pad=1.0, w_pad=0.45)
    fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        path.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--sensor-transform", type=Path, default=DEFAULT_SENSOR_TRANSFORM)
    parser.add_argument("--camera-intrinsics", type=Path, default=DEFAULT_CAMERA_INTRINSICS)
    parser.add_argument("--extrinsic-variant", default=DEFAULT_EXTRINSIC_VARIANT)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--minimum-visibility", type=float, default=0.70)
    parser.add_argument("--minimum-path-length-m", type=float, default=2.0)
    parser.add_argument("--minimum-frame-separation", type=int, default=90)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _configure()
    benchmark_root = args.benchmark_root.resolve()
    data_root = args.data_root.resolve()
    protocol = _read_json(benchmark_root / "effective_protocol.json")
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
    calibration = _load_calibration(
        data_root, "00004", args.sensor_transform.resolve(),
        args.camera_intrinsics.resolve(), args.extrinsic_variant,
    )
    metrics_path = (
        benchmark_root / "figure_source_data" / "selected_advantage_scenes"
        / "all_scene_selection_metrics.csv"
    )
    records = _select_records(
        metrics_path, trajectories, ground_truth, scene_lookup, calibration, data_root,
        args.count, args.minimum_visibility, args.minimum_path_length_m,
        args.minimum_frame_separation,
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
    normalization = mpl.colors.Normalize(vmin=0.0, vmax=tvk_vmax, clip=True)
    output_dir = benchmark_root / "figures" / "camera_bev_tvk_gallery"
    output_dir.mkdir(parents=True, exist_ok=True)
    png_paths: list[Path] = []
    for item in materials:
        frame_id = int(item["record"]["frame_id"])
        base = output_dir / f"camera_bev_tvk_00004_{frame_id:06d}"
        _render_scene(item, calibration, normalization, base)
        png_paths.append(base.with_suffix(".png"))
    _contact_sheet(png_paths, records, output_dir / "gallery_contact_sheet.png")
    with (output_dir / "gallery_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    source_dir = benchmark_root / "figure_source_data" / "camera_bev_tvk_gallery"
    source_dir.mkdir(parents=True, exist_ok=True)
    with (source_dir / "pointwise_tvk_cost.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)
    manifest = {
        "selection_population": len(scene_ids),
        "selection_basis": (
            "held-out sequence 00004; temporal spacing, camera/trajectory visibility, "
            "minimum path length, and recorded-GT geometry only; no planner outcome used"
        ),
        "count": len(records),
        "minimum_visibility": args.minimum_visibility,
        "minimum_path_length_m": args.minimum_path_length_m,
        "minimum_frame_separation": args.minimum_frame_separation,
        "tvk_display_clip": {"rule": "gallery VTF q95", "value": tvk_vmax},
        "records": records,
    }
    (output_dir / "gallery_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

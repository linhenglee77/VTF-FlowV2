"""Render calibrated camera-view and camera--BEV VTF-Flow result figures.

The overlays are geometric projections for qualitative context. They do not
model image-space occlusion and are not used to compute feasibility metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import yaml


TERRAFLOW_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TERRAFLOW_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


DEFAULT_DATA_ROOT = WORKSPACE_ROOT / "data" / "RELLIS3D"
DEFAULT_OUTPUT_ROOT = TERRAFLOW_ROOT / "outputs" / "final_experiments_tvk_final"
DEFAULT_STYLE = TERRAFLOW_ROOT / "configs" / "final_figure_style.json"
DEFAULT_SENSOR_TRANSFORM = TERRAFLOW_ROOT / "configs" / "rellis3d_os1_to_planning_ego.json"
DEFAULT_CAMERA_INTRINSICS = (
    DEFAULT_DATA_ROOT / "processed" / "calibration_variants"
    / "Copy of calibrationdata_pylon_iris5.6" / "calibrationdata_pylon_iris5.6"
    / "pylon_iris56.yaml"
)
DEFAULT_EXTRINSIC_VARIANT = (
    "Copy of Rellis_3D_cam2lidar_20210224"
)

FLOW_COLOR = "#2C7FB8"
VTF_COLOR = "#E67E22"
GT_COLOR = "#111827"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _configure(style: Mapping[str, Any]) -> None:
    size = float(style["font_size_pt"])
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": size,
        "axes.labelsize": size,
        "axes.titlesize": size + 0.5,
        "legend.fontsize": max(5.5, size - 0.5),
        "axes.linewidth": 0.7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def _save(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def _transform_matrix(record: Mapping[str, Any]) -> np.ndarray:
    quaternion = record["q"]
    rotation = Rotation.from_quat([
        float(quaternion["x"]), float(quaternion["y"]),
        float(quaternion["z"]), float(quaternion["w"]),
    ]).as_matrix()
    translation = record["t"]
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = [
        float(translation["x"]), float(translation["y"]),
        float(translation["z"]),
    ]
    return matrix


def _load_calibration(
    data_root: Path,
    sequence: str,
    sensor_transform_path: Path,
    camera_intrinsics_path: Path,
    extrinsic_variant: str,
) -> dict[str, Any]:
    sensor_definition = _read_json(sensor_transform_path)
    ego_from_lidar = np.asarray(sensor_definition["T_ego_sensor"], dtype=np.float64)
    lidar_from_ego = np.linalg.inv(ego_from_lidar)
    full = np.asarray(
        sensor_definition["full_T_base_link_os1_lidar_from_calibration_bag"],
        dtype=np.float64,
    )
    lidar_height_m = float(full[2, 3])

    extrinsic_path = (
        data_root / "processed" / "calibration_variants" / extrinsic_variant
        / "Rellis_3D" / sequence / "transforms.yaml"
    )
    extrinsic_yaml = yaml.safe_load(extrinsic_path.read_text(encoding="utf-8"))
    key = "os1_cloud_node-pylon_camera_node"
    lidar_from_camera = _transform_matrix(extrinsic_yaml[key])
    camera_from_lidar = np.linalg.inv(lidar_from_camera)

    # Direction audit. A planning-ego forward point must lie in front of the
    # camera after ego -> LiDAR -> camera transformation.
    forward_ego = np.array([10.0, 0.0, 0.0, 1.0])
    forward_camera = camera_from_lidar @ lidar_from_ego @ forward_ego
    if forward_camera[2] <= 0.0:
        raise ValueError(
            "released camera-LiDAR transform direction failed the forward-axis audit"
        )

    camera_yaml = yaml.safe_load(camera_intrinsics_path.read_text(encoding="utf-8"))
    intrinsic = np.asarray(camera_yaml["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(
        camera_yaml["distortion_coefficients"]["data"], dtype=np.float64
    )
    return {
        "camera_from_ego": camera_from_lidar @ lidar_from_ego,
        "intrinsic": intrinsic,
        "distortion": distortion,
        "image_width": int(camera_yaml["image_width"]),
        "image_height": int(camera_yaml["image_height"]),
        "lidar_height_m": lidar_height_m,
        "extrinsic_path": str(extrinsic_path.resolve()),
        "intrinsic_path": str(camera_intrinsics_path.resolve()),
        "forward_axis_camera": forward_camera[:3].tolist(),
        "direction_interpretation": (
            "YAML stores T_lidar_camera; inverse used for LiDAR-to-camera projection"
        ),
    }


def _project(
    trajectory: np.ndarray,
    calibration: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(trajectory, dtype=np.float64).copy()
    # Trajectories follow the future LiDAR origin. Shift by the calibrated
    # LiDAR-to-base height only for display on the visible terrain surface.
    points[:, 2] -= float(calibration["lidar_height_m"])
    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    camera = (np.asarray(calibration["camera_from_ego"]) @ homogeneous.T).T[:, :3]
    valid_depth = camera[:, 2] > 0.5
    normalized = camera[:, :2] / np.maximum(camera[:, 2:3], 1e-8)
    x = normalized[:, 0]
    y = normalized[:, 1]
    k1, k2, p1, p2, k3 = np.pad(
        np.asarray(calibration["distortion"], dtype=np.float64),
        (0, max(0, 5 - len(calibration["distortion"]))),
    )[:5]
    radius2 = x * x + y * y
    radial = 1.0 + k1 * radius2 + k2 * radius2**2 + k3 * radius2**3
    distorted_x = x * radial + 2.0 * p1 * x * y + p2 * (radius2 + 2.0 * x * x)
    distorted_y = y * radial + p1 * (radius2 + 2.0 * y * y) + 2.0 * p2 * x * y
    intrinsic = np.asarray(calibration["intrinsic"])
    pixels = np.stack((
        intrinsic[0, 0] * distorted_x + intrinsic[0, 2],
        intrinsic[1, 1] * distorted_y + intrinsic[1, 2],
    ), axis=1)
    visible = (
        valid_depth
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < float(calibration["image_width"]))
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < float(calibration["image_height"]))
    )
    return pixels, visible, camera[:, 2]


def _load_predictions(
    output_root: Path, scene_id: str
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, int]]:
    paths = {
        "Flow": output_root / "main_primary_seed0_A" / "predictions.npz",
        "VTF-Flow": output_root / "main_primary_seed0_VTF" / "predictions.npz",
    }
    trajectories: dict[str, np.ndarray] = {}
    selected: dict[str, int] = {}
    ground_truth: np.ndarray | None = None
    for method, path in paths.items():
        with np.load(path, allow_pickle=False) as archive:
            lookup = {str(value): index for index, value in enumerate(archive["scene_ids"])}
            position = lookup[scene_id]
            current_gt = archive["ground_truth"][position].copy()
            candidates = archive["trajectories"][position].copy()
        if ground_truth is None:
            ground_truth = current_gt
        elif not np.allclose(ground_truth, current_gt, atol=1e-6):
            raise ValueError("prediction archives disagree on ground truth")
        ade = np.linalg.norm(candidates - current_gt[None], axis=-1).mean(axis=-1)
        selected[method] = int(np.argmin(ade))
        trajectories[method] = candidates
    assert ground_truth is not None
    return trajectories, ground_truth, selected


def _camera_image(data_root: Path, sequence: str, frame: int) -> Path:
    directory = data_root / "processed" / "Rellis-3D" / sequence / "pylon_camera_node"
    matches = sorted(directory.glob(f"frame{frame:06d}-*.jpg"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one synchronized camera frame for {sequence}/{frame:06d}, got {len(matches)}"
        )
    return matches[0]


def _draw_projected(
    axis: plt.Axes,
    pixels: np.ndarray,
    visible: np.ndarray,
    *,
    color: str,
    linewidth: float,
    linestyle: str = "-",
    alpha: float = 1.0,
    label: str | None = None,
    zorder: int = 5,
) -> None:
    labelled = False
    visible_indices = np.flatnonzero(visible)
    if len(visible_indices) < 2:
        return
    runs = np.split(visible_indices, np.flatnonzero(np.diff(visible_indices) > 1) + 1)
    for run in runs:
        if len(run) < 2:
            continue
        line, = axis.plot(
            pixels[run, 0], pixels[run, 1],
            color=color, lw=linewidth, ls=linestyle, alpha=alpha,
            label=label if not labelled else None, zorder=zorder,
            solid_capstyle="round",
        )
        line.set_path_effects([
            path_effects.Stroke(
                linewidth=linewidth + 1.1, foreground="white", alpha=0.8 * alpha
            ),
            path_effects.Normal(),
        ])
        labelled = True


def _draw_camera_panel(
    axis: plt.Axes,
    image: np.ndarray,
    trajectories: Mapping[str, np.ndarray],
    ground_truth: np.ndarray,
    selected: Mapping[str, int],
    calibration: Mapping[str, Any],
    *,
    show_other_vtf: bool,
) -> dict[str, float]:
    axis.imshow(image)
    visibility: dict[str, float] = {}
    if show_other_vtf:
        for candidate_index, trajectory in enumerate(trajectories["VTF-Flow"]):
            if candidate_index == selected["VTF-Flow"]:
                continue
            pixels, visible, _ = _project(trajectory, calibration)
            _draw_projected(
                axis, pixels, visible, color=VTF_COLOR, linewidth=0.75,
                alpha=0.20, zorder=3,
            )
    for name, trajectory, color, width, style in (
        ("Flow (best-of-K)", trajectories["Flow"][selected["Flow"]], FLOW_COLOR, 1.4, "--"),
        ("VTF-Flow (best-of-K)", trajectories["VTF-Flow"][selected["VTF-Flow"]], VTF_COLOR, 1.8, "-"),
        ("GT", ground_truth, GT_COLOR, 1.5, "-"),
    ):
        pixels, visible, _ = _project(trajectory, calibration)
        _draw_projected(
            axis, pixels, visible, color=color, linewidth=width,
            linestyle=style, label=name, zorder=6,
        )
        visibility[name] = float(np.mean(visible))
    axis.set_xlim(0, image.shape[1])
    axis.set_ylim(image.shape[0], 0)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    return visibility


def _draw_bev_panel(
    axis: plt.Axes,
    field_path: Path,
    trajectories: Mapping[str, np.ndarray],
    ground_truth: np.ndarray,
    selected: Mapping[str, int],
) -> None:
    with np.load(field_path, allow_pickle=False) as archive:
        feasibility = archive["feasibility"]
        extent = (
            float(archive["x_min_m"]), float(archive["x_max_m"]),
            float(archive["y_min_m"]), float(archive["y_max_m"]),
        )
    axis.imshow(
        feasibility, origin="lower", extent=extent, cmap="viridis",
        vmin=0.0, vmax=1.0, interpolation="nearest", aspect="equal", rasterized=True,
    )
    for trajectory, color, width, style, label in (
        (trajectories["Flow"][selected["Flow"]], FLOW_COLOR, 1.0, "--", "Flow"),
        (trajectories["VTF-Flow"][selected["VTF-Flow"]], VTF_COLOR, 1.4, "-", "VTF-Flow"),
        (ground_truth, GT_COLOR, 1.0, "-", "GT"),
    ):
        points = np.concatenate((np.zeros((1, 3)), trajectory), axis=0)
        line, = axis.plot(
            points[:, 0], points[:, 1], color=color, lw=width, ls=style,
            label=label, solid_capstyle="round",
        )
        line.set_path_effects([
            path_effects.Stroke(linewidth=width + 0.9, foreground="white"),
            path_effects.Normal(),
        ])
    axis.scatter(
        [0.0], [0.0], marker="*", s=22, facecolor="#D62728",
        edgecolor="white", linewidth=0.5, zorder=8,
    )
    axis.set_xlim(extent[0], extent[1])
    axis.set_ylim(extent[2], extent[3])
    axis.set_xlabel("ego-forward $x$ (m)")
    axis.set_ylabel("ego-left $y$ (m)")


def _write_projection_source(
    output_root: Path,
    scene_id: str,
    trajectories: Mapping[str, np.ndarray],
    ground_truth: np.ndarray,
    selected: Mapping[str, int],
    calibration: Mapping[str, Any],
    image_path: Path,
) -> Path:
    directory = output_root / "figure_source_data" / "camera_trajectory_overlays"
    directory.mkdir(parents=True, exist_ok=True)
    sequence, frame_text, _ = scene_id.split(":")
    path = directory / f"camera_projection_{sequence}_{int(frame_text):06d}.csv"
    rows: list[dict[str, Any]] = []
    collections = {
        "Flow": trajectories["Flow"],
        "VTF-Flow": trajectories["VTF-Flow"],
        "GT": ground_truth[None],
    }
    for method, candidates in collections.items():
        for candidate, trajectory in enumerate(candidates):
            pixels, visible, depth = _project(trajectory, calibration)
            for waypoint, (point, pixel) in enumerate(zip(trajectory, pixels)):
                rows.append({
                    "scene_id": scene_id,
                    "image": str(image_path.resolve()),
                    "method": method,
                    "candidate": candidate,
                    "selected_best_of_k": int(
                        method == "GT" or candidate == selected.get(method, -1)
                    ),
                    "waypoint": waypoint,
                    "ego_x_m": float(point[0]),
                    "ego_y_m": float(point[1]),
                    "ego_z_m": float(point[2]),
                    "pixel_u": float(pixel[0]),
                    "pixel_v": float(pixel[1]),
                    "camera_depth_m": float(depth[waypoint]),
                    "visible_in_image": int(visible[waypoint]),
                })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def render_scene(
    record: Mapping[str, Any],
    data_root: Path,
    output_root: Path,
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    sequence = str(record["sequence"])
    frame = int(record["frame"])
    scene_id = str(record["scene_id"])
    image_path = _camera_image(data_root, sequence, frame)
    image = np.asarray(Image.open(image_path).convert("RGB"))
    trajectories, ground_truth, selected = _load_predictions(output_root, scene_id)
    fig, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    visibility = _draw_camera_panel(
        axis, image, trajectories, ground_truth, selected, calibration,
        show_other_vtf=True,
    )
    axis.set_title(
        f"Camera context | RELLIS-3D {sequence}, frame {frame:06d} | {record['category']}",
        loc="left", fontweight="bold",
    )
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, loc="upper left", ncol=3)
    fig.text(
        0.5, 0.012,
        "Geometric projection using released calibration. Other VTF-Flow candidates are faint. "
        "Best-of-K is selected by GT only for offline visualisation; occlusion is not modelled.",
        ha="center", va="bottom", fontsize=5.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
    )
    base = (
        output_root / "figures" / "camera_trajectory_scenes"
        / f"figure_camera_trajectory_{sequence}_{frame:06d}"
    )
    _save(fig, base)
    source = _write_projection_source(
        output_root, scene_id, trajectories, ground_truth, selected,
        calibration, image_path,
    )
    return {
        "category": record["category"],
        "scene_id": scene_id,
        "image": str(image_path.resolve()),
        "figure_base": str(base.resolve()),
        "source_data": str(source.resolve()),
        "visibility": visibility,
        "publication_candidate": bool(
            visibility and min(visibility.values()) >= 0.75
        ),
        "selected_candidates": selected,
    }


def render_triptych(
    records: list[Mapping[str, Any]],
    data_root: Path,
    output_root: Path,
    calibration_by_sequence: Mapping[str, Mapping[str, Any]],
) -> Path:
    # These three categories keep the projected trajectory inside the released
    # camera field of view while retaining distinct path geometry.
    preferred = ("typical_straight", "long_path", "right_turn")
    selected_records = [
        next(record for record in records if record["category"] == category)
        for category in preferred
    ]
    fig, axes = plt.subplots(
        2, 3, figsize=(7.2, 5.0),
        gridspec_kw={"height_ratios": (1.0, 0.78)}, constrained_layout=True,
    )
    for column, record in enumerate(selected_records):
        sequence = str(record["sequence"])
        frame = int(record["frame"])
        image = np.asarray(
            Image.open(_camera_image(data_root, sequence, frame)).convert("RGB")
        )
        trajectories, ground_truth, selected = _load_predictions(
            output_root, str(record["scene_id"])
        )
        _draw_camera_panel(
            axes[0, column], image, trajectories, ground_truth, selected,
            calibration_by_sequence[sequence], show_other_vtf=False,
        )
        axes[0, column].set_title(
            f"{chr(97 + column)}  {record['category'].replace('_', ' ')}\n"
            f"sequence {sequence}, frame {frame:06d}",
            loc="left", fontweight="bold",
        )
        _draw_bev_panel(
            axes[1, column], Path(str(record["field_archive"])),
            trajectories, ground_truth, selected,
        )
        axes[1, column].set_title(
            f"{chr(100 + column)}  Relative terrain feasibility",
            loc="left", fontweight="bold",
        )
        if column > 0:
            axes[1, column].set_ylabel("")
            axes[1, column].set_yticklabels([])
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01))
    fig.text(
        0.5, -0.005,
        "Top: calibrated camera context. Bottom: aligned BEV feasibility. "
        "Lines show offline best-of-K trajectories; projections do not model occlusion.",
        ha="center", va="bottom", fontsize=5.5,
    )
    base = output_root / "figures" / "figure_camera_bev_qualitative"
    _save(fig, base)
    return base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--style", type=Path, default=DEFAULT_STYLE)
    parser.add_argument("--sensor-transform", type=Path, default=DEFAULT_SENSOR_TRANSFORM)
    parser.add_argument("--camera-intrinsics", type=Path, default=DEFAULT_CAMERA_INTRINSICS)
    parser.add_argument("--extrinsic-variant", default=DEFAULT_EXTRINSIC_VARIANT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    _configure(_read_json(args.style.resolve()))
    manifest_path = output_root / "raw_terrain_scene_set_manifest.json"
    manifest = _read_json(manifest_path)
    records = list(manifest["scenes"])
    sequences = sorted({str(record["sequence"]) for record in records})
    calibrations = {
        sequence: _load_calibration(
            data_root, sequence, args.sensor_transform.resolve(),
            args.camera_intrinsics.resolve(), args.extrinsic_variant,
        )
        for sequence in sequences
    }
    rendered = [
        render_scene(
            record, data_root, output_root, calibrations[str(record["sequence"])]
        )
        for record in records
    ]
    triptych = render_triptych(records, data_root, output_root, calibrations)
    output_manifest = {
        "status": "complete",
        "purpose": "qualitative environmental context; not quantitative feasibility evidence",
        "projection": {
            "ground_display_offset": "calibrated LiDAR-to-base height",
            "occlusion_modelled": False,
            "candidate_selection": "offline best-of-K by GT for visualisation only",
            "calibration": calibrations,
        },
        "scenes": rendered,
        "triptych": str(triptych.resolve()),
    }
    path = output_root / "camera_trajectory_result_manifest.json"
    path.write_text(
        json.dumps(
            output_manifest,
            indent=2,
            default=lambda value: value.tolist()
            if isinstance(value, np.ndarray)
            else float(value)
            if isinstance(value, np.floating)
            else int(value)
            if isinstance(value, np.integer)
            else str(value),
        ),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "complete", "manifest": str(path.resolve()),
        "triptych": str(triptych.resolve()), "scene_count": len(rendered),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

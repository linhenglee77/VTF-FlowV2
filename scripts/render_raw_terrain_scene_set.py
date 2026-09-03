"""Render a deterministic set of diverse raw-terrain trajectory figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


TERRAFLOW_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TERRAFLOW_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from TerraFlow.scripts.build_terrain_field import build_archive  # noqa: E402
from TerraFlow.visualization.plot_raw_terrain_trajectory import (  # noqa: E402
    _configure,
    _read_json,
    render,
)


DEFAULT_OUTPUT_ROOT = TERRAFLOW_ROOT / "outputs" / "final_experiments_tvk_final"
DEFAULT_FIELD_CONFIG = TERRAFLOW_ROOT / "configs" / "rellis3d_terrain_field.json"
DEFAULT_STYLE = TERRAFLOW_ROOT / "configs" / "final_figure_style.json"
DEFAULT_SENSOR_TRANSFORM = TERRAFLOW_ROOT / "configs" / "rellis3d_os1_to_planning_ego.json"
DEFAULT_TVK_CONFIG = TERRAFLOW_ROOT / "configs" / "final_tvk_validation.json"


def _trajectory_geometry(ground_truth: np.ndarray) -> dict[str, np.ndarray]:
    origins = np.zeros((len(ground_truth), 1, 3), dtype=ground_truth.dtype)
    segments = np.diff(np.concatenate((origins, ground_truth), axis=1), axis=1)
    return {
        "path_length_m": np.linalg.norm(segments, axis=-1).sum(axis=-1),
        "curvature_proxy": np.linalg.norm(
            np.diff(ground_truth[:, :, :2], n=2, axis=1), axis=-1
        ).sum(axis=-1),
        "final_x_m": ground_truth[:, -1, 0],
        "final_y_m": ground_truth[:, -1, 1],
        "maximum_abs_y_m": np.abs(ground_truth[:, :, 1]).max(axis=-1),
        "minimum_x_m": ground_truth[:, :, 0].min(axis=-1),
        "maximum_x_m": ground_truth[:, :, 0].max(axis=-1),
    }


def select_diverse_scenes(
    prediction_path: Path,
    minimum_frame_separation: int = 100,
) -> list[dict[str, Any]]:
    """Select six scenes using only GT geometry, never terrain outcomes."""

    with np.load(prediction_path, allow_pickle=False) as archive:
        ground_truth = archive["ground_truth"].copy()
        scene_ids = np.asarray([str(value) for value in archive["scene_ids"]])
    geometry = _trajectory_geometry(ground_truth)
    valid = (
        (geometry["minimum_x_m"] >= 0.0)
        & (geometry["maximum_x_m"] <= 23.5)
        & (geometry["maximum_abs_y_m"] <= 11.5)
    )
    median_length = float(np.median(geometry["path_length_m"][valid]))
    scores = (
        ("long_path", geometry["path_length_m"]),
        ("left_turn", geometry["final_y_m"]),
        ("right_turn", -geometry["final_y_m"]),
        ("high_curvature", geometry["curvature_proxy"]),
        (
            "typical_straight",
            -(
                np.abs(geometry["path_length_m"] - median_length)
                + 2.0 * np.abs(geometry["final_y_m"])
            ),
        ),
        ("short_path", -geometry["path_length_m"]),
    )
    selected: list[dict[str, Any]] = []
    for category, score in scores:
        for index in np.argsort(score)[::-1]:
            scene_id = str(scene_ids[index])
            sequence, frame_text, _ = scene_id.split(":")
            frame = int(frame_text)
            if not bool(valid[index]):
                continue
            if any(
                sequence == row["sequence"]
                and abs(frame - int(row["frame"])) < minimum_frame_separation
                for row in selected
            ):
                continue
            selected.append(
                {
                    "category": category,
                    "scene_id": scene_id,
                    "sequence": sequence,
                    "frame": frame,
                    "path_length_m": float(geometry["path_length_m"][index]),
                    "final_x_m": float(geometry["final_x_m"][index]),
                    "final_y_m": float(geometry["final_y_m"][index]),
                    "curvature_proxy": float(geometry["curvature_proxy"][index]),
                }
            )
            break
    if len(selected) != len(scores):
        raise RuntimeError("could not select one valid, separated scene per geometry category")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--field-config", type=Path, default=DEFAULT_FIELD_CONFIG)
    parser.add_argument("--style", type=Path, default=DEFAULT_STYLE)
    parser.add_argument("--sensor-to-ego", type=Path, default=DEFAULT_SENSOR_TRANSFORM)
    parser.add_argument("--tvk-config", type=Path, default=DEFAULT_TVK_CONFIG)
    parser.add_argument("--minimum-frame-separation", type=int, default=100)
    parser.add_argument("--rebuild-fields", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    field_config = args.field_config.resolve()
    style = _read_json(args.style.resolve())
    _configure(style)
    scenes = select_diverse_scenes(
        output_root / "main_primary_seed0_A" / "predictions.npz",
        minimum_frame_separation=args.minimum_frame_separation,
    )
    for record in scenes:
        sequence = str(record["sequence"])
        frame = int(record["frame"])
        field_path = (
            TERRAFLOW_ROOT / "outputs" / "terrain_fields"
            / f"{sequence}_{frame:06d}_verified.npz"
        )
        if args.rebuild_fields or not field_path.is_file():
            build_archive(
                argparse.Namespace(
                    data_root=args.data_root.resolve(),
                    sequence=sequence,
                    frame=frame,
                    sensor="ouster",
                    config=field_config,
                    output=field_path,
                    sensor_to_ego=args.sensor_to_ego.resolve(),
                    allow_unverified_identity=False,
                    geometry_only=False,
                )
            )
        figure, sources, metrics = render(
            field_path=field_path,
            output_root=output_root,
            config_path=field_config,
            case_name="",
            style=style,
            scene_id_override=str(record["scene_id"]),
            tvk_config_path=args.tvk_config.resolve(),
        )
        record["field_archive"] = str(field_path.resolve())
        record["figure_base"] = str(figure.resolve())
        record["source_data"] = [str(path.resolve()) for path in sources]
        record["metrics"] = metrics
    manifest = {
        "selection_rule": (
            "Six deterministic categories selected from seed-0 test GT geometry only: "
            "long path, left turn, right turn, high curvature, typical straight, short path."
        ),
        "minimum_frame_separation": args.minimum_frame_separation,
        "coordinate_convention": "planning ego: x forward, y left, z up",
        "sensor_transform": "reviewed explicit T_ego_sensor",
        "table_displayed_in_figures": False,
        "trajectory_model": "VTF-Flow with unified terrain-vehicle kinematic guidance",
        "tvk_config": str(args.tvk_config.resolve()),
        "scenes": scenes,
    }
    manifest_path = output_root / "raw_terrain_scene_set_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "manifest": str(manifest_path), "scenes": scenes}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

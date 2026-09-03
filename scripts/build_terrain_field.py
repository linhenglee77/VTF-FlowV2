"""Build a local continuous terrain field from one RELLIS-3D LiDAR frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from TerraFlow.terrain.feasibility_field import (  # noqa: E402
    ContinuousTerrainField,
    load_terrain_field_config,
)
from TerraFlow.terrain.terrain_features import (  # noqa: E402
    build_terrain_features,
    stack_feature_names,
    transform_points,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "rellis3d_terrain_field.json"


def resolve_sequence_root(data_root: Path, sequence: str) -> Path:
    """Locate an extracted sequence without baking in a machine-specific root."""

    candidates = (
        data_root / "processed" / "Rellis-3D" / sequence,
        data_root / "Rellis-3D" / sequence,
        data_root / sequence,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    attempted = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"sequence {sequence} not found; attempted: {attempted}")


def load_transform(path: Path) -> np.ndarray:
    """Load an explicitly documented ``T_ego_sensor`` 4x4 matrix."""

    suffix = path.suffix.lower()
    if suffix == ".npy":
        value = np.load(path, allow_pickle=False)
    elif suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        value = raw.get("T_ego_sensor", raw.get("matrix", raw)) if isinstance(raw, dict) else raw
    else:
        value = np.loadtxt(path, dtype=np.float64)
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.size == 12:
        matrix = np.vstack((matrix.reshape(3, 4), np.array([0.0, 0.0, 0.0, 1.0])))
    if matrix.shape != (4, 4):
        raise ValueError("sensor-to-ego transform must contain a 4x4 or 3x4 matrix")
    return matrix


def read_lidar(path: Path) -> np.ndarray:
    """Parse RELLIS-3D float32x4 point records."""

    values = np.fromfile(path, dtype="<f4")
    if values.size % 4:
        raise ValueError(f"point-cloud length is not divisible by four: {path}")
    return values.reshape(-1, 4)


def read_labels(path: Path, expected_points: int) -> np.ndarray:
    """Parse exact uint32 point labels without an undocumented bit mask."""

    labels = np.fromfile(path, dtype="<u4")
    if len(labels) != expected_points:
        raise ValueError(
            f"point/label count mismatch: {expected_points} points vs {len(labels)} labels"
        )
    return labels


def build_archive(args: argparse.Namespace) -> Path:
    """Build, serialize and describe one terrain field archive."""

    definition = load_terrain_field_config(args.config)
    sequence = f"{int(args.sequence):05d}" if str(args.sequence).isdigit() else str(args.sequence)
    frame = int(args.frame)
    sequence_root = resolve_sequence_root(args.data_root.expanduser().resolve(), sequence)
    if args.sensor == "ouster":
        cloud_dir = "os1_cloud_node_kitti_bin"
        label_dir = "os1_cloud_node_semantickitti_label_id"
    else:
        cloud_dir = "vel_cloud_node_kitti_bin"
        label_dir = "vel_cloud_node_semantickitti_label_id"
    cloud_path = sequence_root / cloud_dir / f"{frame:06d}.bin"
    label_path = sequence_root / label_dir / f"{frame:06d}.label"
    if not cloud_path.is_file():
        raise FileNotFoundError(cloud_path)
    points = read_lidar(cloud_path)
    labels = None if args.geometry_only else read_labels(label_path, len(points))

    if args.sensor_to_ego is not None:
        transform = load_transform(args.sensor_to_ego)
        frame_status = "ego_from_explicit_T_ego_sensor"
    elif args.allow_unverified_identity:
        transform = np.eye(4, dtype=np.float64)
        frame_status = "sensor_axes_treated_as_ego_for_diagnostics_only"
    else:
        raise ValueError(
            "RELLIS-3D sensor-to-ego convention is unresolved locally. Provide "
            "--sensor-to-ego or explicitly use --allow-unverified-identity for diagnostics."
        )
    local_points = transform_points(points[:, :3], transform)
    features = build_terrain_features(local_points, labels, definition.feature)
    field = ContinuousTerrainField(features, definition.cost)

    output = args.output
    if output is None:
        output = PROJECT_ROOT / "outputs" / "terrain_fields" / f"{sequence}_{frame:06d}.npz"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    policy = {
        str(label): {
            "name": value.name,
            "cost": value.cost,
            "role": value.role,
            "rationale": value.rationale,
        }
        for label, value in (definition.cost.semantic_classes or {}).items()
    }
    arrays: dict[str, np.ndarray] = {
        name: getattr(features, name).detach().cpu().numpy()
        for name in stack_feature_names()
    }
    arrays.update(
        {
            "terrain_cost": field.cost_map[0, 0].detach().cpu().numpy(),
            "feasibility": field.feasibility_map[0, 0].detach().cpu().numpy(),
            "x_min_m": np.asarray(features.grid.x_min_m, dtype=np.float32),
            "x_max_m": np.asarray(features.grid.x_max_m, dtype=np.float32),
            "y_min_m": np.asarray(features.grid.y_min_m, dtype=np.float32),
            "y_max_m": np.asarray(features.grid.y_max_m, dtype=np.float32),
            "resolution_m": np.asarray(features.grid.resolution_m, dtype=np.float32),
            "sequence": np.asarray(sequence),
            "frame": np.asarray(frame, dtype=np.int32),
            "sensor": np.asarray(args.sensor),
            "coordinate_status": np.asarray(frame_status),
            "T_ego_sensor": transform.astype(np.float64),
            "point_cloud_path": np.asarray(str(cloud_path)),
            "semantic_label_path": np.asarray("" if labels is None else str(label_path)),
            "semantic_provenance": np.asarray(definition.semantic_provenance),
            "label_encoding": np.asarray(definition.label_encoding),
            "semantic_policy_json": np.asarray(json.dumps(policy, sort_keys=True)),
            "config_json": np.asarray(args.config.read_text(encoding="utf-8")),
        }
    )
    np.savez_compressed(output, **arrays)
    valid_fraction = float(features.geometry_valid.float().mean())
    obstacle_fraction = float(features.occupancy.float().mean())
    print(f"Terrain field: {output}")
    print(f"Sequence/frame: {sequence}/{frame:06d}; sensor={args.sensor}")
    print(f"Coordinate status: {frame_status}")
    print(f"Grid: {features.grid.height}x{features.grid.width} at {features.grid.resolution_m:.3f} m")
    print(f"Points: {len(points)}; geometry-valid cells={valid_fraction:.3f}; occupied cells={obstacle_fraction:.3f}")
    print(
        f"Feasibility range: [{float(field.feasibility_map.min()):.4f}, "
        f"{float(field.feasibility_map.max()):.4f}]"
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True, help="Sequence ID, e.g. 00004")
    parser.add_argument("--frame", type=int, required=True, help="Zero-based frame index")
    parser.add_argument("--sensor", choices=("ouster", "velodyne"), default="ouster")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--sensor-to-ego",
        type=Path,
        help="Verified 4x4/3x4 T_ego_sensor in .npy, .json or whitespace text format.",
    )
    parser.add_argument(
        "--allow-unverified-identity",
        action="store_true",
        help="Diagnostic only: treat raw sensor xyz axes as the requested local axes.",
    )
    parser.add_argument("--geometry-only", action="store_true")
    return parser


def main() -> None:
    build_archive(build_parser().parse_args())


if __name__ == "__main__":
    main()

"""Compare binary, terrain-only and vehicle-conditioned fields on saved paths."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import torch

TERRAFLOW_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TERRAFLOW_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from TerraFlow.scripts.build_terrain_field import (  # noqa: E402
    load_transform,
    read_labels,
    read_lidar,
    resolve_sequence_root,
)
from TerraFlow.terrain.feasibility_field import (  # noqa: E402
    ContinuousTerrainField,
    load_terrain_field_config,
)
from TerraFlow.terrain.terrain_features import (  # noqa: E402
    build_terrain_features,
    transform_points,
)
from TerraFlow.terrain.vehicle_conditioned_field import (  # noqa: E402
    BinaryTraversabilityField,
    VehicleConditionedTerrainField,
    load_vehicle_conditioned_config,
    trajectory_motion_state,
)

DEFAULT_CONFIG = TERRAFLOW_ROOT / "configs" / "rellis3d_terrain_field.json"


@dataclass
class MetricAccumulator:
    feasibility: list[np.ndarray] = dataclass_field(default_factory=list)
    cost: list[np.ndarray] = dataclass_field(default_factory=list)
    trajectory_minimum: list[np.ndarray] = dataclass_field(default_factory=list)
    trajectory_mean: list[np.ndarray] = dataclass_field(default_factory=list)
    speed: list[np.ndarray] = dataclass_field(default_factory=list)
    heading_reliability: list[np.ndarray] = dataclass_field(default_factory=list)
    valid_fraction: list[float] = dataclass_field(default_factory=list)
    occupancy_fraction: list[float] = dataclass_field(default_factory=list)
    scenes: set[int] = dataclass_field(default_factory=set)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def path_sources(archive: dict[str, np.ndarray]) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield GT and deterministic/multimodal planner outputs from one archive."""

    gt = np.asarray(archive["gt"], dtype=np.float32)
    regression = np.asarray(archive["regression"], dtype=np.float32)
    flow = np.asarray(archive["flow"], dtype=np.float32)
    if flow.ndim != 3:
        raise ValueError("saved Flow trajectories must have shape [K,H,D]")
    if "flow_ADE" in archive:
        best_index = int(np.argmin(archive["flow_ADE"]))
    else:
        best_index = int(
            np.linalg.norm(flow - gt[None], axis=-1).mean(axis=-1).argmin()
        )
    yield "GT", torch.from_numpy(gt[None])
    yield "Regression", torch.from_numpy(regression[None])
    yield "Flow candidate 0", torch.from_numpy(flow[:1])
    yield f"Flow oracle best-of-{len(flow)}", torch.from_numpy(flow[best_index : best_index + 1])
    yield f"Flow all {len(flow)} candidates", torch.from_numpy(flow)


def append_metrics(
    accumulator: MetricAccumulator,
    feasibility: torch.Tensor,
    cost: torch.Tensor,
    motion: dict[str, torch.Tensor],
    dataset_index: int,
    valid_fraction: float,
    occupancy_fraction: float,
) -> None:
    values = feasibility.detach().cpu().numpy()
    costs = cost.detach().cpu().numpy()
    accumulator.feasibility.append(values.reshape(-1))
    accumulator.cost.append(costs.reshape(-1))
    accumulator.trajectory_minimum.append(values.min(axis=-1).reshape(-1))
    accumulator.trajectory_mean.append(values.mean(axis=-1).reshape(-1))
    accumulator.speed.append(motion["speed"].detach().cpu().numpy().reshape(-1))
    accumulator.heading_reliability.append(
        motion["heading_reliability"].detach().cpu().numpy().reshape(-1)
    )
    accumulator.valid_fraction.append(valid_fraction)
    accumulator.occupancy_fraction.append(occupancy_fraction)
    accumulator.scenes.add(dataset_index)


def summarize(
    values: dict[tuple[str, str], MetricAccumulator],
    violation_threshold: float,
    coordinate_status: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (source, representation), accumulator in values.items():
        feasibility = np.concatenate(accumulator.feasibility)
        cost = np.concatenate(accumulator.cost)
        minimum = np.concatenate(accumulator.trajectory_minimum)
        trajectory_mean = np.concatenate(accumulator.trajectory_mean)
        speed = np.concatenate(accumulator.speed)
        heading_reliability = np.concatenate(accumulator.heading_reliability)
        rows.append(
            {
                "trajectory_source": source,
                "field_representation": representation,
                "scenes": len(accumulator.scenes),
                "trajectories": len(minimum),
                "waypoints": len(feasibility),
                "mean_feasibility": float(feasibility.mean()),
                "median_feasibility": float(np.median(feasibility)),
                "mean_trajectory_feasibility": float(trajectory_mean.mean()),
                "mean_minimum_trajectory_feasibility": float(minimum.mean()),
                f"waypoint_violation_rate_F_lt_{violation_threshold:g}": float(
                    np.mean(feasibility < violation_threshold)
                ),
                f"fully_feasible_trajectory_rate_F_ge_{violation_threshold:g}": float(
                    np.mean(minimum >= violation_threshold)
                ),
                "mean_field_cost": float(cost.mean()),
                "mean_speed_mps": float(speed.mean()),
                "mean_heading_reliability": float(heading_reliability.mean()),
                "heading_reliable_fraction_gt_0.5": float(
                    np.mean(heading_reliability > 0.5)
                ),
                "mean_grid_geometry_valid_fraction": float(
                    np.mean(accumulator.valid_fraction)
                ),
                "mean_grid_occupancy_fraction": float(
                    np.mean(accumulator.occupancy_fraction)
                ),
                "coordinate_status": coordinate_status,
                "F_interpretation": (
                    "relative_continuous_score_threshold_is_diagnostic_only"
                ),
            }
        )
    return rows


def evaluate(args: argparse.Namespace) -> tuple[Path, Path]:
    definition = load_terrain_field_config(args.config)
    vehicle_config = load_vehicle_conditioned_config(args.config)
    manifest = read_manifest(args.cache_root / args.split / "manifest.csv")
    example_paths = sorted(args.examples_dir.glob("*.npz"))
    if not example_paths:
        raise FileNotFoundError(f"no NPZ examples found under {args.examples_dir}")
    if args.sensor_to_ego is not None:
        transform = load_transform(args.sensor_to_ego)
        coordinate_status = (
            "verified_T_planning_ego_os1_lidar_from_local_tf_and_pose_audit:"
            + args.sensor_to_ego.name
        )
    elif args.allow_unverified_identity:
        transform = np.eye(4, dtype=np.float64)
        coordinate_status = "sensor_axes_treated_as_ego_for_diagnostics_only"
    else:
        raise ValueError(
            "Provide --sensor-to-ego or explicitly use --allow-unverified-identity."
        )

    aggregate: dict[tuple[str, str], MetricAccumulator] = {}
    per_scene_rows: list[dict[str, object]] = []
    for position, example_path in enumerate(example_paths, start=1):
        with np.load(example_path, allow_pickle=False) as opened:
            archive = {key: opened[key] for key in opened.files}
        dataset_index = int(archive["dataset_index"])
        if not 0 <= dataset_index < len(manifest):
            raise IndexError(f"dataset index {dataset_index} is outside manifest")
        metadata = manifest[dataset_index]
        sequence = str(metadata["sequence"]).zfill(5)
        frame = int(metadata.get("frame_id", metadata["frame_index"]))
        sequence_root = resolve_sequence_root(args.data_root.resolve(), sequence)
        cloud_path = sequence_root / "os1_cloud_node_kitti_bin" / f"{frame:06d}.bin"
        label_path = sequence_root / "os1_cloud_node_semantickitti_label_id" / f"{frame:06d}.label"
        points = read_lidar(cloud_path)
        labels = read_labels(label_path, len(points))
        local_points = transform_points(points[:, :3], transform)
        features = build_terrain_features(local_points, labels, definition.feature)
        terrain_field = ContinuousTerrainField(features, definition.cost)
        fields = {
            "binary traversability": BinaryTraversabilityField(terrain_field),
            "terrain-only continuous": terrain_field,
            "vehicle-conditioned continuous": VehicleConditionedTerrainField(
                terrain_field, vehicle_config
            ),
        }
        valid_fraction = float(features.geometry_valid.float().mean())
        occupancy_fraction = float(features.occupancy.float().mean())
        for source, trajectories in path_sources(archive):
            motion = trajectory_motion_state(
                trajectories, args.planning_dt_s, vehicle_config
            )
            for representation, field_object in fields.items():
                state = motion if representation == "vehicle-conditioned continuous" else None
                feasibility = field_object.query(trajectories, state)
                cost = field_object.cost(trajectories, state)
                key = (source, representation)
                accumulator = aggregate.setdefault(key, MetricAccumulator())
                append_metrics(
                    accumulator,
                    feasibility,
                    cost,
                    motion,
                    dataset_index,
                    valid_fraction,
                    occupancy_fraction,
                )
                per_scene_rows.append(
                    {
                        "dataset_index": dataset_index,
                        "sequence": sequence,
                        "frame": frame,
                        "trajectory_source": source,
                        "field_representation": representation,
                        "trajectories": trajectories.shape[0],
                        "mean_feasibility": float(feasibility.mean()),
                        "minimum_feasibility": float(feasibility.min()),
                        "waypoint_violation_rate": float(
                            (feasibility < args.violation_threshold).float().mean()
                        ),
                        "mean_field_cost": float(cost.mean()),
                        "mean_speed_mps": float(motion["speed"].mean()),
                        "mean_heading_reliability": float(
                            motion["heading_reliability"].mean()
                        ),
                        "coordinate_status": coordinate_status,
                        "F_interpretation": (
                            "relative_continuous_score_threshold_is_diagnostic_only"
                        ),
                    }
                )
        print(f"processed {position}/{len(example_paths)}: {sequence}/{frame:06d}", flush=True)

    rows = summarize(aggregate, args.violation_threshold, coordinate_status)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    per_scene_output = args.output.with_name(args.output.stem + "_per_scene.csv")
    with per_scene_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_scene_rows[0]))
        writer.writeheader()
        writer.writerows(per_scene_rows)
    print(f"Summary: {args.output.resolve()}")
    print(f"Per-scene: {per_scene_output.resolve()}")
    return args.output, per_scene_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=TERRAFLOW_ROOT / "outputs" / "experiments" / "figures" / "source_data",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output",
        type=Path,
        default=TERRAFLOW_ROOT / "outputs" / "experiments" / "field_comparison.csv",
    )
    parser.add_argument("--planning-dt-s", type=float, default=0.5)
    parser.add_argument("--violation-threshold", type=float, default=0.5)
    parser.add_argument("--sensor-to-ego", type=Path)
    parser.add_argument("--allow-unverified-identity", action="store_true")
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()

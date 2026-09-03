"""Validate terrain-field discrimination and gradients before planner training."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import math
from pathlib import Path
import sys
from typing import Any

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
from TerraFlow.terrain.terrain_features import build_terrain_features, transform_points  # noqa: E402
from TerraFlow.terrain.vehicle_conditioned_field import (  # noqa: E402
    TrajectoryGradientSmoothingConfig,
    VehicleConditionedTerrainField,
    load_gradient_smoothing_config,
    load_vehicle_conditioned_config,
    smooth_trajectory_gradient,
    trajectory_motion_state,
)

DEFAULT_CONFIG = TERRAFLOW_ROOT / "configs" / "rellis3d_terrain_field.json"
PERTURBATION_NAMES = ("smooth_spatial", "local_rotation", "controlled_offset", "obstacle_directed")


@dataclass
class PairedStore:
    gt_mean_f: list[float] = field(default_factory=list)
    perturbed_mean_f: list[float] = field(default_factory=list)
    gt_min_f: list[float] = field(default_factory=list)
    perturbed_min_f: list[float] = field(default_factory=list)
    gt_terrain_cost: list[float] = field(default_factory=list)
    perturbed_terrain_cost: list[float] = field(default_factory=list)
    gt_vehicle_cost: list[float] = field(default_factory=list)
    perturbed_vehicle_cost: list[float] = field(default_factory=list)


@dataclass
class GradientStore:
    magnitudes: list[np.ndarray] = field(default_factory=list)
    obstacle_cosines: list[np.ndarray] = field(default_factory=list)
    jitter_cosines: list[float] = field(default_factory=list)
    jitter_sensitivities: list[float] = field(default_factory=list)
    finite: list[float] = field(default_factory=list)
    attempted: int = 0
    failures: int = 0


@dataclass
class DescentStore:
    initial_f: list[float] = field(default_factory=list)
    final_f: list[float] = field(default_factory=list)
    initial_cost: list[float] = field(default_factory=list)
    final_cost: list[float] = field(default_factory=list)
    initial_occupancy: list[float] = field(default_factory=list)
    final_occupancy: list[float] = field(default_factory=list)
    displacement_mean: list[float] = field(default_factory=list)
    displacement_max: list[float] = field(default_factory=list)
    initial_smoothness: list[float] = field(default_factory=list)
    final_smoothness: list[float] = field(default_factory=list)
    attempted: int = 0
    failures: int = 0


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def smooth_noise(
    horizon: int, generator: torch.Generator, amplitude_m: float
) -> torch.Tensor:
    noise = torch.randn(horizon, 2, generator=generator)
    kernel = torch.tensor([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0])
    kernel = (kernel / kernel.sum()).view(1, 1, -1)
    channels = noise.T.unsqueeze(1)
    smoothed = torch.nn.functional.conv1d(channels, kernel, padding=3)[:, 0].T
    smoothed = smoothed / smoothed.square().mean().sqrt().clamp_min(1e-6)
    phase = torch.linspace(0.0, 1.0, horizon)
    envelope = torch.sin(torch.pi * phase).square().unsqueeze(-1)
    return amplitude_m * envelope * smoothed


def occupied_centres(field_object: ContinuousTerrainField) -> torch.Tensor:
    occupied = torch.nonzero(field_object.features.occupancy > 0.5, as_tuple=False)
    if len(occupied) == 0:
        return torch.empty(0, 2)
    grid = field_object.features.grid
    y = grid.y_min_m + (occupied[:, 0].float() + 0.5) * grid.resolution_m
    x = grid.x_min_m + (occupied[:, 1].float() + 0.5) * grid.resolution_m
    return torch.stack((x, y), dim=-1)


def construct_perturbations(
    gt: torch.Tensor,
    scene_index: int,
    obstacles: torch.Tensor,
    seed: int,
) -> dict[str, torch.Tensor | None]:
    """Construct deterministic endpoint-preserving spatial perturbations."""

    horizon = gt.shape[0]
    phase = torch.linspace(0.0, 1.0, horizon)
    envelope = torch.sin(torch.pi * phase).square()
    generator = torch.Generator().manual_seed(seed + scene_index)
    result: dict[str, torch.Tensor | None] = {}

    smooth = gt.clone()
    smooth[:, :2] += smooth_noise(horizon, generator, amplitude_m=0.55)
    result["smooth_spatial"] = smooth

    centre_index = horizon // 2
    pivot = gt[centre_index, :2]
    sign = 1.0 if scene_index % 2 == 0 else -1.0
    angle = sign * math.radians(10.0)
    rotation = torch.tensor(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=gt.dtype,
    )
    rotated = (gt[:, :2] - pivot) @ rotation.T + pivot
    local_window = torch.exp(
        -0.5 * ((torch.arange(horizon) - centre_index) / max(horizon / 6.0, 1.0)).square()
    ) * envelope
    local = gt.clone()
    local[:, :2] += local_window.unsqueeze(-1) * (rotated - gt[:, :2])
    result["local_rotation"] = local

    direction = gt[-1, :2] - torch.zeros_like(gt[-1, :2])
    tangent = direction / torch.linalg.vector_norm(direction).clamp_min(1e-6)
    normal = sign * torch.stack((-tangent[1], tangent[0]))
    offset = gt.clone()
    offset[:, :2] += 0.75 * envelope.unsqueeze(-1) * normal
    result["controlled_offset"] = offset

    if len(obstacles) == 0:
        result["obstacle_directed"] = None
    else:
        distances = torch.cdist(gt[:, :2], obstacles)
        flat_index = int(distances.argmin())
        waypoint_index = flat_index // len(obstacles)
        obstacle_index = flat_index % len(obstacles)
        vector = obstacles[obstacle_index] - gt[waypoint_index, :2]
        distance = torch.linalg.vector_norm(vector)
        if float(distance) < 1e-5 or float(distance) > 5.0:
            result["obstacle_directed"] = None
        else:
            direction_to_obstacle = vector / distance
            amplitude = float(torch.clamp(0.65 * distance, 0.45, 1.25))
            width = max(horizon / 8.0, 1.0)
            window = torch.exp(
                -0.5 * ((torch.arange(horizon) - waypoint_index) / width).square()
            ) * envelope
            directed = gt.clone()
            directed[:, :2] += amplitude * window.unsqueeze(-1) * direction_to_obstacle
            result["obstacle_directed"] = directed
    return result


def path_metrics(
    path: torch.Tensor,
    terrain: ContinuousTerrainField,
    vehicle: VehicleConditionedTerrainField,
    planning_dt_s: float,
) -> dict[str, Any]:
    batched = path.unsqueeze(0)
    state = trajectory_motion_state(batched, planning_dt_s, vehicle.config)
    terrain_f = terrain.query(batched)
    vehicle_f = vehicle.query(batched, state)
    terrain_cost = terrain.cost(batched)
    vehicle_cost = vehicle.cost(batched, state)
    return {
        "state": state,
        "terrain_f": terrain_f,
        "vehicle_f": vehicle_f,
        "terrain_cost": terrain_cost,
        "vehicle_cost": vehicle_cost,
    }


def component_contributions(
    path: torch.Tensor,
    terrain: ContinuousTerrainField,
    vehicle: VehicleConditionedTerrainField,
    planning_dt_s: float,
) -> dict[str, np.ndarray]:
    batched = path.unsqueeze(0)
    state = trajectory_motion_state(batched, planning_dt_s, vehicle.config)
    terrain_components = terrain.component_costs(batched)
    vehicle_components = vehicle.component_costs(batched, state)
    cfg = terrain.config
    values = {
        "semantic": cfg.semantic_weight * terrain_components["semantic"],
        "slope": cfg.slope_weight * terrain_components["slope"],
        "roughness": cfg.roughness_weight * terrain_components["roughness"],
        "occupancy": cfg.occupancy_weight * terrain_components["occupancy"],
        "clearance": cfg.clearance_weight * terrain_components["clearance"],
        "unknown": cfg.unknown_weight * terrain_components["unknown"],
        "speed_conditioning_additions": vehicle_components["vehicle_additional_cost"],
        "speed_slope_addition": vehicle_components["slope_speed_addition"],
        "speed_roughness_addition": vehicle_components["roughness_speed_addition"],
        "speed_clearance_addition": vehicle_components["clearance_speed_addition"],
    }
    return {name: value.detach().cpu().numpy().reshape(-1) for name, value in values.items()}


def cost_gradient(
    path: torch.Tensor,
    vehicle: VehicleConditionedTerrainField,
    planning_dt_s: float,
) -> tuple[torch.Tensor, float]:
    value = path.detach().clone().requires_grad_(True)
    state = trajectory_motion_state(value.unsqueeze(0), planning_dt_s, vehicle.config)
    objective = vehicle.cost(value.unsqueeze(0), state).mean()
    gradient = torch.autograd.grad(objective, value)[0]
    if not bool(torch.isfinite(objective) and torch.isfinite(gradient).all()):
        raise ValueError("vehicle-conditioned objective or gradient is non-finite")
    return gradient, float(objective.detach())


def gradient_diagnostics(
    path: torch.Tensor,
    vehicle: VehicleConditionedTerrainField,
    obstacles: torch.Tensor,
    planning_dt_s: float,
    jitter_m: float,
    seed: int,
) -> dict[str, Any]:
    gradient, _ = cost_gradient(path, vehicle, planning_dt_s)
    magnitude = torch.linalg.vector_norm(gradient[:, :2], dim=-1)
    generator = torch.Generator().manual_seed(seed)
    jitter = torch.randn(path.shape[0], 2, generator=generator) * jitter_m
    envelope = torch.sin(torch.pi * torch.linspace(0.0, 1.0, path.shape[0])).square()
    jitter = jitter * envelope.unsqueeze(-1)
    perturbed = path.clone()
    perturbed[:, :2] += jitter
    perturbed_gradient, _ = cost_gradient(perturbed, vehicle, planning_dt_s)
    first = gradient[:, :2].reshape(-1)
    second = perturbed_gradient[:, :2].reshape(-1)
    cosine = torch.nn.functional.cosine_similarity(first, second, dim=0, eps=1e-9)
    sensitivity = torch.linalg.vector_norm(second - first) / torch.linalg.vector_norm(
        jitter.reshape(-1)
    ).clamp_min(1e-9)

    obstacle_cosines = torch.empty(0)
    if len(obstacles):
        distances = torch.cdist(path[:, :2], obstacles)
        nearest_distance, nearest_index = distances.min(dim=1)
        active = (nearest_distance <= 2.0) & (magnitude > 1e-8)
        if bool(active.any()):
            away = path[active, :2] - obstacles[nearest_index[active]]
            descent = -gradient[active, :2]
            obstacle_cosines = torch.nn.functional.cosine_similarity(
                descent, away, dim=-1, eps=1e-9
            )
    return {
        "magnitude": magnitude.detach().cpu().numpy(),
        "obstacle_cosines": obstacle_cosines.detach().cpu().numpy(),
        "jitter_cosine": float(cosine.detach()),
        "jitter_sensitivity": float(sensitivity.detach()),
        "finite": float(
            torch.isfinite(gradient).all()
            and torch.isfinite(perturbed_gradient).all()
            and torch.isfinite(cosine)
            and torch.isfinite(sensitivity)
        ),
    }


def smoothness(path: torch.Tensor) -> float:
    if path.shape[0] < 3:
        return 0.0
    return float(torch.linalg.vector_norm(torch.diff(path[:, :2], n=2, dim=0), dim=-1).mean())


def descent_snapshot(
    path: torch.Tensor,
    start: torch.Tensor,
    terrain: ContinuousTerrainField,
    vehicle: VehicleConditionedTerrainField,
    planning_dt_s: float,
) -> dict[str, float]:
    metrics = path_metrics(path, terrain, vehicle, planning_dt_s)
    occupancy = terrain.raw_component_costs(path.unsqueeze(0))["occupancy"]
    displacement = torch.linalg.vector_norm(path[:, :2] - start[:, :2], dim=-1)
    return {
        "feasibility": float(metrics["vehicle_f"].mean()),
        "cost": float(metrics["vehicle_cost"].mean()),
        "occupancy": float((occupancy >= 0.5).float().mean()),
        "displacement_mean": float(displacement.mean()),
        "displacement_max": float(displacement.max()),
        "smoothness": smoothness(path),
    }


def feasibility_descent(
    start: torch.Tensor,
    terrain: ContinuousTerrainField,
    vehicle: VehicleConditionedTerrainField,
    planning_dt_s: float,
    learning_rate_m: float,
    maximum_step_m: float,
    gradient_smoothing: TrajectoryGradientSmoothingConfig,
    checkpoints: tuple[int, ...] = (5, 10, 20),
) -> tuple[dict[str, float], dict[int, dict[str, float]]]:
    """Apply normalized field-cost gradients with fixed perturbed endpoints."""

    initial = descent_snapshot(start, start, terrain, vehicle, planning_dt_s)
    value = start.clone()
    snapshots: dict[int, dict[str, float]] = {}
    for step in range(1, max(checkpoints) + 1):
        gradient, _ = cost_gradient(value, vehicle, planning_dt_s)
        xy_gradient = smooth_trajectory_gradient(
            gradient[:, :2], gradient_smoothing
        )
        rms = xy_gradient.square().mean().sqrt().clamp_min(1e-8)
        update = -learning_rate_m * xy_gradient / rms
        update_norm = torch.linalg.vector_norm(update, dim=-1, keepdim=True)
        update = update * (maximum_step_m / update_norm.clamp_min(maximum_step_m)).clamp(max=1.0)
        update[0] = 0.0
        update[-1] = 0.0
        value[:, :2] = value[:, :2] + update
        if step in checkpoints:
            snapshots[step] = descent_snapshot(value, start, terrain, vehicle, planning_dt_s)
    return initial, snapshots


def paired_summary(store: PairedStore) -> dict[str, float]:
    gt = np.asarray(store.gt_mean_f)
    perturbed = np.asarray(store.perturbed_mean_f)
    difference = gt - perturbed
    standard_deviation = difference.std(ddof=1) if len(difference) > 1 else 0.0
    effect = float(difference.mean() / standard_deviation) if standard_deviation > 0 else float("nan")
    return {
        "n": len(difference),
        "gt_mean_F": float(gt.mean()),
        "perturbed_mean_F": float(perturbed.mean()),
        "gt_minimum_F": float(np.mean(store.gt_min_f)),
        "perturbed_minimum_F": float(np.mean(store.perturbed_min_f)),
        "gt_mean_terrain_cost": float(np.mean(store.gt_terrain_cost)),
        "perturbed_mean_terrain_cost": float(np.mean(store.perturbed_terrain_cost)),
        "gt_mean_vehicle_cost": float(np.mean(store.gt_vehicle_cost)),
        "perturbed_mean_vehicle_cost": float(np.mean(store.perturbed_vehicle_cost)),
        "paired_win_rate_P_F_GT_gt_perturbed": float(np.mean(gt > perturbed)),
        "mean_paired_F_difference": float(difference.mean()),
        "paired_effect_size_dz": effect,
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate(args: argparse.Namespace) -> tuple[Path, Path]:
    definition = load_terrain_field_config(args.config)
    vehicle_config = load_vehicle_conditioned_config(args.config)
    gradient_smoothing = load_gradient_smoothing_config(args.config)
    split_root = args.cache_root / args.split
    manifest = read_manifest(split_root / "manifest.csv")
    trajectories = np.load(split_root / "trajectory.npy", mmap_mode="r")
    sequence = str(args.sequence).zfill(5)
    indices = [
        index for index, row in enumerate(manifest)
        if str(row.get("sequence", "")).zfill(5) == sequence
    ]
    if args.start_position:
        indices = indices[args.start_position :]
    if args.max_scenes is not None:
        indices = indices[: args.max_scenes]
    if not indices:
        raise ValueError(f"sequence {sequence} has no trajectories in split {args.split}")
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
        raise ValueError("Provide --sensor-to-ego or --allow-unverified-identity")

    gradient_positions = set(
        np.linspace(0, len(indices) - 1, min(args.gradient_scenes, len(indices)), dtype=int).tolist()
    )
    descent_positions = set(
        np.linspace(0, len(indices) - 1, min(args.descent_scenes, len(indices)), dtype=int).tolist()
    )
    paired = {name: PairedStore() for name in PERTURBATION_NAMES}
    components: dict[tuple[str, str], list[np.ndarray]] = {}
    gradients = {name: GradientStore() for name in PERTURBATION_NAMES}
    descent = {
        (name, steps): DescentStore()
        for name in PERTURBATION_NAMES for steps in (5, 10, 20)
    }
    per_scene_rows: list[dict[str, Any]] = []
    obstacle_available = 0

    for position, dataset_index in enumerate(indices):
        metadata = manifest[dataset_index]
        frame = int(metadata.get("frame_id", metadata["frame_index"]))
        sequence_root = resolve_sequence_root(args.data_root.resolve(), sequence)
        cloud_path = sequence_root / "os1_cloud_node_kitti_bin" / f"{frame:06d}.bin"
        label_path = sequence_root / "os1_cloud_node_semantickitti_label_id" / f"{frame:06d}.label"
        points = read_lidar(cloud_path)
        labels = read_labels(label_path, len(points))
        local_points = transform_points(points[:, :3], transform)
        features = build_terrain_features(local_points, labels, definition.feature)
        terrain = ContinuousTerrainField(features, definition.cost)
        vehicle = VehicleConditionedTerrainField(terrain, vehicle_config)
        obstacles = occupied_centres(terrain)
        gt = torch.from_numpy(np.array(trajectories[dataset_index], dtype=np.float32, copy=True))
        perturbations = construct_perturbations(gt, dataset_index, obstacles, args.seed)
        gt_metrics = path_metrics(gt, terrain, vehicle, args.planning_dt_s)
        for component, values in component_contributions(
            gt, terrain, vehicle, args.planning_dt_s
        ).items():
            components.setdefault(("GT", component), []).append(values)

        for name, perturbed in perturbations.items():
            if perturbed is None:
                continue
            if name == "obstacle_directed":
                obstacle_available += 1
            metrics = path_metrics(perturbed, terrain, vehicle, args.planning_dt_s)
            store = paired[name]
            store.gt_mean_f.append(float(gt_metrics["vehicle_f"].mean()))
            store.perturbed_mean_f.append(float(metrics["vehicle_f"].mean()))
            store.gt_min_f.append(float(gt_metrics["vehicle_f"].min()))
            store.perturbed_min_f.append(float(metrics["vehicle_f"].min()))
            store.gt_terrain_cost.append(float(gt_metrics["terrain_cost"].mean()))
            store.perturbed_terrain_cost.append(float(metrics["terrain_cost"].mean()))
            store.gt_vehicle_cost.append(float(gt_metrics["vehicle_cost"].mean()))
            store.perturbed_vehicle_cost.append(float(metrics["vehicle_cost"].mean()))
            per_scene_rows.append(
                {
                    "dataset_index": dataset_index,
                    "sequence": sequence,
                    "frame": frame,
                    "perturbation": name,
                    "gt_mean_F": store.gt_mean_f[-1],
                    "perturbed_mean_F": store.perturbed_mean_f[-1],
                    "paired_F_difference": store.gt_mean_f[-1] - store.perturbed_mean_f[-1],
                    "gt_minimum_F": store.gt_min_f[-1],
                    "perturbed_minimum_F": store.perturbed_min_f[-1],
                    "coordinate_status": coordinate_status,
                }
            )
            for component, values in component_contributions(
                perturbed, terrain, vehicle, args.planning_dt_s
            ).items():
                components.setdefault((name, component), []).append(values)

            if position in gradient_positions:
                gradients[name].attempted += 1
                try:
                    diagnostics = gradient_diagnostics(
                        perturbed,
                        vehicle,
                        obstacles,
                        args.planning_dt_s,
                        args.jitter_m,
                        args.seed + dataset_index * 17,
                    )
                    gradients[name].magnitudes.append(diagnostics["magnitude"])
                    if len(diagnostics["obstacle_cosines"]):
                        gradients[name].obstacle_cosines.append(diagnostics["obstacle_cosines"])
                    gradients[name].jitter_cosines.append(diagnostics["jitter_cosine"])
                    gradients[name].jitter_sensitivities.append(diagnostics["jitter_sensitivity"])
                    gradients[name].finite.append(diagnostics["finite"])
                except (RuntimeError, ValueError) as error:
                    gradients[name].failures += 1
                    print(
                        f"gradient diagnostic failed at dataset_index={dataset_index}, "
                        f"perturbation={name}: {error}",
                        flush=True,
                    )

            if position in descent_positions:
                for steps in (5, 10, 20):
                    descent[(name, steps)].attempted += 1
                try:
                    initial, snapshots = feasibility_descent(
                        perturbed,
                        terrain,
                        vehicle,
                        args.planning_dt_s,
                        args.descent_learning_rate_m,
                        args.descent_maximum_step_m,
                        gradient_smoothing,
                    )
                    for steps, final in snapshots.items():
                        descent_store = descent[(name, steps)]
                        descent_store.initial_f.append(initial["feasibility"])
                        descent_store.final_f.append(final["feasibility"])
                        descent_store.initial_cost.append(initial["cost"])
                        descent_store.final_cost.append(final["cost"])
                        descent_store.initial_occupancy.append(initial["occupancy"])
                        descent_store.final_occupancy.append(final["occupancy"])
                        descent_store.displacement_mean.append(final["displacement_mean"])
                        descent_store.displacement_max.append(final["displacement_max"])
                        descent_store.initial_smoothness.append(initial["smoothness"])
                        descent_store.final_smoothness.append(final["smoothness"])
                except (RuntimeError, ValueError) as error:
                    for steps in (5, 10, 20):
                        descent[(name, steps)].failures += 1
                    print(
                        f"descent failed at dataset_index={dataset_index}, "
                        f"perturbation={name}: {error}",
                        flush=True,
                    )

        if (position + 1) % args.progress_every == 0 or position + 1 == len(indices):
            print(f"processed {position + 1}/{len(indices)}", flush=True)

    rows: list[dict[str, Any]] = []
    common = {
        "total_gt_trajectories": len(indices),
        "coordinate_status": coordinate_status,
        "F_interpretation": "relative_continuous_score_not_safety_threshold",
        "spatial_smoothing_sigma_m": definition.cost.spatial_smoothing_sigma_m,
        "trajectory_gradient_smoothing_sigma_waypoints": gradient_smoothing.sigma_waypoints,
    }
    for name, store in paired.items():
        if not store.gt_mean_f:
            continue
        rows.append({"record_type": "paired_discrimination", "perturbation": name, **paired_summary(store), **common})

    for (group, component), arrays in components.items():
        values = np.concatenate(arrays)
        rows.append(
            {
                "record_type": "cost_component",
                "trajectory_group": group,
                "component": component,
                "n_waypoints": len(values),
                "component_mean_contribution": float(values.mean()),
                "component_variance": float(values.var()),
                **common,
            }
        )

    for name, store in gradients.items():
        if not store.magnitudes:
            continue
        magnitude = np.concatenate(store.magnitudes)
        obstacle_cosines = (
            np.concatenate(store.obstacle_cosines) if store.obstacle_cosines else np.asarray([])
        )
        rows.append(
            {
                "record_type": "gradient_quality",
                "perturbation": name,
                "gradient_trajectories": len(store.magnitudes),
                "gradient_attempted": store.attempted,
                "gradient_failure_count": store.failures,
                "gradient_failure_rate": store.failures / max(store.attempted, 1),
                "gradient_waypoints": len(magnitude),
                "gradient_magnitude_mean": float(magnitude.mean()),
                "gradient_magnitude_p05": float(np.quantile(magnitude, 0.05)),
                "gradient_magnitude_p50": float(np.quantile(magnitude, 0.50)),
                "gradient_magnitude_p95": float(np.quantile(magnitude, 0.95)),
                "gradient_magnitude_p99": float(np.quantile(magnitude, 0.99)),
                "gradient_magnitude_max": float(magnitude.max()),
                "zero_gradient_ratio": float(np.mean(magnitude <= args.zero_gradient_threshold)),
                "extreme_gradient_ratio": float(np.mean(magnitude >= args.extreme_gradient_threshold)),
                "near_obstacle_direction_samples": len(obstacle_cosines),
                "mean_descent_away_from_obstacle_cosine": (
                    float(obstacle_cosines.mean()) if len(obstacle_cosines) else "unavailable"
                ),
                "descent_away_from_obstacle_rate": (
                    float(np.mean(obstacle_cosines > 0.0)) if len(obstacle_cosines) else "unavailable"
                ),
                "jitter_gradient_cosine_mean": float(np.mean(store.jitter_cosines)),
                "jitter_gradient_sensitivity_mean_per_m": float(
                    np.mean(store.jitter_sensitivities)
                ),
                "numerically_finite_trajectory_ratio": float(np.mean(store.finite)),
                **common,
            }
        )

    for (name, steps), store in descent.items():
        if not store.initial_f:
            continue
        initial_f = np.asarray(store.initial_f)
        final_f = np.asarray(store.final_f)
        initial_occ = np.asarray(store.initial_occupancy)
        final_occ = np.asarray(store.final_occupancy)
        initial_smooth = np.asarray(store.initial_smoothness)
        final_smooth = np.asarray(store.final_smoothness)
        rows.append(
            {
                "record_type": "gradient_descent_sanity",
                "perturbation": name,
                "steps": steps,
                "descent_trajectories": len(initial_f),
                "descent_attempted": store.attempted,
                "descent_failure_count": store.failures,
                "descent_failure_rate": store.failures / max(store.attempted, 1),
                "initial_mean_F": float(initial_f.mean()),
                "final_mean_F": float(final_f.mean()),
                "mean_F_change": float((final_f - initial_f).mean()),
                "feasibility_improvement_rate": float(np.mean(final_f > initial_f)),
                "initial_mean_vehicle_cost": float(np.mean(store.initial_cost)),
                "final_mean_vehicle_cost": float(np.mean(store.final_cost)),
                "initial_occupancy_violation_rate": float(initial_occ.mean()),
                "final_occupancy_violation_rate": float(final_occ.mean()),
                "occupancy_decrease_rate": float(np.mean(final_occ < initial_occ)),
                "mean_trajectory_displacement_m": float(np.mean(store.displacement_mean)),
                "maximum_trajectory_displacement_m": float(np.max(store.displacement_max)),
                "initial_smoothness_m": float(initial_smooth.mean()),
                "final_smoothness_m": float(final_smooth.mean()),
                "mean_smoothness_ratio": float(
                    np.mean(final_smooth / np.maximum(initial_smooth, 1e-8))
                ),
                **common,
            }
        )

    rows.append(
        {
            "record_type": "coverage",
            "obstacle_directed_available": obstacle_available,
            "obstacle_directed_availability_rate": obstacle_available / len(indices),
            "gradient_scene_target": min(args.gradient_scenes, len(indices)),
            "descent_scene_target": min(args.descent_scenes, len(indices)),
            **common,
        }
    )
    write_rows(args.output, rows)
    detail_path = args.output.with_name(args.output.stem + "_per_scene.csv")
    write_rows(detail_path, per_scene_rows)
    print(f"Summary: {args.output.resolve()}")
    print(f"Per-scene: {detail_path.resolve()}")
    return args.output, detail_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--sequence", default="00004")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output",
        type=Path,
        default=TERRAFLOW_ROOT / "outputs" / "experiments" / "field_guidance_validation.csv",
    )
    parser.add_argument("--planning-dt-s", type=float, default=0.5)
    parser.add_argument("--gradient-scenes", type=int, default=128)
    parser.add_argument("--descent-scenes", type=int, default=128)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--start-position", type=int, default=0)
    parser.add_argument("--seed", type=int, default=9281)
    parser.add_argument("--jitter-m", type=float, default=0.01)
    parser.add_argument("--zero-gradient-threshold", type=float, default=1e-8)
    parser.add_argument("--extreme-gradient-threshold", type=float, default=0.5)
    parser.add_argument("--descent-learning-rate-m", type=float, default=0.05)
    parser.add_argument("--descent-maximum-step-m", type=float, default=0.10)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--sensor-to-ego", type=Path)
    parser.add_argument("--allow-unverified-identity", action="store_true")
    return parser


def main() -> None:
    validate(build_parser().parse_args())


if __name__ == "__main__":
    main()

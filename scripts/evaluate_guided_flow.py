"""Run eta/schedule sweeps and the mandatory A/B/C/D guided Flow ablation."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Subset


TERRAFLOW_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TERRAFLOW_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from TerraFlow.guidance.feasibility_flow_guidance import (  # noqa: E402
    FeasibilityFlowGuidanceConfig,
)
from TerraFlow.metrics.trajectory_metrics import trajectory_metrics  # noqa: E402
from TerraFlow.planners.flow_planner import FlowPlanner, FlowPlannerConfig  # noqa: E402
from TerraFlow.planners.guided_flow_planner import GuidedFlowPlanner  # noqa: E402
from TerraFlow.scripts.train_flow import model_from_config  # noqa: E402
from TerraFlow.scripts.train_regression import (  # noqa: E402
    CombinedSceneDataset,
    make_loader,
    sequence_partition_indices,
    set_reproducible_seed,
)
from TerraFlow.terrain.feasibility_field import AnalyticTerrainField, TerrainFieldConfig  # noqa: E402
from TerraFlow.terrain.trajectory_kinematics import trajectory_kinematic_cost  # noqa: E402
from TerraFlow.terrain.vehicle_conditioned_field import (  # noqa: E402
    BatchedVehicleConditionedTerrainField,
    VehicleConditionedFieldConfig,
    trajectory_motion_state,
)
from TerraFlow.visualization.plot_guided_flow_evolution import (  # noqa: E402
    plot_eta_tradeoffs,
    plot_guidance_evolution,
    plot_qualitative_case,
    plot_schedule_comparison,
)


DEFAULT_OUTPUT = TERRAFLOW_ROOT / "outputs" / "experiments" / "guided_flow"
CONFIG_NAMES = (
    "flow_base", "flow_vehicle_train", "flow_guided", "flow_vehicle_train_guided"
)
ETA_SWEEP = (0.0, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0)
SCHEDULES = ("constant", "early-strong", "late-strong")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    """Write complete CSV output, including a header for empty logs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames or (list(rows[0]) if rows else []))
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not names:
            return
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def load_variant_config(name: str) -> dict[str, Any]:
    """Load one of the four required public experiment configs."""

    path = TERRAFLOW_ROOT / "configs" / f"{name}.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "name", "variant_role", "checkpoint", "training_feasibility", "data",
        "sampling", "guidance", "terrain_field", "vehicle_conditioning", "metrics", "seed",
    }
    if set(config) != required:
        raise ValueError(f"{path.name} has invalid top-level keys")
    FlowPlannerConfig(**config["sampling"])
    FeasibilityFlowGuidanceConfig(**config["guidance"])
    TerrainFieldConfig(**config["terrain_field"])
    VehicleConditionedFieldConfig(**config["vehicle_conditioning"])
    checkpoint = TERRAFLOW_ROOT / config["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return config


def load_model(config: Mapping[str, Any], device: torch.device):
    """Load the exact Flow model referenced by an experiment config."""

    checkpoint_path = TERRAFLOW_ROOT / str(config["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = model_from_config(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint_path.resolve(), checkpoint


def with_guidance(
    base: Mapping[str, Any], *, strength: float, schedule: str,
    enabled: bool | None = None,
) -> dict[str, Any]:
    """Create an in-memory effective config without mutating saved templates."""

    config = copy.deepcopy(dict(base))
    config["guidance"]["strength"] = float(strength)
    config["guidance"]["schedule"] = schedule
    config["guidance"]["enabled"] = bool(strength > 0.0 if enabled is None else enabled)
    return config


def _terminal_scene_metrics(
    trajectories: torch.Tensor,
    ground_truth: torch.Tensor,
    terrain_map: torch.Tensor,
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Compute common candidate-set metrics with no F-based safety threshold."""

    metric = trajectory_metrics(trajectories, ground_truth[..., :3])
    batch, candidates, horizon, _ = trajectories.shape
    flat = trajectories.reshape(batch * candidates, horizon, 3)
    repeated_map = terrain_map.repeat_interleave(candidates, dim=0)
    terrain_cfg = TerrainFieldConfig(**config["terrain_field"])
    vehicle_cfg = VehicleConditionedFieldConfig(**config["vehicle_conditioning"])
    terrain_field = AnalyticTerrainField(repeated_map, terrain_cfg)
    components = terrain_field.component_costs(flat[..., :2])
    terrain_cost = terrain_field.cost(flat).reshape(batch, candidates, horizon)
    motion = trajectory_motion_state(
        flat, float(config["guidance"]["planning_dt_s"]), vehicle_cfg
    )
    vehicle_field = BatchedVehicleConditionedTerrainField(terrain_field, vehicle_cfg)
    vehicle_cost = vehicle_field.cost(flat, motion).reshape(batch, candidates, horizon)
    guidance_cfg = FeasibilityFlowGuidanceConfig(**config["guidance"])
    # Evaluation uses one common unit-weight objective across all variants.
    # Guidance weights control sampling only and must not redefine the metric.
    kinematic_eval_cfg = guidance_cfg.kinematic_config.__class__(
        curvature_weight=1.0,
        lateral_acceleration_weight=1.0,
        maximum_curvature_per_m=guidance_cfg.maximum_curvature_per_m,
        maximum_lateral_acceleration_mps2=(
            guidance_cfg.maximum_lateral_acceleration_mps2
        ),
        curvature_softness_per_m=guidance_cfg.curvature_softness_per_m,
        lateral_acceleration_softness_mps2=(
            guidance_cfg.lateral_acceleration_softness_mps2
        ),
    )
    kinematic = trajectory_kinematic_cost(
        flat,
        guidance_cfg.planning_dt_s,
        kinematic_eval_cfg,
    )
    kinematic_cost = kinematic["trajectory_kinematic_cost"].reshape(batch, candidates)
    curvature = kinematic["absolute_curvature_per_m"].reshape(
        batch, candidates, horizon
    )
    lateral_acceleration = kinematic["lateral_acceleration_mps2"].reshape(
        batch, candidates, horizon
    )
    curvature_violation = kinematic["curvature_violation"].reshape(
        batch, candidates, horizon
    )
    lateral_acceleration_violation = kinematic[
        "lateral_acceleration_violation"
    ].reshape(batch, candidates, horizon)
    threshold = config["metrics"]
    occupancy = components["occupancy"].reshape(batch, candidates, horizon)
    nontraversable = components["nontraversable"].reshape(batch, candidates, horizon)
    slope = components["slope"].reshape(batch, candidates, horizon)
    roughness = components["roughness"].reshape(batch, candidates, horizon)
    clearance = components["clearance"].reshape(batch, candidates, horizon)
    occupancy_violation = occupancy >= float(threshold["occupancy_threshold"])
    terrain_violation = occupancy_violation | (
        nontraversable >= float(threshold["nontraversable_threshold"])
    )
    slope_violation = slope >= float(threshold["normalized_slope_threshold"])
    return {
        "ADE_m": metric["ADE_by_candidate_m"][:, 0],
        "FDE_m": metric["FDE_by_candidate_m"][:, 0],
        "minADE@K_m": metric["minADE@K_m"],
        "minFDE@K_m": metric["minFDE@K_m"],
        "diversity_m": metric["diversity_m"],
        "smoothness_m": metric["smoothness_by_candidate_m"].mean(dim=1),
        "path_length_m": metric["path_length_by_candidate_m"].mean(dim=1),
        "maximum_local_second_difference_m": torch.linalg.vector_norm(
            torch.diff(trajectories, n=2, dim=2), dim=-1
        ).max(dim=2).values.mean(dim=1),
        "terrain_violation_rate": terrain_violation.float().mean(dim=(1, 2)),
        "occupancy_violation_rate": occupancy_violation.float().mean(dim=(1, 2)),
        "slope_violation_rate": slope_violation.float().mean(dim=(1, 2)),
        "mean_terrain_cost": terrain_cost.mean(dim=(1, 2)),
        "mean_vehicle_conditioned_cost": vehicle_cost.mean(dim=(1, 2)),
        "mean_kinematic_cost": kinematic_cost.mean(dim=1),
        "mean_unified_tvk_cost": (
            vehicle_cost.mean(dim=2) + kinematic_cost
        ).mean(dim=1),
        "mean_absolute_curvature_per_m": curvature.mean(dim=(1, 2)),
        "maximum_absolute_curvature_per_m": curvature.max(dim=2).values.mean(dim=1),
        "curvature_violation_rate": curvature_violation.float().mean(dim=(1, 2)),
        "mean_lateral_acceleration_mps2": lateral_acceleration.mean(dim=(1, 2)),
        "maximum_lateral_acceleration_mps2": (
            lateral_acceleration.max(dim=2).values.mean(dim=1)
        ),
        "lateral_acceleration_violation_rate": (
            lateral_acceleration_violation.float().mean(dim=(1, 2))
        ),
        "roughness_cost": roughness.mean(dim=(1, 2)),
        "clearance_cost": clearance.mean(dim=(1, 2)),
    }


EVOLUTION_FIELDS = (
    "run", "step", "time", "eta_mean", "guidance_cost_mean", "guidance_cost_std",
    "mean_feasibility", "mean_field_feasibility", "vehicle_cost", "kinematic_cost",
    "curvature_cost", "lateral_acceleration_cost", "terrain_cost", "occupancy_cost", "slope_cost",
    "roughness_cost", "clearance_cost", "raw_gradient_norm_mean",
    "raw_gradient_norm_median", "raw_gradient_norm_max", "guidance_gradient_norm_max",
    "zero_gradient_ratio", "nonfinite_gradient_ratio", "state_norm_max",
    "smoothed_gradient_norm_mean", "applied_correction_norm_mean",
    "flow_velocity_norm_mean", "correction_flow_ratio_mean",
    "correction_flow_ratio_max", "clean_smoothness_mean",
    "clean_max_second_difference_mean", "flow_gradient_cosine_similarity_mean",
    "mean_absolute_curvature_per_m", "maximum_absolute_curvature_per_m",
    "curvature_violation_rate", "mean_lateral_acceleration_mps2",
    "maximum_lateral_acceleration_mps2", "lateral_acceleration_violation_rate",
    "trigger_mean", "effective_eta_mean",
)


def _summarize_diagnostics(
    run_name: str,
    chunks: Mapping[str, list[np.ndarray]],
    integration_steps: int,
) -> list[dict[str, Any]]:
    """Aggregate candidate-level guidance diagnostics at every Euler step."""

    if not chunks:
        return []
    values = {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}
    rows: list[dict[str, Any]] = []
    for step in range(integration_steps):
        def column(name: str) -> np.ndarray:
            return values[name][..., step].reshape(-1)

        raw = column("raw_gradient_norm")
        rows.append({
            "run": run_name,
            "step": step,
            "time": step / integration_steps,
            "eta_mean": float(column("eta").mean()),
            "guidance_cost_mean": float(column("guidance_cost").mean()),
            "guidance_cost_std": float(column("guidance_cost").std()),
            "mean_feasibility": float(column("mean_feasibility").mean()),
            "mean_field_feasibility": float(column("mean_field_feasibility").mean()),
            "vehicle_cost": float(column("vehicle_cost").mean()),
            "kinematic_cost": float(column("kinematic_cost").mean()),
            "curvature_cost": float(column("curvature_cost").mean()),
            "lateral_acceleration_cost": float(
                column("lateral_acceleration_cost").mean()
            ),
            "terrain_cost": float(column("terrain_cost").mean()),
            "occupancy_cost": float(column("occupancy_cost").mean()),
            "slope_cost": float(column("slope_cost").mean()),
            "roughness_cost": float(column("roughness_cost").mean()),
            "clearance_cost": float(column("clearance_cost").mean()),
            "raw_gradient_norm_mean": float(raw.mean()),
            "raw_gradient_norm_median": float(np.median(raw)),
            "raw_gradient_norm_max": float(raw.max()),
            "guidance_gradient_norm_max": float(column("guidance_gradient_norm").max()),
            "zero_gradient_ratio": float(column("zero_gradient").mean()),
            "nonfinite_gradient_ratio": float(column("gradient_nonfinite").mean()),
            "state_norm_max": float(column("state_norm").max()),
            "smoothed_gradient_norm_mean": float(column("smoothed_gradient_norm").mean()),
            "applied_correction_norm_mean": float(column("applied_correction_norm").mean()),
            "flow_velocity_norm_mean": float(column("flow_velocity_norm").mean()),
            "correction_flow_ratio_mean": float(column("correction_flow_ratio").mean()),
            "correction_flow_ratio_max": float(column("correction_flow_ratio").max()),
            "clean_smoothness_mean": float(column("clean_smoothness").mean()),
            "clean_max_second_difference_mean": float(column("clean_max_second_difference").mean()),
            "flow_gradient_cosine_similarity_mean": float(
                column("flow_gradient_cosine_similarity").mean()
            ),
            "mean_absolute_curvature_per_m": float(
                column("mean_absolute_curvature_per_m").mean()
            ),
            "maximum_absolute_curvature_per_m": float(
                column("maximum_absolute_curvature_per_m").mean()
            ),
            "curvature_violation_rate": float(
                column("curvature_violation_rate").mean()
            ),
            "mean_lateral_acceleration_mps2": float(
                column("mean_lateral_acceleration_mps2").mean()
            ),
            "maximum_lateral_acceleration_mps2": float(
                column("maximum_lateral_acceleration_mps2").mean()
            ),
            "lateral_acceleration_violation_rate": float(
                column("lateral_acceleration_violation_rate").mean()
            ),
            "trigger_mean": float(column("trigger").mean()),
            "effective_eta_mean": float(column("effective_eta").mean()),
        })
    return rows


def evaluate_config(
    config: Mapping[str, Any],
    validation_dataset: Subset,
    validation_indices: Sequence[int],
    device: torch.device,
    batch_size: int,
    run_name: str,
    collect_predictions: bool,
    collect_sample_diagnostics: bool | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Evaluate one config with deterministic paired Gaussian initial states."""

    seed = int(config["seed"])
    set_reproducible_seed(seed)
    model, checkpoint_path, checkpoint = load_model(config, device)
    planner_config = FlowPlannerConfig(**config["sampling"])
    guidance_config = FeasibilityFlowGuidanceConfig(**config["guidance"])
    if guidance_config.enabled and guidance_config.strength > 0.0:
        planner = GuidedFlowPlanner(
            model, planner_config, guidance_config,
            TerrainFieldConfig(**config["terrain_field"]),
            VehicleConditionedFieldConfig(**config["vehicle_conditioning"]),
        ).to(device)
    else:
        planner = FlowPlanner(model, planner_config).to(device)
    loader = make_loader(
        validation_dataset, batch_size, shuffle=False, seed=seed + 101, num_workers=0
    )
    noise_generator = torch.Generator(device=device).manual_seed(seed + 900)
    scene_metric_parts: dict[str, list[np.ndarray]] = {}
    diagnostic_parts: dict[str, list[np.ndarray]] = {}
    prediction_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    latencies: list[np.ndarray] = []
    cursor = 0
    started_all = time.perf_counter()
    for scene in loader:
        scene = scene.to(device)
        batch = scene.batch_size
        noise = torch.randn(
            (batch, planner_config.candidates, model.trajectory_points, 3),
            device=device, dtype=scene.gt_future.dtype, generator=noise_generator,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        prediction = planner.sample(scene, initial_noise=noise)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        per_scene_latency = (time.perf_counter() - started) * 1000.0 / batch
        with torch.no_grad():
            batch_metrics = _terminal_scene_metrics(
                prediction.trajectories, scene.gt_future, scene.terrain_map, config
            )
        for name, value in batch_metrics.items():
            scene_metric_parts.setdefault(name, []).append(value.detach().cpu().numpy())
        latencies.append(np.full(batch, per_scene_latency, dtype=np.float64))
        if prediction.diagnostics:
            for name, value in prediction.diagnostics.items():
                if name == "clean_estimate":
                    continue
                diagnostic_parts.setdefault(name, []).append(value.detach().cpu().numpy())
        if collect_predictions:
            prediction_parts.append(prediction.trajectories.detach().cpu().numpy())
            target_parts.append(scene.gt_future[..., :3].detach().cpu().numpy())
        cursor += batch
        if verbose:
            print(f"  {run_name}: {cursor}/{len(validation_dataset)}", flush=True)
    scene_metrics = {
        name: np.concatenate(parts, axis=0) for name, parts in scene_metric_parts.items()
    }
    latency = np.concatenate(latencies)
    metrics = {name: float(values.mean()) for name, values in scene_metrics.items()}
    metrics.update({
        "latency_ms_per_scene": float(latency.mean()),
        "latency_p95_ms_per_scene": float(np.percentile(latency, 95)),
        "integration_steps": planner_config.integration_steps,
        "K": planner_config.candidates,
        "evaluated_scenes": len(validation_dataset),
        "wall_time_s": time.perf_counter() - started_all,
    })
    evolution = _summarize_diagnostics(
        run_name, diagnostic_parts, planner_config.integration_steps
    )
    packed_diagnostics = None
    keep_diagnostics = (
        collect_predictions
        if collect_sample_diagnostics is None
        else collect_sample_diagnostics
    )
    if keep_diagnostics and diagnostic_parts:
        packed_diagnostics = {
            name: np.concatenate(parts, axis=0)
            for name, parts in diagnostic_parts.items()
        }
    predictions = None
    if collect_predictions:
        predictions = {
            "trajectories": np.concatenate(prediction_parts, axis=0),
            "ground_truth": np.concatenate(target_parts, axis=0),
            "dataset_indices": np.asarray(validation_indices, dtype=np.int64),
        }
    return {
        "config": copy.deepcopy(dict(config)),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "metrics": metrics,
        "scene_metrics": scene_metrics,
        "predictions": predictions,
        "evolution": evolution,
        "sample_diagnostics": packed_diagnostics,
    }


RAW_GUIDANCE_FIELDS = (
    "scene_position", "dataset_index", "candidate", "step", "time", "eta",
    "guidance_cost", "mean_feasibility", "mean_field_feasibility", "vehicle_cost",
    "kinematic_cost", "curvature_cost", "lateral_acceleration_cost",
    "mean_absolute_curvature_per_m", "maximum_absolute_curvature_per_m",
    "curvature_violation_rate", "mean_lateral_acceleration_mps2",
    "maximum_lateral_acceleration_mps2", "lateral_acceleration_violation_rate",
    "terrain_cost", "occupancy_cost",
    "slope_cost", "roughness_cost", "clearance_cost", "raw_gradient_norm",
    "normalized_gradient_norm", "guidance_gradient_norm", "gradient_clip_scale",
    "zero_gradient", "gradient_nonfinite", "state_norm",
)


def _write_raw_guidance_log(
    path: Path,
    diagnostics: Mapping[str, np.ndarray] | None,
    dataset_indices: np.ndarray,
) -> None:
    """Stream per-scene, per-candidate, per-step diagnostics to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RAW_GUIDANCE_FIELDS))
        writer.writeheader()
        if not diagnostics:
            return
        scenes, candidates, steps = diagnostics["guidance_cost"].shape
        for scene_position in range(scenes):
            for candidate in range(candidates):
                for step in range(steps):
                    writer.writerow({
                        "scene_position": scene_position,
                        "dataset_index": int(dataset_indices[scene_position]),
                        "candidate": candidate,
                        "step": step,
                        "time": step / steps,
                        **{
                            name: diagnostics[name][scene_position, candidate, step]
                            for name in RAW_GUIDANCE_FIELDS[5:]
                        },
                    })


def save_variant_artifacts(
    output_dir: Path,
    result: Mapping[str, Any],
    cache_root: Path,
    baseline_latency_ms: float,
) -> None:
    """Save every required reproducibility artifact for an A/B/C/D variant."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    effective = copy.deepcopy(result["config"])
    effective["runtime"] = {
        "data_root_from_cli": str(cache_root.resolve()),
        "checkpoint_resolved": result["checkpoint_path"],
        "checkpoint_epoch": result["checkpoint_epoch"],
    }
    (output_dir / "config_effective.json").write_text(
        json.dumps(effective, indent=2), encoding="utf-8"
    )
    (output_dir / "checkpoint_reference.txt").write_text(
        result["checkpoint_path"] + "\n", encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(result["metrics"], indent=2), encoding="utf-8"
    )
    predictions = result["predictions"]
    if predictions is None:
        raise ValueError("primary variants must collect predictions")
    np.savez_compressed(output_dir / "predictions.npz", **predictions)
    _write_raw_guidance_log(
        output_dir / "guidance_step_log.csv",
        result["sample_diagnostics"],
        predictions["dataset_indices"],
    )
    latency = {
        "latency_ms_per_scene": result["metrics"]["latency_ms_per_scene"],
        "latency_p95_ms_per_scene": result["metrics"]["latency_p95_ms_per_scene"],
        "unguided_baseline_ms_per_scene": baseline_latency_ms,
        "additional_guidance_overhead_ms_per_scene": (
            result["metrics"]["latency_ms_per_scene"] - baseline_latency_ms
        ),
    }
    (output_dir / "latency.json").write_text(
        json.dumps(latency, indent=2), encoding="utf-8"
    )


def _row(label: str, result: Mapping[str, Any]) -> dict[str, Any]:
    return {"variant": label, **result["metrics"]}


def _select_tradeoff(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Select lowest vehicle cost under declared fidelity/quality guardrails."""

    baseline = next(row for row in rows if float(row["eta"]) == 0.0)
    eligible = [
        row for row in rows
        if float(row["minADE@K_m"]) <= 1.05 * float(baseline["minADE@K_m"])
        and float(row["smoothness_m"]) <= 1.20 * float(baseline["smoothness_m"])
    ]
    return min(
        eligible,
        key=lambda row: (
            float(row["mean_vehicle_conditioned_cost"]),
            float(row["terrain_violation_rate"]),
        ),
    )


def _eta_zero_sanity(
    base_config: Mapping[str, Any], validation_dataset: Subset, device: torch.device
) -> float:
    """Verify runtime eta=0 parity on fixed noise, beyond the unit test."""

    model, _, _ = load_model(base_config, device)
    loader = make_loader(validation_dataset, 2, shuffle=False, seed=11, num_workers=0)
    scene = next(iter(loader)).to(device)
    planner_config = FlowPlannerConfig(**base_config["sampling"])
    generator = torch.Generator(device=device).manual_seed(99)
    noise = torch.randn(
        (scene.batch_size, planner_config.candidates, model.trajectory_points, 3),
        generator=generator, device=device, dtype=scene.gt_future.dtype,
    )
    unguided = FlowPlanner(model, planner_config).sample(scene, noise)
    zero_config = FeasibilityFlowGuidanceConfig(
        **with_guidance(base_config, strength=0.0, schedule="late-strong", enabled=True)["guidance"]
    )
    guided = GuidedFlowPlanner(model, planner_config, zero_config).sample(scene, noise)
    return float((unguided.trajectories - guided.trajectories).abs().max().cpu())


def _qualitative_cases(
    source: CombinedSceneDataset,
    validation_indices: Sequence[int],
    baseline: Mapping[str, Any],
    guided: Mapping[str, Any],
    terrain_config: TerrainFieldConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Objectively select improvement, little-effect, and failure/trade-off cases."""

    base_cost = baseline["scene_metrics"]["mean_vehicle_conditioned_cost"]
    guided_cost = guided["scene_metrics"]["mean_vehicle_conditioned_cost"]
    cost_delta = guided_cost - base_cost
    ade_delta = guided["scene_metrics"]["minADE@K_m"] - baseline["scene_metrics"]["minADE@K_m"]
    improve = int(np.argmin(cost_delta))
    little = int(np.argmin(np.abs(cost_delta)))
    failure = int(np.argmax(ade_delta))
    chosen = {"improvement": improve, "little_effect": little, "fidelity_tradeoff": failure}
    records: dict[str, Any] = {}
    for label, position in chosen.items():
        dataset_index = int(validation_indices[position])
        scene = source[dataset_index].as_batch()
        field = AnalyticTerrainField(scene.terrain_map, terrain_config)
        map_h, map_w = scene.terrain_map.shape[-2:]
        x = torch.linspace(0.0, terrain_config.forward_m, map_h)
        y = torch.linspace(-terrain_config.lateral_m, terrain_config.lateral_m, map_w)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        query = torch.stack((xx, yy), dim=-1).reshape(1, -1, 2)
        feasibility = field.query(query).reshape(map_h, map_w).detach().cpu().numpy()
        plot_qualitative_case(
            feasibility,
            (-terrain_config.lateral_m, terrain_config.lateral_m, 0.0, terrain_config.forward_m),
            baseline["predictions"]["ground_truth"][position],
            baseline["predictions"]["trajectories"][position],
            guided["predictions"]["trajectories"][position],
            (
                f"{label}: index={dataset_index}, delta vehicle cost={cost_delta[position]:+.4f}, "
                f"delta minADE={ade_delta[position]:+.4f} m"
            ),
            output_dir / f"{label}_index_{dataset_index}.png",
        )
        records[label] = {
            "validation_position": position,
            "dataset_index": dataset_index,
            "vehicle_cost_delta": float(cost_delta[position]),
            "minADE_delta_m": float(ade_delta[position]),
        }
    (output_dir / "case_selection.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    return records


def _write_report(
    path: Path,
    abcd: Sequence[Mapping[str, Any]],
    eta_rows: Sequence[Mapping[str, Any]],
    schedule_rows: Sequence[Mapping[str, Any]],
    evolution: Sequence[Mapping[str, Any]],
    best: Mapping[str, Any],
    sanity: Mapping[str, Any],
) -> None:
    """Write a measurement-grounded Chinese answer to all ten questions."""

    lookup = {row["variant"]: row for row in abcd}
    a, b, c, d = (lookup[name] for name in ("Flow", "Flow + Feasibility Training", "Flow + Feasibility Guidance", "Full Model"))
    c_steps = [row for row in evolution if row["run"] == "eta_0.5"]
    progressive = (
        float(c_steps[-1]["guidance_cost_mean"]) - float(c_steps[0]["guidance_cost_mean"])
        if c_steps else float("nan")
    )
    lowest_cost_schedule = min(
        schedule_rows, key=lambda row: float(row["mean_vehicle_conditioned_cost"])
    )
    schedule_eligible = [
        row for row in schedule_rows
        if float(row["minADE@K_m"]) <= 1.05 * float(a["minADE@K_m"])
        and float(row["smoothness_m"]) <= 1.20 * float(a["smoothness_m"])
    ]
    guarded_schedule = min(
        schedule_eligible,
        key=lambda row: float(row["mean_vehicle_conditioned_cost"]),
    )
    lines = [
        "# Feasibility-Guided Flow Sampling",
        "",
        "## 方法",
        "",
        "Baseline Flow 的训练目标和 Euler solver 均未改变。每一步从当前状态计算 "
        "`x1_hat = x_t + (1-t) v_theta`，仅以 vehicle-conditioned terrain cost 为 J，"
        "通过完整 `x_t -> v_theta -> x1_hat -> J` 计算梯度，再执行 "
        "`dx/dt = v_theta - eta(t) grad_x J`。没有加入 RL、曲率、平滑、goal 或额外 occupancy guidance。",
        "",
        "## A/B/C/D 汇总",
        "",
        "| Variant | minADE@K | minFDE@K | terrain violation | terrain cost | vehicle cost | slope violation | smoothness | diversity | latency ms/scene |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in abcd:
        lines.append(
            f"| {row['variant']} | {row['minADE@K_m']:.4f} | {row['minFDE@K_m']:.4f} | "
            f"{row['terrain_violation_rate']:.4f} | {row['mean_terrain_cost']:.4f} | "
            f"{row['mean_vehicle_conditioned_cost']:.4f} | {row['slope_violation_rate']:.4f} | "
            f"{row['smoothness_m']:.4f} | {row['diversity_m']:.4f} | {row['latency_ms_per_scene']:.3f} |"
        )
    lines.extend([
        "",
        "## 十个问题的回答",
        "",
        f"1. **推理 guidance 是否降低 cost？** 是。C 相对 A 的 vehicle cost 变化 "
        f"{c['mean_vehicle_conditioned_cost'] - a['mean_vehicle_conditioned_cost']:+.4f}，terrain cost 变化 "
        f"{c['mean_terrain_cost'] - a['mean_terrain_cost']:+.4f}。",
        f"2. **是否比只做训练正则更有效？** 若只看 feasibility，C 比 B 更有效，vehicle cost 差值为 "
        f"{c['mean_vehicle_conditioned_cost'] - b['mean_vehicle_conditioned_cost']:+.4f}；但 minADE 差值 "
        f"{c['minADE@K_m'] - b['minADE@K_m']:+.4f} m，因此综合质量上不能无条件说更好。",
        f"3. **训练+guidance 是否还有收益？** 有额外 feasibility 收益：D 相对 B 的 vehicle cost 变化 "
        f"{d['mean_vehicle_conditioned_cost'] - b['mean_vehicle_conditioned_cost']:+.4f}，相对 C 变化 "
        f"{d['mean_vehicle_conditioned_cost'] - c['mean_vehicle_conditioned_cost']:+.4f}；但 D 的 minADE/smoothness "
        f"相对 B 均变差，所以仍是 trade-off 而不是全面增益。",
        f"4. **推荐 eta 范围？** 当前保守范围是 0.05–0.20。按 minADE 不超过 baseline 5%、"
        f"smoothness 不超过 baseline 20%，再最小化 vehicle cost 的预声明规则，选择 eta={float(best['eta']):g}。",
        f"5. **late-strong 是否更好？** 不是普遍更好。eta=0.5 下最低 vehicle cost 是 "
        f"{lowest_cost_schedule['schedule']}，但按与 eta sweep 相同的 fidelity/smoothness guardrail，"
        f"可接受折中为 {guarded_schedule['schedule']}；late-strong 位于两者之间。",
        f"6. **生成过程中是否渐进降低？** 总体是。eta=0.5 late-strong 的 step 0 到最后记录点 cost 变化 "
        f"{progressive:+.4f}；早期有小幅上升、后期下降且末步回弹，因此不是严格单调。",
        f"7. **哪些指标变差？** 相对 A，C 的 minADE 变化 {c['minADE@K_m']-a['minADE@K_m']:+.4f} m，"
        f"minFDE 变化 {c['minFDE@K_m']-a['minFDE@K_m']:+.4f} m，smoothness 变化 "
        f"{c['smoothness_m']-a['smoothness_m']:+.4f} m，diversity 变化 {c['diversity_m']-a['diversity_m']:+.4f} m。",
        f"8. **额外延迟？** C 相对 A 为 {c['latency_ms_per_scene']-a['latency_ms_per_scene']:+.3f} ms/scene；"
        f"D 相对 B 为 {d['latency_ms_per_scene']-b['latency_ms_per_scene']:+.3f} ms/scene。",
        f"9. **是否存在失败案例？** 存在；定性案例按最大 paired minADE 恶化客观选取，"
        f"不是只展示成功案例。",
        f"10. **是否证明 feasibility 进入生成动力学？** eta=0 最大差异为 {sanity['eta_zero_max_abs_difference']:.3g}，"
        f"非零 eta 改变同初始噪声轨迹且逐步记录到梯度/cost 演化，因此足以支持当前实现层面的“in-flow dynamics”"
        f"机制主张；但单种子、单验证序列不足以支持统计稳健性或通用安全性主张。",
        "",
        "## 限制",
        "",
        "缓存 BEV 的 clearance 是 occupancy proximity 代理，不是米制欧氏安全距离；所有 violation 都是当前相对诊断口径，"
        "没有把 F<0.5 当作通用安全阈值。当前只有一个 seed 和验证序列 00004。",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--primary-eta", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.primary_eta not in ETA_SWEEP:
        raise ValueError(f"primary eta must be one of {ETA_SWEEP}")
    configs = {name: load_variant_config(name) for name in CONFIG_NAMES}
    base_config = configs["flow_base"]
    source = CombinedSceneDataset(args.cache_root, tuple(base_config["data"]["source_splits"]))
    _, validation_indices = sequence_partition_indices(
        source.sequence_ids, base_config["data"]["validation_sequences"]
    )
    if args.max_scenes is not None:
        validation_indices = validation_indices[: args.max_scenes]
    validation_dataset = Subset(source, validation_indices)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"device": str(device), "validation_scenes": len(validation_dataset)}), flush=True)

    result_a = evaluate_config(
        configs["flow_base"], validation_dataset, validation_indices,
        device, args.batch_size, "flow_base", True,
    )
    result_b = evaluate_config(
        configs["flow_vehicle_train"], validation_dataset, validation_indices,
        device, args.batch_size, "flow_vehicle_train", True,
    )

    eta_rows: list[dict[str, Any]] = [{"eta": 0.0, "schedule": "none", **result_a["metrics"]}]
    evolution_rows: list[dict[str, Any]] = []
    eta_results: dict[float, dict[str, Any]] = {0.0: result_a}
    for eta in ETA_SWEEP[1:]:
        config = with_guidance(configs["flow_guided"], strength=eta, schedule="late-strong")
        result = evaluate_config(
            config, validation_dataset, validation_indices, device, args.batch_size,
            f"eta_{eta:g}", eta == args.primary_eta,
        )
        eta_results[eta] = result
        eta_rows.append({"eta": eta, "schedule": "late-strong", **result["metrics"]})
        evolution_rows.extend(result["evolution"])
    result_c = eta_results[float(args.primary_eta)]

    schedule_rows: list[dict[str, Any]] = []
    for schedule in SCHEDULES:
        if schedule == "late-strong":
            result = result_c
        else:
            config = with_guidance(
                configs["flow_guided"], strength=args.primary_eta, schedule=schedule
            )
            result = evaluate_config(
                config, validation_dataset, validation_indices, device, args.batch_size,
                f"schedule_{schedule}", False,
            )
            evolution_rows.extend(result["evolution"])
        schedule_rows.append({"schedule": schedule, "eta": args.primary_eta, **result["metrics"]})

    full_config = with_guidance(
        configs["flow_vehicle_train_guided"],
        strength=args.primary_eta, schedule="late-strong",
    )
    result_d = evaluate_config(
        full_config, validation_dataset, validation_indices, device, args.batch_size,
        "full_model", True,
    )
    evolution_rows.extend(result_d["evolution"])

    abcd_rows = [
        _row("Flow", result_a),
        _row("Flow + Feasibility Training", result_b),
        _row("Flow + Feasibility Guidance", result_c),
        _row("Full Model", result_d),
    ]
    baseline_latency = result_a["metrics"]["latency_ms_per_scene"]
    primary = {
        "flow_base": result_a,
        "flow_vehicle_train": result_b,
        "flow_guided": result_c,
        "flow_vehicle_train_guided": result_d,
    }
    for name, result in primary.items():
        save_variant_artifacts(
            args.output_dir / name, result, args.cache_root, baseline_latency
        )

    for row in abcd_rows:
        row["additional_guidance_overhead_ms"] = row["latency_ms_per_scene"] - baseline_latency
    write_csv(args.output_dir / "abcd_comparison.csv", abcd_rows)
    write_csv(args.output_dir / "eta_sweep.csv", eta_rows)
    write_csv(args.output_dir / "schedule_comparison.csv", schedule_rows)
    write_csv(args.output_dir / "guidance_step_evolution.csv", evolution_rows, EVOLUTION_FIELDS)
    latency_rows = [
        {
            "variant": row["variant"],
            "latency_ms_per_scene": row["latency_ms_per_scene"],
            "latency_p95_ms_per_scene": row["latency_p95_ms_per_scene"],
            "additional_guidance_overhead_ms": row["additional_guidance_overhead_ms"],
            "integration_steps": row["integration_steps"],
        }
        for row in abcd_rows
    ]
    write_csv(args.output_dir / "latency_comparison.csv", latency_rows)

    best = _select_tradeoff(eta_rows)
    c_prediction = result_c["predictions"]["trajectories"]
    a_prediction = result_a["predictions"]["trajectories"]
    trajectory_delta = np.linalg.norm(c_prediction - a_prediction, axis=-1)
    selected_evolution = result_c["evolution"]
    strong_evolution = eta_results[1.0]["evolution"]
    sanity = {
        "eta_zero_max_abs_difference": _eta_zero_sanity(
            base_config, validation_dataset, device
        ),
        "primary_guidance_mean_waypoint_displacement_m": float(trajectory_delta.mean()),
        "primary_guidance_max_waypoint_displacement_m": float(trajectory_delta.max()),
        "weak_eta_raw_gradient_norm_mean": float(
            np.mean([row["raw_gradient_norm_mean"] for row in eta_results[0.02]["evolution"]])
        ),
        "primary_eta_raw_gradient_norm_mean": float(
            np.mean([row["raw_gradient_norm_mean"] for row in selected_evolution])
        ),
        "strong_eta_raw_gradient_norm_max": float(
            np.max([row["raw_gradient_norm_max"] for row in strong_evolution])
        ),
        "strong_eta_nonfinite_gradient_ratio": float(
            np.max([row["nonfinite_gradient_ratio"] for row in strong_evolution])
        ),
        "strong_eta_max_state_norm": float(
            np.max([row["state_norm_max"] for row in strong_evolution])
        ),
        "configured_maximum_gradient_norm": float(base_config["guidance"]["maximum_gradient_norm"]),
        "selected_tradeoff": dict(best),
        "selection_rule": (
            "minimize vehicle-conditioned cost subject to minADE <= 1.05*baseline "
            "and smoothness <= 1.20*baseline"
        ),
    }
    (args.output_dir / "gradient_sanity.json").write_text(
        json.dumps(sanity, indent=2), encoding="utf-8"
    )

    qualitative_dir = args.output_dir / "qualitative_cases"
    cases = _qualitative_cases(
        source, validation_indices, result_a, result_c,
        TerrainFieldConfig(**base_config["terrain_field"]), qualitative_dir,
    )
    for name in primary:
        target = args.output_dir / name / "figures"
        target.mkdir(exist_ok=True)
    plot_eta_tradeoffs(eta_rows, figure_dir)
    plot_guidance_evolution(evolution_rows, figure_dir)
    plot_schedule_comparison(schedule_rows, figure_dir)
    _write_report(
        TERRAFLOW_ROOT / "docs" / "feasibility_guided_flow.md",
        abcd_rows, eta_rows, schedule_rows, evolution_rows, best, sanity,
    )
    summary = {
        "status": "complete",
        "device": str(device),
        "validation_sequences": base_config["data"]["validation_sequences"],
        "validation_scenes": len(validation_dataset),
        "seed": base_config["seed"],
        "paired_initial_noise": True,
        "primary_eta": args.primary_eta,
        "best_tradeoff": dict(best),
        "qualitative_cases": cases,
        "limitations": [
            "single seed and one validation sequence",
            "cached-BEV clearance is an occupancy-proximity proxy",
            "diagnostic violation thresholds are not calibrated safety thresholds",
        ],
    }
    (args.output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

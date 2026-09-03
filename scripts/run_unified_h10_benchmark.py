"""Train and evaluate six planners under one H=10, 5 s RELLIS-3D protocol.

The benchmark rebuilds the learning targets from the first ten 0.5 s samples
of the audited trajectory cache and defines the common goal as the final 5 s
ground-truth point.  It retrains all learning-based planners because the prior
publication checkpoints used thirty trajectory points.  The script is
resumable and never changes the source cache.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset


TERRAFLOW_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TERRAFLOW_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from TerraFlow.datasets.trajectory_builder import (  # noqa: E402
    load_pose_matrices,
    rellis3d_os1_to_planning_ego,
)
from TerraFlow.evaluation.final_experiments import (  # noqa: E402
    SequenceSplit,
    benjamini_hochberg,
    bootstrap_mean_ci,
    paired_wilcoxon,
    partition_sequence_indices,
    scene_identifier,
    terminal_scene_metrics,
    write_csv,
)
from TerraFlow.guidance.feasibility_flow_guidance import (  # noqa: E402
    FeasibilityFlowGuidanceConfig,
)
from TerraFlow.interfaces import SceneBatch, TrajectoryBatch  # noqa: E402
from TerraFlow.metrics.trajectory_metrics import trajectory_metrics  # noqa: E402
from TerraFlow.planners.astar_baseline import (  # noqa: E402
    AStarConfig,
    AStarPlanningError,
    AStarTerrainPlanner,
)
from TerraFlow.planners.constant_velocity import (  # noqa: E402
    ConstantVelocityConfig,
    ConstantVelocityPlanner,
)
from TerraFlow.planners.flow_planner import FlowPlanner, FlowPlannerConfig  # noqa: E402
from TerraFlow.planners.guided_flow_planner import GuidedFlowPlanner  # noqa: E402
from TerraFlow.scripts.run_final_experiments import (  # noqa: E402
    _effective_flow_config,
    _effective_regression_config,
    _load_flow,
    _load_regression,
)
from TerraFlow.scripts.run_flow_feasibility_experiment import (  # noqa: E402
    _train_one as train_flow_variant,
)
from TerraFlow.scripts.train_regression import (  # noqa: E402
    CombinedSceneDataset,
    run_full_training as train_regression,
    set_reproducible_seed,
)
from TerraFlow.terrain.feasibility_field import TerrainFieldConfig  # noqa: E402
from TerraFlow.terrain.trajectory_kinematics import (  # noqa: E402
    TrajectoryKinematicConfig,
)
from TerraFlow.terrain.vehicle_conditioned_field import (  # noqa: E402
    VehicleConditionedFieldConfig,
)


DEFAULT_CONFIG = TERRAFLOW_ROOT / "configs" / "unified_h10_benchmark.json"
DEFAULT_CACHE = WORKSPACE_ROOT / "data" / "RELLIS3D" / "trajectory_cache_h150_s5"
DEFAULT_DATA = WORKSPACE_ROOT / "data" / "RELLIS3D"
DEFAULT_OUTPUT = TERRAFLOW_ROOT / "outputs" / "unified_h10_benchmark"

DISPLAY_NAMES = {
    "CV": "Constant Velocity",
    "ASTAR": "A* terrain planner",
    "REG": "Deterministic regression",
    "FLOW": "Flow baseline",
    "VT": "VTF-Flow w/o kinematic terms",
    "VTF": "VTF-Flow (ours)",
}

SUMMARY_METRICS = (
    "ADE_candidate0_m",
    "FDE_candidate0_m",
    "minADE@K_m",
    "minFDE@K_m",
    "mean_vehicle_conditioned_cost",
    "mean_unified_tvk_cost",
    "terrain_violation_rate",
    "occupancy_violation_rate",
    "slope_violation_rate",
    "curvature_violation_rate",
    "lateral_acceleration_violation_rate",
    "smoothness_m",
    "path_length_m",
    "diversity_m",
    "goal_error_candidate0_m",
    "latency_ms_per_scene",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


class H10PlanningDataset(Dataset):
    """Expose a verified 5 s target and causal recent history for every scene."""

    def __init__(
        self,
        source: CombinedSceneDataset,
        sequence_root: Path,
        *,
        horizon: int,
        history_steps: int,
    ) -> None:
        if horizon <= 0 or history_steps < 2:
            raise ValueError("horizon must be positive and history_steps must be >= 2")
        self.source = source
        self.sequence_ids = list(source.sequence_ids)
        self.horizon = int(horizon)
        self.history_steps = int(history_steps)
        self.history_by_sequence: dict[str, torch.Tensor] = {}
        for sequence in sorted(set(self.sequence_ids)):
            raw = load_pose_matrices(sequence_root / sequence / "poses.txt")
            poses = rellis3d_os1_to_planning_ego(raw)
            rotations = poses[:, :3, :3]
            translations = poses[:, :3, 3]
            history = torch.empty(
                (poses.shape[0], self.history_steps, 3), dtype=torch.float32
            )
            offsets = torch.arange(self.history_steps - 1, -1, -1)
            for frame in range(poses.shape[0]):
                indices = torch.clamp(frame - offsets, min=0)
                delta_world = translations[indices] - translations[frame]
                local = (rotations[frame].transpose(0, 1) @ delta_world.T).T
                history[frame] = local.to(torch.float32)
            self.history_by_sequence[sequence] = history

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> SceneBatch:
        scene = self.source[index]
        if scene.gt_future.shape[0] < self.horizon:
            raise ValueError(
                f"scene {index} contains {scene.gt_future.shape[0]} future points, "
                f"fewer than required H={self.horizon}"
            )
        metadata = dict(scene.metadata)
        sequence = str(metadata["sequence"]).zfill(5)
        frame = int(metadata.get("frame_index", metadata.get("frame_id")))
        future = scene.gt_future[: self.horizon].clone()
        metadata.update(
            {
                "benchmark_horizon_steps": self.horizon,
                "benchmark_goal_definition": "ground_truth_point_at_5_seconds",
            }
        )
        return SceneBatch(
            ego_history=self.history_by_sequence[sequence][frame].clone(),
            gt_future=future,
            goal=future[-1].clone(),
            point_cloud=scene.point_cloud,
            semantic_labels=scene.semantic_labels,
            terrain_map=scene.terrain_map,
            metadata=metadata,
        )


def benchmark_split(config: Mapping[str, Any]) -> SequenceSplit:
    protocol = config["protocol"]
    return SequenceSplit(
        name="primary_h10",
        train=tuple(protocol["train"]),
        validation=tuple(protocol["validation"]),
        test=tuple(protocol["test"]),
    )


def flow_training_config(
    benchmark: Mapping[str, Any], seed: int, *, tvk: bool
) -> dict[str, Any]:
    source = {
        "protocol": {
            "source_splits": list(benchmark["protocol"]["source_splits"])
        },
        "training": {
            "epochs": int(benchmark["training"]["epochs"]),
            "vehicle_lambda": float(benchmark["training"]["vehicle_lambda"]),
        },
        "sampling": dict(benchmark["sampling"]),
    }
    config = _effective_flow_config(source, seed)
    config["model"]["trajectory_points"] = int(
        benchmark["trajectory"]["horizon_steps"]
    )
    config["regularization"]["planning_dt_s"] = float(
        benchmark["trajectory"]["planning_dt_s"]
    )
    if tvk:
        config["regularization"].update(
            {name: float(value) for name, value in benchmark["kinematic"].items()}
        )
    else:
        config["regularization"].update(
            {"curvature_weight": 0.0, "lateral_acceleration_weight": 0.0}
        )
    return config


def regression_training_config(
    benchmark: Mapping[str, Any], seed: int
) -> dict[str, Any]:
    source = {
        "protocol": {
            "source_splits": list(benchmark["protocol"]["source_splits"])
        },
        "training": {"epochs": int(benchmark["training"]["epochs"])},
    }
    config = _effective_regression_config(source, seed)
    config["model"]["horizon"] = int(benchmark["trajectory"]["horizon_steps"])
    return config


def train_learning_methods(
    dataset: H10PlanningDataset,
    indices: Mapping[str, list[int]],
    benchmark: Mapping[str, Any],
    output_root: Path,
    device: torch.device,
) -> dict[int, dict[str, Path]]:
    """Train the four learning-based planners for every configured seed."""

    checkpoints: dict[int, dict[str, Path]] = {}
    train_set = Subset(dataset, indices["train"])
    validation_set = Subset(dataset, indices["validation"])
    for seed_value in benchmark["training"]["seeds"]:
        seed = int(seed_value)
        root = output_root / "checkpoints" / f"seed_{seed}"
        paths = {
            "REG": root / "regression" / "best.pt",
            "FLOW": root / "flow" / "best.pt",
            "VT": root / "flow_vehicle" / "best.pt",
            "VTF": root / "flow_tvk" / "best.pt",
        }
        checkpoints[seed] = paths
        if not paths["REG"].is_file():
            print(f"=== training H10 regression, seed={seed} ===", flush=True)
            set_reproducible_seed(seed)
            regression = regression_training_config(benchmark, seed)
            paths["REG"].parent.mkdir(parents=True, exist_ok=True)
            save_json(paths["REG"].parent / "effective_config.json", regression)
            train_regression(
                train_set,
                validation_set,
                regression,
                paths["REG"].parent,
                device,
                int(benchmark["training"]["epochs"]),
            )
        if not paths["FLOW"].is_file():
            print(f"=== training H10 unguided Flow, seed={seed} ===", flush=True)
            config = flow_training_config(benchmark, seed, tvk=False)
            train_flow_variant(
                config,
                train_set,
                validation_set,
                paths["FLOW"].parent,
                "none",
                0.0,
                device,
                int(benchmark["training"]["epochs"]),
            )
        if not paths["VT"].is_file():
            print(f"=== training H10 VT-only Flow, seed={seed} ===", flush=True)
            config = flow_training_config(benchmark, seed, tvk=False)
            train_flow_variant(
                config,
                train_set,
                validation_set,
                paths["VT"].parent,
                "vehicle",
                float(benchmark["training"]["vehicle_lambda"]),
                device,
                int(benchmark["training"]["epochs"]),
            )
        if not paths["VTF"].is_file():
            print(f"=== training H10 complete TVK Flow, seed={seed} ===", flush=True)
            config = flow_training_config(benchmark, seed, tvk=True)
            train_flow_variant(
                config,
                train_set,
                validation_set,
                paths["VTF"].parent,
                "vehicle",
                float(benchmark["training"]["tvk_lambda"]),
                device,
                int(benchmark["training"]["epochs"]),
            )
    return checkpoints


def guidance_config(
    benchmark: Mapping[str, Any], *, use_kinematics: bool
) -> FeasibilityFlowGuidanceConfig:
    kinematic = benchmark["kinematic"] if use_kinematics else {}
    return FeasibilityFlowGuidanceConfig(
        enabled=True,
        strength=float(benchmark["guidance"]["eta"]),
        schedule=str(benchmark["guidance"]["schedule"]),
        gamma=1.0,
        gradient_normalization="rms",
        maximum_gradient_norm=4.0,
        minimum_gradient_norm=1e-7,
        field_type="vehicle",
        planning_dt_s=float(benchmark["trajectory"]["planning_dt_s"]),
        curvature_weight=float(kinematic.get("curvature_weight", 0.0)),
        lateral_acceleration_weight=float(
            kinematic.get("lateral_acceleration_weight", 0.0)
        ),
        maximum_curvature_per_m=float(
            kinematic.get("maximum_curvature_per_m", 0.35)
        ),
        maximum_lateral_acceleration_mps2=float(
            kinematic.get("maximum_lateral_acceleration_mps2", 2.5)
        ),
        curvature_softness_per_m=float(
            kinematic.get("curvature_softness_per_m", 0.05)
        ),
        lateral_acceleration_softness_mps2=float(
            kinematic.get("lateral_acceleration_softness_mps2", 0.5)
        ),
        minimum_curvature_displacement_m=float(
            kinematic.get("minimum_curvature_displacement_m", 0.1)
        ),
        curvature_reliability_softness_m=float(
            kinematic.get("curvature_reliability_softness_m", 0.02)
        ),
        save_clean_estimate_history=False,
        smoothing_kernel=str(benchmark["guidance"]["smoothing_kernel"]),
        trust_region_rho=None,
        adaptive_trigger_enabled=False,
    )


class RobustAStar:
    """Use hard occupancy constraints, with a recorded soft-cost fallback."""

    def __init__(self, horizon: int) -> None:
        shared = dict(horizon=horizon, forbid_nontraversable=False)
        self.hard = AStarTerrainPlanner(
            AStarConfig(**shared, forbid_occupied=True)
        )
        self.soft = AStarTerrainPlanner(
            AStarConfig(**shared, forbid_occupied=False)
        )

    def __call__(self, scene: SceneBatch) -> tuple[TrajectoryBatch, bool]:
        try:
            return self.hard(scene), False
        except AStarPlanningError:
            return self.soft(scene), True


def planners_for_seed(
    checkpoints: Mapping[str, Path],
    benchmark: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    horizon = int(benchmark["trajectory"]["horizon_steps"])
    flow_plan = FlowPlannerConfig(
        candidates=int(benchmark["sampling"]["candidates"]),
        integration_steps=int(benchmark["sampling"]["integration_steps"]),
        save_integration_history=False,
    )
    flow_model = _load_flow(checkpoints["FLOW"], device)
    vt_model = _load_flow(checkpoints["VT"], device)
    tvk_model = _load_flow(checkpoints["VTF"], device)
    flow_config = flow_training_config(benchmark, 0, tvk=False)
    terrain = TerrainFieldConfig(**flow_config["terrain_field"])
    vehicle = VehicleConditionedFieldConfig(**flow_config["vehicle_conditioning"])
    return {
        "REG": _load_regression(checkpoints["REG"], device),
        "FLOW": FlowPlanner(flow_model, flow_plan).to(device),
        "VT": GuidedFlowPlanner(
            vt_model,
            flow_plan,
            guidance_config(benchmark, use_kinematics=False),
            terrain,
            vehicle,
        ).to(device),
        "VTF": GuidedFlowPlanner(
            tvk_model,
            flow_plan,
            guidance_config(benchmark, use_kinematics=True),
            terrain,
            vehicle,
        ).to(device),
    }


def classical_planners(benchmark: Mapping[str, Any]) -> dict[str, Any]:
    horizon = int(benchmark["trajectory"]["horizon_steps"])
    return {
        "CV": ConstantVelocityPlanner(
            ConstantVelocityConfig(
                horizon=horizon,
                planning_dt_s=float(benchmark["trajectory"]["planning_dt_s"]),
                history_dt_s=float(benchmark["trajectory"]["history_dt_s"]),
                velocity_window=int(benchmark["trajectory"]["history_steps"]) - 1,
                stationary_fallback=True,
            )
        ),
        "ASTAR": RobustAStar(horizon),
    }


def make_noise(
    scene: SceneBatch,
    method: str,
    seed: int,
    scene_position: int,
    candidates: int,
    horizon: int,
) -> torch.Tensor | None:
    if method not in {"FLOW", "VT", "VTF"}:
        return None
    generator = torch.Generator(device=scene.gt_future.device).manual_seed(
        100_000 + seed * 10_000 + scene_position
    )
    return torch.randn(
        (1, candidates, horizon, 3),
        generator=generator,
        device=scene.gt_future.device,
        dtype=scene.gt_future.dtype,
    )


def predict(
    method: str,
    planner: Any,
    scene: SceneBatch,
    noise: torch.Tensor | None,
) -> tuple[TrajectoryBatch, bool]:
    if method == "ASTAR":
        return planner(scene)
    if method in {"FLOW", "VT", "VTF"}:
        return planner.sample(scene, noise), False
    return planner(scene), False


def evaluate_method(
    method: str,
    planner: Any,
    dataset: H10PlanningDataset,
    test_indices: Sequence[int],
    seed: int,
    benchmark: Mapping[str, Any],
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate all 1,909 scenes with batch size one and a common timer boundary."""

    metrics_path = output_dir / "metrics.json"
    scenes_path = output_dir / "scene_level_metrics.csv"
    if metrics_path.is_file() and scenes_path.is_file():
        return load_json(metrics_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    flow_config = flow_training_config(benchmark, seed, tvk=True)
    terrain_config = TerrainFieldConfig(**flow_config["terrain_field"])
    vehicle_config = VehicleConditionedFieldConfig(
        **flow_config["vehicle_conditioning"]
    )
    thresholds = flow_config["metrics"]
    kinematic = TrajectoryKinematicConfig(**benchmark["kinematic"])
    candidates = int(benchmark["sampling"]["candidates"])
    horizon = int(benchmark["trajectory"]["horizon_steps"])
    warmup = min(int(benchmark["latency"]["warmup_scenes"]), len(test_indices))

    for position, dataset_index in enumerate(test_indices[:warmup]):
        scene = dataset[dataset_index].as_batch().to(device)
        noise = make_noise(scene, method, seed, position, candidates, horizon)
        with torch.no_grad() if method not in {"VT", "VTF"} else torch.enable_grad():
            predict(method, planner, scene, noise)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    rows: list[dict[str, Any]] = []
    trajectory_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    latency: list[float] = []
    fallback_count = 0
    for position, dataset_index in enumerate(test_indices):
        scene = dataset[dataset_index].as_batch().to(device)
        noise = make_noise(scene, method, seed, position, candidates, horizon)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.no_grad() if method not in {"VT", "VTF"} else torch.enable_grad():
            prediction, fallback = predict(method, planner, scene, noise)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latency.append(elapsed_ms)
        fallback_count += int(fallback)

        terminal = terminal_scene_metrics(
            prediction.trajectories,
            scene.gt_future,
            scene.terrain_map,
            terrain_config,
            vehicle_config,
            thresholds,
            planning_dt_s=float(benchmark["trajectory"]["planning_dt_s"]),
            kinematic_config=kinematic,
        )
        standard = trajectory_metrics(prediction.trajectories, scene.gt_future)
        trajectory = prediction.trajectories
        path_length = torch.linalg.vector_norm(
            trajectory[:, :, 1:] - trajectory[:, :, :-1], dim=-1
        ).sum(dim=-1).mean(dim=1)
        goal_error = torch.linalg.vector_norm(
            trajectory[:, 0, -1] - scene.goal, dim=-1
        )
        metadata = scene.metadata[0]
        row: dict[str, Any] = {
            "scene_id": scene_identifier(metadata, int(dataset_index)),
            "sequence": str(metadata["sequence"]).zfill(5),
            "frame_id": int(metadata.get("frame_id", metadata.get("frame_index"))),
            "dataset_index": int(dataset_index),
            "method": method,
            "seed": int(seed),
            "K": int(prediction.trajectories.shape[1]),
            "astar_soft_fallback": int(fallback),
            "ADE_candidate0_m": float(standard["ADE_by_candidate_m"][0, 0]),
            "FDE_candidate0_m": float(standard["FDE_by_candidate_m"][0, 0]),
            "path_length_m": float(path_length[0]),
            "goal_error_candidate0_m": float(goal_error[0]),
            "latency_ms_per_scene": elapsed_ms,
        }
        row.update({name: float(value[0]) for name, value in terminal.items()})
        rows.append(row)
        trajectory_chunks.append(trajectory.detach().cpu().numpy())
        target_chunks.append(scene.gt_future.detach().cpu().numpy())
        if (position + 1) % 250 == 0:
            print(
                f"  {method} seed={seed}: {position + 1}/{len(test_indices)} scenes",
                flush=True,
            )

    numeric = [name for name in rows[0] if name not in {
        "scene_id", "sequence", "frame_id", "dataset_index", "method", "seed"
    }]
    summary: dict[str, Any] = {
        "method": method,
        "display_name": DISPLAY_NAMES[method],
        "seed": int(seed),
        "evaluated_scenes": len(rows),
        "K": int(rows[0]["K"]),
        "astar_soft_fallback_count": int(fallback_count),
        "latency_protocol": dict(benchmark["latency"]),
    }
    for name in numeric:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite values in {method}/{name}")
        summary[name] = float(values.mean())
    summary["latency_p50_ms_per_scene"] = float(np.percentile(latency, 50))
    summary["latency_p95_ms_per_scene"] = float(np.percentile(latency, 95))
    save_json(metrics_path, summary)
    write_csv(scenes_path, rows)
    np.savez_compressed(
        output_dir / "predictions.npz",
        trajectories=np.concatenate(trajectory_chunks),
        ground_truth=np.concatenate(target_chunks),
    )
    return summary


def aggregate_results(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for method in DISPLAY_NAMES:
        group = [row for row in records if row["method"] == method]
        if not group:
            continue
        result: dict[str, Any] = {
            "method": method,
            "display_name": DISPLAY_NAMES[method],
            "K": int(group[0]["K"]),
            "n_seeds": len(group),
            "evaluated_scenes": int(group[0]["evaluated_scenes"]),
            "astar_soft_fallback_count": int(group[0].get("astar_soft_fallback_count", 0)),
        }
        for metric in SUMMARY_METRICS:
            values = np.asarray([float(row[metric]) for row in group])
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_sd"] = (
                float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            )
        rows.append(result)
    return pd.DataFrame(rows)


def paired_statistics(
    output_root: Path,
    records: Sequence[Mapping[str, Any]],
    benchmark: Mapping[str, Any],
) -> pd.DataFrame:
    methods = [method for method in DISPLAY_NAMES if method != "VTF"]
    metrics = [
        "ADE_candidate0_m",
        "minADE@K_m",
        "mean_unified_tvk_cost",
        "terrain_violation_rate",
        "curvature_violation_rate",
        "smoothness_m",
    ]
    method_seeds = {
        method: sorted(int(row["seed"]) for row in records if row["method"] == method)
        for method in DISPLAY_NAMES
    }

    def scene_mean(method: str, metric: str) -> tuple[list[str], np.ndarray]:
        mappings = []
        for seed in method_seeds[method]:
            path = output_root / "runs" / f"{method}_seed{seed}" / "scene_level_metrics.csv"
            frame = pd.read_csv(path)
            mappings.append(dict(zip(frame["scene_id"], frame[metric])))
        ids = sorted(mappings[0])
        return ids, np.asarray(
            [[float(values[scene]) for scene in ids] for values in mappings]
        ).mean(axis=0)

    rows = []
    for method in methods:
        for metric_index, metric in enumerate(metrics):
            ids_a, baseline = scene_mean(method, metric)
            ids_b, target = scene_mean("VTF", metric)
            if ids_a != ids_b:
                raise ValueError(f"scene mismatch for {method} vs VTF")
            difference = target - baseline
            estimate, lower, upper = bootstrap_mean_ci(
                difference,
                resamples=int(benchmark["statistics"]["bootstrap_resamples"]),
                seed=int(benchmark["statistics"]["bootstrap_seed"]) + metric_index,
            )
            test = paired_wilcoxon(baseline, target)
            rows.append(
                {
                    "comparison": f"VTF - {method}",
                    "metric": metric,
                    "mean_difference": estimate,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    **test,
                }
            )
    adjusted = benjamini_hochberg([float(row["p_value"]) for row in rows])
    for row, value in zip(rows, adjusted):
        row["p_value_fdr_bh"] = float(value)
    return pd.DataFrame(rows)


def write_formatted_table(summary: pd.DataFrame, output_root: Path) -> None:
    def mean_sd(row: pd.Series, metric: str) -> str:
        mean = float(row[f"{metric}_mean"])
        sd = float(row[f"{metric}_sd"])
        return f"{mean:.4f}" if np.isnan(sd) else f"{mean:.4f} ± {sd:.4f}"

    table = pd.DataFrame(
        {
            "Method": summary["display_name"],
            "K": summary["K"].astype(int),
            "Seeds": summary["n_seeds"].astype(int),
            "Scenes": summary["evaluated_scenes"].astype(int),
            "ADE-0 (m)": [mean_sd(row, "ADE_candidate0_m") for _, row in summary.iterrows()],
            "minADE@K (m)": [mean_sd(row, "minADE@K_m") for _, row in summary.iterrows()],
            "minFDE@K (m)": [mean_sd(row, "minFDE@K_m") for _, row in summary.iterrows()],
            "TVK cost": [mean_sd(row, "mean_unified_tvk_cost") for _, row in summary.iterrows()],
            "Terrain viol.": [mean_sd(row, "terrain_violation_rate") for _, row in summary.iterrows()],
            "Curv. viol.": [mean_sd(row, "curvature_violation_rate") for _, row in summary.iterrows()],
            "Smoothness (m)": [mean_sd(row, "smoothness_m") for _, row in summary.iterrows()],
        }
    )
    table.to_csv(output_root / "unified_main_table.csv", index=False)
    headers = list(table.columns)
    lines = [
        "Unified H=10, 5 s benchmark. Learning methods report mean ± s.d. across three seeds.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    )
    (output_root / "unified_main_table.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output_root / "unified_main_table.tex").write_text(
        table.to_latex(index=False, escape=True), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Require existing H10 checkpoints and only rerun evaluation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark = load_json(args.config)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = CombinedSceneDataset(
        args.cache_root.resolve(), tuple(benchmark["protocol"]["source_splits"])
    )
    dataset = H10PlanningDataset(
        base,
        args.data_root.resolve() / "processed" / "Rellis-3D",
        horizon=int(benchmark["trajectory"]["horizon_steps"]),
        history_steps=int(benchmark["trajectory"]["history_steps"]),
    )
    split = benchmark_split(benchmark)
    indices = partition_sequence_indices(dataset.sequence_ids, split)
    if len(indices["test"]) != 1909:
        raise AssertionError(
            f"expected 1909 primary test scenes, found {len(indices['test'])}"
        )
    save_json(
        output_root / "effective_protocol.json",
        {
            **benchmark,
            "device": str(device),
            "cache_root": str(args.cache_root.resolve()),
            "data_root": str(args.data_root.resolve()),
            "split_counts": {name: len(values) for name, values in indices.items()},
            "goal_definition": "gt_future[9] at 5.0 s for every method",
            "metric_implementation": "terminal_scene_metrics shared by every method",
        },
    )
    if args.skip_training:
        checkpoints = {
            int(seed): {
                method: output_root / "checkpoints" / f"seed_{int(seed)}" / folder / "best.pt"
                for method, folder in {
                    "REG": "regression",
                    "FLOW": "flow",
                    "VT": "flow_vehicle",
                    "VTF": "flow_tvk",
                }.items()
            }
            for seed in benchmark["training"]["seeds"]
        }
        missing = [str(path) for values in checkpoints.values() for path in values.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing H10 checkpoints:\n" + "\n".join(missing))
    else:
        checkpoints = train_learning_methods(
            dataset, indices, benchmark, output_root, device
        )

    records: list[dict[str, Any]] = []
    for method, planner in classical_planners(benchmark).items():
        print(f"=== evaluating {DISPLAY_NAMES[method]} on 1909 scenes ===", flush=True)
        records.append(
            evaluate_method(
                method,
                planner,
                dataset,
                indices["test"],
                0,
                benchmark,
                output_root / "runs" / f"{method}_seed0",
                device,
            )
        )
    for seed in sorted(checkpoints):
        planners = planners_for_seed(checkpoints[seed], benchmark, device)
        for method in ("REG", "FLOW", "VT", "VTF"):
            print(
                f"=== evaluating {DISPLAY_NAMES[method]}, seed={seed}, 1909 scenes ===",
                flush=True,
            )
            records.append(
                evaluate_method(
                    method,
                    planners[method],
                    dataset,
                    indices["test"],
                    seed,
                    benchmark,
                    output_root / "runs" / f"{method}_seed{seed}",
                    device,
                )
            )
        del planners
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = aggregate_results(records)
    summary.to_csv(output_root / "unified_summary.csv", index=False)
    statistics = paired_statistics(output_root, records, benchmark)
    statistics.to_csv(output_root / "paired_statistics.csv", index=False)
    write_formatted_table(summary, output_root)
    print((output_root / "unified_main_table.md").read_text(encoding="utf-8"))
    print(f"Results written to {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

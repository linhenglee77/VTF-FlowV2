"""Train and compare Flow Matching endpoint feasibility regularization."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import DataLoader, Subset


TERRAFLOW_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TERRAFLOW_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from TerraFlow.models.flow_regularization import (  # noqa: E402
    FlowRegularizationConfig,
    regularized_flow_matching_loss,
)
from TerraFlow.planners.flow_planner import FlowPlanner, FlowPlannerConfig  # noqa: E402
from TerraFlow.scripts.train_flow import model_from_config, save_checkpoint, write_csv  # noqa: E402
from TerraFlow.scripts.train_regression import (  # noqa: E402
    CombinedSceneDataset,
    make_loader,
    sequence_partition_indices,
    set_reproducible_seed,
)
from TerraFlow.terrain.feasibility_field import (  # noqa: E402
    AnalyticTerrainField,
    TerrainFieldConfig,
)
from TerraFlow.terrain.vehicle_conditioned_field import (  # noqa: E402
    BatchedVehicleConditionedTerrainField,
    VehicleConditionedFieldConfig,
    trajectory_motion_state,
)


DEFAULT_CONFIG = TERRAFLOW_ROOT / "configs" / "rellis3d_flow_feasibility.json"
DEFAULT_OUTPUT = TERRAFLOW_ROOT / "outputs" / "experiments" / "flow_feasibility"


def load_experiment_config(path: Path) -> dict[str, Any]:
    """Load and validate every hyperparameter used by the comparison."""

    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "model", "data", "training", "sampling", "terrain_field",
        "vehicle_conditioning", "regularization", "metrics", "experiment",
    }
    if set(config) != required:
        raise ValueError(f"config keys must be exactly {sorted(required)}")
    model_from_config(config["model"])
    FlowPlannerConfig(**config["sampling"])
    TerrainFieldConfig(**config["terrain_field"])
    VehicleConditionedFieldConfig(**config["vehicle_conditioning"])
    lambdas = [float(value) for value in config["experiment"]["lambda_feasibility"]]
    if not lambdas or any(value <= 0.0 for value in lambdas):
        raise ValueError("sensitivity lambdas must be a non-empty positive list")
    return config


def _regularization_config(
    config: Mapping[str, Any], mode: str, lambda_feasibility: float
) -> FlowRegularizationConfig:
    regularization = config["regularization"]
    return FlowRegularizationConfig(
        mode=mode,  # type: ignore[arg-type]
        lambda_feasibility=lambda_feasibility,
        lambda_smoothness=float(regularization["lambda_smoothness"]),
        planning_dt_s=float(regularization["planning_dt_s"]),
        curvature_weight=float(regularization.get("curvature_weight", 0.0)),
        lateral_acceleration_weight=float(
            regularization.get("lateral_acceleration_weight", 0.0)
        ),
        maximum_curvature_per_m=float(
            regularization.get("maximum_curvature_per_m", 0.35)
        ),
        maximum_lateral_acceleration_mps2=float(
            regularization.get("maximum_lateral_acceleration_mps2", 2.5)
        ),
        curvature_softness_per_m=float(
            regularization.get("curvature_softness_per_m", 0.05)
        ),
        lateral_acceleration_softness_mps2=float(
            regularization.get("lateral_acceleration_softness_mps2", 0.5)
        ),
        minimum_curvature_displacement_m=float(
            regularization.get("minimum_curvature_displacement_m", 0.1)
        ),
        curvature_reliability_softness_m=float(
            regularization.get("curvature_reliability_softness_m", 0.02)
        ),
    )


def _fixed_validation_losses(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    regularization: FlowRegularizationConfig,
    terrain_config: TerrainFieldConfig,
    vehicle_config: VehicleConditionedFieldConfig,
    seed: int,
) -> dict[str, float]:
    """Evaluate stochastic training terms with fixed x0/t for model selection."""

    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    totals = {"total": 0.0, "flow": 0.0, "feasibility": 0.0, "smoothness": 0.0}
    count = 0
    with torch.no_grad():
        for scene in loader:
            scene = scene.to(device)
            clean = scene.gt_future[..., :3]
            base = torch.randn(clean.shape, device=device, dtype=clean.dtype, generator=generator)
            interpolation_time = torch.rand(
                clean.shape[0], device=device, dtype=clean.dtype, generator=generator
            )
            loss, terms = regularized_flow_matching_loss(
                model, scene, regularization,
                terrain_config=terrain_config,
                vehicle_config=vehicle_config,
                base=base, time=interpolation_time,
            )
            batch = scene.batch_size
            totals["total"] += float(loss) * batch
            totals["flow"] += float(terms["flow_matching_loss"]) * batch
            totals["feasibility"] += float(terms["feasibility_loss"]) * batch
            totals["smoothness"] += float(terms["smoothness_loss"]) * batch
            count += batch
    return {name: value / max(count, 1) for name, value in totals.items()}


def _train_one(
    config: dict[str, Any],
    train_dataset: Subset,
    validation_dataset: Subset,
    output_dir: Path,
    mode: str,
    lambda_feasibility: float,
    device: torch.device,
    epochs_override: int | None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Train one controlled variant and retain the best fixed validation objective."""

    training = config["training"]
    seed = int(training["seed"])
    set_reproducible_seed(seed)
    regularization = _regularization_config(config, mode, lambda_feasibility)
    terrain_config = TerrainFieldConfig(**config["terrain_field"])
    vehicle_config = VehicleConditionedFieldConfig(**config["vehicle_conditioning"])
    effective = json.loads(json.dumps(config))
    effective["active_regularization"] = {
        "mode": mode,
        "lambda_feasibility": lambda_feasibility,
        "lambda_smoothness": regularization.lambda_smoothness,
        "planning_dt_s": regularization.planning_dt_s,
        "curvature_weight": regularization.curvature_weight,
        "lateral_acceleration_weight": regularization.lateral_acceleration_weight,
        "maximum_curvature_per_m": regularization.maximum_curvature_per_m,
        "maximum_lateral_acceleration_mps2": (
            regularization.maximum_lateral_acceleration_mps2
        ),
        "curvature_softness_per_m": regularization.curvature_softness_per_m,
        "lateral_acceleration_softness_mps2": (
            regularization.lateral_acceleration_softness_mps2
        ),
        "minimum_curvature_displacement_m": (
            regularization.minimum_curvature_displacement_m
        ),
        "curvature_reliability_softness_m": (
            regularization.curvature_reliability_softness_m
        ),
        "clean_estimator": "x1_hat = x_t + (1-t) * v_theta",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "effective_config.json").write_text(
        json.dumps(effective, indent=2), encoding="utf-8"
    )
    train_loader = make_loader(
        train_dataset, int(training["batch_size"]), shuffle=True,
        seed=seed + 100, num_workers=int(training["num_workers"]),
    )
    validation_loader = make_loader(
        validation_dataset, int(training["batch_size"]) * 2, shuffle=False,
        seed=seed + 101, num_workers=int(training["num_workers"]),
    )
    model = model_from_config(config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    epochs = int(epochs_override or training["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(training["learning_rate"]) * 0.05
    )
    best_total = float("inf")
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        sums = {"total": 0.0, "flow": 0.0, "feasibility": 0.0, "smoothness": 0.0}
        count = 0
        for scene in train_loader:
            scene = scene.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, terms = regularized_flow_matching_loss(
                model, scene, regularization,
                terrain_config=terrain_config, vehicle_config=vehicle_config,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            optimizer.step()
            batch = scene.batch_size
            sums["total"] += float(loss.detach()) * batch
            sums["flow"] += float(terms["flow_matching_loss"].detach()) * batch
            sums["feasibility"] += float(terms["feasibility_loss"].detach()) * batch
            sums["smoothness"] += float(terms["smoothness_loss"].detach()) * batch
            count += batch
        train_loss = {name: value / max(count, 1) for name, value in sums.items()}
        validation_loss = _fixed_validation_losses(
            model, validation_loader, device, regularization,
            terrain_config, vehicle_config, seed + 200,
        )
        row = {
            "epoch": epoch,
            "train_total_loss": train_loss["total"],
            "train_flow_matching_loss": train_loss["flow"],
            "train_feasibility_loss": train_loss["feasibility"],
            "train_smoothness_loss": train_loss["smoothness"],
            "val_total_loss": validation_loss["total"],
            "val_flow_matching_loss": validation_loss["flow"],
            "val_feasibility_loss": validation_loss["feasibility"],
            "val_smoothness_loss": validation_loss["smoothness"],
            "learning_rate": scheduler.get_last_lr()[0],
        }
        rows.append(row)
        write_csv(output_dir / "training_log.csv", rows)
        checkpoint_config = {
            "model": config["model"], "sampling": config["sampling"],
            "training": config["training"],
        }
        if validation_loss["total"] < best_total:
            best_total = validation_loss["total"]
            save_checkpoint(
                output_dir / "best.pt", model, checkpoint_config, epoch,
                validation_loss,
                {"regularization": effective["active_regularization"],
                 "terrain_field": config["terrain_field"],
                 "vehicle_conditioning": config["vehicle_conditioning"]},
            )
        scheduler.step()
        print(
            f"{output_dir.name} epoch={epoch:03d}/{epochs} "
            f"flow={validation_loss['flow']:.4f} fea={validation_loss['feasibility']:.4f}",
            flush=True,
        )
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    summary = {
        "mode": mode,
        "lambda_feasibility": lambda_feasibility,
        "lambda_smoothness": regularization.lambda_smoothness,
        "epochs": epochs,
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_total_loss": best_total,
        "best_validation_losses": checkpoint["metrics"],
        "training_wall_time_s": time.perf_counter() - started,
        "checkpoint": str((output_dir / "best.pt").resolve()),
    }
    return model, summary


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    config: Mapping[str, Any],
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    """Evaluate common candidate-set accuracy, terrain, motion, and timing metrics."""

    from TerraFlow.metrics.trajectory_metrics import trajectory_metrics

    model.eval()
    planner = FlowPlanner(model, FlowPlannerConfig(**config["sampling"])).to(device)
    terrain_cfg = TerrainFieldConfig(**config["terrain_field"])
    vehicle_cfg = VehicleConditionedFieldConfig(**config["vehicle_conditioning"])
    thresholds = config["metrics"]
    totals = {
        "minADE@K_m": 0.0,
        "minFDE@K_m": 0.0,
        "terrain_violation_rate": 0.0,
        "mean_terrain_cost": 0.0,
        "mean_vehicle_conditioned_cost": 0.0,
        "slope_violation_rate": 0.0,
        "smoothness_m": 0.0,
    }
    scene_count = 0
    candidate_count = 0
    latency_total_ms = 0.0
    for batch_index, scene in enumerate(loader):
        scene = scene.to(device)
        torch.manual_seed(seed + batch_index)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        prediction = planner(scene)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        trajectories = prediction.trajectories
        metric = trajectory_metrics(trajectories, scene.gt_future[..., :3])
        batch, candidates, horizon, _ = trajectories.shape
        flat = trajectories.reshape(batch * candidates, horizon, 3)
        repeated_map = scene.terrain_map.repeat_interleave(candidates, dim=0)
        terrain_field = AnalyticTerrainField(repeated_map, terrain_cfg)
        components = terrain_field.component_costs(flat[..., :2])
        occupancy_violation = components["occupancy"] >= float(thresholds["occupancy_threshold"])
        nontraversable_violation = components["nontraversable"] >= float(
            thresholds["nontraversable_threshold"]
        )
        terrain_violation = (occupancy_violation | nontraversable_violation).float()
        slope_violation = (
            components["slope"] >= float(thresholds["normalized_slope_threshold"])
        ).float()
        motion = trajectory_motion_state(
            flat, float(config["regularization"]["planning_dt_s"]), vehicle_cfg
        )
        vehicle_field = BatchedVehicleConditionedTerrainField(terrain_field, vehicle_cfg)
        totals["minADE@K_m"] += float(metric["minADE@K_m"].sum())
        totals["minFDE@K_m"] += float(metric["minFDE@K_m"].sum())
        totals["terrain_violation_rate"] += float(terrain_violation.mean(dim=-1).sum())
        totals["mean_terrain_cost"] += float(terrain_field.cost(flat).mean(dim=-1).sum())
        totals["mean_vehicle_conditioned_cost"] += float(
            vehicle_field.cost(flat, motion).mean(dim=-1).sum()
        )
        totals["slope_violation_rate"] += float(slope_violation.mean(dim=-1).sum())
        totals["smoothness_m"] += float(metric["smoothness_by_candidate_m"].sum())
        scene_count += batch
        candidate_count += batch * candidates
        latency_total_ms += elapsed_ms
    report = {
        "minADE@K_m": totals["minADE@K_m"] / max(scene_count, 1),
        "minFDE@K_m": totals["minFDE@K_m"] / max(scene_count, 1),
        "terrain_violation_rate": totals["terrain_violation_rate"] / max(candidate_count, 1),
        "mean_terrain_cost": totals["mean_terrain_cost"] / max(candidate_count, 1),
        "mean_vehicle_conditioned_cost": totals["mean_vehicle_conditioned_cost"] / max(candidate_count, 1),
        "slope_violation_rate": totals["slope_violation_rate"] / max(candidate_count, 1),
        "smoothness_m": totals["smoothness_m"] / max(candidate_count, 1),
        "latency_ms_per_scene": latency_total_ms / max(scene_count, 1),
    }
    return report


def _write_aggregate(output_dir: Path, rows: list[dict[str, Any]], split: Mapping[str, Any]) -> None:
    csv_path = output_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    baseline = rows[0]
    lowest_violation = min(rows, key=lambda row: row["terrain_violation_rate"])
    lowest_cost = min(rows, key=lambda row: row["mean_terrain_cost"])
    lowest_minade = min(rows, key=lambda row: row["minADE@K_m"])

    def relative_change(row: Mapping[str, Any], metric: str) -> float:
        return 100.0 * (float(row[metric]) / float(baseline[metric]) - 1.0)

    lines = [
        "# Flow 可行性正则化实验",
        "",
        "所有实验采用相同网络容量、随机种子、序列划分、优化器和 K。原始 "
        "Flow Matching MSE 保持不变；可行性损失作用于 "
        "`x1_hat = x_t + (1-t) v_theta`。",
        "",
        f"训练序列：{', '.join(split['train_sequences'])}；验证序列："
        f"{', '.join(split['validation_sequences'])}；重叠：{split['sequence_overlap']}。",
        "",
        "`terrain_violation_rate` 是所有候选/waypoint 中 occupancy 或 non-traversable "
        "超过配置阈值的比例，只是诊断量，不是经过认证的安全率。没有把任何 F 阈值解释为"
        "安全/不安全。缓存 BEV 的 clearance 是邻障代价代理；米制 clearance 仍需原始 LiDAR。",
        "",
        "| mode | lambda_fea | minADE@K (m) | minFDE@K (m) | terrain violation | terrain cost | vehicle cost | slope violation | smoothness (m) | latency (ms/scene) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['lambda_feasibility']:.3g} | "
            f"{row['minADE@K_m']:.4f} | {row['minFDE@K_m']:.4f} | "
            f"{row['terrain_violation_rate']:.4f} | {row['mean_terrain_cost']:.4f} | "
            f"{row['mean_vehicle_conditioned_cost']:.4f} | {row['slope_violation_rate']:.4f} | "
            f"{row['smoothness_m']:.4f} | {row['latency_ms_per_scene']:.4f} |"
        )
    lines.extend([
        "",
        "## 结果解读",
        "",
        f"- 最低 terrain violation：`{lowest_violation['run']}`，相对原始 Flow "
        f"变化 {relative_change(lowest_violation, 'terrain_violation_rate'):+.2f}%。",
        f"- 最低 mean terrain cost：`{lowest_cost['run']}`，相对原始 Flow "
        f"变化 {relative_change(lowest_cost, 'mean_terrain_cost'):+.2f}%。",
        f"- 最低 minADE@K：`{lowest_minade['run']}`，相对原始 Flow "
        f"变化 {relative_change(lowest_minade, 'minADE@K_m'):+.2f}%。",
        "- 敏感性并非单调：中间 lambda 可能改善某一指标却恶化另一指标，因此不能只按 "
        "L_fea 或单个误差选择模型。",
        "- 这是单随机种子、单验证序列结果，没有置信区间；只能作为下一轮多种子实验的依据。",
        "",
        "只有当地形收益没有以不可接受的精度、平滑性或延迟退化为代价时，才应称正则化版本更好。",
    ])
    (output_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_experiment_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    source = CombinedSceneDataset(args.cache_root, tuple(config["data"]["source_splits"]))
    train_indices, validation_indices = sequence_partition_indices(
        source.sequence_ids, config["data"]["validation_sequences"]
    )
    train_dataset = Subset(source, train_indices)
    validation_dataset = Subset(source, validation_indices)
    split = {
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "train_sequences": sorted({source.sequence_ids[index] for index in train_indices}),
        "validation_sequences": sorted({source.sequence_ids[index] for index in validation_indices}),
        "sequence_overlap": sorted(
            {source.sequence_ids[index] for index in train_indices}
            & {source.sequence_ids[index] for index in validation_indices}
        ),
    }
    (args.output_dir / "sequence_split.json").write_text(
        json.dumps(split, indent=2), encoding="utf-8"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation_loader = make_loader(
        validation_dataset, int(config["training"]["batch_size"]) * 2,
        shuffle=False, seed=int(config["training"]["seed"]) + 101,
        num_workers=int(config["training"]["num_workers"]),
    )
    variants = [("none", 0.0)]
    lambdas = [float(value) for value in config["experiment"]["lambda_feasibility"]]
    variants.extend((mode, value) for mode in ("terrain", "vehicle") for value in lambdas)
    rows: list[dict[str, Any]] = []
    for mode, lambda_feasibility in variants:
        name = "flow" if mode == "none" else f"flow_{mode}_lambda_{lambda_feasibility:g}"
        run_dir = args.output_dir / name
        model, training_summary = _train_one(
            config, train_dataset, validation_dataset, run_dir,
            mode, lambda_feasibility, device, args.epochs,
        )
        metrics = _evaluate(
            model, validation_loader, config, device,
            int(config["training"]["seed"]) + 300,
        )
        row: dict[str, Any] = {
            "run": name,
            "mode": mode,
            "lambda_feasibility": lambda_feasibility,
            "lambda_smoothness": float(config["regularization"]["lambda_smoothness"]),
            "K": int(config["sampling"]["candidates"]),
            **metrics,
            "best_epoch": training_summary["best_epoch"],
            "training_wall_time_s": training_summary["training_wall_time_s"],
        }
        rows.append(row)
        (run_dir / "result.json").write_text(
            json.dumps({"training": training_summary, "metrics": metrics}, indent=2),
            encoding="utf-8",
        )
        _write_aggregate(args.output_dir, rows, split)
        print(json.dumps(row, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

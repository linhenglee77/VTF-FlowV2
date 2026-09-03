"""Validation-selected stage-2 TVK fine-tuning from a trained Flow prior.

The script never reads test metrics.  Each candidate starts from the matching
unguided Flow checkpoint, linearly warms up a low TVK loss weight, and selects
an epoch using fixed-noise planning metrics on sequence-disjoint validation
data.  Its output remains a standard ConditionalTrajectoryFlow checkpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Subset


TERRAFLOW_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TERRAFLOW_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from TerraFlow.evaluation.final_experiments import partition_sequence_indices  # noqa: E402
from TerraFlow.guidance.feasibility_flow_guidance import (  # noqa: E402
    FeasibilityFlowGuidanceConfig,
)
from TerraFlow.models.flow_regularization import (  # noqa: E402
    FlowRegularizationConfig,
    regularized_flow_matching_loss,
)
from TerraFlow.planners.flow_planner import FlowPlanner, FlowPlannerConfig  # noqa: E402
from TerraFlow.planners.guided_flow_planner import GuidedFlowPlanner  # noqa: E402
from TerraFlow.scripts.optimize_vtf_flow_validation import (  # noqa: E402
    PRIMARY_METRICS,
    _evaluate,
    _write_csv,
    select_variant,
)
from TerraFlow.scripts.run_final_experiments import _load_flow  # noqa: E402
from TerraFlow.scripts.run_unified_h10_benchmark import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_CONFIG,
    DEFAULT_DATA,
    H10PlanningDataset,
    benchmark_split,
    flow_training_config,
    guidance_config,
    load_json,
)
from TerraFlow.scripts.train_flow import save_checkpoint  # noqa: E402
from TerraFlow.scripts.train_regression import (  # noqa: E402
    CombinedSceneDataset,
    make_loader,
    set_reproducible_seed,
)
from TerraFlow.terrain.feasibility_field import TerrainFieldConfig  # noqa: E402
from TerraFlow.terrain.vehicle_conditioned_field import (  # noqa: E402
    VehicleConditionedFieldConfig,
)


DEFAULT_FINETUNE = TERRAFLOW_ROOT / "configs" / "vtf_flow_stage2_finetune.json"
DEFAULT_BENCHMARK_ROOT = TERRAFLOW_ROOT / "outputs" / "unified_h10_benchmark"
DEFAULT_OUTPUT = TERRAFLOW_ROOT / "outputs" / "vtf_flow_stage2_finetune"


def warmup_weight(target: float, epoch: int, warmup_epochs: int) -> float:
    """Linearly increase a non-negative regularizer to its target weight."""

    if target < 0.0 or epoch <= 0 or warmup_epochs <= 0:
        raise ValueError("target must be non-negative and epochs must be positive")
    return target * min(float(epoch) / float(warmup_epochs), 1.0)


def _regularization(
    benchmark: Mapping[str, Any], lambda_feasibility: float
) -> FlowRegularizationConfig:
    kinematic = benchmark["kinematic"]
    return FlowRegularizationConfig(
        mode="vehicle",
        lambda_feasibility=lambda_feasibility,
        lambda_smoothness=0.0,
        planning_dt_s=float(benchmark["trajectory"]["planning_dt_s"]),
        curvature_weight=float(kinematic["curvature_weight"]),
        lateral_acceleration_weight=float(kinematic["lateral_acceleration_weight"]),
        maximum_curvature_per_m=float(kinematic["maximum_curvature_per_m"]),
        maximum_lateral_acceleration_mps2=float(
            kinematic["maximum_lateral_acceleration_mps2"]
        ),
        curvature_softness_per_m=float(kinematic["curvature_softness_per_m"]),
        lateral_acceleration_softness_mps2=float(
            kinematic["lateral_acceleration_softness_mps2"]
        ),
        minimum_curvature_displacement_m=float(
            kinematic["minimum_curvature_displacement_m"]
        ),
        curvature_reliability_softness_m=float(
            kinematic["curvature_reliability_softness_m"]
        ),
    )


def _guidance(
    benchmark: Mapping[str, Any], fine_tune: Mapping[str, Any]
) -> FeasibilityFlowGuidanceConfig:
    base = guidance_config(benchmark, use_kinematics=True)
    values = dict(base.__dict__)
    values.update(
        strength=float(fine_tune["guidance"]["eta"]),
        schedule=str(fine_tune["guidance"]["schedule"]),
        gamma=float(fine_tune["guidance"]["gamma"]),
        smoothing_kernel=str(fine_tune["guidance"]["smoothing_kernel"]),
        endpoint_projection=str(fine_tune["guidance"]["endpoint_projection"]),
    )
    return FeasibilityFlowGuidanceConfig(**values)


def _checkpoint_score(
    metrics: Mapping[str, float],
    reference: Mapping[str, float],
    weights: Mapping[str, float],
) -> float:
    return sum(
        float(weights[name]) * float(metrics[name]) / max(float(reference[name]), 1e-12)
        for name in PRIMARY_METRICS
    )


def _train_candidate(
    *,
    seed: int,
    target_lambda: float,
    dataset: H10PlanningDataset,
    train_indices: list[int],
    validation_loader: torch.utils.data.DataLoader,
    validation_positions: list[int],
    benchmark: Mapping[str, Any],
    fine_tune: Mapping[str, Any],
    benchmark_root: Path,
    output_dir: Path,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    set_reproducible_seed(seed)
    source_checkpoint = (
        benchmark_root / "checkpoints" / f"seed_{seed}" / "flow" / "best.pt"
    )
    if not source_checkpoint.is_file():
        raise FileNotFoundError(source_checkpoint)
    model = _load_flow(source_checkpoint, device)
    flow_config = flow_training_config(benchmark, seed, tvk=True)
    flow_config["training"]["seed"] = seed
    plan = FlowPlannerConfig(
        candidates=int(benchmark["sampling"]["candidates"]),
        integration_steps=int(benchmark["sampling"]["integration_steps"]),
        save_integration_history=False,
    )
    terrain = TerrainFieldConfig(**flow_config["terrain_field"])
    vehicle = VehicleConditionedFieldConfig(**flow_config["vehicle_conditioning"])
    reference = _evaluate(
        FlowPlanner(model, plan).to(device), validation_loader, validation_positions,
        seed, benchmark, device,
    )
    reference_primary = {name: float(reference[name]) for name in PRIMARY_METRICS}
    train_loader = make_loader(
        Subset(dataset, train_indices),
        int(fine_tune["batch_size"]),
        shuffle=True,
        seed=seed + 901,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(fine_tune["learning_rate"]),
        weight_decay=float(fine_tune["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(fine_tune["epochs"]),
        eta_min=float(fine_tune["learning_rate"]) * 0.1,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    best_score = float("inf")
    best_row: dict[str, Any] | None = None
    for epoch in range(1, int(fine_tune["epochs"]) + 1):
        active_lambda = warmup_weight(
            target_lambda, epoch, int(fine_tune["warmup_epochs"])
        )
        regularization = _regularization(benchmark, active_lambda)
        model.train()
        sums = {"total": 0.0, "flow": 0.0, "feasibility": 0.0}
        count = 0
        for scene in train_loader:
            scene = scene.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, terms = regularized_flow_matching_loss(
                model,
                scene,
                regularization,
                terrain_config=terrain,
                vehicle_config=vehicle,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(fine_tune["gradient_clip_norm"])
            )
            optimizer.step()
            batch = scene.batch_size
            sums["total"] += float(loss.detach()) * batch
            sums["flow"] += float(terms["flow_matching_loss"].detach()) * batch
            sums["feasibility"] += float(terms["feasibility_loss"].detach()) * batch
            count += batch
        model.eval()
        planner = GuidedFlowPlanner(
            model, plan, _guidance(benchmark, fine_tune), terrain, vehicle
        ).to(device)
        metrics = _evaluate(
            planner, validation_loader, validation_positions, seed, benchmark, device
        )
        score = _checkpoint_score(metrics, reference_primary, fine_tune["metric_weights"])
        accuracy_limit = float(fine_tune["maximum_relative_accuracy_degradation"])
        admissible = (
            metrics["ADE_candidate0_m"]
            <= reference_primary["ADE_candidate0_m"] * (1.0 + accuracy_limit)
            and metrics["minADE@K_m"]
            <= reference_primary["minADE@K_m"] * (1.0 + accuracy_limit)
        )
        row = {
            "kind": "VTF-stage2",
            "seed": seed,
            "target_lambda": target_lambda,
            "epoch": epoch,
            "active_lambda": active_lambda,
            "train_total_loss": sums["total"] / count,
            "train_flow_matching_loss": sums["flow"] / count,
            "train_feasibility_loss": sums["feasibility"] / count,
            **metrics,
            "selection_score": score,
            "accuracy_admissible": admissible,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        rows.append(row)
        if score < best_score:
            best_score = score
            best_row = dict(row)
            save_checkpoint(
                output_dir / "best.pt",
                model,
                flow_config,
                epoch,
                {name: float(metrics[name]) for name in metrics},
                {
                    "stage2_finetune": {
                        "initialized_from": str(source_checkpoint.resolve()),
                        "target_lambda": target_lambda,
                        "active_lambda": active_lambda,
                        "validation_only_selection_score": score,
                        "validation_reference": reference_primary,
                        "guidance": dict(fine_tune["guidance"]),
                    }
                },
            )
        scheduler.step()
        _write_csv(output_dir / "training_log.csv", rows)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    if best_row is None:
        raise RuntimeError("fine-tuning produced no checkpoint")
    return rows, {
        "kind": "VTF",
        "seed": seed,
        "variant": f"lambda_{target_lambda:g}_epoch_{best_row['epoch']}",
        "checkpoint": str((output_dir / "best.pt").resolve()),
        **{name: best_row[name] for name in (*PRIMARY_METRICS, "mean_unified_tvk_cost", "smoothness_m", "diversity_m")},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--finetune-config", type=Path, default=DEFAULT_FINETUNE)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--validation-scenes", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark = load_json(args.benchmark_config)
    fine_tune = load_json(args.finetune_config)
    if args.epochs is not None:
        fine_tune["epochs"] = int(args.epochs)
    if args.validation_scenes is not None:
        fine_tune["validation_scene_count"] = int(args.validation_scenes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = CombinedSceneDataset(
        args.cache_root.resolve(), tuple(benchmark["protocol"]["source_splits"])
    )
    dataset = H10PlanningDataset(
        source,
        args.data_root.resolve() / "processed" / "Rellis-3D",
        horizon=int(benchmark["trajectory"]["horizon_steps"]),
        history_steps=int(benchmark["trajectory"]["history_steps"]),
    )
    indices = partition_sequence_indices(dataset.sequence_ids, benchmark_split(benchmark))
    available = indices["validation"]
    validation_count = min(int(fine_tune["validation_scene_count"]), len(available))
    positions = np.linspace(
        0, len(available) - 1, num=validation_count, dtype=np.int64
    ).tolist()
    selected_indices = [available[position] for position in positions]
    validation_loader = make_loader(
        Subset(dataset, selected_indices),
        int(fine_tune["validation_batch_size"]),
        shuffle=False,
        seed=877,
        num_workers=0,
    )
    all_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    started = time.perf_counter()
    for seed_value in fine_tune["screening_seeds"]:
        seed = int(seed_value)
        for target_lambda_value in fine_tune["lambda_candidates"]:
            target_lambda = float(target_lambda_value)
            run_dir = args.output_dir / f"seed_{seed}" / f"lambda_{target_lambda:g}"
            rows, candidate = _train_candidate(
                seed=seed,
                target_lambda=target_lambda,
                dataset=dataset,
                train_indices=indices["train"],
                validation_loader=validation_loader,
                validation_positions=positions,
                benchmark=benchmark,
                fine_tune=fine_tune,
                benchmark_root=args.benchmark_root,
                output_dir=run_dir,
                device=device,
            )
            all_rows.extend(rows)
            candidates.append(candidate)
    # All candidates share the same pretrained Flow reference for a given seed.
    first_log = all_rows[0]
    reference = {
        name: float(first_log[name]) / max(
            float(first_log["selection_score"]), 1e-12
        )
        for name in PRIMARY_METRICS
    }
    # Reconstruct the exact reference explicitly from the checkpoint metadata.
    checkpoint = torch.load(candidates[0]["checkpoint"], map_location="cpu", weights_only=False)
    reference = {
        name: float(checkpoint["stage2_finetune"]["validation_reference"][name])
        for name in PRIMARY_METRICS
    }
    selected = select_variant(
        candidates,
        reference,
        fine_tune["metric_weights"],
        float(fine_tune["maximum_relative_accuracy_degradation"]),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "all_epoch_metrics.csv", all_rows)
    _write_csv(args.output_dir / "candidate_checkpoints.csv", candidates)
    summary = {
        "selection_split": "validation",
        "selection_sequences": list(benchmark["protocol"]["validation"]),
        "test_sequences_consulted": [],
        "validation_scenes": validation_count,
        "flow_reference": reference,
        "selected": selected,
        "wall_time_s": time.perf_counter() - started,
        "fine_tune_config": fine_tune,
    }
    (args.output_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

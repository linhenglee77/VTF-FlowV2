"""Paired mechanism ablation for feasibility guidance versus filtering/refinement."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from TerraFlow.closed_loop.receding_horizon import BicycleConfig
from TerraFlow.datasets.rellis3d import Rellis3DSceneDataset
from TerraFlow.guidance.feasibility_guidance import FlowGuidanceConfig
from TerraFlow.planners.legacy_guided_flow_planner import (
    LegacyFlowPlannerConfig,
    LegacyGuidedFlowPlanner,
)
from TerraFlow.scripts.evaluate_receding_horizon import (
    load_field,
    load_flow,
    rollout_batch,
)
from TerraFlow.terrain.learned_feasibility_field import LearnedFieldConfig


METHODS = (
    "Unguided Flow",
    "Generate-then-filter",
    "Generate-then-refine",
    "Feasibility-guided Flow",
    "Guided Flow (no reranking)",
)


def mechanism_objective() -> FlowGuidanceConfig:
    return FlowGuidanceConfig(
        enabled=False,
        strength=0.10,
        schedule="late",
        terrain_weight=1.0,
        occupancy_weight=0.10,
        smoothness_weight=0.08,
        curvature_weight=0.12,
        steering_rate_weight=0.08,
        boundary_weight=0.30,
        progress_weight=0.25,
        path_efficiency_weight=0.20,
        initial_heading_weight=0.15,
        gradient_clip=4.0,
        vehicle_conditioned=True,
        planning_dt_s=0.5,
        normalize_objective_terms=True,
    )


def make_mechanism_planner(method, model, checkpoint, field_model, candidates, steps, device):
    objective = mechanism_objective()
    guided = method in {"Feasibility-guided Flow", "Guided Flow (no reranking)"}
    refine = method == "Generate-then-refine"
    guidance = replace(objective, enabled=guided)
    planner_config = LegacyFlowPlannerConfig(
        candidates=candidates,
        integration_steps=steps,
        track_feasibility_history=True,
        terminal_refinement_steps=8 if refine else 0,
        terminal_refinement_strength=0.025,
        score_unified_objective=True,
    )
    field_config = LearnedFieldConfig(vehicle_physics_enabled=True)
    return LegacyGuidedFlowPlanner(
        model=model,
        residual_std=torch.tensor(checkpoint["residual_std_normalized"], device=device),
        metric_scales=torch.tensor(checkpoint["metric_scales"], device=device),
        config=planner_config,
        guidance=guidance,
        terrain_field_model=field_model,
        learned_field_config=field_config,
    ).to(device).eval()


def collapse_history(chunks: list[np.ndarray]) -> tuple[list[float], list[float], int]:
    flattened = [chunk.reshape(-1, chunk.shape[-1]) for chunk in chunks]
    values = np.concatenate(flattened, axis=0)
    return values.mean(axis=0).tolist(), values.std(axis=0).tolist(), len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-cache", type=Path, required=True)
    parser.add_argument("--perception-cache", type=Path, required=True)
    parser.add_argument("--reference-comparison", type=Path, required=True)
    parser.add_argument("--flow-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--field-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--integration-steps", type=int, default=16)
    parser.add_argument("--maximum-time-s", type=float, default=10.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Rellis3DSceneDataset(args.trajectory_cache, "test")
    reference = json.loads(
        (args.reference_comparison / "comparison_report.json").read_text(encoding="utf-8")
    )
    reference_indices = np.asarray(reference["indices"], dtype=int)
    goal_distance = np.linalg.norm(
        np.asarray(dataset.source.goal[reference_indices, :2], dtype=np.float32), axis=1
    )
    eligible = reference_indices[(goal_distance >= 6.0) & (goal_distance <= 23.5)]
    if len(eligible) < args.samples:
        raise ValueError(f"Only {len(eligible)} eligible scenes for {args.samples} requested")
    indices = eligible[np.linspace(0, len(eligible) - 1, args.samples, dtype=int)]
    risk_all = np.load(args.perception_cache / "test" / "risk_target.npy", mmap_mode="r")
    mask_all = np.load(args.perception_cache / "test" / "supervision_mask.npy", mmap_mode="r")
    field_model, field_checkpoint = load_field(args.field_checkpoint, device)
    vehicle = BicycleConfig()

    rows: list[dict] = []
    process_rows: list[dict] = []
    first_seed = None
    trace_payload: dict[str, np.ndarray] = {"index": indices[:8]}
    for checkpoint_path in args.flow_checkpoints:
        model, checkpoint = load_flow(checkpoint_path, device)
        seed = int(checkpoint["config"]["training"]["seed"])
        first_seed = seed if first_seed is None else first_seed
        print(f"seed={seed} checkpoint={checkpoint_path}", flush=True)
        for method in METHODS:
            planner = make_mechanism_planner(
                method, model, checkpoint, field_model,
                args.candidates, args.integration_steps, device,
            )
            selection = (
                "first"
                if method in {"Unguided Flow", "Guided Flow (no reranking)"}
                else "score"
            )
            diagnostics: dict[str, list[np.ndarray]] = {}
            method_traces = []
            for start in range(0, len(indices), args.batch_size):
                batch_indices = indices[start : start + args.batch_size]
                terrain = np.asarray(dataset.source.bev[batch_indices], dtype=np.float32) / 255.0
                goals = np.asarray(dataset.source.goal[batch_indices], dtype=np.float32)
                gt = np.asarray(dataset.source.trajectory[batch_indices], dtype=np.float32)
                true_risk = np.asarray(risk_all[batch_indices, 0], dtype=np.float32) / 255.0
                true_mask = np.asarray(mask_all[batch_indices, 0], dtype=np.float32)
                batch_rows, traces = rollout_batch(
                    planner,
                    terrain,
                    goals,
                    gt,
                    true_risk,
                    true_mask,
                    batch_indices,
                    seed,
                    method,
                    device,
                    vehicle,
                    args.maximum_time_s,
                    selection_policy=selection,
                    diagnostics_sink=diagnostics,
                )
                rows.extend(batch_rows)
                if seed == first_seed:
                    method_traces.extend(traces)
                print(
                    f"  {method}: {min(start + len(batch_indices), len(indices))}/{len(indices)}",
                    flush=True,
                )
            mean, std, observations = collapse_history(
                diagnostics["feasibility_cost_history"]
            )
            for flow_step, (mean_value, std_value) in enumerate(zip(mean, std)):
                process_rows.append({
                    "seed": seed,
                    "method": method,
                    "phase": "flow",
                    "step": flow_step,
                    "cost_mean": mean_value,
                    "cost_std": std_value,
                    "candidate_replans": observations,
                })
            if "refinement_cost_history" in diagnostics:
                mean, std, observations = collapse_history(
                    diagnostics["refinement_cost_history"]
                )
                for refine_step, (mean_value, std_value) in enumerate(zip(mean, std)):
                    process_rows.append({
                        "seed": seed,
                        "method": method,
                        "phase": "refinement",
                        "step": refine_step,
                        "cost_mean": mean_value,
                        "cost_std": std_value,
                        "candidate_replans": observations,
                    })
            if method_traces:
                maximum = int(args.maximum_time_s / vehicle.control_dt_s) + 1
                padded = np.full((min(8, len(method_traces)), maximum, 4), np.nan, dtype=np.float32)
                for index, trace in enumerate(method_traces[:8]):
                    padded[index, : min(len(trace), maximum)] = trace[:maximum]
                trace_payload[method] = padded
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with (args.output_dir / "per_episode_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    seed_rows, method_report = summarize_with_methods(rows)
    with (args.output_dir / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)
    with (args.output_dir / "flow_cost_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(process_rows[0]))
        writer.writeheader()
        writer.writerows(process_rows)
    np.savez_compressed(args.output_dir / "closed_loop_examples.npz", **trace_payload)
    report = {
        "status": "complete",
        "purpose": "separate in-flow guidance from generate-then-filter and generate-then-refine",
        "methods": method_report,
        "seeds": sorted({row["seed"] for row in rows}),
        "samples_per_seed": len(indices),
        "paired_initial_noise": True,
        "shared_terminal_objective": True,
        "objective_config": mechanism_objective().__dict__,
        "learned_field_config": LearnedFieldConfig(vehicle_physics_enabled=True).__dict__,
        "vehicle_conditioned_field": {
            "current_state": ["speed", "ego-relative heading"],
            "trajectory_state": ["waypoint speed", "tangent heading"],
            "terrain_terms": [
                "learned base risk", "learned speed sensitivity",
                "vehicle-width clearance", "longitudinal grade", "cross slope",
            ],
            "dynamics_terms": ["second difference", "curvature", "steering rate"],
        },
        "flow_integration_steps": args.integration_steps,
        "terminal_refinement_steps": 8,
        "field_checkpoint": str(args.field_checkpoint),
        "field_validation": {
            key: field_checkpoint[key]
            for key in ("val_loss", "val_mae", "val_brier", "val_iou")
        },
        "sample_selection": {
            "goal_distance_gate_m": [6.0, 23.5],
            "eligible_before_stratification": len(eligible),
            "indices": indices.tolist(),
        },
        "limitations": [
            "recorded-map receding-horizon replay, not a physical simulator",
            "speed-sensitivity supervision is a physics-informed proxy",
            "z consistency, tire friction, roll and pitch are not directly supervised",
        ],
    }
    (args.output_dir / "mechanism_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def summarize_with_methods(rows):
    numeric = [key for key in rows[0] if key not in {"seed", "index", "method"}]
    seed_summary = []
    for seed in sorted({row["seed"] for row in rows}):
        for method in METHODS:
            subset = [row for row in rows if row["seed"] == seed and row["method"] == method]
            seed_summary.append({
                "seed": seed,
                "method": method,
                **{key: float(np.mean([row[key] for row in subset])) for key in numeric},
            })
    report = {}
    for method in METHODS:
        subset = [row for row in seed_summary if row["method"] == method]
        report[method] = {
            key: {
                "seed_mean": float(np.mean([row[key] for row in subset])),
                "seed_std": (
                    float(np.std([row[key] for row in subset], ddof=1))
                    if len(subset) > 1
                    else 0.0
                ),
            }
            for key in numeric
        }
    return seed_summary, report


if __name__ == "__main__":
    main()

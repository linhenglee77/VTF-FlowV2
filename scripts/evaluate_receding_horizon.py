"""Multi-seed receding-horizon bicycle evaluation of VTF-Flow variants."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from TerraFlow.closed_loop.receding_horizon import (
    BicycleConfig,
    bicycle_step,
    local_path_to_world,
    pure_pursuit_control,
    warp_local_bev,
    world_goal_to_local,
)
from TerraFlow.datasets.rellis3d import Rellis3DSceneDataset
from TerraFlow.guidance.feasibility_guidance import FlowGuidanceConfig
from TerraFlow.interfaces import SceneBatch
from TerraFlow.models.legacy_transformer_flow import LegacyConditionalTrajectoryFlow
from TerraFlow.planners.legacy_guided_flow_planner import FlowPlanner, FlowPlannerConfig
from TerraFlow.terrain.learned_feasibility_field import FeasibilityFieldNet


METHODS = (
    "Flow",
    "Analytic-guided Flow",
    "Learned-guided Flow",
    "VTF-Flow",
)


def load_flow(path: Path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = LegacyConditionalTrajectoryFlow(**checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def load_field(path: Path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = FeasibilityFieldNet(**checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


def make_planner(method, model, checkpoint, field_model, device, candidates, steps):
    base = FlowGuidanceConfig(
        enabled=False, strength=0.0, schedule="late", terrain_weight=1.0,
        occupancy_weight=0.0, smoothness_weight=0.0, curvature_weight=0.0,
        steering_rate_weight=0.0, boundary_weight=0.2, gradient_clip=4.0,
        vehicle_conditioned=True, planning_dt_s=0.5,
    )
    learned = None
    if method == "Analytic-guided Flow":
        guidance = replace(base, enabled=True, strength=0.5)
    elif method == "Learned-guided Flow":
        guidance = replace(base, enabled=True, strength=0.18)
        learned = field_model
    elif method == "VTF-Flow":
        guidance = replace(
            base, enabled=True, strength=0.18, smoothness_weight=0.03,
            curvature_weight=0.08, steering_rate_weight=0.04,
        )
        learned = field_model
    else:
        guidance = base
    return FlowPlanner(
        model=model,
        residual_std=torch.tensor(checkpoint["residual_std_normalized"], device=device),
        metric_scales=torch.tensor(checkpoint["metric_scales"], device=device),
        config=FlowPlannerConfig(candidates=candidates, integration_steps=steps),
        guidance=guidance,
        terrain_field_model=learned,
    ).to(device).eval()


def bilinear(grid: np.ndarray, x: float, y: float, outside: float) -> float:
    if x < 0 or x >= 24.0 or y < -12.0 or y >= 12.0:
        return outside
    row = x / 24.0 * 64 - 0.5
    col = (y + 12.0) / 24.0 * 64 - 0.5
    r0, c0 = int(np.floor(row)), int(np.floor(col))
    dr, dc = row - r0, col - c0
    value = 0.0
    for rr, wr in ((r0, 1 - dr), (r0 + 1, dr)):
        for cc, wc in ((c0, 1 - dc), (c0 + 1, dc)):
            value += wr * wc * float(grid[np.clip(rr, 0, 63), np.clip(cc, 0, 63)])
    return value


def rollout_batch(
    planner,
    terrain: np.ndarray,
    goals: np.ndarray,
    gt: np.ndarray,
    true_risk: np.ndarray,
    true_mask: np.ndarray,
    indices: np.ndarray,
    seed: int,
    method: str,
    device,
    vehicle: BicycleConfig,
    maximum_time_s: float,
    selection_policy: str = "score",
    diagnostics_sink: dict[str, list[np.ndarray]] | None = None,
):
    batch = len(indices)
    initial_bev = torch.from_numpy(terrain).to(device)
    states = np.zeros((batch, 4), dtype=np.float32)
    previous_steering = np.zeros(batch, dtype=np.float32)
    done = np.zeros(batch, dtype=bool)
    out_of_bounds = np.zeros(batch, dtype=bool)
    replans = np.zeros(batch, dtype=int)
    planning_ms = np.zeros(batch, dtype=float)
    traces = [[states[i].copy()] for i in range(batch)]
    risks, knowns, steerings, accelerations = ([[] for _ in range(batch)] for _ in range(4))
    max_replans = int(math.ceil(maximum_time_s / vehicle.replan_interval_s))
    for replan in range(max_replans):
        active = np.flatnonzero(~done)
        if len(active) == 0:
            break
        pose = torch.from_numpy(states[active]).to(device)
        local_bev, _ = warp_local_bev(initial_bev[active], pose)
        local_goal_np = world_goal_to_local(goals[active], states[active])
        local_goal_np[:, 0] = np.clip(local_goal_np[:, 0], 0.25, 24.0)
        local_goal_np[:, 1] = np.clip(local_goal_np[:, 1], -12.0, 12.0)
        local_goal = torch.from_numpy(local_goal_np).to(device)
        scene = SceneBatch(
            ego_history=torch.zeros(len(active), 1, 3, device=device),
            gt_future=torch.zeros(len(active), 30, 3, device=device),
            goal=local_goal,
            point_cloud=None,
            semantic_labels=None,
            terrain_map=local_bev,
            metadata=[{"index": int(indices[i]), "replan": replan} for i in active],
            vehicle_state={
                "speed": pose[:, 3],
                "heading": torch.zeros(len(active), device=device, dtype=pose.dtype),
            },
        )
        torch.manual_seed(seed * 100_000 + int(indices[0]) * 31 + replan)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        prediction = planner(scene)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if diagnostics_sink is not None and prediction.diagnostics is not None:
            for key, value in prediction.diagnostics.items():
                array = value.detach().cpu().numpy()
                diagnostics_sink.setdefault(key, []).append(array)
                if replan == 0:
                    diagnostics_sink.setdefault(f"{key}_first_replan", []).append(array)
        if selection_policy == "first":
            selected_index = torch.zeros(len(active), device=device, dtype=torch.long)
        elif selection_policy == "score":
            selected_index = prediction.scores.argmin(dim=1)
        else:
            raise ValueError(f"Unknown selection policy: {selection_policy}")
        selected = prediction.trajectories[
            torch.arange(len(active), device=device), selected_index
        ].detach().cpu().numpy()
        world_paths = [
            local_path_to_world(selected[local], states[global_index])
            for local, global_index in enumerate(active)
        ]
        for global_index in active:
            replans[global_index] += 1
            planning_ms[global_index] += elapsed_ms / len(active)
        for _ in range(vehicle.controls_per_plan):
            for local, global_index in enumerate(active):
                if done[global_index]:
                    continue
                steering, acceleration = pure_pursuit_control(
                    states[global_index], world_paths[local], goals[global_index],
                    float(previous_steering[global_index]), vehicle,
                )
                states[global_index] = bicycle_step(
                    states[global_index], steering, acceleration, vehicle
                )
                previous_steering[global_index] = steering
                traces[global_index].append(states[global_index].copy())
                steerings[global_index].append(steering)
                accelerations[global_index].append(acceleration)
                x, y = states[global_index, :2]
                risks[global_index].append(bilinear(true_risk[global_index], x, y, outside=1.0))
                knowns[global_index].append(bilinear(true_mask[global_index], x, y, outside=0.0))
                outside = not (0.0 <= x < 24.0 and -12.0 <= y < 12.0)
                reached = np.linalg.norm(states[global_index, :2] - goals[global_index, :2]) < 1.5
                out_of_bounds[global_index] |= outside
                done[global_index] |= outside or reached
    rows = []
    for local, index in enumerate(indices.tolist()):
        risk_values = np.asarray(risks[local], dtype=float)
        known_values = np.asarray(knowns[local], dtype=float)
        steering_values = np.asarray(steerings[local], dtype=float)
        acceleration_values = np.asarray(accelerations[local], dtype=float)
        trace = np.asarray(traces[local], dtype=np.float32)
        final_error = float(np.linalg.norm(states[local, :2] - goals[local, :2]))
        known_obstacle = (risk_values >= 0.5) & (known_values >= 0.5)
        rows.append({
            "seed": seed,
            "index": index,
            "method": method,
            "completion": float(final_error < 1.5 and not out_of_bounds[local]),
            "collision": float(bool(known_obstacle.any())),
            "final_goal_error_m": final_error,
            "known_obstacle_exposure": float(known_obstacle.mean()) if len(risk_values) else 1.0,
            "unknown_fraction": float((known_values < 0.5).mean()) if len(known_values) else 1.0,
            "conservative_risk": float((risk_values * known_values + 0.5 * (1 - known_values)).mean()) if len(risk_values) else 1.0,
            "executed_path_length_m": float(np.linalg.norm(np.diff(trace[:, :2], axis=0), axis=1).sum()),
            "mean_abs_steering_deg": float(np.degrees(np.abs(steering_values)).mean()) if len(steering_values) else 0.0,
            "max_abs_steering_deg": float(np.degrees(np.abs(steering_values)).max()) if len(steering_values) else 0.0,
            "steering_rate_violation": float((np.abs(np.diff(np.r_[0.0, steering_values])) / vehicle.control_dt_s > math.radians(vehicle.maximum_steering_rate_deg_s) + 1e-6).mean()) if len(steering_values) else 0.0,
            "acceleration_jerk_mps3": float((np.abs(np.diff(acceleration_values)) / vehicle.control_dt_s).mean()) if len(acceleration_values) > 1 else 0.0,
            "duration_s": float((len(trace) - 1) * vehicle.control_dt_s),
            "replans": int(replans[local]),
            "planning_time_ms_per_replan": float(planning_ms[local] / max(replans[local], 1)),
            "gt_endpoint_error_m": float(np.linalg.norm(states[local, :2] - gt[local, -1, :2])),
        })
    return rows, traces


def summarize(rows):
    numeric = [key for key in rows[0] if key not in {"seed", "index", "method"}]
    seed_summary = []
    for seed in sorted({row["seed"] for row in rows}):
        for method in METHODS:
            subset = [row for row in rows if row["seed"] == seed and row["method"] == method]
            seed_summary.append({
                "seed": seed, "method": method,
                **{key: float(np.mean([row[key] for row in subset])) for key in numeric},
            })
    report = {}
    for method in METHODS:
        subset = [row for row in seed_summary if row["method"] == method]
        report[method] = {
            key: {
                "seed_mean": float(np.mean([row[key] for row in subset])),
                "seed_std": float(np.std([row[key] for row in subset], ddof=1)) if len(subset) > 1 else 0.0,
            }
            for key in numeric
        }
    return seed_summary, report


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
    reference = json.loads((args.reference_comparison / "comparison_report.json").read_text(encoding="utf-8"))
    reference_indices = np.asarray(reference["indices"], dtype=int)
    goal_distance = np.linalg.norm(
        np.asarray(dataset.source.goal[reference_indices, :2], dtype=np.float32), axis=1
    )
    eligible = reference_indices[(goal_distance >= 6.0) & (goal_distance <= 23.5)]
    if len(eligible) < args.samples:
        raise ValueError(
            f"Only {len(eligible)} reference scenes satisfy the 6.0--23.5 m goal-distance gate"
        )
    indices = eligible[np.linspace(0, len(eligible) - 1, args.samples, dtype=int)]
    risk_all = np.load(args.perception_cache / "test" / "risk_target.npy", mmap_mode="r")
    mask_all = np.load(args.perception_cache / "test" / "supervision_mask.npy", mmap_mode="r")
    field_model, field_checkpoint = load_field(args.field_checkpoint, device)
    vehicle = BicycleConfig()
    rows, trace_payload = [], {"index": indices[:8]}
    first_checkpoint = torch.load(args.flow_checkpoints[0], map_location="cpu", weights_only=False)
    first_seed = int(first_checkpoint["config"]["training"]["seed"])
    for checkpoint_path in args.flow_checkpoints:
        model, checkpoint = load_flow(checkpoint_path, device)
        seed = int(checkpoint["config"]["training"]["seed"])
        print(f"seed={seed} checkpoint={checkpoint_path}", flush=True)
        for method in METHODS:
            planner = make_planner(
                method, model, checkpoint, field_model, device,
                args.candidates, args.integration_steps,
            )
            method_traces = []
            for start in range(0, len(indices), args.batch_size):
                batch_indices = indices[start : start + args.batch_size]
                terrain = np.asarray(dataset.source.bev[batch_indices], dtype=np.float32) / 255.0
                goals = np.asarray(dataset.source.goal[batch_indices], dtype=np.float32)
                gt = np.asarray(dataset.source.trajectory[batch_indices], dtype=np.float32)
                true_risk = np.asarray(risk_all[batch_indices, 0], dtype=np.float32) / 255.0
                true_mask = np.asarray(mask_all[batch_indices, 0], dtype=np.float32)
                batch_rows, traces = rollout_batch(
                    planner, terrain, goals, gt, true_risk, true_mask,
                    batch_indices, seed, method, device, vehicle, args.maximum_time_s,
                )
                rows.extend(batch_rows)
                if seed == first_seed:
                    method_traces.extend(traces)
                print(
                    f"  {method}: {min(start + len(batch_indices), len(indices))}/{len(indices)}",
                    flush=True,
                )
            if method_traces:
                maximum = int(args.maximum_time_s / vehicle.control_dt_s) + 1
                padded = np.full((min(8, len(method_traces)), maximum, 4), np.nan, dtype=np.float32)
                for i, trace in enumerate(method_traces[:8]):
                    padded[i, : min(len(trace), maximum)] = trace[:maximum]
                trace_payload[method] = padded
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    with (args.output_dir / "per_episode_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    seed_rows, method_report = summarize(rows)
    with (args.output_dir / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)
    np.savez_compressed(args.output_dir / "closed_loop_examples.npz", **trace_payload)
    report = {
        "status": "complete",
        "samples_per_seed": len(indices),
        "sample_selection": {
            "source": "fixed comparison-report test indices",
            "goal_distance_gate_m": [6.0, 23.5],
            "eligible_before_stratification": len(eligible),
            "selection": "deterministic evenly spaced positions in eligible order",
            "indices": indices.tolist(),
        },
        "seeds": sorted({row["seed"] for row in rows}),
        "methods": method_report,
        "paired_initial_noise": True,
        "receding_horizon": {
            "observation_refresh": "static initial semantic BEV re-rendered in current ego frame",
            "replan_interval_s": vehicle.replan_interval_s,
            "control_dt_s": vehicle.control_dt_s,
            "maximum_time_s": args.maximum_time_s,
            "candidates": args.candidates,
            "integration_steps": args.integration_steps,
        },
        "vehicle": vehicle.__dict__,
        "field_checkpoint": str(args.field_checkpoint),
        "field_validation": {key: field_checkpoint[key] for key in ("val_loss", "val_mae", "val_brier", "val_iou")},
        "limitations": [
            "local static-map replay, not a photorealistic or hardware simulator",
            "new observations are geometric re-renderings of one recorded frame",
            "semantic risk exposure is not physical collision ground truth",
        ],
    }
    (args.output_dir / "closed_loop_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

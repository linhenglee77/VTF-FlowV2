"""Paired evaluation of unguided and feasibility-guided trajectory Flow."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from TerraFlow.datasets.rellis3d import Rellis3DSceneDataset, collate_scenes
from TerraFlow.evaluation.evaluator import TerraFlowEvaluator
from TerraFlow.guidance.feasibility_guidance import FlowGuidanceConfig
from TerraFlow.models.legacy_transformer_flow import (
    LegacyConditionalTrajectoryFlow as ConditionalTrajectoryFlow,
)
from TerraFlow.planners.legacy_guided_flow_planner import FlowPlanner, FlowPlannerConfig


METHODS = ("Flow", "Feasibility-guided Flow")


def load_model(checkpoint_path: Path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ConditionalTrajectoryFlow(**checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-comparison", type=Path, required=True)
    parser.add_argument("--example-indices", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--integration-steps", type=int, default=16)
    parser.add_argument("--guidance-strength", type=float, default=0.12)
    parser.add_argument("--smoothness-weight", type=float, default=0.0)
    parser.add_argument("--curvature-weight", type=float, default=0.0)
    parser.add_argument("--steering-rate-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7301)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Rellis3DSceneDataset(args.cache_root, "test")
    reference = json.loads(
        (args.reference_comparison / "comparison_report.json").read_text(encoding="utf-8")
    )
    indices = np.asarray(reference["indices"][: args.samples], dtype=int)
    if args.example_indices is not None:
        example_set = set(np.load(args.example_indices)["index"].astype(int).tolist())
    else:
        example_set = set(indices[: min(40, len(indices))].tolist())
    model, checkpoint = load_model(args.checkpoint, device)
    base_guidance = FlowGuidanceConfig(**checkpoint["config"]["guidance"])
    base_guidance = replace(
        base_guidance,
        smoothness_weight=args.smoothness_weight,
        curvature_weight=args.curvature_weight,
        steering_rate_weight=args.steering_rate_weight,
    )
    planner_config = FlowPlannerConfig(
        candidates=args.candidates,
        integration_steps=args.integration_steps,
        anchor_endpoint=True,
        save_integration_history=True,
    )
    residual_std = torch.tensor(checkpoint["residual_std_normalized"], device=device)
    scales = torch.tensor(checkpoint["metric_scales"], device=device)
    planners = {
        "Flow": FlowPlanner(
            model, residual_std, scales, planner_config, replace(base_guidance, enabled=False)
        ).to(device),
        "Feasibility-guided Flow": FlowPlanner(
            model, residual_std, scales, planner_config,
            replace(base_guidance, enabled=True, strength=args.guidance_strength),
        ).to(device),
    }
    evaluator = TerraFlowEvaluator()
    rows = []
    saved = {"index": [], "terrain_map": [], "gt": []}
    for method in METHODS:
        saved[f"{method} candidates"] = []
        saved[f"{method} selected"] = []
        saved[f"{method} integration"] = []

    for batch_start in range(0, len(indices), args.batch_size):
        batch_indices = indices[batch_start : batch_start + args.batch_size]
        scene = collate_scenes([dataset[int(index)] for index in batch_indices]).to(device)
        predictions = {}
        evaluations = {}
        runtimes = {}
        for method, planner in planners.items():
            torch.manual_seed(args.seed + batch_start)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            prediction = planner(scene)
            if device.type == "cuda":
                torch.cuda.synchronize()
            runtimes[method] = (time.perf_counter() - started) * 1000 / len(batch_indices)
            predictions[method] = prediction
            evaluations[method] = evaluator(prediction, scene)
        for local, index in enumerate(batch_indices.tolist()):
            for method in METHODS:
                evaluation = evaluations[method]
                row = {"index": index, "method": method, "planning_time_ms": runtimes[method]}
                for key, value in evaluation.items():
                    if key == "selected_index" or not torch.is_tensor(value):
                        continue
                    row[key] = float(value[local])
                rows.append(row)
            if index in example_set:
                saved["index"].append(index)
                saved["terrain_map"].append(scene.terrain_map[local].cpu().numpy())
                saved["gt"].append(scene.gt_future[local].cpu().numpy())
                for method in METHODS:
                    prediction = predictions[method]
                    selected = int(evaluations[method]["selected_index"][local])
                    saved[f"{method} candidates"].append(
                        prediction.trajectories[local].detach().cpu().numpy()
                    )
                    saved[f"{method} selected"].append(
                        prediction.trajectories[local, selected].detach().cpu().numpy()
                    )
                    saved[f"{method} integration"].append(
                        prediction.integration_history[local, selected].detach().cpu().numpy()
                    )
        print(f"processed {min(batch_start + len(batch_indices), len(indices))}/{len(indices)}", flush=True)

    with (args.output_dir / "per_sample_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        args.output_dir / "flow_examples.npz",
        **{key: np.asarray(value) for key, value in saved.items()},
    )
    numeric = [key for key in rows[0] if key not in {"index", "method"}]
    summary = {}
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        summary[method] = {
            key: {
                "mean": float(np.mean([row[key] for row in subset])),
                "median": float(np.median([row[key] for row in subset])),
            }
            for key in numeric
        }
    report = {
        "status": "complete",
        "samples": len(indices),
        "paired_initial_noise": True,
        "candidates": args.candidates,
        "integration_steps": args.integration_steps,
        "guidance_strength": args.guidance_strength,
        "guidance_components": {
            "terrain_weight": base_guidance.terrain_weight,
            "smoothness_weight": base_guidance.smoothness_weight,
            "curvature_weight": base_guidance.curvature_weight,
            "steering_rate_weight": base_guidance.steering_rate_weight,
            "boundary_weight": base_guidance.boundary_weight,
        },
        "guidance_equation": "dx/dt = v_theta(x,t,c) - eta(t) normalized_gradient(J_feasibility)",
        "selection": "minimum analytic vehicle-conditioned terrain cost for both methods",
        "methods": summary,
        "limitations": [
            "one Flow training seed",
            "analytic feasibility field is derived from a sparse current-frame BEV",
            "no observation refresh or receding-horizon replanning",
        ],
    }
    (args.output_dir / "flow_guidance_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

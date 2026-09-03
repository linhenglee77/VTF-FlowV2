"""Audit feasibility cost inside the first Flow solve under paired initial noise."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from TerraFlow.datasets.rellis3d import Rellis3DSceneDataset
from TerraFlow.interfaces import SceneBatch
from TerraFlow.scripts.evaluate_guidance_mechanisms import METHODS, make_mechanism_planner
from TerraFlow.scripts.evaluate_receding_horizon import load_field, load_flow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-cache", type=Path, required=True)
    parser.add_argument("--reference-comparison", type=Path, required=True)
    parser.add_argument("--flow-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--field-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--integration-steps", type=int, default=16)
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
    indices = eligible[np.linspace(0, len(eligible) - 1, args.samples, dtype=int)]
    field_model, _ = load_field(args.field_checkpoint, device)
    rows = []
    paired_report = {}

    for checkpoint_path in args.flow_checkpoints:
        model, checkpoint = load_flow(checkpoint_path, device)
        seed = int(checkpoint["config"]["training"]["seed"])
        terminal_by_method = {}
        for method in METHODS:
            planner = make_mechanism_planner(
                method, model, checkpoint, field_model,
                args.candidates, args.integration_steps, device,
            )
            flow_chunks, refine_chunks, score_chunks = [], [], []
            for start in range(0, len(indices), args.batch_size):
                batch_indices = indices[start : start + args.batch_size]
                terrain = torch.from_numpy(
                    np.asarray(dataset.source.bev[batch_indices], dtype=np.float32) / 255.0
                ).to(device)
                goals = torch.from_numpy(
                    np.asarray(dataset.source.goal[batch_indices], dtype=np.float32)
                ).to(device)
                scene = SceneBatch(
                    ego_history=torch.zeros(len(batch_indices), 1, 3, device=device),
                    gt_future=torch.zeros(len(batch_indices), 30, 3, device=device),
                    goal=goals,
                    point_cloud=None,
                    semantic_labels=None,
                    terrain_map=terrain,
                    metadata=[{"index": int(index)} for index in batch_indices],
                    vehicle_state={
                        "speed": torch.zeros(len(batch_indices), device=device),
                        "heading": torch.zeros(len(batch_indices), device=device),
                    },
                )
                torch.manual_seed(seed * 100_000 + int(batch_indices[0]) * 31)
                prediction = planner(scene)
                score_chunks.append(prediction.scores.detach().cpu().numpy())
                flow_chunks.append(
                    prediction.diagnostics["feasibility_cost_history"].detach().cpu().numpy()
                )
                if "refinement_cost_history" in prediction.diagnostics:
                    refine_chunks.append(
                        prediction.diagnostics["refinement_cost_history"].detach().cpu().numpy()
                    )
            flow = np.concatenate(flow_chunks, axis=0)
            # Planner scores use the shared objective after the final Flow
            # update and, when applicable, after terminal refinement.
            terminal_by_method[method] = np.concatenate(score_chunks, axis=0)
            for step in range(flow.shape[-1]):
                values = flow[:, :, step].reshape(-1)
                rows.append({
                    "seed": seed,
                    "method": method,
                    "phase": "flow",
                    "step": step,
                    "cost_mean": float(values.mean()),
                    "cost_std_across_candidates": float(values.std()),
                    "candidate_count": len(values),
                })
            if refine_chunks:
                refine = np.concatenate(refine_chunks, axis=0)
                for step in range(refine.shape[-1]):
                    values = refine[:, :, step].reshape(-1)
                    rows.append({
                        "seed": seed,
                        "method": method,
                        "phase": "refinement",
                        "step": step,
                        "cost_mean": float(values.mean()),
                        "cost_std_across_candidates": float(values.std()),
                        "candidate_count": len(values),
                    })
        unguided = terminal_by_method["Unguided Flow"]
        for method, terminal in terminal_by_method.items():
            paired_report.setdefault(method, []).append({
                "seed": seed,
                "terminal_cost_mean": float(terminal.mean()),
                "paired_improvement_mean": float((unguided - terminal).mean()),
                "fraction_better_than_paired_unguided": float((terminal < unguided).mean()),
            })
        del model

    with (args.output_dir / "first_replan_flow_evolution.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paired_rows = [
        {"method": method, **value}
        for method, values in paired_report.items()
        for value in values
    ]
    with (args.output_dir / "first_replan_paired_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    summary = {"samples_per_seed": len(indices), "candidates": args.candidates, "methods": {}}
    for method, values in paired_report.items():
        summary["methods"][method] = {
            key: {
                "seed_mean": float(np.mean([value[key] for value in values])),
                "seed_std": float(np.std([value[key] for value in values], ddof=1)),
            }
            for key in (
                "terminal_cost_mean",
                "paired_improvement_mean",
                "fraction_better_than_paired_unguided",
            )
        }
    (args.output_dir / "first_replan_flow_evolution.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

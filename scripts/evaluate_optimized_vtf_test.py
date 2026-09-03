"""Run the frozen validation-selected VTF sampler once on the test sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset


TERRAFLOW_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TERRAFLOW_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from TerraFlow.evaluation.final_experiments import (  # noqa: E402
    partition_sequence_indices,
    terminal_scene_metrics,
    write_csv,
)
from TerraFlow.guidance.feasibility_flow_guidance import FeasibilityFlowGuidanceConfig  # noqa: E402
from TerraFlow.metrics.trajectory_metrics import trajectory_metrics  # noqa: E402
from TerraFlow.planners.flow_planner import FlowPlannerConfig  # noqa: E402
from TerraFlow.planners.guided_flow_planner import GuidedFlowPlanner  # noqa: E402
from TerraFlow.scripts.optimize_vtf_flow_validation import _fixed_noise  # noqa: E402
from TerraFlow.scripts.run_final_experiments import _load_flow  # noqa: E402
from TerraFlow.scripts.run_unified_h10_benchmark import (  # noqa: E402
    DEFAULT_CACHE, DEFAULT_CONFIG, DEFAULT_DATA, H10PlanningDataset,
    benchmark_split, flow_training_config, guidance_config, load_json,
)
from TerraFlow.scripts.train_regression import CombinedSceneDataset, make_loader  # noqa: E402
from TerraFlow.terrain.feasibility_field import TerrainFieldConfig  # noqa: E402
from TerraFlow.terrain.trajectory_kinematics import TrajectoryKinematicConfig  # noqa: E402
from TerraFlow.terrain.vehicle_conditioned_field import VehicleConditionedFieldConfig  # noqa: E402


DEFAULT_OPTIMIZED = TERRAFLOW_ROOT / "configs" / "optimized_vtf_flow_test.json"
DEFAULT_BENCHMARK_ROOT = TERRAFLOW_ROOT / "outputs" / "unified_h10_benchmark"
DEFAULT_OUTPUT = TERRAFLOW_ROOT / "outputs" / "unified_h10_benchmark_optimized"


def _planner(
    checkpoint: Path,
    benchmark: Mapping[str, Any],
    optimized: Mapping[str, Any],
    seed: int,
    device: torch.device,
) -> GuidedFlowPlanner:
    model = _load_flow(checkpoint, device)
    plan = FlowPlannerConfig(
        candidates=int(optimized["sampling"]["candidates"]),
        integration_steps=int(optimized["sampling"]["integration_steps"]),
        save_integration_history=False,
    )
    flow_cfg = flow_training_config(benchmark, seed, tvk=True)
    terrain = TerrainFieldConfig(**flow_cfg["terrain_field"])
    vehicle = VehicleConditionedFieldConfig(**flow_cfg["vehicle_conditioning"])
    base = guidance_config(benchmark, use_kinematics=True)
    values = dict(base.__dict__)
    values.update(
        strength=float(optimized["guidance"]["eta"]),
        schedule=str(optimized["guidance"]["schedule"]),
        gamma=float(optimized["guidance"]["gamma"]),
        smoothing_kernel=str(optimized["guidance"]["smoothing_kernel"]),
        endpoint_projection=str(optimized["guidance"]["endpoint_projection"]),
        adaptive_trigger_enabled=bool(
            optimized["guidance"]["adaptive_trigger_enabled"]
        ),
    )
    return GuidedFlowPlanner(
        model, plan, FeasibilityFlowGuidanceConfig(**values), terrain, vehicle
    ).to(device)


def _evaluate_seed(
    *,
    seed: int,
    planner: GuidedFlowPlanner,
    dataset: H10PlanningDataset,
    test_indices: list[int],
    benchmark: Mapping[str, Any],
    optimized: Mapping[str, Any],
    output_dir: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = make_loader(
        Subset(dataset, test_indices), batch_size, shuffle=False,
        seed=seed + 601, num_workers=0,
    )
    flow_cfg = flow_training_config(benchmark, seed, tvk=True)
    terrain = TerrainFieldConfig(**flow_cfg["terrain_field"])
    vehicle = VehicleConditionedFieldConfig(**flow_cfg["vehicle_conditioning"])
    kinematic = TrajectoryKinematicConfig(**benchmark["kinematic"])
    candidates = int(optimized["sampling"]["candidates"])
    horizon = int(benchmark["trajectory"]["horizon_steps"])
    rows: list[dict[str, Any]] = []
    trajectory_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    position = 0
    for scene in loader:
        scene = scene.to(device)
        batch = scene.batch_size
        positions = list(range(position, position + batch))
        noise = _fixed_noise(
            positions, seed, candidates, horizon, device,
            str(optimized["sampling"]["noise_protocol"]),
        )
        with torch.enable_grad():
            prediction = planner.sample(scene, noise)
        standard = trajectory_metrics(prediction.trajectories, scene.gt_future)
        terminal = terminal_scene_metrics(
            prediction.trajectories, scene.gt_future, scene.terrain_map,
            terrain, vehicle, flow_cfg["metrics"],
            planning_dt_s=float(benchmark["trajectory"]["planning_dt_s"]),
            kinematic_config=kinematic,
        )
        for local in range(batch):
            metadata = scene.metadata[local]
            row = {
                "scene_id": f"{str(metadata['sequence']).zfill(5)}:{int(metadata.get('frame_id', metadata.get('frame_index')))}:{metadata.get('split', 'unknown')}",
                "sequence": str(metadata["sequence"]).zfill(5),
                "frame_id": int(metadata.get("frame_id", metadata.get("frame_index"))),
                "dataset_index": int(test_indices[position + local]),
                "method": "VTF_OPT",
                "seed": seed,
                "K": candidates,
                "ADE_candidate0_m": float(standard["ADE_by_candidate_m"][local, 0]),
                "FDE_candidate0_m": float(standard["FDE_by_candidate_m"][local, 0]),
            }
            row.update({name: float(value[local]) for name, value in terminal.items()})
            rows.append(row)
        trajectory_chunks.append(prediction.trajectories.detach().cpu().numpy())
        target_chunks.append(scene.gt_future.detach().cpu().numpy())
        position += batch
        if position % 256 < batch:
            print(f"optimized VTF seed={seed}: {position}/{len(test_indices)}", flush=True)
    if len(rows) != len(test_indices):
        raise AssertionError("test evaluation did not cover every scene")
    write_csv(output_dir / "scene_level_metrics.csv", rows)
    np.savez_compressed(
        output_dir / "predictions.npz",
        trajectories=np.concatenate(trajectory_chunks),
        ground_truth=np.concatenate(target_chunks),
    )
    excluded = {"scene_id", "sequence", "frame_id", "dataset_index", "method", "seed"}
    summary: dict[str, Any] = {
        "method": "VTF_OPT", "display_name": "VTF-Flow w/o TVK training (optimized guidance)",
        "seed": seed, "K": candidates, "evaluated_scenes": len(rows),
    }
    for name in rows[0]:
        if name not in excluded:
            summary[name] = float(np.mean([float(row[name]) for row in rows]))
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else float("nan")


def _updated_table(records: list[Mapping[str, Any]], benchmark_root: Path) -> pd.DataFrame:
    source = pd.read_csv(benchmark_root / "unified_summary.csv")
    keep = source[source["method"].isin(["CV", "ASTAR", "REG", "FLOW"])].copy()
    row: dict[str, Any] = {
        "method": "VTF_OPT", "display_name": "VTF-Flow (optimized)",
        "K": 8, "n_seeds": len(records), "evaluated_scenes": 1909,
    }
    metrics = (
        "ADE_candidate0_m", "minADE@K_m", "minFDE@K_m", "diversity_m",
        "mean_unified_tvk_cost", "terrain_violation_rate",
        "curvature_violation_rate", "smoothness_m",
    )
    for metric in metrics:
        mean, sd = _mean_sd([float(record[metric]) for record in records])
        row[f"{metric}_mean"] = mean
        row[f"{metric}_sd"] = sd
    combined = pd.concat((keep, pd.DataFrame([row])), ignore_index=True, sort=False)
    return combined


def _paper_table(summary: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "CV": "Constant Velocity", "ASTAR": "A* terrain planner",
        "REG": "Deterministic regression", "FLOW": "Flow Matching",
        "VTF_OPT": "VTF-Flow w/o TVK training (optimized guidance)",
    }
    rows = []
    for _, source in summary.iterrows():
        rows.append({
            "方法": labels[str(source["method"])],
            "K": int(source["K"]),
            "ADE-0": float(source["ADE_candidate0_m_mean"]),
            "minADE@K": float(source["minADE@K_m_mean"]),
            "多样性": float(source.get("diversity_m_mean", 0.0)),
            "TVK代价": float(source["mean_unified_tvk_cost_mean"]),
            "地形违规率": float(source["terrain_violation_rate_mean"]),
            "曲率违规率": float(source["curvature_violation_rate_mean"]),
            "平顺性": float(source["smoothness_m_mean"]),
        })
    return pd.DataFrame(rows)


def _markdown_table(table: pd.DataFrame) -> str:
    """Render the compact paper table without optional pandas dependencies."""

    columns = list(table.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---:" if index else "---" for index, _ in enumerate(columns)) + "|",
    ]
    for _, row in table.iterrows():
        values = []
        for name in columns:
            value = row[name]
            values.append(f"{float(value):.4f}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--optimized-config", type=Path, default=DEFAULT_OPTIMIZED)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-test-scenes", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark = load_json(args.benchmark_config)
    optimized = load_json(args.optimized_config)
    if optimized["selected_on"]["test_metrics_used_for_selection"]:
        raise ValueError("optimized configuration must not be selected on test metrics")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = CombinedSceneDataset(
        args.cache_root.resolve(), tuple(benchmark["protocol"]["source_splits"])
    )
    dataset = H10PlanningDataset(
        source, args.data_root.resolve() / "processed" / "Rellis-3D",
        horizon=int(benchmark["trajectory"]["horizon_steps"]),
        history_steps=int(benchmark["trajectory"]["history_steps"]),
    )
    indices = partition_sequence_indices(dataset.sequence_ids, benchmark_split(benchmark))
    if len(indices["test"]) != 1909:
        raise AssertionError(f"expected 1909 test scenes, got {len(indices['test'])}")
    test_indices = indices["test"]
    if args.max_test_scenes is not None:
        test_indices = test_indices[: int(args.max_test_scenes)]
    records = []
    for seed_value in optimized["seeds"]:
        seed = int(seed_value)
        checkpoint = (
            args.benchmark_root / "checkpoints" / f"seed_{seed}" / "flow" / "best.pt"
        )
        planner = _planner(checkpoint, benchmark, optimized, seed, device)
        records.append(_evaluate_seed(
            seed=seed, planner=planner, dataset=dataset,
            test_indices=test_indices, benchmark=benchmark, optimized=optimized,
            output_dir=args.output_dir / "runs" / f"VTF_OPT_seed{seed}",
            device=device, batch_size=args.batch_size,
        ))
        del planner
        if device.type == "cuda":
            torch.cuda.empty_cache()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = _updated_table(records, args.benchmark_root)
    summary.to_csv(args.output_dir / "updated_unified_summary.csv", index=False)
    table = _paper_table(summary)
    table.to_csv(args.output_dir / "table2_optimized.csv", index=False)
    (args.output_dir / "table2_optimized.md").write_text(
        _markdown_table(table), encoding="utf-8"
    )
    (args.output_dir / "table2_optimized.tex").write_text(
        table.to_latex(index=False, float_format="%.4f"), encoding="utf-8"
    )
    (args.output_dir / "effective_test_protocol.json").write_text(
        json.dumps({
            "benchmark": benchmark, "optimized": optimized,
            "test_sequences": benchmark["protocol"]["test"],
            "test_scenes": len(test_indices), "device": str(device),
            "batch_size": args.batch_size,
        }, indent=2), encoding="utf-8"
    )
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

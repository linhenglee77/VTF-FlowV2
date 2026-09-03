"""Compare IID and antithetic K-sample protocols on validation data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import Subset


TERRAFLOW_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TERRAFLOW_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from TerraFlow.evaluation.final_experiments import partition_sequence_indices  # noqa: E402
from TerraFlow.guidance.feasibility_flow_guidance import FeasibilityFlowGuidanceConfig  # noqa: E402
from TerraFlow.planners.flow_planner import FlowPlanner, FlowPlannerConfig  # noqa: E402
from TerraFlow.planners.guided_flow_planner import GuidedFlowPlanner  # noqa: E402
from TerraFlow.scripts.optimize_vtf_flow_validation import _evaluate, _write_csv  # noqa: E402
from TerraFlow.scripts.run_final_experiments import _load_flow  # noqa: E402
from TerraFlow.scripts.run_unified_h10_benchmark import (  # noqa: E402
    DEFAULT_CACHE, DEFAULT_CONFIG, DEFAULT_DATA, H10PlanningDataset,
    benchmark_split, flow_training_config, guidance_config, load_json,
)
from TerraFlow.scripts.train_regression import CombinedSceneDataset, make_loader  # noqa: E402
from TerraFlow.terrain.feasibility_field import TerrainFieldConfig  # noqa: E402
from TerraFlow.terrain.vehicle_conditioned_field import VehicleConditionedFieldConfig  # noqa: E402


DEFAULT_BENCHMARK_ROOT = TERRAFLOW_ROOT / "outputs" / "unified_h10_benchmark"
DEFAULT_VTF = (
    TERRAFLOW_ROOT / "outputs" / "vtf_flow_stage2_finetune" /
    "seed_0" / "lambda_0.25" / "best.pt"
)
DEFAULT_OUTPUT = TERRAFLOW_ROOT / "outputs" / "vtf_flow_sampling_protocols"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--vtf-checkpoint", type=Path, default=DEFAULT_VTF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-scenes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark = load_json(args.benchmark_config)
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
    available = indices["validation"]
    count = min(args.validation_scenes, len(available))
    positions = np.linspace(0, len(available) - 1, num=count, dtype=np.int64).tolist()
    selected_indices = [available[position] for position in positions]
    loader = make_loader(
        Subset(dataset, selected_indices), 8, shuffle=False, seed=433,
        num_workers=0,
    )
    plan = FlowPlannerConfig(
        candidates=int(benchmark["sampling"]["candidates"]),
        integration_steps=int(benchmark["sampling"]["integration_steps"]),
        save_integration_history=False,
    )
    flow_checkpoint = (
        args.benchmark_root / "checkpoints" / f"seed_{args.seed}" / "flow" / "best.pt"
    )
    flow_model = _load_flow(flow_checkpoint, device)
    vtf_model = _load_flow(args.vtf_checkpoint, device)
    cfg = flow_training_config(benchmark, args.seed, tvk=True)
    terrain = TerrainFieldConfig(**cfg["terrain_field"])
    vehicle = VehicleConditionedFieldConfig(**cfg["vehicle_conditioning"])
    base_guidance = guidance_config(benchmark, use_kinematics=True)
    values = dict(base_guidance.__dict__)
    values.update(
        strength=0.05,
        schedule="late-strong",
        gamma=1.0,
        smoothing_kernel="kernel_3",
        endpoint_projection="terminal",
    )
    vtf_planner = GuidedFlowPlanner(
        vtf_model, plan, FeasibilityFlowGuidanceConfig(**values), terrain, vehicle
    ).to(device)
    rows = []
    for protocol in ("iid", "antithetic"):
        flow_metrics = _evaluate(
            FlowPlanner(flow_model, plan).to(device), loader, positions, args.seed,
            benchmark, device, protocol,
        )
        rows.append({"method": "Flow", "noise_protocol": protocol, **flow_metrics})
        vtf_metrics = _evaluate(
            vtf_planner, loader, positions, args.seed, benchmark, device, protocol,
        )
        rows.append({"method": "VTF-stage2", "noise_protocol": protocol, **vtf_metrics})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "validation_sampling_protocols.csv", rows)
    summary = {
        "selection_split": "validation",
        "test_sequences_consulted": [],
        "evaluated_scenes": count,
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

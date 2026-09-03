"""Select VTF-Flow inference guidance only on the held-out validation sequence.

This script intentionally never evaluates the configured test sequence.  It
uses identical scene-wise Gaussian noise for all variants, reports all tried
configurations, marks the Pareto set, and writes a frozen selected config that
can subsequently be evaluated once on the test sequence.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Subset


TERRAFLOW_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TERRAFLOW_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from TerraFlow.evaluation.final_experiments import (  # noqa: E402
    partition_sequence_indices,
    terminal_scene_metrics,
)
from TerraFlow.guidance.feasibility_flow_guidance import (  # noqa: E402
    FeasibilityFlowGuidanceConfig,
)
from TerraFlow.metrics.trajectory_metrics import trajectory_metrics  # noqa: E402
from TerraFlow.planners.flow_planner import FlowPlanner, FlowPlannerConfig  # noqa: E402
from TerraFlow.planners.guided_flow_planner import GuidedFlowPlanner  # noqa: E402
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
from TerraFlow.scripts.train_regression import (  # noqa: E402
    CombinedSceneDataset,
    make_loader,
)
from TerraFlow.terrain.feasibility_field import TerrainFieldConfig  # noqa: E402
from TerraFlow.terrain.trajectory_kinematics import TrajectoryKinematicConfig  # noqa: E402
from TerraFlow.terrain.vehicle_conditioned_field import (  # noqa: E402
    VehicleConditionedFieldConfig,
)


DEFAULT_SEARCH = TERRAFLOW_ROOT / "configs" / "vtf_flow_validation_search.json"
DEFAULT_BENCHMARK_ROOT = TERRAFLOW_ROOT / "outputs" / "unified_h10_benchmark"
DEFAULT_OUTPUT = TERRAFLOW_ROOT / "outputs" / "vtf_flow_validation_search"
PRIMARY_METRICS = (
    "ADE_candidate0_m",
    "minADE@K_m",
    "terrain_violation_rate",
)
REPORTED_METRICS = (*PRIMARY_METRICS, "mean_unified_tvk_cost", "smoothness_m", "diversity_m")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fixed_noise(
    positions: Sequence[int], seed: int, candidates: int, horizon: int,
    device: torch.device, protocol: str = "iid",
) -> torch.Tensor:
    chunks = []
    for position in positions:
        generator = torch.Generator().manual_seed(100_000 + seed * 10_000 + int(position))
        if protocol == "iid":
            noise = torch.randn((candidates, horizon, 3), generator=generator)
        elif protocol == "antithetic":
            if candidates % 2 != 0:
                raise ValueError("antithetic noise requires an even candidate count")
            base = torch.randn((candidates // 2, horizon, 3), generator=generator)
            noise = torch.stack((base, -base), dim=1).reshape(candidates, horizon, 3)
        else:
            raise ValueError("noise protocol must be 'iid' or 'antithetic'")
        chunks.append(noise)
    return torch.stack(chunks).to(device)


def _evaluate(
    planner: FlowPlanner | GuidedFlowPlanner,
    loader: torch.utils.data.DataLoader,
    positions: Sequence[int],
    seed: int,
    benchmark: Mapping[str, Any],
    device: torch.device,
    noise_protocol: str = "iid",
) -> dict[str, float]:
    flow_config = flow_training_config(benchmark, seed, tvk=True)
    terrain = TerrainFieldConfig(**flow_config["terrain_field"])
    vehicle = VehicleConditionedFieldConfig(**flow_config["vehicle_conditioning"])
    kinematic = TrajectoryKinematicConfig(**benchmark["kinematic"])
    candidates = int(benchmark["sampling"]["candidates"])
    horizon = int(benchmark["trajectory"]["horizon_steps"])
    totals = {name: 0.0 for name in REPORTED_METRICS}
    count = 0
    offset = 0
    planner.eval()
    for scene in loader:
        scene = scene.to(device)
        batch = scene.batch_size
        batch_positions = positions[offset : offset + batch]
        noise = _fixed_noise(
            batch_positions, seed, candidates, horizon, device, noise_protocol
        )
        with torch.enable_grad() if isinstance(planner, GuidedFlowPlanner) else torch.no_grad():
            prediction = planner.sample(scene, noise)
        standard = trajectory_metrics(prediction.trajectories, scene.gt_future)
        terminal = terminal_scene_metrics(
            prediction.trajectories,
            scene.gt_future,
            scene.terrain_map,
            terrain,
            vehicle,
            flow_config["metrics"],
            planning_dt_s=float(benchmark["trajectory"]["planning_dt_s"]),
            kinematic_config=kinematic,
        )
        values = {
            "ADE_candidate0_m": standard["ADE_by_candidate_m"][:, 0],
            "minADE@K_m": standard["minADE@K_m"],
            "terrain_violation_rate": terminal["terrain_violation_rate"],
            "mean_unified_tvk_cost": terminal["mean_unified_tvk_cost"],
            "smoothness_m": terminal["smoothness_m"],
            "diversity_m": terminal["diversity_m"],
        }
        for name, value in values.items():
            totals[name] += float(value.detach().sum().cpu())
        count += batch
        offset += batch
    if count != len(positions):
        raise AssertionError("validation loader did not cover the requested positions")
    return {name: total / count for name, total in totals.items()}


def _is_dominated(row: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> bool:
    for other in rows:
        no_worse = all(float(other[name]) <= float(row[name]) for name in PRIMARY_METRICS)
        strictly_better = any(float(other[name]) < float(row[name]) for name in PRIMARY_METRICS)
        if no_worse and strictly_better:
            return True
    return False


def select_variant(
    rows: list[dict[str, Any]],
    reference: Mapping[str, float],
    weights: Mapping[str, float],
    maximum_relative_accuracy_degradation: float,
) -> dict[str, Any]:
    """Select a validation Pareto point without consulting test outcomes."""

    if set(weights) != set(PRIMARY_METRICS):
        raise ValueError(f"metric_weights must contain exactly {PRIMARY_METRICS}")
    if any(float(value) < 0.0 for value in weights.values()):
        raise ValueError("metric weights must be non-negative")
    candidates = [row for row in rows if row["kind"] == "VTF"]
    for row in candidates:
        row["pareto"] = not _is_dominated(row, candidates)
        row["accuracy_admissible"] = (
            float(row["ADE_candidate0_m"])
            <= float(reference["ADE_candidate0_m"]) * (1.0 + maximum_relative_accuracy_degradation)
            and float(row["minADE@K_m"])
            <= float(reference["minADE@K_m"]) * (1.0 + maximum_relative_accuracy_degradation)
        )
        row["selection_score"] = sum(
            float(weights[name]) * float(row[name]) / max(float(reference[name]), 1e-12)
            for name in PRIMARY_METRICS
        )
    admissible = [row for row in candidates if row["pareto"] and row["accuracy_admissible"]]
    pool = admissible or [row for row in candidates if row["pareto"]]
    return min(pool, key=lambda row: float(row["selection_score"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--search-config", type=Path, default=DEFAULT_SEARCH)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument(
        "--vtf-checkpoint",
        type=Path,
        help="Optional checkpoint override for validation-only method screening.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark = load_json(args.benchmark_config)
    search = load_json(args.search_config)
    if search["selection_split"] != "validation":
        raise ValueError("hyperparameter selection is restricted to the validation split")
    if str(search["selection_sequence"]).zfill(5) not in benchmark["protocol"]["validation"]:
        raise ValueError("selection_sequence must be one of the benchmark validation sequences")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = CombinedSceneDataset(args.cache_root.resolve(), tuple(benchmark["protocol"]["source_splits"]))
    dataset = H10PlanningDataset(
        source,
        args.data_root.resolve() / "processed" / "Rellis-3D",
        horizon=int(benchmark["trajectory"]["horizon_steps"]),
        history_steps=int(benchmark["trajectory"]["history_steps"]),
    )
    split_indices = partition_sequence_indices(dataset.sequence_ids, benchmark_split(benchmark))
    available = split_indices["validation"]
    maximum = int(args.max_scenes or search["coarse_scene_count"])
    count = min(maximum, len(available))
    validation_positions = np.linspace(
        0, len(available) - 1, num=count, dtype=np.int64
    ).tolist()
    chosen = [available[position] for position in validation_positions]
    loader = make_loader(
        Subset(dataset, chosen),
        int(search["batch_size"]),
        shuffle=False,
        seed=73,
        num_workers=0,
    )
    rows: list[dict[str, Any]] = []
    for seed_value in search["seeds"]:
        seed = int(seed_value)
        flow_checkpoint = args.benchmark_root / "checkpoints" / f"seed_{seed}" / "flow" / "best.pt"
        vtf_checkpoint = (
            args.vtf_checkpoint
            if args.vtf_checkpoint is not None
            else args.benchmark_root / "checkpoints" / f"seed_{seed}" / "flow_tvk" / "best.pt"
        )
        if not flow_checkpoint.is_file() or not vtf_checkpoint.is_file():
            raise FileNotFoundError("required unified benchmark checkpoints are missing")
        flow_model = _load_flow(flow_checkpoint, device)
        plan = FlowPlannerConfig(
            candidates=int(benchmark["sampling"]["candidates"]),
            integration_steps=int(benchmark["sampling"]["integration_steps"]),
            save_integration_history=False,
        )
        reference_metrics = _evaluate(
            FlowPlanner(flow_model, plan).to(device), loader, validation_positions,
            seed, benchmark, device,
        )
        rows.append({"kind": "FLOW", "seed": seed, "variant": "flow_reference", **reference_metrics})
        del flow_model
        vtf_model = _load_flow(vtf_checkpoint, device)
        flow_cfg = flow_training_config(benchmark, seed, tvk=True)
        terrain_cfg = TerrainFieldConfig(**flow_cfg["terrain_field"])
        vehicle_cfg = VehicleConditionedFieldConfig(**flow_cfg["vehicle_conditioning"])
        for variant in search["variants"]:
            base = guidance_config(benchmark, use_kinematics=True)
            values = dict(base.__dict__)
            values.update(
                strength=float(variant["eta"]),
                schedule=str(variant["schedule"]),
                gamma=float(variant["gamma"]),
                smoothing_kernel=str(variant["smoothing_kernel"]),
                endpoint_projection=str(variant["endpoint_projection"]),
                adaptive_trigger_enabled=bool(
                    variant.get("adaptive_trigger_enabled", False)
                ),
                trigger_alpha=float(variant.get("trigger_alpha", 10.0)),
                trigger_reference_cost=float(
                    variant.get("trigger_reference_cost", 0.5)
                ),
            )
            planner = GuidedFlowPlanner(
                vtf_model, plan, FeasibilityFlowGuidanceConfig(**values),
                terrain_cfg, vehicle_cfg,
            ).to(device)
            metrics = _evaluate(planner, loader, validation_positions, seed, benchmark, device)
            rows.append({
                "kind": "VTF", "seed": seed, "variant": variant["name"],
                "eta": variant["eta"], "schedule": variant["schedule"],
                "gamma": variant["gamma"], "smoothing_kernel": variant["smoothing_kernel"],
                "endpoint_projection": variant["endpoint_projection"], **metrics,
                "adaptive_trigger_enabled": variant.get(
                    "adaptive_trigger_enabled", False
                ),
                "trigger_alpha": variant.get("trigger_alpha", 10.0),
                "trigger_reference_cost": variant.get(
                    "trigger_reference_cost", 0.5
                ),
            })
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
        del vtf_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # The current search is intentionally seed-0 coarse screening.  Multi-seed
    # confirmation can reuse the same script after adding seeds to the config.
    reference_rows = [row for row in rows if row["kind"] == "FLOW"]
    reference = {
        name: float(np.mean([float(row[name]) for row in reference_rows]))
        for name in PRIMARY_METRICS
    }
    selected = select_variant(
        rows,
        reference,
        search["metric_weights"],
        float(search["maximum_relative_accuracy_degradation"]),
    )
    for row in rows:
        row["selected"] = row is selected
        row.setdefault("pareto", "")
        row.setdefault("accuracy_admissible", "")
        row.setdefault("selection_score", "")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "validation_search.csv", rows)
    frozen = {
        "selection_split": "validation",
        "selection_sequences": list(benchmark["protocol"]["validation"]),
        "test_sequences_consulted": [],
        "evaluated_validation_scenes": len(chosen),
        "selected_variant": selected,
        "flow_validation_reference": reference,
        "search_config": search,
    }
    (args.output_dir / "selected_guidance_config.json").write_text(
        json.dumps(frozen, indent=2), encoding="utf-8"
    )
    print(json.dumps(frozen, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

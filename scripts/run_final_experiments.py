"""Train and evaluate frozen VTF-Flow variants for final experiments.

This script uses sequence-disjoint splits, retrains all learning-based models
for every configured seed, and stores complete reproducibility artifacts.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Subset


TERRAFLOW_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TERRAFLOW_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from TerraFlow.evaluation.final_experiments import (  # noqa: E402
    SequenceSplit,
    partition_sequence_indices,
    save_json,
    scene_identifier,
    summarize_scene_rows,
    terminal_scene_metrics,
    write_csv,
)
from TerraFlow.guidance.feasibility_flow_guidance import (  # noqa: E402
    FeasibilityFlowGuidanceConfig,
)
from TerraFlow.planners.flow_planner import FlowPlanner, FlowPlannerConfig  # noqa: E402
from TerraFlow.planners.guided_flow_planner import GuidedFlowPlanner  # noqa: E402
from TerraFlow.planners.regression_planner import RegressionPlanner  # noqa: E402
from TerraFlow.scripts.run_flow_feasibility_experiment import (  # noqa: E402
    _train_one as train_flow_variant,
    load_experiment_config,
)
from TerraFlow.scripts.train_flow import model_from_config  # noqa: E402
from TerraFlow.scripts.train_regression import (  # noqa: E402
    CombinedSceneDataset,
    _model_config as regression_model_config,
    load_config as load_regression_config,
    make_loader,
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


DEFAULT_CONFIG = TERRAFLOW_ROOT / "configs" / "final_experiments.json"
DEFAULT_CACHE = TERRAFLOW_ROOT.parent / "data" / "RELLIS3D" / "trajectory_cache_h150_s5"
DEFAULT_OUTPUT = TERRAFLOW_ROOT / "outputs" / "final_experiments"
METRIC_CONFIG = TERRAFLOW_ROOT / "configs" / "rellis3d_flow_feasibility.json"
REGRESSION_CONFIG = TERRAFLOW_ROOT / "configs" / "rellis3d_regression_matched_flow.json"


def _git_commit() -> str:
    """Return the current commit or an explicit unavailable marker."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=TERRAFLOW_ROOT,
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable_not_a_git_worktree"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _effective_flow_config(final: Mapping[str, Any], seed: int) -> dict[str, Any]:
    config = load_experiment_config(METRIC_CONFIG)
    config["data"]["source_splits"] = list(final["protocol"]["source_splits"])
    config["training"]["seed"] = int(seed)
    config["training"]["epochs"] = int(final["training"]["epochs"])
    config["sampling"].update({
        "candidates": int(final["sampling"]["candidates"]),
        "integration_steps": int(final["sampling"]["integration_steps"]),
    })
    config["experiment"]["lambda_feasibility"] = [
        float(final["training"]["vehicle_lambda"])
    ]
    return config


def _effective_regression_config(final: Mapping[str, Any], seed: int) -> dict[str, Any]:
    config = load_regression_config(REGRESSION_CONFIG)
    config["data"]["source_splits"] = list(final["protocol"]["source_splits"])
    config["training"]["seed"] = int(seed)
    config["training"]["epochs"] = int(final["training"]["epochs"])
    return config


def _train_models(
    source: CombinedSceneDataset,
    indices: Mapping[str, list[int]],
    split: SequenceSplit,
    seed: int,
    final: Mapping[str, Any],
    output_root: Path,
    device: torch.device,
) -> dict[str, Path]:
    """Train Regression, Flow, and vehicle-regularized Flow from scratch."""

    checkpoint_root = output_root / "checkpoints" / split.name / f"seed_{seed}"
    train_set = Subset(source, indices["train"])
    validation_set = Subset(source, indices["validation"])
    split_payload = {
        **split.as_dict(),
        "train_scenes": len(train_set),
        "validation_scenes": len(validation_set),
        "test_scenes": len(indices["test"]),
    }
    save_json(checkpoint_root / "split_definition.json", split_payload)
    flow_config = _effective_flow_config(final, seed)
    regression_config = _effective_regression_config(final, seed)
    checkpoints = {
        "R": checkpoint_root / "regression" / "best.pt",
        "A": checkpoint_root / "flow" / "best.pt",
        "B": checkpoint_root / "flow_vehicle" / "best.pt",
    }
    if not checkpoints["R"].is_file():
        set_reproducible_seed(seed)
        regression_dir = checkpoints["R"].parent
        regression_dir.mkdir(parents=True, exist_ok=True)
        save_json(regression_dir / "effective_config.json", regression_config)
        train_regression(
            train_set, validation_set, regression_config, regression_dir, device,
            int(final["training"]["epochs"]),
        )
    if not checkpoints["A"].is_file():
        train_flow_variant(
            flow_config, train_set, validation_set, checkpoints["A"].parent,
            "none", 0.0, device, int(final["training"]["epochs"]),
        )
    if not checkpoints["B"].is_file():
        train_flow_variant(
            flow_config, train_set, validation_set, checkpoints["B"].parent,
            "vehicle", float(final["training"]["vehicle_lambda"]), device,
            int(final["training"]["epochs"]),
        )
    return checkpoints


def _load_flow(path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = model_from_config(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def _load_regression(path: Path, device: torch.device) -> RegressionPlanner:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = RegressionPlanner(regression_model_config(checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def _guidance_config(
    final: Mapping[str, Any], *, eta: float, smoothing: str,
    field_type: str = "vehicle", use_kinematics: bool = False,
) -> FeasibilityFlowGuidanceConfig:
    """Build the frozen inference guidance without trust region or trigger."""

    kinematic = final.get("kinematic", {}) if use_kinematics else {}
    return FeasibilityFlowGuidanceConfig(
        enabled=eta > 0.0,
        strength=float(eta),
        schedule=str(final["guidance"]["schedule"]),
        gamma=1.0,
        gradient_normalization="rms",
        maximum_gradient_norm=4.0,
        minimum_gradient_norm=1e-7,
        field_type=field_type,  # type: ignore[arg-type]
        planning_dt_s=0.5,
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
        smoothing_kernel=smoothing,
        trust_region_rho=None,
        adaptive_trigger_enabled=False,
    )


def _evaluate_one(
    *,
    method: str,
    checkpoint_path: Path,
    source: CombinedSceneDataset,
    dataset_indices: Sequence[int],
    split: SequenceSplit,
    seed: int,
    final: Mapping[str, Any],
    output_dir: Path,
    device: torch.device,
    candidates: int | None = None,
    integration_steps: int | None = None,
    eta: float = 0.0,
    smoothing: str = "none",
    field_type: str = "vehicle",
    use_kinematic_guidance: bool = False,
    maximum_noise_candidates: int | None = None,
) -> dict[str, Any]:
    """Evaluate one frozen variant and save complete scene-level artifacts."""

    if (output_dir / "metrics.json").is_file() and (output_dir / "scene_level_metrics.csv").is_file():
        return _load_json(output_dir / "metrics.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    flow_config = _effective_flow_config(final, seed)
    terrain_config = TerrainFieldConfig(**flow_config["terrain_field"])
    vehicle_config = VehicleConditionedFieldConfig(**flow_config["vehicle_conditioning"])
    thresholds = flow_config["metrics"]
    evaluation_set = Subset(source, list(dataset_indices))
    loader = make_loader(
        evaluation_set, int(flow_config["training"]["batch_size"]), shuffle=False,
        seed=seed + 9000, num_workers=int(flow_config["training"]["num_workers"]),
    )
    is_regression = method == "R"
    planner_config = FlowPlannerConfig(
        candidates=int(candidates or final["sampling"]["candidates"]),
        integration_steps=int(integration_steps or final["sampling"]["integration_steps"]),
        save_integration_history=False,
    )
    if is_regression:
        planner: Any = _load_regression(checkpoint_path, device)
        model = None
    else:
        model = _load_flow(checkpoint_path, device)
        if eta > 0.0:
            planner = GuidedFlowPlanner(
                model, planner_config,
                _guidance_config(
                    final, eta=eta, smoothing=smoothing, field_type=field_type,
                    use_kinematics=use_kinematic_guidance,
                ),
                terrain_config, vehicle_config,
            ).to(device)
        else:
            planner = FlowPlanner(model, planner_config).to(device)
    scene_rows: list[dict[str, Any]] = []
    trajectory_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    id_chunks: list[str] = []
    index_chunks: list[int] = []
    latency_samples: list[float] = []
    position = 0
    with torch.no_grad() if is_regression or eta == 0.0 else torch.enable_grad():
        for batch_index, scene in enumerate(loader):
            scene = scene.to(device)
            batch = scene.batch_size
            if is_regression:
                noise = None
            else:
                noise_k = int(maximum_noise_candidates or planner_config.candidates)
                generator = torch.Generator(device=device).manual_seed(
                    100_000 + seed * 10_000 + batch_index
                )
                full_noise = torch.randn(
                    (batch, noise_k, model.trajectory_points, 3),
                    generator=generator, device=device, dtype=scene.gt_future.dtype,
                )
                noise = full_noise[:, :planner_config.candidates]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            prediction = planner(scene) if is_regression else planner.sample(scene, noise)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            latency_per_scene = elapsed_ms / batch
            latency_samples.extend([latency_per_scene] * batch)
            metric = terminal_scene_metrics(
                prediction.trajectories, scene.gt_future, scene.terrain_map,
                terrain_config, vehicle_config, thresholds,
                kinematic_config=TrajectoryKinematicConfig(**final.get("kinematic", {})),
            )
            metric_np = {name: value.detach().cpu().numpy() for name, value in metric.items()}
            metadata = scene.metadata
            for local in range(batch):
                dataset_index = int(dataset_indices[position + local])
                meta = metadata[local]
                row: dict[str, Any] = {
                    "scene_id": scene_identifier(meta, dataset_index),
                    "sequence": str(meta.get("sequence", "unknown")).zfill(5),
                    "frame_id": meta.get("frame_id", meta.get("frame_index", "unknown")),
                    "dataset_index": dataset_index,
                    "method": method,
                    "seed": seed,
                    "split": split.name,
                }
                row.update({name: float(values[local]) for name, values in metric_np.items()})
                scene_rows.append(row)
                id_chunks.append(row["scene_id"])
                index_chunks.append(dataset_index)
            trajectory_chunks.append(prediction.trajectories.detach().cpu().numpy())
            target_chunks.append(scene.gt_future.detach().cpu().numpy())
            position += batch
    if position != len(dataset_indices):
        raise AssertionError(f"missing scenes: evaluated {position}, expected {len(dataset_indices)}")
    if len(set(id_chunks)) != len(id_chunks):
        raise ValueError("scene identifiers are not unique")
    summary = summarize_scene_rows(scene_rows)
    summary.update({
        "method": method,
        "seed": int(seed),
        "split": split.name,
        "checkpoint": str(checkpoint_path.resolve()),
        "latency_ms_per_scene": float(np.mean(latency_samples)),
        "latency_p95_ms_per_scene": float(np.percentile(latency_samples, 95.0)),
        "K": 1 if is_regression else planner_config.candidates,
        "integration_steps": 0 if is_regression else planner_config.integration_steps,
        "eta": float(eta),
        "smoothing_kernel": smoothing,
        "field_type": field_type,
        "kinematic_guidance": bool(use_kinematic_guidance),
    })
    effective = {
        "method": method,
        "seed": seed,
        "split": split.as_dict(),
        "checkpoint": str(checkpoint_path.resolve()),
        "sampling": {
            "K": summary["K"], "integration_steps": summary["integration_steps"],
            "solver": final["sampling"]["solver"],
        },
        "guidance": {
            "eta": eta, "smoothing_kernel": smoothing, "field_type": field_type,
            "kinematic_guidance": bool(use_kinematic_guidance),
            "kinematic": final.get("kinematic", {}),
            "trust_region": False, "adaptive_trigger": False,
        },
        "metric_directionality": {
            "lower_is_better": [
                "minADE@K_m", "minFDE@K_m", "mean_vehicle_conditioned_cost",
                "mean_terrain_cost", "terrain_violation_rate", "occupancy_violation_rate",
                "slope_violation_rate", "roughness_cost", "clearance_cost",
                "smoothness_m", "maximum_local_second_difference_m", "latency_ms_per_scene",
                "mean_kinematic_cost", "mean_unified_tvk_cost", "curvature_cost",
                "lateral_acceleration_cost", "curvature_violation_rate",
                "lateral_acceleration_violation_rate", "mean_absolute_curvature_per_m",
                "mean_lateral_acceleration_mps2",
            ],
            "higher_is_better": ["diversity_m"],
        },
        "git_commit": _git_commit(),
    }
    save_json(output_dir / "effective_config.json", effective)
    save_json(output_dir / "split_definition.json", split.as_dict())
    save_json(output_dir / "metrics.json", summary)
    save_json(output_dir / "latency.json", {
        "mean_ms_per_scene": summary["latency_ms_per_scene"],
        "p95_ms_per_scene": summary["latency_p95_ms_per_scene"],
        "scene_count": len(latency_samples),
    })
    (output_dir / "seed.txt").write_text(f"{seed}\n", encoding="utf-8")
    (output_dir / "checkpoint_reference.txt").write_text(
        str(checkpoint_path.resolve()) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / "scene_level_metrics.csv", scene_rows)
    np.savez_compressed(
        output_dir / "predictions.npz",
        trajectories=np.concatenate(trajectory_chunks),
        ground_truth=np.concatenate(target_chunks),
        scene_ids=np.asarray(id_chunks),
        dataset_indices=np.asarray(index_chunks, dtype=np.int64),
    )
    return summary


def _registry_row(
    experiment_name: str, seed: int, split: SequenceSplit,
    checkpoint: Path, status: str = "complete",
) -> dict[str, Any]:
    return {
        "experiment_name": experiment_name,
        "seed": seed,
        "split": split.name,
        "checkpoint": str(checkpoint.resolve()),
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _run_main_split(
    source: CombinedSceneDataset,
    split: SequenceSplit,
    seeds: Sequence[int],
    final: Mapping[str, Any],
    output_root: Path,
    device: torch.device,
    registry: list[dict[str, Any]],
) -> dict[int, dict[str, Path]]:
    indices = partition_sequence_indices(source.sequence_ids, split)
    checkpoint_map: dict[int, dict[str, Path]] = {}
    for seed in seeds:
        print(f"=== {split.name} seed={seed}: training ===", flush=True)
        checkpoints = _train_models(source, indices, split, seed, final, output_root, device)
        checkpoint_map[seed] = checkpoints
        methods = {
            "R": (checkpoints["R"], 0.0, "none"),
            "A": (checkpoints["A"], 0.0, "none"),
            "B": (checkpoints["B"], 0.0, "none"),
            "C": (checkpoints["A"], float(final["guidance"]["eta"]), "kernel_3"),
            "D": (checkpoints["B"], float(final["guidance"]["eta"]), "kernel_3"),
        }
        for method, (checkpoint, eta, smoothing) in methods.items():
            experiment_name = f"main_{split.name}_seed{seed}_{method}"
            print(f"  evaluating {experiment_name}", flush=True)
            _evaluate_one(
                method=method, checkpoint_path=checkpoint, source=source,
                dataset_indices=indices["test"], split=split, seed=seed, final=final,
                output_dir=output_root / experiment_name, device=device,
                eta=eta, smoothing=smoothing,
            )
            registry.append(_registry_row(experiment_name, seed, split, checkpoint))
    return checkpoint_map


def _run_sensitivities(
    source: CombinedSceneDataset,
    split: SequenceSplit,
    checkpoints: Mapping[str, Path],
    final: Mapping[str, Any],
    output_root: Path,
    device: torch.device,
    registry: list[dict[str, Any]],
) -> None:
    indices = partition_sequence_indices(source.sequence_ids, split)
    test_indices = indices["test"]
    seed = 0
    # Final VTF-Flow eta sensitivity.
    for eta in final["sensitivity"]["eta"]:
        name = f"eta_primary_seed0_eta{float(eta):g}"
        _evaluate_one(
            method="D", checkpoint_path=checkpoints["B"], source=source,
            dataset_indices=test_indices, split=split, seed=seed, final=final,
            output_dir=output_root / name, device=device, eta=float(eta),
            smoothing="kernel_3" if float(eta) > 0.0 else "none",
        )
        registry.append(_registry_row(name, seed, split, checkpoints["B"]))
    # Final VTF-Flow Euler-step sensitivity.
    for steps in final["sensitivity"]["integration_steps"]:
        name = f"steps_primary_seed0_s{int(steps)}"
        _evaluate_one(
            method="D", checkpoint_path=checkpoints["B"], source=source,
            dataset_indices=test_indices, split=split, seed=seed, final=final,
            output_dir=output_root / name, device=device,
            integration_steps=int(steps), eta=float(final["guidance"]["eta"]),
            smoothing="kernel_3",
        )
        registry.append(_registry_row(name, seed, split, checkpoints["B"]))
    # K sensitivity uses prefixes of the exact same K=10 noise tensor.
    max_k = max(int(value) for value in final["sensitivity"]["candidates"])
    for candidates in final["sensitivity"]["candidates"]:
        name = f"k_primary_seed0_k{int(candidates)}"
        _evaluate_one(
            method="D", checkpoint_path=checkpoints["B"], source=source,
            dataset_indices=test_indices, split=split, seed=seed, final=final,
            output_dir=output_root / name, device=device, candidates=int(candidates),
            eta=float(final["guidance"]["eta"]), smoothing="kernel_3",
            maximum_noise_candidates=max_k,
        )
        registry.append(_registry_row(name, seed, split, checkpoints["B"]))
    # Guidance design on baseline Flow: no, raw, smoothed.
    for label, eta, smoothing in (
        ("none", 0.0, "none"),
        ("raw", float(final["guidance"]["eta"]), "none"),
        ("smoothed", float(final["guidance"]["eta"]), "kernel_3"),
    ):
        name = f"guidance_primary_seed0_{label}"
        _evaluate_one(
            method="C" if eta > 0.0 else "A", checkpoint_path=checkpoints["A"],
            source=source, dataset_indices=test_indices, split=split, seed=seed,
            final=final, output_dir=output_root / name, device=device,
            eta=eta, smoothing=smoothing,
        )
        registry.append(_registry_row(name, seed, split, checkpoints["A"]))
    # Representation ablation. Hard binary traversability is non-differentiable,
    # so its inference result is the unguided Flow; continuous fields receive the
    # identical smooth guidance mechanism.
    for label, eta, field_type in (
        ("binary", 0.0, "vehicle"),
        ("terrain_continuous", float(final["guidance"]["eta"]), "terrain"),
        ("vehicle_continuous", float(final["guidance"]["eta"]), "vehicle"),
    ):
        name = f"field_primary_seed0_{label}"
        _evaluate_one(
            method="C" if eta > 0.0 else "A", checkpoint_path=checkpoints["A"],
            source=source, dataset_indices=test_indices, split=split, seed=seed,
            final=final, output_dir=output_root / name, device=device,
            eta=eta, smoothing="kernel_3" if eta > 0.0 else "none",
            field_type=field_type,
        )
        registry.append(_registry_row(name, seed, split, checkpoints["A"]))
    # Per-sequence diagnostics using primary seed 0. Roles are retained so that
    # train/validation rows are never presented as held-out generalization.
    for sequence in ("00000", "00001", "00002", "00003", "00004"):
        sequence_indices = [
            index for index, value in enumerate(source.sequence_ids)
            if str(value).zfill(5) == sequence
        ]
        for method, checkpoint, eta in (
            ("A", checkpoints["A"], 0.0),
            ("D", checkpoints["B"], float(final["guidance"]["eta"])),
        ):
            name = f"per_sequence_primary_seed0_{sequence}_{method}"
            _evaluate_one(
                method=method, checkpoint_path=checkpoint, source=source,
                dataset_indices=sequence_indices, split=split, seed=seed, final=final,
                output_dir=output_root / name, device=device, eta=eta,
                smoothing="kernel_3" if eta > 0.0 else "none",
            )
            registry.append(_registry_row(name, seed, split, checkpoint))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-sensitivities", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    final = _load_json(args.config)
    source = CombinedSceneDataset(args.cache_root, tuple(final["protocol"]["source_splits"]))
    splits = {
        name: SequenceSplit.from_mapping(name, final["protocol"][name])
        for name in ("primary", "swapped")
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    save_json(args.output_root / "effective_config.json", {
        **final,
        "cache_root": str(args.cache_root.resolve()),
        "git_commit": _git_commit(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    })
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    registry: list[dict[str, Any]] = []
    primary_checkpoints = _run_main_split(
        source, splits["primary"], [int(seed) for seed in final["seeds"]],
        final, args.output_root, device, registry,
    )
    _run_main_split(
        source, splits["swapped"], [int(seed) for seed in final["swapped_seeds"]],
        final, args.output_root, device, registry,
    )
    if not args.skip_sensitivities:
        _run_sensitivities(
            source, splits["primary"], primary_checkpoints[0], final,
            args.output_root, device, registry,
        )
    # Preserve existing completed registry rows when resuming an interrupted run.
    registry_path = args.output_root / "experiment_registry.csv"
    existing: list[dict[str, Any]] = []
    if registry_path.is_file():
        with registry_path.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    by_name = {str(row["experiment_name"]): row for row in existing}
    by_name.update({str(row["experiment_name"]): row for row in registry})
    write_csv(registry_path, list(by_name.values()))
    print(json.dumps({
        "status": "complete", "experiments": len(by_name),
        "output_root": str(args.output_root.resolve()),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

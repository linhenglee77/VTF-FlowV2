"""Run resumable sequence-level Flow versus final VTF-Flow robustness folds."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
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
    terminal_scene_metrics,
    write_csv,
)
from TerraFlow.evaluation.planning_claim_metrics import (  # noqa: E402
    PlanningClaimMetricConfig,
    candidate_claim_metrics,
    compliance_mask,
    derive_independent_component_maps,
)
from TerraFlow.evaluation.sequence_robustness import (  # noqa: E402
    SequenceHoldoutFold,
    build_fixed_validation_holdouts,
    sequence_level_method_effects,
)
from TerraFlow.metrics.trajectory_metrics import trajectory_metrics  # noqa: E402
from TerraFlow.planners.flow_planner import FlowPlanner, FlowPlannerConfig  # noqa: E402
from TerraFlow.planners.guided_flow_planner import GuidedFlowPlanner  # noqa: E402
from TerraFlow.scripts.optimize_vtf_flow_validation import _fixed_noise  # noqa: E402
from TerraFlow.scripts.run_final_experiments import _load_flow  # noqa: E402
from TerraFlow.scripts.run_flow_feasibility_experiment import (  # noqa: E402
    _train_one as train_flow_variant,
)
from TerraFlow.scripts.run_unified_h10_benchmark import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_CONFIG as DEFAULT_BENCHMARK,
    DEFAULT_DATA,
    H10PlanningDataset,
    flow_training_config,
    guidance_config,
    load_json,
)
from TerraFlow.scripts.train_regression import (  # noqa: E402
    CombinedSceneDataset,
    make_loader,
)
from TerraFlow.terrain.feasibility_field import TerrainFieldConfig  # noqa: E402
from TerraFlow.terrain.trajectory_kinematics import (  # noqa: E402
    TrajectoryKinematicConfig,
)
from TerraFlow.terrain.vehicle_conditioned_field import (  # noqa: E402
    VehicleConditionedFieldConfig,
)


DEFAULT_PROTOCOL = TERRAFLOW_ROOT / "configs" / "sequence_holdout_robustness.json"
DEFAULT_OUTPUT = TERRAFLOW_ROOT / "outputs" / "sequence_holdout_robustness"
METHODS = ("FLOW", "VTF_V2")
METRICS = (
    "ADE_candidate0_m",
    "minADE@K_m",
    "mean_unified_tvk_cost",
    "terrain_violation_rate",
    "occupancy_exposure_rate",
    "nontraversable_exposure_rate",
    "slope_exposure_rate",
    "roughness_mean",
    "clearance_q05_m",
    "curvature_violation_rate_independent",
    "smoothness_m_independent",
    "compliant_candidate_rate_q80",
    "GCCR_at_K_q80",
)
HIGHER_IS_BETTER = {
    "clearance_q05_m",
    "compliant_candidate_rate_q80",
    "GCCR_at_K_q80",
}


def load_protocol(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_sequence_split(fold: SequenceHoldoutFold) -> SequenceSplit:
    return SequenceSplit(
        name=fold.name,
        train=fold.train,
        validation=fold.validation,
        test=fold.test,
    )


def train_fold_models(
    dataset: H10PlanningDataset,
    indices: Mapping[str, list[int]],
    fold: SequenceHoldoutFold,
    seed: int,
    benchmark: Mapping[str, Any],
    protocol: Mapping[str, Any],
    output_root: Path,
    device: torch.device,
    skip_training: bool,
) -> dict[str, Path]:
    """Train or resume the two controlled models for one outer fold."""

    root = output_root / "checkpoints" / fold.name / f"seed_{seed}"
    paths = {
        "FLOW": root / "flow" / "best.pt",
        "VTF_V2": root / "flow_tvk" / "best.pt",
    }
    if skip_training:
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing requested checkpoints: {missing}")
        return paths
    train_set = Subset(dataset, indices["train"])
    validation_set = Subset(dataset, indices["validation"])
    save_json(
        root / "split_definition.json",
        {
            "fold": fold.name,
            "train": list(fold.train),
            "validation": list(fold.validation),
            "test": list(fold.test),
            "train_scenes": len(train_set),
            "validation_scenes": len(validation_set),
            "test_scenes": len(indices["test"]),
        },
    )
    epochs = int(protocol["epochs"])
    if not paths["FLOW"].is_file():
        config = flow_training_config(benchmark, seed, tvk=False)
        config["training"]["epochs"] = epochs
        config["data"]["validation_sequences"] = list(fold.validation)
        print(f"training {fold.name} seed={seed} FLOW", flush=True)
        train_flow_variant(
            config,
            train_set,
            validation_set,
            paths["FLOW"].parent,
            "none",
            0.0,
            device,
            epochs,
        )
    if not paths["VTF_V2"].is_file():
        config = flow_training_config(benchmark, seed, tvk=True)
        config["training"]["epochs"] = epochs
        config["data"]["validation_sequences"] = list(fold.validation)
        print(f"training {fold.name} seed={seed} VTF_V2", flush=True)
        train_flow_variant(
            config,
            train_set,
            validation_set,
            paths["VTF_V2"].parent,
            "vehicle",
            float(benchmark["training"]["tvk_lambda"]),
            device,
            epochs,
        )
    return paths


def planners_for_fold(
    paths: Mapping[str, Path],
    benchmark: Mapping[str, Any],
    protocol: Mapping[str, Any],
    device: torch.device,
) -> dict[str, FlowPlanner | GuidedFlowPlanner]:
    candidates = int(benchmark["sampling"]["candidates"])
    steps = int(benchmark["sampling"]["integration_steps"])
    plan = FlowPlannerConfig(candidates=candidates, integration_steps=steps)
    flow_model = _load_flow(paths["FLOW"], device)
    vtf_model = _load_flow(paths["VTF_V2"], device)
    training = flow_training_config(benchmark, 0, tvk=True)
    terrain = TerrainFieldConfig(**training["terrain_field"])
    vehicle = VehicleConditionedFieldConfig(**training["vehicle_conditioning"])
    guidance = guidance_config(benchmark, use_kinematics=True)
    frozen = protocol["guidance"]
    guidance = replace(
        guidance,
        strength=float(frozen["eta"]),
        schedule=str(frozen["schedule"]),
        gamma=float(frozen["gamma"]),
        smoothing_kernel=str(frozen["smoothing_kernel"]),
        endpoint_projection=str(frozen["endpoint_projection"]),
    )
    return {
        "FLOW": FlowPlanner(flow_model, plan).to(device),
        "VTF_V2": GuidedFlowPlanner(
            vtf_model, plan, guidance, terrain, vehicle
        ).to(device),
    }


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ignored = {"scene_id", "sequence", "frame_id", "dataset_index", "method", "seed"}
    output: dict[str, Any] = {
        "method": rows[0]["method"],
        "seed": int(rows[0]["seed"]),
        "test_sequence": rows[0]["sequence"],
        "evaluated_scenes": len(rows),
    }
    for name in rows[0]:
        if name in ignored:
            continue
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite values in {name}")
        output[name] = float(values.mean())
    return output


def evaluate_fold(
    dataset: H10PlanningDataset,
    test_indices: Sequence[int],
    fold: SequenceHoldoutFold,
    seed: int,
    planners: Mapping[str, FlowPlanner | GuidedFlowPlanner],
    benchmark: Mapping[str, Any],
    protocol: Mapping[str, Any],
    envelopes: Mapping[str, Mapping[str, float]],
    output_root: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Evaluate paired methods in batches and retain scene-level source data."""

    training = flow_training_config(benchmark, seed, tvk=True)
    terrain = TerrainFieldConfig(**training["terrain_field"])
    vehicle = VehicleConditionedFieldConfig(**training["vehicle_conditioning"])
    kinematic = TrajectoryKinematicConfig(**benchmark["kinematic"])
    metric_config = PlanningClaimMetricConfig(
        planning_dt_s=float(benchmark["trajectory"]["planning_dt_s"]),
        occupancy_threshold=float(training["metrics"]["occupancy_threshold"]),
        traversability_threshold=1.0
        - float(training["metrics"]["nontraversable_threshold"]),
        normalized_slope_threshold=float(
            training["metrics"]["normalized_slope_threshold"]
        ),
    )
    loader = make_loader(
        Subset(dataset, list(test_indices)),
        int(protocol["batch_size_evaluation"]),
        shuffle=False,
        seed=seed + 701,
        num_workers=0,
    )
    candidates = int(benchmark["sampling"]["candidates"])
    horizon = int(benchmark["trajectory"]["horizon_steps"])
    all_summaries = []
    rows_by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    trajectory_chunks: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}
    ground_truth_chunks: list[np.ndarray] = []
    offset = 0
    for scene in loader:
        batch = scene.batch_size
        positions = list(range(offset, offset + batch))
        scene = scene.to(device)
        noise = _fixed_noise(positions, seed, candidates, horizon, device)
        components = derive_independent_component_maps(
            scene.terrain_map, terrain, metric_config
        )
        predictions = {}
        with torch.no_grad():
            predictions["FLOW"] = planners["FLOW"].sample(scene, noise)
        with torch.enable_grad():
            predictions["VTF_V2"] = planners["VTF_V2"].sample(scene, noise)
        ground_truth_chunks.append(scene.gt_future.detach().cpu().numpy())
        for method, prediction in predictions.items():
            trajectories = prediction.trajectories.detach()
            standard = trajectory_metrics(trajectories, scene.gt_future)
            terminal = terminal_scene_metrics(
                trajectories,
                scene.gt_future,
                scene.terrain_map,
                terrain,
                vehicle,
                training["metrics"],
                planning_dt_s=float(benchmark["trajectory"]["planning_dt_s"]),
                kinematic_config=kinematic,
            )
            independent = candidate_claim_metrics(
                trajectories, scene.goal, components, metric_config, kinematic
            )
            masks = {
                label: compliance_mask(independent, envelope)
                for label, envelope in envelopes.items()
            }
            for local, metadata in enumerate(scene.metadata):
                row: dict[str, Any] = {
                    "scene_id": scene_identifier(
                        metadata, int(test_indices[offset + local])
                    ),
                    "sequence": fold.test_sequence,
                    "frame_id": int(
                        metadata.get("frame_id", metadata.get("frame_index"))
                    ),
                    "dataset_index": int(test_indices[offset + local]),
                    "method": method,
                    "seed": seed,
                    "ADE_candidate0_m": float(
                        standard["ADE_by_candidate_m"][local, 0]
                    ),
                    "minADE@K_m": float(standard["minADE@K_m"][local]),
                    "mean_unified_tvk_cost": float(
                        terminal["mean_unified_tvk_cost"][local]
                    ),
                    "terrain_violation_rate": float(
                        terminal["terrain_violation_rate"][local]
                    ),
                }
                for name, values in independent.items():
                    output_name = {
                        "curvature_violation_rate": "curvature_violation_rate_independent",
                        "lateral_acceleration_violation_rate": "lateral_acceleration_violation_rate_independent",
                        "smoothness_m": "smoothness_m_independent",
                    }.get(name, name)
                    row[output_name] = float(values[local].mean())
                for label, mask in masks.items():
                    row[f"compliant_candidate_rate_{label}"] = float(
                        mask[local].float().mean()
                    )
                    row[f"GCCR_at_K_{label}"] = float(mask[local].any())
                rows_by_method[method].append(row)
            trajectory_chunks[method].append(trajectories.cpu().numpy())
        offset += batch
        print(
            f"{fold.name} seed={seed}: {offset}/{len(test_indices)}",
            flush=True,
        )
    for method in METHODS:
        run_root = output_root / "runs" / fold.name / f"seed_{seed}" / method
        run_root.mkdir(parents=True, exist_ok=True)
        write_csv(run_root / "scene_level_metrics.csv", rows_by_method[method])
        summary = summarize_rows(rows_by_method[method])
        save_json(run_root / "summary.json", summary)
        np.savez_compressed(
            run_root / "predictions.npz",
            trajectories=np.concatenate(trajectory_chunks[method]),
            ground_truth=np.concatenate(ground_truth_chunks),
        )
        all_summaries.append(summary)
    return all_summaries


def cached_fold_summaries(
    fold: SequenceHoldoutFold,
    seed: int,
    output_root: Path,
) -> list[dict[str, Any]] | None:
    rows = []
    for method in METHODS:
        path = output_root / "runs" / fold.name / f"seed_{seed}" / method / "summary.json"
        if not path.is_file():
            return None
        rows.append(load_json(path))
    return rows


def write_aggregate_outputs(
    summaries: Sequence[Mapping[str, Any]], output_root: Path
) -> None:
    run_summary = pd.DataFrame(summaries).sort_values(
        ["test_sequence", "seed", "method"]
    )
    run_summary.to_csv(output_root / "run_summary.csv", index=False)
    effects = sequence_level_method_effects(run_summary, METRICS)
    effects.to_csv(output_root / "sequence_level_effects.csv", index=False)
    flow_seed = run_summary[run_summary["method"] == "FLOW"].set_index(
        ["test_sequence", "seed"]
    )
    vtf_seed = run_summary[run_summary["method"] == "VTF_V2"].set_index(
        ["test_sequence", "seed"]
    )
    if not flow_seed.index.equals(vtf_seed.index):
        raise ValueError("seed-level paired runs do not align")
    rows = []
    for metric in METRICS:
        difference = effects[f"{metric}_difference"].to_numpy(dtype=np.float64)
        seed_difference = (
            vtf_seed[metric].to_numpy(dtype=np.float64)
            - flow_seed[metric].to_numpy(dtype=np.float64)
        )
        if metric in HIGHER_IS_BETTER:
            wins = difference > 0.0
            seed_wins = seed_difference > 0.0
        else:
            wins = difference < 0.0
            seed_wins = seed_difference < 0.0
        flow_mean = float(effects[f"{metric}_FLOW"].mean())
        target_mean = float(effects[f"{metric}_VTF_V2"].mean())
        mean_difference = float(difference.mean())
        rows.append(
            {
                "metric": metric,
                "n_held_out_sequences": len(effects),
                "n_seed_sequence_pairs": len(seed_difference),
                "flow_sequence_mean": flow_mean,
                "vtf_v2_sequence_mean": target_mean,
                "mean_paired_difference": mean_difference,
                "relative_change_pct": 100.0 * mean_difference / flow_mean,
                "paired_difference_sd": float(difference.std(ddof=1))
                if len(difference) > 1
                else float("nan"),
                "sequences_improved": int(wins.sum()),
                "seed_sequence_pairs_improved": int(seed_wins.sum()),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_root / "robustness_summary.csv", index=False)
    table_rows = []
    for row in rows:
        table_rows.append(
            "| {metric} | {flow:.6f} | {vtf:.6f} | {delta:+.6f} | {relative:+.2f}% | {wins}/{n} | {seed_wins}/{seed_n} |".format(
                metric=row["metric"],
                flow=row["flow_sequence_mean"],
                vtf=row["vtf_v2_sequence_mean"],
                delta=row["mean_paired_difference"],
                relative=row["relative_change_pct"],
                wins=row["sequences_improved"],
                n=row["n_held_out_sequences"],
                seed_wins=row["seed_sequence_pairs_improved"],
                seed_n=row["n_seed_sequence_pairs"],
            )
        )
    report = f"""# VTF-Flow 序列留出稳健性分析

## 统计设计

- 外层留出序列：{', '.join(effects['test_sequence'])}；
- 独立分析单位：留出序列，n={len(effects)}；
- 随机种子作为训练技术重复，先在每个序列内部平均；
- 相邻帧仅用于估计该序列的平均规划表现，不作为独立统计样本；
- 最终引导参数在运行前冻结为 terminal projection、eta=0.075；
- 00003 保持为开发验证序列，00004 不进入本次外层测试折；
- 所有运行均固定随机种子，但 PyTorch 提示 CUDA 上 adaptive pooling 与 grid sampling 的反向传播不保证逐比特确定性，因此种子重复用于估计实现层面的技术波动，而非声称 bitwise reproducibility。

## 序列级配对结果

| 指标 | Flow | VTF-Flow | VTF−Flow | 相对变化 | 改善序列 | 改善种子—序列对 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

## 解释边界

这是同一数据集内部的跨序列稳健性分析，不是外部数据集确认。由于独立序列数量较少，不报告把帧视为独立样本的 p 值；主要证据为序列级配对效应、方向一致性及其实际量级。CUDA 算子的确定性边界不改变已保存输出，但限制了跨硬件环境的逐比特复现声明。
"""
    (output_root / "sequence_holdout_report_zh.md").write_text(
        report, encoding="utf-8"
    )
    latex = summary[
        [
            "metric",
            "flow_sequence_mean",
            "vtf_v2_sequence_mean",
            "mean_paired_difference",
            "relative_change_pct",
            "sequences_improved",
            "seed_sequence_pairs_improved",
        ]
    ].rename(
        columns={
            "metric": "Metric",
            "flow_sequence_mean": "Flow",
            "vtf_v2_sequence_mean": "VTF-Flow",
            "mean_paired_difference": "Paired difference",
            "relative_change_pct": "Relative change (\\%)",
            "sequences_improved": "Sequences improved",
            "seed_sequence_pairs_improved": "Seed--sequence pairs improved",
        }
    )
    (output_root / "sequence_holdout_table.tex").write_text(
        latex.to_latex(index=False, escape=False, float_format="%.4f"),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--folds", nargs="+")
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    benchmark = load_json(args.benchmark)
    seeds = args.seeds or [int(protocol["screening_seed"])]
    requested_folds = {
        str(value).zfill(5) for value in (args.folds or protocol["outer_test_sequences"])
    }
    folds = build_fixed_validation_holdouts(
        protocol["available_sequences"],
        protocol["development_validation_sequence"],
        protocol["outer_test_sequences"],
    )
    folds = tuple(fold for fold in folds if fold.test_sequence in requested_folds)
    if not folds:
        raise ValueError("no requested outer folds remain")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = CombinedSceneDataset(
        args.cache_root.resolve(), tuple(protocol["source_splits"])
    )
    dataset = H10PlanningDataset(
        source,
        args.data_root.resolve() / "processed" / "Rellis-3D",
        horizon=int(benchmark["trajectory"]["horizon_steps"]),
        history_steps=int(benchmark["trajectory"]["history_steps"]),
    )
    envelopes_path = TERRAFLOW_ROOT / protocol["candidate_envelopes"]
    envelopes = load_json(envelopes_path)
    summaries: list[dict[str, Any]] = []
    effective_folds = []
    for fold in folds:
        split = as_sequence_split(fold)
        indices = partition_sequence_indices(dataset.sequence_ids, split)
        effective_folds.append(
            {
                "fold": fold.name,
                "train": list(fold.train),
                "validation": list(fold.validation),
                "test": list(fold.test),
                "counts": {name: len(value) for name, value in indices.items()},
            }
        )
        for seed in seeds:
            cached = cached_fold_summaries(fold, seed, output_root)
            if cached is not None:
                summaries.extend(cached)
                print(f"using cached {fold.name} seed={seed}", flush=True)
                continue
            paths = train_fold_models(
                dataset,
                indices,
                fold,
                seed,
                benchmark,
                protocol,
                output_root,
                device,
                args.skip_training,
            )
            planners = planners_for_fold(paths, benchmark, protocol, device)
            summaries.extend(
                evaluate_fold(
                    dataset,
                    indices["test"],
                    fold,
                    seed,
                    planners,
                    benchmark,
                    protocol,
                    envelopes,
                    output_root,
                    device,
                )
            )
            del planners
            if device.type == "cuda":
                torch.cuda.empty_cache()
    save_json(
        output_root / "effective_protocol.json",
        {
            **protocol,
            "executed_seeds": seeds,
            "executed_folds": effective_folds,
            "device": str(device),
            "test_sequence_used_for_parameter_selection": False,
        },
    )
    write_aggregate_outputs(summaries, output_root)
    print(output_root / "sequence_holdout_report_zh.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

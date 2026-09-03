"""Validate in-flow TVK correction on the validation sequence only."""

from __future__ import annotations

import argparse
import csv
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
    partition_sequence_indices,
    terminal_scene_metrics,
)
from TerraFlow.guidance.feasibility_flow_guidance import (  # noqa: E402
    FeasibilityFlowGuidanceConfig,
)
from TerraFlow.metrics.trajectory_metrics import trajectory_metrics  # noqa: E402
from TerraFlow.planners.flow_planner import FlowPlanner, FlowPlannerConfig  # noqa: E402
from TerraFlow.planners.guided_flow_planner import GuidedFlowPlanner  # noqa: E402
from TerraFlow.scripts.optimize_vtf_flow_validation import _fixed_noise  # noqa: E402
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


DEFAULT_MECHANISM = TERRAFLOW_ROOT / "configs" / "inflow_mechanism_validation.json"
DEFAULT_BENCHMARK = TERRAFLOW_ROOT / "outputs" / "unified_h10_benchmark"
DEFAULT_OUTPUT = TERRAFLOW_ROOT / "outputs" / "inflow_mechanism_validation"
CURVE_METRICS = (
    "ADE_candidate0_m",
    "minADE@K_m",
    "mean_unified_tvk_cost",
    "terrain_violation_rate",
    "curvature_violation_rate",
    "smoothness_m",
    "goal_error_mean_m",
)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def history_metrics(
    history: torch.Tensor,
    scene: Any,
    terrain: TerrainFieldConfig,
    vehicle: VehicleConditionedFieldConfig,
    thresholds: Mapping[str, float],
    kinematic: TrajectoryKinematicConfig,
    planning_dt_s: float,
) -> dict[str, torch.Tensor]:
    """Evaluate all Euler states at once and return ``[B,S]`` metrics."""

    if history.ndim != 5:
        raise ValueError("history must have shape [B,K,S,H,3]")
    batch, candidates, steps, horizon, dimension = history.shape
    trajectories = history.permute(0, 2, 1, 3, 4).reshape(
        batch * steps, candidates, horizon, dimension
    )
    ground_truth = scene.gt_future.repeat_interleave(steps, dim=0)
    terrain_map = scene.terrain_map.repeat_interleave(steps, dim=0)
    goals = scene.goal.repeat_interleave(steps, dim=0)
    standard = trajectory_metrics(trajectories, ground_truth)
    terminal = terminal_scene_metrics(
        trajectories,
        ground_truth,
        terrain_map,
        terrain,
        vehicle,
        thresholds,
        planning_dt_s=planning_dt_s,
        kinematic_config=kinematic,
    )
    goal_error = torch.linalg.vector_norm(
        trajectories[:, :, -1] - goals[:, None], dim=-1
    ).mean(dim=1)
    values = {
        "ADE_candidate0_m": standard["ADE_by_candidate_m"][:, 0],
        "minADE@K_m": standard["minADE@K_m"],
        "mean_unified_tvk_cost": terminal["mean_unified_tvk_cost"],
        "terrain_violation_rate": terminal["terrain_violation_rate"],
        "curvature_violation_rate": terminal["curvature_violation_rate"],
        "smoothness_m": terminal["smoothness_m"],
        "goal_error_mean_m": goal_error,
    }
    return {name: value.reshape(batch, steps) for name, value in values.items()}


def variant_guidance(
    benchmark: Mapping[str, Any], variant: Mapping[str, Any]
) -> FeasibilityFlowGuidanceConfig:
    base = guidance_config(benchmark, use_kinematics=True)
    values = dict(base.__dict__)
    values.update(
        strength=float(variant["eta"]),
        schedule=str(variant["schedule"]),
        gamma=float(variant["gamma"]),
        smoothing_kernel=str(variant["smoothing_kernel"]),
        endpoint_projection=str(variant["endpoint_projection"]),
    )
    return FeasibilityFlowGuidanceConfig(**values)


def summarize_final_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    reference = frame[frame.variant == "unguided_same_vtf_checkpoint"]
    for variant in frame.variant.unique():
        if variant == "unguided_same_vtf_checkpoint":
            continue
        guided = frame[frame.variant == variant]
        paired = reference.merge(
            guided,
            on=["seed", "validation_position"],
            suffixes=("_reference", "_guided"),
            validate="one_to_one",
        )
        row: dict[str, Any] = {
            "variant": variant,
            "n_pairs": len(paired),
        }
        for metric in CURVE_METRICS:
            ref = paired[f"{metric}_reference"].to_numpy(dtype=np.float64)
            val = paired[f"{metric}_guided"].to_numpy(dtype=np.float64)
            difference = val - ref
            row[f"{metric}_reference"] = float(ref.mean())
            row[f"{metric}_guided"] = float(val.mean())
            row[f"{metric}_difference"] = float(difference.mean())
            row[f"{metric}_improvement_rate"] = float((difference < -1e-10).mean())
        for metric in (
            "mean_output_displacement_m",
            "p95_output_displacement_m",
            "maximum_output_displacement_m",
            "terminal_output_displacement_m",
        ):
            row[metric] = float(paired[f"{metric}_guided"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def select_pareto_variant(
    summary: pd.DataFrame,
    constraints: Mapping[str, float],
) -> tuple[pd.Series, pd.DataFrame]:
    """Select maximum TVK reduction under predeclared behavior constraints."""

    audited = summary.copy()
    audited["relative_ade_increase"] = (
        audited["ADE_candidate0_m_difference"]
        / audited["ADE_candidate0_m_reference"].clip(lower=1e-12)
    )
    audited["relative_minade_increase"] = (
        audited["minADE@K_m_difference"]
        / audited["minADE@K_m_reference"].clip(lower=1e-12)
    )
    audited["admissible"] = (
        (
            audited["relative_ade_increase"]
            <= float(constraints["maximum_relative_ade_increase"])
        )
        & (
            audited["relative_minade_increase"]
            <= float(constraints["maximum_relative_minade_increase"])
        )
        & (
            audited["goal_error_mean_m_difference"]
            <= float(constraints["maximum_goal_error_increase_m"])
        )
        & (
            audited["smoothness_m_difference"]
            <= float(constraints["maximum_smoothness_increase_m"])
        )
    )
    admissible = audited[audited["admissible"]]
    if admissible.empty:
        raise RuntimeError("no guidance variant satisfies the selection constraints")
    selected = admissible.sort_values(
        ["mean_unified_tvk_cost_difference", "minADE@K_m_difference"]
    ).iloc[0]
    return selected, audited


def write_mechanism_report(
    path: Path,
    audited: pd.DataFrame,
    selected: pd.Series,
    constraints: Mapping[str, float],
) -> None:
    """Write a Chinese validation-only mechanism and selection report."""

    rows = []
    for _, row in audited.iterrows():
        rows.append(
            "| {variant} | {dade:+.6f} | {dmin:+.6f} | {dtvk:+.6f} | "
            "{dterrain:+.6f} | {dcurv:+.6f} | {dsmooth:+.6f} | {dgoal:+.6f} | {ok} |".format(
                variant=row["variant"],
                dade=row["ADE_candidate0_m_difference"],
                dmin=row["minADE@K_m_difference"],
                dtvk=row["mean_unified_tvk_cost_difference"],
                dterrain=row["terrain_violation_rate_difference"],
                dcurv=row["curvature_violation_rate_difference"],
                dsmooth=row["smoothness_m_difference"],
                dgoal=row["goal_error_mean_m_difference"],
                ok="是" if row["admissible"] else "否",
            )
        )
    content = f"""# VTF-Flow 生成期机制验证与参数选择

## 验证边界

本实验仅使用验证序列 00003，在相同完整 VTF-Flow 权重、相同初始噪声和相同 16 步 Euler 积分下，将各引导变体与无引导轨迹逐场景配对。测试序列 00004 未参与本轮参数选择。

## 预声明选择约束

- ADE-0 相对增幅不超过 {float(constraints['maximum_relative_ade_increase']):.2%}；
- minADE@8 相对增幅不超过 {float(constraints['maximum_relative_minade_increase']):.2%}；
- 平均目标误差增量不超过 {float(constraints['maximum_goal_error_increase_m']):.4f} m；
- 平顺性指标不得增加。

在满足上述行为保持约束的方案中，选择平均 TVK 势下降最多者。

## 配对结果

| 引导方案 | ΔADE-0 (m) | ΔminADE@8 (m) | ΔTVK | Δ地形违规率 | Δ曲率违规率 | Δ平顺性 | Δ目标误差 (m) | 可接受 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
{chr(10).join(rows)}

## 选择结果

验证集选择方案为 **{selected['variant']}**。相对于同权重无引导 Flow，其平均 TVK 势变化为 {selected['mean_unified_tvk_cost_difference']:+.6f}，ADE-0 变化为 {selected['ADE_candidate0_m_difference']:+.6f} m，minADE@8 变化为 {selected['minADE@K_m_difference']:+.6f} m，平均输出位移为 {selected['mean_output_displacement_m']:.6f} m。

这说明现有强引导的行为偏移主要来自未保持目标端点且强度偏大；端点投影和较弱引导可在保留生成先验的同时稳定降低 TVK 势。该配置是下一轮独立测试的候选版本，不能用当前已查看过的测试序列重新调参后替换既有主结果。
"""
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mechanism-config", type=Path, default=DEFAULT_MECHANISM)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark = load_json(args.benchmark_config)
    mechanism = load_json(args.mechanism_config)
    if mechanism["selection_split"] != "validation":
        raise ValueError("mechanism validation is restricted to validation data")
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
    split_indices = partition_sequence_indices(dataset.sequence_ids, benchmark_split(benchmark))
    available = split_indices["validation"]
    count = min(int(mechanism["scene_count"]), len(available))
    validation_positions = np.linspace(
        0, len(available) - 1, num=count, dtype=np.int64
    ).tolist()
    selected_indices = [available[position] for position in validation_positions]
    loader = make_loader(
        Subset(dataset, selected_indices),
        int(mechanism["batch_size"]),
        shuffle=False,
        seed=514,
        num_workers=0,
    )
    final_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    steps = int(benchmark["sampling"]["integration_steps"])
    candidates = int(benchmark["sampling"]["candidates"])
    horizon = int(benchmark["trajectory"]["horizon_steps"])
    for seed_value in mechanism["seeds"]:
        seed = int(seed_value)
        checkpoint = (
            args.benchmark_root.resolve()
            / "checkpoints" / f"seed_{seed}" / "flow_tvk" / "best.pt"
        )
        model = _load_flow(checkpoint, device)
        plan = FlowPlannerConfig(
            candidates=candidates,
            integration_steps=steps,
            save_integration_history=True,
        )
        flow_config = flow_training_config(benchmark, seed, tvk=True)
        terrain = TerrainFieldConfig(**flow_config["terrain_field"])
        vehicle = VehicleConditionedFieldConfig(**flow_config["vehicle_conditioning"])
        kinematic = TrajectoryKinematicConfig(**benchmark["kinematic"])
        reference_planner = FlowPlanner(model, plan).to(device)
        planners = {
            str(variant["name"]): GuidedFlowPlanner(
                model,
                plan,
                variant_guidance(benchmark, variant),
                terrain,
                vehicle,
            ).to(device)
            for variant in mechanism["variants"]
        }
        offset = 0
        for scene in loader:
            scene = scene.to(device)
            batch = scene.batch_size
            positions = validation_positions[offset : offset + batch]
            noise = _fixed_noise(positions, seed, candidates, horizon, device)
            with torch.no_grad():
                reference_prediction = reference_planner.sample(scene, noise)
            if reference_prediction.integration_history is None:
                raise AssertionError("reference history was not retained")
            predictions = {"unguided_same_vtf_checkpoint": reference_prediction}
            for name, planner in planners.items():
                with torch.enable_grad():
                    predictions[name] = planner.sample(scene, noise)
            reference_final = reference_prediction.trajectories
            for variant, prediction in predictions.items():
                if prediction.integration_history is None:
                    raise AssertionError("integration history was not retained")
                curves = history_metrics(
                    prediction.integration_history,
                    scene,
                    terrain,
                    vehicle,
                    flow_config["metrics"],
                    kinematic,
                    float(benchmark["trajectory"]["planning_dt_s"]),
                )
                displacement = torch.linalg.vector_norm(
                    prediction.trajectories - reference_final, dim=-1
                )
                for local, position in enumerate(positions):
                    final_row: dict[str, Any] = {
                        "variant": variant,
                        "seed": seed,
                        "validation_position": int(position),
                        "dataset_index": int(selected_indices[offset + local]),
                        "mean_output_displacement_m": float(displacement[local].mean()),
                        "p95_output_displacement_m": float(
                            torch.quantile(displacement[local], 0.95)
                        ),
                        "maximum_output_displacement_m": float(displacement[local].max()),
                        "terminal_output_displacement_m": float(
                            displacement[local, :, -1].mean()
                        ),
                    }
                    scene_curve_rows = [
                        {
                            "variant": variant,
                            "seed": seed,
                            "validation_position": int(position),
                            "step": step + 1,
                        }
                        for step in range(steps)
                    ]
                    for metric, values in curves.items():
                        final_row[metric] = float(values[local, -1])
                        for step in range(steps):
                            scene_curve_rows[step][metric] = float(values[local, step])
                    curve_rows.extend(scene_curve_rows)
                    final_rows.append(final_row)
            offset += batch
            print(f"seed={seed}: {offset}/{count}", flush=True)
        del model, planners, reference_planner
        torch.cuda.empty_cache() if device.type == "cuda" else None

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    final_frame = pd.DataFrame(final_rows)
    curve_frame = pd.DataFrame(curve_rows)
    final_frame.to_csv(output_root / "scene_level_final_pairs.csv", index=False)
    curve_frame.to_csv(output_root / "integration_curves.csv", index=False)
    summary = summarize_final_pairs(final_frame)
    selected, audited = select_pareto_variant(
        summary, mechanism["selection_constraints"]
    )
    audited.to_csv(output_root / "variant_summary.csv", index=False)
    seed_summaries = []
    for seed, seed_frame in final_frame.groupby("seed"):
        seed_summary = summarize_final_pairs(seed_frame)
        seed_summary.insert(1, "seed", int(seed))
        seed_summaries.append(seed_summary)
    pd.concat(seed_summaries, ignore_index=True).to_csv(
        output_root / "variant_seed_summary.csv", index=False
    )
    curve_summary = (
        curve_frame.groupby(["variant", "step"], as_index=False)[list(CURVE_METRICS)]
        .agg(["mean", "std"])
    )
    curve_summary.columns = [
        "_".join(str(value) for value in column if str(value))
        for column in curve_summary.columns.to_flat_index()
    ]
    curve_summary.to_csv(output_root / "integration_curve_summary.csv", index=False)
    selected_protocol = {
        "status": "validation_selected_not_independently_retested",
        "selection_split": mechanism["selection_split"],
        "selection_sequence": mechanism["selection_sequence"],
        "selected_variant": selected["variant"],
        "selection_constraints": mechanism["selection_constraints"],
        "guidance": next(
            variant
            for variant in mechanism["variants"]
            if variant["name"] == selected["variant"]
        ),
    }
    (output_root / "selected_validation_variant.json").write_text(
        json.dumps(selected_protocol, indent=2), encoding="utf-8"
    )
    write_mechanism_report(
        output_root / "inflow_mechanism_report_zh.md",
        audited,
        selected,
        mechanism["selection_constraints"],
    )
    (output_root / "effective_protocol.json").write_text(
        json.dumps(
            {
                "selection_split": "validation",
                "selection_sequence": mechanism["selection_sequence"],
                "scene_count": count,
                "seeds": mechanism["seeds"],
                "same_checkpoint_and_noise": True,
                "test_sequence_consulted": False,
                "variants": mechanism["variants"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"selected_validation_variant={selected['variant']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

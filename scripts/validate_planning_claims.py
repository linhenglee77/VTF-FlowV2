"""Audit VTF-Flow claims with independent BEV components and candidate availability.

Thresholds for the demonstration constraint envelope are fitted only from the
validation-sequence GT component distributions.  Frozen test predictions are
then evaluated without changing any model or test-time parameter.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch


TERRAFLOW_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TERRAFLOW_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from TerraFlow.evaluation.final_experiments import (  # noqa: E402
    partition_sequence_indices,
    scene_identifier,
)
from TerraFlow.evaluation.planning_claim_metrics import (  # noqa: E402
    PlanningClaimMetricConfig,
    candidate_claim_metrics,
    compliance_mask,
    compliant_diversity,
    derive_independent_component_maps,
    fit_demonstration_envelope,
)
from TerraFlow.scripts.run_unified_h10_benchmark import (  # noqa: E402
    H10PlanningDataset,
    benchmark_split,
)
from TerraFlow.scripts.train_regression import CombinedSceneDataset  # noqa: E402
from TerraFlow.terrain.feasibility_field import TerrainFieldConfig  # noqa: E402
from TerraFlow.terrain.trajectory_kinematics import (  # noqa: E402
    TrajectoryKinematicConfig,
)


DEFAULT_CONFIG = TERRAFLOW_ROOT / "configs" / "unified_h10_benchmark.json"
DEFAULT_CACHE = WORKSPACE_ROOT / "data" / "RELLIS3D" / "trajectory_cache_h150_s5"
DEFAULT_DATA = WORKSPACE_ROOT / "data" / "RELLIS3D"
DEFAULT_BENCHMARK = TERRAFLOW_ROOT / "outputs" / "unified_h10_benchmark"
DEFAULT_OPTIMIZED = TERRAFLOW_ROOT / "outputs" / "unified_h10_benchmark_optimized"
DEFAULT_OUTPUT = TERRAFLOW_ROOT / "outputs" / "planning_claim_validation"

DISPLAY_NAMES = {
    "CV": "Constant Velocity",
    "ASTAR": "A* terrain planner",
    "REG": "Deterministic regression",
    "FLOW": "Flow Matching",
    "VTF": "VTF-Flow (complete)",
    "VTF_OPT": "VTF-Flow w/o TVK training (optimized guidance)",
}
METHOD_ORDER = ["CV", "ASTAR", "REG", "FLOW", "VTF", "VTF_OPT"]
SCENE_METRICS = (
    "ADE_candidate0_m",
    "minADE@K_m",
    "mean_unified_tvk_cost",
    "terrain_violation_rate",
    "goal_error_mean_m",
    "occupancy_exposure_rate",
    "nontraversable_exposure_rate",
    "slope_exposure_rate",
    "roughness_mean",
    "clearance_min_m",
    "clearance_q05_m",
    "curvature_violation_rate_independent",
    "lateral_acceleration_violation_rate_independent",
    "smoothness_m_independent",
    "compliant_candidate_rate",
    "GCCR_at_K",
    "multi_compliant_scene",
    "compliant_diversity_m",
    "compliant_candidate_rate_q80",
    "GCCR_at_K_q80",
    "compliant_candidate_rate_q90",
    "GCCR_at_K_q90",
    "compliant_candidate_rate_q95",
    "GCCR_at_K_q95",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_specs(benchmark_root: Path, optimized_root: Path) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    seeds_by_method = {
        "CV": [0], "ASTAR": [0], "REG": [0, 1, 2],
        "FLOW": [0, 1, 2], "VTF": [0, 1, 2],
    }
    for method in METHOD_ORDER:
        if method == "VTF_OPT":
            seeds = [0, 1, 2]
            root = optimized_root
        else:
            seeds = seeds_by_method.get(method, [])
            root = benchmark_root
        for seed in seeds:
            directory = root / "runs" / f"{method}_seed{seed}"
            prediction_path = directory / "predictions.npz"
            metric_path = directory / "scene_level_metrics.csv"
            if prediction_path.is_file() and metric_path.is_file():
                specs.append(
                    {
                        "method": method,
                        "seed": seed,
                        "directory": directory,
                        "prediction_path": prediction_path,
                        "metric_path": metric_path,
                    }
                )
    return specs


def stack_scenes(
    dataset: H10PlanningDataset, indices: Iterable[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    scenes = [dataset[int(index)] for index in indices]
    maps = torch.stack([scene.terrain_map for scene in scenes])
    ground_truth = torch.stack([scene.gt_future for scene in scenes])
    goals = torch.stack([scene.goal for scene in scenes])
    identifiers = [
        scene_identifier(scene.metadata, int(index))
        for scene, index in zip(scenes, indices)
    ]
    return maps, ground_truth, goals, identifiers


def fit_validation_envelopes(
    dataset: H10PlanningDataset,
    validation_indices: list[int],
    terrain_config: TerrainFieldConfig,
    metric_config: PlanningClaimMetricConfig,
    kinematic_config: TrajectoryKinematicConfig,
    batch_size: int,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    collected: dict[str, list[torch.Tensor]] = {}
    for start in range(0, len(validation_indices), batch_size):
        batch_indices = validation_indices[start : start + batch_size]
        maps, ground_truth, goals, _ = stack_scenes(dataset, batch_indices)
        components = derive_independent_component_maps(
            maps, terrain_config, metric_config
        )
        metrics = candidate_claim_metrics(
            ground_truth[:, None], goals, components, metric_config, kinematic_config
        )
        for name, values in metrics.items():
            collected.setdefault(name, []).append(values.cpu())
    joined = {name: torch.cat(values) for name, values in collected.items()}
    envelopes = {}
    for percentile in (80, 90, 95):
        upper = percentile / 100.0
        lower = 1.0 - upper
        variant = replace(
            metric_config,
            envelope_upper_quantile=upper,
            envelope_lower_quantile=lower,
        )
        envelopes[f"q{percentile}"] = fit_demonstration_envelope(joined, variant)
    validation_summary = {
        f"{name}_mean": float(values.mean())
        for name, values in joined.items()
    }
    return envelopes, validation_summary


def seed_and_method_summary(scene_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows: list[dict[str, Any]] = []
    for (method, seed), group in scene_frame.groupby(["method", "seed"], sort=False):
        row: dict[str, Any] = {
            "method": method,
            "display_name": DISPLAY_NAMES[str(method)],
            "seed": int(seed),
            "K": int(group["K"].iloc[0]),
            "evaluated_scenes": len(group),
        }
        for metric in SCENE_METRICS:
            values = group[metric].to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            row[metric] = float(finite.mean()) if finite.size else float("nan")
            row[f"{metric}_n"] = int(finite.size)
        seed_rows.append(row)
    seed_frame = pd.DataFrame(seed_rows)
    method_rows: list[dict[str, Any]] = []
    for method, group in seed_frame.groupby("method", sort=False):
        row = {
            "method": method,
            "display_name": DISPLAY_NAMES[str(method)],
            "K": int(group["K"].iloc[0]),
            "n_seeds": len(group),
            "evaluated_scenes": int(group["evaluated_scenes"].iloc[0]),
        }
        for metric in SCENE_METRICS:
            values = group[metric].to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(finite.mean()) if finite.size else float("nan")
            row[f"{metric}_sd"] = (
                float(finite.std(ddof=1)) if finite.size > 1 else float("nan")
            )
        method_rows.append(row)
    return seed_frame, pd.DataFrame(method_rows)


def paired_summary(scene_frame: pd.DataFrame) -> pd.DataFrame:
    flow = scene_frame[scene_frame.method == "FLOW"]
    vtf = scene_frame[scene_frame.method == "VTF"]
    paired = flow.merge(
        vtf,
        on=["scene_id", "seed"],
        suffixes=("_flow", "_vtf"),
        validate="one_to_one",
    )
    higher_is_better = {
        "clearance_min_m", "clearance_q05_m", "compliant_candidate_rate",
        "GCCR_at_K", "multi_compliant_scene", "compliant_diversity_m",
    }
    rows: list[dict[str, Any]] = []
    for metric in SCENE_METRICS:
        flow_values = paired[f"{metric}_flow"].to_numpy(dtype=np.float64)
        vtf_values = paired[f"{metric}_vtf"].to_numpy(dtype=np.float64)
        finite = np.isfinite(flow_values) & np.isfinite(vtf_values)
        flow_values, vtf_values = flow_values[finite], vtf_values[finite]
        difference = vtf_values - flow_values
        tolerance = 1e-10
        if metric in higher_is_better or metric.startswith(
            ("GCCR_at_K_q", "compliant_candidate_rate_q")
        ):
            wins = difference > tolerance
            losses = difference < -tolerance
        else:
            wins = difference < -tolerance
            losses = difference > tolerance
        ties = ~(wins | losses)
        non_ties = int(wins.sum() + losses.sum())
        rows.append(
            {
                "comparison": "VTF_minus_Flow",
                "metric": metric,
                "n_pairs": int(difference.size),
                "mean_flow": float(flow_values.mean()),
                "mean_vtf": float(vtf_values.mean()),
                "mean_difference": float(difference.mean()),
                "win_rate": float(wins.mean()),
                "tie_rate": float(ties.mean()),
                "loss_rate": float(losses.mean()),
                "non_tie_win_rate": (
                    float(wins.sum() / non_ties) if non_ties else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate_predictions(
    dataset: H10PlanningDataset,
    test_indices: list[int],
    specs: list[dict[str, Any]],
    envelopes: Mapping[str, Mapping[str, float]],
    terrain_config: TerrainFieldConfig,
    metric_config: PlanningClaimMetricConfig,
    kinematic_config: TrajectoryKinematicConfig,
    batch_size: int,
) -> pd.DataFrame:
    predictions: dict[tuple[str, int], np.ndarray] = {}
    metric_lookup: dict[tuple[str, int], dict[str, dict[str, str]]] = {}
    for spec in specs:
        archive = np.load(spec["prediction_path"])
        values = np.asarray(archive["trajectories"], dtype=np.float32)
        if values.shape[0] != len(test_indices):
            raise ValueError(f"scene count mismatch for {spec['prediction_path']}")
        key = (str(spec["method"]), int(spec["seed"]))
        predictions[key] = values
        rows = read_rows(spec["metric_path"])
        metric_lookup[key] = {row["scene_id"]: row for row in rows}

    output_rows: list[dict[str, Any]] = []
    for start in range(0, len(test_indices), batch_size):
        batch_indices = test_indices[start : start + batch_size]
        maps, _, goals, identifiers = stack_scenes(dataset, batch_indices)
        components = derive_independent_component_maps(
            maps, terrain_config, metric_config
        )
        for spec in specs:
            method, seed = str(spec["method"]), int(spec["seed"])
            key = (method, seed)
            trajectories = torch.from_numpy(predictions[key][start : start + len(batch_indices)])
            metrics = candidate_claim_metrics(
                trajectories, goals, components, metric_config, kinematic_config
            )
            mask = compliance_mask(metrics, envelopes["q95"])
            sensitivity_masks = {
                label: compliance_mask(metrics, envelope)
                for label, envelope in envelopes.items()
            }
            diversity, has_pair = compliant_diversity(trajectories, mask)
            for local, scene_id in enumerate(identifiers):
                source = metric_lookup[key][scene_id]
                row: dict[str, Any] = {
                    "scene_id": scene_id,
                    "sequence": source["sequence"],
                    "frame_id": int(source["frame_id"]),
                    "dataset_index": int(source["dataset_index"]),
                    "method": method,
                    "display_name": DISPLAY_NAMES[method],
                    "seed": seed,
                    "K": int(trajectories.shape[1]),
                    "ADE_candidate0_m": float(source["ADE_candidate0_m"]),
                    "minADE@K_m": float(source["minADE@K_m"]),
                    "mean_unified_tvk_cost": float(source["mean_unified_tvk_cost"]),
                    "terrain_violation_rate": float(source["terrain_violation_rate"]),
                }
                name_map = {
                    "goal_error_m": "goal_error_mean_m",
                    "curvature_violation_rate": "curvature_violation_rate_independent",
                    "lateral_acceleration_violation_rate": (
                        "lateral_acceleration_violation_rate_independent"
                    ),
                    "smoothness_m": "smoothness_m_independent",
                }
                for name, values in metrics.items():
                    output_name = name_map.get(name, name)
                    row[output_name] = float(values[local].mean())
                row["compliant_candidate_rate"] = float(mask[local].float().mean())
                row["GCCR_at_K"] = float(mask[local].any())
                row["multi_compliant_scene"] = float(has_pair[local])
                row["compliant_diversity_m"] = float(diversity[local])
                for label, sensitivity in sensitivity_masks.items():
                    row[f"compliant_candidate_rate_{label}"] = float(
                        sensitivity[local].float().mean()
                    )
                    row[f"GCCR_at_K_{label}"] = float(sensitivity[local].any())
                output_rows.append(row)
        print(
            f"independent metrics: {min(start + batch_size, len(test_indices))}/"
            f"{len(test_indices)} scenes",
            flush=True,
        )
    return pd.DataFrame(output_rows)


def write_report(
    output: Path,
    method_summary: pd.DataFrame,
    paired: pd.DataFrame,
    envelopes: Mapping[str, Mapping[str, float]],
) -> None:
    lookup = method_summary.set_index("method")
    flow, vtf = lookup.loc["FLOW"], lookup.loc["VTF"]
    pair = paired.set_index("metric")

    def value(row: pd.Series, name: str) -> float:
        return float(row[f"{name}_mean"])

    behavior_bounded = (
        value(vtf, "ADE_candidate0_m") <= 1.02 * value(flow, "ADE_candidate0_m")
        and value(vtf, "minADE@K_m") <= 1.02 * value(flow, "minADE@K_m")
    )
    raw_improvements = {
        "occupancy": value(vtf, "occupancy_exposure_rate") < value(flow, "occupancy_exposure_rate"),
        "nontraversability": value(vtf, "nontraversable_exposure_rate") < value(flow, "nontraversable_exposure_rate"),
        "slope": value(vtf, "slope_exposure_rate") < value(flow, "slope_exposure_rate"),
        "clearance": value(vtf, "clearance_q05_m") > value(flow, "clearance_q05_m"),
        "curvature": value(vtf, "curvature_violation_rate_independent") < value(flow, "curvature_violation_rate_independent"),
    }
    availability_improved = value(vtf, "GCCR_at_K") > value(flow, "GCCR_at_K")
    potential_win = float(pair.loc["mean_unified_tvk_cost", "win_rate"])
    lines = [
        "# VTF-Flow 建模主张验证报告",
        "",
        "## 结论摘要",
        "",
        f"- 行为偏差保持在相对 Flow 的 2% 容差内：{'是' if behavior_bounded else '否'}。",
        f"- 冻结验证包络下 GCCR@K 提高：{'是' if availability_improved else '否'}。",
        f"- 独立分量改善数：{sum(raw_improvements.values())}/{len(raw_improvements)}。",
        f"- 完整 VTF-Flow 的 TVK 场景配对改善率：{potential_win * 100:.2f}%。",
        "",
        "这里的约束包络由验证序列 GT 的分量分布冻结，只表示与示范地形—运动学暴露范围一致；它不是安全概率或真实车辆失效标签。",
        "",
        "## 冻结示范约束包络",
        "",
    ]
    for label, envelope in envelopes.items():
        lines.append(f"- **{label}**")
        for name, threshold in envelope.items():
            lines.append(f"  - `{name}`: {threshold:.6f}")
    lines.extend(["", "## Flow 与完整 VTF-Flow", ""])
    names = [
        ("ADE_candidate0_m", "ADE-0"),
        ("minADE@K_m", "minADE@K"),
        ("GCCR_at_K", "GCCR@K"),
        ("compliant_candidate_rate", "候选约束满足率"),
        ("occupancy_exposure_rate", "原始占据暴露率"),
        ("nontraversable_exposure_rate", "非通行暴露率"),
        ("slope_exposure_rate", "坡度暴露率"),
        ("clearance_q05_m", "净空 Q05 (m)"),
        ("curvature_violation_rate_independent", "曲率违规率"),
        ("mean_unified_tvk_cost", "TVK 代价"),
    ]
    lines.append("| 指标 | Flow | 完整 VTF-Flow | VTF-Flow−Flow | 配对胜率 |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, label in names:
        row = pair.loc[name]
        lines.append(
            f"| {label} | {float(row['mean_flow']):.6f} | "
            f"{float(row['mean_vtf']):.6f} | {float(row['mean_difference']):+.6f} | "
            f"{float(row['win_rate']) * 100:.2f}% |"
        )
    lines.extend(["", "## 解释边界", ""])
    lines.extend(
        [
            "- TVK 代价是方法内部相对势；独立分量表用于降低循环论证风险。",
            "- 米制净空来自点云派生 BEV 占据掩膜的欧氏距离变换，不是车辆包络碰撞验证。",
            "- 1,909 个测试场景属于同一序列的相邻帧，配对胜率按场景描述，不用于声称跨序列显著性。",
            "- ADE/minADE 保留为行为合理性参考；GT 不是地形或车辆运动学最优轨迹。",
        ]
    )
    lines.extend(["", "## 风险—覆盖敏感性", ""])
    lines.append("| 包络 | Flow GCCR@8 | 完整 VTF-Flow GCCR@8 | 差值 |")
    lines.append("|---|---:|---:|---:|")
    for label in ("q80", "q90", "q95"):
        row = pair.loc[f"GCCR_at_K_{label}"]
        lines.append(
            f"| {label} | {float(row['mean_flow']):.6f} | "
            f"{float(row['mean_vtf']):.6f} | {float(row['mean_difference']):+.6f} |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--optimized-root", type=Path, default=DEFAULT_OPTIMIZED)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    benchmark = load_json(args.config)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    base = CombinedSceneDataset(
        args.cache_root.resolve(), tuple(benchmark["protocol"]["source_splits"])
    )
    dataset = H10PlanningDataset(
        base,
        args.data_root.resolve() / "processed" / "Rellis-3D",
        horizon=int(benchmark["trajectory"]["horizon_steps"]),
        history_steps=int(benchmark["trajectory"]["history_steps"]),
    )
    split = benchmark_split(benchmark)
    indices = partition_sequence_indices(dataset.sequence_ids, split)
    flow_config = load_json(
        args.benchmark_root.resolve()
        / "checkpoints" / "seed_0" / "flow_tvk" / "effective_config.json"
    )
    terrain_config = TerrainFieldConfig(**flow_config["terrain_field"])
    kinematic_config = TrajectoryKinematicConfig(**benchmark["kinematic"])
    metric_config = PlanningClaimMetricConfig(
        forward_m=terrain_config.forward_m,
        lateral_m=terrain_config.lateral_m,
        occupancy_threshold=float(flow_config["metrics"]["occupancy_threshold"]),
        traversability_threshold=(
            1.0 - float(flow_config["metrics"]["nontraversable_threshold"])
        ),
        normalized_slope_threshold=float(
            flow_config["metrics"]["normalized_slope_threshold"]
        ),
        planning_dt_s=float(benchmark["trajectory"]["planning_dt_s"]),
    )
    envelopes, validation_summary = fit_validation_envelopes(
        dataset,
        indices["validation"],
        terrain_config,
        metric_config,
        kinematic_config,
        args.batch_size,
    )
    save_json(output_root / "validation_demonstration_envelope.json", envelopes)
    save_json(output_root / "validation_gt_component_summary.json", validation_summary)
    specs = run_specs(args.benchmark_root.resolve(), args.optimized_root.resolve())
    scene_frame = evaluate_predictions(
        dataset,
        indices["test"],
        specs,
        envelopes,
        terrain_config,
        metric_config,
        kinematic_config,
        args.batch_size,
    )
    scene_frame.to_csv(output_root / "scene_level_claim_metrics.csv", index=False)
    seed_frame, method_frame = seed_and_method_summary(scene_frame)
    seed_frame.to_csv(output_root / "seed_summary.csv", index=False)
    method_frame.to_csv(output_root / "method_summary.csv", index=False)
    paired = paired_summary(scene_frame)
    paired.to_csv(output_root / "paired_flow_vs_complete_vtf.csv", index=False)
    write_report(
        output_root / "claim_validation_report_zh.md",
        method_frame,
        paired,
        envelopes,
    )
    save_json(
        output_root / "effective_protocol.json",
        {
            "split": split.as_dict(),
            "split_counts": {name: len(values) for name, values in indices.items()},
            "metric_config": metric_config.__dict__,
            "kinematic_config": kinematic_config.__dict__,
            "candidate_compliance_definition": (
                "goal tolerance plus component-wise validation-GT envelope"
            ),
            "test_parameters_selected_from_test": False,
            "clearance_interpretation": (
                "Euclidean distance on the point-cloud-derived BEV occupancy mask; "
                "not vehicle-body collision clearance"
            ),
        },
    )
    print(f"claim validation written to {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

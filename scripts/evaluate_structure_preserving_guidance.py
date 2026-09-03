"""Evaluate smoothing, trust-region, and adaptive feasibility Flow guidance."""

from __future__ import annotations

import argparse
import copy
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

from TerraFlow.guidance.structure_preserving_guidance import paired_correction_metrics  # noqa: E402
from TerraFlow.scripts.evaluate_guided_flow import (  # noqa: E402
    evaluate_config,
    load_variant_config,
    write_csv,
)
from TerraFlow.scripts.train_regression import (  # noqa: E402
    CombinedSceneDataset,
    make_loader,
    sequence_partition_indices,
)
from TerraFlow.terrain.feasibility_field import AnalyticTerrainField, TerrainFieldConfig  # noqa: E402
from TerraFlow.terrain.vehicle_conditioned_field import (  # noqa: E402
    BatchedVehicleConditionedTerrainField,
    VehicleConditionedFieldConfig,
    trajectory_motion_state,
)
from TerraFlow.visualization.plot_structure_preserving_guidance import (  # noqa: E402
    plot_flow_step_structure,
    plot_main_comparison,
    plot_multi_method_case,
)


DEFAULT_CONFIG = TERRAFLOW_ROOT / "configs" / "structure_preserving_guidance.json"
DEFAULT_OUTPUT = (
    TERRAFLOW_ROOT / "outputs" / "experiments" / "structure_preserving_guidance"
)
METHOD_LABELS = {
    "unguided": "Unguided",
    "raw": "Raw",
    "smoothed": "Smoothed",
    "trust": "Trust region",
    "smooth_trust": "Smooth + trust",
    "adaptive": "Adaptive",
}


def load_config(path: Path) -> dict[str, Any]:
    """Load the bounded Step 11.5 experiment grid."""

    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "base_variant_config", "checkpoint", "data", "sampling", "primary",
        "sweep", "adaptive_trigger", "selection", "seed",
    }
    if set(config) != required:
        raise ValueError(f"config keys must be exactly {sorted(required)}")
    if config["sweep"]["eta"] != [0.05, 0.10, 0.20, 0.50]:
        raise ValueError("eta sweep must remain the requested conservative grid")
    if config["sweep"]["rho"] != [0.10, 0.20, 0.30]:
        raise ValueError("rho sweep must remain [0.10,0.20,0.30]")
    return config


def method_config(
    base: Mapping[str, Any],
    method: str,
    eta: float,
    rho: float | None,
    trigger_reference_cost: float,
    experiment: Mapping[str, Any],
) -> dict[str, Any]:
    """Create one paired method config while changing only inference guidance."""

    config = copy.deepcopy(dict(base))
    primary = experiment["primary"]
    trigger_cfg = experiment["adaptive_trigger"]
    guidance = config["guidance"]
    guidance.update({
        "enabled": method != "unguided" and eta > 0.0,
        "strength": float(eta),
        "schedule": primary["schedule"],
        "smoothing_kernel": (
            primary["smoothing_kernel"]
            if method in {"smoothed", "smooth_trust", "adaptive"}
            else "none"
        ),
        "trust_region_rho": (
            float(rho) if method in {"trust", "smooth_trust", "adaptive"} else None
        ),
        "trust_region_scope": primary["trust_region_scope"],
        "adaptive_trigger_enabled": method == "adaptive",
        "trigger_alpha": float(trigger_cfg["alpha"]),
        "trigger_reference_cost": float(trigger_reference_cost),
    })
    config["name"] = f"structure_{method}_eta_{eta:g}_rho_{rho}"
    config["structure_method"] = method
    return config


@torch.no_grad()
def estimate_training_gt_reference(
    dataset: Subset,
    base_config: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    """Estimate GT vehicle-cost percentiles using training sequences only."""

    loader = make_loader(dataset, batch_size, shuffle=False, seed=73, num_workers=0)
    terrain_cfg = TerrainFieldConfig(**base_config["terrain_field"])
    vehicle_cfg = VehicleConditionedFieldConfig(**base_config["vehicle_conditioning"])
    costs: list[np.ndarray] = []
    for scene in loader:
        scene = scene.to(device)
        trajectories = scene.gt_future[..., :3]
        terrain_field = AnalyticTerrainField(scene.terrain_map, terrain_cfg)
        vehicle_field = BatchedVehicleConditionedTerrainField(terrain_field, vehicle_cfg)
        motion = trajectory_motion_state(
            trajectories, float(base_config["guidance"]["planning_dt_s"]), vehicle_cfg
        )
        cost = vehicle_field.cost(trajectories, motion).mean(dim=1)
        costs.append(cost.cpu().numpy())
    values = np.concatenate(costs)
    return {
        "source": "GT trajectories from training sequences only",
        "samples": int(values.size),
        "percentiles": {
            str(percentile): float(np.percentile(values, percentile))
            for percentile in (50.0, 75.0, 90.0)
        },
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def add_paired_metrics(
    result: dict[str, Any],
    baseline: Mapping[str, Any],
    epsilon: float,
) -> None:
    """Attach per-scene and aggregate structure-correction diagnostics."""

    guided_tensor = torch.from_numpy(result["predictions"]["trajectories"])
    base_tensor = torch.from_numpy(baseline["predictions"]["trajectories"])
    correction = paired_correction_metrics(guided_tensor, base_tensor)
    correction_numpy = {name: value.numpy() for name, value in correction.items()}
    result["correction_scene_metrics"] = correction_numpy
    for name, value in correction_numpy.items():
        result["metrics"][name] = float(value.mean())
    feasibility_gain = (
        baseline["metrics"]["mean_vehicle_conditioned_cost"]
        - result["metrics"]["mean_vehicle_conditioned_cost"]
    )
    fidelity_loss = max(
        result["metrics"]["minADE@K_m"] - baseline["metrics"]["minADE@K_m"], 0.0
    )
    smoothness_loss = max(
        result["metrics"]["smoothness_m"] - baseline["metrics"]["smoothness_m"], 0.0
    )
    result["metrics"].update({
        "feasibility_gain": feasibility_gain,
        "fidelity_loss": fidelity_loss,
        "smoothness_loss": smoothness_loss,
        "feasibility_gain_per_fidelity_loss": feasibility_gain / (fidelity_loss + epsilon),
        "feasibility_gain_per_smoothness_loss": feasibility_gain / (smoothness_loss + epsilon),
    })


def _result_row(
    method: str, eta: float, rho: float | None, result: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "method": METHOD_LABELS[method],
        "method_key": method,
        "eta": eta,
        "rho": "" if rho is None else rho,
        **result["metrics"],
    }


def _terrain_context(
    source: CombinedSceneDataset,
    dataset_index: int,
    terrain_config: TerrainFieldConfig,
) -> tuple[np.ndarray, np.ndarray]:
    scene = source[dataset_index].as_batch()
    field = AnalyticTerrainField(scene.terrain_map, terrain_config)
    height, width = scene.terrain_map.shape[-2:]
    x = torch.linspace(0.0, terrain_config.forward_m, height)
    y = torch.linspace(-terrain_config.lateral_m, terrain_config.lateral_m, width)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    query = torch.stack((xx, yy), dim=-1).reshape(1, -1, 2)
    feasibility = field.query(query).reshape(height, width).detach().cpu().numpy()
    return feasibility, scene.gt_future[0].cpu().numpy()


def analyze_cases(
    results: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Any],
    source: CombinedSceneDataset,
    terrain_config: TerrainFieldConfig,
    output_dir: Path,
    selection: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Audit the known failure plus deterministic improvement/trade-off cases."""

    combined = results["smooth_trust"]
    indices = baseline["predictions"]["dataset_indices"]
    known_matches = np.flatnonzero(indices == 6172)
    known = int(known_matches[0]) if known_matches.size == 1 else -1
    base_vehicle = baseline["scene_metrics"]["mean_vehicle_conditioned_cost"]
    combined_vehicle = combined["scene_metrics"]["mean_vehicle_conditioned_cost"]
    feasibility_gain = base_vehicle - combined_vehicle
    fidelity_loss = (
        combined["scene_metrics"]["minADE@K_m"]
        - baseline["scene_metrics"]["minADE@K_m"]
    )
    smoothness_loss = (
        combined["scene_metrics"]["smoothness_m"]
        - baseline["scene_metrics"]["smoothness_m"]
    )
    best_feasibility = int(np.argmax(feasibility_gain))
    worst_fidelity = int(np.argmax(fidelity_loss))
    if known < 0:
        known = worst_fidelity
    ade_limit = float(selection["maximum_relative_minade_degradation"])
    smooth_limit = float(selection["maximum_relative_smoothness_degradation"])
    eligible = (
        fidelity_loss <= ade_limit * np.maximum(
            baseline["scene_metrics"]["minADE@K_m"], 1e-6
        )
    ) & (
        smoothness_loss <= smooth_limit * np.maximum(
            baseline["scene_metrics"]["smoothness_m"], 1e-6
        )
    )
    eligible_positions = np.flatnonzero(eligible)
    if eligible_positions.size:
        best_tradeoff = int(
            eligible_positions[np.argmax(feasibility_gain[eligible_positions])]
        )
        tradeoff_rule = "maximum feasibility gain under per-scene 1% minADE and 5% smoothness limits"
    else:
        denominator = np.maximum(fidelity_loss, 0.0) + np.maximum(smoothness_loss, 0.0) + 1e-6
        best_tradeoff = int(np.argmax(feasibility_gain / denominator))
        tradeoff_rule = "fallback maximum feasibility gain divided by positive structure loss"
    cases = {
        "previous_failure": known,
        "best_feasibility_improvement": best_feasibility,
        "worst_fidelity_loss": worst_fidelity,
        "best_overall_tradeoff": best_tradeoff,
    }
    rows: list[dict[str, Any]] = []
    records: dict[str, Any] = {"selection_rule": tradeoff_rule, "cases": {}}
    output_dir.mkdir(parents=True, exist_ok=True)
    for case_name, position in cases.items():
        dataset_index = int(indices[position])
        records["cases"][case_name] = {
            "validation_position": position,
            "dataset_index": dataset_index,
            "combined_feasibility_gain": float(feasibility_gain[position]),
            "combined_minADE_change_m": float(fidelity_loss[position]),
            "combined_smoothness_change_m": float(smoothness_loss[position]),
        }
        trajectory_map: dict[str, np.ndarray] = {}
        for method in ("unguided", "raw", "smoothed", "trust", "smooth_trust"):
            result = results[method]
            trajectory_map[METHOD_LABELS[method]] = result["predictions"]["trajectories"][position]
            row = {
                "case": case_name,
                "dataset_index": dataset_index,
                "method": METHOD_LABELS[method],
                **{
                    name: float(values[position])
                    for name, values in result["scene_metrics"].items()
                },
                **{
                    name: float(values[position])
                    for name, values in result.get("correction_scene_metrics", {}).items()
                },
            }
            rows.append(row)
        context, ground_truth = _terrain_context(source, dataset_index, terrain_config)
        plot_multi_method_case(
            context,
            (-terrain_config.lateral_m, terrain_config.lateral_m, 0.0, terrain_config.forward_m),
            ground_truth,
            trajectory_map,
            output_dir / f"{case_name}_index_{dataset_index}.png",
            f"{case_name}: RELLIS validation index {dataset_index}",
        )
    (output_dir / "case_selection.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    return rows, records


def _write_report(
    path: Path,
    main_rows: Sequence[Mapping[str, Any]],
    step_rows: Sequence[Mapping[str, Any]],
    gt_reference: Mapping[str, Any],
    case_records: Mapping[str, Any],
) -> None:
    lookup = {row["method_key"]: row for row in main_rows}
    base, raw, smooth, trust, combined, adaptive = (
        lookup[name] for name in (
            "unguided", "raw", "smoothed", "trust", "smooth_trust", "adaptive"
        )
    )
    step_lookup: dict[str, list[Mapping[str, Any]]] = {}
    for row in step_rows:
        step_lookup.setdefault(str(row["method_key"]), []).append(row)
    combined_steps = sorted(step_lookup["smooth_trust"], key=lambda row: int(row["step"]))
    raw_ratio = float(np.mean([float(row["correction_flow_ratio_mean"]) for row in step_lookup["raw"]]))
    combined_ratio = float(np.mean([float(row["correction_flow_ratio_mean"]) for row in combined_steps]))
    raw_cos = float(np.mean([float(row["flow_gradient_cosine_similarity_mean"]) for row in step_lookup["raw"]]))
    # Prefer the simplest method when the configured trust region is inactive.
    # At the primary setting the trust and no-trust predictions can therefore
    # be exactly equal; calling the combined method superior would incorrectly
    # attribute the smoothing benefit to the trust region.
    retained = smooth
    if (
        adaptive["mean_vehicle_conditioned_cost"] < combined["mean_vehicle_conditioned_cost"]
        and adaptive["minADE@K_m"] <= combined["minADE@K_m"]
        and adaptive["smoothness_m"] <= combined["smoothness_m"]
    ):
        retained = adaptive
    lines = [
        "# Structure-Preserving Feasibility-Guided Flow",
        "",
        "训练好的 Flow、Euler solver、初始高斯噪声、验证序列和 vehicle-conditioned field 均保持不变。"
        "本实验只修改推理期 feasibility correction。",
        "",
        "## 主对照（eta=0.2，rho=0.2）",
        "",
        "| Method | minADE@K | vehicle cost | terrain violation | smoothness | max local 2nd diff | mean correction | max correction | diversity | latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in main_rows:
        lines.append(
            f"| {row['method']} | {row['minADE@K_m']:.4f} | "
            f"{row['mean_vehicle_conditioned_cost']:.4f} | {row['terrain_violation_rate']:.4f} | "
            f"{row['smoothness_m']:.4f} | {row['maximum_local_second_difference_m']:.4f} | "
            f"{row['mean_waypoint_correction_m']:.4f} | {row['maximum_waypoint_correction_m']:.4f} | "
            f"{row['diversity_m']:.4f} | {row['latency_ms_per_scene']:.3f} |"
        )
    lines.extend([
        "",
        "## 十个问题",
        "",
        f"1. **Raw guidance 为什么破坏 smoothness？** waypoint cost gradient 含沿 H 的高频分量；"
        f"raw 在归一化后每步施加 correction，smoothness 相对 unguided 变化 "
        f"{raw['smoothness_m']-base['smoothness_m']:+.4f} m，最大局部二阶差分变化 "
        f"{raw['maximum_local_second_difference_m']-base['maximum_local_second_difference_m']:+.4f} m。",
        f"2. **平滑是否减少局部畸变？** Smoothed 相对 Raw 的 smoothness 变化 "
        f"{smooth['smoothness_m']-raw['smoothness_m']:+.4f} m，最大局部二阶差分变化 "
        f"{smooth['maximum_local_second_difference_m']-raw['maximum_local_second_difference_m']:+.4f} m。",
        f"3. **Trust region 是否阻止 correction 主导 Flow？** Raw 平均 correction/Flow 比为 "
        f"{raw_ratio:.4f}，Smooth+Trust 为 {combined_ratio:.4f}。主配置下 trust region 未触发，"
        f"因此它只是保护上界，不能解释本次改善。",
        f"4. **组合方法是否改善 trade-off？** Smooth+Trust 相对 Raw 的 vehicle cost 差值 "
        f"{combined['mean_vehicle_conditioned_cost']-raw['mean_vehicle_conditioned_cost']:+.4f}，minADE 差值 "
        f"{combined['minADE@K_m']-raw['minADE@K_m']:+.4f} m，smoothness 差值 "
        f"{combined['smoothness_m']-raw['smoothness_m']:+.4f} m。",
        f"5. **Adaptive trigger 是否有用？** J_ref={gt_reference['percentiles']['75.0']:.4f} 仅来自训练 GT 75 分位，"
        f"不是安全阈值。Adaptive 相对组合方法的 vehicle cost/minADE/smoothness 差值依次为 "
        f"{adaptive['mean_vehicle_conditioned_cost']-combined['mean_vehicle_conditioned_cost']:+.4f}/"
        f"{adaptive['minADE@K_m']-combined['minADE@K_m']:+.4f}/"
        f"{adaptive['smoothness_m']-combined['smoothness_m']:+.4f}。",
        f"6. **实际 correction 多大？** 保留候选方法平均 waypoint correction 为 "
        f"{retained['mean_waypoint_correction_m']:.4f} m，endpoint correction 为 "
        f"{retained['endpoint_correction_m']:.4f} m。",
        f"7. **何时与 Flow 冲突？** Raw 的平均 cos(Flow, gradient)={raw_cos:.4f}；"
        f"逐步符号和变化见 flow_step_diagnostics.csv，正值意味着减去 gradient 会抵消部分 Flow 速度。",
        f"8. **是否保持 diversity？** 保留候选方法相对 unguided 的 diversity 变化 "
        f"{retained['diversity_m']-base['diversity_m']:+.4f} m。",
        f"9. **额外延迟？** 保留候选相对 unguided 增加 "
        f"{retained['latency_ms_per_scene']-base['latency_ms_per_scene']:+.3f} ms/scene。",
        f"10. **最终保留哪个设计？** 按 feasibility、minADE、smoothness、correction bound 和延迟共同判断，"
        f"当前保留 `{retained['method']}`；这不是通用最优性声明。",
        "",
        "## 失败案例与证据边界",
        "",
        f"复用了 dataset index 6172；同时按确定性规则选择最佳 feasibility、最差 fidelity 和最佳折中案例。"
        f"最佳折中规则为：{case_records['selection_rule']}。",
        "当前仍为单 seed、单验证序列；clearance 是缓存 BEV 邻障代理，J_ref 和所有 violation 均不是校准安全阈值。",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-scenes", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment = load_config(args.config)
    base_config = load_variant_config("flow_guided")
    base_config["sampling"] = experiment["sampling"]
    base_config["seed"] = experiment["seed"]
    source = CombinedSceneDataset(args.cache_root, tuple(experiment["data"]["source_splits"]))
    train_indices, validation_indices = sequence_partition_indices(
        source.sequence_ids, experiment["data"]["validation_sequences"]
    )
    if args.max_scenes is not None:
        validation_indices = validation_indices[: args.max_scenes]
    train_dataset = Subset(source, train_indices)
    validation_dataset = Subset(source, validation_indices)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    print(json.dumps({"device": str(device), "validation_scenes": len(validation_dataset)}), flush=True)

    gt_reference = estimate_training_gt_reference(
        train_dataset, base_config, device, args.batch_size * 2
    )
    reference_percentile = str(float(experiment["adaptive_trigger"]["gt_cost_percentile"]))
    trigger_reference = float(gt_reference["percentiles"][reference_percentile])
    (args.output_dir / "training_gt_cost_reference.json").write_text(
        json.dumps(gt_reference, indent=2), encoding="utf-8"
    )

    baseline_config = method_config(
        base_config, "unguided", 0.0, None, trigger_reference, experiment
    )
    baseline = evaluate_config(
        baseline_config, validation_dataset, validation_indices, device,
        args.batch_size, "unguided", True, False, True,
    )
    epsilon = float(experiment["selection"]["ratio_epsilon"])
    add_paired_metrics(baseline, baseline, epsilon)
    sweep_rows: list[dict[str, Any]] = []
    main_results: dict[str, dict[str, Any]] = {"unguided": baseline}
    primary_eta = float(experiment["primary"]["eta"])
    primary_rho = float(experiment["primary"]["trust_region_rho"])

    combinations: list[tuple[str, float, float | None]] = []
    for eta in experiment["sweep"]["eta"]:
        combinations.extend((("raw", eta, None), ("smoothed", eta, None)))
        for rho in experiment["sweep"]["rho"]:
            combinations.extend((("trust", eta, rho), ("smooth_trust", eta, rho)))
    for method, eta, rho in combinations:
        is_main = eta == primary_eta and (
            (rho is None and method in {"raw", "smoothed"})
            or (rho == primary_rho and method in {"trust", "smooth_trust"})
        )
        config = method_config(
            base_config, method, float(eta), None if rho is None else float(rho),
            trigger_reference, experiment,
        )
        result = evaluate_config(
            config, validation_dataset, validation_indices, device,
            args.batch_size, f"{method}_eta{eta}_rho{rho}", True, is_main, is_main,
        )
        add_paired_metrics(result, baseline, epsilon)
        sweep_rows.append(_result_row(method, float(eta), rho, result))
        if is_main:
            main_results[method] = result
        print(
            f"completed method={method} eta={eta} rho={rho} "
            f"minADE={result['metrics']['minADE@K_m']:.4f} "
            f"vehicle={result['metrics']['mean_vehicle_conditioned_cost']:.4f}",
            flush=True,
        )

    adaptive_config = method_config(
        base_config, "adaptive", primary_eta, primary_rho,
        trigger_reference, experiment,
    )
    adaptive = evaluate_config(
        adaptive_config, validation_dataset, validation_indices, device,
        args.batch_size, "adaptive", True, True, True,
    )
    add_paired_metrics(adaptive, baseline, epsilon)
    main_results["adaptive"] = adaptive

    main_rows = [_result_row("unguided", 0.0, None, baseline)] + [
        _result_row(method, primary_eta, primary_rho if method in {"trust", "smooth_trust", "adaptive"} else None, main_results[method])
        for method in ("raw", "smoothed", "trust", "smooth_trust", "adaptive")
    ]
    write_csv(args.output_dir / "main_comparison.csv", main_rows)
    write_csv(args.output_dir / "eta_rho_sweep.csv", sweep_rows)
    correction_names = (
        "mean_waypoint_correction_m", "maximum_waypoint_correction_m",
        "mean_trajectory_max_correction_m", "endpoint_correction_m",
        "second_difference_change_m", "maximum_local_second_difference_m",
        "feasibility_gain", "fidelity_loss", "smoothness_loss",
        "feasibility_gain_per_fidelity_loss", "feasibility_gain_per_smoothness_loss",
    )
    correction_rows = [
        {"method": row["method"], **{name: row[name] for name in correction_names}}
        for row in main_rows
    ]
    write_csv(args.output_dir / "correction_metrics.csv", correction_rows)

    step_rows: list[dict[str, Any]] = []
    for method in ("raw", "smoothed", "trust", "smooth_trust", "adaptive"):
        for row in main_results[method]["evolution"]:
            step_rows.append({"method": METHOD_LABELS[method], "method_key": method, **row})
    write_csv(args.output_dir / "flow_step_diagnostics.csv", step_rows)
    latency_rows = [
        {
            "method": row["method"],
            "latency_ms_per_scene": row["latency_ms_per_scene"],
            "guidance_overhead_ms_per_scene": (
                row["latency_ms_per_scene"] - main_rows[0]["latency_ms_per_scene"]
            ),
            "integration_steps": row["integration_steps"],
        }
        for row in main_rows
    ]
    write_csv(args.output_dir / "latency.csv", latency_rows)

    failure_rows, case_records = analyze_cases(
        main_results, baseline, source,
        TerrainFieldConfig(**base_config["terrain_field"]),
        args.output_dir / "qualitative_cases", experiment["selection"],
    )
    write_csv(args.output_dir / "failure_case_analysis.csv", failure_rows)
    plot_flow_step_structure(step_rows, figure_dir)
    plot_main_comparison(main_rows, figure_dir)

    effective = copy.deepcopy(experiment)
    effective["runtime"] = {
        "cache_root_from_cli": str(args.cache_root.resolve()),
        "device": str(device),
        "train_sequences": sorted({source.sequence_ids[index] for index in train_indices}),
        "validation_sequences": sorted({source.sequence_ids[index] for index in validation_indices}),
        "validation_scenes": len(validation_dataset),
        "trigger_reference_cost": trigger_reference,
        "checkpoint_resolved": str((TERRAFLOW_ROOT / experiment["checkpoint"]).resolve()),
    }
    (args.output_dir / "config_effective.json").write_text(
        json.dumps(effective, indent=2), encoding="utf-8"
    )
    (args.output_dir / "checkpoint_reference.txt").write_text(
        effective["runtime"]["checkpoint_resolved"] + "\n", encoding="utf-8"
    )
    prediction_payload: dict[str, np.ndarray] = {
        "dataset_indices": baseline["predictions"]["dataset_indices"],
        "ground_truth": baseline["predictions"]["ground_truth"],
    }
    for method, result in main_results.items():
        prediction_payload[f"trajectories_{method}"] = result["predictions"]["trajectories"]
    np.savez_compressed(args.output_dir / "main_predictions.npz", **prediction_payload)
    _write_report(
        TERRAFLOW_ROOT / "docs" / "structure_preserving_guidance.md",
        main_rows, step_rows, gt_reference, case_records,
    )
    summary = {
        "status": "complete",
        "validation_scenes": len(validation_dataset),
        "seed": experiment["seed"],
        "paired_initial_noise": True,
        "training_gt_trigger_reference": trigger_reference,
        "primary_eta": primary_eta,
        "primary_rho": primary_rho,
        "main_methods": main_rows,
        "case_selection": case_records,
        "limitations": [
            "single seed and validation sequence 00004",
            "J_ref is a training-data activation reference, not a safety threshold",
            "cached-BEV clearance is an occupancy-proximity proxy",
        ],
    }
    (args.output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "output": str(args.output_dir.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

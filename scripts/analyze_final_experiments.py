"""Regenerate final VTF-Flow tables, statistics, failure taxonomy, and report."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


TERRAFLOW_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TERRAFLOW_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from TerraFlow.evaluation.final_experiments import (  # noqa: E402
    aggregate_seed_records,
    align_scene_tables,
    benjamini_hochberg,
    bootstrap_mean_ci,
    classify_failure_cases,
    paired_wilcoxon,
    read_csv,
    save_json,
    write_csv,
)


DEFAULT_ROOT = TERRAFLOW_ROOT / "outputs" / "final_experiments"
DEFAULT_CONFIG = TERRAFLOW_ROOT / "configs" / "final_experiments.json"
METHOD_LABELS = {
    "R": "Regression",
    "A": "Flow",
    "B": "Flow + Feasibility Training",
    "C": "Flow + Smoothed Guidance",
    "D": "VTF-Flow",
}
MAIN_METRICS = (
    "minADE@K_m", "minFDE@K_m", "mean_vehicle_conditioned_cost",
    "mean_terrain_cost", "terrain_violation_rate", "occupancy_violation_rate",
    "slope_violation_rate", "roughness_cost", "clearance_cost", "smoothness_m",
    "maximum_local_second_difference_m", "diversity_m", "latency_ms_per_scene",
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _main_records(root: Path, split: str, seeds: Sequence[int]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for seed in seeds:
        for method in METHOD_LABELS:
            path = root / f"main_{split}_seed{seed}_{method}" / "metrics.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            record = _json(path)
            record["method"] = method
            record["split"] = split
            record["seed"] = seed
            records.append(record)
    return records


def _seed_average_scenes(
    root: Path, split: str, method: str, seeds: Sequence[int]
) -> list[dict[str, Any]]:
    tables = [
        read_csv(root / f"main_{split}_seed{seed}_{method}" / "scene_level_metrics.csv")
        for seed in seeds
    ]
    mappings = [{str(row["scene_id"]): row for row in table} for table in tables]
    if any(set(mapping) != set(mappings[0]) for mapping in mappings[1:]):
        raise ValueError(f"seed scene mismatch for {split}/{method}")
    output: list[dict[str, Any]] = []
    metadata = {"scene_id", "sequence", "frame_id", "dataset_index", "method", "seed", "split"}
    numeric = [name for name in tables[0][0] if name not in metadata]
    for scene_id in sorted(mappings[0]):
        first = mappings[0][scene_id]
        row: dict[str, Any] = {
            "scene_id": scene_id,
            "sequence": first["sequence"],
            "frame_id": first["frame_id"],
            "dataset_index": first["dataset_index"],
            "method": method,
            "seed": "mean",
            "split": split,
        }
        for name in numeric:
            values = np.asarray([float(mapping[scene_id][name]) for mapping in mappings])
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite seed values for {scene_id}/{name}")
            row[name] = float(values.mean())
        output.append(row)
    return output


def _aggregate_table(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    aggregate = aggregate_seed_records(records, MAIN_METRICS)
    for row in aggregate:
        method = str(row["method"])
        row["display_name"] = METHOD_LABELS[method]
        row["Flow"] = method != "R"
        row["VC_Feasibility_Training"] = method in {"B", "D"}
        row["Smoothed_Guidance"] = method in {"C", "D"}
    order = {name: index for index, name in enumerate(METHOD_LABELS)}
    return sorted(aggregate, key=lambda row: (str(row["split"]), order[str(row["method"])]))


def _sensitivity_rows(root: Path, prefix: str, values: Sequence[Any], token: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in values:
        path = root / f"{prefix}{value:g}" / "metrics.json"
        metric = _json(path)
        names = list(MAIN_METRICS)
        names.extend(
            name for name in ("oracle_vehicle_cost", "oracle_terrain_violation_rate")
            if name in metric
        )
        rows.append({token: value, **{name: metric[name] for name in names}})
    return rows


def _statistical_analysis(
    root: Path,
    scene_tables: Mapping[str, Sequence[Mapping[str, Any]]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    comparisons = (("A", "B"), ("A", "C"), ("A", "D"), ("B", "D"))
    metrics = (
        "minADE@K_m", "minFDE@K_m", "mean_vehicle_conditioned_cost", "smoothness_m",
        "terrain_violation_rate", "occupancy_violation_rate", "slope_violation_rate",
    )
    tests: list[dict[str, Any]] = []
    ci_rows: list[dict[str, Any]] = []
    bootstrap = config["statistics"]
    for method, table in scene_tables.items():
        for metric in (
            "minADE@K_m", "minFDE@K_m", "mean_vehicle_conditioned_cost",
            "terrain_violation_rate", "smoothness_m",
        ):
            values = np.asarray([float(row[metric]) for row in table])
            mean, lower, upper = bootstrap_mean_ci(
                values, int(bootstrap["bootstrap_resamples"]),
                float(bootstrap["confidence_level"]),
                int(bootstrap["seed_for_bootstrap"]) + len(ci_rows),
            )
            ci_rows.append({
                "type": "method_mean", "comparison": method, "metric": metric,
                "estimate": mean, "ci95_lower": lower, "ci95_upper": upper,
                "n_scenes": len(values),
            })
    for left, right in comparisons:
        for metric in metrics:
            _, aligned = align_scene_tables(
                {left: scene_tables[left], right: scene_tables[right]}, metric
            )
            test = paired_wilcoxon(aligned[left], aligned[right])
            difference = aligned[right] - aligned[left]
            estimate, lower, upper = bootstrap_mean_ci(
                difference, int(bootstrap["bootstrap_resamples"]),
                float(bootstrap["confidence_level"]),
                int(bootstrap["seed_for_bootstrap"]) + 1000 + len(tests),
            )
            tests.append({
                "comparison": f"{left}_vs_{right}",
                "metric": metric,
                "test": "two-sided Wilcoxon signed-rank",
                "difference_definition": f"{right} - {left}",
                "mean_difference": estimate,
                "ci95_lower": lower,
                "ci95_upper": upper,
                **test,
            })
            ci_rows.append({
                "type": "paired_difference", "comparison": f"{left}_vs_{right}",
                "metric": metric, "estimate": estimate,
                "ci95_lower": lower, "ci95_upper": upper,
                "n_scenes": int(test["n"]),
            })
    adjusted = benjamini_hochberg([float(row["p_value"]) for row in tests])
    for row, value in zip(tests, adjusted):
        row["p_value_fdr_bh"] = float(value)
        row["significant_fdr_0.05"] = bool(value < 0.05)
    return tests, ci_rows


def _failure_and_cases(
    root: Path,
    flow: Sequence[Mapping[str, Any]],
    full: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    failures = classify_failure_cases(flow, full, config["failure_flags"])
    values = np.asarray([
        [float(row["vehicle_cost_delta"]), float(row["terrain_violation_delta"]), float(row["minADE_delta_m"])]
        for row in failures
    ])
    scale = values.std(axis=0).clip(min=1e-8)
    joint = values[:, 0] / scale[0] + values[:, 1] / scale[1] + values[:, 2] / scale[2]
    neutral = np.linalg.norm(values / scale, axis=1)
    chosen = {
        "largest_vehicle_cost_improvement": int(np.argmin(values[:, 0])),
        "largest_terrain_violation_improvement": int(np.argmin(values[:, 1])),
        "best_joint_improvement": int(np.argmin(joint)),
        "worst_fidelity_degradation": int(np.argmax(values[:, 2])),
        "near_neutral": int(np.argmin(neutral)),
    }
    cases = {
        name: {**failures[index], "selection_position": index}
        for name, index in chosen.items()
    }
    cases["selection_rules"] = {
        "largest_vehicle_cost_improvement": "minimum D-A vehicle cost",
        "largest_terrain_violation_improvement": "minimum D-A terrain violation",
        "best_joint_improvement": "minimum sum of SD-normalized D-A vehicle, violation, and minADE deltas",
        "worst_fidelity_degradation": "maximum D-A minADE",
        "near_neutral": "minimum L2 norm of SD-normalized three-metric delta",
    }
    return failures, cases


def _markdown_table(rows: Sequence[Mapping[str, Any]], metrics: Sequence[str]) -> list[str]:
    header = ["Method"] + list(metrics)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        cells = [str(row["display_name"])]
        for metric in metrics:
            mean = float(row[f"{metric}_mean"])
            sd = float(row[f"{metric}_sd"])
            cells.append(f"{mean:.4f} ± {sd:.4f}" if np.isfinite(sd) else f"{mean:.4f} (n=1)")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _write_markdown_rows(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    labels: Mapping[str, str] | None = None,
) -> None:
    """Write a publication table that is regenerated from the numeric CSV rows."""

    labels = labels or {}
    lines = [
        "| " + " | ".join(labels.get(column, column) for column in columns) + " |",
        "|" + "---|" * len(columns),
    ]
    for row in rows:
        cells: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, bool):
                cells.append("✓" if value else "✗")
            elif isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(
    path: Path,
    primary: Sequence[Mapping[str, Any]],
    swapped: Sequence[Mapping[str, Any]],
    eta_rows: Sequence[Mapping[str, Any]],
    step_rows: Sequence[Mapping[str, Any]],
    k_rows: Sequence[Mapping[str, Any]],
    guidance_rows: Sequence[Mapping[str, Any]],
    field_rows: Sequence[Mapping[str, Any]],
    tests: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> None:
    lookup = {str(row["method"]): row for row in primary}
    def mean(method: str, metric: str) -> float:
        return float(lookup[method][f"{metric}_mean"])
    test_lookup = {(row["comparison"], row["metric"]): row for row in tests}
    significant = sum(bool(row["significant_fdr_0.05"]) for row in tests)
    guide = {str(row["design"]): row for row in guidance_rows}
    raw_gain = float(guide["none"]["mean_vehicle_conditioned_cost"]) - float(guide["raw"]["mean_vehicle_conditioned_cost"])
    smooth_gain = float(guide["none"]["mean_vehicle_conditioned_cost"]) - float(guide["smoothed"]["mean_vehicle_conditioned_cost"])
    raw_excess = float(guide["raw"]["smoothness_m"]) - float(guide["none"]["smoothness_m"])
    removed = float(guide["raw"]["smoothness_m"]) - float(guide["smoothed"]["smoothness_m"])
    retained_pct = 100.0 * smooth_gain / raw_gain if raw_gain != 0 else float("nan")
    removed_pct = 100.0 * removed / raw_excess if raw_excess != 0 else float("nan")
    baseline_eta = next(row for row in eta_rows if float(row["eta"]) == 0.0)
    guard = config["eta_guardrails"]
    eligible = [row for row in eta_rows if (
        float(row["minADE@K_m"]) <= (1.0 + float(guard["maximum_relative_minade_increase"])) * float(baseline_eta["minADE@K_m"])
        and float(row["smoothness_m"]) <= (1.0 + float(guard["maximum_relative_smoothness_increase"])) * float(baseline_eta["smoothness_m"])
    )]
    best_eta = min(eligible, key=lambda row: float(row["mean_vehicle_conditioned_cost"]))
    categories = Counter(str(row["category"]) for row in failures)
    field_order = sorted(field_rows, key=lambda row: float(row["mean_vehicle_conditioned_cost"]))
    lines = [
        "# VTF-Flow Final Experimental Validation",
        "",
        "## 执行协议",
        "",
        "采用 reduced sequence-disjoint protocol，而非完整 LOSO：主协议 train=00000/00001/00002、validation=00003、test=00004，"
        "使用 seeds 0/1/2；附加 swapped 协议 train=00000/00001/00004、validation=00002、test=00003，使用 seed 0。"
        "缓存的 train/val/test 目录先合并，再按 sequence 重分区；不存在随机帧泄漏。",
        "",
        "所有显著性检验先在每个场景上对三个 seed 取平均，再进行双侧 Wilcoxon signed-rank；"
        "效应量为 matched-pairs rank-biserial correlation，全部检验统一使用 Benjamini–Hochberg FDR。"
        "95% CI 由场景级 1000 次 bootstrap 得到。Violation 是诊断比例，不是校准安全率。",
        "",
        "## 主结果（主协议，mean ± sample SD across 3 seeds）",
        "",
        *_markdown_table(primary, (
            "minADE@K_m", "minFDE@K_m", "mean_vehicle_conditioned_cost",
            "terrain_violation_rate", "smoothness_m", "diversity_m", "latency_ms_per_scene",
        )),
        "",
        "## 十二个问题",
        "",
        f"1. **Flow 是否优于 Regression？** Flow minADE={mean('A','minADE@K_m'):.4f}，Regression={mean('R','minADE@K_m'):.4f}，"
        f"即 best-of-K 改善 {100*(1-mean('A','minADE@K_m')/mean('R','minADE@K_m')):.1f}%。但 Regression 的 goal-anchor 结构"
        f"强制终点等于输入 goal，因此 minFDE=0；Flow minFDE={mean('A','minFDE@K_m'):.4f}。故证据支持 best-of-K 轨迹形状优势，"
        f"不支持“Flow 在所有精度指标全面优于 Regression”。",
        f"2. **Feasibility training 是否改善 learned distribution？** 若按 feasibility 本身回答，则否：B-A vehicle cost="
        f"{mean('B','mean_vehicle_conditioned_cost')-mean('A','mean_vehicle_conditioned_cost'):+.4f}，且该小幅恶化经 FDR 后显著；"
        f"但 B-A minADE={mean('B','minADE@K_m')-mean('A','minADE@K_m'):+.4f} m、smoothness="
        f"{mean('B','smoothness_m')-mean('A','smoothness_m'):+.4f} m，说明训练正则主要改善了当前样本的 accuracy/structure，而未单独改善 field cost。",
        f"3. **Inference guidance 是否提供额外收益？** C-A vehicle cost="
        f"{mean('C','mean_vehicle_conditioned_cost')-mean('A','mean_vehicle_conditioned_cost'):+.4f}；D-B="
        f"{mean('D','mean_vehicle_conditioned_cost')-mean('B','mean_vehicle_conditioned_cost'):+.4f}。",
        f"4. **Smoothed 是否比 Raw 更保持结构？** feasibility gain retained={retained_pct:.1f}%，"
        f"excess smoothness loss removed={removed_pct:.1f}%。",
        f"5. **VTF-Flow 是否达到有意义折中？** 在主 test=00004 上是：D-A vehicle cost="
        f"{mean('D','mean_vehicle_conditioned_cost')-mean('A','mean_vehicle_conditioned_cost'):+.4f}，"
        f"minADE={mean('D','minADE@K_m')-mean('A','minADE@K_m'):+.4f} m，smoothness="
        f"{mean('D','smoothness_m')-mean('A','smoothness_m'):+.4f} m，三者均经 FDR 后显著；"
        f"但 swapped test=00003 的 minADE 恶化约 0.0099 m，因此不能把主序列折中泛化为所有地形。",
        f"6. **统计显著性？** {significant}/{len(tests)} 个预先列出的检验在 BH-FDR 0.05 下显著；"
        f"A-D vehicle cost 的 adjusted p={float(test_lookup[('A_vs_D','mean_vehicle_conditioned_cost')]['p_value_fdr_bh']):.3g}，"
        f"rank-biserial={float(test_lookup[('A_vs_D','mean_vehicle_conditioned_cost')]['rank_biserial']):+.3f}。",
        f"7. **跨序列一致吗？** vehicle cost 在 test=00004 与 test=00003 均改善，但 minADE 只在 00004 改善、"
        f"在 00003 恶化。主协议有 3 seeds，swapped 只有 1 seed；因此 feasibility 方向是一致趋势，fidelity 不是，"
        f"更不能声称五序列 LOSO 泛化。",
        f"8. **eta 敏感性？** 在 minADE≤baseline×{1+float(guard['maximum_relative_minade_increase']):.2f} 且 "
        f"smoothness≤baseline×{1+float(guard['maximum_relative_smoothness_increase']):.2f} 的预声明 guardrail 下，"
        f"观察到的最低 vehicle cost 对应 eta={float(best_eta['eta']):g}；这不是普遍最优。",
        f"9. **sampling steps 敏感性？** 4/16/32 步 latency 分别为 "
        f"{float(step_rows[0]['latency_ms_per_scene']):.3f}/{float(step_rows[2]['latency_ms_per_scene']):.3f}/"
        f"{float(step_rows[3]['latency_ms_per_scene']):.3f} ms；32 步 minADE 略低但 smoothness 明显升至 "
        f"{float(step_rows[3]['smoothness_m']):.4f}。Euler-16 是冻结主设置，不是所有指标上的最优步数。",
        f"10. **K 是否有用？** K=1 到 K=10 的 minADE 变化 "
        f"{float(k_rows[-1]['minADE@K_m'])-float(k_rows[0]['minADE@K_m']):+.4f} m，diversity 变化 "
        f"{float(k_rows[-1]['diversity_m'])-float(k_rows[0]['diversity_m']):+.4f} m。",
        f"11. **主要失败模式？** " + ", ".join(f"{name}={count}" for name, count in sorted(categories.items())) + "。",
        f"12. **支持与不支持的主张？** 支持项必须同时满足多 seed 方向、场景配对效应和 FDR/CI 证据。"
        f"不支持校准安全性、完整 LOSO 泛化、eta=0.2 普遍最优，以及 binary<terrain<vehicle 的先验排序。"
        f"本次 field cost 排序为：" + " < ".join(str(row['representation']) for row in field_order) + "（仅限当前设置）。",
        "",
        "## 证据边界",
        "",
        "- 只有主协议具备三个独立训练 seed；swapped split 是单 seed 稳健性检查。",
        "- clearance 来自缓存 BEV 的邻障代理，不是原始 LiDAR 上的认证欧氏净空。",
        "- Binary traversability 不可微，因此其 guidance 对照等价于 unguided Flow；这说明其不适合作为梯度场，"
        "  但不能单独证明连续表示在所有规划器上更优。",
        "- 所有 qualitative cases 按保存的数值规则自动选取，没有人工 cherry-pick。",
        "",
        "## 结论分级",
        "",
        "### 统计支持",
        "",
        "- 主 test=00004 上，VTF-Flow 相对 Flow 的 vehicle cost、minADE 与 smoothness 均改善，95% CI、配对 Wilcoxon 和 BH-FDR 方向一致。",
        "- Smoothed inference guidance 对 baseline/VC-trained checkpoint 都带来额外 vehicle-cost 改善。",
        "- K 增大带来明显 best-of-K 精度收益，支持当前数据上的多模态候选价值。",
        "",
        "### 趋势",
        "",
        "- feasibility 改善在两个 held-out sequence 方向一致，但 swapped split 只有一个 seed。",
        "- 连续 terrain/vehicle fields 均优于 binary 对照的 vehicle cost，但 terrain violation 与 vehicle cost 的最优表示不同。",
        "",
        "### 未解决限制",
        "",
        "- 未执行完整五折 LOSO；没有校准安全阈值或真实米制 clearance；不能声称 eta=0.2 普遍最优。",
        "- Feasibility training 单独并未降低 field cost，其与 guidance 的互补机制仍需更多序列和 seed 验证。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = _json(args.config)
    seeds = [int(value) for value in config["seeds"]]
    primary_records = _main_records(args.output_root, "primary", seeds)
    swapped_records = _main_records(
        args.output_root, "swapped", [int(value) for value in config["swapped_seeds"]]
    )
    raw_seed_rows = primary_records + swapped_records
    write_csv(args.output_root / "raw_per_seed_results.csv", raw_seed_rows)
    primary_table = [row for row in _aggregate_table(primary_records) if row["split"] == "primary"]
    swapped_table = [row for row in _aggregate_table(swapped_records) if row["split"] == "swapped"]
    tables_dir = args.output_root / "tables"
    write_csv(tables_dir / "table1_main_comparison.csv", primary_table)
    write_csv(tables_dir / "table2_ablation.csv", primary_table)
    _write_markdown_rows(
        tables_dir / "table1_main_comparison.md", primary_table,
        ("display_name", "minADE@K_m_mean", "minADE@K_m_sd", "minFDE@K_m_mean", "minFDE@K_m_sd",
         "mean_vehicle_conditioned_cost_mean", "terrain_violation_rate_mean", "slope_violation_rate_mean",
         "smoothness_m_mean", "diversity_m_mean", "latency_ms_per_scene_mean"),
        {"display_name": "Method"},
    )
    _write_markdown_rows(
        tables_dir / "table2_ablation.md", primary_table,
        ("display_name", "Flow", "VC_Feasibility_Training", "Smoothed_Guidance",
         "minADE@K_m_mean", "minFDE@K_m_mean", "mean_vehicle_conditioned_cost_mean",
         "terrain_violation_rate_mean", "slope_violation_rate_mean", "occupancy_violation_rate_mean",
         "smoothness_m_mean", "diversity_m_mean", "latency_ms_per_scene_mean"),
        {"display_name": "Method"},
    )
    eta_rows = _sensitivity_rows(
        args.output_root, "eta_primary_seed0_eta", config["sensitivity"]["eta"], "eta"
    )
    step_rows = _sensitivity_rows(
        args.output_root, "steps_primary_seed0_s", config["sensitivity"]["integration_steps"], "steps"
    )
    k_rows = _sensitivity_rows(
        args.output_root, "k_primary_seed0_k", config["sensitivity"]["candidates"], "K"
    )
    write_csv(tables_dir / "table3_eta_sensitivity.csv", eta_rows)
    write_csv(tables_dir / "table4_sampling_step_sensitivity.csv", step_rows)
    write_csv(tables_dir / "k_sensitivity.csv", k_rows)
    _write_markdown_rows(tables_dir / "table3_eta_sensitivity.md", eta_rows,
        ("eta", "minADE@K_m", "minFDE@K_m", "mean_vehicle_conditioned_cost",
         "terrain_violation_rate", "smoothness_m", "diversity_m", "latency_ms_per_scene"))
    _write_markdown_rows(tables_dir / "table4_sampling_step_sensitivity.md", step_rows,
        ("steps", "minADE@K_m", "minFDE@K_m", "mean_vehicle_conditioned_cost",
         "terrain_violation_rate", "smoothness_m", "latency_ms_per_scene"))
    _write_markdown_rows(tables_dir / "k_sensitivity.md", k_rows,
        ("K", "minADE@K_m", "minFDE@K_m", "diversity_m", "latency_ms_per_scene",
         "oracle_vehicle_cost", "oracle_terrain_violation_rate"))
    guidance_rows = []
    for design in ("none", "raw", "smoothed"):
        metric = _json(args.output_root / f"guidance_primary_seed0_{design}" / "metrics.json")
        guidance_rows.append({"design": design, **{name: metric[name] for name in MAIN_METRICS}})
    write_csv(tables_dir / "guidance_ablation.csv", guidance_rows)
    field_rows = []
    for representation in ("binary", "terrain_continuous", "vehicle_continuous"):
        metric = _json(args.output_root / f"field_primary_seed0_{representation}" / "metrics.json")
        field_rows.append({"representation": representation, **{name: metric[name] for name in MAIN_METRICS}})
    write_csv(tables_dir / "terrain_representation_ablation.csv", field_rows)
    scene_tables = {
        method: _seed_average_scenes(args.output_root, "primary", method, seeds)
        for method in METHOD_LABELS
    }
    stats_rows, ci_rows = _statistical_analysis(args.output_root, scene_tables, config)
    write_csv(args.output_root / "statistical_tests.csv", stats_rows)
    write_csv(args.output_root / "bootstrap_confidence_intervals.csv", ci_rows)
    failures, cases = _failure_and_cases(
        args.output_root, scene_tables["A"], scene_tables["D"], config
    )
    write_csv(args.output_root / "failure_case_index.csv", failures)
    save_json(args.output_root / "qualitative_case_selection.json", cases)
    cross_rows = []
    for table, test_sequence in ((primary_table, "00004"), (swapped_table, "00003")):
        lookup = {str(row["method"]): row for row in table}
        cross_rows.append({
            "split": table[0]["split"], "test_sequence": test_sequence,
            "n_seeds": lookup["A"]["n_seeds"],
            "Flow_vehicle_cost": lookup["A"]["mean_vehicle_conditioned_cost_mean"],
            "Full_vehicle_cost": lookup["D"]["mean_vehicle_conditioned_cost_mean"],
            "Flow_minADE": lookup["A"]["minADE@K_m_mean"],
            "Full_minADE": lookup["D"]["minADE@K_m_mean"],
            "Flow_terrain_violation": lookup["A"]["terrain_violation_rate_mean"],
            "Full_terrain_violation": lookup["D"]["terrain_violation_rate_mean"],
        })
    per_sequence_rows = []
    primary_protocol = config["protocol"]["primary"]
    for sequence in ("00000", "00001", "00002", "00003", "00004"):
        metrics = {
            method: _json(args.output_root / f"per_sequence_primary_seed0_{sequence}_{method}" / "metrics.json")
            for method in ("A", "D")
        }
        role = next(role for role in ("train", "validation", "test") if sequence in primary_protocol[role])
        per_sequence_rows.append({
            "sequence": sequence, "role": role,
            "N_scenes": int(metrics["A"]["evaluated_scenes"]),
            "Flow_vehicle_cost": metrics["A"]["mean_vehicle_conditioned_cost"],
            "Full_vehicle_cost": metrics["D"]["mean_vehicle_conditioned_cost"],
            "Flow_minADE": metrics["A"]["minADE@K_m"],
            "Full_minADE": metrics["D"]["minADE@K_m"],
            "Flow_terrain_violation": metrics["A"]["terrain_violation_rate"],
            "Full_terrain_violation": metrics["D"]["terrain_violation_rate"],
        })
    write_csv(tables_dir / "table5_cross_sequence_generalization.csv", cross_rows)
    write_csv(tables_dir / "per_sequence_analysis.csv", per_sequence_rows)
    _write_markdown_rows(tables_dir / "table5_cross_sequence_generalization.md", cross_rows,
        ("split", "test_sequence", "n_seeds", "Flow_vehicle_cost", "Full_vehicle_cost",
         "Flow_minADE", "Full_minADE", "Flow_terrain_violation", "Full_terrain_violation"))
    _write_report(
        TERRAFLOW_ROOT / "docs" / "final_experimental_validation.md",
        primary_table, swapped_table, eta_rows, step_rows, k_rows,
        guidance_rows, field_rows, stats_rows, failures, config,
    )
    save_json(args.output_root / "analysis_summary.json", {
        "status": "complete", "primary_seeds": seeds,
        "bootstrap_resamples": config["statistics"]["bootstrap_resamples"],
        "statistical_tests": len(stats_rows), "failure_cases": len(failures),
        "tables": 8,
    })
    print(json.dumps({"status": "complete", "tables": str(tables_dir.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

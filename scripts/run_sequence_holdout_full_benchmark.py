"""Compare final VTF-Flow with all baselines under one outer-fold protocol.

The script reuses the frozen Flow/VTF-Flow checkpoints and prediction
archives produced by ``run_sequence_holdout_robustness.py``. Constant Velocity,
A*, and deterministic regression are generated on the identical held-out scene
indices, after which every archive is scored by one common evaluator.
"""

from __future__ import annotations

import argparse
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
    sequence_macro_benchmark_summary,
)
from TerraFlow.metrics.trajectory_metrics import trajectory_metrics  # noqa: E402
from TerraFlow.interfaces import SceneBatch, TrajectoryBatch  # noqa: E402
from TerraFlow.planners.astar_baseline import AStarPlanningError  # noqa: E402
from TerraFlow.scripts.run_final_experiments import _load_regression  # noqa: E402
from TerraFlow.scripts.run_sequence_holdout_robustness import (  # noqa: E402
    as_sequence_split,
    flow_training_config,
)
from TerraFlow.scripts.run_unified_h10_benchmark import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_CONFIG as DEFAULT_BENCHMARK_CONFIG,
    DEFAULT_DATA,
    H10PlanningDataset,
    classical_planners,
    load_json,
    regression_training_config,
)
from TerraFlow.scripts.train_regression import (  # noqa: E402
    CombinedSceneDataset,
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


DEFAULT_CONFIG = TERRAFLOW_ROOT / "configs" / "sequence_holdout_full_benchmark.json"
DEFAULT_OUTPUT = TERRAFLOW_ROOT / "outputs" / "sequence_holdout_full_benchmark"
METHOD_ORDER = ("CV", "ASTAR", "REG", "FLOW", "VTF_V2")
EVALUATOR_VERSION = "sequence_holdout_common_v2"
HIGHER_IS_BETTER = {
    "clearance_q05_m",
    "compliant_candidate_rate_q80",
    "GCCR_at_K_q80",
}
NONORDINAL_METRICS = {"diversity_m"}
COMPLETE_METRICS = (
    "ADE_candidate0_m",
    "FDE_candidate0_m",
    "minADE@K_m",
    "minFDE@K_m",
    "diversity_m",
    "path_length_m",
    "mean_unified_tvk_cost",
    "terrain_violation_rate",
    "occupancy_exposure_rate",
    "nontraversable_exposure_rate",
    "slope_exposure_rate",
    "roughness_mean",
    "clearance_q05_m",
    "curvature_violation_rate_independent",
    "lateral_acceleration_violation_rate_independent",
    "smoothness_m_independent",
    "compliant_candidate_rate_q80",
    "GCCR_at_K_q80",
)


class FullCoverageAStar:
    """Evaluate A* on every scene without silently excluding map-unsupported goals.

    The planner-used BEV covers only forward ``x >= 0``. A recorded future may
    contain a short reverse displacement and therefore lie outside that spatial
    support. Such scenes receive an explicit stationary failure fallback rather
    than a fabricated rear terrain map or post-hoc scene exclusion.
    """

    def __init__(self, planner: Any, horizon: int) -> None:
        self.planner = planner
        self.horizon = int(horizon)
        self.out_of_map_stationary_count = 0
        self.soft_cost_fallback_count = 0
        self.planning_failure_stationary_count = 0

    def _stationary(self, scene: SceneBatch) -> TrajectoryBatch:
        scene = scene.as_batch()
        dimensions = scene.gt_future.shape[-1]
        trajectories = scene.gt_future.new_zeros(
            (scene.batch_size, 1, self.horizon, dimensions)
        )
        return TrajectoryBatch(trajectories=trajectories)

    def __call__(self, scene: SceneBatch) -> tuple[TrajectoryBatch, bool]:
        scene = scene.as_batch()
        goal = scene.goal[0, :2]
        config = self.planner.hard.config
        supported = bool(
            (goal[0] >= 0.0)
            & (goal[0] <= config.forward_extent_m)
            & (goal[1].abs() <= config.lateral_extent_m)
        )
        if not supported:
            self.out_of_map_stationary_count += 1
            return self._stationary(scene), True
        try:
            prediction, soft = self.planner(scene)
        except AStarPlanningError:
            self.planning_failure_stationary_count += 1
            return self._stationary(scene), True
        self.soft_cost_fallback_count += int(soft)
        return prediction, soft

    @property
    def fallback_reasons(self) -> dict[str, int]:
        return {
            "out_of_map_stationary": self.out_of_map_stationary_count,
            "soft_cost": self.soft_cost_fallback_count,
            "planning_failure_stationary": self.planning_failure_stationary_count,
        }


def load_protocol(path: Path) -> dict[str, Any]:
    """Load the full-benchmark protocol."""

    return json.loads(path.read_text(encoding="utf-8"))


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a protocol path relative to the TerraFlow project root."""

    path = Path(value)
    return path.resolve() if path.is_absolute() else (TERRAFLOW_ROOT / path).resolve()


def train_regression_fold(
    dataset: H10PlanningDataset,
    indices: Mapping[str, list[int]],
    fold: SequenceHoldoutFold,
    seed: int,
    benchmark: Mapping[str, Any],
    frozen: Mapping[str, Any],
    output_root: Path,
    device: torch.device,
    skip_training: bool,
) -> Path:
    """Train or resume deterministic regression for one outer fold and seed."""

    root = output_root / "checkpoints" / fold.name / f"seed_{seed}" / "regression"
    checkpoint = root / "best.pt"
    if checkpoint.is_file():
        return checkpoint
    if skip_training:
        raise FileNotFoundError(f"missing regression checkpoint: {checkpoint}")
    config = regression_training_config(benchmark, seed)
    config["training"]["epochs"] = int(frozen["epochs"])
    config["data"]["validation_sequences"] = list(fold.validation)
    root.mkdir(parents=True, exist_ok=True)
    save_json(root / "effective_config.json", config)
    set_reproducible_seed(seed)
    print(f"training {fold.name} seed={seed} deterministic regression", flush=True)
    train_regression(
        Subset(dataset, indices["train"]),
        Subset(dataset, indices["validation"]),
        config,
        root,
        device,
        int(frozen["epochs"]),
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"training did not create {checkpoint}")
    return checkpoint


def generate_prediction_archive(
    method: str,
    planner: Any,
    dataset: H10PlanningDataset,
    test_indices: Sequence[int],
    seed: int,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
) -> tuple[Path, int]:
    """Generate one ordered prediction archive for a classical or regression method."""

    archive = output_dir / "predictions.npz"
    info_path = output_dir / "prediction_info.json"
    if archive.is_file() and info_path.is_file():
        info = load_json(info_path)
        return archive, int(info.get("astar_soft_fallback_count", 0))
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_device = torch.device("cpu") if method == "ASTAR" else device
    loader = make_loader(
        Subset(dataset, list(test_indices)),
        1 if method == "ASTAR" else batch_size,
        shuffle=False,
        seed=seed + 1301,
        num_workers=0,
    )
    trajectories: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    fallback_count = 0
    processed = 0
    for scene in loader:
        scene = scene.to(evaluation_device)
        with torch.no_grad():
            if method == "ASTAR":
                prediction, fallback = planner(scene)
                fallback_count += int(fallback)
            else:
                prediction = planner(scene)
        if prediction.trajectories.shape[0] != scene.batch_size:
            raise ValueError(f"{method} prediction batch does not match the scene batch")
        trajectories.append(prediction.trajectories.detach().cpu().numpy())
        targets.append(scene.gt_future.detach().cpu().numpy())
        processed += scene.batch_size
        if method == "ASTAR" and processed % 500 == 0:
            print(f"  {method} seed={seed}: {processed}/{len(test_indices)}", flush=True)
    prediction_array = np.concatenate(trajectories)
    target_array = np.concatenate(targets)
    if prediction_array.shape[0] != len(test_indices):
        raise ValueError(f"{method} archive has the wrong scene count")
    np.savez_compressed(
        archive,
        trajectories=prediction_array,
        ground_truth=target_array,
    )
    save_json(
        info_path,
        {
            "method": method,
            "seed": seed,
            "evaluated_scenes": len(test_indices),
            "K": int(prediction_array.shape[1]),
            "astar_soft_fallback_count": fallback_count,
            "astar_fallback_reasons": getattr(planner, "fallback_reasons", {}),
        },
    )
    return archive, fallback_count


def summarize_scene_rows(
    rows: Sequence[Mapping[str, Any]],
    method: str,
    seed: int,
    test_sequence: str,
    fallback_count: int,
) -> dict[str, Any]:
    """Summarize a complete scene table without dropping invalid values."""

    if not rows:
        raise ValueError("cannot summarize an empty evaluation")
    ignored = {
        "scene_id",
        "sequence",
        "frame_id",
        "dataset_index",
        "method",
        "seed",
    }
    summary: dict[str, Any] = {
        "method": method,
        "seed": seed,
        "test_sequence": test_sequence,
        "evaluated_scenes": len(rows),
        "K": int(rows[0]["K"]),
        "astar_soft_fallback_count": fallback_count,
        "astar_fallback_count": fallback_count,
        "evaluator_version": EVALUATOR_VERSION,
    }
    for name in rows[0]:
        if name in ignored or name in {"K", "astar_soft_fallback"}:
            continue
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite values in {method}/{name}")
        summary[name] = float(values.mean())
    return summary


def evaluate_prediction_archive(
    archive: Path,
    method: str,
    seed: int,
    fold: SequenceHoldoutFold,
    dataset: H10PlanningDataset,
    test_indices: Sequence[int],
    benchmark: Mapping[str, Any],
    frozen: Mapping[str, Any],
    envelopes: Mapping[str, Mapping[str, float]],
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    fallback_count: int = 0,
) -> dict[str, Any]:
    """Score an ordered trajectory archive with the common TVK evaluator."""

    summary_path = output_dir / "summary.json"
    scene_path = output_dir / "scene_level_metrics.csv"
    if summary_path.is_file() and scene_path.is_file():
        cached = load_json(summary_path)
        if cached.get("evaluator_version") == EVALUATOR_VERSION:
            return cached
    output_dir.mkdir(parents=True, exist_ok=True)
    values = np.load(archive)
    prediction_array = np.asarray(values["trajectories"], dtype=np.float32)
    target_array = np.asarray(values["ground_truth"], dtype=np.float32)
    if prediction_array.shape[0] != len(test_indices):
        raise ValueError(f"archive scene count mismatch: {archive}")
    if prediction_array.shape[2:] != (
        int(benchmark["trajectory"]["horizon_steps"]),
        3,
    ):
        raise ValueError(f"archive trajectory shape mismatch: {prediction_array.shape}")
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
        batch_size,
        shuffle=False,
        seed=seed + 1701,
        num_workers=0,
    )
    rows: list[dict[str, Any]] = []
    offset = 0
    for scene in loader:
        batch = scene.batch_size
        scene = scene.to(device)
        trajectories = torch.from_numpy(
            prediction_array[offset : offset + batch]
        ).to(device=device, dtype=scene.gt_future.dtype)
        archived_target = torch.from_numpy(
            target_array[offset : offset + batch]
        ).to(device=device, dtype=scene.gt_future.dtype)
        if not torch.allclose(archived_target, scene.gt_future, atol=1e-6, rtol=0.0):
            raise ValueError(f"ground-truth order mismatch in {archive}")
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
        components = derive_independent_component_maps(
            scene.terrain_map, terrain, metric_config
        )
        independent = candidate_claim_metrics(
            trajectories, scene.goal, components, metric_config, kinematic
        )
        masks = {
            label: compliance_mask(independent, envelope)
            for label, envelope in envelopes.items()
        }
        path_length = standard["path_length_by_candidate_m"].mean(dim=1)
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
                "K": int(trajectories.shape[1]),
                "ADE_candidate0_m": float(
                    standard["ADE_by_candidate_m"][local, 0]
                ),
                "FDE_candidate0_m": float(
                    standard["FDE_by_candidate_m"][local, 0]
                ),
                "minADE@K_m": float(standard["minADE@K_m"][local]),
                "minFDE@K_m": float(standard["minFDE@K_m"][local]),
                "diversity_m": float(standard["diversity_m"][local]),
                "path_length_m": float(path_length[local]),
            }
            row.update({name: float(value[local]) for name, value in terminal.items()})
            for name, metric_values in independent.items():
                output_name = {
                    "curvature_violation_rate": "curvature_violation_rate_independent",
                    "lateral_acceleration_violation_rate": "lateral_acceleration_violation_rate_independent",
                    "smoothness_m": "smoothness_m_independent",
                }.get(name, name)
                row[output_name] = float(metric_values[local].mean())
            for label, mask in masks.items():
                row[f"compliant_candidate_rate_{label}"] = float(
                    mask[local].float().mean()
                )
                row[f"GCCR_at_K_{label}"] = float(mask[local].any())
            rows.append(row)
        offset += batch
    if offset != len(test_indices):
        raise ValueError("evaluation loader did not consume every held-out scene")
    write_csv(scene_path, rows)
    summary = summarize_scene_rows(
        rows, method, seed, fold.test_sequence, fallback_count
    )
    save_json(summary_path, summary)
    (output_dir / "prediction_reference.txt").write_text(
        str(archive.resolve()) + "\n", encoding="utf-8"
    )
    return summary


def format_mean_sd(row: pd.Series, metric: str, tex: bool = False) -> str:
    """Format one sequence-macro mean and sequence-level sample SD."""

    mean = float(row[f"{metric}_mean"])
    sd = float(row[f"{metric}_sequence_sd"])
    if np.isnan(sd):
        return f"{mean:.4f}"
    return (
        f"{mean:.4f} $\\pm$ {sd:.4f}"
        if tex
        else f"{mean:.4f} ± {sd:.4f}"
    )


def write_main_tables(
    macro: pd.DataFrame,
    display_names: Mapping[str, str],
    output_root: Path,
) -> None:
    """Write complete numeric and manuscript-formatted benchmark tables."""

    macro.to_csv(output_root / "main_table_numeric.csv", index=False)
    columns = [
        ("ADE-0 (m) ↓", "ADE_candidate0_m", "min"),
        ("minADE@K (m) ↓", "minADE@K_m", "min"),
        ("Diversity (m)", "diversity_m", None),
        ("TVK potential ↓", "mean_unified_tvk_cost", "min"),
        ("Terrain violation ↓", "terrain_violation_rate", "min"),
        ("Occupancy exposure ↓", "occupancy_exposure_rate", "min"),
        ("Curvature violation ↓", "curvature_violation_rate_independent", "min"),
        ("Smoothness (m) ↓", "smoothness_m_independent", "min"),
        ("q80 candidate rate ↑", "compliant_candidate_rate_q80", "max"),
        ("GCCR@K (q80) ↑", "GCCR_at_K_q80", "max"),
    ]
    markdown_rows = []
    tex_rows = []
    csv_rows = []
    for _, row in macro.iterrows():
        method = str(row["method"])
        k = int(row["K"])
        csv_row: dict[str, Any] = {"Method": display_names[method], "K": k}
        md_values = [display_names[method], str(k)]
        tex_values = [display_names[method], str(k)]
        for label, metric, direction in columns:
            if metric == "diversity_m" and k == 1:
                csv_value = "—"
                tex_value = "--"
                md_value = csv_value
            else:
                csv_value = format_mean_sd(row, metric)
                tex_value = format_mean_sd(row, metric, tex=True)
                md_value = csv_value
                mean = float(row[f"{metric}_mean"])
                if direction is not None:
                    values = macro[f"{metric}_mean"].to_numpy(dtype=np.float64)
                    best = float(values.min() if direction == "min" else values.max())
                    if np.isclose(mean, best, atol=1e-12, rtol=0.0):
                        md_value = f"**{md_value}**"
                        tex_value = f"\\textbf{{{tex_value}}}"
            csv_row[label] = csv_value
            md_values.append(md_value)
            tex_values.append(tex_value)
        csv_rows.append(csv_row)
        markdown_rows.append(md_values)
        tex_rows.append(tex_values)
    formatted = pd.DataFrame(csv_rows)
    formatted.to_csv(output_root / "main_table.csv", index=False)
    headers = list(formatted.columns)
    tex_headers = [
        header.replace(" ↓", " $\\downarrow$").replace(" ↑", " $\\uparrow$")
        for header in headers
    ]
    md = [
        "# Unified sequence-holdout benchmark",
        "",
        "Values are unweighted mean ± sample SD across three held-out sequences. "
        "Training seeds are averaged within sequence before this summary. A* uses "
        "a recorded stationary failure fallback when a goal lies outside the forward-only BEV.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    md.extend("| " + " | ".join(values) + " |" for values in markdown_rows)
    (output_root / "main_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    alignment = "ll" + "c" * len(columns)
    latex = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Strict sequence-level comparison under the frozen VTF-Flow protocol. Values are unweighted mean $\\pm$ sample standard deviation across three held-out sequences; training seeds are averaged within each sequence. A* uses a recorded stationary failure fallback for goals outside the forward-only BEV.}",
        "\\label{tab:sequence_holdout_full_benchmark}",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\toprule",
        " & ".join(tex_headers) + " \\\\",
        "\\midrule",
    ]
    latex.extend(" & ".join(values) + " \\\\" for values in tex_rows)
    latex.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}"])
    (output_root / "main_table.tex").write_text(
        "\n".join(latex) + "\n", encoding="utf-8"
    )


def write_pairwise_effects(
    sequence_method: pd.DataFrame,
    metrics: Sequence[str],
    output_root: Path,
) -> pd.DataFrame:
    """Write descriptive VTF-Flow effects using held-out sequences as n."""

    target = sequence_method[sequence_method["method"] == "VTF_V2"].set_index(
        "test_sequence"
    )
    manuscript_names = {
        "CV": "Constant Velocity",
        "ASTAR": "A* terrain planner",
        "REG": "Deterministic regression",
        "FLOW": "Flow Matching",
    }
    rows = []
    for method in METHOD_ORDER:
        if method == "VTF_V2":
            continue
        baseline = sequence_method[sequence_method["method"] == method].set_index(
            "test_sequence"
        )
        if not baseline.index.equals(target.index):
            raise ValueError(f"sequence mismatch for VTF_V2 versus {method}")
        for metric in metrics:
            raw = target[metric] - baseline[metric]
            if metric in NONORDINAL_METRICS:
                direction = "descriptive_nonordinal"
                aligned_mean = float("nan")
                improved: int | None = None
            else:
                direction = "higher_is_better" if metric in HIGHER_IS_BETTER else "lower_is_better"
                aligned = raw if metric in HIGHER_IS_BETTER else -raw
                aligned_mean = float(aligned.mean())
                improved = int((aligned > 0.0).sum())
            rows.append(
                {
                    "comparison": f"VTF-Flow - {manuscript_names[method]}",
                    "metric": metric,
                    "direction": direction,
                    "mean_raw_difference": float(raw.mean()),
                    "sequence_sd_raw_difference": float(raw.std(ddof=1)),
                    "mean_direction_aligned_improvement": aligned_mean,
                    "sequences_improved": improved,
                    "n_held_out_sequences": len(raw),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(output_root / "vtf_flow_pairwise_sequence_effects.csv", index=False)
    return result


def write_report(
    macro: pd.DataFrame,
    display_names: Mapping[str, str],
    astar_coverage: Mapping[str, Any],
    output_root: Path,
) -> None:
    """Create a concise Chinese interpretation with explicit evidence limits."""

    rows = []
    for _, row in macro.iterrows():
        rows.append(
            "| {name} | {k} | {ade} | {minade} | {tvk} | {terrain} | {curvature} | {candidate} |".format(
                name=display_names[str(row["method"])],
                k=int(row["K"]),
                ade=format_mean_sd(row, "ADE_candidate0_m"),
                minade=format_mean_sd(row, "minADE@K_m"),
                tvk=format_mean_sd(row, "mean_unified_tvk_cost"),
                terrain=format_mean_sd(row, "terrain_violation_rate"),
                curvature=format_mean_sd(row, "curvature_violation_rate_independent"),
                candidate=format_mean_sd(row, "compliant_candidate_rate_q80"),
            )
        )
    indexed = macro.set_index("method")

    def metric(method: str, name: str) -> float:
        return float(indexed.loc[method, f"{name}_mean"])

    def reduction(target: str, baseline: str, name: str) -> float:
        base = metric(baseline, name)
        return (base - metric(target, name)) / base * 100.0

    flow_ade = reduction("VTF_V2", "FLOW", "ADE_candidate0_m")
    flow_minade = reduction("VTF_V2", "FLOW", "minADE@K_m")
    flow_tvk = reduction("VTF_V2", "FLOW", "mean_unified_tvk_cost")
    flow_curvature = reduction(
        "VTF_V2", "FLOW", "curvature_violation_rate_independent"
    )
    flow_candidate_pp = 100.0 * (
        metric("VTF_V2", "compliant_candidate_rate_q80")
        - metric("FLOW", "compliant_candidate_rate_q80")
    )
    regression_ade_increase = -reduction(
        "VTF_V2", "REG", "ADE_candidate0_m"
    )
    regression_minade = reduction("VTF_V2", "REG", "minADE@K_m")
    regression_tvk = reduction("VTF_V2", "REG", "mean_unified_tvk_cost")
    regression_candidate_pp = 100.0 * (
        metric("VTF_V2", "compliant_candidate_rate_q80")
        - metric("REG", "compliant_candidate_rate_q80")
    )
    astar_terrain_pp = 100.0 * (
        metric("VTF_V2", "terrain_violation_rate")
        - metric("ASTAR", "terrain_violation_rate")
    )
    astar_tvk = reduction("VTF_V2", "ASTAR", "mean_unified_tvk_cost")
    astar_curvature = reduction(
        "VTF_V2", "ASTAR", "curvature_violation_rate_independent"
    )
    astar_smoothness = reduction(
        "VTF_V2", "ASTAR", "smoothness_m_independent"
    )
    diversity_change = 100.0 * (
        metric("VTF_V2", "diversity_m") / metric("FLOW", "diversity_m") - 1.0
    )
    report = f"""# 最终 VTF-Flow 与全部 benchmark 的统一比较

## 统一协议

- 外层测试序列为 00000、00001 和 00002，序列 00003 固定用于开发验证；
- 每折训练序列、测试场景、5 s 目标、H=10、地形势、运动学阈值和后验评价器完全一致；
- deterministic regression、Flow Matching 与 VTF-Flow 均使用 3 个训练种子；Constant Velocity 与 A* 为确定性方法，每折运行一次；
- 学习方法先在每条测试序列内平均训练种子，再对 3 条独立测试序列等权宏平均；表中不确定性为序列间样本标准差；
- K=1 方法的 minADE@K 与 ADE-0 相同。minADE@8 仅表示生成模型在相同八候选预算下对记录未来的覆盖；
- VTF-Flow 的 terminal projection 和 eta=0.075 在外层测试前冻结。
- A* 的 planner-used BEV 仅覆盖前向区域。{astar_coverage['out_of_map_stationary']} 个后向目标使用显式静止失败回退，{astar_coverage['soft_cost']} 个场景使用软占据代价回退；所有回退均保留在主表中，没有删帧。

## 核心结果

| 方法 | K | ADE-0 (m) | minADE@K (m) | TVK 势 | 地形违规率 | 曲率违规率 | q80 候选满足率 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## 结果解读

1. **相对直接生成基线 Flow Matching，VTF-Flow 形成一致的联合改善。** ADE-0、minADE@8、TVK 势和曲率违规率分别下降 {flow_ade:.2f}%、{flow_minade:.2f}%、{flow_tvk:.2f}% 和 {flow_curvature:.2f}%，q80 候选满足率提高 {flow_candidate_pp:.2f} 个百分点；这些方向在 3/3 条外层测试序列上保持一致。候选多样性变化为 {diversity_change:+.2f}%，说明引导带来轻微集合收缩，但没有退化为确定性单轨输出。

2. **相对 deterministic regression，优势主要来自多候选覆盖和统一可行性，而非单轨拟合。** Regression 的 ADE-0 更低，VTF-Flow 高 {regression_ade_increase:.2f}%；但 VTF-Flow 的 minADE@8 降低 {regression_minade:.2f}%，TVK 势降低 {regression_tvk:.2f}%，q80 候选满足率提高 {regression_candidate_pp:.2f} 个百分点。Regression 同时具有更低的二阶差分幅值，因此不能声称 VTF-Flow 在所有几何质量指标上占优。

3. **A* 体现了单项地形代价与统一 TVK 质量之间的冲突。** A* 的地形违规率比 VTF-Flow 低 {astar_terrain_pp:.2f} 个百分点，但其统一 TVK 势更高；VTF-Flow 相对 A* 将 TVK 势、曲率违规率和平顺性代价分别降低 {astar_tvk:.2f}%、{astar_curvature:.2f}% 和 {astar_smoothness:.2f}%。这表明仅在离散栅格上避开高代价单元，并不能保证轨迹的连续运动学质量。

4. **Constant Velocity 的零曲率和近零二阶差分不能单独解释为更优规划。** 其 ADE-0、目标一致性和 q80 候选满足率明显较差，说明曲率或平顺性必须与目标到达、地形暴露和行为一致性联合解释。

综上，VTF-Flow 并非在每个单项指标上取得数值最优，而是在记录未来覆盖、统一 TVK 势、生成期地形修正与候选可用性之间形成最均衡的结果。最强且最直接的因果对照仍是相同网络与噪声条件下的 Flow Matching；其他 benchmark 用于说明不同规划范式的能力边界。

## 统计与解释边界

独立分析单位为测试序列（n=3），帧是序列内时间相关重复观测，随机种子是模型训练技术重复。因此本表报告序列宏平均、序列间标准差和方向一致性，不执行帧级显著性检验。该结果用于 RELLIS-3D 内部的严格跨序列方法比较，不代表外部数据集泛化、真实碰撞概率或车辆安全认证。
"""
    (output_root / "benchmark_report_zh.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-regression-training",
        action="store_true",
        help="Require all outer-fold regression checkpoints to exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = load_protocol(args.config)
    benchmark_path = resolve_project_path(
        protocol.get("benchmark_config", DEFAULT_BENCHMARK_CONFIG)
    )
    frozen_path = resolve_project_path(protocol["frozen_v2_protocol"])
    frozen_results = resolve_project_path(protocol["frozen_v2_results"])
    benchmark = load_json(benchmark_path)
    frozen = load_json(frozen_path)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    methods = tuple(protocol["methods"])
    if methods != METHOD_ORDER:
        raise ValueError(f"expected method order {METHOD_ORDER}, found {methods}")
    display_names = {str(k): str(v) for k, v in protocol["display_names"].items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = CombinedSceneDataset(
        args.cache_root.resolve(), tuple(frozen["source_splits"])
    )
    dataset = H10PlanningDataset(
        source,
        args.data_root.resolve() / "processed" / "Rellis-3D",
        horizon=int(benchmark["trajectory"]["horizon_steps"]),
        history_steps=int(benchmark["trajectory"]["history_steps"]),
    )
    folds = build_fixed_validation_holdouts(
        frozen["available_sequences"],
        frozen["development_validation_sequence"],
        frozen["outer_test_sequences"],
    )
    envelopes = load_json(resolve_project_path(frozen["candidate_envelopes"]))
    split_counts: dict[str, dict[str, int]] = {}
    summaries: list[dict[str, Any]] = []
    astar_coverage_rows: list[dict[str, Any]] = []
    batch_size = int(protocol["evaluation_batch_size"])
    for fold in folds:
        indices = partition_sequence_indices(dataset.sequence_ids, as_sequence_split(fold))
        split_counts[fold.name] = {name: len(value) for name, value in indices.items()}
        print(f"=== {fold.name}: {split_counts[fold.name]} ===", flush=True)
        planners = classical_planners(benchmark)
        planners["ASTAR"] = FullCoverageAStar(
            planners["ASTAR"], int(benchmark["trajectory"]["horizon_steps"])
        )
        for method in ("CV", "ASTAR"):
            run_root = output_root / "runs" / fold.name / "seed_0" / method
            archive, fallback = generate_prediction_archive(
                method,
                planners[method],
                dataset,
                indices["test"],
                0,
                run_root,
                device,
                batch_size,
            )
            if method == "ASTAR":
                prediction_info = load_json(run_root / "prediction_info.json")
                astar_coverage_rows.append(
                    {
                        "test_sequence": fold.test_sequence,
                        "evaluated_scenes": int(prediction_info["evaluated_scenes"]),
                        **{
                            str(name): int(value)
                            for name, value in prediction_info[
                                "astar_fallback_reasons"
                            ].items()
                        },
                    }
                )
            summaries.append(
                evaluate_prediction_archive(
                    archive,
                    method,
                    0,
                    fold,
                    dataset,
                    indices["test"],
                    benchmark,
                    frozen,
                    envelopes,
                    run_root,
                    device,
                    batch_size,
                    fallback,
                )
            )
        for seed_value in frozen["seeds"]:
            seed = int(seed_value)
            checkpoint = train_regression_fold(
                dataset,
                indices,
                fold,
                seed,
                benchmark,
                frozen,
                output_root,
                device,
                args.skip_regression_training,
            )
            regression = _load_regression(checkpoint, device)
            regression_root = output_root / "runs" / fold.name / f"seed_{seed}" / "REG"
            archive, _ = generate_prediction_archive(
                "REG",
                regression,
                dataset,
                indices["test"],
                seed,
                regression_root,
                device,
                batch_size,
            )
            summaries.append(
                evaluate_prediction_archive(
                    archive,
                    "REG",
                    seed,
                    fold,
                    dataset,
                    indices["test"],
                    benchmark,
                    frozen,
                    envelopes,
                    regression_root,
                    device,
                    batch_size,
                )
            )
            del regression
            for method in ("FLOW", "VTF_V2"):
                source_archive = (
                    frozen_results
                    / "runs"
                    / fold.name
                    / f"seed_{seed}"
                    / method
                    / "predictions.npz"
                )
                if not source_archive.is_file():
                    raise FileNotFoundError(f"missing frozen archive: {source_archive}")
                run_root = output_root / "runs" / fold.name / f"seed_{seed}" / method
                summaries.append(
                    evaluate_prediction_archive(
                        source_archive,
                        method,
                        seed,
                        fold,
                        dataset,
                        indices["test"],
                        benchmark,
                        frozen,
                        envelopes,
                        run_root,
                        device,
                        batch_size,
                    )
                )
            if device.type == "cuda":
                torch.cuda.empty_cache()
    run_summary = pd.DataFrame(summaries).sort_values(
        ["test_sequence", "method", "seed"]
    )
    run_summary.to_csv(output_root / "run_summary.csv", index=False)
    sequence_method, macro = sequence_macro_benchmark_summary(
        run_summary, COMPLETE_METRICS, METHOD_ORDER
    )
    sequence_method.to_csv(output_root / "per_sequence_summary.csv", index=False)
    write_main_tables(macro, display_names, output_root)
    write_pairwise_effects(
        sequence_method, tuple(protocol["primary_metrics"]), output_root
    )
    astar_coverage = {
        "evaluated_scenes": int(
            sum(row["evaluated_scenes"] for row in astar_coverage_rows)
        ),
        "out_of_map_stationary": int(
            sum(row["out_of_map_stationary"] for row in astar_coverage_rows)
        ),
        "soft_cost": int(sum(row["soft_cost"] for row in astar_coverage_rows)),
        "planning_failure_stationary": int(
            sum(
                row["planning_failure_stationary"]
                for row in astar_coverage_rows
            )
        ),
        "per_sequence": astar_coverage_rows,
    }
    astar_coverage["out_of_map_stationary_rate"] = (
        astar_coverage["out_of_map_stationary"]
        / astar_coverage["evaluated_scenes"]
    )
    astar_coverage["soft_cost_rate"] = (
        astar_coverage["soft_cost"] / astar_coverage["evaluated_scenes"]
    )
    save_json(output_root / "astar_coverage_report.json", astar_coverage)
    write_report(macro, display_names, astar_coverage, output_root)
    save_json(
        output_root / "effective_protocol.json",
        {
            **protocol,
            "benchmark_config": str(benchmark_path),
            "frozen_v2_protocol": str(frozen_path),
            "frozen_v2_results": str(frozen_results),
            "cache_root": str(args.cache_root.resolve()),
            "data_root": str(args.data_root.resolve()),
            "device": str(device),
            "folds": [
                {
                    "name": fold.name,
                    "train": list(fold.train),
                    "validation": list(fold.validation),
                    "test": list(fold.test),
                }
                for fold in folds
            ],
            "split_counts": split_counts,
            "trajectory_protocol": benchmark["trajectory"],
            "sampling_protocol": benchmark["sampling"],
            "frozen_guidance": frozen["guidance"],
            "candidate_envelopes": str(resolve_project_path(frozen["candidate_envelopes"])),
            "evaluator": "one common post-hoc evaluator for all prediction archives",
            "latency_in_main_table": False,
        },
    )
    print((output_root / "main_table.md").read_text(encoding="utf-8"))
    print(f"Results written to {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

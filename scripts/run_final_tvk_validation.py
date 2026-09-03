"""Run the formal unified terrain--vehicle kinematic VTF-Flow validation.

The unchanged Flow and vehicle-terrain checkpoints are reused from the frozen
publication experiment.  A new Flow model is trained with non-zero curvature
and lateral-acceleration terms, then evaluated with and without in-flow TVK
guidance under the same sequence-disjoint protocol and paired Gaussian noise.
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
    SequenceSplit,
    benjamini_hochberg,
    bootstrap_mean_ci,
    paired_wilcoxon,
    partition_sequence_indices,
    read_csv,
    save_json,
    write_csv,
)
from TerraFlow.scripts.run_final_experiments import (  # noqa: E402
    _effective_flow_config,
    _evaluate_one,
)
from TerraFlow.scripts.run_flow_feasibility_experiment import (  # noqa: E402
    _train_one as train_flow_variant,
)
from TerraFlow.scripts.train_regression import CombinedSceneDataset  # noqa: E402


DEFAULT_CONFIG = TERRAFLOW_ROOT / "configs" / "final_tvk_validation.json"
DEFAULT_CACHE = WORKSPACE_ROOT / "data" / "RELLIS3D" / "trajectory_cache_h150_s5"
DEFAULT_BASELINE_ROOT = TERRAFLOW_ROOT / "outputs" / "final_experiments"
DEFAULT_OUTPUT = TERRAFLOW_ROOT / "outputs" / "final_experiments_tvk_final"

METHODS = {
    "A": "Flow baseline",
    "C_VT": "VTF-Flow w/o feasibility training and kinematic terms",
    "D_VT": "VTF-Flow w/o kinematic terms",
    "T_TVK": "VTF-Flow w/o inference-time guidance",
    "G_TVK": "VTF-Flow w/o feasibility training",
    "VTF": "VTF-Flow (ours)",
}

METRICS = (
    "minADE@K_m",
    "minFDE@K_m",
    "mean_vehicle_conditioned_cost",
    "mean_kinematic_cost",
    "mean_unified_tvk_cost",
    "terrain_violation_rate",
    "occupancy_violation_rate",
    "slope_violation_rate",
    "curvature_violation_rate",
    "lateral_acceleration_violation_rate",
    "smoothness_m",
    "diversity_m",
    "latency_ms_per_scene",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint(
    root: Path, split: str, seed: int, name: str
) -> Path:
    path = root / "checkpoints" / split / f"seed_{seed}" / name / "best.pt"
    if not path.is_file():
        raise FileNotFoundError(f"required frozen checkpoint is missing: {path}")
    return path


def _tvk_training_config(final: Mapping[str, Any], seed: int) -> dict[str, Any]:
    config = _effective_flow_config(final, seed)
    config["regularization"].update(
        {name: float(value) for name, value in final["kinematic"].items()}
    )
    config["experiment"]["lambda_feasibility"] = [
        float(final["training"]["tvk_lambda"])
    ]
    return config


def _train_tvk(
    source: CombinedSceneDataset,
    indices: Mapping[str, list[int]],
    split: SequenceSplit,
    seed: int,
    final: Mapping[str, Any],
    output_root: Path,
    device: torch.device,
) -> Path:
    output_dir = output_root / "checkpoints" / split.name / f"seed_{seed}" / "flow_tvk"
    checkpoint = output_dir / "best.pt"
    if checkpoint.is_file():
        return checkpoint
    config = _tvk_training_config(final, seed)
    _, summary = train_flow_variant(
        config,
        Subset(source, indices["train"]),
        Subset(source, indices["validation"]),
        output_dir,
        "vehicle",
        float(final["training"]["tvk_lambda"]),
        device,
        int(final["training"]["epochs"]),
    )
    save_json(output_dir / "training_summary.json", summary)
    return checkpoint


def _method_specs(
    baseline_root: Path,
    tvk_checkpoint: Path,
    split: SequenceSplit,
    seed: int,
    eta: float,
) -> dict[str, tuple[Path, float, bool]]:
    flow = _checkpoint(baseline_root, split.name, seed, "flow")
    vehicle = _checkpoint(baseline_root, split.name, seed, "flow_vehicle")
    return {
        "A": (flow, 0.0, False),
        "C_VT": (flow, eta, False),
        "D_VT": (vehicle, eta, False),
        "T_TVK": (tvk_checkpoint, 0.0, False),
        "G_TVK": (flow, eta, True),
        "VTF": (tvk_checkpoint, eta, True),
    }


def _run_split(
    source: CombinedSceneDataset,
    split: SequenceSplit,
    seeds: Sequence[int],
    final: Mapping[str, Any],
    baseline_root: Path,
    output_root: Path,
    device: torch.device,
) -> None:
    indices = partition_sequence_indices(source.sequence_ids, split)
    for seed in seeds:
        print(f"=== TVK formal validation: {split.name}, seed={seed} ===", flush=True)
        tvk_checkpoint = _train_tvk(
            source, indices, split, int(seed), final, output_root, device
        )
        specs = _method_specs(
            baseline_root, tvk_checkpoint, split, int(seed),
            float(final["guidance"]["eta"]),
        )
        for method, (checkpoint, eta, tvk_guidance) in specs.items():
            name = f"main_{split.name}_seed{seed}_{method}"
            print(f"  evaluating {name}", flush=True)
            _evaluate_one(
                method=method,
                checkpoint_path=checkpoint,
                source=source,
                dataset_indices=indices["test"],
                split=split,
                seed=int(seed),
                final=final,
                output_dir=output_root / name,
                device=device,
                eta=eta,
                smoothing=(
                    str(final["guidance"]["smoothing_kernel"])
                    if eta > 0.0 else "none"
                ),
                field_type="vehicle",
                use_kinematic_guidance=tvk_guidance,
            )


def _seed_summaries(output_root: Path, split: str, seeds: Sequence[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        records = [
            _load_json(output_root / f"main_{split}_seed{seed}_{method}" / "metrics.json")
            for seed in seeds
        ]
        row: dict[str, Any] = {
            "split": split,
            "method": method,
            "display_name": METHODS[method],
            "n_seeds": len(records),
        }
        for metric in METRICS:
            values = np.asarray([float(record[metric]) for record in records])
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = (
                float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            )
        rows.append(row)
    return rows


def _scene_seed_mean(
    output_root: Path, split: str, method: str, seeds: Sequence[int], metric: str
) -> tuple[list[str], np.ndarray]:
    per_seed: list[dict[str, float]] = []
    for seed in seeds:
        rows = read_csv(
            output_root / f"main_{split}_seed{seed}_{method}" / "scene_level_metrics.csv"
        )
        per_seed.append({str(row["scene_id"]): float(row[metric]) for row in rows})
    ids = sorted(per_seed[0])
    if any(set(values) != set(ids) for values in per_seed[1:]):
        raise ValueError(f"scene mismatch across seeds for {split}/{method}/{metric}")
    return ids, np.asarray(
        [[values[scene_id] for scene_id in ids] for values in per_seed],
        dtype=np.float64,
    ).mean(axis=0)


def _paired_statistics(
    output_root: Path,
    seeds: Sequence[int],
    final: Mapping[str, Any],
) -> list[dict[str, Any]]:
    comparisons = (
        ("A", "VTF"),
        ("C_VT", "G_TVK"),
        ("D_VT", "VTF"),
        ("T_TVK", "VTF"),
    )
    metrics = (
        "minADE@K_m",
        "mean_vehicle_conditioned_cost",
        "mean_unified_tvk_cost",
        "terrain_violation_rate",
        "curvature_violation_rate",
        "lateral_acceleration_violation_rate",
        "smoothness_m",
    )
    rows: list[dict[str, Any]] = []
    for baseline, target in comparisons:
        for metric_index, metric in enumerate(metrics):
            ids_a, a = _scene_seed_mean(output_root, "primary", baseline, seeds, metric)
            ids_b, b = _scene_seed_mean(output_root, "primary", target, seeds, metric)
            if ids_a != ids_b:
                raise ValueError("paired scene IDs differ")
            difference = b - a
            estimate, lower, upper = bootstrap_mean_ci(
                difference,
                resamples=int(final["statistics"]["bootstrap_resamples"]),
                seed=int(final["statistics"]["seed_for_bootstrap"])
                + metric_index,
            )
            test = paired_wilcoxon(a, b)
            rows.append({
                "comparison": f"{baseline}_vs_{target}",
                "difference_definition": f"{target} - {baseline}",
                "metric": metric,
                "mean_difference": estimate,
                "ci95_lower": lower,
                "ci95_upper": upper,
                **test,
            })
    adjusted = benjamini_hochberg([float(row["p_value"]) for row in rows])
    for row, value in zip(rows, adjusted):
        row["p_value_fdr_bh"] = float(value)
        row["significant_fdr_0.05"] = bool(value < 0.05)
    return rows


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _write_report(
    output_root: Path,
    primary: Sequence[Mapping[str, Any]],
    swapped: Sequence[Mapping[str, Any]],
    statistics: Sequence[Mapping[str, Any]],
) -> None:
    tables = output_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    write_csv(tables / "tvk_main_comparison.csv", primary)
    write_csv(tables / "tvk_cross_sequence.csv", swapped)
    write_csv(output_root / "tvk_statistical_tests.csv", statistics)

    columns = (
        "minADE@K_m", "mean_vehicle_conditioned_cost", "mean_kinematic_cost",
        "mean_unified_tvk_cost", "curvature_violation_rate",
        "lateral_acceleration_violation_rate", "smoothness_m",
    )
    header = (
        "| Method | minADE@K (m) | Vehicle cost | Kinematic cost | TVK cost | "
        "Curvature violation | Lateral-accel violation | Smoothness (m) |"
    )
    divider = "|---|---:|---:|---:|---:|---:|---:|---:|"
    lines = ["# Formal TVK validation", "", header, divider]
    for row in primary:
        cells = []
        for metric in columns:
            mean = float(row[f"{metric}_mean"])
            sd = float(row[f"{metric}_sd"])
            cells.append(f"{_fmt(mean)} ± {_fmt(sd)}")
        lines.append(f"| {row['display_name']} | " + " | ".join(cells) + " |")

    stat_lookup = {
        (str(row["comparison"]), str(row["metric"])): row for row in statistics
    }
    vtf = next(row for row in primary if row["method"] == "VTF")
    flow = next(row for row in primary if row["method"] == "A")
    without_kinematics = next(row for row in primary if row["method"] == "D_VT")
    lines.extend([
        "",
        "## Evidence interpretation",
        "",
        (
            "The primary protocol uses three independently trained seeds and a held-out "
            "test sequence. All feasibility and violation quantities are model-based "
            "diagnostics, not calibrated safety probabilities. Scene-level paired "
            "intervals are descriptive because neighbouring frames are temporally "
            "correlated; cross-sequence robustness is reported separately."
        ),
        "",
        (
            f"- VTF-Flow (ours) versus the Flow baseline: unified TVK cost "
            f"{float(flow['mean_unified_tvk_cost_mean']):.4f} -> "
            f"{float(vtf['mean_unified_tvk_cost_mean']):.4f}; minADE@K "
            f"{float(flow['minADE@K_m_mean']):.4f} -> "
            f"{float(vtf['minADE@K_m_mean']):.4f}."
        ),
        (
            f"- VTF-Flow (ours) versus VTF-Flow w/o kinematic terms: unified "
            f"TVK cost {float(without_kinematics['mean_unified_tvk_cost_mean']):.4f} -> "
            f"{float(vtf['mean_unified_tvk_cost_mean']):.4f}; curvature violation "
            f"{float(without_kinematics['curvature_violation_rate_mean']):.4f} -> "
            f"{float(vtf['curvature_violation_rate_mean']):.4f}."
        ),
    ])
    for metric in (
        "minADE@K_m", "mean_unified_tvk_cost", "curvature_violation_rate",
        "lateral_acceleration_violation_rate", "smoothness_m",
    ):
        row = stat_lookup[("A_vs_VTF", metric)]
        lines.append(
            f"- Paired A vs VTF, {metric}: mean difference "
            f"{float(row['mean_difference']):+.6f}, 95% CI "
            f"[{float(row['ci95_lower']):+.6f}, {float(row['ci95_upper']):+.6f}], "
            f"BH-FDR p={float(row['p_value_fdr_bh']):.3g}."
        )
    lines.extend([
        "",
        "## Boundary",
        "",
        "Curvature and lateral acceleration are trajectory-derived kinematic proxies. "
        "A continuous displacement-reliability gate attenuates curvature when adjacent "
        "waypoint displacement is insufficient for a stable estimate. "
        "The nominal limits are planning hyperparameters because RELLIS-3D does not "
        "provide reliable tyre-terrain friction or complete vehicle parameters. The "
        "results therefore do not establish full vehicle dynamics or formal safety.",
        "",
    ])
    (output_root / "final_tvk_validation.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    (TERRAFLOW_ROOT / "docs" / "final_tvk_validation.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    final = _load_json(args.config.resolve())
    source = CombinedSceneDataset(
        args.cache_root.resolve(), tuple(final["protocol"]["source_splits"])
    )
    splits = {
        name: SequenceSplit.from_mapping(name, final["protocol"][name])
        for name in ("primary", "swapped")
    }
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    save_json(output_root / "effective_config.json", {
        **final,
        "cache_root": str(args.cache_root.resolve()),
        "baseline_root": str(args.baseline_root.resolve()),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "interpretation": "relative feasibility diagnostics, not safety thresholds",
    })
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.skip_training:
        for split_name, seeds in (
            ("primary", final["seeds"]), ("swapped", final["swapped_seeds"])
        ):
            for seed in seeds:
                _checkpoint(output_root, split_name, int(seed), "flow_tvk")
    _run_split(
        source, splits["primary"], [int(value) for value in final["seeds"]],
        final, args.baseline_root.resolve(), output_root, device,
    )
    _run_split(
        source, splits["swapped"],
        [int(value) for value in final["swapped_seeds"]], final,
        args.baseline_root.resolve(), output_root, device,
    )
    primary = _seed_summaries(output_root, "primary", final["seeds"])
    swapped = _seed_summaries(output_root, "swapped", final["swapped_seeds"])
    statistics = _paired_statistics(output_root, final["seeds"], final)
    _write_report(output_root, primary, swapped, statistics)
    print(json.dumps({
        "status": "complete",
        "output_root": str(output_root),
        "primary_seeds": final["seeds"],
        "swapped_seeds": final["swapped_seeds"],
        "methods": METHODS,
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

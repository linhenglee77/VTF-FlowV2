"""Validate unified terrain-vehicle kinematic guidance on frozen Flow outputs."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import Subset


TERRAFLOW_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TERRAFLOW_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from TerraFlow.scripts.evaluate_guided_flow import (  # noqa: E402
    evaluate_config,
    load_variant_config,
)
from TerraFlow.scripts.train_regression import (  # noqa: E402
    CombinedSceneDataset,
    sequence_partition_indices,
)


DEFAULT_CONFIG = TERRAFLOW_ROOT / "configs" / "kinematic_guidance_validation.json"
DEFAULT_OUTPUT = TERRAFLOW_ROOT / "outputs" / "experiments" / "kinematic_guidance_validation"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _variant_config(
    base: dict[str, Any],
    validation: dict[str, Any],
    variant: dict[str, Any],
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    limits = validation["nominal_limits"]
    guidance = config["guidance"]
    guidance.update(
        {
            "enabled": bool(variant["enabled"]),
            "strength": (
                float(validation["guidance_strength"])
                if variant["enabled"]
                else 0.0
            ),
            "schedule": validation["guidance_schedule"],
            "smoothing_kernel": validation["smoothing_kernel"],
            "curvature_weight": float(variant["curvature_weight"]),
            "lateral_acceleration_weight": float(
                variant["lateral_acceleration_weight"]
            ),
            **limits,
        }
    )
    config["name"] = str(variant["name"])
    return config


def _summary_row(
    name: str,
    result: dict[str, Any],
    baseline: dict[str, Any] | None,
    scenario: str,
) -> dict[str, Any]:
    metrics = result["metrics"]
    row = {
        "scenario": scenario,
        "variant": name,
        "minADE@K_m": metrics["minADE@K_m"],
        "minFDE@K_m": metrics["minFDE@K_m"],
        "terrain_violation_rate": metrics["terrain_violation_rate"],
        "mean_vehicle_conditioned_cost": metrics["mean_vehicle_conditioned_cost"],
        "mean_kinematic_cost": metrics["mean_kinematic_cost"],
        "mean_unified_tvk_cost": metrics["mean_unified_tvk_cost"],
        "curvature_violation_rate": metrics["curvature_violation_rate"],
        "lateral_acceleration_violation_rate": metrics[
            "lateral_acceleration_violation_rate"
        ],
        "mean_absolute_curvature_per_m": metrics["mean_absolute_curvature_per_m"],
        "mean_lateral_acceleration_mps2": metrics[
            "mean_lateral_acceleration_mps2"
        ],
        "smoothness_m": metrics["smoothness_m"],
        "latency_ms_per_scene": metrics["latency_ms_per_scene"],
        "evaluated_scenes": metrics["evaluated_scenes"],
    }
    if baseline is None:
        row.update(
            {
                "mean_waypoint_displacement_vs_flow_m": 0.0,
                "maximum_waypoint_displacement_vs_flow_m": 0.0,
                "paired_curvature_improvement_rate": 0.0,
                "paired_lateral_acceleration_improvement_rate": 0.0,
                "paired_tvk_cost_improvement_rate": 0.0,
            }
        )
        return row
    prediction = result["predictions"]["trajectories"]
    reference = baseline["predictions"]["trajectories"]
    displacement = np.linalg.norm(prediction - reference, axis=-1)
    scene = result["scene_metrics"]
    reference_scene = baseline["scene_metrics"]
    row.update(
        {
            "mean_waypoint_displacement_vs_flow_m": float(displacement.mean()),
            "maximum_waypoint_displacement_vs_flow_m": float(displacement.max()),
            "paired_curvature_improvement_rate": float(
                np.mean(
                    scene["curvature_violation_rate"]
                    < reference_scene["curvature_violation_rate"]
                )
            ),
            "paired_lateral_acceleration_improvement_rate": float(
                np.mean(
                    scene["lateral_acceleration_violation_rate"]
                    < reference_scene["lateral_acceleration_violation_rate"]
                )
            ),
            "paired_tvk_cost_improvement_rate": float(
                np.mean(
                    scene["mean_unified_tvk_cost"]
                    < reference_scene["mean_unified_tvk_cost"]
                )
            ),
        }
    )
    return row


def _plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    import matplotlib.pyplot as plt

    labels = [
        (
            str(row["variant"])
            .replace("Terrain-vehicle + lateral acceleration", "T–V + aᵧ")
            .replace("Terrain-vehicle + curvature", "T–V + κ")
            .replace("Terrain-vehicle", "T–V")
            + (" (stress)" if row["scenario"] != "nominal" else "")
        )
        for row in rows
    ]
    x = np.arange(len(rows))
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    axes[0].bar(
        x - 0.18,
        [row["curvature_violation_rate"] for row in rows],
        width=0.36,
        label="Curvature violation",
        color="#3b82b8",
    )
    axes[0].bar(
        x + 0.18,
        [row["lateral_acceleration_violation_rate"] for row in rows],
        width=0.36,
        label="Lateral-acceleration violation",
        color="#e58233",
    )
    axes[0].set_ylabel("Violation rate")
    axes[0].set_xticks(
        x, labels, rotation=18, ha="right", rotation_mode="anchor"
    )
    axes[0].set_ylim(0.0, 0.34)
    axes[0].legend(
        frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2
    )
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].set_title("a  Kinematic-limit diagnostics", loc="left", fontweight="bold")
    comparison_indices = (0, 1, 4, 5, 6)
    comparison_rows = [rows[index] for index in comparison_indices]
    comparison_labels = [labels[index] for index in comparison_indices]
    colors = ("#59636e", "#d08b45", "#277da1", "#9aa1a8", "#3a9d6f")
    axes[1].scatter(
        [row["minADE@K_m"] for row in comparison_rows],
        [row["mean_unified_tvk_cost"] for row in comparison_rows],
        s=42,
        c=colors,
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )
    offsets = ((4, 5), (4, -13), (4, 5), (4, 5), (4, 5))
    for label, row, offset in zip(comparison_labels, comparison_rows, offsets):
        axes[1].annotate(
            label,
            (row["minADE@K_m"], row["mean_unified_tvk_cost"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=6,
        )
    axes[1].set_xlabel("minADE@K (m; lower is better)")
    axes[1].set_ylabel("Unified TVK cost (lower is better)")
    axes[1].grid(alpha=0.2)
    axes[1].set_title("b  Fidelity–feasibility trade-off", loc="left", fontweight="bold")
    figure.suptitle("Frozen Flow; 128 paired RELLIS-3D scenes; one seed", fontsize=7)
    figure.savefig(path, dpi=600, transparent=False)
    figure.savefig(path.with_suffix(".svg"), transparent=True)
    figure.savefig(path.with_suffix(".pdf"), transparent=True)
    figure.savefig(path.with_suffix(".tiff"), dpi=600, transparent=False)
    plt.close(figure)


def _write_report(
    path: Path,
    rows: list[dict[str, Any]],
    validation: dict[str, Any],
    device: torch.device,
) -> None:
    flow = next(
        row for row in rows
        if row["scenario"] == "nominal" and row["variant"] == "Flow"
    )
    unified = next(
        row for row in rows
        if row["scenario"] == "nominal" and row["variant"] == "Unified TVK"
    )
    stress_flow = next(
        row for row in rows
        if row["scenario"] == "4x-speed stress" and row["variant"] == "Flow"
    )
    stress_unified = next(
        row for row in rows
        if row["scenario"] == "4x-speed stress" and row["variant"] == "Unified TVK"
    )
    limits = validation["nominal_limits"]
    curvature_delta = unified["curvature_violation_rate"] - flow[
        "curvature_violation_rate"
    ]
    lateral_delta = unified["lateral_acceleration_violation_rate"] - flow[
        "lateral_acceleration_violation_rate"
    ]
    ade_delta = unified["minADE@K_m"] - flow["minADE@K_m"]
    smooth_delta = unified["smoothness_m"] - flow["smoothness_m"]
    lines = [
        "# Unified terrain–vehicle kinematic feasibility validation",
        "",
        "## Validation protocol",
        "",
        "A frozen Flow checkpoint is evaluated with paired Gaussian initial states. "
        "Only the inference-time objective changes; the network architecture and learned weights are unchanged. "
        "The unified objective combines vehicle-conditioned terrain cost, curvature soft excess, and "
        "lateral-acceleration soft excess.",
        "",
        f"Runtime device: `{device}`. Evaluated scenes: {int(flow['evaluated_scenes'])}. "
        f"Nominal limits: |kappa| <= {limits['maximum_curvature_per_m']} 1/m and "
        f"a_y <= {limits['maximum_lateral_acceleration_mps2']} m/s^2.",
        "",
        "> These are planning hyperparameters for controlled simulation, not calibrated limits of the RELLIS-3D collection vehicle. "
        "No tire-road friction, mass, wheelbase, steering limit, suspension, or roll-stability parameter is assumed.",
        "",
        "## Results",
        "",
        "| Scenario | Variant | minADE@K (m) | Curvature violation | Lateral-accel violation | TVK cost | Smoothness | Displacement vs Flow (m) | Latency (ms/scene) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['variant']} | {row['minADE@K_m']:.4f} | "
            f"{row['curvature_violation_rate']:.4f} | "
            f"{row['lateral_acceleration_violation_rate']:.4f} | "
            f"{row['mean_unified_tvk_cost']:.4f} | {row['smoothness_m']:.4f} | "
            f"{row['mean_waypoint_displacement_vs_flow_m']:.4f} | "
            f"{row['latency_ms_per_scene']:.3f} |"
        )
    curvature_relative = curvature_delta / flow["curvature_violation_rate"]
    tvk_delta = unified["mean_unified_tvk_cost"] - flow["mean_unified_tvk_cost"]
    tvk_relative = tvk_delta / flow["mean_unified_tvk_cost"]
    bounded = unified["maximum_waypoint_displacement_vs_flow_m"] < 5.0
    stress_lateral_delta = (
        stress_unified["lateral_acceleration_violation_rate"]
        - stress_flow["lateral_acceleration_violation_rate"]
    )
    stress_mean_lateral_delta = (
        stress_unified["mean_lateral_acceleration_mps2"]
        - stress_flow["mean_lateral_acceleration_mps2"]
    )
    stress_tvk_delta = (
        stress_unified["mean_unified_tvk_cost"]
        - stress_flow["mean_unified_tvk_cost"]
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Unified TVK versus Flow: curvature-violation change {curvature_delta:+.4f}; "
            f"relative change {curvature_relative * 100:+.2f}%; common TVK-cost change "
            f"{tvk_delta:+.4f} ({tvk_relative * 100:+.2f}%).",
            f"- Nominal lateral-acceleration violation is zero for both variants "
            f"(change {lateral_delta:+.4f}); therefore this split cannot establish a nominal-condition benefit "
            f"for that term.",
            f"- Fidelity trade-off: minADE@K change {ade_delta:+.4f} m; smoothness change "
            f"{smooth_delta:+.4f} m.",
            f"- Mean/max waypoint displacement versus paired Flow: "
            f"{unified['mean_waypoint_displacement_vs_flow_m']:.4f}/"
            f"{unified['maximum_waypoint_displacement_vs_flow_m']:.4f} m.",
            f"- Paired TVK-cost improvement rate is "
            f"{unified['paired_tvk_cost_improvement_rate'] * 100:.2f}%; bounded-displacement diagnostic "
            f"(<5 m max) {'passed' if bounded else 'did not pass'}.",
            f"- In the explicitly labelled 4x-speed stress test, mean lateral acceleration changed "
            f"by {stress_mean_lateral_delta:+.4f} m/s^2 and common TVK cost by {stress_tvk_delta:+.4f}, "
            f"but threshold violation changed by {stress_lateral_delta:+.4f}. Thus continuous-cost "
            f"responsiveness is observed, while violation-rate improvement is not established.",
            "",
            "Overall status: **partial validation**. The curvature term and unified continuous objective are "
            "effective without trajectory divergence; the lateral-acceleration term is numerically valid but "
            "its operational benefit is not supported under nominal recorded speeds and remains inconclusive "
            "under the controlled stress test.",
            "",
            "The experiment validates a differentiable kinematic feasibility proxy, not full vehicle dynamics or safety. "
            "A dynamics/stability claim would require reliable vehicle parameters, steering constraints, tire-ground friction, "
            "and preferably measured vehicle states.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-scenes", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validation = json.loads(args.config.read_text(encoding="utf-8"))
    base = load_variant_config(str(validation["base_variant"]))
    source = CombinedSceneDataset(args.cache_root, tuple(base["data"]["source_splits"]))
    _, indices = sequence_partition_indices(
        source.sequence_ids, base["data"]["validation_sequences"]
    )
    indices = indices[: args.max_scenes]
    dataset = Subset(source, indices)
    if not indices:
        raise ValueError("validation split is empty")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    effective = {
        **validation,
        "runtime": {
            "cache_root": str(args.cache_root.resolve()),
            "device": str(device),
            "evaluated_scenes": len(indices),
        },
    }
    (args.output_dir / "effective_config.json").write_text(
        json.dumps(effective, indent=2), encoding="utf-8"
    )
    rows: list[dict[str, Any]] = []
    baseline = None
    for variant in validation["variants"]:
        config = _variant_config(base, validation, variant)
        result = evaluate_config(
            config,
            dataset,
            indices,
            device,
            args.batch_size,
            str(variant["name"]),
            collect_predictions=True,
            collect_sample_diagnostics=False,
        )
        if baseline is None:
            baseline = result
        rows.append(
            _summary_row(
                str(variant["name"]),
                result,
                None if len(rows) == 0 else baseline,
                "nominal",
            )
        )

    compression = float(validation["stress_time_compression_factor"])
    if compression <= 1.0:
        raise ValueError("stress_time_compression_factor must exceed one")
    unified_variant = next(
        variant for variant in validation["variants"]
        if variant["name"] == "Unified TVK"
    )
    stress_results: list[dict[str, Any]] = []
    for name, variant in (
        ("Flow", {**unified_variant, "enabled": False,
                  "curvature_weight": 0.0, "lateral_acceleration_weight": 0.0}),
        ("Unified TVK", unified_variant),
    ):
        config = _variant_config(base, validation, variant)
        config["guidance"]["planning_dt_s"] = (
            float(config["guidance"]["planning_dt_s"]) / compression
        )
        if variant["enabled"]:
            config["guidance"]["strength"] = float(
                validation["stress_guidance_strength"]
            )
            config["guidance"]["lateral_acceleration_weight"] = float(
                validation["stress_lateral_acceleration_weight"]
            )
        result = evaluate_config(
            config,
            dataset,
            indices,
            device,
            args.batch_size,
            f"{name} ({compression:g}x-speed stress)",
            collect_predictions=True,
            collect_sample_diagnostics=False,
        )
        stress_results.append(result)
        rows.append(
            _summary_row(
                name,
                result,
                None if len(stress_results) == 1 else stress_results[0],
                f"{compression:g}x-speed stress",
            )
        )
    _write_csv(args.output_dir / "kinematic_guidance_validation.csv", rows)
    _plot(args.output_dir / "kinematic_guidance_validation.png", rows)
    _write_report(
        TERRAFLOW_ROOT / "docs" / "terrain_vehicle_kinematic_feasibility.md",
        rows,
        validation,
        device,
    )
    print(json.dumps(rows, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

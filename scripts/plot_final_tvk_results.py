"""Plot the formal unified terrain--vehicle kinematic validation results.

The figure is intentionally evidence-led: it reports the accuracy--feasibility
trade-off, paired changes against Flow, kinematic violation rates, and the
held-out sequence-swap check.  All feasibility quantities are relative model
diagnostics rather than calibrated safety probabilities.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TERRAFLOW_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = TERRAFLOW_ROOT / "outputs" / "final_experiments_tvk_final"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _rows_by_method(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["method"]: row for row in rows}


def _relative_change(target: float, baseline: float) -> float:
    return 100.0 * (target - baseline) / max(abs(baseline), 1e-12)


def _save_all(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", transparent=True)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", transparent=True)
    fig.savefig(base.with_suffix(".png"), dpi=360, bbox_inches="tight", facecolor="white")
    fig.savefig(
        base.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
        facecolor="white", pil_kwargs={"compression": "tiff_lzw"},
    )


def render(output_root: Path) -> Path:
    tables = output_root / "tables"
    primary = _rows_by_method(_read_csv(tables / "tvk_main_comparison.csv"))
    swapped = _rows_by_method(_read_csv(tables / "tvk_cross_sequence.csv"))
    statistics = _read_csv(output_root / "tvk_statistical_tests.csv")
    paired = {
        row["metric"]: row
        for row in statistics
        if row["comparison"] == "A_vs_VTF"
    }

    methods = ["A", "C_VT", "D_VT", "T_TVK", "G_TVK", "VTF"]
    labels = {
        "A": "Flow baseline",
        "C_VT": "w/o training & kinematics",
        "D_VT": "w/o kinematic terms",
        "T_TVK": "w/o inference guidance",
        "G_TVK": "w/o feasibility training",
        "VTF": "VTF-Flow (ours)",
    }
    colors = {
        "A": "#77828f",
        "C_VT": "#3b82a0",
        "D_VT": "#2563a5",
        "T_TVK": "#c58b2b",
        "G_TVK": "#2b8a6e",
        "VTF": "#a62d4f",
    }

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8.2,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.5,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.fontsize": 7.1,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.35), constrained_layout=True)

    # a | Accuracy--feasibility operating points.
    ax = axes[0, 0]
    for method in methods:
        row = primary[method]
        ax.errorbar(
            _number(row, "minADE@K_m_mean"),
            _number(row, "mean_unified_tvk_cost_mean"),
            xerr=_number(row, "minADE@K_m_sd"),
            yerr=_number(row, "mean_unified_tvk_cost_sd"),
            fmt="o", ms=5.2 if method == "VTF" else 4.2,
            color=colors[method], ecolor=colors[method], elinewidth=0.8,
            capsize=2.0, label=labels[method], zorder=4 if method == "VTF" else 3,
        )
    ax.set_xlabel(r"minADE@8 (m)  $\leftarrow$ lower")
    ax.set_ylabel(r"Unified TVK cost  $\leftarrow$ lower")
    ax.set_title("a  Accuracy–feasibility trade-off", loc="left", fontweight="bold")
    ax.grid(color="#d7dce2", linewidth=0.5, alpha=0.75)
    ax.legend(frameon=False, ncol=2, handletextpad=0.3, columnspacing=0.8)

    # b | Paired proposed-method effect with scene-level bootstrap intervals.
    ax = axes[0, 1]
    metric_info = [
        ("minADE@K_m", "minADE@8"),
        ("mean_vehicle_conditioned_cost", "Vehicle cost"),
        ("mean_unified_tvk_cost", "Unified TVK cost"),
        ("curvature_violation_rate", "Curvature violations"),
        ("smoothness_m", "Smoothness"),
    ]
    flow = primary["A"]
    effects: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for metric, _ in metric_info:
        baseline = _number(flow, f"{metric}_mean")
        stat = paired[metric]
        effects.append(100.0 * _number(stat, "mean_difference") / max(abs(baseline), 1e-12))
        lows.append(100.0 * _number(stat, "ci95_lower") / max(abs(baseline), 1e-12))
        highs.append(100.0 * _number(stat, "ci95_upper") / max(abs(baseline), 1e-12))
    ypos = np.arange(len(metric_info))
    effects_array = np.asarray(effects)
    ax.errorbar(
        effects_array, ypos,
        xerr=np.vstack((effects_array - np.asarray(lows), np.asarray(highs) - effects_array)),
        fmt="o", color=colors["VTF"], ecolor="#5f6570", elinewidth=1.0,
        capsize=2.2, ms=4.8,
    )
    ax.axvline(0.0, color="#252a30", linewidth=0.8)
    ax.set_yticks(ypos, [label for _, label in metric_info])
    ax.invert_yaxis()
    ax.set_xlabel("Paired change, VTF-Flow − Flow (%)")
    ax.set_title("b  VTF-Flow paired effect (95% CI)", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#d7dce2", linewidth=0.5, alpha=0.75)

    # c | Kinematic violation rates, shown separately from terrain cost.
    ax = axes[1, 0]
    selected = ["A", "D_VT", "VTF"]
    x = np.arange(len(selected))
    width = 0.34
    curvature = [100.0 * _number(primary[m], "curvature_violation_rate_mean") for m in selected]
    lateral = [100.0 * _number(primary[m], "lateral_acceleration_violation_rate_mean") for m in selected]
    curvature_sd = [100.0 * _number(primary[m], "curvature_violation_rate_sd") for m in selected]
    lateral_sd = [100.0 * _number(primary[m], "lateral_acceleration_violation_rate_sd") for m in selected]
    ax.bar(x - width / 2, curvature, width, yerr=curvature_sd, color="#8759a6", label="Curvature", capsize=2)
    ax.bar(x + width / 2, lateral, width, yerr=lateral_sd, color="#d98b3a", label="Lateral acceleration", capsize=2)
    ax.set_xticks(
        x, [labels[m] for m in selected], rotation=12, ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylabel("Violation rate (%)")
    ax.set_title("c  Nominal kinematic-limit diagnostics", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#d7dce2", linewidth=0.5, alpha=0.75)
    ax.legend(frameon=False, ncol=2, loc="upper right")

    # d | Sequence-swap direction check.
    ax = axes[1, 1]
    splits = [("Primary test: 00004", primary), ("Swapped test: 00003", swapped)]
    x = np.arange(len(splits))
    width = 0.32
    minade_delta = [
        _relative_change(_number(rows["VTF"], "minADE@K_m_mean"), _number(rows["A"], "minADE@K_m_mean"))
        for _, rows in splits
    ]
    tvk_delta = [
        _relative_change(
            _number(rows["VTF"], "mean_unified_tvk_cost_mean"),
            _number(rows["A"], "mean_unified_tvk_cost_mean"),
        )
        for _, rows in splits
    ]
    ax.bar(x - width / 2, minade_delta, width, color="#4169a1", label="Minimum ADE@8")
    ax.bar(x + width / 2, tvk_delta, width, color="#2b8a6e", label="Unified TVK cost")
    ax.axhline(0.0, color="#252a30", linewidth=0.8)
    ax.set_xticks(x, [label for label, _ in splits])
    ax.set_ylabel("Relative change vs Flow (%)")
    ax.set_title("d  Held-out sequence robustness", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#d7dce2", linewidth=0.5, alpha=0.75)
    ax.legend(frameon=False, ncol=2, loc="best")

    fig.suptitle(
        "Unified terrain–vehicle kinematic validation",
        fontsize=10.6, fontweight="bold", color="#173c5d",
    )
    fig.text(
        0.5, -0.012,
        "Mean ± SD across three primary seeds; feasibility values are relative diagnostics, not safety probabilities.",
        ha="center", fontsize=7.2, color="#555d66",
    )
    base = output_root / "figures" / "figure_tvk_quantitative_summary"
    _save_all(fig, base)
    plt.close(fig)
    return base.with_suffix(".png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    output_root = parse_args().output_root.resolve()
    figure = render(output_root)
    print(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

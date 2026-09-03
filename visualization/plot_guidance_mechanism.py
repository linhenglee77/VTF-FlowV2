"""Publication figures for the paired feasibility-guidance mechanism audit."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    "Unguided Flow",
    "Generate-then-filter",
    "Generate-then-refine",
    "Feasibility-guided Flow",
    "Guided Flow (no reranking)",
)
DISPLAY = {
    "Unguided Flow": "Unguided",
    "Generate-then-filter": "Filter",
    "Generate-then-refine": "Refine",
    "Feasibility-guided Flow": "Guided + rank",
    "Guided Flow (no reranking)": "Guided only",
}
COLORS = {
    "Unguided Flow": "#667085",
    "Generate-then-filter": "#E07A00",
    "Generate-then-refine": "#7A68CC",
    "Feasibility-guided Flow": "#009E73",
    "Guided Flow (no reranking)": "#3977B8",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def configure() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def save_bundle(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.tiff", dpi=600, bbox_inches="tight")


def seed_series(rows, method, phase):
    selected = [row for row in rows if row["method"] == method and row["phase"] == phase]
    seeds = sorted({int(row["seed"]) for row in selected})
    steps = sorted({int(row["step"]) for row in selected})
    values = np.asarray([
        [
            float(next(row["cost_mean"] for row in selected if int(row["seed"]) == seed and int(row["step"]) == step))
            for step in steps
        ]
        for seed in seeds
    ])
    return np.asarray(steps), values


def metric_seed_values(rows, method, metric):
    return np.asarray([
        float(row[metric]) for row in rows if row["method"] == method
    ])


def add_panel_label(axis, label):
    axis.text(-0.14, 1.07, label, transform=axis.transAxes, fontsize=9, fontweight="bold")


def mechanism_figure(evolution, paired, closed_loop, output_dir):
    fig = plt.figure(figsize=(7.244094488, 3.35), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1, 1])
    ax_process = fig.add_subplot(grid[:, 0])
    ax_tradeoff = fig.add_subplot(grid[0, 1])
    ax_paired = fig.add_subplot(grid[1, 1])

    for method, linestyle, zorder in (
        ("Unguided Flow", "-", 2),
        ("Generate-then-filter", "--", 3),
        ("Feasibility-guided Flow", "-", 5),
    ):
        steps, values = seed_series(evolution, method, "flow")
        mean = values.mean(axis=0)
        spread = values.std(axis=0, ddof=1)
        label = DISPLAY[method]
        if method == "Generate-then-filter":
            label += " (same Flow path)"
        ax_process.plot(steps, mean, color=COLORS[method], lw=1.8, ls=linestyle, label=label, zorder=zorder)
        ax_process.fill_between(steps, mean - spread, mean + spread, color=COLORS[method], alpha=0.10, lw=0)
    refine_steps, refine = seed_series(evolution, "Generate-then-refine", "refinement")
    refine_x = refine_steps + 16
    refine_mean = refine.mean(axis=0)
    refine_sd = refine.std(axis=0, ddof=1)
    ax_process.plot(refine_x, refine_mean, color=COLORS["Generate-then-refine"], lw=1.8, label="Post-hoc refine")
    ax_process.fill_between(refine_x, refine_mean - refine_sd, refine_mean + refine_sd, color=COLORS["Generate-then-refine"], alpha=0.10, lw=0)
    ax_process.axvline(15.5, color="#B8BDC7", lw=0.8, ls=":")
    ax_process.text(7.5, 1.04, "Flow evolution", ha="center", color="#667085")
    ax_process.text(19.5, 1.04, "Post-process", ha="center", color="#667085")
    ax_process.set_xlabel("Optimization step")
    ax_process.set_ylabel("Unified feasibility objective")
    ax_process.set_title("Guidance changes the trajectory before terminal selection", loc="left")
    ax_process.set_xlim(0, 23)
    ax_process.set_ylim(0.34, 1.12)
    ax_process.grid(axis="y", color="#E5E7EB", lw=0.6)
    ax_process.legend(loc="lower left", frameon=False, ncol=2)
    add_panel_label(ax_process, "a")

    for method in METHODS:
        time_values = metric_seed_values(closed_loop, method, "planning_time_ms_per_replan")
        collision_values = 100 * metric_seed_values(closed_loop, method, "collision")
        ax_tradeoff.scatter(time_values, collision_values, s=14, color=COLORS[method], alpha=0.28, edgecolors="none")
        ax_tradeoff.scatter(time_values.mean(), collision_values.mean(), s=42, color=COLORS[method], edgecolors="white", linewidths=0.7, label=DISPLAY[method], zorder=5)
    ax_tradeoff.set_xlabel("Planning time (ms/replan)")
    ax_tradeoff.set_ylabel("Collision episodes (%)")
    ax_tradeoff.set_title("Closed-loop safety–compute trade-off", loc="left")
    ax_tradeoff.grid(color="#E5E7EB", lw=0.6)
    ax_tradeoff.legend(loc="upper right", fontsize=5.2, handletextpad=0.3, borderaxespad=0.2)
    add_panel_label(ax_tradeoff, "b")

    paired_methods = METHODS[1:]
    x = np.arange(len(paired_methods))
    for index, method in enumerate(paired_methods):
        values = np.asarray([
            float(row["paired_improvement_mean"])
            for row in paired if row["method"] == method
        ])
        ax_paired.scatter(np.full(len(values), index), values, s=15, color=COLORS[method], alpha=0.38, zorder=3)
        ax_paired.errorbar(index, values.mean(), yerr=values.std(ddof=1), fmt="o", ms=5, color=COLORS[method], capsize=2.5, lw=1.2, zorder=4)
    ax_paired.axhline(0, color="#8C939F", lw=0.8)
    ax_paired.set_xticks(x, [DISPLAY[method] for method in paired_methods], rotation=24, ha="right", rotation_mode="anchor")
    ax_paired.set_ylabel("Paired terminal objective improvement")
    ax_paired.set_title("Only guidance/refine changes candidate feasibility", loc="left")
    ax_paired.grid(axis="y", color="#E5E7EB", lw=0.6)
    add_panel_label(ax_paired, "c")

    fig.suptitle("Feasibility guidance acts within Flow, while post-hoc refinement remains the stronger optimizer", fontsize=10)
    save_bundle(fig, output_dir, "guidance_mechanism_evidence")
    plt.close(fig)


def ablation_figure(closed_loop, output_dir):
    fig, axes = plt.subplots(1, 4, figsize=(7.244094488, 2.65), constrained_layout=True)
    panels = (
        ("completion", "Completion (%)", 100.0, True),
        ("collision", "Collision episodes (%)", 100.0, False),
        ("conservative_risk", "Conservative executed risk", 1.0, False),
        ("mean_abs_steering_deg", "Mean |steering| (deg)", 1.0, False),
    )
    x = np.arange(len(METHODS))
    for panel_index, (axis, (metric, ylabel, scale, higher)) in enumerate(zip(axes, panels)):
        for method_index, method in enumerate(METHODS):
            values = scale * metric_seed_values(closed_loop, method, metric)
            axis.scatter(np.full(len(values), method_index), values, s=15, color=COLORS[method], alpha=0.35)
            axis.errorbar(method_index, values.mean(), yerr=values.std(ddof=1), fmt="o", ms=5.5, color=COLORS[method], capsize=2.5, lw=1.2)
        axis.set_xticks(x, [DISPLAY[method] for method in METHODS], rotation=31, ha="right", rotation_mode="anchor")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#E5E7EB", lw=0.6)
        axis.set_title("higher is better" if higher else "lower is better", color="#666666", fontsize=6.5)
        add_panel_label(axis, chr(ord("a") + panel_index))
    fig.suptitle("Unified closed-loop ablation separates guidance from filtering and refinement", fontsize=10)
    save_bundle(fig, output_dir, "guidance_closed_loop_ablation")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    configure()
    evolution = read_csv(args.input_dir / "first_replan_flow_evolution.csv")
    paired = read_csv(args.input_dir / "first_replan_paired_summary.csv")
    closed_loop = read_csv(args.input_dir / "per_seed_metrics.csv")
    mechanism_figure(evolution, paired, closed_loop, args.output_dir)
    ablation_figure(closed_loop, args.output_dir)


if __name__ == "__main__":
    main()

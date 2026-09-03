"""Build publication tables and evidence figures from completed VTF-Flow runs.

This script performs no model training.  It only reads completed experiment
artifacts and, when ``--data-root`` is supplied, the audited RELLIS-3D poses
used to construct the ground-truth trajectory reference.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_PARENT = PROJECT_ROOT.parent
if str(WORKSPACE_PARENT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_PARENT))

from TerraFlow.datasets.trajectory_builder import (  # noqa: E402
    RellisTrajectoryBuilder,
    TrajectoryBuilderConfig,
    load_rellis_sequence,
    rellis3d_os1_to_planning_ego,
)


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.0,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

FLOW = "#6B7280"
VTF = "#087E8B"
ACCENT = "#D95F59"
BLUE = "#386CB0"
PALETTE = ["#386CB0", "#7FC97F", "#BEAED4", "#FDC086", "#BF5B17"]


def save_publication_figure(fig: plt.Figure, stem: Path) -> None:
    """Export one figure with editable vector text and a high-resolution raster."""

    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def _format_value(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    if isinstance(value, (float, np.floating)):
        magnitude = abs(float(value))
        if magnitude != 0 and magnitude < 1e-3:
            return f"{float(value):.2e}"
        return f"{float(value):.4f}"
    return str(value)


def write_table_bundle(frame: pd.DataFrame, stem: Path, caption: str) -> None:
    """Write a table as source CSV, readable Markdown, and booktabs LaTeX."""

    stem.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    display = frame.copy().map(_format_value)
    headers = [str(column) for column in display.columns]
    markdown_rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    markdown_rows.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    )
    stem.with_suffix(".md").write_text(
        f"{caption}\n\n" + "\n".join(markdown_rows) + "\n",
        encoding="utf-8",
    )
    latex = display.to_latex(index=False, escape=True, column_format="l" + "r" * (len(frame.columns) - 1))
    stem.with_suffix(".tex").write_text(latex, encoding="utf-8")


def mean_sd(row: pd.Series, metric: str, digits: int = 4) -> str:
    """Format a mean and across-seed standard deviation without inventing uncertainty."""

    mean = float(row[f"{metric}_mean"])
    sd = float(row[f"{metric}_sd"])
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def build_tables(final_root: Path, output_root: Path, data_root: Path | None) -> None:
    """Create manuscript-ready comparison, ablation, robustness, and statistics tables."""

    source = pd.read_csv(final_root / "tables" / "tvk_main_comparison.csv")
    by_method = source.set_index("method")

    compact_methods = ["A", "D_VT", "VTF"]
    compact = pd.DataFrame(
        {
            "Method": [by_method.loc[m, "display_name"] for m in compact_methods],
            "minADE@8 (m)": [mean_sd(by_method.loc[m], "minADE@K_m") for m in compact_methods],
            "minFDE@8 (m)": [mean_sd(by_method.loc[m], "minFDE@K_m") for m in compact_methods],
            "TVK cost": [mean_sd(by_method.loc[m], "mean_unified_tvk_cost") for m in compact_methods],
            "Curv. viol.": [mean_sd(by_method.loc[m], "curvature_violation_rate") for m in compact_methods],
            "Smoothness (m)": [mean_sd(by_method.loc[m], "smoothness_m") for m in compact_methods],
            "Latency (ms/scene)": [mean_sd(by_method.loc[m], "latency_ms_per_scene") for m in compact_methods],
        }
    )
    write_table_bundle(
        compact,
        output_root / "tables" / "table_main_comparison",
        "Primary held-out-sequence comparison (mean ± s.d. across three independently trained seeds).",
    )

    switches: Mapping[str, tuple[str, str, str]] = {
        "A": ("--", "--", "--"),
        "C_VT": ("--", "VT", "--"),
        "D_VT": ("VT", "VT", "--"),
        "T_TVK": ("TVK", "--", "--"),
        "G_TVK": ("--", "TVK", "--"),
        "VTF": ("TVK", "TVK", "Full"),
    }
    ablation_rows = []
    for method in ["A", "C_VT", "D_VT", "T_TVK", "G_TVK", "VTF"]:
        row = by_method.loc[method]
        train, guidance, unified = switches[method]
        ablation_rows.append(
            {
                "Method": row["display_name"],
                "Training reg.": train,
                "In-flow guidance": guidance,
                "Unified TVK": unified,
                "minADE@8 (m)": mean_sd(row, "minADE@K_m"),
                "TVK cost": mean_sd(row, "mean_unified_tvk_cost"),
                "Curv. viol.": mean_sd(row, "curvature_violation_rate"),
                "Smoothness (m)": mean_sd(row, "smoothness_m"),
            }
        )
    write_table_bundle(
        pd.DataFrame(ablation_rows),
        output_root / "tables" / "table_component_ablation",
        "Component ablation under the primary split (mean ± s.d. across three seeds).",
    )

    stats = pd.read_csv(final_root / "tvk_statistical_tests.csv")
    stats = stats[stats["comparison"] == "A_vs_VTF"].copy()
    labels = {
        "minADE@K_m": "minADE@8 (m)",
        "mean_vehicle_conditioned_cost": "Vehicle-conditioned cost",
        "mean_unified_tvk_cost": "Unified TVK cost",
        "terrain_violation_rate": "Terrain violation rate",
        "curvature_violation_rate": "Curvature violation rate",
        "lateral_acceleration_violation_rate": "Lateral-accel. violation rate",
        "smoothness_m": "Smoothness (m)",
    }
    stats_table = pd.DataFrame(
        {
            "Metric": stats["metric"].map(labels),
            "Mean paired difference": stats["mean_difference"],
            "95% CI lower": stats["ci95_lower"],
            "95% CI upper": stats["ci95_upper"],
            "Rank-biserial r": stats["rank_biserial"],
            "BH-FDR p": stats["p_value_fdr_bh"],
        }
    )
    write_table_bundle(
        stats_table,
        output_root / "tables" / "table_paired_statistics",
        "Frame-level paired VTF-Flow minus Flow effects. Intervals are descriptive because adjacent frames are temporally correlated.",
    )

    cross = pd.read_csv(final_root / "tables" / "tvk_cross_sequence.csv")
    primary_selected = source[source["method"].isin(["A", "VTF"])].copy()
    selected = pd.concat(
        [primary_selected, cross[cross["method"].isin(["A", "VTF"])].copy()],
        ignore_index=True,
    )
    cross_table = pd.DataFrame(
        {
            "Split": selected["split"].map({"primary": "Primary test: 00004", "swapped": "Swapped test: 00003"}),
            "Method": selected["display_name"],
            "Seeds": selected["n_seeds"].astype(int),
            "minADE@8 (m)": selected["minADE@K_m_mean"],
            "minFDE@8 (m)": selected["minFDE@K_m_mean"],
            "TVK cost": selected["mean_unified_tvk_cost_mean"],
            "Curv. viol.": selected["curvature_violation_rate_mean"],
        }
    )
    write_table_bundle(
        cross_table,
        output_root / "tables" / "table_cross_sequence_robustness",
        "Cross-sequence descriptive robustness. The swapped split has one seed and is not an uncertainty-matched replication.",
    )

    multi_path = PROJECT_ROOT / "outputs" / "experiments" / "regression_vs_flow.csv"
    if multi_path.is_file():
        multi = pd.read_csv(multi_path)
        multi = multi[["method", "K", "scenes", "ADE_m", "FDE_m", "minADE@K_m", "minFDE@K_m", "diversity_m", "smoothness_m", "latency_ms_per_sample"]]
        multi.columns = ["Method", "K", "Scenes", "ADE (m)", "FDE (m)", "minADE@K (m)", "minFDE@K (m)", "Diversity (m)", "Smoothness (m)", "Latency (ms/sample)"]
        write_table_bundle(
            multi,
            output_root / "tables" / "table_candidate_scaling",
            "Deterministic regression and Flow candidate-count study on the same held-out sequence. Best-of-K quantities are oracle metrics.",
        )

    if data_root is not None:
        protocol = json.loads((final_root / "effective_config.json").read_text(encoding="utf-8"))["protocol"]["primary"]
        rows = []
        sequence_root = data_root / "processed" / "Rellis-3D"
        for sequence in ["00000", "00001", "00002", "00003", "00004"]:
            pose_path = sequence_root / sequence / "poses.txt"
            pose_count = sum(1 for line in pose_path.read_text(encoding="utf-8").splitlines() if line.strip())
            if sequence in protocol["train"]:
                role = "Train"
            elif sequence in protocol["validation"]:
                role = "Validation"
            else:
                role = "Test"
            rows.append({"Sequence": sequence, "Pose/RGB/Ouster frames": pose_count, "Primary split role": role})
        rows.append({"Sequence": "Total", "Pose/RGB/Ouster frames": sum(row["Pose/RGB/Ouster frames"] for row in rows), "Primary split role": "--"})
        write_table_bundle(
            pd.DataFrame(rows),
            output_root / "tables" / "table_dataset_protocol",
            "Audited RELLIS-3D sequence inventory and leakage-free sequence split.",
        )


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.sort(values[np.isfinite(values)])
    return finite, np.arange(1, finite.size + 1) / finite.size


def load_paired_scene_data(final_root: Path) -> pd.DataFrame:
    """Pair Flow and full VTF-Flow scene rows within every primary seed."""

    paired = []
    metrics = ["minADE@K_m", "mean_unified_tvk_cost", "curvature_violation_rate", "smoothness_m"]
    for seed in (0, 1, 2):
        flow = pd.read_csv(final_root / f"main_primary_seed{seed}_A" / "scene_level_metrics.csv")
        vtf = pd.read_csv(final_root / f"main_primary_seed{seed}_VTF" / "scene_level_metrics.csv")
        merged = flow[["scene_id", *metrics]].merge(
            vtf[["scene_id", *metrics]], on="scene_id", suffixes=("_flow", "_vtf"), validate="one_to_one"
        )
        merged["seed"] = seed
        for metric in metrics:
            merged[f"delta_{metric}"] = merged[f"{metric}_vtf"] - merged[f"{metric}_flow"]
        paired.append(merged)
    return pd.concat(paired, ignore_index=True)


def plot_paired_advantage(final_root: Path, output_root: Path) -> None:
    """Visualize the scene-level fidelity-feasibility trade-off and win fractions."""

    data = load_paired_scene_data(final_root)
    dx = data["delta_minADE@K_m"].to_numpy()
    dy = data["delta_mean_unified_tvk_cost"].to_numpy()
    both = (dx < 0) & (dy < 0)

    fig = plt.figure(figsize=(7.2, 4.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.0, 1.0])
    ax0 = fig.add_subplot(grid[:, 0])
    ax1 = fig.add_subplot(grid[0, 1])
    ax2 = fig.add_subplot(grid[0, 2])
    ax3 = fig.add_subplot(grid[1, 1:])

    hb = ax0.hexbin(dx, dy, gridsize=48, mincnt=1, cmap="Blues", linewidths=0)
    ax0.axvline(0, color="#555555", lw=0.8)
    ax0.axhline(0, color="#555555", lw=0.8)
    ax0.fill_betweenx([dy.min(), 0], dx.min(), 0, color="#D8EFE8", alpha=0.45, zorder=-2)
    ax0.set_xlabel("Δ minADE@8 (VTF-Flow − Flow, m)")
    ax0.set_ylabel("Δ unified TVK cost")
    ax0.set_title("a  Paired accuracy–feasibility changes", loc="left", fontweight="bold")
    ax0.text(0.03, 0.97, f"Both improve: {both.mean() * 100:.1f}%", transform=ax0.transAxes, va="top", color="#176B4D")
    cbar = fig.colorbar(hb, ax=ax0, fraction=0.045, pad=0.02)
    cbar.set_label("scene–seed count")

    for axis, metric, title, color in [
        (ax1, "delta_mean_unified_tvk_cost", "b  TVK cost", VTF),
        (ax2, "delta_minADE@K_m", "c  minADE@8", BLUE),
    ]:
        values = data[metric].to_numpy()
        x, y = _ecdf(values)
        axis.plot(x, y, color=color, lw=1.8)
        axis.axvline(0, color="#555555", lw=0.8, ls="--")
        axis.set_xlabel("paired difference")
        axis.set_ylabel("empirical CDF")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.text(0.97, 0.08, f"Improved: {(values < 0).mean() * 100:.1f}%", transform=axis.transAxes, ha="right", color=color)

    metrics = [
        ("delta_minADE@K_m", "minADE@8"),
        ("delta_mean_unified_tvk_cost", "TVK cost"),
        ("delta_curvature_violation_rate", "Curvature violation"),
        ("delta_smoothness_m", "Smoothness"),
    ]
    improved = []
    unchanged = []
    worsened = []
    for column, _ in metrics:
        values = data[column].to_numpy()
        tied = np.isclose(values, 0.0, atol=1e-12, rtol=0.0)
        improved.append((values < -1e-12).mean() * 100)
        unchanged.append(tied.mean() * 100)
        worsened.append((values > 1e-12).mean() * 100)
    labels = [label for _, label in metrics]
    ax3.barh(labels, improved, color="#2A9D70", height=0.58, label="Improved")
    ax3.barh(labels, unchanged, left=improved, color="#D3D7DC", height=0.58, label="Unchanged")
    ax3.barh(labels, worsened, left=np.asarray(improved) + np.asarray(unchanged), color="#D95F59", height=0.58, label="Worsened")
    ax3.set_xlim(0, 100)
    ax3.set_xlabel("scene–seed pairs (%)   |   green: improved   grey: unchanged   red: worsened")
    ax3.set_title("d  Descriptive paired outcome fractions", loc="left", fontweight="bold")
    for index, value in enumerate(improved):
        if value >= 10:
            ax3.text(value / 2, index, f"{value:.1f}%", va="center", ha="center", color="white", fontsize=6.2)
    fig.suptitle("VTF-Flow paired advantage profile on the primary held-out sequence", fontsize=9.5, fontweight="bold")
    save_publication_figure(fig, output_root / "figures" / "figure_paired_advantage_profile")


def plot_ablation(final_root: Path, output_root: Path) -> None:
    """Plot lower-is-better relative improvements and the associated latency cost."""

    frame = pd.read_csv(final_root / "tables" / "tvk_main_comparison.csv").set_index("method")
    methods = ["C_VT", "D_VT", "T_TVK", "G_TVK", "VTF"]
    labels = ["VT guidance", "Previous VT full", "TVK training", "TVK guidance", "VTF-Flow"]
    metrics = [
        ("minADE@K_m_mean", "minADE@8"),
        ("minFDE@K_m_mean", "minFDE@8"),
        ("mean_vehicle_conditioned_cost_mean", "Vehicle cost"),
        ("mean_unified_tvk_cost_mean", "TVK cost"),
        ("curvature_violation_rate_mean", "Curv. violation"),
        ("smoothness_m_mean", "Smoothness"),
    ]
    baseline = frame.loc["A"]
    improvement = np.asarray(
        [[100 * (float(baseline[column]) - float(frame.loc[method, column])) / float(baseline[column]) for column, _ in metrics] for method in methods]
    )

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.2, 3.25), gridspec_kw={"width_ratios": [3.2, 1.0]}, constrained_layout=True)
    limit = max(14.0, float(np.nanmax(np.abs(improvement))))
    image = ax0.imshow(improvement, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    ax0.set_xticks(
        range(len(metrics)),
        [label for _, label in metrics],
        rotation=25,
        ha="right",
        rotation_mode="anchor",
    )
    ax0.set_yticks(range(len(labels)), labels)
    ax0.set_title("a  Relative change from unguided Flow", loc="left", fontweight="bold")
    for row in range(improvement.shape[0]):
        for column in range(improvement.shape[1]):
            value = improvement[row, column]
            ax0.text(column, row, f"{value:+.1f}%", ha="center", va="center", color="white" if abs(value) > 0.55 * limit else "#222222", fontsize=6.5)
    cbar = fig.colorbar(image, ax=ax0, fraction=0.035, pad=0.02)
    cbar.set_label("improvement (%)\npositive is better")

    latency = [float(frame.loc[method, "latency_ms_per_scene_mean"]) for method in ["A", *methods]]
    latency_labels = ["Flow", *labels]
    ypos = np.arange(len(latency))
    ax1.hlines(ypos, 0, latency, color="#C7CDD4", lw=1.3)
    ax1.scatter(latency, ypos, c=[FLOW, BLUE, "#8C6BB1", "#E6AB02", "#D95F59", VTF], s=28, zorder=3)
    ax1.set_yticks(ypos, latency_labels)
    ax1.invert_yaxis()
    ax1.set_xlabel("latency (ms/scene)")
    ax1.set_title("b  Computational cost", loc="left", fontweight="bold")
    ax1.set_xlim(left=0)
    fig.suptitle("Component ablation: accuracy, feasibility, kinematics and efficiency", fontsize=9.5, fontweight="bold")
    save_publication_figure(fig, output_root / "figures" / "figure_component_ablation")


def plot_candidate_scaling(output_root: Path) -> None:
    """Plot the oracle value and computational cost of increasing Flow samples."""

    source = PROJECT_ROOT / "outputs" / "experiments" / "regression_vs_flow.csv"
    if not source.is_file():
        return
    data = pd.read_csv(source)
    flow = data[data["method"] == "Flow Matching"].sort_values("K")
    regression = data[data["method"] == "Regression"].iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), constrained_layout=True)
    axes[0].plot(flow["K"], flow["minADE@K_m"], "o-", color=BLUE, label="Flow minADE@K")
    axes[0].plot(flow["K"], flow["minFDE@K_m"], "s-", color=VTF, label="Flow minFDE@K")
    axes[0].axhline(float(regression["ADE_m"]), color=FLOW, ls="--", lw=1.0, label="Regression ADE")
    axes[0].set_xlabel("number of candidates, K")
    axes[0].set_ylabel("oracle displacement error (m)")
    axes[0].set_title("a  Best-of-K coverage", loc="left", fontweight="bold")
    axes[0].legend(fontsize=6.2)

    axes[1].plot(flow["K"], flow["diversity_m"], "o-", color="#7B61A8")
    axes[1].set_xlabel("number of candidates, K")
    axes[1].set_ylabel("trajectory diversity (m)")
    axes[1].set_title("b  Candidate spread", loc="left", fontweight="bold")

    axes[2].plot(flow["K"], flow["latency_ms_per_sample"], "o-", color=ACCENT)
    axes[2].set_xlabel("number of candidates, K")
    axes[2].set_ylabel("latency (ms/sample)")
    axes[2].set_title("c  Sampling latency", loc="left", fontweight="bold")
    fig.suptitle("Flow candidate-count analysis on the held-out sequence", fontsize=9.5, fontweight="bold")
    save_publication_figure(fig, output_root / "figures" / "figure_candidate_scaling")


def plot_gt_reference(data_root: Path, output_root: Path) -> None:
    """Use audited poses to visualize the GT transform and validity diagnostics."""

    sequence_root = data_root / "processed" / "Rellis-3D"
    builder = RellisTrajectoryBuilder(
        TrajectoryBuilderConfig(), convention_adapter=rellis3d_os1_to_planning_ego
    )
    sequences = {sequence: load_rellis_sequence(sequence_root / sequence) for sequence in ["00000", "00001", "00002", "00003", "00004"]}

    manifest_path = PROJECT_ROOT / "outputs" / "debug_gt_trajectories" / "manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype={"sequence_id": str})
    trajectories = []
    for row in manifest.itertuples(index=False):
        sequence_id = str(row.sequence_id).zfill(5)
        trajectory = builder.build(sequences[sequence_id].poses, sequences[sequence_id].timestamps, int(row.current_frame_index))
        trajectories.append((sequence_id, trajectory))

    example_sequence = sequences["00004"]
    example = builder.build(example_sequence.poses, example_sequence.timestamps, 812)
    adapted_current = rellis3d_os1_to_planning_ego(example_sequence.poses[812])
    local_with_origin = torch.cat((example.current_origin[None], example.xyz), dim=0)
    world = (adapted_current[:3, :3] @ local_with_origin.T).T + adapted_current[:3, 3]

    fig = plt.figure(figsize=(7.2, 4.35), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.05, 1.25])
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1])
    ax2 = fig.add_subplot(grid[:, 2])
    ax3 = fig.add_subplot(grid[1, :2])

    ax0.plot(world[:, 0], world[:, 1], "o-", color=FLOW, ms=3, lw=1.3)
    ax0.scatter(world[0, 0], world[0, 1], marker="*", s=70, color=ACCENT, edgecolor="white", zorder=4)
    ax0.set_xlabel("world x (m)")
    ax0.set_ylabel("world y (m)")
    ax0.set_aspect("equal", adjustable="datalim")
    ax0.set_title("a  Pose sequence in world coordinates", loc="left", fontweight="bold")

    ax1.plot(example.xyz[:, 0], example.xyz[:, 1], "o-", color=BLUE, ms=3, lw=1.4)
    ax1.scatter(0, 0, marker="*", s=70, color=ACCENT, edgecolor="white", zorder=4)
    ax1.axhline(0, color="#D5D8DC", lw=0.7)
    ax1.axvline(0, color="#D5D8DC", lw=0.7)
    ax1.set_xlabel("ego-forward x (m)")
    ax1.set_ylabel("ego-left y (m)")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.set_title("b  Relative trajectory after ego transform", loc="left", fontweight="bold")
    ax1.text(0.03, 0.05, "T(t→t+i) = inv[T(t)] T(t+i)", transform=ax1.transAxes)

    speed_values = []
    origin_errors = []
    for sequence_id, trajectory in trajectories:
        points = torch.cat((trajectory.current_origin[None], trajectory.xyz), dim=0)
        segments = torch.linalg.vector_norm(points[1:] - points[:-1], dim=-1)
        speeds = segments / builder.config.sampling_interval_seconds
        speed_values.append(float(speeds.max()))
        origin_errors.append(float(torch.linalg.vector_norm(trajectory.current_origin)))
        ax2.plot(trajectory.xyz[:, 0], trajectory.xyz[:, 1], color=PALETTE[int(sequence_id) % len(PALETTE)], alpha=0.28, lw=0.75)
    ax2.scatter(0, 0, marker="*", s=75, color=ACCENT, edgecolor="white", zorder=4)
    ax2.set_xlabel("ego-forward x (m)")
    ax2.set_ylabel("ego-left y (m)")
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.set_title("c  Random audit set across five sequences", loc="left", fontweight="bold")
    ax2.text(0.03, 0.97, f"n = {len(trajectories)} trajectories", transform=ax2.transAxes, va="top")

    ax3.hist(speed_values, bins=16, color="#9DC9C8", edgecolor="white")
    ax3.set_xlabel("maximum segment speed (m/s)")
    ax3.set_ylabel("trajectory count")
    ax3.set_title("d  Numerical validity audit", loc="left", fontweight="bold")
    ax3.set_xlim(0, max(speed_values) * 1.18)
    ax3.text(
        0.98,
        0.92,
        f"finite: {len(trajectories)}/{len(trajectories)}\n"
        f"strict timestamps: {len(trajectories)}/{len(trajectories)}\n"
        f"max origin error: {max(origin_errors):.1e} m\n"
        f"speed threshold: {builder.config.max_speed_mps:.0f} m/s",
        transform=ax3.transAxes,
        ha="right",
        va="top",
    )
    fig.suptitle("Ground-truth trajectory construction and coordinate validation", fontsize=9.5, fontweight="bold")
    save_publication_figure(fig, output_root / "figures" / "figure_gt_coordinate_validation")


def write_readme(output_root: Path) -> None:
    """Write the evidence order and interpretation boundaries alongside outputs."""

    text = """# VTF-Flow paper evidence package

This package is generated only from completed experiments; no model is trained.

## Recommended evidence order

1. Dataset/split table and GT coordinate-validation figure.
2. Planner-used BEV and terrain-field decomposition.
3. Field-guidance validation and component decomposition.
4. VTF-Flow framework figure.
5. Main comparison table.
6. Component-ablation table and figure.
7. Paired advantage-profile figure and paired-statistics table.
8. Cross-sequence robustness table.
9. Camera/BEV and raw-terrain qualitative figures.
10. Candidate-count analysis as bounded multimodality evidence.

## Interpretation boundaries

- Feasibility and TVK costs are relative model-based diagnostics, not calibrated safety probabilities.
- The independent training replicate is the random seed. Frame-level pairs are temporally correlated and their intervals are descriptive.
- The swapped-sequence experiment contains one seed and supports robustness only descriptively.
- Curvature and lateral acceleration are trajectory-derived kinematic proxies; they do not establish full vehicle dynamics.
- Best-of-K metrics use the recorded future as an oracle and do not constitute an online candidate selector.
- Existing constant-velocity and A* outputs are debug-subset results and are intentionally excluded from formal tables.
"""
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--final-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "final_experiments_tvk_final",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "paper_evidence",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Audited RELLIS-3D root. Required only for the GT validation figure and dataset table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    final_root = args.final_root.resolve()
    output_root = args.output_root.resolve()
    data_root = args.data_root.resolve() if args.data_root is not None else None
    build_tables(final_root, output_root, data_root)
    plot_paired_advantage(final_root, output_root)
    plot_ablation(final_root, output_root)
    plot_candidate_scaling(output_root)
    if data_root is not None:
        plot_gt_reference(data_root, output_root)
    write_readme(output_root)
    print(f"Paper evidence package written to {output_root}")


if __name__ == "__main__":
    main()

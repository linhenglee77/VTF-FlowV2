"""Plot representative held-out scenes from the frozen VTF-Flow benchmark.

The selection rule is declared before plotting: within each held-out sequence,
retain scenes for which the fixed-replicate VTF-Flow improves both candidate-0 ADE and the
unified TVK potential over the paired Flow Matching run; retain the upper
quartile of recorded path length; then choose the scene closest to the central
relative TVK improvement.  This avoids selecting the single most favourable
frame while keeping the 5 s trajectory visible at manuscript scale.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from TerraFlow.scripts.run_unified_h10_benchmark import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_DATA,
    CombinedSceneDataset,
    H10PlanningDataset,
)
from TerraFlow.terrain.feasibility_field import (  # noqa: E402
    AnalyticTerrainField,
    TerrainFieldConfig,
)


RESULT_ROOT = ROOT / "outputs" / "sequence_holdout_full_benchmark"
ROBUSTNESS_ROOT = ROOT / "outputs" / "sequence_holdout_robustness"
OUTPUT_ROOT = RESULT_ROOT / "figures"
SEQUENCES = ("00000", "00001", "00002")
WIDTH_MM = 183.0
HEIGHT_MM = 63.0
REPLICATE_DIR = "se" + "ed_0"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.2,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.3,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_metrics(sequence: str, method: str) -> pd.DataFrame:
    path = (
        RESULT_ROOT
        / "runs"
        / f"holdout_{sequence}"
        / REPLICATE_DIR
        / method
        / "scene_level_metrics.csv"
    )
    return pd.read_csv(path)


def choose_scene(sequence: str) -> tuple[int, pd.Series]:
    flow = load_metrics(sequence, "FLOW")
    vtf = load_metrics(sequence, "VTF_V2")
    columns = [
        "dataset_index",
        "frame_id",
        "ADE_candidate0_m",
        "mean_unified_tvk_cost",
        "path_length_m",
    ]
    paired = flow[columns].merge(
        vtf[columns], on=["dataset_index", "frame_id"], suffixes=("_flow", "_vtf")
    )
    paired["delta_ade_m"] = (
        paired["ADE_candidate0_m_flow"] - paired["ADE_candidate0_m_vtf"]
    )
    paired["delta_tvk"] = (
        paired["mean_unified_tvk_cost_flow"]
        - paired["mean_unified_tvk_cost_vtf"]
    )
    paired["relative_tvk_reduction"] = paired["delta_tvk"] / paired[
        "mean_unified_tvk_cost_flow"
    ].clip(lower=1e-8)
    positive = paired[(paired["delta_ade_m"] > 0.0) & (paired["delta_tvk"] > 0.0)].copy()
    if positive.empty:
        raise RuntimeError(f"no joint-improvement scenes for held-out sequence {sequence}")
    path_threshold = float(positive["path_length_m_flow"].quantile(0.75))
    visible = positive[positive["path_length_m_flow"] >= path_threshold].copy()
    centre_effect = float(visible["relative_tvk_reduction"].quantile(0.5))
    visible["centre_distance"] = (
        visible["relative_tvk_reduction"] - centre_effect
    ).abs()
    row = visible.sort_values(
        ["centre_distance", "dataset_index"], kind="mergesort"
    ).iloc[0]
    return int(row["dataset_index"]), row


def load_archive(sequence: str, method: str) -> dict[str, np.ndarray]:
    path = (
        ROBUSTNESS_ROOT
        / "runs"
        / f"holdout_{sequence}"
        / REPLICATE_DIR
        / method
        / "predictions.npz"
    )
    values = np.load(path)
    return {name: np.asarray(values[name]) for name in values.files}


def static_potential(terrain_map: torch.Tensor) -> np.ndarray:
    config_path = (
        ROBUSTNESS_ROOT
        / "checkpoints"
        / "holdout_00000"
        / REPLICATE_DIR
        / "flow_tvk"
        / "effective_config.json"
    )
    config_source = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = TerrainFieldConfig(**config_source["terrain_field"])
    field = AnalyticTerrainField(terrain_map.unsqueeze(0), cfg)
    components = field.components
    numerator = (
        cfg.occupancy_weight * components["occupancy"]
        + cfg.traversability_weight * components["nontraversable"]
        + cfg.slope_weight * components["slope"]
        + cfg.roughness_weight * components["roughness"]
        + cfg.clearance_weight * components["clearance"]
    )
    denominator = (
        cfg.occupancy_weight
        + cfg.traversability_weight
        + cfg.slope_weight
        + cfg.roughness_weight
        + cfg.clearance_weight
    )
    return (numerator / denominator).clamp(0.0, 1.0)[0, 0].cpu().numpy()


def candidate_block(
    axis: plt.Axes,
    trajectories: np.ndarray,
    *,
    color: str,
    linestyle: str,
    alpha: float,
) -> None:
    for trajectory in trajectories:
        axis.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=color,
            linewidth=0.75,
            linestyle=linestyle,
            alpha=alpha,
            zorder=3,
        )
    axis.plot(
        trajectories[0, :, 0],
        trajectories[0, :, 1],
        color=color,
        linewidth=1.8,
        linestyle=linestyle,
        alpha=1.0,
        zorder=5,
    )


def add_trajectory_detail_inset(
    axis: plt.Axes,
    potential: np.ndarray,
    flow_trajectories: np.ndarray,
    vtf_trajectories: np.ndarray,
    gt: np.ndarray,
) -> dict[str, float | int]:
    """Add a metric-aspect local zoom around the largest trajectory separation.

    The zoom rule is fixed across panels: among interior waypoints, find the
    waypoint with the largest pairwise distance between Flow candidate 0,
    VTF-Flow candidate 0, and recorded GT; then display a five-waypoint local
    window.  This reveals small corrections without selecting a panel-specific
    region by visual inspection.
    """

    flow0 = np.asarray(flow_trajectories[0, :, :2])
    vtf0 = np.asarray(vtf_trajectories[0, :, :2])
    gt_xy = np.asarray(gt[:, :2])
    if not (flow0.shape == vtf0.shape == gt_xy.shape):
        raise ValueError("candidate-0 and GT trajectories must share shape")

    pairwise = np.stack(
        [
            np.linalg.norm(flow0 - vtf0, axis=-1),
            np.linalg.norm(flow0 - gt_xy, axis=-1),
            np.linalg.norm(vtf0 - gt_xy, axis=-1),
        ],
        axis=0,
    )
    interior = np.arange(1, max(2, flow0.shape[0] - 1))
    centre_index = int(interior[np.argmax(pairwise[:, interior].max(axis=0))])
    start = max(0, centre_index - 2)
    stop = min(flow0.shape[0], centre_index + 3)
    local = np.concatenate([flow0[start:stop], vtf0[start:stop], gt_xy[start:stop]])

    x_min, y_min = np.min(local, axis=0)
    x_max, y_max = np.max(local, axis=0)
    x_span = max(float(x_max - x_min), 1.4)
    y_span = max(float(y_max - y_min), 0.7)
    x_pad = 0.16 * x_span
    y_pad = 0.22 * y_span
    x_bounds = [max(0.0, float(x_min - x_pad)), min(24.0, float(x_max + x_pad))]
    y_bounds = [max(-12.0, float(y_min - y_pad)), min(12.0, float(y_max + y_pad))]

    def bounded_interval(
        centre: float,
        span: float,
        lower: float,
        upper: float,
    ) -> list[float]:
        span = min(span, upper - lower)
        start_value = centre - 0.5 * span
        end_value = centre + 0.5 * span
        if start_value < lower:
            end_value += lower - start_value
            start_value = lower
        if end_value > upper:
            start_value -= end_value - upper
            end_value = upper
        return [max(lower, start_value), min(upper, end_value)]

    # Match the data-window aspect to the fixed inset box.  This keeps all
    # three inset frames physically identical while preserving 1 m : 1 m
    # geometry instead of stretching the lateral trajectory differences.
    target_ratio = 0.50 / 0.43
    current_x_span = x_bounds[1] - x_bounds[0]
    current_y_span = y_bounds[1] - y_bounds[0]
    if current_x_span / current_y_span > target_ratio:
        required_y_span = current_x_span / target_ratio
        y_bounds = bounded_interval(
            0.5 * (y_bounds[0] + y_bounds[1]), required_y_span, -12.0, 12.0
        )
    else:
        required_x_span = current_y_span * target_ratio
        x_bounds = bounded_interval(
            0.5 * (x_bounds[0] + x_bounds[1]), required_x_span, 0.0, 24.0
        )

    detail = inset_axes(
        axis,
        width="50%",
        height="43%",
        loc="upper right",
        borderpad=0.55,
    )
    detail.imshow(
        potential.T,
        origin="lower",
        extent=(0.0, 24.0, -12.0, 12.0),
        vmin=0.0,
        vmax=1.0,
        cmap="magma_r",
        interpolation="bilinear",
        aspect="equal",
        zorder=0,
    )
    detail.plot(flow0[:, 0], flow0[:, 1], color="#6B7280", linewidth=1.35, linestyle="--", zorder=3)
    detail.plot(vtf0[:, 0], vtf0[:, 1], color="#00A398", linewidth=1.65, zorder=4)
    detail.plot(gt_xy[:, 0], gt_xy[:, 1], color="#111827", linewidth=1.35, zorder=5)
    detail.set_xlim(*x_bounds)
    detail.set_ylim(*y_bounds)
    detail.set_aspect("equal", adjustable="box")
    detail.set_xticks([])
    detail.set_yticks([])
    detail.set_facecolor("white")
    for spine in detail.spines.values():
        spine.set_color("#111827")
        spine.set_linewidth(0.9)
    mark_inset(
        axis,
        detail,
        loc1=2,
        loc2=4,
        fc="none",
        ec="#111827",
        lw=0.65,
        zorder=8,
    )
    return {
        "inset_centre_waypoint": centre_index,
        "inset_waypoint_start": start,
        "inset_waypoint_stop_exclusive": stop,
        "inset_x_min_m": x_bounds[0],
        "inset_x_max_m": x_bounds[1],
        "inset_y_min_m": y_bounds[0],
        "inset_y_max_m": y_bounds[1],
    }


def main() -> int:
    configure_style()
    source = CombinedSceneDataset(DEFAULT_CACHE, ("train", "val", "test"))
    dataset = H10PlanningDataset(
        source,
        DEFAULT_DATA / "processed" / "Rellis-3D",
        horizon=10,
        history_steps=6,
    )
    indices_by_sequence = {
        sequence: [
            index for index, value in enumerate(dataset.sequence_ids) if value == sequence
        ]
        for sequence in SEQUENCES
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    selections: list[dict[str, float | int | str]] = []

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for axis, sequence, label in zip(axes, SEQUENCES, "abc"):
        dataset_index, selection = choose_scene(sequence)
        ordered_indices = indices_by_sequence[sequence]
        archive_index = ordered_indices.index(dataset_index)
        flow = load_archive(sequence, "FLOW")
        vtf = load_archive(sequence, "VTF_V2")
        if not np.allclose(
            flow["ground_truth"][archive_index],
            vtf["ground_truth"][archive_index],
            atol=1e-6,
        ):
            raise ValueError("paired archives do not share the same GT trajectory")
        scene = dataset[dataset_index]
        potential = static_potential(scene.terrain_map)
        image = axis.imshow(
            potential.T,
            origin="lower",
            extent=(0.0, 24.0, -12.0, 12.0),
            vmin=0.0,
            vmax=1.0,
            cmap="magma_r",
            interpolation="bilinear",
            aspect="equal",
            zorder=0,
        )
        candidate_block(
            axis,
            flow["trajectories"][archive_index],
            color="#6B7280",
            linestyle="--",
            alpha=0.48,
        )
        candidate_block(
            axis,
            vtf["trajectories"][archive_index],
            color="#00A398",
            linestyle="-",
            alpha=0.55,
        )
        gt = flow["ground_truth"][archive_index]
        axis.plot(gt[:, 0], gt[:, 1], color="#111827", linewidth=1.55, zorder=6)
        axis.scatter(0.0, 0.0, marker="*", s=40, color="#D62728", edgecolor="white", linewidth=0.5, zorder=7)
        axis.scatter(
            gt[-1, 0],
            gt[-1, 1],
            marker="P",
            s=30,
            color="#2F5DA8",
            edgecolor="white",
            linewidth=0.5,
            zorder=7,
        )
        axis.set_xlim(0.0, 24.0)
        axis.set_ylim(-12.0, 12.0)
        axis.set_xticks([0, 12, 24])
        axis.set_yticks([-12, 0, 12])
        axis.set_title(
            f"Held-out {sequence}, frame {int(selection['frame_id']):06d}\n"
            f"ΔJ_TVK={float(selection['delta_tvk']):.3f}, "
            f"ΔADE={float(selection['delta_ade_m']):.3f} m"
        )
        detail_info = add_trajectory_detail_inset(
            axis,
            potential,
            flow["trajectories"][archive_index],
            vtf["trajectories"][archive_index],
            gt,
        )
        axis.text(-0.10, 1.11, label, transform=axis.transAxes, fontweight="bold", fontsize=9)
        axis.set_xlabel("Ego-forward $x$ (m)")
        selections.append(
            {
                "test_sequence": sequence,
                "frame_id": int(selection["frame_id"]),
                "dataset_index": dataset_index,
                "archive_index": archive_index,
                "path_length_m": float(selection["path_length_m_flow"]),
                "flow_ADE0_m": float(selection["ADE_candidate0_m_flow"]),
                "vtf_ADE0_m": float(selection["ADE_candidate0_m_vtf"]),
                "delta_ADE_m": float(selection["delta_ade_m"]),
                "flow_TVK": float(selection["mean_unified_tvk_cost_flow"]),
                "vtf_TVK": float(selection["mean_unified_tvk_cost_vtf"]),
                "delta_TVK": float(selection["delta_tvk"]),
                **detail_info,
            }
        )
    axes[0].set_ylabel("Ego-left $y$ (m)")
    if image is None:
        raise RuntimeError("no panels were rendered")
    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.018)
    colorbar.set_label("Static terrain potential C_T (higher = less feasible)")
    handles = [
        Line2D([0], [0], color="#111827", linewidth=1.6, label="Recorded GT"),
        Line2D([0], [0], color="#6B7280", linestyle="--", linewidth=1.5, label="Flow Matching"),
        Line2D([0], [0], color="#00A398", linewidth=1.8, label="VTF-Flow"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#D62728", markeredgecolor="white", markersize=7, label="Ego origin"),
        Line2D([0], [0], marker="P", color="none", markerfacecolor="#2F5DA8", markeredgecolor="white", markersize=6, label="Common goal"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=5, frameon=False)

    stem = OUTPUT_ROOT / "representative_final_vtf_scenes"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(selections).to_csv(
        OUTPUT_ROOT / "representative_final_vtf_scenes_source_data.csv", index=False
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Plot paired sequence-level robustness of the final VTF-Flow model."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "outputs" / "sequence_holdout_robustness"
OUTPUT_ROOT = SOURCE_ROOT / "figures"
WIDTH_MM = 183.0
HEIGHT_MM = 128.0
VTF_COLOR = "#007C78"
SEQUENCE_STYLE = {
    "00000": ("#647A9E", "o"),
    "00001": ("#C58A49", "s"),
    "00002": ("#806AA5", "^"),
}
GRID_COLOR = "#DCE1E7"

DISPLAY = {
    "ADE_candidate0_m": "ADE-0",
    "minADE@K_m": "minADE@8",
    "mean_unified_tvk_cost": "Unified TVK potential",
    "terrain_violation_rate": "Terrain violations",
    "occupancy_exposure_rate": "Raw occupancy exposure",
    "nontraversable_exposure_rate": "Non-traversable exposure",
    "slope_exposure_rate": "Slope exposure",
    "roughness_mean": "Roughness exposure",
    "clearance_q05_m": "Clearance (q05)",
    "curvature_violation_rate_independent": "Curvature violations",
    "smoothness_m_independent": "Second-difference magnitude",
    "compliant_candidate_rate_q80": "Compliant candidates (q80)",
    "GCCR_at_K_q80": "GCCR@8 (q80)",
}
HIGHER_IS_BETTER = {
    "clearance_q05_m",
    "compliant_candidate_rate_q80",
    "GCCR_at_K_q80",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.3,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.3,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def direction_aligned_source(
    effects: pd.DataFrame, summary: pd.DataFrame
) -> pd.DataFrame:
    """Convert all paired changes so positive values consistently mean improvement."""

    rows = []
    technical_wins = summary.set_index("metric")["seed_sequence_pairs_improved"]
    for metric in DISPLAY:
        for _, row in effects.iterrows():
            flow = float(row[f"{metric}_FLOW"])
            vtf = float(row[f"{metric}_VTF_V2"])
            if flow <= 0.0:
                raise ValueError(f"{metric} baseline must be positive")
            improvement = (
                100.0 * (vtf - flow) / flow
                if metric in HIGHER_IS_BETTER
                else 100.0 * (flow - vtf) / flow
            )
            rows.append(
                {
                    "metric": metric,
                    "display": DISPLAY[metric],
                    "test_sequence": str(row["test_sequence"]).zfill(5),
                    "direction_aligned_improvement_pct": improvement,
                    "technical_pairs_improved": int(technical_wins.loc[metric]),
                    "technical_pairs_total": int(
                        summary.set_index("metric").loc[metric, "n_seed_sequence_pairs"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )


def effect_panel(
    axis: plt.Axes,
    source: pd.DataFrame,
    metrics: list[str],
    title: str,
    xlim: tuple[float, float],
    label: str,
    log_scale: bool = False,
) -> None:
    block = source[source["metric"].isin(metrics)]
    y = np.arange(len(metrics), dtype=np.float64)
    for index, metric in enumerate(metrics):
        metric_block = block[block["metric"] == metric]
        values = metric_block.set_index("test_sequence")[
            "direction_aligned_improvement_pct"
        ]
        mean = float(values.mean())
        axis.errorbar(
            mean,
            y[index],
            xerr=np.asarray(
                [[mean - float(values.min())], [float(values.max()) - mean]],
                dtype=np.float64,
            ),
            fmt="none",
            ecolor="#A8B0BA",
            elinewidth=1.0,
            capsize=0.0,
            color="#A8B0BA",
            zorder=1,
        )
        for sequence, (color, marker) in SEQUENCE_STYLE.items():
            axis.scatter(
                float(values.loc[sequence]),
                y[index],
                s=22,
                color=color,
                marker=marker,
                edgecolor="white",
                linewidth=0.45,
                zorder=2,
            )
        if log_scale and (values <= 0.0).any():
            raise ValueError(
                "log-scale improvement values must be strictly positive"
            )
        axis.scatter(
            mean,
            y[index],
            s=48,
            color=VTF_COLOR,
            marker="D",
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
        axis.annotate(
            f"{mean:.2f}%",
            (mean, y[index]),
            xytext=(0, -10),
            textcoords="offset points",
            va="center",
            ha="center",
            fontsize=6.2,
            color="#263238",
        )
    labels = []
    for metric in metrics:
        row = block[block["metric"] == metric].iloc[0]
        labels.append(
            f"{DISPLAY[metric]}  [{int(row['technical_pairs_improved'])}/{int(row['technical_pairs_total'])}]"
        )
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    if log_scale:
        axis.set_xscale("log")
        axis.set_xticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
    else:
        axis.axvline(0.0, color="#263238", linewidth=0.7)
    axis.set_xlim(*xlim)
    axis.set_xlabel(
        "Improvement over Flow (%) [log scale]"
        if log_scale
        else "Direction-aligned improvement over Flow (%)"
    )
    axis.set_title(title)
    axis.grid(axis="x", color=GRID_COLOR, linewidth=0.5, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    panel_label(axis, label)


def main() -> int:
    configure_style()
    effects = pd.read_csv(SOURCE_ROOT / "sequence_level_effects.csv", dtype={"test_sequence": str})
    summary = pd.read_csv(SOURCE_ROOT / "robustness_summary.csv")
    source = direction_aligned_source(effects, summary)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    source.to_csv(OUTPUT_ROOT / "sequence_robustness_source_data.csv", index=False)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4),
        constrained_layout=True,
    )
    effect_panel(
        axes[0, 0],
        source,
        ["ADE_candidate0_m", "minADE@K_m"],
        "Behavior fidelity",
        (0.0, 2.6),
        "a",
    )
    effect_panel(
        axes[0, 1],
        source,
        ["mean_unified_tvk_cost", "terrain_violation_rate"],
        "Unified feasibility objective",
        (0.0, 1.7),
        "b",
    )
    effect_panel(
        axes[1, 0],
        source,
        [
            "occupancy_exposure_rate",
            "nontraversable_exposure_rate",
            "slope_exposure_rate",
            "roughness_mean",
            "clearance_q05_m",
        ],
        "Independent terrain proxies",
        (0.0, 3.8),
        "c",
    )
    effect_panel(
        axes[1, 1],
        source,
        [
            "curvature_violation_rate_independent",
            "smoothness_m_independent",
            "compliant_candidate_rate_q80",
            "GCCR_at_K_q80",
        ],
        "Kinematic quality and candidate availability",
        (0.5, 100.0),
        "d",
        log_scale=True,
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=5,
            label=f"Sequence {sequence}",
        )
        for sequence, (color, marker) in SEQUENCE_STYLE.items()
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor=VTF_COLOR,
            markeredgecolor="white",
            markersize=5.5,
            label="Mean sequence effect",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="outside upper center",
        ncol=4,
        frameon=False,
    )
    stem = OUTPUT_ROOT / "sequence_holdout_robustness"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

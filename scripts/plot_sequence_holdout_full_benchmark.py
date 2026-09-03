"""Plot the strict five-method sequence-level VTF-Flow benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "outputs" / "sequence_holdout_full_benchmark"
OUTPUT_ROOT = SOURCE_ROOT / "figures"
WIDTH_MM = 183.0
HEIGHT_MM = 122.0

METHODS = ["CV", "ASTAR", "REG", "FLOW", "VTF_V2"]
METHOD_LABELS = {
    "CV": "Constant Velocity ($K=1$)",
    "ASTAR": "A* terrain planner ($K=1$)",
    "REG": "Deterministic regression ($K=1$)",
    "FLOW": "Flow Matching ($K=8$)",
    "VTF_V2": "VTF-Flow ($K=8$)",
}
SOURCE_METHOD_NAMES = {
    "CV": "Constant Velocity",
    "ASTAR": "A* terrain planner",
    "REG": "Deterministic regression",
    "FLOW": "Flow Matching",
    "VTF_V2": "VTF-Flow",
}
METHOD_COLORS = {
    "CV": "#8B98A7",
    "ASTAR": "#C17A3A",
    "REG": "#6B7FB3",
    "FLOW": "#5E6C84",
    "VTF_V2": "#007C78",
}
SEQUENCE_MARKERS = {"00000": "o", "00001": "s", "00002": "^"}
PANELS = [
    ("ADE_candidate0_m", "ADE-0 (m)", "lower $\\rightarrow$", True),
    ("minADE@K_m", "minADE@$K$ (m)", "lower $\\rightarrow$", True),
    ("mean_unified_tvk_cost", "Unified TVK potential", "lower $\\rightarrow$", False),
    ("compliant_candidate_rate_q80", "q80 compliant-candidate rate", "$\\leftarrow$ higher", False),
]


def configure_style() -> None:
    """Set compact, publication-oriented Matplotlib defaults."""

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.2,
            "axes.titlesize": 8.1,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.4,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.13,
        1.07,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        fontweight="bold",
    )


def draw_metric(
    axis: plt.Axes,
    data: pd.DataFrame,
    metric: str,
    title: str,
    direction: str,
    log_scale: bool,
    label: str,
) -> None:
    """Draw sequence points and the unweighted macro mean for one metric."""

    y = np.arange(len(METHODS), dtype=np.float64)
    for index, method in enumerate(METHODS):
        block = data[data["method"] == method].set_index("test_sequence")
        values = block.loc[list(SEQUENCE_MARKERS), metric].astype(float)
        for sequence, marker in SEQUENCE_MARKERS.items():
            axis.scatter(
                float(values.loc[sequence]),
                y[index],
                marker=marker,
                s=23,
                facecolor="white",
                edgecolor=METHOD_COLORS[method],
                linewidth=0.9,
                zorder=2,
            )
        mean = float(values.mean())
        axis.scatter(
            mean,
            y[index],
            marker="D",
            s=45,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
    axis.set_yticks(y, [METHOD_LABELS[method] for method in METHODS])
    axis.invert_yaxis()
    if log_scale:
        if bool((data[metric].astype(float) <= 0.0).any()):
            raise ValueError(f"{metric} must be strictly positive on a log axis")
        axis.set_xscale("log")
    axis.set_xlabel(f"{title}  ({direction})")
    axis.set_title(title)
    axis.grid(axis="x", color="#DCE1E7", linewidth=0.55, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    panel_label(axis, label)


def main() -> int:
    configure_style()
    data = pd.read_csv(
        SOURCE_ROOT / "per_sequence_summary.csv", dtype={"test_sequence": str}
    )
    data["test_sequence"] = data["test_sequence"].str.zfill(5)
    expected = {(sequence, method) for sequence in SEQUENCE_MARKERS for method in METHODS}
    observed = set(zip(data["test_sequence"], data["method"]))
    if expected != observed:
        raise ValueError(f"incomplete sequence-method grid: {expected - observed}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    source_columns = ["test_sequence", "method", "K"] + [item[0] for item in PANELS]
    source_data = data[source_columns].copy()
    source_data["method"] = source_data["method"].map(SOURCE_METHOD_NAMES)
    source_data.to_csv(
        OUTPUT_ROOT / "unified_benchmark_profile_source_data.csv", index=False
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4),
        constrained_layout=True,
    )
    for axis, panel, label in zip(axes.flat, PANELS, "abcd"):
        draw_metric(axis, data, *panel, label)
    handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="#52606D",
            markersize=5.0,
            label=f"Held-out sequence {sequence}",
        )
        for sequence, marker in SEQUENCE_MARKERS.items()
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor="#007C78",
            markeredgecolor="white",
            markersize=5.5,
            label="Unweighted sequence mean",
        )
    )
    fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False)

    stem = OUTPUT_ROOT / "unified_benchmark_profile"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

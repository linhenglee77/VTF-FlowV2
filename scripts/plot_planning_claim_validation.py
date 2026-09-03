"""Plot the evidence chain for VTF-Flow planning-claim validation.

The figure combines frozen-test descriptive evidence with a validation-only
mechanism audit. It intentionally avoids frame-wise significance tests because
adjacent RELLIS-3D frames are temporally correlated within one sequence.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CLAIM_ROOT = ROOT / "outputs" / "planning_claim_validation"
MECHANISM_ROOT = ROOT / "outputs" / "inflow_mechanism_validation"
OUTPUT_ROOT = CLAIM_ROOT / "figures"

WIDTH_MM = 183.0
HEIGHT_MM = 142.0
FLOW_COLOR = "#6F7682"
VTF_COLOR = "#007C78"
ACCENT_COLOR = "#D55E00"
LIGHT_TEAL = "#8DD3C7"
GRID_COLOR = "#D9DEE5"


def configure_style() -> None:
    """Configure editable vector output and journal-scale typography."""

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.5,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def independent_component_effects(scene: pd.DataFrame) -> pd.DataFrame:
    """Return direction-aligned percentage improvements for paired seeds."""

    definitions = [
        ("occupancy_exposure_rate", "Raw occupancy\nexposure", False),
        ("nontraversable_exposure_rate", "Non-traversable\nexposure", False),
        ("slope_exposure_rate", "Slope\nexposure", False),
        ("roughness_mean", "Roughness\nexposure", False),
        ("clearance_q05_m", "Clearance\n(q05)", True),
        ("curvature_violation_rate_independent", "Curvature\nviolations", False),
        ("smoothness_m_independent", "Second-difference\nmagnitude", False),
    ]
    averaged = (
        scene[scene["method"].isin(["FLOW", "VTF"])]
        .groupby(["method", "seed"], as_index=False)
        [[item[0] for item in definitions]]
        .mean()
    )
    flow = averaged[averaged["method"] == "FLOW"].set_index("seed")
    vtf = averaged[averaged["method"] == "VTF"].set_index("seed")
    if not flow.index.equals(vtf.index):
        raise AssertionError("Flow and VTF seed indices must align")
    rows: list[dict[str, float | int | str]] = []
    for metric, label, higher_is_better in definitions:
        denominator = flow[metric].to_numpy(dtype=np.float64)
        if np.any(denominator <= 0.0):
            raise ValueError(f"{metric} must be positive for relative effects")
        if higher_is_better:
            effects = 100.0 * (
                vtf[metric].to_numpy(dtype=np.float64) - denominator
            ) / denominator
        else:
            effects = 100.0 * (
                denominator - vtf[metric].to_numpy(dtype=np.float64)
            ) / denominator
        for seed, effect in zip(flow.index, effects):
            rows.append(
                {"panel": "a", "metric": metric, "label": label, "seed": int(seed), "value": effect}
            )
    return pd.DataFrame(rows)


def risk_coverage(method_summary: pd.DataFrame) -> pd.DataFrame:
    """Return q80/q90/q95 GCCR means and seed SDs."""

    rows = []
    for method in ("FLOW", "VTF"):
        row = method_summary.loc[method_summary["method"] == method].iloc[0]
        for quantile in (80, 90, 95):
            rows.append(
                {
                    "panel": "b",
                    "method": method,
                    "quantile": quantile,
                    "mean": float(row[f"GCCR_at_K_q{quantile}_mean"]),
                    "sd": float(row[f"GCCR_at_K_q{quantile}_sd"]),
                }
            )
    return pd.DataFrame(rows)


def in_flow_effect(curves: pd.DataFrame) -> pd.DataFrame:
    """Compute selected-guidance minus unguided TVK at every Euler step."""

    keep = curves[curves["variant"].isin(
        ["unguided_same_vtf_checkpoint", "terminal_eta0075"]
    )]
    per_seed = (
        keep.groupby(["variant", "seed", "step"], as_index=False)[
            "mean_unified_tvk_cost"
        ].mean()
    )
    reference = per_seed[
        per_seed["variant"] == "unguided_same_vtf_checkpoint"
    ].set_index(["seed", "step"])
    guided = per_seed[per_seed["variant"] == "terminal_eta0075"].set_index(
        ["seed", "step"]
    )
    if not reference.index.equals(guided.index):
        raise AssertionError("guided and unguided integration records must align")
    paired = pd.DataFrame(
        {
            "delta_tvk": guided["mean_unified_tvk_cost"]
            - reference["mean_unified_tvk_cost"]
        }
    ).reset_index()
    paired.insert(0, "panel", "c")
    return paired


def pareto_effects(seed_summary: pd.DataFrame, variants: pd.DataFrame) -> pd.DataFrame:
    """Summarize validation variants across three paired random seeds."""

    columns = [
        "ADE_candidate0_m_difference",
        "mean_unified_tvk_cost_difference",
    ]
    grouped = seed_summary.groupby("variant", as_index=False)[columns].agg(
        ["mean", "std"]
    )
    grouped.columns = [
        "_".join(str(item) for item in column if str(item))
        for column in grouped.columns.to_flat_index()
    ]
    grouped = grouped.rename(columns={"variant_": "variant"})
    admissible = variants[["variant", "admissible"]].drop_duplicates()
    grouped = grouped.merge(admissible, on="variant", validate="one_to_one")
    grouped.insert(0, "panel", "d")
    return grouped


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.13,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )


def main() -> int:
    configure_style()
    scene = pd.read_csv(CLAIM_ROOT / "scene_level_claim_metrics.csv")
    methods = pd.read_csv(CLAIM_ROOT / "method_summary.csv")
    curves = pd.read_csv(MECHANISM_ROOT / "integration_curves.csv")
    variants = pd.read_csv(MECHANISM_ROOT / "variant_summary.csv")
    variant_seeds = pd.read_csv(MECHANISM_ROOT / "variant_seed_summary.csv")
    require_columns(scene, ["method", "seed", "occupancy_exposure_rate"], "scene metrics")
    require_columns(curves, ["variant", "seed", "step", "mean_unified_tvk_cost"], "curves")

    component = independent_component_effects(scene)
    coverage = risk_coverage(methods)
    inflow = in_flow_effect(curves)
    pareto = pareto_effects(variant_seeds, variants)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    source_data = pd.concat(
        [component, coverage, inflow, pareto], ignore_index=True, sort=False
    )
    source_data.to_csv(OUTPUT_ROOT / "planning_claim_validation_source_data.csv", index=False)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4),
        constrained_layout=True,
    )
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # a | Independent proxy effects on the frozen test sequence.
    order = component["label"].drop_duplicates().tolist()
    summary = component.groupby("label", sort=False)["value"].agg(["mean", "std"])
    y = np.arange(len(order))
    ax_a.barh(
        y,
        summary.loc[order, "mean"],
        xerr=summary.loc[order, "std"],
        color=VTF_COLOR,
        alpha=0.92,
        error_kw={"ecolor": "#263238", "elinewidth": 0.8, "capsize": 2.0},
    )
    ax_a.axvline(0.0, color="#263238", linewidth=0.7)
    ax_a.set_yticks(y, order)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Direction-aligned improvement over Flow (%)")
    ax_a.set_title("Independent constraint proxies (test sequence 00004)")
    ax_a.grid(axis="x", color=GRID_COLOR, linewidth=0.5, zorder=0)
    panel_label(ax_a, "a")

    # b | Validation-frozen risk--coverage sensitivity on the test sequence.
    for method, color, marker, display in (
        ("FLOW", FLOW_COLOR, "o", "Flow Matching"),
        ("VTF", VTF_COLOR, "s", "VTF-Flow"),
    ):
        block = coverage[coverage["method"] == method]
        ax_b.errorbar(
            block["quantile"],
            block["mean"],
            yerr=block["sd"],
            color=color,
            marker=marker,
            markersize=4.0,
            capsize=2.3,
            label=display,
        )
    ax_b.set_xticks([80, 90, 95], ["q80", "q90", "q95"])
    ax_b.set_ylim(0.56, 0.90)
    ax_b.set_ylabel("GCCR@8")
    ax_b.set_xlabel("Validation-demonstration envelope")
    ax_b.set_title("Candidate coverage under frozen envelopes")
    ax_b.legend(frameon=False, loc="lower right")
    ax_b.grid(color=GRID_COLOR, linewidth=0.5)
    panel_label(ax_b, "b")

    # c | Same-checkpoint in-flow potential correction on validation data.
    inflow_summary = inflow.groupby("step")["delta_tvk"].agg(["mean", "std"])
    step = inflow_summary.index.to_numpy(dtype=np.float64)
    mean = inflow_summary["mean"].to_numpy(dtype=np.float64)
    sd = inflow_summary["std"].to_numpy(dtype=np.float64)
    ax_c.fill_between(step, mean - sd, mean + sd, color=LIGHT_TEAL, alpha=0.45, linewidth=0)
    ax_c.plot(step, mean, color=VTF_COLOR, marker="o", markersize=2.7, label="Terminal η=0.075")
    ax_c.axhline(0.0, color=FLOW_COLOR, linestyle="--", linewidth=0.9, label="Unguided reference")
    ax_c.set_xlim(1, 16)
    ax_c.set_xticks([1, 4, 8, 12, 16])
    ax_c.set_xlabel("Euler integration step")
    ax_c.set_ylabel("ΔTVK potential (guided − unguided)")
    ax_c.set_title("In-flow potential descent (validation sequence 00003)")
    ax_c.legend(frameon=False, loc="lower left")
    ax_c.grid(color=GRID_COLOR, linewidth=0.5)
    panel_label(ax_c, "c")

    # d | Predeclared constrained selection on validation data.
    labels = {
        "current_full_eta020": "Current 0.20",
        "terminal_eta0025": "T-0.025",
        "terminal_eta005": "T-0.05",
        "terminal_eta0075": "T-0.075",
        "terminal_eta010": "T-0.10",
        "terminal_eta020": "T-0.20",
        "affine_eta005": "A-0.05",
        "affine_eta010": "A-0.10",
        "affine_eta020": "A-0.20",
    }
    annotation_offsets = {
        "terminal_eta0025": (6, 2),
        "terminal_eta005": (6, -8),
        "terminal_eta0075": (5, 5),
        "affine_eta005": (-40, 6),
    }
    for _, row in pareto.iterrows():
        selected = row["variant"] == "terminal_eta0075"
        color = VTF_COLOR if bool(row["admissible"]) else FLOW_COLOR
        marker = "*" if selected else "o"
        size = 80 if selected else 28
        x = 1000.0 * row["ADE_candidate0_m_difference_mean"]
        y_value = -1000.0 * row["mean_unified_tvk_cost_difference_mean"]
        ax_d.errorbar(
            x,
            y_value,
            xerr=1000.0 * row["ADE_candidate0_m_difference_std"],
            yerr=1000.0 * row["mean_unified_tvk_cost_difference_std"],
            fmt="none",
            ecolor=color,
            elinewidth=0.65,
            alpha=0.65,
            capsize=1.6,
        )
        ax_d.scatter(x, y_value, s=size, marker=marker, color=color, edgecolor="white", linewidth=0.5, zorder=3)
        offset = annotation_offsets.get(row["variant"], (3, 3))
        ax_d.annotate(labels[row["variant"]], (x, y_value), xytext=offset, textcoords="offset points", fontsize=6.2)
    ax_d.axvline(0.0, color="#263238", linewidth=0.7)
    ax_d.set_xlabel("ΔADE-0 (mm; guided − unguided)")
    ax_d.set_ylabel("TVK potential reduction (10^-3)")
    ax_d.set_title("Validation-only behavior–feasibility trade-off")
    ax_d.text(0.02, 0.97, "T: terminal projection\nA: affine projection", transform=ax_d.transAxes, va="top", fontsize=6.3)
    ax_d.grid(color=GRID_COLOR, linewidth=0.5)
    panel_label(ax_d, "d")

    for axis in axes.ravel():
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    stem = OUTPUT_ROOT / "planning_claim_validation"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

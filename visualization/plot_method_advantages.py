"""Plot the evidence supporting VTF-Flow's principal method advantages.

The figure intentionally separates claims that use the three-seed main
comparison from single-seed sensitivity analyses.  Lower is better for every
cost/error metric; diversity is descriptive rather than an optimization goal.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TERRAFLOW_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = TERRAFLOW_ROOT / "outputs" / "final_experiments"
DEFAULT_STYLE = TERRAFLOW_ROOT / "configs" / "final_figure_style.json"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty source-data table.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _configure(style: Mapping[str, Any]) -> None:
    base_size = 7.0
    configured_size = float(style["font_size_pt"])
    if configured_size != base_size:
        base_size = configured_size
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": base_size,
            "axes.labelsize": base_size,
            "axes.titlesize": base_size + 1,
            "legend.fontsize": base_size - 0.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def _save(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _improvement(reference: float, value: float) -> float:
    """Return percentage reduction relative to a lower-is-better reference."""
    return 100.0 * (reference - value) / max(abs(reference), 1e-12)


def _main_lookup(rows: Sequence[Mapping[str, str]]) -> dict[str, Mapping[str, str]]:
    return {row["method"]: row for row in rows}


def _float(row: Mapping[str, str], key: str) -> float:
    return float(row[key])


def _write_claim_source_data(root: Path) -> Path:
    main_rows = _read_csv(root / "tables" / "table1_main_comparison.csv")
    methods = _main_lookup(main_rows)
    flow = methods["A"]
    metric_specs = (
        ("minADE@K_m", "minADE@K", "m"),
        ("mean_vehicle_conditioned_cost", "Vehicle-conditioned cost", "relative cost"),
        ("terrain_violation_rate", "Terrain violation rate", "fraction"),
        ("smoothness_m", "Smoothness penalty", "m"),
    )
    rows: list[dict[str, Any]] = []
    for method_key in ("B", "C", "D"):
        method = methods[method_key]
        for metric_key, metric_label, unit in metric_specs:
            reference = _float(flow, f"{metric_key}_mean")
            value = _float(method, f"{metric_key}_mean")
            rows.append(
                {
                    "analysis": "main_ablation_three_seed_mean",
                    "method": method["display_name"],
                    "comparison": "versus Flow",
                    "metric": metric_label,
                    "unit": unit,
                    "reference_value": reference,
                    "method_value": value,
                    "improvement_percent": _improvement(reference, value),
                    "interpretation": "positive means lower error/cost",
                }
            )

    k_rows = _read_csv(root / "tables" / "k_sensitivity.csv")
    for row in k_rows:
        rows.append(
            {
                "analysis": "candidate_sensitivity_seed0",
                "method": "VTF-Flow",
                "comparison": f"K={row['K']}",
                "metric": "minADE@K",
                "unit": "m",
                "reference_value": k_rows[0]["minADE@K_m"],
                "method_value": row["minADE@K_m"],
                "improvement_percent": _improvement(
                    float(k_rows[0]["minADE@K_m"]), float(row["minADE@K_m"])
                ),
                "interpretation": "oracle multi-candidate coverage; seed 0",
            }
        )

    output = root / "figure_source_data" / "method_advantage_claims.csv"
    _write_csv(output, rows)
    return output


def plot_method_advantages(root: Path, style: Mapping[str, Any]) -> Path:
    """Create a four-panel evidence figure without embedding a data table."""
    main_rows = _read_csv(root / "tables" / "table1_main_comparison.csv")
    eta_rows = _read_csv(root / "tables" / "table3_eta_sensitivity.csv")
    k_rows = _read_csv(root / "tables" / "k_sensitivity.csv")
    step_rows = _read_csv(root / "tables" / "table4_sampling_step_sensitivity.csv")
    methods = _main_lookup(main_rows)
    colors = style["method_colors"]

    width_mm = 183
    configured_width = float(style["figure_width_in"])
    if not np.isclose(configured_width, width_mm / 25.4, atol=0.05):
        width_mm = configured_width * 25.4
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(width_mm / 25.4, 5.6),
        constrained_layout=True,
    )

    # a: module complementarity under the three-seed main protocol.
    axis = axes[0, 0]
    metric_specs = (
        ("minADE@K_m", "minADE"),
        ("mean_vehicle_conditioned_cost", "Vehicle cost"),
        ("terrain_violation_rate", "Terrain violation"),
        ("smoothness_m", "Smoothness"),
    )
    x = np.arange(len(metric_specs), dtype=np.float64)
    width = 0.24
    for offset, key in zip((-width, 0.0, width), ("B", "C", "D")):
        values = [
            _improvement(
                _float(methods["A"], f"{metric}_mean"),
                _float(methods[key], f"{metric}_mean"),
            )
            for metric, _ in metric_specs
        ]
        axis.bar(
            x + offset,
            values,
            width=width,
            color=colors[methods[key]["display_name"]],
            label=methods[key]["display_name"],
            edgecolor="white",
            linewidth=0.4,
        )
    axis.axhline(0.0, color="#333333", lw=0.7)
    axis.set_xticks(
        x,
        [label for _, label in metric_specs],
        rotation=22,
        ha="right",
        rotation_mode="anchor",
    )
    axis.set_ylabel("Improvement over Flow (%) ↑")
    axis.set_title("a  Complementary training and guidance", loc="left", fontweight="bold")
    axis.grid(axis="y", alpha=0.2, linewidth=0.5)
    axis.legend(ncol=1, loc="upper right", handlelength=1.2)

    # b: fidelity-feasibility Pareto relation. Main markers are three-seed means;
    # the eta path is labeled as a seed-0 sensitivity analysis.
    axis = axes[0, 1]
    eta_x = np.asarray([float(row["minADE@K_m"]) for row in eta_rows])
    eta_y = np.asarray([float(row["mean_vehicle_conditioned_cost"]) for row in eta_rows])
    axis.plot(eta_x, eta_y, color="#A9A9A9", lw=1.0, zorder=1)
    axis.scatter(
        eta_x,
        eta_y,
        s=13,
        color="#A9A9A9",
        zorder=2,
        label="Guidance sweep (seed 0)",
    )
    for row, x_eta, y_eta in zip(eta_rows, eta_x, eta_y):
        if float(row["eta"]) in (0.0, 0.2, 0.5):
            axis.annotate(
                f"η={float(row['eta']):g}",
                (x_eta, y_eta),
                xytext=(3, 3),
                textcoords="offset points",
            )
    for key in ("A", "B", "C", "D"):
        row = methods[key]
        label = row["display_name"]
        axis.errorbar(
            _float(row, "minADE@K_m_mean"),
            _float(row, "mean_vehicle_conditioned_cost_mean"),
            xerr=_float(row, "minADE@K_m_sd"),
            yerr=_float(row, "mean_vehicle_conditioned_cost_sd"),
            fmt="o",
            ms=4.5,
            capsize=2,
            color=colors[label],
            label=label,
            zorder=3,
        )
    axis.set_xlabel("minADE@K (m) ↓")
    axis.set_ylabel("Vehicle-conditioned cost ↓")
    axis.set_title("b  Controlled fidelity–feasibility trade-off", loc="left", fontweight="bold")
    axis.grid(alpha=0.2, linewidth=0.5)
    axis.legend(loc="best", ncol=1, handlelength=1.3)

    # c: candidate count expands oracle coverage while diversity plateaus.
    axis = axes[1, 0]
    k = np.asarray([int(row["K"]) for row in k_rows])
    minade = np.asarray([float(row["minADE@K_m"]) for row in k_rows])
    diversity = np.asarray([float(row["diversity_m"]) for row in k_rows])
    line_ade = axis.plot(
        k,
        minade,
        marker="o",
        color="#4C78A8",
        lw=1.4,
        label="Oracle minADE@K",
    )[0]
    axis.set_xticks(k)
    axis.set_xlabel("Number of candidates K")
    axis.set_ylabel("minADE@K (m) ↓", color="#4C78A8")
    axis.tick_params(axis="y", labelcolor="#4C78A8")
    axis.grid(alpha=0.2, linewidth=0.5)
    twin = axis.twinx()
    twin.spines["right"].set_visible(True)
    line_div = twin.plot(
        k, diversity, marker="s", color="#D46A4C", lw=1.2, label="Diversity"
    )[0]
    twin.set_ylabel("Diversity (m)", color="#D46A4C")
    twin.tick_params(axis="y", labelcolor="#D46A4C")
    gain_k8 = _improvement(minade[0], minade[np.where(k == 8)[0][0]])
    axis.annotate(
        f"K=1→8: {gain_k8:.1f}% lower",
        xy=(8, minade[np.where(k == 8)[0][0]]),
        xytext=(4.0, minade[0] - 0.018),
        arrowprops={"arrowstyle": "->", "lw": 0.7, "color": "#555555"},
    )
    axis.set_title("c  Multi-candidate coverage (seed 0)", loc="left", fontweight="bold")
    axis.legend(
        [line_ade, line_div],
        ["Oracle minADE@K", "Diversity"],
        loc="center right",
    )

    # d: integration budget exposes the latency/quality/smoothness boundary.
    axis = axes[1, 1]
    steps = np.asarray([int(row["steps"]) for row in step_rows])
    latency = np.asarray([float(row["latency_ms_per_scene"]) for row in step_rows])
    step_minade = np.asarray([float(row["minADE@K_m"]) for row in step_rows])
    smoothness = np.asarray([float(row["smoothness_m"]) for row in step_rows])
    bars = axis.bar(steps, latency, width=3.0, color="#B9C9D8", label="Latency")
    axis.set_xticks(steps)
    axis.set_xlabel("Euler integration steps")
    axis.set_ylabel("Latency (ms/scene)")
    axis.grid(axis="y", alpha=0.2, linewidth=0.5)
    twin = axis.twinx()
    twin.spines["right"].set_visible(True)
    line = twin.plot(
        steps,
        step_minade,
        marker="o",
        color="#D46A4C",
        lw=1.3,
        label="Oracle minADE@K",
    )[0]
    twin.set_ylabel("minADE@K (m) ↓", color="#D46A4C")
    twin.tick_params(axis="y", labelcolor="#D46A4C")
    axis.axvline(16, color="#555555", ls="--", lw=0.8)
    axis.annotate("selected", (16, latency[2]), xytext=(11.2, latency[2] + 0.50))
    smoothness_change = 100.0 * (smoothness[-1] - smoothness[2]) / smoothness[2]
    latency_change = 100.0 * (latency[-1] - latency[2]) / latency[2]
    axis.annotate(
        f"32 steps\n+{latency_change:.0f}% latency\n+{smoothness_change:.0f}% smoothness",
        xy=(32, latency[-1]),
        xytext=(20.0, latency[-1] - 0.55),
        arrowprops={"arrowstyle": "->", "lw": 0.7, "color": "#555555"},
    )
    axis.set_title("d  Explicit computation budget (seed 0)", loc="left", fontweight="bold")
    axis.legend([bars, line], ["Latency", "Oracle minADE@K"], loc="upper left")

    base = root / "figures" / "figure_J_method_advantages"
    _save(fig, base)
    return base


def _update_manifest(root: Path) -> None:
    path = root / "figures" / "figure_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    sources = list(manifest.get("quantitative_source_tables", []))
    for item in (
        "tables/k_sensitivity.csv",
        "figure_source_data/method_advantage_claims.csv",
    ):
        if item not in sources:
            sources.append(item)
    manifest["quantitative_source_tables"] = sources
    manifest["method_advantage_figure"] = "figure_J_method_advantages"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--style", type=Path, default=DEFAULT_STYLE)
    args = parser.parse_args()
    style = json.loads(args.style.read_text(encoding="utf-8"))
    _configure(style)
    source = _write_claim_source_data(args.results_root)
    figure = plot_method_advantages(args.results_root, style)
    _update_manifest(args.results_root)
    print(f"source_data={source}")
    print(f"figure={figure}")


if __name__ == "__main__":
    main()

"""Create publication figures for the paired VTF-Flow experiment.

The quantitative panel always uses every paired observation. Qualitative
examples are selected deterministically from the saved example pool and are
reported in ``selected_examples.csv`` to keep the image plate auditable.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update(
    {
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.0,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    }
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from TerraFlow.terrain.feasibility_field import AnalyticTerrainField


FLOW = "Flow"
GUIDED = "Feasibility-guided Flow"
FLOW_COLOR = "#D97706"
GUIDED_COLOR = "#009E73"
GT_COLOR = "#3775BA"
NEUTRAL = "#B8B8B8"
TEXT = "#303030"

METRICS = [
    ("selected_mean_terrain_cost", "Terrain cost", "lower"),
    ("selected_occupancy_violation_rate", "Occupancy violations", "lower"),
    ("selected_slope_violation_rate", "Slope violations", "lower"),
    ("selected_ADE_m", "Selected ADE", "lower"),
    ("selected_smoothness_m", "Second difference", "lower"),
    ("diversity_m", "Candidate diversity", "higher"),
]


def _read_paired(csv_path: Path) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    indices = np.array(sorted({int(row["index"]) for row in rows}), dtype=int)
    by_method = {}
    for method in (FLOW, GUIDED):
        lookup = {int(row["index"]): row for row in rows if row["method"] == method}
        if set(lookup) != set(indices.tolist()):
            raise ValueError(f"Unpaired rows for {method}")
        by_method[method] = {
            key: np.array([float(lookup[int(index)][key]) for index in indices], dtype=float)
            for key, _, _ in METRICS
        }
        by_method[method]["planning_time_ms"] = np.array(
            [float(lookup[int(index)]["planning_time_ms"]) for index in indices], dtype=float
        )
    return indices, by_method


def _bootstrap_summary(
    indices: np.ndarray,
    values: dict[str, dict[str, np.ndarray]],
    draws: int = 5000,
    seed: int = 20260828,
) -> list[dict[str, float | str]]:
    rng = np.random.default_rng(seed)
    n = len(indices)
    boot_index = rng.integers(0, n, size=(draws, n))
    summary = []
    for key, label, direction in METRICS:
        baseline = values[FLOW][key]
        guided = values[GUIDED][key]
        baseline_mean = float(baseline.mean())
        guided_mean = float(guided.mean())
        sign = -1.0 if direction == "lower" else 1.0
        # Positive values always mean an improvement after orienting the metric.
        effect = sign * 100.0 * (guided_mean - baseline_mean) / max(abs(baseline_mean), 1e-12)
        base_boot = baseline[boot_index].mean(axis=1)
        guided_boot = guided[boot_index].mean(axis=1)
        boot = sign * 100.0 * (guided_boot - base_boot) / np.maximum(
            np.abs(base_boot), 1e-12
        )
        lo, hi = np.quantile(boot, [0.025, 0.975])
        summary.append(
            {
                "metric": key,
                "display_name": label,
                "direction": direction,
                "n": n,
                "flow_mean": baseline_mean,
                "guided_mean": guided_mean,
                "improvement_percent": float(effect),
                "ci95_low": float(lo),
                "ci95_high": float(hi),
            }
        )
    return summary


def _write_dicts(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_quantitative(summary: list[dict], output_dir: Path) -> None:
    labels = [row["display_name"] for row in summary]
    effect = np.array([row["improvement_percent"] for row in summary])
    lo = np.array([row["ci95_low"] for row in summary])
    hi = np.array([row["ci95_high"] for row in summary])
    colors = np.where(lo > 0, GUIDED_COLOR, np.where(hi < 0, "#B64342", "#767676"))
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 3.25))
    ax.axvline(0, color="#777777", lw=0.8, ls="--", zorder=0)
    for yi, estimate, low, high, color in zip(y, effect, lo, hi, colors):
        ax.errorbar(
            estimate, yi,
            xerr=np.array([[estimate - low], [high - estimate]]),
            fmt="o", ms=6.5, color=color, ecolor=color, elinewidth=2.0,
            capsize=0, markeredgecolor="white", markeredgewidth=0.6, zorder=3,
        )
        anchor = high if estimate >= 0 else low
        ha = "left" if estimate >= 0 else "right"
        ax.annotate(
            f"{estimate:+.1f}%",
            (anchor, yi),
            xytext=(5 if estimate >= 0 else -5, 0),
            textcoords="offset points",
            va="center",
            ha=ha,
            fontsize=6.5,
            color=TEXT,
        )
    ax.set_yticks(y, labels)
    ax.set_xlabel("Paired relative improvement (%)   ← worse     better →")
    ax.set_title("Feasibility guidance trades small imitation loss for safer terrain", loc="left", pad=9)
    ax.grid(axis="x", color="#E8E8E8", lw=0.6, zorder=-2)
    fig.text(
        0.94, 0.055, "mean; 95% paired bootstrap CI; n=512",
        ha="right", va="bottom", fontsize=6.2, color="#666666",
    )
    ax.text(-0.14, 1.04, "a", transform=ax.transAxes, fontweight="bold", fontsize=10)
    fig.subplots_adjust(left=0.25, right=0.94, bottom=0.23, top=0.84)
    _save_figure(fig, output_dir / "flow_guidance_quantitative")


def _cost_map(terrain_map: np.ndarray) -> np.ndarray:
    field = AnalyticTerrainField(torch.from_numpy(terrain_map).float().unsqueeze(0))
    forward = torch.linspace(0.0, 24.0, 128)
    lateral = torch.linspace(-12.0, 12.0, 128)
    xx, yy = torch.meshgrid(forward, lateral, indexing="ij")
    xyz = torch.stack((xx, yy, torch.zeros_like(xx)), dim=-1).unsqueeze(0)
    return field.cost(xyz).squeeze(0).detach().cpu().numpy()


def _choose_examples(
    payload: np.lib.npyio.NpzFile,
    metric_indices: np.ndarray,
    values: dict[str, dict[str, np.ndarray]],
    count: int,
) -> np.ndarray:
    metric_lookup = {int(index): i for i, index in enumerate(metric_indices)}
    candidates = []
    for local, index in enumerate(payload["index"].astype(int)):
        pos = metric_lookup[int(index)]
        terrain_gain = (
            values[FLOW]["selected_mean_terrain_cost"][pos]
            - values[GUIDED]["selected_mean_terrain_cost"][pos]
        )
        ade_penalty = max(
            0.0,
            values[GUIDED]["selected_ADE_m"][pos] - values[FLOW]["selected_ADE_m"][pos],
        )
        candidates.append((terrain_gain - 0.15 * ade_penalty, local, int(index), terrain_gain))
    candidates.sort(reverse=True)
    # Use separated ranks instead of adjacent frames to reduce visual redundancy.
    pool = candidates[: max(count * 3, count)]
    chosen = [pool[i] for i in np.linspace(0, len(pool) - 1, count, dtype=int)]
    return np.array([entry[1] for entry in chosen], dtype=int)


def _draw_bev(
    ax: plt.Axes,
    cost: np.ndarray,
    candidates: np.ndarray,
    selected: np.ndarray,
    gt: np.ndarray,
    color: str,
    title: str,
) -> None:
    ax.imshow(
        cost.T,
        origin="lower",
        extent=(0, 24, -12, 12),
        cmap="RdYlGn_r",
        vmin=0,
        vmax=1,
        alpha=0.83,
        interpolation="bilinear",
        aspect="equal",
    )
    ax.contour(
        np.linspace(0, 24, cost.shape[0]),
        np.linspace(-12, 12, cost.shape[1]),
        cost.T,
        levels=[0.35, 0.55],
        colors=["#555555", "#202020"],
        linewidths=[0.45, 0.75],
        alpha=0.65,
    )
    for path in candidates:
        ax.plot(path[:, 0], path[:, 1], color=NEUTRAL, lw=0.75, alpha=0.62, zorder=2)
    ax.plot(gt[:, 0], gt[:, 1], color=GT_COLOR, lw=1.4, ls=(0, (3, 2)), zorder=4)
    ax.plot(selected[:, 0], selected[:, 1], color=color, lw=2.0, zorder=5)
    ax.scatter([0], [0], s=17, color="#202020", edgecolor="white", linewidth=0.45, zorder=6)
    ax.set_xlim(0, 24)
    ax.set_ylim(-12, 12)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=7.2, pad=3)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)


def plot_qualitative(
    payload: np.lib.npyio.NpzFile,
    chosen: np.ndarray,
    output_dir: Path,
) -> list[dict]:
    count = len(chosen)
    fig, axes = plt.subplots(2, count, figsize=(7.25, 4.15), sharex=True, sharey=True)
    audit_rows = []
    for column, local in enumerate(chosen):
        index = int(payload["index"][local])
        cost = _cost_map(payload["terrain_map"][local])
        for row, (method, color) in enumerate(((FLOW, FLOW_COLOR), (GUIDED, GUIDED_COLOR))):
            ax = axes[row, column]
            _draw_bev(
                ax,
                cost,
                payload[f"{method} candidates"][local],
                payload[f"{method} selected"][local],
                payload["gt"][local],
                color,
                f"test index {index}" if row == 0 else "",
            )
            if column == 0:
                ax.set_ylabel(f"{method}\nLateral [m]", color=color, fontweight="bold")
            if row == 1:
                ax.set_xlabel("Forward [m]")
        audit_rows.append({"panel_column": column + 1, "test_index": index})
    axes[0, 0].text(-0.28, 1.08, "a", transform=axes[0, 0].transAxes, fontweight="bold", fontsize=10)
    axes[1, 0].text(-0.28, 1.08, "b", transform=axes[1, 0].transAxes, fontweight="bold", fontsize=10)
    handles = [
        plt.Line2D([0], [0], color=NEUTRAL, lw=1.3, label="Candidates (K=8)"),
        plt.Line2D([0], [0], color=GT_COLOR, lw=1.4, ls=(0, (3, 2)), label="GT future"),
        plt.Line2D([0], [0], color=FLOW_COLOR, lw=2.0, label=FLOW),
        plt.Line2D([0], [0], color=GUIDED_COLOR, lw=2.0, label=GUIDED),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.48, -0.01))
    scalar = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=plt.Normalize(0, 1))
    color_ax = fig.add_axes((0.945, 0.18, 0.010, 0.66))
    colorbar = fig.colorbar(scalar, cax=color_ax)
    colorbar.set_label("Terrain cost", fontsize=6.5)
    colorbar.ax.tick_params(labelsize=6, width=0.6, length=2)
    fig.suptitle("Same scene and initial noise: guidance shifts candidate trajectories away from high-cost terrain", y=0.985)
    fig.subplots_adjust(left=0.09, right=0.93, bottom=0.14, top=0.90, wspace=0.08, hspace=0.08)
    _save_figure(fig, output_dir / "flow_guidance_candidates")
    return audit_rows


def plot_integration(payload: np.lib.npyio.NpzFile, local: int, output_dir: Path) -> None:
    history = payload[f"{GUIDED} integration"][local]
    candidate = payload[f"{GUIDED} candidates"][local]
    gt = payload["gt"][local]
    cost = _cost_map(payload["terrain_map"][local])
    steps = np.linspace(0, len(history) - 1, 5, dtype=int)
    fig, axes = plt.subplots(1, len(steps), figsize=(7.25, 1.95), sharex=True, sharey=True)
    for panel, (ax, step) in enumerate(zip(axes, steps)):
        ax.imshow(cost.T, origin="lower", extent=(0, 24, -12, 12), cmap="RdYlGn_r", vmin=0, vmax=1, alpha=0.83, aspect="equal")
        if panel == len(steps) - 1:
            for path in candidate:
                ax.plot(path[:, 0], path[:, 1], color=NEUTRAL, lw=0.65, alpha=0.45)
        ax.plot(gt[:, 0], gt[:, 1], color=GT_COLOR, lw=1.1, ls=(0, (3, 2)))
        ax.plot(history[step, :, 0], history[step, :, 1], color=GUIDED_COLOR, lw=1.8)
        ax.scatter([0], [0], s=13, color="#202020", edgecolor="white", linewidth=0.4, zorder=5)
        ax.set_xlim(0, 24)
        ax.set_ylim(-12, 12)
        ax.set_aspect("equal")
        ax.set_title(f"ODE step {step + 1}/16")
        ax.set_xlabel("Forward [m]")
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
    axes[0].set_ylabel("Lateral [m]")
    axes[0].text(-0.30, 1.08, "a", transform=axes[0].transAxes, fontweight="bold", fontsize=10)
    fig.suptitle("Feasibility gradients act throughout Flow integration", y=0.99)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.19, top=0.84, wspace=0.08)
    _save_figure(fig, output_dir / "flow_guidance_integration")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    indices, values = _read_paired(args.results_dir / "per_sample_metrics.csv")
    summary = _bootstrap_summary(indices, values)
    _write_dicts(args.output_dir / "paired_bootstrap_summary.csv", summary)
    plot_quantitative(summary, args.output_dir)
    payload = np.load(args.results_dir / "flow_examples.npz", allow_pickle=False)
    chosen = _choose_examples(payload, indices, values, args.examples)
    audit = plot_qualitative(payload, chosen, args.output_dir)
    _write_dicts(args.output_dir / "selected_examples.csv", audit)
    plot_integration(payload, int(chosen[0]), args.output_dir)
    print(f"Saved figures and source tables to {args.output_dir}")


if __name__ == "__main__":
    main()

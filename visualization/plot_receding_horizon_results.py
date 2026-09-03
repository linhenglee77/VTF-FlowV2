"""Publication figures for learned fields and five-seed receding-horizon tests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7.0,
    "axes.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})

METHODS = ("Flow", "Analytic-guided Flow", "Learned-guided Flow", "VTF-Flow")
COLORS = {
    "Flow": "#596780",
    "Analytic-guided Flow": "#D97706",
    "Learned-guided Flow": "#009E73",
    "VTF-Flow": "#7C6CCF",
}


def save_figure(fig, base: Path):
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def read_csv(path: Path):
    return list(csv.DictReader(path.open(encoding="utf-8")))


def plot_closed_loop(seed_rows, output_dir: Path):
    panels = [
        ("completion", "Completion (%)", 100.0, True),
        ("collision", "Episodes touching known risk (%)", 100.0, False),
        ("conservative_risk", "Conservative executed risk", 1.0, False),
        ("mean_abs_steering_deg", "Mean |steering| (deg)", 1.0, False),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(7.25, 2.45))
    for panel, (ax, (key, label, scale, higher)) in enumerate(zip(axes, panels)):
        for x, method in enumerate(METHODS):
            values = np.asarray([float(row[key]) * scale for row in seed_rows if row["method"] == method])
            assert len(values) == 5 and np.isfinite(values).all()
            jitter = np.linspace(-0.045, 0.045, len(values))
            ax.scatter(np.full(len(values), x) + jitter, values, s=14, color=COLORS[method], alpha=0.72, zorder=3)
            mean, std = values.mean(), values.std(ddof=1)
            ax.errorbar(x, mean, yerr=std, fmt="o", ms=6.0, color=COLORS[method], ecolor=COLORS[method], elinewidth=1.5, capsize=3, markeredgecolor="white", markeredgewidth=0.6, zorder=4)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels(["Flow", "Analytic", "Learned", "VTF-Flow"], rotation=25, ha="right", rotation_mode="anchor")
        ax.set_ylabel(label)
        ax.grid(axis="y", color="#E7E7E7", lw=0.55, zorder=-2)
        ax.text(-0.18, 1.04, chr(ord("a") + panel), transform=ax.transAxes, fontweight="bold", fontsize=9)
        direction = "higher is better" if higher else "lower is better"
        ax.set_title(direction, fontsize=6.3, color="#666666", pad=4)
    handles = [plt.Line2D([0], [0], marker="o", color=COLORS[m], lw=0, label=m) for m in METHODS]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.91))
    fig.suptitle("Five-seed receding-horizon evaluation reveals a safety–completion trade-off", y=1.01)
    fig.text(0.995, 0.015, "mean ± seed SD; 64 paired test scenes per seed", ha="right", fontsize=6.2, color="#666666")
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.23, top=0.73, wspace=0.40)
    save_figure(fig, output_dir / "receding_horizon_5seed")


def plot_field_comparison(report, output_dir: Path):
    order = ("Binary traversability", "Analytic terrain field", "Learned feasibility field")
    labels = ("Binary traversability\n(label-copy reference)", "Analytic field", "Learned geometry field")
    colors = ("#A8A8A8", "#D97706", "#009E73")
    mae = np.asarray([report["metrics"][name]["masked_mae"] for name in order])
    iou = np.asarray([report["metrics"][name]["masked_iou_at_0.5"] for name in order])
    assert np.all(mae > 0) and np.all(iou > 0)
    mae = np.maximum(mae, 1e-12)
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.55))
    for ax, values, ylabel, panel in (
        (axes[0], mae, "Masked MAE (log scale)", "a"),
        (axes[1], iou, "Masked IoU at 0.5", "b"),
    ):
        bars = ax.bar(range(3), values, color=colors, edgecolor="white", linewidth=0.7)
        ax.set_xticks(range(3), labels, rotation=16, ha="right", rotation_mode="anchor")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E7E7E7", lw=0.55, zorder=-2)
        ax.text(-0.14, 1.04, panel, transform=ax.transAxes, fontweight="bold", fontsize=9)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value * (1.20 if ax is axes[0] else 1.001), f"{value:.3g}", ha="center", va="bottom", fontsize=6.2)
    axes[0].set_yscale("log")
    axes[0].tick_params(axis="y", labelsize=8)
    axes[1].set_ylim(0.92, 1.005)
    fig.suptitle("Geometry-only learning improves the analytic field without reading the target-like channel", y=1.01)
    fig.text(0.995, 0.015, "sequence-disjoint RELLIS-3D test; 2,443 frames; 4,553,680 valid cells", ha="right", fontsize=6.2, color="#666666")
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.31, top=0.80, wspace=0.30)
    save_figure(fig, output_dir / "learned_field_test")


def plot_traces(payload, trajectory_cache: Path, perception_cache: Path, output_dir: Path):
    terrain_goal = np.load(trajectory_cache / "test" / "goal.npy", mmap_mode="r")
    gt_all = np.load(trajectory_cache / "test" / "trajectory.npy", mmap_mode="r")
    risk_all = np.load(perception_cache / "test" / "risk_target.npy", mmap_mode="r")
    mask_all = np.load(perception_cache / "test" / "supervision_mask.npy", mmap_mode="r")
    local_columns = np.asarray([0, 3, 7], dtype=int)
    fig, axes = plt.subplots(4, 3, figsize=(7.25, 7.25), sharex=True, sharey=True)
    for column, local in enumerate(local_columns):
        index = int(payload["index"][local])
        risk = np.asarray(risk_all[index, 0], dtype=np.float32) / 255.0
        known = np.asarray(mask_all[index, 0], dtype=np.float32)
        display = risk * known + 0.5 * (1.0 - known)
        goal, gt = np.asarray(terrain_goal[index]), np.asarray(gt_all[index])
        for row, method in enumerate(METHODS):
            ax = axes[row, column]
            ax.imshow(display.T, origin="lower", extent=(0, 24, -12, 12), cmap="RdYlGn_r", vmin=0, vmax=1, alpha=0.84, aspect="equal")
            trace = payload[method][local]
            trace = trace[np.isfinite(trace[:, 0])]
            ax.plot(gt[:, 0], gt[:, 1], color="#3775BA", lw=1.0, ls=(0, (3, 2)))
            ax.plot(trace[:, 0], trace[:, 1], color=COLORS[method], lw=2.0)
            ax.scatter([0], [0], s=13, color="#202020", edgecolor="white", linewidth=0.4, zorder=5)
            ax.scatter([goal[0]], [goal[1]], marker="*", s=36, color="#F4D03F", edgecolor="#303030", linewidth=0.5, zorder=5)
            ax.set_xlim(0, 24); ax.set_ylim(-12, 12); ax.set_aspect("equal")
            ax.spines["top"].set_visible(True); ax.spines["right"].set_visible(True)
            if row == 0:
                ax.set_title(f"test index {index}")
            if column == 0:
                ax.set_ylabel(f"{method}\nLateral [m]", color=COLORS[method], fontweight="bold")
            if row == 3:
                ax.set_xlabel("Forward [m]")
    axes[0, 0].text(-0.24, 1.06, "a", transform=axes[0, 0].transAxes, fontweight="bold", fontsize=9)
    handles = [
        plt.Line2D([0], [0], color="#3775BA", lw=1.1, ls=(0, (3, 2)), label="GT future"),
        plt.Line2D([0], [0], color="#555555", lw=2.0, label="Executed bicycle trajectory"),
        plt.Line2D([0], [0], marker="*", color="none", markerfacecolor="#F4D03F", markeredgecolor="#303030", label="Goal"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.005))
    scalar = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=plt.Normalize(0, 1))
    color_ax = fig.add_axes((0.945, 0.15, 0.012, 0.70))
    colorbar = fig.colorbar(scalar, cax=color_ax)
    colorbar.set_label("Semantic risk (unknown = 0.5)", fontsize=6.5)
    colorbar.ax.tick_params(labelsize=6, width=0.6, length=2)
    fig.suptitle("Executed trajectories are replanned every 0.5 s from the updated ego frame", y=0.995)
    fig.subplots_adjust(left=0.13, right=0.93, bottom=0.075, top=0.955, wspace=0.08, hspace=0.08)
    save_figure(fig, output_dir / "receding_horizon_examples")


def write_training_summary(flow_dirs, output_dir: Path):
    rows = []
    for directory in flow_dirs:
        summary = json.loads((directory / "training_summary.json").read_text(encoding="utf-8"))
        seed = int(directory.name.rsplit("seed", 1)[1])
        rows.append({"seed": seed, "best_val_loss": summary["best_val_loss"], "epochs": summary["epochs"], "wall_time_s": summary["wall_time_s"], "checkpoint": str(directory / "best.pt")})
    with (output_dir / "multi_seed_training_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--closed-loop-dir", type=Path, required=True)
    parser.add_argument("--field-report", type=Path, required=True)
    parser.add_argument("--trajectory-cache", type=Path, required=True)
    parser.add_argument("--perception-cache", type=Path, required=True)
    parser.add_argument("--flow-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_rows = read_csv(args.closed_loop_dir / "per_seed_metrics.csv")
    plot_closed_loop(seed_rows, args.output_dir)
    plot_field_comparison(json.loads(args.field_report.read_text(encoding="utf-8")), args.output_dir)
    payload = np.load(args.closed_loop_dir / "closed_loop_examples.npz")
    plot_traces(payload, args.trajectory_cache, args.perception_cache, args.output_dir)
    write_training_summary(args.flow_dirs, args.output_dir)
    print(f"Saved figures and source tables to {args.output_dir}")


if __name__ == "__main__":
    main()

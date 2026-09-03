"""Generate reproducible final VTF-Flow figures from saved source tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


TERRAFLOW_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TERRAFLOW_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from TerraFlow.evaluation.final_experiments import read_csv, write_csv  # noqa: E402
from TerraFlow.scripts.train_regression import CombinedSceneDataset  # noqa: E402
from TerraFlow.terrain.feasibility_field import AnalyticTerrainField, TerrainFieldConfig  # noqa: E402


DEFAULT_ROOT = TERRAFLOW_ROOT / "outputs" / "final_experiments"
DEFAULT_CACHE = TERRAFLOW_ROOT.parent / "data" / "RELLIS3D" / "trajectory_cache_h150_s5"
DEFAULT_STYLE = TERRAFLOW_ROOT / "configs" / "final_figure_style.json"
TERRAIN_CONFIG = TERRAFLOW_ROOT / "configs" / "rellis3d_flow_feasibility.json"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _configure(style: Mapping[str, Any]) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": float(style["font_size_pt"]),
        "axes.labelsize": float(style["font_size_pt"]),
        "axes.titlesize": float(style["font_size_pt"]) + 1,
        "legend.fontsize": float(style["font_size_pt"]) - 0.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    })


def _save(fig: plt.Figure, base: Path, dpi: int) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _float(rows: Sequence[Mapping[str, Any]], name: str) -> np.ndarray:
    return np.asarray([float(row[name]) for row in rows], dtype=np.float64)


def plot_method_overview(root: Path, figures: Path, style: Mapping[str, Any]) -> None:
    rows = read_csv(root / "tables" / "table1_main_comparison.csv")
    colors = style["method_colors"]
    labels = [row["display_name"] for row in rows]
    metrics = (
        ("minADE@K_m", "minADE@K (m)"),
        ("mean_vehicle_conditioned_cost", "Vehicle-conditioned cost"),
        ("terrain_violation_rate", "Terrain violation"),
        ("smoothness_m", "Smoothness (m)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(float(style["figure_width_in"]), 5.2), constrained_layout=True)
    for panel, (axis, (metric, title)) in enumerate(zip(axes.flat, metrics)):
        means = _float(rows, f"{metric}_mean")
        sds = _float(rows, f"{metric}_sd")
        axis.bar(
            np.arange(len(rows)), means, yerr=sds,
            color=[colors[label] for label in labels], edgecolor="white", linewidth=0.5,
            capsize=2,
        )
        axis.set_xticks(
            np.arange(len(rows)), labels, rotation=28, ha="right",
            rotation_mode="anchor",
        )
        axis.set_ylabel(title)
        axis.set_title(f"{chr(97 + panel)}  {title}", loc="left", fontweight="bold")
        axis.grid(axis="y", alpha=0.2, linewidth=0.5)
    _save(fig, figures / "figure_A_method_overview", int(style["dpi"]))


def plot_pareto(root: Path, figures: Path, style: Mapping[str, Any]) -> None:
    rows = read_csv(root / "tables" / "table1_main_comparison.csv")
    eta = read_csv(root / "tables" / "table3_eta_sensitivity.csv")
    colors = style["method_colors"]
    fig, axis = plt.subplots(figsize=(3.5, 3.4), constrained_layout=True)
    eta_x = _float(eta, "minADE@K_m")
    eta_y = _float(eta, "mean_vehicle_conditioned_cost")
    axis.plot(eta_x, eta_y, color="#A0A0A0", lw=1.0, zorder=1)
    for row, x, y in zip(eta, eta_x, eta_y):
        axis.scatter(x, y, s=20, color="#A0A0A0", zorder=2)
        axis.annotate(f"η={float(row['eta']):g}", (x, y), xytext=(3, 3), textcoords="offset points")
    for row in rows:
        label = row["display_name"]
        x = float(row["minADE@K_m_mean"])
        y = float(row["mean_vehicle_conditioned_cost_mean"])
        axis.errorbar(
            x, y, xerr=float(row["minADE@K_m_sd"]),
            yerr=float(row["mean_vehicle_conditioned_cost_sd"]),
            fmt="o", ms=5, capsize=2, color=colors[label], label=label, zorder=3,
        )
    axis.set_xlabel("minADE@K (m) ↓")
    axis.set_ylabel("Vehicle-conditioned cost ↓")
    axis.set_title("Fidelity–feasibility trade-off", loc="left", fontweight="bold")
    axis.grid(alpha=0.2, linewidth=0.5)
    axis.legend(loc="best")
    _save(fig, figures / "figure_B_fidelity_feasibility_pareto", int(style["dpi"]))


def plot_eta(root: Path, figures: Path, style: Mapping[str, Any]) -> None:
    rows = read_csv(root / "tables" / "table3_eta_sensitivity.csv")
    eta = _float(rows, "eta")
    metrics = (
        ("minADE@K_m", "minADE@K (m)", "eta_vs_minADE"),
        ("mean_vehicle_conditioned_cost", "Vehicle-conditioned cost", "eta_vs_vehicle_cost"),
        ("terrain_violation_rate", "Terrain violation", "eta_vs_terrain_violation"),
        ("smoothness_m", "Smoothness (m)", "eta_vs_smoothness"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(float(style["figure_width_in"]), 4.8), constrained_layout=True)
    for panel, (axis, (metric, ylabel, filename)) in enumerate(zip(axes.flat, metrics)):
        values = _float(rows, metric)
        axis.plot(eta, values, marker="o", lw=1.4, color="#D46A4C")
        axis.axvline(0.2, color="#777777", ls="--", lw=0.8)
        axis.set_xlabel("Guidance strength η")
        axis.set_ylabel(ylabel)
        axis.set_title(f"{chr(97 + panel)}  {ylabel}", loc="left", fontweight="bold")
        axis.grid(alpha=0.2, linewidth=0.5)
        single, single_axis = plt.subplots(figsize=(3.5, 2.7), constrained_layout=True)
        single_axis.plot(eta, values, marker="o", lw=1.4, color="#D46A4C")
        single_axis.axvline(0.2, color="#777777", ls="--", lw=0.8)
        single_axis.set(xlabel="Guidance strength η", ylabel=ylabel)
        single_axis.grid(alpha=0.2, linewidth=0.5)
        _save(single, figures / filename, int(style["dpi"]))
    _save(fig, figures / "figure_C_eta_sensitivity", int(style["dpi"]))


def plot_steps(root: Path, figures: Path, style: Mapping[str, Any]) -> None:
    rows = read_csv(root / "tables" / "table4_sampling_step_sensitivity.csv")
    steps = _float(rows, "steps")
    metrics = (
        ("minADE@K_m", "minADE@K (m)"),
        ("mean_vehicle_conditioned_cost", "Vehicle-conditioned cost"),
        ("terrain_violation_rate", "Terrain violation"),
        ("latency_ms_per_scene", "Latency (ms/scene)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(float(style["figure_width_in"]), 4.8), constrained_layout=True)
    for panel, (axis, (metric, ylabel)) in enumerate(zip(axes.flat, metrics)):
        axis.plot(steps, _float(rows, metric), marker="o", color="#4C78A8", lw=1.4)
        axis.axvline(16, color="#777777", ls="--", lw=0.8)
        axis.set_xticks(steps)
        axis.set(xlabel="Euler integration steps", ylabel=ylabel)
        axis.set_title(f"{chr(97 + panel)}  {ylabel}", loc="left", fontweight="bold")
        axis.grid(alpha=0.2, linewidth=0.5)
    _save(fig, figures / "figure_D_sampling_steps", int(style["dpi"]))


def plot_sequences(root: Path, figures: Path, style: Mapping[str, Any]) -> None:
    rows = read_csv(root / "tables" / "per_sequence_analysis.csv")
    sequence = [row["sequence"] for row in rows]
    metrics = (
        ("vehicle_cost", "Vehicle cost improvement (%)"),
        ("minADE", "minADE improvement (%)"),
        ("terrain_violation", "Terrain violation improvement (%)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(float(style["figure_width_in"]), 2.7), constrained_layout=True)
    role_colors = {"train": "#B8B8B8", "validation": "#F2A65A", "test": "#D46A4C"}
    for panel, (axis, (suffix, ylabel)) in enumerate(zip(axes, metrics)):
        flow = _float(rows, f"Flow_{suffix}")
        full = _float(rows, f"Full_{suffix}")
        improvement = 100.0 * (flow - full) / np.maximum(np.abs(flow), 1e-12)
        axis.bar(sequence, improvement, color=[role_colors[row["role"]] for row in rows])
        axis.axhline(0.0, color="black", lw=0.7)
        axis.set_ylabel(ylabel)
        axis.set_title(f"{chr(97 + panel)}", loc="left", fontweight="bold")
        plt.setp(
            axis.get_xticklabels(), rotation=35, ha="right",
            rotation_mode="anchor",
        )
        axis.grid(axis="y", alpha=0.2, linewidth=0.5)
    axes[0].legend(
        handles=[Patch(facecolor=color, label=role.capitalize()) for role, color in role_colors.items()],
        loc="upper right",
    )
    _save(fig, figures / "figure_E_per_sequence", int(style["dpi"]))


def _prepare_qualitative_source(root: Path, cache_root: Path) -> list[Path]:
    selection = _json(root / "qualitative_case_selection.json")
    source = CombinedSceneDataset(cache_root, ("train", "val", "test"))
    a_npz = np.load(root / "main_primary_seed0_A" / "predictions.npz")
    d_npz = np.load(root / "main_primary_seed0_D" / "predictions.npz")
    a_lookup = {str(value): index for index, value in enumerate(a_npz["scene_ids"])}
    d_lookup = {str(value): index for index, value in enumerate(d_npz["scene_ids"])}
    terrain_cfg = TerrainFieldConfig(**_json(TERRAIN_CONFIG)["terrain_field"])
    source_dir = root / "figure_source_data"
    paths: list[Path] = []
    for label, record in selection.items():
        if label == "selection_rules":
            continue
        scene_id = str(record["scene_id"])
        if scene_id not in a_lookup or scene_id not in d_lookup:
            raise ValueError(f"qualitative scene missing from paired predictions: {scene_id}")
        dataset_index = int(record["dataset_index"])
        scene = source[dataset_index].as_batch()
        field = AnalyticTerrainField(scene.terrain_map, terrain_cfg)
        map_h, map_w = scene.terrain_map.shape[-2:]
        x = np.linspace(0.0, terrain_cfg.forward_m, map_h)
        y = np.linspace(-terrain_cfg.lateral_m, terrain_cfg.lateral_m, map_w)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        import torch
        query = torch.from_numpy(np.stack((xx, yy), axis=-1).reshape(1, -1, 2)).float()
        feasibility = field.query(query).reshape(map_h, map_w).detach().cpu().numpy()
        a_pos, d_pos = a_lookup[scene_id], d_lookup[scene_id]
        a_traj = a_npz["trajectories"][a_pos]
        d_traj = d_npz["trajectories"][d_pos]
        gt = a_npz["ground_truth"][a_pos]
        rows: list[dict[str, Any]] = []
        for i in range(map_h):
            for j in range(map_w):
                rows.append({
                    "kind": "terrain", "candidate": -1, "waypoint": i * map_w + j,
                    "x": float(xx[i, j]), "y": float(yy[i, j]), "z": 0.0,
                    "feasibility": float(feasibility[i, j]), "selected": 0,
                    "scene_id": scene_id,
                })
        for kind, trajectories in (("Flow", a_traj), ("Full", d_traj)):
            best = int(np.linalg.norm(trajectories - gt[None], axis=-1).mean(axis=-1).argmin())
            for candidate, trajectory in enumerate(trajectories):
                for waypoint, point in enumerate(trajectory):
                    rows.append({
                        "kind": kind, "candidate": candidate, "waypoint": waypoint,
                        "x": float(point[0]), "y": float(point[1]), "z": float(point[2]),
                        "feasibility": "", "selected": int(candidate == best),
                        "scene_id": scene_id,
                    })
        for waypoint, point in enumerate(gt):
            rows.append({
                "kind": "GT", "candidate": -1, "waypoint": waypoint,
                "x": float(point[0]), "y": float(point[1]), "z": float(point[2]),
                "feasibility": "", "selected": 0, "scene_id": scene_id,
            })
        path = source_dir / f"qualitative_{label}.csv"
        write_csv(path, rows)
        paths.append(path)
    return paths


def plot_qualitative(root: Path, cache_root: Path, figures: Path, style: Mapping[str, Any]) -> None:
    source_paths = _prepare_qualitative_source(root, cache_root)
    for path in source_paths:
        rows = read_csv(path)
        terrain = [row for row in rows if row["kind"] == "terrain"]
        xs = np.unique(_float(terrain, "x"))
        ys = np.unique(_float(terrain, "y"))
        feasibility = _float(terrain, "feasibility").reshape(len(xs), len(ys))
        fig, axes = plt.subplots(1, 2, figsize=(float(style["figure_width_in"]), 3.4), sharex=True, sharey=True, constrained_layout=True)
        for panel, (axis, method) in enumerate(zip(axes, ("Flow", "Full"))):
            image = axis.imshow(
                feasibility, extent=(ys.min(), ys.max(), xs.min(), xs.max()),
                origin="lower", aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0,
            )
            method_rows = [row for row in rows if row["kind"] == method]
            candidates = sorted({int(row["candidate"]) for row in method_rows})
            gt = [row for row in rows if row["kind"] == "GT"]
            for candidate in candidates:
                trajectory = sorted(
                    [row for row in method_rows if int(row["candidate"]) == candidate],
                    key=lambda row: int(row["waypoint"]),
                )
                axis.plot(_float(trajectory, "y"), _float(trajectory, "x"), color="white", alpha=0.42, lw=0.8)
                if int(trajectory[0]["selected"]) == 1:
                    axis.plot(
                        _float(trajectory, "y"), _float(trajectory, "x"),
                        color="#32C5FF", alpha=1.0, lw=1.5, label="Selected (minADE)",
                    )
            axis.plot(_float(gt, "y"), _float(gt, "x"), color="#D73027", lw=1.8, label="GT")
            axis.scatter([0.0], [0.0], marker="*", s=32, color="black", zorder=4)
            display_method = "VTF-Flow" if method == "Full" else method
            axis.set_title(
                f"{chr(97 + panel)}  {display_method}",
                loc="left",
                fontweight="bold",
            )
            axis.set_xlabel("Ego lateral y (m)")
            axis.legend(loc="upper right")
            axis.grid(False)
        axes[0].set_ylabel("Ego forward x (m)")
        fig.colorbar(image, ax=axes, label="Terrain feasibility F (relative)", fraction=0.025)
        _save(fig, figures / f"figure_F_{path.stem.replace('qualitative_', '')}", int(style["dpi"]))


def plot_failures(root: Path, figures: Path, style: Mapping[str, Any]) -> None:
    rows = read_csv(root / "failure_case_index.csv")
    categories = sorted({row["category"] for row in rows})
    counts = [sum(row["category"] == category for row in rows) for category in categories]
    colors = {category: color for category, color in zip(categories, ("#4C78A8", "#F2A65A", "#72A0C1", "#D46A4C"))}
    fig, axes = plt.subplots(1, 2, figsize=(float(style["figure_width_in"]), 3.0), constrained_layout=True)
    axes[0].barh(np.arange(len(categories)), counts, color=[colors[value] for value in categories])
    axes[0].set_yticks(np.arange(len(categories)), [value.replace("_", " ") for value in categories])
    axes[0].set_xlabel("Scenes")
    axes[0].set_title("a  Failure taxonomy", loc="left", fontweight="bold")
    for category in categories:
        subset = [row for row in rows if row["category"] == category]
        axes[1].scatter(
            _float(subset, "minADE_delta_m"), _float(subset, "vehicle_cost_delta"),
            s=7, alpha=0.45, color=colors[category], label=category.replace("_", " "),
            rasterized=True,
        )
    axes[1].axhline(0.0, color="black", lw=0.7)
    axes[1].axvline(0.0, color="black", lw=0.7)
    axes[1].set(
        xlabel="Δ minADE, VTF-Flow − Flow (m)",
        ylabel="Δ vehicle cost, VTF-Flow − Flow",
    )
    axes[1].set_title("b  Paired scene changes", loc="left", fontweight="bold")
    axes[1].legend(fontsize=5.5)
    axes[1].grid(alpha=0.15, linewidth=0.5)
    _save(fig, figures / "figure_G_failure_cases", int(style["dpi"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--style", type=Path, default=DEFAULT_STYLE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    style = _json(args.style)
    _configure(style)
    figures = args.output_root / "figures"
    plot_method_overview(args.output_root, figures, style)
    plot_pareto(args.output_root, figures, style)
    plot_eta(args.output_root, figures, style)
    plot_steps(args.output_root, figures, style)
    plot_sequences(args.output_root, figures, style)
    plot_qualitative(args.output_root, args.cache_root, figures, style)
    plot_failures(args.output_root, figures, style)
    save_manifest = {
        "backend": "Python/matplotlib",
        "quantitative_source_tables": [
            "tables/table1_main_comparison.csv", "tables/table3_eta_sensitivity.csv",
            "tables/table4_sampling_step_sensitivity.csv", "tables/per_sequence_analysis.csv",
            "failure_case_index.csv",
        ],
        "qualitative_source_pattern": "figure_source_data/qualitative_*.csv",
        "formats": ["svg", "pdf", "tiff", "png"],
    }
    (figures / "figure_manifest.json").write_text(
        json.dumps(save_manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "figures": str(figures.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

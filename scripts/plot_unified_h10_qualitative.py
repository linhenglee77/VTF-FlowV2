"""Plot traceable qualitative scenes from the unified H=10 benchmark.

Scenes are selected from cross-seed scene-level metrics using fixed criteria,
not by manual visual inspection.  The exported figure shows the actual cached
planner terrain map and frozen seed-0 predictions.  All candidates are shown;
the minimum-ADE member is emphasized only as a standard best-of-K diagnostic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch


TERRAFLOW_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TERRAFLOW_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from TerraFlow.scripts.train_regression import CombinedSceneDataset  # noqa: E402
from TerraFlow.terrain.feasibility_field import (  # noqa: E402
    AnalyticTerrainField,
    TerrainFieldConfig,
)


DEFAULT_BENCHMARK = TERRAFLOW_ROOT / "outputs" / "unified_h10_benchmark"
DEFAULT_CACHE = WORKSPACE_ROOT / "data" / "RELLIS3D" / "trajectory_cache_h150_s5"

METHODS = ("FLOW", "VT", "VTF")
DISPLAY_NAMES = {
    "FLOW": "Flow baseline",
    "VT": "VTF-Flow w/o kinematic terms",
    "VTF": "VTF-Flow (ours)",
}
COLORS = {
    "FLOW": "#7C8796",
    "VT": "#3478A8",
    "VTF": "#C84D58",
}
CATEGORY_TITLES = {
    "terrain": "Terrain-violation reduction",
    "kinematic": "Kinematic-feasibility correction",
    "smoothness": "Trajectory-coherence improvement",
    "balanced": "Balanced terrain–kinematic gain",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _scene_means(benchmark_root: Path, method: str) -> pd.DataFrame:
    """Average matched scene metrics over all available training seeds."""

    frames = []
    for path in sorted((benchmark_root / "runs").glob(f"{method}_seed*/scene_level_metrics.csv")):
        frame = pd.read_csv(path)
        frame["seed_source"] = int(path.parent.name.split("seed")[-1])
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no scene-level runs found for {method}")
    combined = pd.concat(frames, ignore_index=True)
    identifiers = ["scene_id", "sequence", "frame_id", "dataset_index"]
    numeric = [
        name for name in combined.select_dtypes(include=[np.number]).columns
        if name not in {"seed", "seed_source", "K", "dataset_index", "frame_id"}
    ]
    means = combined.groupby(identifiers, as_index=False)[numeric].mean()
    return means.set_index("scene_id").add_prefix(f"{method}_")


def build_selection_frame(benchmark_root: Path) -> pd.DataFrame:
    """Build the auditable cross-seed advantage frame for all 1909 scenes."""

    merged = pd.concat(
        [_scene_means(benchmark_root, method) for method in METHODS], axis=1
    )
    merged["delta_tvk_vs_flow"] = (
        merged["FLOW_mean_unified_tvk_cost"]
        - merged["VTF_mean_unified_tvk_cost"]
    )
    merged["delta_tvk_vs_wo_kin"] = (
        merged["VT_mean_unified_tvk_cost"]
        - merged["VTF_mean_unified_tvk_cost"]
    )
    merged["delta_terrain_violation_vs_flow"] = (
        merged["FLOW_terrain_violation_rate"]
        - merged["VTF_terrain_violation_rate"]
    )
    merged["delta_curvature_violation_vs_flow"] = (
        merged["FLOW_curvature_violation_rate"]
        - merged["VTF_curvature_violation_rate"]
    )
    merged["delta_curvature_violation_vs_wo_kin"] = (
        merged["VT_curvature_violation_rate"]
        - merged["VTF_curvature_violation_rate"]
    )
    merged["delta_smoothness_vs_flow"] = (
        merged["FLOW_smoothness_m"] - merged["VTF_smoothness_m"]
    )
    merged["delta_minade_ours_minus_flow"] = (
        merged["VTF_minADE@K_m"] - merged["FLOW_minADE@K_m"]
    )
    improvements = [
        "delta_tvk_vs_flow",
        "delta_tvk_vs_wo_kin",
        "delta_terrain_violation_vs_flow",
        "delta_curvature_violation_vs_flow",
        "delta_curvature_violation_vs_wo_kin",
        "delta_smoothness_vs_flow",
    ]
    ranks = merged[improvements].rank(pct=True)
    fidelity_penalty = merged["delta_minade_ours_minus_flow"].clip(lower=0.0)
    merged["balanced_score"] = ranks.mean(axis=1) - 2.0 * fidelity_penalty
    return merged


def select_scenes(
    frame: pd.DataFrame,
    *,
    minimum_frame_separation: int,
) -> list[dict[str, Any]]:
    """Select four distinct advantage mechanisms using declared rules."""

    eligible = frame[
        (frame["delta_tvk_vs_flow"] > 0.0)
        & (frame["delta_tvk_vs_wo_kin"] > 0.0)
        & (frame["delta_minade_ours_minus_flow"] <= 0.03)
        & (frame["VTF_minADE@K_m"] <= 0.25)
        & (frame["VTF_path_length_m"] >= 2.0)
    ].copy()
    rules = (
        (
            "terrain",
            "delta_terrain_violation_vs_flow",
            eligible[eligible["VTF_path_length_m"] >= 4.0],
        ),
        (
            "kinematic",
            "kinematic_score",
            eligible.assign(
                kinematic_score=(
                    eligible["delta_curvature_violation_vs_flow"]
                    + eligible["delta_curvature_violation_vs_wo_kin"]
                )
            ),
        ),
        (
            "smoothness",
            "delta_smoothness_vs_flow",
            eligible[
                (eligible["VTF_path_length_m"] >= 4.0)
                & (eligible["delta_curvature_violation_vs_flow"] >= 0.0)
            ],
        ),
        ("balanced", "balanced_score", eligible),
    )
    selected: list[dict[str, Any]] = []
    for category, score_name, candidates in rules:
        for scene_id, row in candidates.sort_values(score_name, ascending=False).iterrows():
            sequence = str(int(row["VTF_sequence"])).zfill(5)
            frame_id = int(row["VTF_frame_id"])
            if any(
                record["sequence"] == sequence
                and abs(record["frame_id"] - frame_id) < minimum_frame_separation
                for record in selected
            ):
                continue
            selected.append(
                {
                    "category": category,
                    "selection_metric": score_name,
                    "scene_id": str(scene_id),
                    "sequence": sequence,
                    "frame_id": frame_id,
                    "dataset_index": int(row["VTF_dataset_index"]),
                    "cross_seed_delta_tvk_vs_flow": float(row["delta_tvk_vs_flow"]),
                    "cross_seed_delta_tvk_vs_wo_kin": float(row["delta_tvk_vs_wo_kin"]),
                    "cross_seed_delta_terrain_violation_vs_flow": float(
                        row["delta_terrain_violation_vs_flow"]
                    ),
                    "cross_seed_delta_curvature_violation_vs_flow": float(
                        row["delta_curvature_violation_vs_flow"]
                    ),
                    "cross_seed_delta_smoothness_vs_flow": float(
                        row["delta_smoothness_vs_flow"]
                    ),
                    "cross_seed_delta_minade_ours_minus_flow": float(
                        row["delta_minade_ours_minus_flow"]
                    ),
                }
            )
            break
    if len(selected) != 4:
        raise RuntimeError(f"selected {len(selected)} scenes instead of four")
    return selected


def _load_predictions(benchmark_root: Path) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    trajectories: dict[str, np.ndarray] = {}
    ground_truth: np.ndarray | None = None
    scene_ids: list[str] | None = None
    for method in METHODS:
        run = benchmark_root / "runs" / f"{method}_seed0"
        with np.load(run / "predictions.npz", allow_pickle=False) as archive:
            trajectories[method] = archive["trajectories"].copy()
            current_gt = archive["ground_truth"].copy()
        current_ids = pd.read_csv(run / "scene_level_metrics.csv")["scene_id"].astype(str).tolist()
        if ground_truth is None:
            ground_truth = current_gt
            scene_ids = current_ids
        elif not np.array_equal(current_gt, ground_truth) or current_ids != scene_ids:
            raise ValueError("prediction archives are not scene-aligned")
    assert ground_truth is not None and scene_ids is not None
    return trajectories, ground_truth, scene_ids


def _terrain_cost_map(
    terrain_map: torch.Tensor,
    config: TerrainFieldConfig,
    *,
    samples: int = 160,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    field = AnalyticTerrainField(terrain_map.unsqueeze(0), config)
    x = torch.linspace(0.0, config.forward_m, samples)
    y = torch.linspace(-config.lateral_m, config.lateral_m, samples)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    query = torch.stack((xx, yy), dim=-1).unsqueeze(0)
    with torch.no_grad():
        cost = field.cost(query)[0].cpu().numpy().T
    return cost, (0.0, config.forward_m, -config.lateral_m, config.lateral_m)


def _best_candidate(trajectories: np.ndarray, ground_truth: np.ndarray) -> int:
    ade = np.linalg.norm(trajectories - ground_truth[None, ...], axis=-1).mean(axis=-1)
    return int(np.argmin(ade))


def _axis_limits(paths: list[np.ndarray]) -> tuple[tuple[float, float], tuple[float, float]]:
    xy = np.concatenate([np.zeros((1, 2))] + [path[..., :2].reshape(-1, 2) for path in paths])
    x_max = min(24.0, max(3.5, float(np.nanmax(xy[:, 0])) + 0.55))
    y_min = max(-12.0, float(np.nanmin(xy[:, 1])) - 0.55)
    y_max = min(12.0, float(np.nanmax(xy[:, 1])) + 0.55)
    if y_max - y_min < 2.4:
        center = 0.5 * (y_min + y_max)
        y_min, y_max = center - 1.2, center + 1.2
    return (0.0, x_max), (y_min, y_max)


def _save_all(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", transparent=True)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", transparent=True)
    fig.savefig(base.with_suffix(".png"), dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(
        base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def render(
    benchmark_root: Path,
    cache_root: Path,
    selected: list[dict[str, Any]],
) -> Path:
    protocol = _load_json(benchmark_root / "effective_protocol.json")
    effective = _load_json(
        benchmark_root / "checkpoints" / "seed_0" / "flow_tvk" / "effective_config.json"
    )
    terrain_config = TerrainFieldConfig(**effective["terrain_field"])
    dataset = CombinedSceneDataset(cache_root, tuple(protocol["protocol"]["source_splits"]))
    predictions, ground_truth, scene_ids = _load_predictions(benchmark_root)
    scene_lookup = {scene_id: index for index, scene_id in enumerate(scene_ids)}

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.2,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.fontsize": 6.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(7.2, 6.0), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=(0.13, 1.0, 1.0))
    legend_axis = fig.add_subplot(grid[0, :])
    legend_axis.axis("off")
    axes = np.asarray(
        [
            [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])],
            [fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[2, 1])],
        ]
    )
    images = []
    source_rows: list[dict[str, Any]] = []
    for panel_index, (axis, record) in enumerate(zip(axes.flat, selected)):
        scene_id = record["scene_id"]
        prediction_index = scene_lookup[scene_id]
        scene = dataset[int(record["dataset_index"])]
        cost_map, extent = _terrain_cost_map(scene.terrain_map.float(), terrain_config)
        image = axis.imshow(
            cost_map,
            origin="lower",
            extent=extent,
            cmap="YlOrBr",
            vmin=0.0,
            vmax=1.0,
            interpolation="bilinear",
            alpha=0.92,
            aspect="equal",
        )
        images.append(image)
        gt = ground_truth[prediction_index]
        all_paths = [gt]
        for method in METHODS:
            candidates = predictions[method][prediction_index]
            best = _best_candidate(candidates, gt)
            all_paths.extend(list(candidates))
            for candidate_index, candidate in enumerate(candidates):
                is_best = candidate_index == best
                axis.plot(
                    candidate[:, 0],
                    candidate[:, 1],
                    color=COLORS[method],
                    linewidth=1.65 if is_best else 0.65,
                    alpha=0.98 if is_best else 0.13,
                    zorder=5 if is_best else 3,
                )
            source_rows.extend(
                {
                    "scene_id": scene_id,
                    "method": DISPLAY_NAMES[method],
                    "candidate": candidate_index,
                    "is_minade_candidate": candidate_index == best,
                    "step": step,
                    "x_m": float(candidate[step, 0]),
                    "y_m": float(candidate[step, 1]),
                    "z_m": float(candidate[step, 2]),
                }
                for candidate_index, candidate in enumerate(candidates)
                for step in range(candidate.shape[0])
            )
        axis.plot(gt[:, 0], gt[:, 1], color="#171A1F", linewidth=2.0, zorder=7)
        axis.scatter(0.0, 0.0, marker="*", s=62, color="#D62F2F", edgecolor="white", linewidth=0.7, zorder=9)
        axis.scatter(
            gt[-1, 0], gt[-1, 1], marker="o", s=48, facecolor="white",
            edgecolor="#2155A6", linewidth=1.3, zorder=8,
        )
        axis.scatter(gt[-1, 0], gt[-1, 1], marker="+", s=58, color="#2155A6", linewidth=1.0, zorder=9)
        xlim, ylim = _axis_limits(all_paths)
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        axis.set_xlabel("Ego-forward x (m)")
        axis.set_ylabel("Ego-left y (m)")
        axis.grid(color="white", linewidth=0.35, alpha=0.35)
        letter = chr(ord("a") + panel_index)
        axis.set_title(
            f"{letter}  {CATEGORY_TITLES[record['category']]}\n"
            f"sequence {record['sequence']}, frame {record['frame_id']:06d}",
            loc="left",
            fontweight="bold",
        )
        annotation = (
            f"Reduction vs Flow: TVK cost {record['cross_seed_delta_tvk_vs_flow']:.3f}"
            "\n"
            rf"terrain {record['cross_seed_delta_terrain_violation_vs_flow'] * 100.0:.1f} pp; "
            rf"curvature {record['cross_seed_delta_curvature_violation_vs_flow'] * 100.0:.1f} pp"
        )
        axis.text(
            0.02, 0.02, annotation, transform=axis.transAxes, ha="left", va="bottom",
            fontsize=6.4, color="#20252B",
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
            zorder=10,
        )

    legend_handles = [
        Line2D([0], [0], color="#171A1F", lw=2.0, label="GT trajectory"),
        Line2D([0], [0], color=COLORS["FLOW"], lw=1.7, label=DISPLAY_NAMES["FLOW"]),
        Line2D([0], [0], color=COLORS["VT"], lw=1.7, label=DISPLAY_NAMES["VT"]),
        Line2D([0], [0], color=COLORS["VTF"], lw=1.7, label=DISPLAY_NAMES["VTF"]),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#D62F2F", markeredgecolor="white", markersize=8, label="Ego origin"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#2155A6", markersize=6, label="5 s goal"),
    ]
    legend_axis.legend(
        handles=legend_handles,
        loc="center",
        ncol=3,
        frameon=False,
        columnspacing=1.1,
        handlelength=2.2,
    )
    colorbar = fig.colorbar(images[0], ax=axes, fraction=0.025, pad=0.018, shrink=0.88)
    colorbar.set_label(r"Derived terrain cost $C_T$ (low $\rightarrow$ high)")

    source_dir = benchmark_root / "figure_source_data" / "selected_advantage_scenes"
    source_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(source_rows).to_csv(source_dir / "trajectory_coordinates.csv", index=False)
    base = benchmark_root / "figures" / "figure_selected_advantage_scenes"
    _save_all(fig, base)
    plt.close(fig)
    return base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--minimum-frame-separation", type=int, default=150)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark_root = args.benchmark_root.resolve()
    selection = build_selection_frame(benchmark_root)
    selected = select_scenes(
        selection, minimum_frame_separation=args.minimum_frame_separation
    )
    source_dir = benchmark_root / "figure_source_data" / "selected_advantage_scenes"
    source_dir.mkdir(parents=True, exist_ok=True)
    selection.to_csv(source_dir / "all_scene_selection_metrics.csv")
    pd.DataFrame(selected).to_csv(source_dir / "selected_scenes.csv", index=False)
    manifest = {
        "selection_population": 1909,
        "selection_statistics": "cross-seed means over seeds 0, 1, and 2",
        "eligibility": (
            "VTF-Flow improves TVK cost versus both Flow and the no-kinematics "
            "ablation, minADE degradation versus Flow is <= 0.03 m, absolute "
            "VTF-Flow minADE is <= 0.25 m, and path length >= 2 m"
        ),
        "categories": list(CATEGORY_TITLES),
        "minimum_frame_separation": args.minimum_frame_separation,
        "displayed_predictions": "frozen seed-0 K=8 predictions; all candidates shown",
        "emphasized_candidate": "minimum-ADE member, used only for qualitative best-of-K diagnosis",
        "background": "derived terrain cost from the exact cached planner-used BEV",
        "selected_scenes": selected,
    }
    (source_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    base = render(benchmark_root, args.cache_root.resolve(), selected)
    caption = (
        "Representative held-out RELLIS-3D scenes selected by fixed cross-seed criteria. "
        "The background is the derived planner terrain cost. Thin curves show all eight "
        "samples and thick curves identify the minimum-ADE sample for visualization only. "
        "Panel annotations report cross-seed scene-level changes of VTF-Flow relative to "
        "the Flow baseline; positive reductions indicate lower cost or violation. These "
        "examples illustrate mechanisms and do not replace the 1909-scene aggregate results."
    )
    (benchmark_root / "figures" / "figure_selected_advantage_scenes_caption.txt").write_text(
        caption + "\n", encoding="utf-8"
    )
    qa_note = """# Qualitative figure QA

- Core conclusion: unified TVK guidance can reduce terrain or kinematic diagnostics while retaining bounded trajectory error in representative held-out scenes.
- Archetype: image plate + quantitative annotations; Python/matplotlib only.
- Final size: 182.9 mm wide; SVG/PDF editable text; TIFF at 600 dpi.
- Panels a--d: one held-out scene per declared mechanism; all K=8 candidates are visible and the minimum-ADE member is emphasized.
- Selection: all 1909 test scenes were scored using cross-seed means; no scene was selected by visual inspection.
- Uncertainty: annotations are cross-seed scene means without error bars because each panel is a spatial scene example. Seed variability and paired uncertainty belong to the aggregate table/statistical results.
- Image integrity: the terrain background is queried from the unchanged cached planner BEV using the shared benchmark terrain-cost implementation; no local image enhancement or selective terrain editing was applied.
- Reviewer boundary: the selected panels are qualitative mechanism illustrations, not population-level evidence and not calibrated safety demonstrations.
"""
    (benchmark_root / "figures" / "figure_selected_advantage_scenes_qa.md").write_text(
        qa_note, encoding="utf-8"
    )
    print(json.dumps({"figure_base": str(base), "selected_scenes": selected}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

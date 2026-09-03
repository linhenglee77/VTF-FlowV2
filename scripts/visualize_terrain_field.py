"""Visualize a serialized terrain field and an aligned GT trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def load_trajectory(path: Path) -> np.ndarray:
    """Load ``[H, >=2]`` GT coordinates from NPY or a named NPZ array."""

    if path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            keys = ("gt_trajectory", "gt_future", "gt", "trajectory")
            key = next((candidate for candidate in keys if candidate in archive), None)
            if key is None:
                raise KeyError(f"trajectory archive must contain one of {keys}")
            value = archive[key]
    else:
        raise ValueError("GT trajectory must be .npy or .npz")
    trajectory = np.asarray(value, dtype=np.float32)
    if trajectory.ndim == 3 and trajectory.shape[0] == 1:
        trajectory = trajectory[0]
    if trajectory.ndim != 2 or trajectory.shape[1] < 2:
        raise ValueError("GT trajectory must have shape [H, >=2]")
    return trajectory


def _overlay(axis: plt.Axes, trajectory: np.ndarray, hero: bool = False) -> None:
    line, = axis.plot(
        trajectory[:, 0], trajectory[:, 1], color="#111827", lw=2.0 if hero else 1.3,
        label="GT trajectory", zorder=10,
    )
    line.set_path_effects([path_effects.Stroke(linewidth=3.4 if hero else 2.4, foreground="white"), path_effects.Normal()])
    axis.scatter([0.0], [0.0], marker="*", s=55 if hero else 28, color="#d62728", edgecolor="white", linewidth=0.6, zorder=11)


def _image(
    axis: plt.Axes,
    values: np.ndarray,
    extent: tuple[float, float, float, float],
    title: str,
    cmap: str,
    trajectory: np.ndarray,
    label: str,
    mask: np.ndarray | None = None,
) -> None:
    shown = np.ma.array(values, mask=mask) if mask is not None else values
    image = axis.imshow(shown, origin="lower", extent=extent, cmap=cmap, interpolation="nearest", aspect="equal")
    _overlay(axis, trajectory)
    axis.set_title(title, fontweight="bold")
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    colorbar.set_label(label)


def render(field_path: Path, trajectory_path: Path, output: Path) -> list[Path]:
    """Render the hero feasibility panel and six source feature panels."""

    trajectory = load_trajectory(trajectory_path)
    with np.load(field_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    extent = (
        float(data["x_min_m"]), float(data["x_max_m"]),
        float(data["y_min_m"]), float(data["y_max_m"]),
    )
    geometry_mask = ~data["geometry_valid"].astype(bool)
    slope_mask = ~data["slope_valid"].astype(bool)
    semantic_values = data["semantic_class"].astype(np.int64)
    semantic_mask = ~data["semantic_valid"].astype(bool)
    policy = json.loads(str(data["semantic_policy_json"]))
    encountered = sorted(int(value) for value in np.unique(semantic_values[~semantic_mask]))
    display = np.full_like(semantic_values, np.nan, dtype=np.float32)
    for index, label_id in enumerate(encountered):
        display[semantic_values == label_id] = index
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, max(len(encountered), 1)))
    semantic_cmap = ListedColormap(colors)
    semantic_norm = BoundaryNorm(np.arange(-0.5, len(encountered) + 0.5), semantic_cmap.N)

    figure = plt.figure(figsize=(7.2, 5.2), constrained_layout=True)
    layout = figure.add_gridspec(3, 4, width_ratios=(1.4, 1.4, 1.0, 1.0))
    hero = figure.add_subplot(layout[:, :2])
    support = [figure.add_subplot(layout[row, col]) for row in range(3) for col in range(2, 4)]

    feasibility_image = hero.imshow(
        data["feasibility"], origin="lower", extent=extent, cmap="viridis",
        vmin=0.0, vmax=1.0, interpolation="nearest", aspect="equal",
    )
    _overlay(hero, trajectory, hero=True)
    hero.set_title("Continuous terrain feasibility", fontweight="bold", fontsize=10)
    hero.set_xlabel("Local x (m)")
    hero.set_ylabel("Local y (m)")
    hero.legend(loc="upper left")
    colorbar = figure.colorbar(feasibility_image, ax=hero, fraction=0.035, pad=0.02)
    colorbar.set_label("F(x,y), higher is more feasible")

    _image(support[0], data["elevation_m"], extent, "Elevation", "terrain", trajectory, "m", geometry_mask)
    _image(support[1], data["slope_deg"], extent, "Local slope", "magma", trajectory, "degrees", slope_mask)
    _image(support[2], data["roughness_m"], extent, "Height roughness", "cividis", trajectory, "m (standard deviation)", geometry_mask)

    semantic_image = support[3].imshow(
        np.ma.array(display, mask=semantic_mask), origin="lower", extent=extent,
        cmap=semantic_cmap, norm=semantic_norm, interpolation="nearest", aspect="equal",
    )
    _overlay(support[3], trajectory)
    support[3].set_title("Dominant semantic class", fontweight="bold")
    if encountered:
        semantic_bar = figure.colorbar(semantic_image, ax=support[3], fraction=0.046, pad=0.03, ticks=np.arange(len(encountered)))
        semantic_bar.ax.set_yticklabels([policy.get(str(value), {}).get("name", str(value)) for value in encountered])
    _image(support[4], data["occupancy"], extent, "Obstacle occupancy", "gray_r", trajectory, "occupied", None)
    _image(support[5], data["clearance_m"], extent, "Obstacle clearance", "Blues", trajectory, "m", None)
    for axis in support:
        axis.set_xlabel("Local x (m)")
        axis.set_ylabel("Local y (m)")

    sequence = str(data.get("sequence", "?"))
    frame = int(data.get("frame", -1))
    status = str(data.get("coordinate_status", "unknown"))
    figure.suptitle(
        f"RELLIS-3D terrain field | sequence {sequence}, frame {frame:06d}\n"
        f"Coordinate status: {status}",
        fontsize=10,
    )
    output = output.with_suffix("")
    output.parent.mkdir(parents=True, exist_ok=True)
    png_path = output.with_suffix(".png")
    tiff_path = output.with_suffix(".tiff")
    svg_path = output.with_suffix(".svg")
    pdf_path = output.with_suffix(".pdf")
    figure.savefig(png_path, dpi=600, bbox_inches="tight")
    figure.savefig(tiff_path, dpi=600, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    generated = [png_path, tiff_path, svg_path, pdf_path]
    plt.close(figure)
    return generated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/terrain_fields/terrain_field_visualization"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generated = render(args.field.resolve(), args.trajectory.resolve(), args.output.resolve())
    print("Generated:")
    for path in generated:
        print(f"  {path}")


if __name__ == "__main__":
    main()

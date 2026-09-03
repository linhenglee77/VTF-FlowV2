"""Plot the planner-used static terrain-potential components for Results Sec. 5.1.

The input NPZ must contain the frozen arrays exported by
``render_real_method_modules.py``.  No component is recomputed here: this
script only applies the coordinate orientation and bilinear display used by
the planner's continuous queries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE: Final[Path] = (
    ROOT
    / "outputs"
    / "unified_h10_benchmark"
    / "figures"
    / "method_framework_real"
    / "source_data"
    / "static_terrain_components.npz"
)
DEFAULT_SOURCE_MANIFEST: Final[Path] = DEFAULT_SOURCE.with_name("manifest.json")
DEFAULT_OUTPUT: Final[Path] = (
    ROOT / "outputs" / "experiments" / "field_guidance_validation_figures"
)
EXTENT: Final[tuple[float, float, float, float]] = (0.0, 24.0, -12.0, 12.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render real planner-used static potential components."
    )
    parser.add_argument("--source-npz", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    required = (
        "occupancy",
        "nontraversable",
        "slope",
        "roughness",
        "clearance",
        "static_terrain_cost",
    )
    with np.load(path) as archive:
        missing = [key for key in required if key not in archive.files]
        if missing:
            raise KeyError(f"missing required potential arrays: {missing}")
        arrays = {key: np.asarray(archive[key], dtype=np.float32) for key in required}

    for key, values in arrays.items():
        if values.ndim != 2:
            raise ValueError(f"{key} must be two-dimensional, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{key} contains non-finite values")
        if float(values.min()) < -1e-6 or float(values.max()) > 1.0 + 1e-6:
            raise ValueError(f"{key} must be normalized to [0, 1]")
    return arrays


def _load_scene_metadata(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"sequence": "unresolved", "frame_id": "unresolved"}
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return {
        "sequence": str(manifest.get("sequence", "unresolved")),
        "frame_id": manifest.get("frame_id", "unresolved"),
        "scene_id": str(manifest.get("scene_id", "unresolved")),
    }


def _save_figure(fig: plt.Figure, output_dir: Path) -> list[Path]:
    stem = output_dir / "static_potential_component_fields"
    paths = [stem.with_suffix(suffix) for suffix in (".png", ".pdf", ".svg", ".tiff")]
    common = {"bbox_inches": "tight", "pad_inches": 0.025, "transparent": False}
    fig.savefig(paths[0], dpi=600, **common)
    fig.savefig(paths[1], **common)
    fig.savefig(paths[2], **common)
    fig.savefig(paths[3], dpi=600, pil_kwargs={"compression": "tiff_lzw"}, **common)
    return paths


def _render(
    arrays: dict[str, np.ndarray], metadata: dict[str, object], output_dir: Path
) -> list[Path]:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.6,
            "axes.titlesize": 8.4,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )

    panels = (
        ("occupancy", r"Occupancy potential $C_o$"),
        ("nontraversable", r"Non-traversability potential $C_{nt}$"),
        ("slope", r"Slope potential $C_s$"),
        ("roughness", r"Roughness potential $C_r$"),
        ("clearance", r"Obstacle-proximity potential $C_d$"),
        ("static_terrain_cost", r"Unified static potential $C_T$"),
    )
    labels = ("a", "b", "c", "d", "e", "f")

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(7.2047, 4.5669),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")
    common_image = None
    for axis, (key, title), label in zip(axes.flat, panels, labels):
        # Component tensors are stored as [forward, lateral]; imshow expects
        # [vertical, horizontal].  The sampled static cost is already [y, x].
        image_values = arrays[key] if key == "static_terrain_cost" else arrays[key].T
        common_image = axis.imshow(
            image_values,
            origin="lower",
            extent=EXTENT,
            cmap="magma_r",
            vmin=0.0,
            vmax=1.0,
            interpolation="bilinear",
            aspect="equal",
            rasterized=True,
        )
        axis.set_title(title, pad=4.0)
        axis.text(
            -0.13,
            1.06,
            label,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
        )
        axis.set_xlim(EXTENT[0], EXTENT[1])
        axis.set_ylim(EXTENT[2], EXTENT[3])
        axis.set_xticks((0.0, 12.0, 24.0))
        axis.set_yticks((-12.0, 0.0, 12.0))
        axis.grid(False)

    for axis in axes[:, 0]:
        axis.set_ylabel("Ego-left y (m)")
    for axis in axes[-1, :]:
        axis.set_xlabel("Ego-forward x (m)")

    if common_image is None:
        raise RuntimeError("no panels were rendered")
    colorbar = fig.colorbar(
        common_image,
        ax=axes,
        location="right",
        fraction=0.028,
        pad=0.025,
        ticks=(0.0, 0.5, 1.0),
    )
    colorbar.set_label(r"Normalized potential (low $\rightarrow$ high)", labelpad=5.0)
    colorbar.outline.set_linewidth(0.65)

    frame_id = metadata["frame_id"]
    frame_text = f"{int(frame_id):06d}" if isinstance(frame_id, int) else str(frame_id)
    fig.suptitle(
        "Planner-used static terrain-feasibility potentials | "
        f"RELLIS-3D {metadata['sequence']}, frame {frame_text}",
        fontsize=9.4,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        -0.018,
        "Fixed planning-ego fields; higher values denote larger local potential, "
        "not calibrated risk probability.",
        ha="center",
        va="top",
        fontsize=7.2,
        color="#4A4A4A",
    )

    paths = _save_figure(fig, output_dir)
    plt.close(fig)
    return paths


def _write_provenance(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, object],
    source_path: Path,
    output_dir: Path,
) -> None:
    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(source_dir / "static_potential_component_fields.npz", **arrays)
    manifest = {
        "figure": "static_potential_component_fields",
        "purpose": "Results Sec. 5.1 qualitative decomposition of fixed spatial potentials",
        "source_npz": str(source_path.resolve()),
        "scene": metadata,
        "extent_m": {
            "x_min": EXTENT[0],
            "x_max": EXTENT[1],
            "y_min": EXTENT[2],
            "y_max": EXTENT[3],
        },
        "display": {
            "normalization": "fixed [0, 1] for every panel",
            "interpolation": "bilinear, matching continuous planner queries",
            "component_orientation": "stored [forward, lateral], transposed for plotting",
            "static_cost_orientation": "stored [lateral, forward] sampled map",
        },
        "scope_boundary": (
            "Only fixed spatial terrain components are shown. Vehicle-state, curvature, "
            "and lateral-acceleration terms are candidate-dependent and are not fixed 2D fields."
        ),
    }
    with (source_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    caption = (
        "图 2｜规划器使用的静态地形可行性势分解。a--e 分别为占据、非可通行性、"
        "坡度、粗糙度和障碍接近度势；f 为采用冻结配置融合得到的静态地形势 C_T。"
        "所有面板来自 RELLIS-3D sequence 00004/frame 001124，使用相同规划自车坐标、"
        "[0,1] 色阶和双线性显示。数值越高表示局部势越大，并非标定安全概率。车辆状态、"
        "曲率和横向加速度项依赖候选轨迹，因此不作为固定二维热力图显示。"
    )
    (output_dir / "figure_caption_zh.txt").write_text(caption, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays = _load_arrays(args.source_npz)
    metadata = _load_scene_metadata(args.source_manifest)
    paths = _render(arrays, metadata, args.output_dir)
    _write_provenance(arrays, metadata, args.source_npz, args.output_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

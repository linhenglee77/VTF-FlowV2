"""Randomly render timestamp-sampled RELLIS-3D ground-truth trajectories.

The plot axes preserve the audited pose matrices numerically. They are not
labeled forward/left because the local files do not verify those semantics.
Optional LiDAR uses raw point-cloud x/y and is visibly marked unverified because
the pose-to-LiDAR frame relationship has not been established.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_PARENT = PROJECT_ROOT.parent
if str(WORKSPACE_PARENT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_PARENT))

from TerraFlow.datasets.trajectory_builder import (  # noqa: E402
    EgoTrajectory,
    RellisSequence,
    RellisTrajectoryBuilder,
    TrajectoryBuilderConfig,
    TrajectoryConstructionError,
    load_rellis_sequence,
)
from TerraFlow.scripts.inspect_rellis3d import discover_sequence_directories  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "debug_gt_trajectories"


def load_lidar_xy(path: Path, max_points: int = 50_000) -> Optional[torch.Tensor]:
    """Load raw first-two LiDAR channels for an explicitly unverified overlay."""

    if not path.is_file():
        return None
    byte_size = path.stat().st_size
    if byte_size % 16:
        raise TrajectoryConstructionError(
            f"LiDAR file size is not float32x4 aligned: {path}"
        )
    values = torch.from_file(str(path), dtype=torch.float32, size=byte_size // 4)
    points = values.reshape(-1, 4)[:, :2]
    finite = torch.isfinite(points).all(dim=1)
    points = points[finite]
    if points.shape[0] > max_points:
        stride = max(1, points.shape[0] // max_points)
        points = points[::stride][:max_points]
    return points


def render_trajectory(
    trajectory: EgoTrajectory,
    sequence_id: str,
    output_path: Path,
    lidar_xy: Optional[torch.Tensor] = None,
) -> None:
    """Render origin, future xyz trajectory, and an optional raw LiDAR BEV."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xyz = trajectory.xyz.detach().cpu()
    origin = trajectory.current_origin.detach().cpu()
    figure, axis = plt.subplots(figsize=(7.2, 7.2), constrained_layout=True)

    if lidar_xy is not None and lidar_xy.numel():
        lidar = lidar_xy.detach().cpu()
        axis.scatter(
            lidar[:, 0],
            lidar[:, 1],
            s=0.25,
            c="#7f8c8d",
            alpha=0.22,
            linewidths=0,
            rasterized=True,
            label="raw LiDAR xy (alignment unverified)",
        )

    axis.plot(
        xyz[:, 0],
        xyz[:, 1],
        color="#1565c0",
        linewidth=2.2,
        marker="o",
        markersize=5,
        label="future GT samples",
        zorder=3,
    )
    scatter = axis.scatter(
        xyz[:, 0],
        xyz[:, 1],
        c=xyz[:, 2],
        cmap="viridis",
        s=42,
        edgecolors="white",
        linewidths=0.5,
        zorder=4,
    )
    for step, point in enumerate(xyz, start=1):
        axis.annotate(
            str(step),
            (float(point[0]), float(point[1])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="#0d47a1",
        )
    axis.scatter(
        [float(origin[0])],
        [float(origin[1])],
        marker="*",
        s=180,
        color="#d32f2f",
        edgecolors="white",
        linewidths=0.8,
        label="current ego origin",
        zorder=5,
    )

    all_xy = torch.cat((origin[:2].reshape(1, 2), xyz[:, :2]), dim=0)
    if lidar_xy is not None and lidar_xy.numel():
        lidar_for_limits = lidar_xy.detach().cpu()
        lower = torch.quantile(lidar_for_limits, 0.01, dim=0)
        upper = torch.quantile(lidar_for_limits, 0.99, dim=0)
        all_xy = torch.cat((all_xy, lower.reshape(1, 2), upper.reshape(1, 2)), dim=0)
    minimum = all_xy.min(dim=0).values
    maximum = all_xy.max(dim=0).values
    span = torch.clamp(maximum - minimum, min=4.0)
    margin = torch.maximum(span * 0.25, torch.tensor([2.0, 2.0]))
    axis.set_xlim(float(minimum[0] - margin[0]), float(maximum[0] + margin[0]))
    axis.set_ylim(float(minimum[1] - margin[1]), float(maximum[1] + margin[1]))
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    axis.axhline(0.0, color="black", linewidth=0.6, alpha=0.3)
    axis.axvline(0.0, color="black", linewidth=0.6, alpha=0.3)
    axis.set_xlabel("local pose coordinate 0 (axis semantics unresolved)")
    axis.set_ylabel("local pose coordinate 1 (axis semantics unresolved)")
    axis.set_title(
        f"RELLIS-3D {sequence_id}, frame {trajectory.current_frame_index:06d}\n"
        f"H={trajectory.xyz.shape[0]}, max speed={trajectory.validity.max_speed:.2f} "
        "pose-units/s"
    )
    axis.legend(loc="best", fontsize=8)
    colorbar = figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("local pose coordinate 2 (semantics unresolved)")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def collect_candidates(
    sequences: Sequence[RellisSequence], builder: RellisTrajectoryBuilder
) -> List[Tuple[RellisSequence, int]]:
    """Collect all valid-by-coverage current indices across sequences."""

    candidates: List[Tuple[RellisSequence, int]] = []
    for sequence in sequences:
        candidates.extend(
            (sequence, index)
            for index in builder.valid_current_indices(
                sequence.poses, sequence.timestamps
            )
        )
    return candidates


def generate_visualizations(
    data_root: Path,
    output_dir: Path,
    config: TrajectoryBuilderConfig,
    num_samples: int = 75,
    seed: int = 7,
    include_lidar: bool = False,
) -> List[Dict[str, object]]:
    """Randomly build and render valid trajectories, returning manifest rows."""

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    sequence_dirs = discover_sequence_directories(data_root.expanduser().resolve())
    if not sequence_dirs:
        raise TrajectoryConstructionError(f"no sequence payloads found under {data_root}")
    sequences = [load_rellis_sequence(path) for path in sequence_dirs]
    builder = RellisTrajectoryBuilder(config=config)
    candidates = collect_candidates(sequences, builder)
    if len(candidates) < num_samples:
        raise TrajectoryConstructionError(
            f"requested {num_samples} examples but only {len(candidates)} have coverage"
        )

    generator = random.Random(seed)
    generator.shuffle(candidates)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    rejected = 0
    for sequence, frame_index in candidates:
        if len(rows) >= num_samples:
            break
        try:
            trajectory = builder.build(
                sequence.poses, sequence.timestamps, current_index=frame_index
            )
        except TrajectoryConstructionError:
            rejected += 1
            continue

        lidar_xy = None
        if include_lidar:
            lidar_path = (
                sequence.sequence_dir
                / "os1_cloud_node_kitti_bin"
                / f"{frame_index:06d}.bin"
            )
            lidar_xy = load_lidar_xy(lidar_path)
        filename = f"{sequence.sequence_id}_{frame_index:06d}.png"
        render_trajectory(
            trajectory,
            sequence_id=sequence.sequence_id,
            output_path=output_dir / filename,
            lidar_xy=lidar_xy,
        )
        rows.append(
            {
                "sequence_id": sequence.sequence_id,
                "current_frame_index": frame_index,
                "current_timestamp_s": float(sequence.timestamps[frame_index]),
                "horizon_seconds": config.horizon_seconds,
                "sampling_interval_seconds": config.sampling_interval_seconds,
                "num_future_steps": config.num_future_steps,
                "interpolated": config.interpolate,
                "max_step_pose_units": trajectory.validity.max_step_distance,
                "max_speed_pose_units_per_s": trajectory.validity.max_speed,
                "lidar_overlay": include_lidar and lidar_xy is not None,
                "image": filename,
            }
        )

    if len(rows) < num_samples:
        raise TrajectoryConstructionError(
            f"only {len(rows)} of {num_samples} requested samples passed validity; "
            f"{rejected} candidates were rejected"
        )
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def build_argument_parser() -> argparse.ArgumentParser:
    """Create command-line arguments for sampling, validity, and rendering."""

    parser = argparse.ArgumentParser(
        description="Render random ego-centric future trajectories from RELLIS-3D poses."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-samples", type=int, default=75)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--horizon-seconds", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--min-future-frames", type=int, default=10)
    parser.add_argument("--no-interpolation", action="store_true")
    parser.add_argument("--max-speed", type=float, default=30.0)
    parser.add_argument("--max-teleport-distance", type=float, default=20.0)
    parser.add_argument(
        "--with-lidar",
        action="store_true",
        help=(
            "Overlay raw LiDAR xy. Alignment with pose-local axes is unverified "
            "and is marked as such in every plot."
        ),
    )
    return parser


def main() -> None:
    """Run random trajectory construction and rendering."""

    args = build_argument_parser().parse_args()
    config = TrajectoryBuilderConfig(
        horizon_seconds=args.horizon_seconds,
        sampling_interval_seconds=args.dt,
        min_future_frames=args.min_future_frames,
        interpolate=not args.no_interpolation,
        max_speed_mps=args.max_speed,
        max_teleport_distance=args.max_teleport_distance,
    )
    rows = generate_visualizations(
        data_root=args.data_root,
        output_dir=args.output_dir,
        config=config,
        num_samples=args.num_samples,
        seed=args.seed,
        include_lidar=args.with_lidar,
    )
    print(f"Rendered {len(rows)} trajectories to {args.output_dir.resolve()}")
    print(f"Manifest: {(args.output_dir / 'manifest.csv').resolve()}")
    print(
        "Coordinate warning: pose and optional LiDAR axis semantics remain "
        "unverified; no hidden axis correction was applied."
    )


if __name__ == "__main__":
    main()

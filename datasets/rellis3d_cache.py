"""Build and read the leakage-controlled RELLIS-3D trajectory cache.

This is the exact preprocessing path used by the sequence-holdout VTF-Flow
experiments.  It is intentionally kept inside the repository so a clean clone
does not depend on sibling development directories.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SEQUENCES = ("00000", "00001", "00002", "00003", "00004")
TRAVERSABLE_IDS = np.asarray([1, 3, 10, 23, 31, 33, 34], dtype=np.uint16)
CAMERA_RE = re.compile(r"frame(?P<frame>\d+)-(?P<sec>\d+)_(?P<msec>\d+)\.jpg$")


@dataclass(frozen=True)
class WindowRecord:
    """One cacheable trajectory window."""

    split: str
    sequence: str
    frame_index: int
    frame_id: int


class _FixedFrameTrajectoryBuilder:
    """Reproduce the fixed-frame trajectory construction used in the paper."""

    raw_to_ego = np.diag([-1.0, -1.0, 1.0, 1.0])

    def __init__(self, data_root: str | Path):
        root = Path(data_root).expanduser().resolve()
        self.root = next(
            (path for path in (root / "Rellis-3D", root) if (path / "00000").is_dir()),
            None,
        )
        if self.root is None:
            raise FileNotFoundError(
                f"Could not find Rellis-3D/00000 under {root}. "
                "Pass the processed dataset directory."
            )
        self._poses: dict[str, np.ndarray] = {}
        self._frame_ids: dict[str, np.ndarray] = {}

    def sequence_dir(self, sequence: str) -> Path:
        if sequence not in SEQUENCES:
            raise ValueError(f"Unknown RELLIS-3D sequence: {sequence}")
        return self.root / sequence

    def poses(self, sequence: str) -> np.ndarray:
        if sequence not in self._poses:
            pose_file = self.sequence_dir(sequence) / "poses.txt"
            raw = np.loadtxt(pose_file, dtype=np.float64)
            if raw.ndim != 2 or raw.shape[1] != 12:
                raise ValueError(f"Expected 12 columns in {pose_file}, got {raw.shape}")
            if not np.isfinite(raw).all():
                raise ValueError(f"Non-finite values found in {pose_file}")
            matrices = np.repeat(np.eye(4, dtype=np.float64)[None], len(raw), axis=0)
            matrices[:, :3, :] = raw.reshape(-1, 3, 4)
            self._poses[sequence] = matrices
        return self._poses[sequence]

    def frame_ids(self, sequence: str) -> np.ndarray:
        if sequence not in self._frame_ids:
            cloud_dir = self.sequence_dir(sequence) / "os1_cloud_node_kitti_bin"
            frame_ids = np.asarray(
                sorted(int(path.stem) for path in cloud_dir.glob("*.bin")), dtype=np.int64
            )
            if len(frame_ids) != len(self.poses(sequence)):
                raise ValueError(
                    f"Pose/cloud count mismatch for {sequence}: "
                    f"{len(self.poses(sequence))} poses vs {len(frame_ids)} clouds"
                )
            self._frame_ids[sequence] = frame_ids
        return self._frame_ids[sequence]

    def build_xyz(
        self,
        sequence: str,
        frame_index: int,
        horizon: int,
        stride: int,
    ) -> np.ndarray:
        poses = self.poses(sequence)
        last = frame_index + horizon
        if frame_index < 0 or last >= len(poses):
            raise IndexError(
                f"Invalid window for {sequence}: index={frame_index}, "
                f"horizon={horizon}, pose_count={len(poses)}"
            )
        indices = np.arange(frame_index, last + 1, stride, dtype=np.int64)
        if indices[-1] != last:
            indices = np.append(indices, last)
        relative_raw = np.einsum(
            "ij,njk->nik", np.linalg.inv(poses[frame_index]), poses[indices]
        )
        relative_ego = np.einsum(
            "ij,njk,kl->nil", self.raw_to_ego, relative_raw, self.raw_to_ego
        )
        xyz = relative_ego[:, :3, 3]
        if not np.allclose(xyz[0], 0.0, atol=1e-8):
            raise AssertionError(f"Ego-origin invariant failed: {xyz[0]}")
        return xyz

    def transform_points_to_ego(self, xyz_raw: np.ndarray) -> np.ndarray:
        xyz_raw = np.asarray(xyz_raw, dtype=np.float64)
        if xyz_raw.ndim != 2 or xyz_raw.shape[1] != 3:
            raise ValueError(f"Expected Nx3 points, got {xyz_raw.shape}")
        return xyz_raw @ self.raw_to_ego[:3, :3].T


def parse_official_splits(metadata_dir: Path) -> dict[tuple[str, int], str]:
    """Map ``(sequence, frame_id)`` to the official RELLIS-3D LiDAR split."""

    result: dict[tuple[str, int], str] = {}
    for split in ("train", "val", "test"):
        split_path = metadata_dir / f"pt_{split}.lst"
        for line in split_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cloud_rel = line.split()[0].replace("\\", "/")
            parts = cloud_rel.split("/")
            key = (parts[0], int(Path(parts[2]).stem))
            if key in result:
                raise ValueError(f"Duplicate official split assignment for {key}")
            result[key] = split
    return result


def leakage_controlled_windows(
    builder: _FixedFrameTrajectoryBuilder,
    split_map: dict[tuple[str, int], str],
    horizon: int,
    isolation: int,
) -> dict[str, list[WindowRecord]]:
    """Return windows whose used frames and isolation halo contain one split."""

    records: dict[str, list[WindowRecord]] = {"train": [], "val": [], "test": []}
    for sequence in SEQUENCES:
        frame_ids = builder.frame_ids(sequence)
        id_to_index = {int(frame_id): index for index, frame_id in enumerate(frame_ids)}
        for frame_id in frame_ids:
            frame_id_int = int(frame_id)
            split = split_map[(sequence, frame_id_int)]
            index = id_to_index[frame_id_int]
            if index + horizon >= len(frame_ids):
                continue
            used_ids = frame_ids[index : index + horizon + 1]
            if any(split_map.get((sequence, int(other))) != split for other in used_ids):
                continue
            low = max(0, index - isolation)
            high = min(len(frame_ids), index + horizon + isolation + 1)
            halo_ids = frame_ids[low:high]
            if any(split_map.get((sequence, int(other))) != split for other in halo_ids):
                continue
            records[split].append(WindowRecord(split, sequence, index, frame_id_int))
    return records


def semantic_bev(
    builder: _FixedFrameTrajectoryBuilder,
    sequence: str,
    frame_id: int,
    grid_size: int = 64,
    forward_m: float = 24.0,
    lateral_m: float = 12.0,
) -> np.ndarray:
    """Create uint8 traversability, obstacle-density, and height BEV channels."""

    sequence_dir = builder.sequence_dir(sequence)
    cloud_path = sequence_dir / "os1_cloud_node_kitti_bin" / f"{frame_id:06d}.bin"
    label_path = (
        sequence_dir
        / "os1_cloud_node_semantickitti_label_id"
        / f"{frame_id:06d}.label"
    )
    cloud = np.fromfile(cloud_path, dtype=np.float32).reshape(-1, 4)
    labels = np.fromfile(label_path, dtype=np.uint32) & 0xFFFF
    if len(cloud) != len(labels):
        raise ValueError(f"Point/label count mismatch for {sequence}/{frame_id:06d}")

    xyz = builder.transform_points_to_ego(cloud[:, :3]).astype(np.float32)
    valid = (
        (xyz[:, 0] >= 0.0)
        & (xyz[:, 0] < forward_m)
        & (xyz[:, 1] >= -lateral_m)
        & (xyz[:, 1] < lateral_m)
        & (xyz[:, 2] >= -3.0)
        & (xyz[:, 2] <= 2.0)
    )
    xyz = xyz[valid]
    labels = labels[valid].astype(np.uint16)
    row = np.minimum((xyz[:, 0] / forward_m * grid_size).astype(np.int64), grid_size - 1)
    col = np.minimum(
        ((xyz[:, 1] + lateral_m) / (2.0 * lateral_m) * grid_size).astype(np.int64),
        grid_size - 1,
    )
    cell = row * grid_size + col
    cells = grid_size * grid_size
    all_count = np.bincount(cell, minlength=cells).astype(np.float32)
    traversable = np.isin(labels, TRAVERSABLE_IDS)
    trav_count = np.bincount(cell[traversable], minlength=cells).astype(np.float32)
    obstacle = (~traversable) & (labels != 0) & (labels != 7)
    obs_count = np.bincount(cell[obstacle], minlength=cells).astype(np.float32)
    height_sum = np.bincount(cell, weights=xyz[:, 2], minlength=cells).astype(np.float32)

    trav_fraction = np.divide(
        trav_count, all_count, out=np.zeros_like(trav_count), where=all_count > 0
    )
    obs_density = 1.0 - np.exp(-obs_count / 3.0)
    mean_height = np.divide(
        height_sum, all_count, out=np.zeros_like(height_sum), where=all_count > 0
    )
    height_norm = np.where(
        all_count > 0, np.clip((mean_height + 2.5) / 4.5, 0.0, 1.0), 0.0
    )
    bev = np.stack([trav_fraction, obs_density, height_norm]).reshape(
        3, grid_size, grid_size
    )
    return np.rint(bev * 255.0).astype(np.uint8)


def build_cache(
    data_root: Path,
    output_dir: Path,
    horizon: int = 150,
    trajectory_stride: int = 5,
    isolation: int = 150,
    grid_size: int = 64,
) -> dict[str, object]:
    """Build the exact cached inputs used by the VTF-Flow experiments."""

    builder = _FixedFrameTrajectoryBuilder(data_root)
    metadata_dir = builder.root.parent / "metadata"
    split_map = parse_official_splits(metadata_dir)
    records = leakage_controlled_windows(builder, split_map, horizon, isolation)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "dataset_root": str(builder.root),
        "official_split": str(metadata_dir),
        "horizon_frames": horizon,
        "trajectory_stride": trajectory_stride,
        "trajectory_points_excluding_origin": horizon // trajectory_stride,
        "isolation_frames": isolation,
        "nominal_frame_rate_hz": 10.0,
        "bev_shape": [3, grid_size, grid_size],
        "bev_channels": ["traversable_fraction", "obstacle_density", "mean_height"],
        "planner_axes": "x forward, y left, z up",
        "normalization_scales_m": [24.0, 12.0, 3.0],
        "counts": {split: len(rows) for split, rows in records.items()},
    }

    for split, split_records in records.items():
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        sample_count = len(split_records)
        points = horizon // trajectory_stride
        bev_array = np.lib.format.open_memmap(
            split_dir / "bev.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(sample_count, 3, grid_size, grid_size),
        )
        trajectory_array = np.lib.format.open_memmap(
            split_dir / "trajectory.npy",
            mode="w+",
            dtype=np.float32,
            shape=(sample_count, points, 3),
        )
        goal_array = np.lib.format.open_memmap(
            split_dir / "goal.npy",
            mode="w+",
            dtype=np.float32,
            shape=(sample_count, 3),
        )
        manifest_rows: list[dict[str, object]] = []
        for row_index, record in enumerate(split_records):
            xyz = builder.build_xyz(
                record.sequence, record.frame_index, horizon, trajectory_stride
            )[1:].astype(np.float32)
            if len(xyz) != points:
                raise AssertionError(f"Unexpected trajectory length: {len(xyz)}")
            bev_array[row_index] = semantic_bev(
                builder, record.sequence, record.frame_id, grid_size=grid_size
            )
            trajectory_array[row_index] = xyz
            goal_array[row_index] = xyz[-1]
            manifest_rows.append(
                {
                    "row": row_index,
                    "split": split,
                    "sequence": record.sequence,
                    "frame_index": record.frame_index,
                    "frame_id": record.frame_id,
                    "goal_forward_m": float(xyz[-1, 0]),
                    "goal_lateral_m": float(xyz[-1, 1]),
                    "goal_elevation_m": float(xyz[-1, 2]),
                }
            )
            if (row_index + 1) % 250 == 0 or row_index + 1 == sample_count:
                print(f"[{split}] {row_index + 1}/{sample_count}", flush=True)
        bev_array.flush()
        trajectory_array.flush()
        goal_array.flush()
        with (split_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "row",
                "split",
                "sequence",
                "frame_index",
                "frame_id",
                "goal_forward_m",
                "goal_lateral_m",
                "goal_elevation_m",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)

    (output_dir / "dataset_config.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


class CachedTrajectoryDataset:
    """Memory-mapped reader for a generated trajectory cache."""

    def __init__(self, cache_root: Path, split: str):
        self.cache_root = Path(cache_root)
        self.split = split
        split_dir = self.cache_root / split
        self.bev = np.load(split_dir / "bev.npy", mmap_mode="r")
        self.trajectory = np.load(split_dir / "trajectory.npy", mmap_mode="r")
        self.goal = np.load(split_dir / "goal.npy", mmap_mode="r")
        config = json.loads(
            (self.cache_root / "dataset_config.json").read_text(encoding="utf-8")
        )
        self.scales = np.asarray(config["normalization_scales_m"], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.trajectory)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray(self.bev[index], dtype=np.float32) / 255.0,
            np.asarray(self.trajectory[index], dtype=np.float32) / self.scales,
            np.asarray(self.goal[index], dtype=np.float32) / self.scales,
        )

    def close(self) -> None:
        """Release memory-map file handles, including on Windows."""

        for array in (self.bev, self.trajectory, self.goal):
            memory_map = getattr(array, "_mmap", None)
            if memory_map is not None:
                memory_map.close()

"""Unit tests for timestamped ego-centric trajectory construction."""

from pathlib import Path
import struct
import tempfile
import unittest

import torch

from TerraFlow.datasets.trajectory_builder import (
    RellisTrajectoryBuilder,
    TrajectoryBuilderConfig,
    TrajectoryConstructionError,
    load_pose_matrices,
    load_rgb_timestamps,
    rellis3d_os1_to_planning_ego,
    relative_future_translations,
    validate_trajectory,
)
from TerraFlow.scripts.visualize_gt_trajectories import load_lidar_xy


def make_pose(x: float, y: float, z: float, yaw_radians: float = 0.0) -> torch.Tensor:
    """Create a float64 rigid transform for deterministic tests."""

    cosine = torch.cos(torch.tensor(yaw_radians, dtype=torch.float64))
    sine = torch.sin(torch.tensor(yaw_radians, dtype=torch.float64))
    pose = torch.eye(4, dtype=torch.float64)
    pose[:2, :2] = torch.tensor(
        [[cosine, -sine], [sine, cosine]], dtype=torch.float64
    )
    pose[:3, 3] = torch.tensor([x, y, z], dtype=torch.float64)
    return pose


class TrajectoryBuilderTest(unittest.TestCase):
    """Verify transform algebra, sampling, parsing, and validity checks."""

    def test_default_configuration_produces_ten_future_steps(self) -> None:
        config = TrajectoryBuilderConfig()
        self.assertEqual(config.horizon_seconds, 5.0)
        self.assertEqual(config.sampling_interval_seconds, 0.5)
        self.assertEqual(config.num_future_steps, 10)

    def test_relative_transform_uses_current_ego_rotation(self) -> None:
        current = make_pose(10.0, 0.0, 2.0, yaw_radians=torch.pi / 2)
        future = torch.stack(
            [
                make_pose(10.0, 1.0, 2.0, yaw_radians=torch.pi / 2),
                make_pose(10.0, 2.0, 3.0, yaw_radians=torch.pi / 2),
            ]
        )
        xyz, origin = relative_future_translations(current, future)
        expected = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 1.0]], dtype=torch.float64)
        self.assertTrue(torch.allclose(xyz, expected, atol=1e-12))
        self.assertTrue(torch.allclose(origin, torch.zeros(3, dtype=torch.float64)))

    def test_nonuniform_timestamps_are_interpolated(self) -> None:
        timestamps = torch.tensor([0.0, 0.4, 1.1, 1.7], dtype=torch.float64)
        poses = torch.stack([make_pose(2.0 * time, 0.0, 0.0) for time in timestamps])
        config = TrajectoryBuilderConfig(
            horizon_seconds=1.0,
            sampling_interval_seconds=0.5,
            min_future_frames=2,
            interpolate=True,
            max_speed_mps=5.0,
            max_teleport_distance=2.0,
        )
        result = RellisTrajectoryBuilder(config).build(poses, timestamps, current_index=0)
        expected = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64)
        self.assertTrue(torch.allclose(result.xyz, expected, atol=1e-12))
        self.assertEqual(result.xyz.shape, (2, 3))
        self.assertTrue(result.validity.is_valid)

    def test_nearest_sampling_can_disable_interpolation(self) -> None:
        timestamps = torch.tensor([0.0, 0.4, 1.1, 1.7], dtype=torch.float64)
        poses = torch.stack([make_pose(float(time), 0.0, 0.0) for time in timestamps])
        config = TrajectoryBuilderConfig(
            horizon_seconds=1.0,
            sampling_interval_seconds=0.5,
            min_future_frames=2,
            interpolate=False,
            max_speed_mps=5.0,
            max_teleport_distance=2.0,
        )
        result = RellisTrajectoryBuilder(config).build(poses, timestamps, current_index=0)
        expected = torch.tensor([[0.4, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=torch.float64)
        self.assertTrue(torch.allclose(result.xyz, expected))
        self.assertEqual(result.source_frame_indices.tolist(), [1, 2])

    def test_pose_convention_adapter_is_isolated_and_explicit(self) -> None:
        current = make_pose(0.0, 0.0, 0.0)
        future = torch.stack([make_pose(2.0, 3.0, 0.0)])

        def swap_translation_axes(poses: torch.Tensor) -> torch.Tensor:
            converted = poses.clone()
            converted[..., 0, 3] = poses[..., 1, 3]
            converted[..., 1, 3] = poses[..., 0, 3]
            return converted

        raw_xyz, _ = relative_future_translations(current, future)
        converted_xyz, _ = relative_future_translations(
            current, future, convention_adapter=swap_translation_axes
        )
        self.assertTrue(
            torch.equal(raw_xyz, torch.tensor([[2.0, 3.0, 0.0]], dtype=torch.float64))
        )
        self.assertTrue(
            torch.equal(
                converted_xyz, torch.tensor([[3.0, 2.0, 0.0]], dtype=torch.float64)
            )
        )

    def test_verified_rellis_adapter_maps_raw_negative_x_motion_forward(self) -> None:
        current = make_pose(0.0, 0.0, 0.0)
        future = torch.stack([make_pose(-2.0, -0.5, 0.25)])
        xyz, origin = relative_future_translations(
            current,
            future,
            convention_adapter=rellis3d_os1_to_planning_ego,
        )
        expected = torch.tensor([[2.0, 0.5, 0.25]], dtype=torch.float64)
        self.assertTrue(torch.equal(xyz, expected))
        self.assertTrue(torch.equal(origin, torch.zeros(3, dtype=torch.float64)))

    def test_non_monotonic_dataset_timestamps_are_rejected(self) -> None:
        poses = torch.stack([make_pose(0.0, 0.0, 0.0) for _ in range(3)])
        timestamps = torch.tensor([0.0, 0.5, 0.4], dtype=torch.float64)
        config = TrajectoryBuilderConfig(
            horizon_seconds=0.5,
            sampling_interval_seconds=0.5,
            min_future_frames=1,
        )
        with self.assertRaisesRegex(TrajectoryConstructionError, "strictly increasing"):
            RellisTrajectoryBuilder(config).build(poses, timestamps, current_index=0)

    def test_validity_detects_nonfinite_and_teleportation(self) -> None:
        config = TrajectoryBuilderConfig(
            horizon_seconds=1.0,
            sampling_interval_seconds=0.5,
            min_future_frames=1,
            max_speed_mps=10.0,
            max_teleport_distance=3.0,
        )
        timestamps = torch.tensor([0.0, 0.5, 1.0])
        teleport = validate_trajectory(
            torch.tensor([[1.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
            torch.zeros(3),
            timestamps,
            config,
        )
        nonfinite = validate_trajectory(
            torch.tensor([[1.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]]),
            torch.zeros(3),
            timestamps,
            config,
        )
        self.assertFalse(teleport.no_teleportation)
        self.assertFalse(teleport.speed_within_limit)
        self.assertFalse(nonfinite.finite)
        self.assertFalse(nonfinite.is_valid)

    def test_pose_and_rgb_timestamp_file_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pose_path = root / "poses.txt"
            pose_path.write_text(
                "1 0 0 1 0 1 0 2 0 0 1 3\n"
                "1 0 0 2 0 1 0 3 0 0 1 4\n",
                encoding="utf-8",
            )
            image_dir = root / "pylon_camera_node"
            image_dir.mkdir()
            (image_dir / "frame000000-10_250.jpg").write_bytes(b"")
            (image_dir / "frame000001-10_750.jpg").write_bytes(b"")

            poses = load_pose_matrices(pose_path)
            timestamps = load_rgb_timestamps(image_dir, expected_frames=2)
            self.assertEqual(poses.shape, (2, 4, 4))
            self.assertEqual(poses[0, :3, 3].tolist(), [1.0, 2.0, 3.0])
            self.assertTrue(torch.allclose(timestamps, torch.tensor([10.25, 10.75], dtype=torch.float64)))

    def test_optional_lidar_bev_loader_reads_float32x4(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cloud_path = Path(temp_dir) / "000000.bin"
            cloud_path.write_bytes(
                struct.pack("<8f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
            )
            xy = load_lidar_xy(cloud_path)
            self.assertIsNotNone(xy)
            assert xy is not None
            self.assertTrue(torch.equal(xy, torch.tensor([[1.0, 2.0], [5.0, 6.0]])))


if __name__ == "__main__":
    unittest.main()

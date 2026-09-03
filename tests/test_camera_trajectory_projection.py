"""Numerical tests for the camera-view trajectory projection."""

from __future__ import annotations

import unittest

import numpy as np

from TerraFlow.scripts.render_camera_trajectory_results import _project


class CameraTrajectoryProjectionTests(unittest.TestCase):
    def test_pinhole_projection_is_finite_and_centered(self) -> None:
        calibration = {
            "camera_from_ego": np.eye(4),
            "intrinsic": np.array(
                [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]
            ),
            "distortion": np.zeros(5),
            "image_width": 100,
            "image_height": 80,
            "lidar_height_m": 0.0,
        }
        trajectory = np.array([[0.0, 0.0, 5.0], [0.5, 0.0, 5.0]])
        pixels, visible, depth = _project(trajectory, calibration)
        np.testing.assert_allclose(pixels[0], [50.0, 40.0])
        np.testing.assert_allclose(pixels[1], [60.0, 40.0])
        self.assertTrue(visible.all())
        self.assertTrue(np.isfinite(depth).all())

    def test_points_behind_camera_are_not_visible(self) -> None:
        calibration = {
            "camera_from_ego": np.eye(4),
            "intrinsic": np.eye(3),
            "distortion": np.zeros(5),
            "image_width": 10,
            "image_height": 10,
            "lidar_height_m": 0.0,
        }
        _, visible, _ = _project(np.array([[0.0, 0.0, -1.0]]), calibration)
        self.assertFalse(bool(visible[0]))


if __name__ == "__main__":
    unittest.main()


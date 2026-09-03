"""Unit tests for the unified real-data figure construction helpers."""

from __future__ import annotations

import unittest

import numpy as np

from TerraFlow.scripts.render_unified_raw_terrain_scenes import (
    _minimum_ade_candidate,
    _planning_view_extent,
    _trajectory_geometry,
)


class UnifiedRawTerrainFigureTest(unittest.TestCase):
    """Verify deterministic candidate selection and planning-view geometry."""

    def test_minimum_ade_candidate_uses_all_waypoints_and_xyz(self) -> None:
        ground_truth = np.zeros((3, 3), dtype=np.float32)
        candidates = np.stack(
            (
                np.full((3, 3), 2.0, dtype=np.float32),
                np.full((3, 3), 0.25, dtype=np.float32),
                np.full((3, 3), 1.0, dtype=np.float32),
            )
        )
        self.assertEqual(_minimum_ade_candidate(candidates, ground_truth), 1)

    def test_minimum_ade_candidate_rejects_incompatible_shapes(self) -> None:
        with self.assertRaises(ValueError):
            _minimum_ade_candidate(
                np.zeros((2, 4, 3), dtype=np.float32),
                np.zeros((3, 3), dtype=np.float32),
            )

    def test_trajectory_geometry_includes_ego_origin_segment(self) -> None:
        ground_truth = np.array(
            [[[3.0, 4.0, 0.0], [6.0, 8.0, 0.0]]], dtype=np.float32
        )
        geometry = _trajectory_geometry(ground_truth)
        self.assertAlmostEqual(float(geometry["path_length_m"][0]), 10.0)
        self.assertAlmostEqual(float(geometry["final_x_m"][0]), 6.0)
        self.assertAlmostEqual(float(geometry["maximum_abs_y_m"][0]), 8.0)

    def test_planning_view_is_shared_bounded_and_not_smaller_than_eight_metres(self) -> None:
        full_extent = (-3.0, 24.0, -12.0, 12.0)
        paths = (
            np.array([[0.0, 0.0, 0.0], [10.0, 1.0, 0.0]], dtype=np.float32),
            np.array([[0.0, 0.0, 0.0], [12.0, -2.0, 0.0]], dtype=np.float32),
        )
        view = _planning_view_extent(full_extent, paths)
        self.assertGreaterEqual(view[0], full_extent[0])
        self.assertLessEqual(view[1], full_extent[1])
        self.assertGreaterEqual(view[2], full_extent[2])
        self.assertLessEqual(view[3], full_extent[3])
        self.assertGreaterEqual(view[1], 13.5)
        self.assertGreaterEqual(view[3] - view[2], 8.0)


if __name__ == "__main__":
    unittest.main()

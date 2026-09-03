"""Synthetic numerical tests for the geometry-semantic terrain field."""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from TerraFlow.terrain.feasibility_field import (
    ContinuousTerrainField,
    ContinuousTerrainFieldConfig,
    SemanticClassPolicy,
    load_terrain_field_config,
)
from TerraFlow.terrain.terrain_features import (
    TerrainFeatureConfig,
    TerrainFeatures,
    TerrainGridSpec,
    build_terrain_features,
    obstacle_clearance,
)


def make_features(
    occupancy: torch.Tensor | None = None,
    slope: torch.Tensor | None = None,
) -> TerrainFeatures:
    """Create a fully observed 2x2 grid with controlled costs."""

    grid = TerrainGridSpec(0.0, 2.0, 0.0, 2.0, 1.0)
    shape = (2, 2)
    zeros = torch.zeros(shape, dtype=torch.float32)
    valid = torch.ones(shape, dtype=torch.bool)
    return TerrainFeatures(
        grid=grid,
        elevation_m=zeros.clone(),
        slope_deg=zeros.clone() if slope is None else slope.float(),
        roughness_m=zeros.clone(),
        semantic_class=torch.full(shape, 1, dtype=torch.long),
        occupancy=zeros.clone() if occupancy is None else occupancy.float(),
        clearance_m=torch.full(shape, 10.0),
        point_count=torch.full(shape, 5, dtype=torch.long),
        geometry_valid=valid.clone(),
        slope_valid=valid.clone(),
        semantic_valid=valid.clone(),
    )


def cost_config(**overrides: float) -> ContinuousTerrainFieldConfig:
    values = {
        "occupancy_weight": 0.0,
        "slope_weight": 0.0,
        "roughness_weight": 0.0,
        "semantic_weight": 0.0,
        "clearance_weight": 0.0,
        "unknown_weight": 0.0,
        "slope_reference_deg": 20.0,
        "roughness_reference_m": 0.2,
        "clearance_reference_m": 2.0,
        "semantic_classes": {1: SemanticClassPolicy("dirt", 0.0, "ground", "test")},
    }
    values.update(overrides)
    return ContinuousTerrainFieldConfig(**values)


class TerrainFieldSyntheticTest(unittest.TestCase):
    def test_obstacle_has_lower_feasibility(self) -> None:
        occupancy = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
        field = ContinuousTerrainField(
            make_features(occupancy=occupancy), cost_config(occupancy_weight=3.0)
        )
        clear = field.query(torch.tensor([[0.5, 0.5]]))
        obstacle = field.query(torch.tensor([[1.5, 0.5]]))
        self.assertGreater(float(clear), float(obstacle))
        self.assertAlmostEqual(float(clear), 1.0, places=6)

    def test_steep_area_has_lower_feasibility(self) -> None:
        slope = torch.tensor([[0.0, 30.0], [0.0, 0.0]])
        field = ContinuousTerrainField(
            make_features(slope=slope), cost_config(slope_weight=2.0)
        )
        flat = field.query(torch.tensor([[0.5, 0.5]]))
        steep = field.query(torch.tensor([[1.5, 0.5]]))
        self.assertGreater(float(flat), float(steep))

    def test_query_is_bilinear_at_cell_centres(self) -> None:
        occupancy = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        field = ContinuousTerrainField(
            make_features(occupancy=occupancy), cost_config(occupancy_weight=1.0)
        )
        centre = field.query(torch.tensor([[1.0, 1.0]]))
        expected = 0.5 * (1.0 + math.exp(-1.0))
        self.assertAlmostEqual(float(centre), expected, places=6)
        xyz = torch.tensor([[1.0, 1.0, 8.0]], requires_grad=True)
        field.query(xyz).sum().backward()
        self.assertTrue(torch.isfinite(xyz.grad).all())

    def test_spatial_smoothing_spreads_cost_but_preserves_raw_occupancy(self) -> None:
        occupancy = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
        field = ContinuousTerrainField(
            make_features(occupancy=occupancy),
            cost_config(
                occupancy_weight=1.0,
                spatial_smoothing_sigma_m=0.5,
            ),
        )
        clear_xy = torch.tensor([[0.5, 0.5]])
        self.assertEqual(float(field.raw_component_costs(clear_xy)["occupancy"]), 0.0)
        self.assertGreater(float(field.component_costs(clear_xy)["occupancy"]), 0.0)
        self.assertAlmostEqual(float(field.components["occupancy"].sum()), 1.0, places=5)

    def test_point_aggregation_uses_geometry_and_exact_semantics(self) -> None:
        grid = TerrainGridSpec(0.0, 4.0, 0.0, 4.0, 1.0)
        config = TerrainFeatureConfig(
            grid=grid,
            minimum_points_per_cell=3,
            elevation_percentile=0.0,
            obstacle_height_threshold_m=0.5,
            minimum_obstacle_points=1,
            maximum_clearance_m=4.0,
            semantic_obstacle_ids=(9,),
        )
        points = np.asarray(
            [
                [1.1, 1.1, 0.0], [1.2, 1.1, 0.8], [1.1, 1.2, 1.0],
                [2.1, 1.1, 0.0], [2.2, 1.1, 0.0], [2.1, 1.2, 0.0],
            ],
            dtype=np.float32,
        )
        labels = np.asarray([9, 9, 1, 1, 1, 1], dtype=np.uint32)
        features = build_terrain_features(points, labels, config)
        self.assertEqual(int(features.semantic_class[1, 1]), 9)
        self.assertEqual(float(features.occupancy[1, 1]), 1.0)
        self.assertEqual(float(features.occupancy[1, 2]), 0.0)
        self.assertEqual(float(features.clearance_m[1, 1]), 0.0)
        self.assertGreater(float(features.clearance_m[1, 2]), 0.0)

    def test_float32_upper_boundary_never_indexes_past_grid(self) -> None:
        grid = TerrainGridSpec(0.0, 24.0, -12.0, 12.0, 0.25)
        almost_upper = np.nextafter(np.float32(12.0), np.float32(-np.inf))
        points = np.asarray(
            [[1.0, almost_upper, 0.0], [1.0, 12.0, 0.0]], dtype=np.float32
        )
        labels = np.asarray([1, 1], dtype=np.uint32)
        features = build_terrain_features(
            points,
            labels,
            TerrainFeatureConfig(grid=grid, minimum_points_per_cell=1),
        )
        self.assertEqual(features.point_count.shape, (96, 96))
        self.assertLessEqual(int(features.point_count.sum()), 1)

    def test_default_semantic_policy_is_documented(self) -> None:
        definition = load_terrain_field_config(
            __import__("pathlib").Path(__file__).parents[1]
            / "configs"
            / "rellis3d_terrain_field.json"
        )
        self.assertIn("not RELLIS-3D annotations", definition.semantic_provenance)
        self.assertTrue(definition.label_encoding.startswith("Exact uint32"))
        for policy in (definition.cost.semantic_classes or {}).values():
            self.assertTrue(policy.rationale)

    def test_clearance_is_zero_on_obstacle_and_increases_outward(self) -> None:
        occupancy = np.zeros((5, 5), dtype=bool)
        occupancy[2, 2] = True
        clearance = obstacle_clearance(occupancy, 1.0, 10.0)
        self.assertEqual(float(clearance[2, 2]), 0.0)
        self.assertLess(float(clearance[2, 3]), float(clearance[2, 4]))


if __name__ == "__main__":
    unittest.main()

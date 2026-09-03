"""Synthetic tests for continuous vehicle-state terrain conditioning."""

from __future__ import annotations

import math
import unittest

import torch

from TerraFlow.terrain.feasibility_field import (
    ContinuousTerrainField,
    ContinuousTerrainFieldConfig,
    SemanticClassPolicy,
)
from TerraFlow.terrain.terrain_features import TerrainFeatures, TerrainGridSpec
from TerraFlow.terrain.vehicle_conditioned_field import (
    TrajectoryGradientSmoothingConfig,
    VehicleConditionedFieldConfig,
    VehicleConditionedTerrainField,
    smooth_trajectory_gradient,
    trajectory_motion_state,
)


def make_base_field() -> ContinuousTerrainField:
    grid = TerrainGridSpec(0.0, 4.0, -1.0, 1.0, 1.0)
    shape = (2, 4)
    zeros = torch.zeros(shape)
    valid = torch.ones(shape, dtype=torch.bool)
    slope = torch.tensor([[0.0, 25.0, 25.0, 0.0], [0.0, 25.0, 25.0, 0.0]])
    roughness = torch.tensor([[0.0, 0.2, 0.2, 0.0], [0.0, 0.2, 0.2, 0.0]])
    clearance = torch.tensor([[5.0, 5.0, 0.0, 0.0], [5.0, 5.0, 0.0, 0.0]])
    features = TerrainFeatures(
        grid=grid,
        elevation_m=zeros.clone(),
        slope_deg=slope,
        roughness_m=roughness,
        semantic_class=torch.ones(shape, dtype=torch.long),
        occupancy=zeros.clone(),
        clearance_m=clearance,
        point_count=torch.full(shape, 5, dtype=torch.long),
        geometry_valid=valid.clone(),
        slope_valid=valid.clone(),
        semantic_valid=valid.clone(),
    )
    config = ContinuousTerrainFieldConfig(
        occupancy_weight=0.0,
        slope_weight=0.0,
        roughness_weight=0.0,
        semantic_weight=0.0,
        clearance_weight=0.0,
        unknown_weight=1.0,
        slope_reference_deg=25.0,
        roughness_reference_m=0.2,
        clearance_reference_m=2.0,
        semantic_classes={1: SemanticClassPolicy("dirt", 0.0, "ground", "test")},
    )
    return ContinuousTerrainField(features, config)


class VehicleConditionedFieldTest(unittest.TestCase):
    def test_temporal_gradient_smoothing_preserves_constant_and_reduces_impulse(self) -> None:
        config = TrajectoryGradientSmoothingConfig(sigma_waypoints=1.0)
        constant = torch.ones(2, 8, 3)
        self.assertTrue(torch.allclose(smooth_trajectory_gradient(constant, config), constant))
        impulse = torch.zeros(8, 2)
        impulse[4] = 1.0
        smoothed = smooth_trajectory_gradient(impulse, config)
        self.assertLess(float(smoothed.max()), 1.0)
        self.assertGreater(float(smoothed[3, 0]), 0.0)
        self.assertTrue(torch.isfinite(smoothed).all())

    def test_zero_speed_reduces_exactly_to_terrain_only(self) -> None:
        base = make_base_field()
        field = VehicleConditionedTerrainField(base)
        points = torch.tensor([[1.5, 0.5], [2.5, 0.5]])
        terrain_only = base.query(points)
        stationary = field.query(points, {"speed": torch.zeros(2)})
        self.assertTrue(torch.allclose(terrain_only, stationary, atol=1e-7))

    def test_speed_increases_slope_roughness_and_clearance_penalty(self) -> None:
        field = VehicleConditionedTerrainField(make_base_field())
        points = torch.tensor([[1.5, 0.5], [2.5, 0.5]])
        slow = field.query(points, {"speed": torch.zeros(2)})
        fast = field.query(points, {"speed": torch.full((2,), 3.0)})
        self.assertTrue(torch.all(fast < slow))
        components = field.component_costs(points, {"speed": torch.full((2,), 3.0)})
        self.assertTrue(torch.all(components["slope_speed_addition"] > 0.0))
        self.assertTrue(torch.all(components["roughness_speed_addition"] > 0.0))

    def test_reliable_heading_changes_lookahead_clearance(self) -> None:
        config = VehicleConditionedFieldConfig(
            slope_speed_gain=0.0,
            roughness_speed_gain=0.0,
            heading_lookahead_time_s=0.5,
        )
        field = VehicleConditionedTerrainField(make_base_field(), config)
        point = torch.tensor([[1.5, 0.5]])
        state_toward = {
            "speed": torch.tensor([3.0]),
            "heading": torch.tensor([0.0]),
            "heading_reliability": torch.tensor([1.0]),
        }
        state_away = {
            "speed": torch.tensor([3.0]),
            "heading": torch.tensor([math.pi]),
            "heading_reliability": torch.tensor([1.0]),
        }
        self.assertLess(float(field.query(point, state_toward)), float(field.query(point, state_away)))

    def test_query_is_differentiable_in_position_and_speed(self) -> None:
        field = VehicleConditionedTerrainField(make_base_field())
        point = torch.tensor([[1.6, 0.5]], requires_grad=True)
        speed = torch.tensor([2.0], requires_grad=True)
        field.query(point, {"speed": speed}).sum().backward()
        self.assertTrue(torch.isfinite(point.grad).all())
        self.assertTrue(torch.isfinite(speed.grad).all())
        self.assertNotEqual(float(speed.grad.abs()), 0.0)

    def test_trajectory_state_softly_marks_stationary_heading_unreliable(self) -> None:
        path = torch.tensor(
            [[[0.0, 0.0], [0.0, 0.0], [0.5, 0.0]]], requires_grad=True
        )
        state = trajectory_motion_state(path, planning_dt_s=0.5)
        self.assertLess(float(state["heading_reliability"][0, 1].detach()), 0.1)
        self.assertGreater(float(state["heading_reliability"][0, 2].detach()), 0.9)
        self.assertAlmostEqual(float(state["speed"][0, 2].detach()), 1.0, places=6)
        (state["speed"].sum() + state["heading"].sum()).backward()
        self.assertTrue(torch.isfinite(path.grad).all())


if __name__ == "__main__":
    unittest.main()

"""Synthetic tests for planning-claim diagnostics."""

from __future__ import annotations

import unittest

import torch

from TerraFlow.evaluation.planning_claim_metrics import (
    PlanningClaimMetricConfig,
    candidate_claim_metrics,
    compliance_mask,
    compliant_diversity,
    derive_independent_component_maps,
    fit_demonstration_envelope,
)
from TerraFlow.terrain.feasibility_field import TerrainFieldConfig
from TerraFlow.terrain.trajectory_kinematics import TrajectoryKinematicConfig


class PlanningClaimMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metric = PlanningClaimMetricConfig(
            forward_m=8.0,
            lateral_m=4.0,
            goal_tolerance_m=0.5,
        )
        self.terrain = TerrainFieldConfig(forward_m=8.0, lateral_m=4.0)
        self.kinematic = TrajectoryKinematicConfig(
            maximum_curvature_per_m=0.35,
            maximum_lateral_acceleration_mps2=2.5,
        )

    def test_obstacle_path_has_more_exposure_and_less_clearance(self) -> None:
        terrain = torch.zeros(1, 3, 8, 8)
        terrain[:, 0] = 1.0
        terrain[:, 1, 4, 4] = 1.0
        components = derive_independent_component_maps(
            terrain, self.terrain, self.metric
        )
        safe = torch.tensor([[[[1.0, -3.0, 0.0], [3.0, -3.0, 0.0]]]])
        obstacle = torch.tensor([[[[3.5, 0.5, 0.0], [4.5, 0.5, 0.0]]]])
        safe_metrics = candidate_claim_metrics(
            safe, safe[:, 0, -1], components, self.metric, self.kinematic
        )
        obstacle_metrics = candidate_claim_metrics(
            obstacle, obstacle[:, 0, -1], components, self.metric, self.kinematic
        )
        self.assertGreater(
            float(obstacle_metrics["occupancy_exposure_rate"][0, 0]),
            float(safe_metrics["occupancy_exposure_rate"][0, 0]),
        )
        self.assertLess(
            float(obstacle_metrics["clearance_q05_m"][0, 0]),
            float(safe_metrics["clearance_q05_m"][0, 0]),
        )

    def test_validation_envelope_and_compliance_are_deterministic(self) -> None:
        values = {
            "goal_error_m": torch.zeros(4, 1),
            "occupancy_exposure_rate": torch.tensor([[0.0], [0.0], [0.1], [0.2]]),
            "nontraversable_exposure_rate": torch.zeros(4, 1),
            "slope_exposure_rate": torch.zeros(4, 1),
            "roughness_mean": torch.zeros(4, 1),
            "clearance_q05_m": torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
            "curvature_violation_rate": torch.zeros(4, 1),
            "lateral_acceleration_violation_rate": torch.zeros(4, 1),
        }
        envelope = fit_demonstration_envelope(values, self.metric)
        mask = compliance_mask(values, envelope)
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(mask.shape, (4, 1))
        self.assertTrue(bool(mask[1, 0]))

    def test_compliant_diversity_ignores_noncompliant_candidates(self) -> None:
        trajectories = torch.zeros(1, 3, 2, 3)
        trajectories[0, 1, :, 1] = 2.0
        trajectories[0, 2, :, 1] = 20.0
        values, has_pair = compliant_diversity(
            trajectories, torch.tensor([[True, True, False]])
        )
        self.assertTrue(bool(has_pair[0]))
        self.assertAlmostEqual(float(values[0]), 2.0, places=5)


if __name__ == "__main__":
    unittest.main()

"""Numerical and gradient tests for trajectory-level kinematic feasibility."""

from __future__ import annotations

import math
import unittest

import torch

from TerraFlow.terrain.trajectory_kinematics import (
    TrajectoryKinematicConfig,
    trajectory_kinematic_cost,
    trajectory_kinematic_quantities,
)


class TrajectoryKinematicTests(unittest.TestCase):
    def test_straight_constant_speed_has_zero_curvature(self) -> None:
        x = torch.arange(1.0, 7.0)
        trajectory = torch.stack((x, torch.zeros_like(x), torch.zeros_like(x)), dim=-1)[None]
        quantities = trajectory_kinematic_quantities(trajectory, planning_dt_s=0.5)
        torch.testing.assert_close(
            quantities["absolute_curvature_per_m"],
            torch.zeros_like(quantities["absolute_curvature_per_m"]),
            atol=1e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            quantities["lateral_acceleration_mps2"],
            torch.zeros_like(quantities["lateral_acceleration_mps2"]),
            atol=1e-6,
            rtol=0.0,
        )

    def test_circular_arc_recovers_inverse_radius(self) -> None:
        radius = 8.0
        theta = torch.linspace(0.08, 0.48, 6)
        xy = torch.stack(
            (radius * torch.sin(theta), radius * (1.0 - torch.cos(theta))), dim=-1
        )
        trajectory = torch.cat((xy, torch.zeros(6, 1)), dim=-1)[None]
        quantities = trajectory_kinematic_quantities(trajectory, planning_dt_s=0.5)
        expected = torch.full_like(quantities["absolute_curvature_per_m"], 1.0 / radius)
        torch.testing.assert_close(
            quantities["absolute_curvature_per_m"], expected, atol=3e-4, rtol=3e-3
        )

    def test_lateral_acceleration_increases_when_dt_decreases(self) -> None:
        radius = 6.0
        theta = torch.linspace(0.1, 0.6, 6)
        xy = torch.stack(
            (radius * torch.sin(theta), radius * (1.0 - torch.cos(theta))), dim=-1
        )
        trajectory = torch.cat((xy, torch.zeros(6, 1)), dim=-1)[None]
        slow = trajectory_kinematic_quantities(trajectory, planning_dt_s=1.0)
        fast = trajectory_kinematic_quantities(trajectory, planning_dt_s=0.5)
        torch.testing.assert_close(
            slow["absolute_curvature_per_m"], fast["absolute_curvature_per_m"]
        )
        ratio = fast["lateral_acceleration_mps2"] / slow[
            "lateral_acceleration_mps2"
        ].clamp_min(1e-8)
        torch.testing.assert_close(ratio, torch.full_like(ratio, 4.0), atol=1e-4, rtol=1e-4)

    def test_soft_limits_distinguish_gentle_and_sharp_paths(self) -> None:
        x = torch.linspace(0.5, 5.0, 10)
        gentle = torch.stack((x, 0.02 * x.square(), torch.zeros_like(x)), dim=-1)
        sharp = torch.stack((x, 0.40 * torch.sin(1.6 * x), torch.zeros_like(x)), dim=-1)
        trajectories = torch.stack((gentle, sharp), dim=0)
        cfg = TrajectoryKinematicConfig(
            curvature_weight=0.7,
            lateral_acceleration_weight=0.7,
            maximum_curvature_per_m=0.25,
            maximum_lateral_acceleration_mps2=2.0,
        )
        cost = trajectory_kinematic_cost(trajectories, 0.25, cfg)
        self.assertGreater(
            float(cost["trajectory_kinematic_cost"][1]),
            float(cost["trajectory_kinematic_cost"][0]),
        )
        self.assertGreater(
            float(cost["curvature_violation"][1].float().mean()),
            float(cost["curvature_violation"][0].float().mean()),
        )

    def test_gradient_descent_reduces_cost_without_nonfinite_values(self) -> None:
        x = torch.linspace(0.5, 5.0, 10)
        y = 0.5 * torch.sin(2.2 * x)
        trajectory = torch.stack((x, y, torch.zeros_like(x)), dim=-1)[None]
        cfg = TrajectoryKinematicConfig(
            curvature_weight=0.5,
            lateral_acceleration_weight=0.5,
            maximum_curvature_per_m=0.3,
            maximum_lateral_acceleration_mps2=2.5,
        )
        initial = trajectory.clone()
        current = trajectory.clone()
        costs = []
        for _ in range(8):
            current = current.detach().requires_grad_(True)
            objective = trajectory_kinematic_cost(current, 0.5, cfg)[
                "trajectory_kinematic_cost"
            ].mean()
            costs.append(float(objective.detach()))
            gradient = torch.autograd.grad(objective, current)[0]
            self.assertTrue(torch.isfinite(gradient).all())
            norm = torch.linalg.vector_norm(gradient.flatten()).clamp_min(1e-8)
            current = current - 0.03 * gradient / norm
        final_cost = float(
            trajectory_kinematic_cost(current.detach(), 0.5, cfg)[
                "trajectory_kinematic_cost"
            ].mean()
        )
        self.assertLess(final_cost, costs[0])
        displacement = torch.linalg.vector_norm((current.detach() - initial).flatten())
        self.assertLess(float(displacement), 8 * 0.03001)

    def test_zero_weights_preserve_zero_added_cost(self) -> None:
        trajectory = torch.randn(3, 7, 3)
        cost = trajectory_kinematic_cost(trajectory, 0.5)
        self.assertEqual(float(cost["pointwise_kinematic_cost"].abs().sum()), 0.0)

    def test_near_stationary_pose_jitter_is_reliability_attenuated(self) -> None:
        # Millimetre-scale alternating pose noise produces a numerically large
        # raw turning angle but does not provide enough displacement for a
        # reliable curvature estimate.
        x = torch.arange(1.0, 11.0) * 0.008
        y = torch.where(
            torch.arange(10) % 2 == 0,
            torch.full((10,), 0.006),
            torch.full((10,), -0.006),
        )
        trajectory = torch.stack((x, y, torch.zeros_like(x)), dim=-1)[None]
        quantities = trajectory_kinematic_quantities(trajectory, planning_dt_s=0.5)
        self.assertLess(float(quantities["curvature_reliability"].max()), 1e-3)
        self.assertLess(float(quantities["absolute_curvature_per_m"].max()), 0.2)

    def test_regular_waypoint_spacing_keeps_curvature_reliable(self) -> None:
        radius = 8.0
        theta = torch.linspace(0.08, 0.48, 6)
        xy = torch.stack(
            (radius * torch.sin(theta), radius * (1.0 - torch.cos(theta))), dim=-1
        )
        trajectory = torch.cat((xy, torch.zeros(6, 1)), dim=-1)[None]
        quantities = trajectory_kinematic_quantities(trajectory, planning_dt_s=0.5)
        self.assertGreater(float(quantities["curvature_reliability"].min()), 0.999)


if __name__ == "__main__":
    unittest.main()

"""Tests for structure-preserving inference-only guidance operators."""

from __future__ import annotations

import unittest

import torch

from TerraFlow.guidance.feasibility_flow_guidance import FeasibilityFlowGuidanceConfig
from TerraFlow.guidance.structure_preserving_guidance import (
    adaptive_feasibility_trigger,
    apply_relative_trust_region,
    paired_correction_metrics,
    project_goal_anchored_correction,
    smooth_trajectory_gradient,
)
from TerraFlow.interfaces import BasePlanner, TrajectoryBatch
from TerraFlow.models.flow_network import ConditionalTrajectoryFlow
from TerraFlow.planners.flow_planner import FlowPlanner, FlowPlannerConfig
from TerraFlow.planners.guided_flow_planner import GuidedFlowPlanner
from TerraFlow.tests.test_feasibility_flow_guidance import guidance_scene


class GradientSmoothingTests(unittest.TestCase):
    def test_smoothing_preserves_shape_and_spatial_channels(self) -> None:
        gradient = torch.randn(4, 9, 3)
        for kernel in ("none", "kernel_3", "kernel_5"):
            smoothed = smooth_trajectory_gradient(gradient, kernel)  # type: ignore[arg-type]
            self.assertEqual(smoothed.shape, gradient.shape)

    def test_constant_gradient_remains_constant(self) -> None:
        gradient = torch.full((2, 8, 3), 2.75)
        for kernel in ("kernel_3", "kernel_5"):
            torch.testing.assert_close(
                smooth_trajectory_gradient(gradient, kernel),  # type: ignore[arg-type]
                gradient,
            )

    def test_combined_smoothing_and_trust_region_is_differentiable(self) -> None:
        gradient = torch.randn(3, 8, 3, requires_grad=True)
        velocity = torch.randn_like(gradient)
        smoothed = smooth_trajectory_gradient(gradient, "kernel_3")
        correction, _ = apply_relative_trust_region(
            0.2 * smoothed, velocity, rho=0.1
        )
        correction.square().sum().backward()
        self.assertIsNotNone(gradient.grad)
        self.assertTrue(torch.isfinite(gradient.grad).all())


class TrustRegionTests(unittest.TestCase):
    def test_trajectory_correction_never_exceeds_relative_limit(self) -> None:
        torch.manual_seed(5)
        correction = 10.0 * torch.randn(6, 12, 3)
        velocity = torch.randn_like(correction)
        for rho in (0.05, 0.10, 0.20, 0.30, 0.50):
            applied, diagnostics = apply_relative_trust_region(
                correction, velocity, rho
            )
            applied_norm = torch.linalg.vector_norm(applied.flatten(1), dim=1)
            flow_norm = torch.linalg.vector_norm(velocity.flatten(1), dim=1)
            self.assertTrue(bool((applied_norm <= rho * flow_norm + 1e-6).all()))
            self.assertTrue(torch.isfinite(diagnostics["correction_flow_ratio"]).all())

    def test_waypoint_scope_is_also_bounded(self) -> None:
        correction = torch.full((2, 5, 3), 8.0)
        velocity = torch.randn_like(correction)
        applied, _ = apply_relative_trust_region(
            correction, velocity, rho=0.2, scope="waypoint"
        )
        self.assertTrue(bool((
            torch.linalg.vector_norm(applied, dim=-1)
            <= 0.2 * torch.linalg.vector_norm(velocity, dim=-1) + 1e-6
        ).all()))


class GoalAnchorProjectionTests(unittest.TestCase):
    def test_terminal_projection_only_zeros_terminal_correction(self) -> None:
        correction = torch.randn(3, 7, 3)
        projected = project_goal_anchored_correction(correction, "terminal")
        torch.testing.assert_close(projected[:, :-1], correction[:, :-1])
        torch.testing.assert_close(projected[:, -1], torch.zeros_like(projected[:, -1]))

    def test_affine_projection_preserves_endpoint_and_gradients(self) -> None:
        correction = torch.randn(2, 6, 3, requires_grad=True)
        projected = project_goal_anchored_correction(correction, "affine")
        torch.testing.assert_close(projected[:, -1], torch.zeros_like(projected[:, -1]))
        projected.square().sum().backward()
        self.assertIsNotNone(correction.grad)
        self.assertTrue(torch.isfinite(correction.grad).all())

    def test_none_projection_is_exact_identity(self) -> None:
        correction = torch.randn(2, 5, 3)
        projected = project_goal_anchored_correction(correction, "none")
        self.assertIs(projected, correction)


class TriggerAndMetricTests(unittest.TestCase):
    def test_trigger_is_bounded_and_monotonic(self) -> None:
        cost = torch.linspace(0.0, 1.0, 101)
        trigger = adaptive_feasibility_trigger(cost, alpha=12.0, reference_cost=0.55)
        self.assertTrue(bool(((trigger >= 0.0) & (trigger <= 1.0)).all()))
        self.assertTrue(bool((torch.diff(trigger) >= 0.0).all()))

    def test_paired_correction_metrics_are_exact(self) -> None:
        unguided = torch.zeros(1, 1, 3, 3)
        guided = unguided.clone()
        guided[0, 0, :, 0] = torch.tensor([1.0, 2.0, 3.0])
        metrics = paired_correction_metrics(guided, unguided)
        torch.testing.assert_close(metrics["mean_waypoint_correction_m"], torch.tensor([2.0]))
        torch.testing.assert_close(metrics["maximum_waypoint_correction_m"], torch.tensor([3.0]))
        torch.testing.assert_close(metrics["endpoint_correction_m"], torch.tensor([3.0]))
        torch.testing.assert_close(metrics["second_difference_change_m"], torch.tensor([0.0]))


class StructurePreservingPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(13)
        self.scene = guidance_scene(batch=1)
        self.model = ConditionalTrajectoryFlow(
            trajectory_points=6, hidden_dim=64, layers=1
        )
        torch.nn.init.normal_(self.model.output.weight, std=0.01)
        self.planner_config = FlowPlannerConfig(candidates=2, integration_steps=4)
        self.noise = torch.randn(1, 2, 6, 3)

    def test_eta_zero_remains_exact_and_interface_is_unchanged(self) -> None:
        baseline = FlowPlanner(self.model, self.planner_config).sample(
            self.scene, self.noise
        )
        planner = GuidedFlowPlanner(
            self.model,
            self.planner_config,
            FeasibilityFlowGuidanceConfig(
                strength=0.0,
                smoothing_kernel="kernel_3",
                trust_region_rho=0.1,
                adaptive_trigger_enabled=True,
            ),
        )
        self.assertIsInstance(planner, BasePlanner)
        prediction = planner.sample(self.scene, self.noise)
        self.assertIsInstance(prediction, TrajectoryBatch)
        torch.testing.assert_close(
            prediction.trajectories, baseline.trajectories, rtol=0.0, atol=0.0
        )

    def test_all_requested_eta_rho_values_remain_finite(self) -> None:
        for eta in (0.05, 0.10, 0.20, 0.50):
            for rho in (0.10, 0.20, 0.30):
                planner = GuidedFlowPlanner(
                    self.model,
                    self.planner_config,
                    FeasibilityFlowGuidanceConfig(
                        strength=eta,
                        smoothing_kernel="kernel_3",
                        trust_region_rho=rho,
                    ),
                )
                prediction = planner.sample(self.scene, self.noise)
                self.assertEqual(tuple(prediction.trajectories.shape), (1, 2, 6, 3))
                self.assertTrue(torch.isfinite(prediction.trajectories).all())


if __name__ == "__main__":
    unittest.main()

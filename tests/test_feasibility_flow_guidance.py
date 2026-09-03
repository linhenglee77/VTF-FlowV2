"""Automated checks for clean-space feasibility-guided Flow sampling."""

from __future__ import annotations

import unittest

import torch

from TerraFlow.guidance.feasibility_flow_guidance import (
    FeasibilityFlowGuidanceConfig,
    clean_feasibility_gradient,
    normalize_and_clip_gradient,
)
from TerraFlow.guidance.guidance_schedule import (
    GuidanceScheduleConfig,
    guidance_weight,
)
from TerraFlow.interfaces import BasePlanner, SceneBatch, TrajectoryBatch
from TerraFlow.models.flow_network import (
    ConditionalTrajectoryFlow,
    estimate_clean_trajectory,
)
from TerraFlow.planners.flow_planner import FlowPlanner, FlowPlannerConfig
from TerraFlow.planners.guided_flow_planner import GuidedFlowPlanner


def guidance_scene(batch: int = 2, horizon: int = 6) -> SceneBatch:
    """Build a small scene with spatially varying differentiable BEV costs."""

    torch.manual_seed(23)
    history = torch.zeros(batch, 2, 3)
    goal = torch.tensor([[3.0, 0.5, 0.0], [2.5, -0.5, 0.0]])[:batch]
    alpha = torch.linspace(1.0 / horizon, 1.0, horizon)[None, :, None]
    future = alpha * goal[:, None]
    height, width = 24, 24
    x = torch.linspace(0.0, 1.0, height)[None, None, :, None]
    y = torch.linspace(0.0, 1.0, width)[None, None, None, :]
    traversable = (0.85 - 0.25 * y + 0.05 * x).expand(batch, 1, height, width)
    occupancy = (0.1 + 0.5 * x * y).expand(batch, 1, height, width)
    elevation = (0.2 + 0.35 * x + 0.1 * y).expand(batch, 1, height, width)
    terrain = torch.cat((traversable, occupancy, elevation), dim=1).contiguous()
    return SceneBatch(
        ego_history=history,
        gt_future=future,
        goal=goal,
        point_cloud=None,
        semantic_labels=None,
        terrain_map=terrain,
        metadata=[{} for _ in range(batch)],
    )


class GuidanceScheduleTests(unittest.TestCase):
    def test_schedule_values(self) -> None:
        time = torch.tensor([0.0, 0.25, 1.0])
        torch.testing.assert_close(
            guidance_weight(time, 0.2, GuidanceScheduleConfig("constant")),
            torch.full_like(time, 0.2),
        )
        torch.testing.assert_close(
            guidance_weight(time, 0.2, GuidanceScheduleConfig("early-strong")),
            0.2 * (1.0 - time),
        )
        torch.testing.assert_close(
            guidance_weight(time, 0.2, GuidanceScheduleConfig("late-strong")),
            0.2 * time,
        )

    def test_late_increases_and_early_decreases(self) -> None:
        time = torch.linspace(0.0, 1.0, 9)
        late = guidance_weight(time, 1.0, GuidanceScheduleConfig("late-strong", 2.0))
        early = guidance_weight(time, 1.0, GuidanceScheduleConfig("early-strong", 2.0))
        self.assertTrue(bool((torch.diff(late) >= 0.0).all()))
        self.assertTrue(bool((torch.diff(early) <= 0.0).all()))


class GuidanceGradientTests(unittest.TestCase):
    def test_gradient_clipping_limits_per_trajectory_norm(self) -> None:
        gradient = torch.full((3, 6, 3), 100.0)
        clipped, diagnostics = normalize_and_clip_gradient(
            gradient, "none", maximum_norm=0.75
        )
        norms = torch.linalg.vector_norm(clipped.flatten(start_dim=1), dim=1)
        self.assertTrue(bool((norms <= 0.750001).all()))
        self.assertTrue(bool((diagnostics["gradient_clip_scale"] < 1.0).all()))

    def test_gradient_propagates_through_clean_estimate(self) -> None:
        state = torch.randn(2, 5, 3, requires_grad=True)
        time = torch.tensor([0.25, 0.75])
        velocity = 2.0 * state
        clean = estimate_clean_trajectory(state, time, velocity)
        gradient = torch.autograd.grad(clean.square().sum(), state)[0]
        expected_scale = (3.0 - 2.0 * time)[:, None, None]
        torch.testing.assert_close(gradient, 2.0 * clean * expected_scale)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_vehicle_guidance_gradient_is_finite(self) -> None:
        scene = guidance_scene()
        model = ConditionalTrajectoryFlow(trajectory_points=6, hidden_dim=64, layers=1)
        torch.nn.init.normal_(model.output.weight, std=0.01)
        state = torch.randn(2, 6, 3)
        time = torch.tensor([0.3, 0.7])
        condition = model.encode_condition(
            scene.ego_history, scene.goal, scene.terrain_map
        )
        _, gradient, diagnostics = clean_feasibility_gradient(
            model, state, time, condition, scene.terrain_map,
            FeasibilityFlowGuidanceConfig(field_type="vehicle"),
        )
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertFalse(bool(diagnostics["gradient_nonfinite"].any()))
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_unified_kinematic_guidance_reports_finite_terms(self) -> None:
        scene = guidance_scene()
        model = ConditionalTrajectoryFlow(trajectory_points=6, hidden_dim=64, layers=1)
        torch.nn.init.normal_(model.output.weight, std=0.01)
        state = torch.randn(2, 6, 3)
        time = torch.tensor([0.3, 0.7])
        condition = model.encode_condition(
            scene.ego_history, scene.goal, scene.terrain_map
        )
        _, gradient, diagnostics = clean_feasibility_gradient(
            model,
            state,
            time,
            condition,
            scene.terrain_map,
            FeasibilityFlowGuidanceConfig(
                field_type="vehicle",
                curvature_weight=0.5,
                lateral_acceleration_weight=0.5,
            ),
        )
        self.assertTrue(torch.isfinite(gradient).all())
        for name in (
            "vehicle_cost",
            "kinematic_cost",
            "curvature_violation_rate",
            "lateral_acceleration_violation_rate",
        ):
            self.assertIn(name, diagnostics)
            self.assertTrue(torch.isfinite(diagnostics[name]).all())


class GuidedPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(31)
        self.scene = guidance_scene(batch=1)
        self.model = ConditionalTrajectoryFlow(
            trajectory_points=6, hidden_dim=64, layers=1
        )
        # The production checkpoint has a trained state-dependent correction.
        # Give the tiny synthetic model the same property; the default output
        # layer is deliberately zero-initialized for stable FM training.
        torch.nn.init.normal_(self.model.output.weight, std=0.01)
        self.planner_config = FlowPlannerConfig(candidates=3, integration_steps=4)
        self.noise = torch.randn(1, 3, 6, 3)

    def test_eta_zero_exactly_reproduces_unguided_output(self) -> None:
        baseline = FlowPlanner(self.model, self.planner_config).sample(
            self.scene, self.noise
        )
        guided = GuidedFlowPlanner(
            self.model,
            self.planner_config,
            FeasibilityFlowGuidanceConfig(enabled=True, strength=0.0),
        ).sample(self.scene, self.noise)
        torch.testing.assert_close(guided.trajectories, baseline.trajectories, rtol=0, atol=0)

    def test_guided_output_shape_interface_and_diagnostics(self) -> None:
        planner = GuidedFlowPlanner(
            self.model,
            self.planner_config,
            FeasibilityFlowGuidanceConfig(strength=0.2, schedule="late-strong"),
        )
        self.assertIsInstance(planner, BasePlanner)
        prediction = planner.sample(self.scene, self.noise)
        self.assertIsInstance(prediction, TrajectoryBatch)
        self.assertEqual(tuple(prediction.trajectories.shape), (1, 3, 6, 3))
        self.assertIsNotNone(prediction.diagnostics)
        assert prediction.diagnostics is not None
        self.assertEqual(tuple(prediction.diagnostics["vehicle_cost"].shape), (1, 3, 4))
        self.assertTrue(torch.isfinite(prediction.trajectories).all())

    def test_nonzero_guidance_changes_paired_trajectory(self) -> None:
        baseline = FlowPlanner(self.model, self.planner_config).sample(
            self.scene, self.noise
        )
        guided = GuidedFlowPlanner(
            self.model,
            self.planner_config,
            FeasibilityFlowGuidanceConfig(strength=0.5, schedule="constant"),
        ).sample(self.scene, self.noise)
        self.assertGreater(
            float((guided.trajectories - baseline.trajectories).abs().max()), 1e-7
        )


if __name__ == "__main__":
    unittest.main()

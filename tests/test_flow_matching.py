"""Numerical and interface tests for minimal conditional Flow Matching."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from TerraFlow.interfaces import SceneBatch
from TerraFlow.models.flow_network import (
    ConditionalTrajectoryFlow,
    estimate_clean_trajectory,
    linear_flow_matching_sample,
)
from TerraFlow.models.flow_regularization import (
    FlowRegularizationConfig,
    regularized_flow_matching_loss,
    trajectory_smoothness_loss,
)
from TerraFlow.planners.flow_planner import FlowPlanner, FlowPlannerConfig


def flow_scene(batch: int = 2, horizon: int = 6) -> SceneBatch:
    history = torch.zeros(batch, 2, 3)
    history[:, 0, 0] = -0.1
    goal = torch.tensor([[3.0, 0.5, 0.1], [2.0, -0.4, -0.1]])[:batch]
    alpha = torch.linspace(1.0 / horizon, 1.0, horizon)[None, :, None]
    future = alpha * goal[:, None]
    future[:, :, 1] += 0.25 * torch.sin(torch.linspace(0.0, torch.pi, horizon))[None]
    return SceneBatch(
        ego_history=history,
        gt_future=future,
        goal=goal,
        point_cloud=None,
        semantic_labels=None,
        terrain_map=torch.rand(batch, 3, 32, 32),
        metadata=[{} for _ in range(batch)],
    )


class ConstantVelocityNetwork(nn.Module):
    """Small duck-typed velocity field for exact Euler tests."""

    def __init__(self, horizon: int, velocity: float) -> None:
        super().__init__()
        self.trajectory_points = horizon
        self.velocity = velocity

    def encode_condition(self, history, goal, terrain):
        return goal.new_zeros(goal.shape[0], 1)

    def forward(self, state, time, condition):
        return torch.full_like(state, self.velocity)


class LinearConstructionTests(unittest.TestCase):
    def test_exact_linear_interpolation_and_target_velocity(self) -> None:
        clean = torch.tensor([[[2.0, 4.0, 6.0], [4.0, 8.0, 12.0]]])
        base = torch.tensor([[[0.0, 2.0, 4.0], [2.0, 4.0, 6.0]]])
        time = torch.tensor([0.25])
        state, target, returned_base, returned_time = linear_flow_matching_sample(
            clean, base, time
        )
        torch.testing.assert_close(state, 0.75 * base + 0.25 * clean)
        torch.testing.assert_close(target, clean - base)
        torch.testing.assert_close(returned_base, base)
        torch.testing.assert_close(returned_time, time)

    def test_invalid_time_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "\[0,1\]"):
            linear_flow_matching_sample(
                torch.zeros(1, 2, 3), time=torch.tensor([1.1])
            )

    def test_clean_endpoint_estimate_is_exact_for_target_velocity(self) -> None:
        clean = torch.randn(2, 5, 3)
        base = torch.randn_like(clean)
        time = torch.tensor([0.1, 0.8])
        state, target, _, _ = linear_flow_matching_sample(clean, base, time)
        torch.testing.assert_close(
            estimate_clean_trajectory(state, time, target), clean
        )


class FlowNetworkTests(unittest.TestCase):
    def test_velocity_and_loss_shapes(self) -> None:
        torch.manual_seed(3)
        scene = flow_scene()
        model = ConditionalTrajectoryFlow(
            trajectory_points=6, hidden_dim=64, layers=1
        )
        clean = scene.gt_future
        base = torch.randn_like(clean)
        time = torch.tensor([0.2, 0.8])
        loss, parts = model.flow_matching_loss(
            clean,
            scene.ego_history,
            scene.goal,
            scene.terrain_map,
            base=base,
            time=time,
        )
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(tuple(parts["predicted_velocity"].shape), (2, 6, 3))
        torch.testing.assert_close(parts["target_velocity"], clean - base)

    def test_loss_is_exact_mean_squared_velocity_error(self) -> None:
        scene = flow_scene(batch=1)
        model = ConditionalTrajectoryFlow(
            trajectory_points=6, hidden_dim=64, layers=1
        )
        for parameter in model.parameters():
            nn.init.zeros_(parameter)
        base = torch.ones_like(scene.gt_future)
        loss, _ = model.flow_matching_loss(
            scene.gt_future,
            scene.ego_history,
            scene.goal,
            scene.terrain_map,
            base=base,
            time=torch.tensor([0.4]),
        )
        loss_parts = model.flow_matching_loss(
            scene.gt_future,
            scene.ego_history,
            scene.goal,
            scene.terrain_map,
            base=base,
            time=torch.tensor([0.4]),
        )[1]
        expected = torch.mean(
            (loss_parts["predicted_velocity"] - loss_parts["target_velocity"]).square()
        )
        torch.testing.assert_close(loss, expected)

    def test_tiny_fixed_batch_loss_can_decrease(self) -> None:
        torch.manual_seed(9)
        scene = flow_scene()
        model = ConditionalTrajectoryFlow(
            trajectory_points=6, hidden_dim=64, layers=1
        )
        base = torch.randn_like(scene.gt_future)
        time = torch.tensor([0.25, 0.75])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        def loss_value() -> torch.Tensor:
            return model.flow_matching_loss(
                scene.gt_future,
                scene.ego_history,
                scene.goal,
                scene.terrain_map,
                base=base,
                time=time,
            )[0]

        initial = float(loss_value().detach())
        for _ in range(40):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_value()
            loss.backward()
            optimizer.step()
        final = float(loss_value().detach())
        self.assertLess(final, initial * 0.25)


class FlowRegularizationTests(unittest.TestCase):
    def test_disabled_regularization_preserves_exact_fm_objective(self) -> None:
        torch.manual_seed(14)
        scene = flow_scene()
        model = ConditionalTrajectoryFlow(trajectory_points=6, hidden_dim=64, layers=1)
        base = torch.randn_like(scene.gt_future)
        time = torch.tensor([0.3, 0.7])
        expected = model.flow_matching_loss(
            scene.gt_future,
            scene.ego_history,
            scene.goal,
            scene.terrain_map,
            base=base,
            time=time,
        )[0]
        actual, terms = regularized_flow_matching_loss(
            model,
            scene,
            FlowRegularizationConfig(mode="none"),
            base=base,
            time=time,
        )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(terms["flow_matching_loss"], expected)

    def test_terrain_and_vehicle_losses_are_finite_and_differentiable(self) -> None:
        torch.manual_seed(17)
        scene = flow_scene()
        for mode in ("terrain", "vehicle"):
            model = ConditionalTrajectoryFlow(
                trajectory_points=6, hidden_dim=64, layers=1
            )
            total, terms = regularized_flow_matching_loss(
                model,
                scene,
                FlowRegularizationConfig(
                    mode=mode, lambda_feasibility=0.5, planning_dt_s=0.5
                ),
                base=torch.zeros_like(scene.gt_future),
                time=torch.full((scene.batch_size,), 0.5),
            )
            self.assertTrue(torch.isfinite(total))
            self.assertGreaterEqual(float(terms["feasibility_loss"].detach()), 0.0)
            self.assertLessEqual(float(terms["feasibility_loss"].detach()), 1.0)
            total.backward()
            gradient_norm = sum(
                float(parameter.grad.abs().sum())
                for parameter in model.parameters()
                if parameter.grad is not None
            )
            self.assertGreater(gradient_norm, 0.0)

    def test_unified_kinematic_training_terms_are_optional_and_differentiable(self) -> None:
        torch.manual_seed(19)
        scene = flow_scene()
        model = ConditionalTrajectoryFlow(trajectory_points=6, hidden_dim=64, layers=1)
        total, terms = regularized_flow_matching_loss(
            model,
            scene,
            FlowRegularizationConfig(
                mode="vehicle",
                lambda_feasibility=0.5,
                planning_dt_s=0.5,
                curvature_weight=0.25,
                lateral_acceleration_weight=0.25,
            ),
            base=torch.zeros_like(scene.gt_future),
            time=torch.full((scene.batch_size,), 0.5),
        )
        for name in (
            "mean_kinematic_cost",
            "mean_curvature_cost",
            "mean_lateral_acceleration_cost",
        ):
            self.assertIn(name, terms)
            self.assertTrue(torch.isfinite(terms[name]))
        total.backward()
        self.assertGreater(
            sum(
                float(parameter.grad.abs().sum())
                for parameter in model.parameters()
                if parameter.grad is not None
            ),
            0.0,
        )

    def test_smoothness_is_a_separate_second_difference_loss(self) -> None:
        straight = torch.tensor(
            [[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]
        )
        bent = straight.clone()
        bent[:, 1, 1] = 1.0
        torch.testing.assert_close(trajectory_smoothness_loss(straight), torch.tensor(0.0))
        self.assertGreater(float(trajectory_smoothness_loss(bent)), 0.0)


class EulerPlannerTests(unittest.TestCase):
    def test_all_supported_step_counts_integrate_constant_velocity(self) -> None:
        scene = flow_scene(batch=1)
        initial = torch.zeros(1, 3, 6, 3)
        for steps in (4, 8, 16, 32):
            planner = FlowPlanner(
                ConstantVelocityNetwork(6, velocity=2.0),  # type: ignore[arg-type]
                FlowPlannerConfig(candidates=3, integration_steps=steps),
            )
            prediction = planner.sample(scene, initial_noise=initial)
            self.assertEqual(tuple(prediction.trajectories.shape), (1, 3, 6, 3))
            torch.testing.assert_close(
                prediction.trajectories, torch.full_like(prediction.trajectories, 2.0)
            )
            self.assertIsNone(prediction.scores)

    def test_independent_candidates_and_history_shape(self) -> None:
        scene = flow_scene()
        model = ConditionalTrajectoryFlow(
            trajectory_points=6, hidden_dim=64, layers=1
        )
        planner = FlowPlanner(
            model,
            FlowPlannerConfig(
                candidates=4, integration_steps=8, save_integration_history=True
            ),
        )
        prediction = planner(scene)
        self.assertEqual(tuple(prediction.trajectories.shape), (2, 4, 6, 3))
        self.assertEqual(tuple(prediction.integration_history.shape), (2, 4, 8, 6, 3))
        self.assertGreater(
            float((prediction.trajectories[:, 0] - prediction.trajectories[:, 1]).abs().sum()),
            0.0,
        )

    def test_unsupported_step_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "one of"):
            FlowPlannerConfig(integration_steps=2)


if __name__ == "__main__":
    unittest.main()

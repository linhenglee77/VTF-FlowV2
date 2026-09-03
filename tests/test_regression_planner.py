"""Tests for deterministic neural trajectory regression and sequence splits."""

from __future__ import annotations

import unittest

import torch

from TerraFlow.interfaces import SceneBatch
from TerraFlow.planners.regression_planner import RegressionPlanner, RegressionPlannerConfig
from TerraFlow.scripts.train_regression import sequence_partition_indices


def regression_scene(batch: int = 2, horizon: int = 6) -> SceneBatch:
    history = torch.zeros(batch, 3, 3)
    history[:, :, 0] = torch.tensor([-0.2, -0.1, 0.0])
    goal = torch.tensor([[4.0, 1.0, 0.2], [3.0, -1.0, -0.1]])[:batch]
    alpha = torch.linspace(1.0 / horizon, 1.0, horizon)[None, :, None]
    future = alpha * goal[:, None]
    future[:, :, 1] += 0.2 * torch.sin(torch.linspace(0.0, torch.pi, horizon))[None]
    return SceneBatch(
        ego_history=history,
        gt_future=future,
        goal=goal,
        point_cloud=None,
        semantic_labels=None,
        terrain_map=torch.rand(batch, 3, 32, 32),
        metadata=[{"sequence": f"{index:05d}"} for index in range(batch)],
    )


class RegressionPlannerTests(unittest.TestCase):
    def test_raw_and_shared_interface_shapes(self) -> None:
        scene = regression_scene()
        planner = RegressionPlanner(
            RegressionPlannerConfig(horizon=6, feature_dim=64, decoder_hidden_dim=64)
        )
        self.assertEqual(tuple(planner.predict_trajectory(scene).shape), (2, 6, 3))
        self.assertEqual(tuple(planner(scene).trajectories.shape), (2, 1, 6, 3))

    def test_smooth_l1_loss_is_finite_and_backpropagates(self) -> None:
        scene = regression_scene()
        planner = RegressionPlanner(
            RegressionPlannerConfig(horizon=6, feature_dim=64, decoder_hidden_dim=64)
        )
        loss = planner.trajectory_loss(scene, "smooth_l1")
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in planner.parameters()))

    def test_l1_loss_name_and_horizon_validation(self) -> None:
        scene = regression_scene()
        planner = RegressionPlanner(
            RegressionPlannerConfig(horizon=6, feature_dim=64, decoder_hidden_dim=64)
        )
        self.assertTrue(torch.isfinite(planner.trajectory_loss(scene, "l1")))
        with self.assertRaisesRegex(ValueError, "unsupported"):
            planner.trajectory_loss(scene, "mse")  # type: ignore[arg-type]
        wrong = RegressionPlanner(
            RegressionPlannerConfig(horizon=5, feature_dim=64, decoder_hidden_dim=64)
        )
        with self.assertRaisesRegex(ValueError, "horizon"):
            wrong.trajectory_loss(scene)


class SequencePartitionTests(unittest.TestCase):
    def test_validation_holds_out_complete_sequences(self) -> None:
        sequence_ids = ["00000", "00000", "00002", "00003", "00003"]
        train, validation = sequence_partition_indices(sequence_ids, ["00003"])
        self.assertEqual(train, [0, 1, 2])
        self.assertEqual(validation, [3, 4])
        self.assertFalse(
            {sequence_ids[index] for index in train}
            & {sequence_ids[index] for index in validation}
        )

    def test_missing_validation_sequence_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "absent"):
            sequence_partition_indices(["00000", "00002"], ["00004"])


if __name__ == "__main__":
    unittest.main()

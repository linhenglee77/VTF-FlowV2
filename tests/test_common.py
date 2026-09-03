"""Tests for shared data contracts."""

import unittest

import torch

from TerraFlow import SceneBatch, TrajectoryBatch
from TerraFlow.tests.helpers import make_scene_batch


class CommonContractTest(unittest.TestCase):
    """Verify the shared dataclass contracts."""

    def setUp(self) -> None:
        self.scene_batch = make_scene_batch()

    def test_scene_batch_exposes_required_fields(self) -> None:
        """The stable scene fields and batch size are available to every module."""

        scene_batch: SceneBatch = self.scene_batch
        self.assertEqual(scene_batch.batch_size, 2)
        self.assertEqual(scene_batch.ego_history.shape, (2, 4, 3))
        self.assertEqual(scene_batch.gt_future.shape, (2, 5, 3))
        self.assertEqual(scene_batch.goal.shape, (2, 3))
        self.assertEqual(scene_batch.point_cloud.shape, (2, 8, 4))
        self.assertEqual(scene_batch.semantic_labels.shape, (2, 8))
        self.assertEqual(scene_batch.terrain_map.shape, (2, 2, 16, 16))
        self.assertEqual(scene_batch.metadata["coordinate_frame"], "current_ego")

    def test_scene_batch_to_preserves_metadata(self) -> None:
        """Tensor conversion leaves non-tensor provenance untouched."""

        converted = self.scene_batch.to(dtype=torch.float64)
        self.assertEqual(converted.gt_future.dtype, torch.float64)
        self.assertIs(converted.metadata, self.scene_batch.metadata)

    def test_trajectory_batch_accepts_version_one_xyz(self) -> None:
        """Version-one candidates follow [B, K, H, 3]."""

        batch = TrajectoryBatch(
            trajectories=torch.zeros(2, 4, 5, 3),
            scores=torch.zeros(2, 4),
        )
        self.assertEqual(batch.batch_size, 2)
        self.assertEqual(batch.num_candidates, 4)

    def test_trajectory_batch_rejects_wrong_rank(self) -> None:
        """Missing the candidate axis violates the stable interface."""

        with self.assertRaisesRegex(ValueError, r"\[B, K, H, D\]"):
            TrajectoryBatch(trajectories=torch.zeros(2, 5, 3))

    def test_trajectory_batch_rejects_mismatched_scores(self) -> None:
        """Scores must align exactly with batch and candidate axes."""

        with self.assertRaisesRegex(ValueError, "scores must have shape"):
            TrajectoryBatch(
                trajectories=torch.zeros(2, 4, 5, 3),
                scores=torch.zeros(2, 5),
            )


if __name__ == "__main__":
    unittest.main()

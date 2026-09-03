import torch

from TerraFlow.interfaces import SceneBatch, TrajectoryBatch


def test_scene_and_trajectory_shapes():
    scene = SceneBatch(
        ego_history=torch.zeros(1, 3),
        gt_future=torch.zeros(10, 3),
        goal=torch.zeros(3),
        point_cloud=None,
        semantic_labels=None,
        terrain_map=torch.zeros(3, 64, 64),
        metadata={},
    ).as_batch()
    assert scene.gt_future.shape == (1, 10, 3)
    prediction = TrajectoryBatch(torch.zeros(1, 4, 10, 3), torch.zeros(1, 4))
    assert prediction.trajectories.shape == (1, 4, 10, 3)

"""Shared test data builders for VTF-Flow interface tests."""

from typing import Any, Mapping

import torch

from TerraFlow import SceneBatch


def make_scene_batch() -> SceneBatch:
    """Create a small, internally consistent ego-frame scene batch."""

    batch_size = 2
    horizon = 5
    metadata: Mapping[str, Any] = {
        "sequence_id": ["00000", "00001"],
        "frame_id": [10, 20],
        "coordinate_frame": "current_ego",
    }
    return SceneBatch(
        ego_history=torch.zeros(batch_size, 4, 3),
        gt_future=torch.zeros(batch_size, horizon, 3),
        goal=torch.zeros(batch_size, 3),
        point_cloud=torch.zeros(batch_size, 8, 4),
        semantic_labels=torch.zeros(batch_size, 8, dtype=torch.long),
        terrain_map=torch.zeros(batch_size, 2, 16, 16),
        metadata=metadata,
    )

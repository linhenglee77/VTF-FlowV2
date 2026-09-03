"""Adapter from the leakage-controlled RELLIS-3D cache to SceneBatch."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from TerraFlow.datasets.rellis3d_cache import CachedTrajectoryDataset
from TerraFlow.interfaces import SceneBatch


class Rellis3DSceneDataset(Dataset):
    """Metric trajectory scenes backed by the existing NumPy cache."""

    def __init__(self, cache_root: Path | str, split: str):
        self.source = CachedTrajectoryDataset(Path(cache_root), split)
        manifest_path = Path(cache_root) / split / "manifest.csv"
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            self.manifest = list(csv.DictReader(handle))

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> SceneBatch:
        terrain = torch.from_numpy(
            np.asarray(self.source.bev[index], dtype=np.float32) / 255.0
        )
        future = torch.from_numpy(
            np.array(self.source.trajectory[index], dtype=np.float32, copy=True)
        )
        goal = torch.from_numpy(np.array(self.source.goal[index], dtype=np.float32, copy=True))
        return SceneBatch(
            ego_history=torch.zeros(1, 3, dtype=torch.float32),
            gt_future=future,
            goal=goal,
            point_cloud=None,
            semantic_labels=None,
            terrain_map=terrain,
            metadata={**self.manifest[index], "cache_row": index},
        )


def collate_scenes(scenes: list[SceneBatch]) -> SceneBatch:
    """Stack tensor fields while retaining per-scene metadata."""
    return SceneBatch(
        ego_history=torch.stack([scene.ego_history for scene in scenes]),
        gt_future=torch.stack([scene.gt_future for scene in scenes]),
        goal=torch.stack([scene.goal for scene in scenes]),
        point_cloud=None,
        semantic_labels=None,
        terrain_map=torch.stack([scene.terrain_map for scene in scenes]),
        metadata=[scene.metadata for scene in scenes],
    )

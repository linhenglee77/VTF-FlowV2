"""Aligned RELLIS-3D supervision for learning a continuous feasibility field."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class RellisFeasibilityDataset(Dataset):
    """Pair trajectory-cache BEV inputs with semantic risk and valid-cell masks.

    The caches are produced from the same sequence-disjoint split and retain the
    same row order. Shape and length checks make this alignment explicit rather
    than silently assuming it.
    """

    def __init__(self, trajectory_cache: Path | str, perception_cache: Path | str, split: str):
        trajectory_root = Path(trajectory_cache) / split
        perception_root = Path(perception_cache) / split
        self.terrain = np.load(trajectory_root / "bev.npy", mmap_mode="r")
        self.risk = np.load(perception_root / "risk_target.npy", mmap_mode="r")
        self.mask = np.load(perception_root / "supervision_mask.npy", mmap_mode="r")
        if not (len(self.terrain) == len(self.risk) == len(self.mask)):
            raise ValueError("Trajectory and perception caches are not row-aligned")
        if self.terrain.shape[1:] != (3, 64, 64):
            raise ValueError(f"Unexpected terrain shape: {self.terrain.shape}")
        if self.risk.shape[1:] != (1, 64, 64) or self.mask.shape[1:] != (1, 64, 64):
            raise ValueError("Risk and supervision mask must have shape [N,1,64,64]")

    def __len__(self) -> int:
        return len(self.terrain)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "terrain": torch.from_numpy(
                np.array(self.terrain[index], dtype=np.float32, copy=True) / 255.0
            ),
            "risk": torch.from_numpy(
                np.array(self.risk[index], dtype=np.float32, copy=True) / 255.0
            ),
            "mask": torch.from_numpy(
                np.array(self.mask[index], dtype=np.float32, copy=True)
            ),
        }

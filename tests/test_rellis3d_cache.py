"""Tests for the self-contained RELLIS-3D cache reader and split parser."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from TerraFlow.datasets.rellis3d_cache import CachedTrajectoryDataset, parse_official_splits


class Rellis3DCacheTests(unittest.TestCase):
    def test_split_parser_accepts_released_path_format(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = {
                "train": "00000/os1_cloud_node_kitti_bin/000123.bin x\n",
                "val": "00001\\os1_cloud_node_kitti_bin\\000456.bin x\n",
                "test": "00002/os1_cloud_node_kitti_bin/000789.bin x\n",
            }
            for split, content in rows.items():
                (root / f"pt_{split}.lst").write_text(content, encoding="utf-8")
            mapping = parse_official_splits(root)
            self.assertEqual(mapping[("00000", 123)], "train")
            self.assertEqual(mapping[("00001", 456)], "val")
            self.assertEqual(mapping[("00002", 789)], "test")

    def test_cached_dataset_applies_paper_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_root = root / "train"
            split_root.mkdir()
            np.save(split_root / "bev.npy", np.full((1, 3, 2, 2), 255, dtype=np.uint8))
            np.save(
                split_root / "trajectory.npy",
                np.asarray([[[24.0, 12.0, 3.0]]], dtype=np.float32),
            )
            np.save(split_root / "goal.npy", np.asarray([[24.0, 12.0, 3.0]], dtype=np.float32))
            (root / "dataset_config.json").write_text(
                json.dumps({"normalization_scales_m": [24.0, 12.0, 3.0]}),
                encoding="utf-8",
            )
            dataset = CachedTrajectoryDataset(root, "train")
            try:
                bev, trajectory, goal = dataset[0]
                np.testing.assert_allclose(bev, 1.0)
                np.testing.assert_allclose(trajectory, 1.0)
                np.testing.assert_allclose(goal, 1.0)
            finally:
                dataset.close()


if __name__ == "__main__":
    unittest.main()

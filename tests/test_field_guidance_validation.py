"""Tests for deterministic perturbation construction and paired summaries."""

from __future__ import annotations

import unittest

import torch

from TerraFlow.scripts.validate_field_guidance import (
    PairedStore,
    construct_perturbations,
    paired_summary,
)


class FieldGuidanceValidationTest(unittest.TestCase):
    def test_perturbations_are_finite_and_preserve_endpoints(self) -> None:
        x = torch.linspace(0.5, 10.0, 20)
        gt = torch.stack((x, torch.zeros_like(x), torch.zeros_like(x)), dim=-1)
        obstacles = torch.tensor([[5.0, 1.0], [8.0, -2.0]])
        perturbations = construct_perturbations(gt, 4, obstacles, seed=9)
        self.assertEqual(set(perturbations), {
            "smooth_spatial", "local_rotation", "controlled_offset", "obstacle_directed"
        })
        for value in perturbations.values():
            self.assertIsNotNone(value)
            assert value is not None
            self.assertEqual(value.shape, gt.shape)
            self.assertTrue(torch.isfinite(value).all())
            self.assertTrue(torch.allclose(value[0], gt[0], atol=1e-6))
            self.assertTrue(torch.allclose(value[-1], gt[-1], atol=1e-6))

    def test_paired_summary_uses_within_scene_effect(self) -> None:
        store = PairedStore(
            gt_mean_f=[0.8, 0.6, 0.7],
            perturbed_mean_f=[0.7, 0.5, 0.65],
            gt_min_f=[0.3, 0.2, 0.25],
            perturbed_min_f=[0.2, 0.1, 0.2],
            gt_terrain_cost=[1.0, 1.0, 1.0],
            perturbed_terrain_cost=[1.1, 1.1, 1.1],
            gt_vehicle_cost=[1.2, 1.2, 1.2],
            perturbed_vehicle_cost=[1.3, 1.3, 1.3],
        )
        summary = paired_summary(store)
        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["paired_win_rate_P_F_GT_gt_perturbed"], 1.0)
        self.assertGreater(summary["paired_effect_size_dz"], 0.0)


if __name__ == "__main__":
    unittest.main()

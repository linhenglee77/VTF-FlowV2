"""Tests for validation-only multi-objective guidance selection."""

from __future__ import annotations

import unittest

import torch

from TerraFlow.scripts.optimize_vtf_flow_validation import _fixed_noise, select_variant


class ValidationSelectionTests(unittest.TestCase):
    def test_antithetic_noise_contains_exact_pairs(self) -> None:
        noise = _fixed_noise([0, 4], 2, 8, 5, torch.device("cpu"), "antithetic")
        torch.testing.assert_close(noise[:, 0::2], -noise[:, 1::2])

    def test_dominated_variant_is_not_selected(self) -> None:
        rows = [
            {"kind": "VTF", "variant": "a", "ADE_candidate0_m": 1.0, "minADE@K_m": 1.0, "terrain_violation_rate": 1.0},
            {"kind": "VTF", "variant": "b", "ADE_candidate0_m": 0.9, "minADE@K_m": 0.9, "terrain_violation_rate": 0.9},
        ]
        selected = select_variant(
            rows,
            {"ADE_candidate0_m": 1.0, "minADE@K_m": 1.0, "terrain_violation_rate": 1.0},
            {"ADE_candidate0_m": 0.4, "minADE@K_m": 0.3, "terrain_violation_rate": 0.3},
            0.01,
        )
        self.assertEqual(selected["variant"], "b")
        self.assertFalse(rows[0]["pareto"])

    def test_accuracy_constraint_prevents_cost_only_selection(self) -> None:
        rows = [
            {"kind": "VTF", "variant": "balanced", "ADE_candidate0_m": 0.99, "minADE@K_m": 0.99, "terrain_violation_rate": 0.8},
            {"kind": "VTF", "variant": "cost_only", "ADE_candidate0_m": 1.2, "minADE@K_m": 1.2, "terrain_violation_rate": 0.1},
        ]
        selected = select_variant(
            rows,
            {"ADE_candidate0_m": 1.0, "minADE@K_m": 1.0, "terrain_violation_rate": 1.0},
            {"ADE_candidate0_m": 0.2, "minADE@K_m": 0.2, "terrain_violation_rate": 0.6},
            0.01,
        )
        self.assertEqual(selected["variant"], "balanced")


if __name__ == "__main__":
    unittest.main()

"""Tests for sequence-level holdout protocol helpers."""

from __future__ import annotations

import unittest

import pandas as pd

from TerraFlow.evaluation.sequence_robustness import (
    build_fixed_validation_holdouts,
    sequence_macro_benchmark_summary,
    sequence_level_method_effects,
)


class SequenceRobustnessTest(unittest.TestCase):
    def test_fixed_validation_folds_are_disjoint(self) -> None:
        folds = build_fixed_validation_holdouts(
            ["00000", "00001", "00002", "00003", "00004"],
            "00003",
            ["00000", "00001", "00002"],
        )
        self.assertEqual(len(folds), 3)
        for fold in folds:
            self.assertFalse(set(fold.train) & set(fold.validation))
            self.assertFalse(set(fold.train) & set(fold.test))
            self.assertFalse(set(fold.validation) & set(fold.test))
            self.assertNotIn("00003", fold.test)

    def test_development_sequence_cannot_be_tested(self) -> None:
        with self.assertRaises(ValueError):
            build_fixed_validation_holdouts(
                ["00000", "00003", "00004"], "00003", ["00003"]
            )

    def test_seed_averaging_precedes_sequence_effect(self) -> None:
        frame = pd.DataFrame(
            [
                {"test_sequence": "00000", "method": "FLOW", "seed": 0, "cost": 4.0},
                {"test_sequence": "00000", "method": "FLOW", "seed": 1, "cost": 6.0},
                {"test_sequence": "00000", "method": "VTF_V2", "seed": 0, "cost": 3.0},
                {"test_sequence": "00000", "method": "VTF_V2", "seed": 1, "cost": 5.0},
            ]
        )
        result = sequence_level_method_effects(frame, ["cost"])
        self.assertAlmostEqual(result.loc[0, "cost_FLOW"], 5.0)
        self.assertAlmostEqual(result.loc[0, "cost_VTF_V2"], 4.0)
        self.assertAlmostEqual(result.loc[0, "cost_difference"], -1.0)

    def test_macro_benchmark_averages_seeds_before_sequences(self) -> None:
        frame = pd.DataFrame(
            [
                {"test_sequence": "00000", "method": "A", "seed": 0, "K": 1, "cost": 2.0},
                {"test_sequence": "00000", "method": "A", "seed": 1, "K": 1, "cost": 4.0},
                {"test_sequence": "00001", "method": "A", "seed": 0, "K": 1, "cost": 9.0},
                {"test_sequence": "00000", "method": "B", "seed": 0, "K": 8, "cost": 1.0},
                {"test_sequence": "00001", "method": "B", "seed": 0, "K": 8, "cost": 5.0},
            ]
        )
        sequence, macro = sequence_macro_benchmark_summary(
            frame, ["cost"], ["A", "B"]
        )
        a_sequence = sequence[sequence["method"] == "A"].sort_values(
            "test_sequence"
        )
        self.assertEqual(a_sequence["cost"].tolist(), [3.0, 9.0])
        self.assertAlmostEqual(
            float(macro.loc[macro["method"] == "A", "cost_mean"].iloc[0]), 6.0
        )
        self.assertEqual(
            int(macro.loc[macro["method"] == "B", "K"].iloc[0]), 8
        )


if __name__ == "__main__":
    unittest.main()

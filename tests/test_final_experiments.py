"""Tests for publication-level experiment integrity utilities."""

from __future__ import annotations

import unittest

import numpy as np

from TerraFlow.evaluation.final_experiments import (
    SequenceSplit,
    aggregate_seed_records,
    align_scene_tables,
    benjamini_hochberg,
    bootstrap_mean_ci,
    classify_failure_cases,
    paired_wilcoxon,
    partition_sequence_indices,
    summarize_scene_rows,
)


class SequenceIntegrityTests(unittest.TestCase):
    def test_split_rejects_any_sequence_overlap(self) -> None:
        with self.assertRaises(ValueError):
            SequenceSplit("bad", ("00000",), ("00001",), ("00000",))

    def test_partition_keeps_complete_sequences(self) -> None:
        split = SequenceSplit("ok", ("00000",), ("00001",), ("00002",))
        result = partition_sequence_indices(["00000", "00001", "00002", "00000"], split)
        self.assertEqual(result, {"train": [0, 3], "validation": [1], "test": [2]})


class AggregateIntegrityTests(unittest.TestCase):
    def test_seed_mean_and_sample_sd_match_raw_records(self) -> None:
        rows = [
            {"split": "p", "method": "A", "seed": 0, "metric": 1.0},
            {"split": "p", "method": "A", "seed": 1, "metric": 3.0},
            {"split": "p", "method": "A", "seed": 2, "metric": 5.0},
        ]
        result = aggregate_seed_records(rows, ["metric"])[0]
        self.assertAlmostEqual(result["metric_mean"], 3.0)
        self.assertAlmostEqual(result["metric_sd"], 2.0)

    def test_duplicate_seed_is_rejected(self) -> None:
        rows = [
            {"split": "p", "method": "A", "seed": 0, "metric": 1.0},
            {"split": "p", "method": "A", "seed": 0, "metric": 2.0},
        ]
        with self.assertRaises(ValueError):
            aggregate_seed_records(rows, ["metric"])

    def test_missing_paired_scene_is_rejected(self) -> None:
        tables = {
            "A": [{"scene_id": "a", "metric": 1.0}],
            "D": [{"scene_id": "b", "metric": 1.0}],
        }
        with self.assertRaises(ValueError):
            align_scene_tables(tables, "metric")

    def test_nonfinite_scene_metric_is_not_silently_dropped(self) -> None:
        with self.assertRaises(ValueError):
            summarize_scene_rows([{"scene_id": "a", "metric": float("nan")}])


class StatisticalTests(unittest.TestCase):
    def test_bh_adjustment_is_monotone_in_rank(self) -> None:
        raw = np.array([0.04, 0.001, 0.02, 0.5])
        adjusted = benjamini_hochberg(raw)
        ranked = adjusted[np.argsort(raw)]
        self.assertTrue(np.all(np.diff(ranked) >= -1e-12))
        self.assertTrue(np.all((adjusted >= 0.0) & (adjusted <= 1.0)))

    def test_bootstrap_is_scene_level_deterministic_and_requires_1000(self) -> None:
        values = np.arange(10, dtype=np.float64)
        first = bootstrap_mean_ci(values, resamples=1000, seed=7)
        second = bootstrap_mean_ci(values, resamples=1000, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first[0], 4.5)
        with self.assertRaises(ValueError):
            bootstrap_mean_ci(values, resamples=999, seed=7)

    def test_wilcoxon_effect_sign_tracks_paired_direction(self) -> None:
        result = paired_wilcoxon(np.array([3.0, 4.0, 5.0]), np.array([2.0, 3.0, 4.0]))
        self.assertLess(result["rank_biserial"], 0.0)


class FailureTaxonomyTests(unittest.TestCase):
    def test_all_four_categories_are_deterministic(self) -> None:
        flow = []
        full = []
        deltas = ((1, 1), (-1, 1), (1, -1), (-1, -1))
        for index, (ade_delta, cost_delta) in enumerate(deltas):
            base = {
                "scene_id": str(index), "sequence": "00004", "dataset_index": index,
                "minADE@K_m": 2.0, "mean_vehicle_conditioned_cost": 2.0,
                "terrain_violation_rate": 0.2, "smoothness_m": 0.1,
            }
            flow.append(base)
            full.append({
                **base, "minADE@K_m": 2.0 + ade_delta,
                "mean_vehicle_conditioned_cost": 2.0 + cost_delta,
            })
        rows = classify_failure_cases(flow, full, {
            "large_minade_degradation_m": 0.1,
            "large_vehicle_cost_reduction": 0.02,
            "large_smoothness_increase_m": 0.002,
            "high_terrain_violation_rate": 0.5,
        })
        self.assertEqual(len({row["category"] for row in rows}), 4)


if __name__ == "__main__":
    unittest.main()

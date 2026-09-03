"""Numerical tests for unified trajectory and feasibility evaluation."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from TerraFlow.evaluation import TerraFlowEvaluator
from TerraFlow.metrics.feasibility_metrics import feasibility_metrics
from TerraFlow.metrics.trajectory_metrics import trajectory_metrics
from TerraFlow.scripts.evaluate_predictions import evaluate_saved_predictions


class TrajectoryMetricsTest(unittest.TestCase):
    """Verify metric values using analytically simple trajectories."""

    def test_k1_ade_and_fde_use_euclidean_distance(self) -> None:
        ground_truth = torch.zeros(1, 2, 3)
        prediction = torch.tensor(
            [[[[3.0, 4.0, 0.0], [3.0, 4.0, 0.0]]]]
        )
        metrics = trajectory_metrics(prediction, ground_truth)
        self.assertTrue(torch.equal(metrics["ADE"], torch.tensor([5.0])))
        self.assertTrue(torch.equal(metrics["FDE"], torch.tensor([5.0])))
        self.assertTrue(torch.equal(metrics["ADE_m"], torch.tensor([5.0])))
        self.assertTrue(torch.equal(metrics["FDE_m"], torch.tensor([5.0])))
        self.assertTrue(
            torch.equal(metrics["ADE_by_candidate_m"], torch.tensor([[5.0]]))
        )
        self.assertTrue(
            torch.equal(metrics["FDE_by_candidate_m"], torch.tensor([[5.0]]))
        )
        self.assertTrue(torch.equal(metrics["minADE@K_m"], torch.tensor([5.0])))
        self.assertTrue(torch.equal(metrics["minFDE@K_m"], torch.tensor([5.0])))

    def test_minade_and_minfde_select_best_of_k(self) -> None:
        ground_truth = torch.zeros(1, 3, 3)
        prediction = torch.zeros(1, 2, 3, 3)
        prediction[:, 0, :, 0] = torch.tensor([1.0, 2.0, 3.0])
        metrics = trajectory_metrics(prediction, ground_truth)
        self.assertTrue(
            torch.allclose(
                metrics["ADE_by_candidate_m"], torch.tensor([[2.0, 0.0]])
            )
        )
        self.assertTrue(
            torch.allclose(
                metrics["FDE_by_candidate_m"], torch.tensor([[3.0, 0.0]])
            )
        )
        self.assertTrue(torch.equal(metrics["minADE@K_m"], torch.tensor([0.0])))
        self.assertTrue(torch.equal(metrics["minFDE@K_m"], torch.tensor([0.0])))
        self.assertTrue(torch.equal(metrics["minADE@K"], torch.tensor([0.0])))
        self.assertTrue(torch.equal(metrics["minFDE@K"], torch.tensor([0.0])))

    def test_diversity_path_length_and_smoothness(self) -> None:
        prediction = torch.zeros(1, 2, 3, 3)
        prediction[0, 0, :, :2] = torch.tensor(
            [[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]]
        )
        prediction[0, 1, :, :2] = prediction[0, 0, :, :2] + torch.tensor(
            [3.0, 4.0]
        )
        metrics = trajectory_metrics(prediction, torch.zeros(1, 3, 3))
        self.assertTrue(torch.allclose(metrics["diversity_m"], torch.tensor([5.0])))
        self.assertTrue(
            torch.allclose(
                metrics["path_length_by_candidate_m"], torch.tensor([[10.0, 10.0]])
            )
        )
        self.assertTrue(
            torch.equal(
                metrics["smoothness_by_candidate_m"], torch.tensor([[0.0, 0.0]])
            )
        )

    def test_k1_diversity_and_short_horizon_smoothness_are_zero(self) -> None:
        prediction = torch.zeros(2, 1, 2, 3)
        metrics = trajectory_metrics(prediction, torch.zeros(2, 2, 3))
        self.assertTrue(torch.equal(metrics["diversity_m"], torch.zeros(2)))
        self.assertTrue(
            torch.equal(metrics["smoothness_by_candidate_m"], torch.zeros(2, 1))
        )


class FeasibilityAvailabilityTest(unittest.TestCase):
    """Ensure unavailable terrain inputs never generate fabricated numbers."""

    def test_missing_terrain_returns_explicit_unavailable_states(self) -> None:
        trajectories = torch.zeros(1, 2, 4, 3)
        states = feasibility_metrics(trajectories, terrain_map=None)
        for name in (
            "occupancy_violation_rate",
            "traversability_violation_rate",
            "slope_violation_rate",
            "mean_terrain_cost",
            "minimum_obstacle_clearance",
            "elevation_consistency_error",
        ):
            self.assertEqual(states[name]["status"], "unavailable")
            self.assertIsNone(states[name]["values"])
            self.assertTrue(states[name]["reason"])
        self.assertEqual(states["curvature_violation_rate"]["status"], "available")
        self.assertTrue(
            torch.equal(
                states["curvature_violation_rate"]["values"], torch.zeros(1, 2)
            )
        )


class UnifiedEvaluatorTest(unittest.TestCase):
    """Verify selection, latency, and serialized CLI outputs."""

    def test_score_selected_metrics_and_latency(self) -> None:
        ground_truth = torch.zeros(1, 3, 3)
        prediction = torch.ones(1, 2, 3, 3)
        prediction[:, 1] = 0.0
        scores = torch.tensor([[2.0, 1.0]])
        result = TerraFlowEvaluator().evaluate_tensors(
            prediction,
            ground_truth,
            scores=scores,
            terrain_map=None,
            inference_latency_ms=2.5,
        )
        self.assertEqual(result["selected_index"].tolist(), [1])
        self.assertTrue(torch.equal(result["ADE_m"], torch.tensor([0.0])))
        self.assertTrue(torch.equal(result["FDE_m"], torch.tensor([0.0])))
        self.assertTrue(
            torch.equal(result["inference_latency_ms"], torch.tensor([2.5]))
        )
        self.assertEqual(result["inference_latency"]["status"], "available")

    def test_saved_npz_writes_json_csv_and_terminal_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            saved_path = root / "predictions.npz"
            predictions = np.zeros((2, 1, 3, 3), dtype=np.float32)
            ground_truth = np.zeros((2, 3, 3), dtype=np.float32)
            predictions[1, 0, :, 0] = 1.0
            np.savez(
                saved_path,
                predictions=predictions,
                gt_future=ground_truth,
                inference_latency_ms=np.asarray([1.0, 3.0], dtype=np.float32),
            )
            output_dir = root / "evaluation"
            summary, rows, terminal = evaluate_saved_predictions(
                saved_path, output_dir
            )

            self.assertEqual(len(rows), 2)
            self.assertAlmostEqual(summary["metrics"]["ADE_m"]["mean"], 0.5)
            self.assertIn("VTF-Flow trajectory evaluation", terminal)
            self.assertTrue((output_dir / "evaluation_summary.json").is_file())
            self.assertTrue((output_dir / "per_scene_metrics.csv").is_file())
            loaded_json = json.loads(
                (output_dir / "evaluation_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                loaded_json["feasibility_metrics"]["occupancy_violation_rate"][
                    "status"
                ],
                "unavailable",
            )
            with (output_dir / "per_scene_metrics.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                csv_rows = list(csv.DictReader(stream))
            self.assertEqual(len(csv_rows), 2)
            self.assertEqual(
                csv_rows[0]["occupancy_violation_rate_status"], "unavailable"
            )


if __name__ == "__main__":
    unittest.main()

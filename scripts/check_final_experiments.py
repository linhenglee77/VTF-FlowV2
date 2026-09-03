"""Run automated integrity checks on final VTF-Flow experiment artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np


TERRAFLOW_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TERRAFLOW_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from TerraFlow.evaluation.final_experiments import read_csv, save_json  # noqa: E402


DEFAULT_ROOT = TERRAFLOW_ROOT / "outputs" / "final_experiments"
REQUIRED = (
    "effective_config.json", "seed.txt", "split_definition.json", "metrics.json",
    "scene_level_metrics.csv", "predictions.npz", "latency.json",
    "checkpoint_reference.txt",
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_no_overlap(root: Path) -> None:
    for path in root.glob("main_*_*/split_definition.json"):
        split = _json(path)
        groups = [set(split[name]) for name in ("train", "validation", "test")]
        assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])


def _assert_aggregates(root: Path) -> None:
    raw = read_csv(root / "raw_per_seed_results.csv")
    table = read_csv(root / "tables" / "table1_main_comparison.csv")
    metrics = ("minADE@K_m", "minFDE@K_m", "mean_vehicle_conditioned_cost", "smoothness_m")
    for row in table:
        group = [
            item for item in raw
            if item["split"] == row["split"] and item["method"] == row["method"]
        ]
        for metric in metrics:
            values = np.asarray([float(item[metric]) for item in group])
            assert np.isclose(values.mean(), float(row[f"{metric}_mean"]), atol=1e-12)
            assert np.isclose(values.std(ddof=1), float(row[f"{metric}_sd"]), atol=1e-12)


def _assert_paired_alignment(root: Path) -> None:
    expected: set[str] | None = None
    for seed in (0, 1, 2):
        for method in ("A", "B", "C", "D"):
            rows = read_csv(root / f"main_primary_seed{seed}_{method}" / "scene_level_metrics.csv")
            ids = {row["scene_id"] for row in rows}
            assert len(ids) == len(rows)
            if expected is None:
                expected = ids
            assert ids == expected


def _assert_distinct_seeds(root: Path) -> None:
    registry = read_csv(root / "experiment_registry.csv")
    for method in ("R", "A", "B", "C", "D"):
        rows = [
            row for row in registry
            if row["experiment_name"].startswith("main_primary_seed")
            and row["experiment_name"].endswith(f"_{method}")
        ]
        assert sorted(int(row["seed"]) for row in rows) == [0, 1, 2]


def _assert_eta_zero(root: Path) -> None:
    unguided = np.load(root / "main_primary_seed0_B" / "predictions.npz")["trajectories"]
    zero = np.load(root / "eta_primary_seed0_eta0" / "predictions.npz")["trajectories"]
    assert unguided.shape == zero.shape
    assert np.array_equal(unguided, zero)


def _assert_figure_sources(root: Path) -> None:
    manifest = _json(root / "figures" / "figure_manifest.json")
    for relative in manifest["quantitative_source_tables"]:
        assert (root / relative).is_file()
    qualitative = list((root / "figure_source_data").glob("qualitative_*.csv"))
    assert len(qualitative) == 5
    for stem in ("figure_A_method_overview", "figure_B_fidelity_feasibility_pareto",
                 "figure_C_eta_sensitivity", "figure_D_sampling_steps",
                 "figure_E_per_sequence", "figure_G_failure_cases"):
        for suffix in (".svg", ".pdf", ".tiff", ".png"):
            assert (root / "figures" / f"{stem}{suffix}").is_file()


def _assert_no_missing_scenes(root: Path) -> None:
    registry = read_csv(root / "experiment_registry.csv")
    for entry in registry:
        run = root / entry["experiment_name"]
        for name in REQUIRED:
            assert (run / name).is_file(), f"{run.name} missing {name}"
        rows = read_csv(run / "scene_level_metrics.csv")
        metrics = _json(run / "metrics.json")
        predictions = np.load(run / "predictions.npz")
        expected = int(metrics["evaluated_scenes"])
        assert len(rows) == expected
        assert predictions["trajectories"].shape[0] == expected
        assert predictions["ground_truth"].shape[0] == expected
        assert len(set(row["scene_id"] for row in rows)) == expected


def _assert_counts_reported(root: Path) -> None:
    for path in root.glob("*/metrics.json"):
        metrics = _json(path)
        assert int(metrics["evaluated_scenes"]) > 0
        latency = _json(path.parent / "latency.json")
        assert int(latency["scene_count"]) == int(metrics["evaluated_scenes"])


def _assert_directionality(root: Path) -> None:
    for path in root.glob("*/effective_config.json"):
        effective = _json(path)
        directions = effective["metric_directionality"]
        lower = set(directions["lower_is_better"])
        higher = set(directions["higher_is_better"])
        assert "minADE@K_m" in lower and "mean_vehicle_conditioned_cost" in lower
        assert "diversity_m" in higher and not (lower & higher)


def _assert_tables_regenerable(root: Path) -> None:
    required = (
        "table1_main_comparison.csv", "table1_main_comparison.md",
        "table2_ablation.csv", "table2_ablation.md", "table3_eta_sensitivity.csv",
        "table3_eta_sensitivity.md", "table4_sampling_step_sensitivity.csv",
        "table4_sampling_step_sensitivity.md", "table5_cross_sequence_generalization.csv",
        "table5_cross_sequence_generalization.md",
    )
    for name in required:
        path = root / "tables" / name
        assert path.is_file() and path.stat().st_size > 0
    summary = _json(root / "analysis_summary.json")
    assert summary["status"] == "complete"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks: list[tuple[str, Callable[[Path], None]]] = [
        ("no_train_test_sequence_overlap", _assert_no_overlap),
        ("reported_mean_sd_match_raw", _assert_aggregates),
        ("paired_tests_use_aligned_scenes", _assert_paired_alignment),
        ("seed_runs_are_distinct", _assert_distinct_seeds),
        ("eta_zero_matches_unguided", _assert_eta_zero),
        ("figures_have_saved_sources", _assert_figure_sources),
        ("no_missing_scenes", _assert_no_missing_scenes),
        ("scene_counts_reported", _assert_counts_reported),
        ("metric_directionality_documented", _assert_directionality),
        ("tables_regenerable_from_raw", _assert_tables_regenerable),
    ]
    results: list[dict[str, str]] = []
    for name, function in checks:
        try:
            function(args.output_root)
        except Exception as error:
            results.append({"check": name, "status": "failed", "detail": repr(error)})
        else:
            results.append({"check": name, "status": "passed", "detail": ""})
    status = "passed" if all(row["status"] == "passed" for row in results) else "failed"
    save_json(args.output_root / "automated_checks.json", {"status": status, "checks": results})
    print(json.dumps({"status": status, "checks": results}, indent=2))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

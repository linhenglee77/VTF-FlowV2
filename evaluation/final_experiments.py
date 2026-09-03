"""Reproducible utilities for VTF-Flow publication-level experiments.

The functions in this module operate on complete scenes.  They never treat
waypoints or candidates as statistically independent observations.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from scipy import stats

from TerraFlow.metrics.trajectory_metrics import trajectory_metrics
from TerraFlow.terrain.feasibility_field import AnalyticTerrainField, TerrainFieldConfig
from TerraFlow.terrain.trajectory_kinematics import (
    TrajectoryKinematicConfig,
    trajectory_kinematic_cost,
)
from TerraFlow.terrain.vehicle_conditioned_field import (
    BatchedVehicleConditionedTerrainField,
    VehicleConditionedFieldConfig,
    trajectory_motion_state,
)


LOWER_IS_BETTER = {
    "minADE@K_m", "minFDE@K_m", "mean_vehicle_conditioned_cost",
    "mean_terrain_cost", "terrain_violation_rate", "occupancy_violation_rate",
    "slope_violation_rate", "roughness_cost", "clearance_cost",
    "smoothness_m", "maximum_local_second_difference_m", "latency_ms_per_scene",
    "mean_kinematic_cost", "mean_unified_tvk_cost", "curvature_cost",
    "lateral_acceleration_cost", "curvature_violation_rate",
    "lateral_acceleration_violation_rate", "mean_absolute_curvature_per_m",
    "mean_lateral_acceleration_mps2",
}
HIGHER_IS_BETTER = {"diversity_m"}


@dataclass(frozen=True)
class SequenceSplit:
    """A sequence-disjoint train/validation/test definition."""

    name: str
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def __post_init__(self) -> None:
        groups = [set(self.train), set(self.validation), set(self.test)]
        if any(not group for group in groups):
            raise ValueError("train, validation, and test sequence groups must be non-empty")
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("train/validation/test sequence overlap is forbidden")

    @classmethod
    def from_mapping(cls, name: str, values: Mapping[str, Sequence[str]]) -> "SequenceSplit":
        """Normalize sequence identifiers to five digits."""

        return cls(
            name=name,
            train=tuple(str(value).zfill(5) for value in values["train"]),
            validation=tuple(str(value).zfill(5) for value in values["validation"]),
            test=tuple(str(value).zfill(5) for value in values["test"]),
        )

    def role(self, sequence: str) -> str:
        """Return the configured role of one sequence."""

        sequence = str(sequence).zfill(5)
        for role in ("train", "validation", "test"):
            if sequence in getattr(self, role):
                return role
        return "unused"

    def as_dict(self) -> dict[str, Any]:
        """Serialize the split for experiment artifacts."""

        return {
            "name": self.name,
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
            "sequence_overlap": [],
        }


def partition_sequence_indices(
    sequence_ids: Sequence[str], split: SequenceSplit
) -> dict[str, list[int]]:
    """Partition complete sequences and reject missing or overlapping data."""

    available = {str(value).zfill(5) for value in sequence_ids}
    required = set(split.train) | set(split.validation) | set(split.test)
    missing = required - available
    if missing:
        raise ValueError(f"configured sequences absent from source: {sorted(missing)}")
    result = {
        role: [
            index for index, sequence in enumerate(sequence_ids)
            if str(sequence).zfill(5) in set(getattr(split, role))
        ]
        for role in ("train", "validation", "test")
    }
    if any(not values for values in result.values()):
        raise ValueError("each sequence role must contain at least one scene")
    index_sets = [set(result[role]) for role in ("train", "validation", "test")]
    if index_sets[0] & index_sets[1] or index_sets[0] & index_sets[2] or index_sets[1] & index_sets[2]:
        raise AssertionError("scene leakage detected across split roles")
    return result


def scene_identifier(metadata: Mapping[str, Any], dataset_index: int) -> str:
    """Create a stable sequence/frame identifier for paired alignment."""

    sequence = str(metadata.get("sequence", "unknown")).zfill(5)
    frame = metadata.get("frame_id", metadata.get("frame_index", dataset_index))
    source_split = metadata.get("split", "unknown")
    return f"{sequence}:{frame}:{source_split}"


def terminal_scene_metrics(
    trajectories: torch.Tensor,
    ground_truth: torch.Tensor,
    terrain_map: torch.Tensor,
    terrain_config: TerrainFieldConfig,
    vehicle_config: VehicleConditionedFieldConfig,
    thresholds: Mapping[str, float],
    planning_dt_s: float = 0.5,
    kinematic_config: TrajectoryKinematicConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Compute one scalar per scene for every final evaluation metric.

    Feasibility summaries average over all candidates and waypoints.  Additional
    ``oracle_*`` fields score the candidate selected by minimum ADE and are used
    only for the K-sensitivity analysis.
    """

    metric = trajectory_metrics(trajectories, ground_truth[..., :3])
    batch, candidates, horizon, _ = trajectories.shape
    flat = trajectories.reshape(batch * candidates, horizon, 3)
    repeated_map = terrain_map.repeat_interleave(candidates, dim=0)
    terrain_field = AnalyticTerrainField(repeated_map, terrain_config)
    components = terrain_field.component_costs(flat[..., :2])
    terrain_cost = terrain_field.cost(flat).reshape(batch, candidates, horizon)
    motion = trajectory_motion_state(flat, planning_dt_s, vehicle_config)
    vehicle_field = BatchedVehicleConditionedTerrainField(terrain_field, vehicle_config)
    vehicle_cost = vehicle_field.cost(flat, motion).reshape(batch, candidates, horizon)
    kinematic = trajectory_kinematic_cost(
        flat,
        planning_dt_s,
        kinematic_config or TrajectoryKinematicConfig(),
    )
    kinematic_cost = kinematic["trajectory_kinematic_cost"].reshape(batch, candidates)
    curvature_cost = kinematic["weighted_curvature_cost"].reshape(
        batch, candidates, horizon
    )
    lateral_cost = kinematic["weighted_lateral_acceleration_cost"].reshape(
        batch, candidates, horizon
    )
    absolute_curvature = kinematic["absolute_curvature_per_m"].reshape(
        batch, candidates, horizon
    )
    lateral_acceleration = kinematic["lateral_acceleration_mps2"].reshape(
        batch, candidates, horizon
    )
    curvature_violation = kinematic["curvature_violation"].reshape(
        batch, candidates, horizon
    )
    lateral_acceleration_violation = kinematic[
        "lateral_acceleration_violation"
    ].reshape(batch, candidates, horizon)
    occupancy = components["occupancy"].reshape(batch, candidates, horizon)
    nontraversable = components["nontraversable"].reshape(batch, candidates, horizon)
    slope = components["slope"].reshape(batch, candidates, horizon)
    roughness = components["roughness"].reshape(batch, candidates, horizon)
    clearance = components["clearance"].reshape(batch, candidates, horizon)
    occupancy_violation = occupancy >= float(thresholds["occupancy_threshold"])
    terrain_violation = occupancy_violation | (
        nontraversable >= float(thresholds["nontraversable_threshold"])
    )
    slope_violation = slope >= float(thresholds["normalized_slope_threshold"])
    second = torch.linalg.vector_norm(torch.diff(trajectories, n=2, dim=2), dim=-1)
    best = metric["ADE_by_candidate_m"].argmin(dim=1)
    batch_index = torch.arange(batch, device=trajectories.device)
    return {
        "minADE@K_m": metric["minADE@K_m"],
        "minFDE@K_m": metric["minFDE@K_m"],
        "diversity_m": metric["diversity_m"],
        "smoothness_m": metric["smoothness_by_candidate_m"].mean(dim=1),
        "maximum_local_second_difference_m": second.max(dim=2).values.mean(dim=1),
        "mean_terrain_cost": terrain_cost.mean(dim=(1, 2)),
        "mean_vehicle_conditioned_cost": vehicle_cost.mean(dim=(1, 2)),
        "mean_kinematic_cost": kinematic_cost.mean(dim=1),
        "mean_unified_tvk_cost": (
            vehicle_cost.mean(dim=2) + kinematic_cost
        ).mean(dim=1),
        "curvature_cost": curvature_cost.mean(dim=(1, 2)),
        "lateral_acceleration_cost": lateral_cost.mean(dim=(1, 2)),
        "curvature_violation_rate": curvature_violation.float().mean(dim=(1, 2)),
        "lateral_acceleration_violation_rate": (
            lateral_acceleration_violation.float().mean(dim=(1, 2))
        ),
        "mean_absolute_curvature_per_m": absolute_curvature.mean(dim=(1, 2)),
        "mean_lateral_acceleration_mps2": lateral_acceleration.mean(dim=(1, 2)),
        "terrain_violation_rate": terrain_violation.float().mean(dim=(1, 2)),
        "occupancy_violation_rate": occupancy_violation.float().mean(dim=(1, 2)),
        "slope_violation_rate": slope_violation.float().mean(dim=(1, 2)),
        "roughness_cost": roughness.mean(dim=(1, 2)),
        "clearance_cost": clearance.mean(dim=(1, 2)),
        "oracle_vehicle_cost": vehicle_cost.mean(dim=2)[batch_index, best],
        "oracle_terrain_violation_rate": terrain_violation.float().mean(dim=2)[batch_index, best],
    }


def summarize_scene_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Average numeric scene metrics without silently dropping non-finite rows."""

    if not rows:
        raise ValueError("cannot summarize zero scenes")
    ignored = {"scene_id", "sequence", "frame_id", "dataset_index", "method", "seed", "split"}
    names = [name for name in rows[0] if name not in ignored]
    summary: dict[str, float] = {}
    for name in names:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite values in scene metric {name}")
        summary[name] = float(values.mean())
    summary["evaluated_scenes"] = float(len(rows))
    return summary


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write rows with a stable header."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a complete CSV table."""

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate_seed_records(
    records: Sequence[Mapping[str, Any]], metric_names: Sequence[str]
) -> list[dict[str, Any]]:
    """Compute mean and sample SD across distinct seed-level records."""

    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for record in records:
        key = (str(record["split"]), str(record["method"]))
        groups.setdefault(key, []).append(record)
    output: list[dict[str, Any]] = []
    for (split, method), group in sorted(groups.items()):
        seeds = [int(row["seed"]) for row in group]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate seeds for {split}/{method}")
        row: dict[str, Any] = {"split": split, "method": method, "n_seeds": len(group)}
        for metric in metric_names:
            values = np.asarray([float(item[metric]) for item in group], dtype=np.float64)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        output.append(row)
    return output


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Return Benjamini-Hochberg FDR-adjusted p-values."""

    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must be a one-dimensional array in [0,1]")
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0.0, 1.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def paired_wilcoxon(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Wilcoxon signed-rank test and matched-pairs rank-biserial effect size."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("paired arrays must be aligned one-dimensional vectors")
    difference = y - x
    nonzero = difference[difference != 0.0]
    if nonzero.size == 0:
        return {"statistic": 0.0, "p_value": 1.0, "rank_biserial": 0.0, "n": float(len(x))}
    test = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    ranks = stats.rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0.0].sum())
    negative = float(ranks[nonzero < 0.0].sum())
    effect = (positive - negative) / max(positive + negative, 1e-12)
    return {
        "statistic": float(test.statistic),
        "p_value": float(test.pvalue),
        "rank_biserial": float(effect),
        "n": float(len(x)),
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap a mean over scenes, never over candidates or waypoints."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be a finite non-empty scene vector")
    if resamples < 1000:
        raise ValueError("publication bootstrap requires at least 1000 resamples")
    generator = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 100):
        count = min(100, resamples - start)
        indices = generator.integers(0, values.size, size=(count, values.size))
        means[start:start + count] = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(values.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def align_scene_tables(
    tables: Mapping[str, Sequence[Mapping[str, Any]]], metric: str
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Align multiple methods by exact scene ID and reject missing scenes."""

    if not tables:
        raise ValueError("at least one scene table is required")
    mappings = {
        name: {str(row["scene_id"]): float(row[metric]) for row in rows}
        for name, rows in tables.items()
    }
    id_sets = [set(values) for values in mappings.values()]
    if any(ids != id_sets[0] for ids in id_sets[1:]):
        raise ValueError("paired scene tables are not exactly aligned")
    scene_ids = sorted(id_sets[0])
    return scene_ids, {
        name: np.asarray([values[scene_id] for scene_id in scene_ids], dtype=np.float64)
        for name, values in mappings.items()
    }


def classify_failure_cases(
    flow_rows: Sequence[Mapping[str, Any]],
    full_rows: Sequence[Mapping[str, Any]],
    flags: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Classify every aligned scene using deterministic metric directions."""

    flow = {str(row["scene_id"]): row for row in flow_rows}
    full = {str(row["scene_id"]): row for row in full_rows}
    if set(flow) != set(full):
        raise ValueError("failure taxonomy requires identical Flow/Full scenes")
    output: list[dict[str, Any]] = []
    for scene_id in sorted(flow):
        a, d = flow[scene_id], full[scene_id]
        fidelity_delta = float(d["minADE@K_m"]) - float(a["minADE@K_m"])
        feasibility_delta = float(d["mean_vehicle_conditioned_cost"]) - float(a["mean_vehicle_conditioned_cost"])
        violation_delta = float(d["terrain_violation_rate"]) - float(a["terrain_violation_rate"])
        smoothness_delta = float(d["smoothness_m"]) - float(a["smoothness_m"])
        fidelity_improved = fidelity_delta < 0.0
        feasibility_improved = feasibility_delta < 0.0
        if feasibility_improved and not fidelity_improved:
            category = "feasibility_improved_fidelity_degraded"
        elif fidelity_improved and not feasibility_improved:
            category = "fidelity_improved_feasibility_degraded"
        elif feasibility_improved and fidelity_improved:
            category = "both_improved"
        else:
            category = "both_degraded"
        output.append({
            "scene_id": scene_id,
            "sequence": a["sequence"],
            "dataset_index": a["dataset_index"],
            "category": category,
            "minADE_delta_m": fidelity_delta,
            "vehicle_cost_delta": feasibility_delta,
            "terrain_violation_delta": violation_delta,
            "smoothness_delta_m": smoothness_delta,
            "large_minADE_degradation": fidelity_delta >= float(flags["large_minade_degradation_m"]),
            "large_vehicle_cost_reduction": feasibility_delta <= -float(flags["large_vehicle_cost_reduction"]),
            "large_smoothness_increase": smoothness_delta >= float(flags["large_smoothness_increase_m"]),
            "high_terrain_violation": float(d["terrain_violation_rate"]) >= float(flags["high_terrain_violation_rate"]),
        })
    return output


def save_json(path: Path, value: Any) -> None:
    """Save UTF-8 JSON with stable indentation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


__all__ = [
    "HIGHER_IS_BETTER", "LOWER_IS_BETTER", "SequenceSplit",
    "aggregate_seed_records", "align_scene_tables", "benjamini_hochberg",
    "bootstrap_mean_ci", "classify_failure_cases", "paired_wilcoxon",
    "partition_sequence_indices", "read_csv", "save_json", "scene_identifier",
    "summarize_scene_rows", "terminal_scene_metrics", "write_csv",
]

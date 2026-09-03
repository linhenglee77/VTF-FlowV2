"""Sequence-level holdout protocol helpers for planning robustness studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

import pandas as pd


@dataclass(frozen=True)
class SequenceHoldoutFold:
    """One outer fold with disjoint train, validation, and test sequences."""

    name: str
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def __post_init__(self) -> None:
        groups = [set(self.train), set(self.validation), set(self.test)]
        if not all(groups):
            raise ValueError("train, validation, and test groups must be non-empty")
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("sequence groups must be mutually disjoint")

    @property
    def test_sequence(self) -> str:
        """Return the single held-out sequence."""

        if len(self.test) != 1:
            raise ValueError("a holdout fold must contain exactly one test sequence")
        return self.test[0]


def build_fixed_validation_holdouts(
    all_sequences: Sequence[str],
    validation_sequence: str,
    test_sequences: Iterable[str],
) -> tuple[SequenceHoldoutFold, ...]:
    """Build folds while keeping the development sequence out of outer tests."""

    normalized = tuple(sorted({str(value).zfill(5) for value in all_sequences}))
    validation = str(validation_sequence).zfill(5)
    if validation not in normalized:
        raise ValueError("validation sequence is unavailable")
    tests = tuple(str(value).zfill(5) for value in test_sequences)
    if len(set(tests)) != len(tests):
        raise ValueError("test sequences must be unique")
    if validation in tests:
        raise ValueError("the development sequence cannot be an outer test fold")
    unavailable = sorted(set(tests) - set(normalized))
    if unavailable:
        raise ValueError(f"unavailable test sequences: {unavailable}")
    folds = []
    for test in tests:
        train = tuple(
            sequence
            for sequence in normalized
            if sequence not in {validation, test}
        )
        folds.append(
            SequenceHoldoutFold(
                name=f"holdout_{test}",
                train=train,
                validation=(validation,),
                test=(test,),
            )
        )
    return tuple(folds)


def sequence_level_method_effects(
    run_summary: pd.DataFrame,
    metrics: Sequence[str],
    baseline: str = "FLOW",
    target: str = "VTF_V2",
) -> pd.DataFrame:
    """Average technical seeds within sequence, then compute paired effects."""

    required = {"test_sequence", "method", "seed", *metrics}
    missing = sorted(required - set(run_summary.columns))
    if missing:
        raise ValueError(f"run summary is missing columns: {missing}")
    sequence_method = (
        run_summary.groupby(["test_sequence", "method"], as_index=False)[list(metrics)]
        .mean()
    )
    base = sequence_method[sequence_method["method"] == baseline].set_index(
        "test_sequence"
    )
    target_frame = sequence_method[
        sequence_method["method"] == target
    ].set_index("test_sequence")
    if not base.index.equals(target_frame.index):
        raise ValueError("baseline and target sequences do not align")
    rows = []
    for sequence in base.index:
        row = {"test_sequence": sequence}
        for metric in metrics:
            row[f"{metric}_{baseline}"] = float(base.loc[sequence, metric])
            row[f"{metric}_{target}"] = float(target_frame.loc[sequence, metric])
            row[f"{metric}_difference"] = float(
                target_frame.loc[sequence, metric] - base.loc[sequence, metric]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def sequence_macro_benchmark_summary(
    run_summary: pd.DataFrame,
    metrics: Sequence[str],
    method_order: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate technical seeds within sequence before macro averaging.

    The input contains one row per method, held-out sequence, and technical
    training seed. Deterministic baselines may contain a single seed row. The
    returned sequence table has one row per method and held-out sequence; the
    macro table reports an unweighted mean and sample standard deviation across
    independent held-out sequences.
    """

    required = {"test_sequence", "method", "seed", "K", *metrics}
    missing = sorted(required - set(run_summary.columns))
    if missing:
        raise ValueError(f"run summary is missing columns: {missing}")
    unknown = sorted(set(run_summary["method"]) - set(method_order))
    if unknown:
        raise ValueError(f"method order omits methods: {unknown}")
    k_counts = run_summary.groupby("method")["K"].nunique()
    inconsistent = sorted(k_counts[k_counts != 1].index)
    if inconsistent:
        raise ValueError(f"candidate count varies within methods: {inconsistent}")
    sequence_method = (
        run_summary.groupby(["test_sequence", "method"], as_index=False)[
            ["K", *metrics]
        ]
        .mean()
    )
    sequence_method["K"] = sequence_method["K"].astype(int)
    expected_sequences = sorted(sequence_method["test_sequence"].unique())
    rows = []
    for method in method_order:
        block = sequence_method[sequence_method["method"] == method]
        observed = sorted(block["test_sequence"].unique())
        if observed != expected_sequences:
            raise ValueError(
                f"method {method} has sequences {observed}, expected {expected_sequences}"
            )
        row: dict[str, float | int | str] = {
            "method": method,
            "K": int(block["K"].iloc[0]),
            "n_held_out_sequences": len(block),
        }
        for metric in metrics:
            values = block[metric].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite values in {method}/{metric}")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sequence_sd"] = (
                float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            )
        rows.append(row)
    return sequence_method, pd.DataFrame(rows)


__all__ = [
    "SequenceHoldoutFold",
    "build_fixed_validation_holdouts",
    "sequence_macro_benchmark_summary",
    "sequence_level_method_effects",
]

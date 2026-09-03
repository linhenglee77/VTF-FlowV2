"""Train and verify the deterministic VTF-Flow trajectory-regression baseline.

The mandatory overfit gate runs before normal training. Validation samples are
held out by complete sequence ID, never by randomly selecting neighboring
frames from a training sequence.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# Required by CUDA/cuBLAS for reproducible matrix-multiplication algorithms.
# It must be set before the first CUDA context is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import DataLoader, Dataset, Subset


TERRAFLOW_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TERRAFLOW_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from TerraFlow.datasets.rellis3d import Rellis3DSceneDataset, collate_scenes  # noqa: E402
from TerraFlow.evaluation import EvaluatorConfig, TerraFlowEvaluator, timed_planner_call  # noqa: E402
from TerraFlow.interfaces import SceneBatch  # noqa: E402
from TerraFlow.metrics import FeasibilityMetricConfig  # noqa: E402
from TerraFlow.planners.regression_planner import (  # noqa: E402
    RegressionPlanner,
    RegressionPlannerConfig,
)


DEFAULT_CONFIG = TERRAFLOW_ROOT / "configs" / "rellis3d_regression.json"
DEFAULT_OUTPUT = TERRAFLOW_ROOT / "outputs" / "regression"


class CombinedSceneDataset(Dataset):
    """Concatenate named cache splits while retaining sequence provenance."""

    def __init__(self, cache_root: Path, source_splits: Sequence[str]) -> None:
        if not source_splits:
            raise ValueError("source_splits cannot be empty")
        self.datasets = [Rellis3DSceneDataset(cache_root, split) for split in source_splits]
        self.records: list[tuple[int, int]] = []
        self.sequence_ids: list[str] = []
        for dataset_index, dataset in enumerate(self.datasets):
            for local_index, metadata in enumerate(dataset.manifest):
                if "sequence" not in metadata:
                    raise ValueError("cache manifest lacks required sequence column")
                self.records.append((dataset_index, local_index))
                self.sequence_ids.append(str(metadata["sequence"]).zfill(5))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> SceneBatch:
        dataset_index, local_index = self.records[index]
        return self.datasets[dataset_index][local_index]


def sequence_partition_indices(
    sequence_ids: Sequence[str], validation_sequences: Iterable[str]
) -> tuple[list[int], list[int]]:
    """Partition indices using complete sequence IDs with an overlap assertion."""

    validation_set = {str(value).zfill(5) for value in validation_sequences}
    if not validation_set:
        raise ValueError("at least one validation sequence must be configured")
    available = {str(value).zfill(5) for value in sequence_ids}
    missing = validation_set - available
    if missing:
        raise ValueError(f"validation sequences absent from source data: {sorted(missing)}")
    train = [
        index
        for index, sequence in enumerate(sequence_ids)
        if str(sequence).zfill(5) not in validation_set
    ]
    validation = [
        index
        for index, sequence in enumerate(sequence_ids)
        if str(sequence).zfill(5) in validation_set
    ]
    if not train or not validation:
        raise ValueError("sequence split must leave non-empty train and validation sets")
    train_sequences = {str(sequence_ids[index]).zfill(5) for index in train}
    val_sequences = {str(sequence_ids[index]).zfill(5) for index in validation}
    if train_sequences & val_sequences:
        raise AssertionError("sequence leakage detected between training and validation")
    return train, validation


def set_reproducible_seed(seed: int) -> None:
    """Seed Python, NumPy, CPU, and CUDA generators deterministically."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _model_config(values: Mapping[str, Any]) -> RegressionPlannerConfig:
    allowed = set(RegressionPlannerConfig.__dataclass_fields__)
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown model configuration keys: {sorted(unknown)}")
    normalized = dict(values)
    if "metric_scales" in normalized:
        normalized["metric_scales"] = tuple(float(value) for value in normalized["metric_scales"])
    return RegressionPlannerConfig(**normalized)


def load_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate the regression JSON configuration."""

    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"model", "data", "training", "overfit_test"}
    if set(config) != required:
        raise ValueError(
            f"config top-level keys must be exactly {sorted(required)}, got {sorted(config)}"
        )
    _model_config(config["model"])
    if config["training"].get("loss") not in {"l1", "smooth_l1"}:
        raise ValueError("training.loss must be 'l1' or 'smooth_l1'")
    return config


def make_loader(
    dataset: Dataset,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    """Build a reproducibly seeded SceneBatch data loader."""

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_scenes,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def train_epoch(
    model: RegressionPlanner,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    training: Mapping[str, Any],
) -> float:
    """Run one supervised trajectory-regression epoch."""

    model.train()
    total, count = 0.0, 0
    for scene in loader:
        scene = scene.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = model.trajectory_loss(
            scene,
            loss_name=training["loss"],
            beta=float(training.get("smooth_l1_beta", 1.0)),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training.get("gradient_clip_norm", 1.0))
        )
        optimizer.step()
        total += float(loss.detach()) * scene.batch_size
        count += scene.batch_size
    return total / max(count, 1)


@torch.no_grad()
def evaluate(
    model: RegressionPlanner,
    loader: DataLoader,
    device: torch.device,
    training: Mapping[str, Any],
) -> dict[str, float]:
    """Measure loss, ADE, FDE, smoothness, and per-sample latency."""

    model.eval()
    evaluator = TerraFlowEvaluator(
        EvaluatorConfig(
            feasibility=FeasibilityMetricConfig(planning_dt_s=0.5)
        )
    )
    totals = {"loss": 0.0, "ADE_m": 0.0, "FDE_m": 0.0, "smoothness_m": 0.0}
    latency_total, count = 0.0, 0
    for scene in loader:
        scene = scene.to(device)
        prediction, latency_ms = timed_planner_call(model, scene)
        loss = model.trajectory_loss(
            scene,
            loss_name=training["loss"],
            beta=float(training.get("smooth_l1_beta", 1.0)),
        )
        result = evaluator(prediction, scene, inference_latency_ms=latency_ms)
        batch = scene.batch_size
        totals["loss"] += float(loss) * batch
        for name in ("ADE_m", "FDE_m", "smoothness_m"):
            totals[name] += float(result[name].sum().detach().cpu())
        latency_total += latency_ms * batch
        count += batch
    report = {name: value / max(count, 1) for name, value in totals.items()}
    report["latency_ms_per_sample"] = latency_total / max(count, 1)
    return report


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically replace a compact epoch CSV log."""

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


@torch.no_grad()
def save_trajectory_plots(
    model: RegressionPlanner,
    dataset: Dataset,
    output_dir: Path,
    device: torch.device,
    examples: int,
) -> list[str]:
    """Save XY and elevation predicted-vs-GT plots for manual inspection."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    paths: list[str] = []
    for index in range(min(examples, len(dataset))):
        scene = dataset[index]
        prediction = model(scene.to(device)).trajectories[0, 0].detach().cpu()
        target = scene.gt_future.detach().cpu()
        figure, (axis_xy, axis_z) = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)
        axis_xy.scatter([0.0], [0.0], marker="*", s=100, color="black", label="ego origin")
        axis_xy.plot(target[:, 0], target[:, 1], "o-", ms=2.5, lw=1.5, label="GT")
        axis_xy.plot(prediction[:, 0], prediction[:, 1], "o-", ms=2.5, lw=1.5, label="prediction")
        axis_xy.set_xlabel("ego x (m)")
        axis_xy.set_ylabel("ego y (m)")
        axis_xy.set_aspect("equal", adjustable="datalim")
        axis_xy.grid(alpha=0.25)
        axis_xy.legend()
        steps = np.arange(1, len(target) + 1)
        axis_z.plot(steps, target[:, 2], label="GT z")
        axis_z.plot(steps, prediction[:, 2], label="prediction z")
        axis_z.set_xlabel("future waypoint")
        axis_z.set_ylabel("ego z (m)")
        axis_z.grid(alpha=0.25)
        axis_z.legend()
        metadata = scene.metadata if isinstance(scene.metadata, Mapping) else {}
        figure.suptitle(
            f"Regression overfit: sequence {metadata.get('sequence', '?')}, "
            f"frame {metadata.get('frame_id', '?')}"
        )
        path = output_dir / f"sample_{index:02d}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(str(path.resolve()))
    return paths


def save_checkpoint(
    path: Path,
    model: RegressionPlanner,
    config: Mapping[str, Any],
    epoch: int,
    metrics: Mapping[str, float],
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Persist all state required to reproduce deterministic inference."""

    payload = {
        "model": model.state_dict(),
        "model_config": dict(config["model"]),
        "epoch": epoch,
        "metrics": dict(metrics),
        "seed": int(config["training"]["seed"]),
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def run_overfit_gate(
    train_dataset: Dataset,
    config: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    epoch_override: int | None = None,
) -> tuple[RegressionPlanner, dict[str, Any]]:
    """Fit exactly 20 configured samples and enforce quantitative success."""

    overfit = config["overfit_test"]
    training = config["training"]
    sample_count = int(overfit["samples"])
    if len(train_dataset) < sample_count:
        raise ValueError(f"overfit test needs {sample_count} training samples")
    selected = sorted(
        random.Random(int(training["seed"])).sample(range(len(train_dataset)), sample_count)
    )
    subset = Subset(train_dataset, selected)
    loader = make_loader(
        subset,
        batch_size=sample_count,
        shuffle=True,
        seed=int(training["seed"]) + 1,
        num_workers=0,
    )
    eval_loader = make_loader(
        subset,
        batch_size=sample_count,
        shuffle=False,
        seed=int(training["seed"]) + 2,
        num_workers=0,
    )
    model = RegressionPlanner(_model_config(config["model"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(overfit["learning_rate"]),
        weight_decay=0.0,
    )
    epochs = int(epoch_override or overfit["epochs"])
    initial = evaluate(model, eval_loader, device, training)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, loader, optimizer, device, training)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            metrics = evaluate(model, eval_loader, device, training)
            row = {"epoch": epoch, "train_loss": train_loss, **metrics}
            rows.append(row)
            write_csv(output_dir / "overfit_training_log.csv", rows)
            if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
                print(
                    f"overfit epoch={epoch:03d}/{epochs} loss={metrics['loss']:.5f} "
                    f"ADE={metrics['ADE_m']:.4f} FDE={metrics['FDE_m']:.4f}",
                    flush=True,
                )
    final = evaluate(model, eval_loader, device, training)
    reduction = 1.0 - final["loss"] / max(initial["loss"], 1e-12)
    passed = (
        final["ADE_m"] <= float(overfit["target_ade_m"])
        and reduction >= float(overfit["required_loss_reduction"])
    )
    plot_paths = save_trajectory_plots(
        model,
        subset,
        output_dir / "predicted_vs_gt",
        device,
        int(overfit["plot_examples"]),
    )
    summary: dict[str, Any] = {
        "status": "passed" if passed else "failed",
        "samples": sample_count,
        "sample_indices_within_training_partition": selected,
        "epochs": epochs,
        "initial": initial,
        "final": final,
        "loss_reduction_fraction": reduction,
        "target_ade_m": float(overfit["target_ade_m"]),
        "required_loss_reduction": float(overfit["required_loss_reduction"]),
        "wall_time_s": time.perf_counter() - started,
        "plots": plot_paths,
    }
    save_checkpoint(
        output_dir / "overfit_checkpoint.pt",
        model,
        config,
        epochs,
        final,
        {"overfit_summary": summary},
    )
    (output_dir / "overfit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if not passed:
        raise RuntimeError(
            "20-sample overfit gate failed: "
            f"ADE={final['ADE_m']:.4f} m, reduction={reduction:.3f}"
        )
    return model, summary


def run_full_training(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    config: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    epoch_override: int | None = None,
) -> dict[str, Any]:
    """Train on training sequences and checkpoint sequence-held-out validation."""

    training = config["training"]
    seed = int(training["seed"])
    batch_size = int(training["batch_size"])
    train_loader = make_loader(
        train_dataset,
        batch_size,
        shuffle=True,
        seed=seed + 10,
        num_workers=int(training.get("num_workers", 0)),
    )
    val_loader = make_loader(
        validation_dataset,
        batch_size * 2,
        shuffle=False,
        seed=seed + 11,
        num_workers=int(training.get("num_workers", 0)),
    )
    model = RegressionPlanner(_model_config(config["model"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    epochs = int(epoch_override or training["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(training["learning_rate"]) * 0.05
    )
    best_ade = float("inf")
    best_metrics: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, training)
        validation = evaluate(model, val_loader, device, training)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{key}": value for key, value in validation.items()},
            "learning_rate": scheduler.get_last_lr()[0],
        }
        rows.append(row)
        write_csv(output_dir / "training_log.csv", rows)
        save_checkpoint(output_dir / "last.pt", model, config, epoch, validation)
        if validation["ADE_m"] < best_ade:
            best_ade = validation["ADE_m"]
            best_metrics = validation
            save_checkpoint(output_dir / "best.pt", model, config, epoch, validation)
        scheduler.step()
        print(
            f"epoch={epoch:03d}/{epochs} train={train_loss:.5f} "
            f"val_ADE={validation['ADE_m']:.4f} val_FDE={validation['FDE_m']:.4f}",
            flush=True,
        )
    summary = {
        "status": "complete",
        "epochs": epochs,
        "best_validation": best_metrics,
        "wall_time_s": time.perf_counter() - started,
        "best_checkpoint": str((output_dir / "best.pt").resolve()),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, help="override full-training epochs")
    parser.add_argument("--overfit-epochs", type=int, help="override overfit-gate epochs")
    parser.add_argument("--seed", type=int, help="override reproducible seed")
    parser.add_argument(
        "--overfit-only",
        action="store_true",
        help="run the mandatory 20-sample verification without full training",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    seed = int(config["training"]["seed"])
    set_reproducible_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = CombinedSceneDataset(
        args.cache_root, tuple(config["data"]["source_splits"])
    )
    train_indices, validation_indices = sequence_partition_indices(
        source.sequence_ids, config["data"]["validation_sequences"]
    )
    train_dataset = Subset(source, train_indices)
    validation_dataset = Subset(source, validation_indices)
    train_sequences = sorted({source.sequence_ids[index] for index in train_indices})
    validation_sequences = sorted(
        {source.sequence_ids[index] for index in validation_indices}
    )
    split_report = {
        "source_splits": config["data"]["source_splits"],
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "train_sequences": train_sequences,
        "validation_sequences": validation_sequences,
        "sequence_overlap": sorted(set(train_sequences) & set(validation_sequences)),
    }
    if split_report["sequence_overlap"]:
        raise AssertionError("training/validation sequence leakage detected")
    (args.output_dir / "sequence_split.json").write_text(
        json.dumps(split_report, indent=2), encoding="utf-8"
    )
    print(json.dumps({"device": str(device), **split_report}, indent=2), flush=True)

    _, overfit_summary = run_overfit_gate(
        train_dataset,
        config,
        args.output_dir,
        device,
        epoch_override=args.overfit_epochs,
    )
    print(json.dumps({"overfit_test": overfit_summary}, indent=2), flush=True)
    if args.overfit_only:
        return 0
    full_summary = run_full_training(
        train_dataset,
        validation_dataset,
        config,
        args.output_dir,
        device,
        epoch_override=args.epochs,
    )
    print(json.dumps({"full_training": full_summary}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Train minimal conditional trajectory Flow Matching with an overfit gate."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import DataLoader, Dataset, Subset


TERRAFLOW_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TERRAFLOW_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from TerraFlow.evaluation import TerraFlowEvaluator, timed_planner_call  # noqa: E402
from TerraFlow.models.flow_network import ConditionalTrajectoryFlow  # noqa: E402
from TerraFlow.planners.flow_planner import FlowPlanner, FlowPlannerConfig  # noqa: E402
from TerraFlow.scripts.train_regression import (  # noqa: E402
    CombinedSceneDataset,
    make_loader,
    sequence_partition_indices,
    set_reproducible_seed,
)


DEFAULT_CONFIG = TERRAFLOW_ROOT / "configs" / "rellis3d_flow.json"
DEFAULT_OUTPUT = TERRAFLOW_ROOT / "outputs" / "flow"


def model_from_config(values: Mapping[str, Any]) -> ConditionalTrajectoryFlow:
    """Construct the network while normalizing JSON arrays to tuple fields."""

    normalized = dict(values)
    if "metric_scales" in normalized:
        normalized["metric_scales"] = tuple(normalized["metric_scales"])
    return ConditionalTrajectoryFlow(**normalized)


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate Flow architecture, data, optimization, and sampling."""

    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"model", "data", "training", "overfit_test", "sampling"}
    if set(config) != required:
        raise ValueError(
            f"config top-level keys must be exactly {sorted(required)}, got {sorted(config)}"
        )
    model_from_config(config["model"])
    FlowPlannerConfig(**config["sampling"])
    if int(config["overfit_test"]["samples"]) != 20:
        raise ValueError("the mandatory overfit test must use exactly 20 samples")
    return config


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Replace an epoch log with a complete CSV snapshot."""

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def save_checkpoint(
    path: Path,
    model: ConditionalTrajectoryFlow,
    config: Mapping[str, Any],
    epoch: int,
    metrics: Mapping[str, float],
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Save the velocity network and all inference configuration."""

    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "model_config": dict(config["model"]),
        "sampling_config": dict(config["sampling"]),
        "epoch": epoch,
        "metrics": dict(metrics),
        "seed": int(config["training"]["seed"]),
        "flow_definition": {
            "base": "x0 ~ N(0,I)",
            "interpolation": "x_t = (1-t)*x0 + t*x1",
            "target_velocity": "u_t = x1 - x0",
            "loss": "mean squared error(v_theta, u_t)",
        },
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train_epoch(
    model: ConditionalTrajectoryFlow,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float,
) -> float:
    """Run one epoch of the exact conditional Flow Matching objective."""

    model.train()
    total, count = 0.0, 0
    for scene in loader:
        scene = scene.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.flow_matching_loss(
            scene.gt_future[..., :3],
            scene.ego_history,
            scene.goal,
            scene.terrain_map,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        total += float(loss.detach()) * scene.batch_size
        count += scene.batch_size
    return total / max(count, 1)


@torch.no_grad()
def fixed_flow_loss(
    model: ConditionalTrajectoryFlow,
    loader: DataLoader,
    device: torch.device,
    seed: int,
) -> float:
    """Evaluate loss with fixed ``x0`` and ``t`` for comparable epochs."""

    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    total, count = 0.0, 0
    for scene in loader:
        scene = scene.to(device)
        clean = scene.gt_future[..., :3]
        base = torch.randn(
            clean.shape, dtype=clean.dtype, device=device, generator=generator
        )
        interpolation_time = torch.rand(
            clean.shape[0], dtype=clean.dtype, device=device, generator=generator
        )
        loss, _ = model.flow_matching_loss(
            clean,
            scene.ego_history,
            scene.goal,
            scene.terrain_map,
            base=base,
            time=interpolation_time,
        )
        total += float(loss) * scene.batch_size
        count += scene.batch_size
    return total / max(count, 1)


@torch.no_grad()
def sampling_metrics(
    model: ConditionalTrajectoryFlow,
    loader: DataLoader,
    device: torch.device,
    planner_config: FlowPlannerConfig,
    seed: int,
) -> dict[str, float]:
    """Report ADE/FDE, best-of-K, diversity, smoothness, and latency."""

    model.eval()
    planner = FlowPlanner(model, planner_config).to(device)
    evaluator = TerraFlowEvaluator()
    totals = {
        "ADE_m": 0.0,
        "FDE_m": 0.0,
        "minADE@K_m": 0.0,
        "minFDE@K_m": 0.0,
        "diversity_m": 0.0,
        "smoothness_m": 0.0,
    }
    latency_total, count = 0.0, 0
    for batch_index, scene in enumerate(loader):
        scene = scene.to(device)
        torch.manual_seed(seed + batch_index)
        prediction, latency_ms = timed_planner_call(planner, scene)
        result = evaluator(prediction, scene, inference_latency_ms=latency_ms)
        batch = scene.batch_size
        for name in totals:
            totals[name] += float(result[name].sum().detach().cpu())
        latency_total += latency_ms * batch
        count += batch
    report = {name: value / max(count, 1) for name, value in totals.items()}
    report["latency_ms_per_sample"] = latency_total / max(count, 1)
    return report


@torch.no_grad()
def save_overfit_plot(
    model: ConditionalTrajectoryFlow,
    dataset: Dataset,
    output_path: Path,
    device: torch.device,
    config: FlowPlannerConfig,
    seed: int,
) -> None:
    """Plot GT and all learned Flow candidates for the first overfit scene."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scene = dataset[0]
    torch.manual_seed(seed)
    prediction = FlowPlanner(model, config)(scene.to(device)).trajectories[0].cpu()
    target = scene.gt_future.cpu()
    error = torch.linalg.vector_norm(prediction - target[None], dim=-1).mean(dim=-1)
    best = int(error.argmin())
    figure, axis = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    axis.scatter([0.0], [0.0], marker="*", s=100, color="black", label="ego origin")
    for candidate in prediction:
        axis.plot(candidate[:, 0], candidate[:, 1], color="#60a5fa", alpha=0.22, lw=1.0)
    axis.plot(target[:, 0], target[:, 1], color="black", lw=2.2, label="GT")
    axis.plot(
        prediction[best, :, 0],
        prediction[best, :, 1],
        color="#dc2626",
        lw=2.0,
        label="best Flow sample",
    )
    axis.set_xlabel("ego x (m)")
    axis.set_ylabel("ego y (m)")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.25)
    axis.legend()
    axis.set_title("20-sample Flow Matching overfit check")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def run_overfit_gate(
    train_dataset: Dataset,
    config: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    epoch_override: int | None,
) -> dict[str, Any]:
    """Train exactly 20 samples and block full training unless it succeeds."""

    training = config["training"]
    overfit = config["overfit_test"]
    seed = int(training["seed"])
    samples = int(overfit["samples"])
    indices = sorted(random.Random(seed).sample(range(len(train_dataset)), samples))
    subset = Subset(train_dataset, indices)
    train_loader = make_loader(subset, samples, shuffle=True, seed=seed + 1, num_workers=0)
    eval_loader = make_loader(subset, samples, shuffle=False, seed=seed + 2, num_workers=0)
    model = model_from_config(config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(overfit["learning_rate"]), weight_decay=0.0
    )
    planner_config = FlowPlannerConfig(**config["sampling"])
    epochs = int(epoch_override or overfit["epochs"])
    initial_loss = fixed_flow_loss(model, eval_loader, device, seed + 20)
    initial_metrics = sampling_metrics(
        model, eval_loader, device, planner_config, seed + 30
    )
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            float(training["gradient_clip_norm"]),
        )
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            validation_loss = fixed_flow_loss(model, eval_loader, device, seed + 20)
            row = {
                "epoch": epoch,
                "train_flow_loss": train_loss,
                "fixed_flow_loss": validation_loss,
            }
            rows.append(row)
            write_csv(output_dir / "overfit_training_log.csv", rows)
            if epoch == 1 or epoch % 100 == 0 or epoch == epochs:
                print(
                    f"overfit epoch={epoch:03d}/{epochs} "
                    f"flow_loss={validation_loss:.5f}",
                    flush=True,
                )
    final_loss = fixed_flow_loss(model, eval_loader, device, seed + 20)
    final_metrics = sampling_metrics(model, eval_loader, device, planner_config, seed + 30)
    loss_reduction = 1.0 - final_loss / max(initial_loss, 1e-12)
    passed = (
        loss_reduction >= float(overfit["required_loss_reduction"])
        and final_metrics["minADE@K_m"] <= float(overfit["target_minADE_m"])
    )
    plot_path = output_dir / "overfit_flow_vs_gt.png"
    save_overfit_plot(model, subset, plot_path, device, planner_config, seed + 30)
    summary = {
        "status": "passed" if passed else "failed",
        "samples": samples,
        "indices_within_training_partition": indices,
        "epochs": epochs,
        "initial_fixed_flow_loss": initial_loss,
        "final_fixed_flow_loss": final_loss,
        "loss_reduction_fraction": loss_reduction,
        "initial_sampling": initial_metrics,
        "final_sampling": final_metrics,
        "target_minADE_m": float(overfit["target_minADE_m"]),
        "required_loss_reduction": float(overfit["required_loss_reduction"]),
        "plot": str(plot_path.resolve()),
        "wall_time_s": time.perf_counter() - started,
    }
    save_checkpoint(
        output_dir / "overfit_checkpoint.pt",
        model,
        config,
        epochs,
        {"flow_loss": final_loss, **final_metrics},
        {"overfit_summary": summary},
    )
    (output_dir / "overfit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if not passed:
        raise RuntimeError(
            "Flow overfit gate failed: "
            f"reduction={loss_reduction:.3f}, minADE={final_metrics['minADE@K_m']:.3f} m"
        )
    return summary


def run_full_training(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    config: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    epoch_override: int | None,
) -> dict[str, Any]:
    """Train after the gate and evaluate the held-out validation sequence."""

    training = config["training"]
    seed = int(training["seed"])
    batch_size = int(training["batch_size"])
    train_loader = make_loader(
        train_dataset,
        batch_size,
        shuffle=True,
        seed=seed + 100,
        num_workers=int(training["num_workers"]),
    )
    val_loader = make_loader(
        validation_dataset,
        batch_size * 2,
        shuffle=False,
        seed=seed + 101,
        num_workers=int(training["num_workers"]),
    )
    model = model_from_config(config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    epochs = int(epoch_override or training["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(training["learning_rate"]) * 0.05
    )
    best_loss = float("inf")
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            float(training["gradient_clip_norm"]),
        )
        validation_loss = fixed_flow_loss(model, val_loader, device, seed + 200)
        row = {
            "epoch": epoch,
            "train_flow_loss": train_loss,
            "val_flow_loss": validation_loss,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        rows.append(row)
        write_csv(output_dir / "training_log.csv", rows)
        metrics = {"flow_loss": validation_loss}
        save_checkpoint(output_dir / "last.pt", model, config, epoch, metrics)
        if validation_loss < best_loss:
            best_loss = validation_loss
            save_checkpoint(output_dir / "best.pt", model, config, epoch, metrics)
        scheduler.step()
        print(
            f"epoch={epoch:03d}/{epochs} train={train_loss:.5f} "
            f"val={validation_loss:.5f}",
            flush=True,
        )
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    final_metrics = sampling_metrics(
        model,
        val_loader,
        device,
        FlowPlannerConfig(**config["sampling"]),
        seed + 300,
    )
    summary = {
        "status": "complete",
        "epochs": epochs,
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_flow_loss": best_loss,
        "validation_sampling": final_metrics,
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
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--overfit-epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overfit-only", action="store_true")
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
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "train_sequences": train_sequences,
        "validation_sequences": validation_sequences,
        "sequence_overlap": sorted(set(train_sequences) & set(validation_sequences)),
    }
    (args.output_dir / "sequence_split.json").write_text(
        json.dumps(split_report, indent=2), encoding="utf-8"
    )
    print(json.dumps({"device": str(device), **split_report}, indent=2), flush=True)
    overfit = run_overfit_gate(
        train_dataset, config, args.output_dir, device, args.overfit_epochs
    )
    print(json.dumps({"overfit_test": overfit}, indent=2), flush=True)
    if args.overfit_only:
        return 0
    summary = run_full_training(
        train_dataset,
        validation_dataset,
        config,
        args.output_dir,
        device,
        args.epochs,
    )
    print(json.dumps({"full_training": summary}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

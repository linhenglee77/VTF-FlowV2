"""Train the learned RELLIS-3D terrain feasibility field."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from TerraFlow.datasets.feasibility_dataset import RellisFeasibilityDataset
from TerraFlow.terrain.learned_feasibility_field import (
    FeasibilityFieldNet,
    analytic_speed_sensitivity_target,
)


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def loss_terms(model, batch, sensitivity_weight: float, smoothness_weight: float):
    terrain = batch["terrain"]
    risk = batch["risk"]
    mask = batch["mask"]
    prediction = model(terrain)
    semantic = masked_mean(
        F.binary_cross_entropy_with_logits(prediction["base_logit"], risk, reduction="none"),
        mask,
    )
    sensitivity_target = analytic_speed_sensitivity_target(terrain)
    sensitivity = masked_mean(
        (torch.sigmoid(prediction["speed_sensitivity_logit"]) - sensitivity_target).square(),
        mask,
    )
    probability = torch.sigmoid(prediction["base_logit"])
    smoothness = (
        torch.diff(probability, dim=-1).abs().mean()
        + torch.diff(probability, dim=-2).abs().mean()
    )
    total = semantic + sensitivity_weight * sensitivity + smoothness_weight * smoothness
    return total, semantic, sensitivity, smoothness, probability


@torch.no_grad()
def validate(model, loader, device, sensitivity_weight, smoothness_weight):
    model.eval()
    totals = {"loss": 0.0, "mae": 0.0, "brier": 0.0, "intersection": 0.0, "union": 0.0, "pixels": 0.0}
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        total, _, _, _, probability = loss_terms(
            model, batch, sensitivity_weight, smoothness_weight
        )
        mask = batch["mask"]
        risk = batch["risk"]
        pixels = float(mask.sum())
        totals["loss"] += float(total) * len(risk)
        totals["mae"] += float(((probability - risk).abs() * mask).sum())
        totals["brier"] += float(((probability - risk).square() * mask).sum())
        predicted_positive = probability >= 0.5
        target_positive = risk >= 0.5
        valid = mask >= 0.5
        totals["intersection"] += float((predicted_positive & target_positive & valid).sum())
        totals["union"] += float(((predicted_positive | target_positive) & valid).sum())
        totals["pixels"] += pixels
    return {
        "val_loss": totals["loss"] / len(loader.dataset),
        "val_mae": totals["mae"] / max(totals["pixels"], 1.0),
        "val_brier": totals["brier"] / max(totals["pixels"], 1.0),
        "val_iou": totals["intersection"] / max(totals["union"], 1.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-cache", type=Path, required=True)
    parser.add_argument("--perception-cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    training = dict(config["training"])
    if args.seed is not None:
        training["seed"] = args.seed
    if args.epochs is not None:
        training["epochs"] = args.epochs
    config["training"] = training
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_set = RellisFeasibilityDataset(args.trajectory_cache, args.perception_cache, "train")
    val_set = RellisFeasibilityDataset(args.trajectory_cache, args.perception_cache, "val")
    train_loader = DataLoader(train_set, batch_size=training["batch_size"], shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=training["batch_size"] * 2, shuffle=False, num_workers=0)
    model = FeasibilityFieldNet(**config["model"]).to(device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training["epochs"], eta_min=training["learning_rate"] * 0.05)
    rows, best = [], float("inf")
    started = time.perf_counter()
    for epoch in range(training["epochs"]):
        model.train()
        total_loss, seen = 0.0, 0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, _, _, _, _ = loss_terms(
                model, batch, training["sensitivity_weight"], training["smoothness_weight"]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                for ema_parameter, parameter in zip(ema.parameters(), model.parameters()):
                    ema_parameter.mul_(0.99).add_(parameter, alpha=0.01)
                for ema_buffer, buffer in zip(ema.buffers(), model.buffers()):
                    ema_buffer.copy_(buffer)
            total_loss += float(loss.detach()) * len(batch["terrain"])
            seen += len(batch["terrain"])
        scheduler.step()
        validation = validate(
            ema, val_loader, device,
            training["sensitivity_weight"], training["smoothness_weight"],
        )
        row = {"epoch": epoch + 1, "train_loss": total_loss / seen, **validation, "learning_rate": scheduler.get_last_lr()[0]}
        rows.append(row)
        checkpoint = {
            "model": ema.state_dict(), "config": config, "epoch": epoch + 1,
            **validation,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if validation["val_loss"] < best:
            best = validation["val_loss"]
            torch.save(checkpoint, args.output_dir / "best.pt")
        print(
            f"epoch={epoch + 1:03d}/{training['epochs']} train={row['train_loss']:.5f} "
            f"val={row['val_loss']:.5f} mae={row['val_mae']:.4f} iou={row['val_iou']:.4f}",
            flush=True,
        )
    with (args.output_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "complete", "seed": seed, "epochs": training["epochs"],
        "best_val_loss": best, "wall_time_s": time.perf_counter() - started,
        "checkpoint": str(args.output_dir / "best.pt"),
        "supervision": "RELLIS-3D semantic risk on valid mask; analytic slope/clearance only for speed-sensitivity auxiliary head",
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

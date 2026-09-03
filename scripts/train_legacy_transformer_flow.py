"""Reproduce the checkpoint-stable Transformer Flow five-seed experiment."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from TerraFlow.datasets.rellis3d import Rellis3DSceneDataset, collate_scenes
from TerraFlow.models.legacy_transformer_flow import LegacyConditionalTrajectoryFlow


def goal_residual(future, goal, residual_std, scales):
    points = future.shape[1]
    alpha = torch.linspace(1.0 / points, 1.0, points, device=future.device)[None, :, None]
    return (future / scales[None, None] - alpha * (goal / scales[None])[:, None]) / residual_std[None, None]


@torch.no_grad()
def validate(model, loader, device, residual_std, scales):
    model.eval(); total = 0.0; count = 0
    for scene in loader:
        scene = scene.to(device)
        clean = goal_residual(scene.gt_future, scene.goal, residual_std, scales)
        loss, _ = model.flow_matching_loss(clean, scene.terrain_map, scene.goal / scales)
        total += float(loss) * len(clean); count += len(clean)
    return total / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    training = dict(config["training"]); training["seed"] = args.seed
    if args.epochs is not None: training["epochs"] = args.epochs
    config["training"] = training
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_data = Rellis3DSceneDataset(args.cache_root, "train")
    val_data = Rellis3DSceneDataset(args.cache_root, "val")
    scales_np = np.asarray(train_data.source.scales, dtype=np.float32)
    points = train_data.source.trajectory.shape[1]
    alpha = np.linspace(1.0 / points, 1.0, points, dtype=np.float32)[None, :, None]
    normalized_future = np.asarray(train_data.source.trajectory, dtype=np.float32) / scales_np
    normalized_goal = np.asarray(train_data.source.goal, dtype=np.float32) / scales_np
    residual_std_np = np.maximum(np.std(normalized_future - alpha * normalized_goal[:, None], axis=(0, 1)), 1e-4).astype(np.float32)
    scales = torch.tensor(scales_np, device=device); residual_std = torch.tensor(residual_std_np, device=device)
    train_loader = DataLoader(train_data, batch_size=training["batch_size"], shuffle=True, num_workers=0, collate_fn=collate_scenes, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=training["batch_size"] * 2, shuffle=False, num_workers=0, collate_fn=collate_scenes)
    model = LegacyConditionalTrajectoryFlow(**config["model"]).to(device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters(): parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training["epochs"], eta_min=training["learning_rate"] * 0.05)
    rows, best = [], float("inf"); started = time.perf_counter()
    for epoch in range(training["epochs"]):
        model.train(); total = 0.0; count = 0
        for scene in train_loader:
            scene = scene.to(device)
            clean = goal_residual(scene.gt_future, scene.goal, residual_std, scales)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model.flow_matching_loss(clean, scene.terrain_map, scene.goal / scales)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            with torch.no_grad():
                for ema_parameter, parameter in zip(ema.parameters(), model.parameters()):
                    ema_parameter.mul_(0.995).add_(parameter, alpha=0.005)
                for ema_buffer, buffer in zip(ema.buffers(), model.buffers()): ema_buffer.copy_(buffer)
            total += float(loss.detach()) * len(clean); count += len(clean)
        scheduler.step(); validation = validate(ema, val_loader, device, residual_std, scales)
        row = {"epoch": epoch + 1, "train_loss": total / count, "val_loss": validation, "learning_rate": scheduler.get_last_lr()[0]}; rows.append(row)
        checkpoint = {"model": ema.state_dict(), "config": config, "residual_std_normalized": residual_std_np.tolist(), "metric_scales": scales_np.tolist(), "epoch": epoch + 1, "val_loss": validation, "architecture": "legacy_transformer_flow_v1"}
        torch.save(checkpoint, args.output_dir / "last.pt")
        if validation < best: best = validation; torch.save(checkpoint, args.output_dir / "best.pt")
        print(f"seed={args.seed} epoch={epoch + 1:03d}/{training['epochs']} train={row['train_loss']:.5f} val={validation:.5f}", flush=True)
    with (args.output_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {"status": "complete", "seed": args.seed, "best_val_loss": best, "epochs": training["epochs"], "wall_time_s": time.perf_counter() - started, "checkpoint": str(args.output_dir / "best.pt"), "architecture": "legacy_transformer_flow_v1"}
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

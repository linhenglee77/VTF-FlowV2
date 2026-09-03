"""Held-out comparison of binary, analytic and learned feasibility fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from TerraFlow.datasets.feasibility_dataset import RellisFeasibilityDataset
from TerraFlow.terrain.feasibility_field import AnalyticTerrainField
from TerraFlow.terrain.learned_feasibility_field import FeasibilityFieldNet


def grid_points(batch: int, device, dtype):
    forward = torch.linspace(24.0 / 128, 24.0 - 24.0 / 128, 64, device=device, dtype=dtype)
    lateral = torch.linspace(-12.0 + 12.0 / 64, 12.0 - 12.0 / 64, 64, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(forward, lateral, indexing="ij")
    return torch.stack([xx, yy, torch.zeros_like(xx)], dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)


def update(totals, name, probability, target, mask):
    valid = mask >= 0.5
    error = probability - target
    predicted = probability >= 0.5
    positive = target >= 0.5
    totals[name]["absolute"] += float((error.abs() * mask).sum())
    totals[name]["square"] += float((error.square() * mask).sum())
    totals[name]["intersection"] += float((predicted & positive & valid).sum())
    totals[name]["union"] += float(((predicted | positive) & valid).sum())
    totals[name]["pixels"] += float(mask.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-cache", type=Path, required=True)
    parser.add_argument("--perception-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = FeasibilityFieldNet(**checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = RellisFeasibilityDataset(args.trajectory_cache, args.perception_cache, "test")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    names = ("Binary traversability", "Analytic terrain field", "Learned feasibility field")
    totals = {name: {key: 0.0 for key in ("absolute", "square", "intersection", "union", "pixels")} for name in names}
    with torch.no_grad():
        for batch in loader:
            terrain, target, mask = (batch[key].to(device) for key in ("terrain", "risk", "mask"))
            binary = 1.0 - terrain[:, 0:1]
            points = grid_points(len(terrain), device, terrain.dtype)
            analytic = AnalyticTerrainField(terrain).cost(points).unsqueeze(1)
            learned = torch.sigmoid(model(terrain)["base_logit"])
            for name, probability in zip(names, (binary, analytic, learned)):
                update(totals, name, probability, target, mask)
    metrics = {}
    for name, value in totals.items():
        metrics[name] = {
            "masked_mae": value["absolute"] / value["pixels"],
            "masked_brier": value["square"] / value["pixels"],
            "masked_iou_at_0.5": value["intersection"] / max(value["union"], 1.0),
            "valid_pixels": int(value["pixels"]),
        }
    report = {
        "status": "complete", "split": "sequence-disjoint test", "samples": len(dataset),
        "target": "RELLIS-3D projected semantic risk on supervision_mask",
        "metrics": metrics,
        "checkpoint": str(args.checkpoint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Build the leakage-controlled cache used by the paper experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from TerraFlow.datasets.rellis3d_cache import build_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=150)
    parser.add_argument("--trajectory-stride", type=int, default=5)
    parser.add_argument("--isolation", type=int, default=150)
    parser.add_argument("--grid-size", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_cache(
        args.data_root,
        args.output_dir,
        horizon=args.horizon,
        trajectory_stride=args.trajectory_stride,
        isolation=args.isolation,
        grid_size=args.grid_size,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

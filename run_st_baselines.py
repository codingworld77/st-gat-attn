#!/usr/bin/env python3
"""Train STGCN / ASTGCN / ST-Transformer baselines with the existing pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ST spatiotemporal baselines.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["stgcn", "astgcn", "st_transformer"],
        choices=["stgcn", "astgcn", "st_transformer"],
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["temporal", "spatial"],
        choices=["temporal", "spatial", "emergent"],
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    for model in args.models:
        for split in args.splits:
            cmd = [
                sys.executable,
                str(root / "train.py"),
                "--config",
                str(args.config),
                "--model",
                model,
                "--split",
                split,
                "--epochs",
                str(args.epochs),
            ]
            if args.prepare_data:
                cmd.append("--prepare-data")
            if args.device:
                cmd.extend(["--device", args.device])
            print(f"\n=== {' '.join(cmd)} ===")
            subprocess.run(cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full ST-GAT-Attn experiment matrix.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--models", nargs="+", default=["b1", "b2", "b3", "st_gat_attn"])
    parser.add_argument("--splits", nargs="+", default=["temporal", "spatial", "emergent"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--prepare-data", action="store_true")
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
            ]
            if args.epochs is not None:
                cmd.extend(["--epochs", str(args.epochs)])
            if args.prepare_data:
                cmd.append("--prepare-data")
            print(f"\n=== Running: {' '.join(cmd)} ===")
            subprocess.run(cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()

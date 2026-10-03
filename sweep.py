from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for ST-GAT-Attn.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--splits", nargs="+", default=["temporal", "spatial"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    sweep_dir = root / "artifacts" / "sweeps" / datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir.mkdir(parents=True, exist_ok=True)

    grid = {
        "k_neighbors": [0, 4, 6, 8],
        "hidden_dim": [64, 128],
        "learning_rate": [0.001, 0.0005],
        "num_gat_layers": [1, 2],
    }
    combos = [dict(zip(grid.keys(), values)) for values in itertools.product(*grid.values())]

    manifest = []
    for combo in combos:
        for split in args.splits:
            run_name = (
                f"sweep_st_gat_attn_{split}_k{combo['k_neighbors']}_"
                f"h{combo['hidden_dim']}_lr{combo['learning_rate']}_"
                f"l{combo['num_gat_layers']}"
            )
            cmd = [
                sys.executable,
                str(root / "train.py"),
                "--config",
                str(args.config),
                "--model",
                "st_gat_attn",
                "--split",
                split,
                "--epochs",
                str(args.epochs),
                "--run-name",
                run_name,
                "--k-neighbors",
                str(combo["k_neighbors"]),
                "--hidden-dim",
                str(combo["hidden_dim"]),
                "--learning-rate",
                str(combo["learning_rate"]),
                "--num-gat-layers",
                str(combo["num_gat_layers"]),
            ]
            if args.prepare_data:
                cmd.append("--prepare-data")

            entry = {"run_name": run_name, "split": split, **combo, "cmd": cmd}
            manifest.append(entry)

            if args.dry_run:
                print(" ".join(cmd))
                continue

            print(f"\n=== Sweep run: {run_name} ===")
            subprocess.run(cmd, cwd=root, check=True)

    (sweep_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSweep manifest saved to {sweep_dir / 'manifest.json'}")

    if not args.dry_run:
        report_cmd = [sys.executable, str(root / "compare_results.py"), "--tag", "sweep_st_gat_attn"]
        subprocess.run(report_cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from st_gat_attn.config import Config, set_seed
from st_gat_attn.data.dataset import build_dataloaders
from st_gat_attn.data.preprocessing import load_processed_data, prepare_processed_data
from st_gat_attn.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ST-GAT-Attn climate forecasting models.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["b1", "b2", "b3", "st_gat_attn", "stgcn", "astgcn", "st_transformer"],
    )
    parser.add_argument("--split", type=str, default=None, choices=["temporal", "spatial", "emergent"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--prepare-data", action="store_true", help="Rebuild processed parquet cache.")
    parser.add_argument("--k-neighbors", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-gat-layers", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None, help="Optional custom run directory name suffix.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {}
    if args.model:
        overrides["model"] = args.model
    if args.split:
        overrides["split"] = args.split
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.device:
        overrides["device"] = args.device
    if args.k_neighbors is not None:
        overrides["k_neighbors"] = args.k_neighbors
    if args.hidden_dim is not None:
        overrides["hidden_dim"] = args.hidden_dim
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate
    if args.num_gat_layers is not None:
        overrides["num_gat_layers"] = args.num_gat_layers

    config_path = args.config.resolve()
    config = Config.from_yaml(config_path, overrides=overrides)
    set_seed(config.seed)

    if args.prepare_data or not (config.processed_dir / "panel.parquet").exists():
        print("Preparing processed dataset...")
        prepare_processed_data(
            data_dir=config.data_dir,
            coords_path=config.station_coords,
            processed_dir=config.processed_dir,
            holdout_fraction=config.spatial_holdout_fraction,
            holdout_seed=config.spatial_holdout_seed,
        )

    bundle = load_processed_data(config.processed_dir)
    loaders, scaler_meta = build_dataloaders(
        bundle=bundle,
        seq_len=config.seq_len,
        split=config.split,
        train_years=config.train_years,
        val_years=config.val_years,
        test_years=config.test_years,
        batch_size=config.batch_size,
        k_neighbors=config.k_neighbors,
        rain_threshold=config.rain_threshold_mm,
        num_workers=config.num_workers,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_label = args.run_name or f"{config.model}_{config.split}_{timestamp}"
    run_dir = config.output_dir / run_label
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config.__dict__, indent=2, default=str), encoding="utf-8")
    (run_dir / "scalers.json").write_text(json.dumps(scaler_meta, indent=2), encoding="utf-8")

    print(f"Model={config.model} | split={config.split} | train batches={len(loaders.train)} | device={config.resolve_device()}")
    trainer = Trainer(config, loaders, run_dir)
    result = trainer.train()

    print("\nTest metrics:")
    for key, value in sorted(result.test_metrics.items()):
        print(f"  {key}: {value:.4f}")
    print(f"\nArtifacts saved to: {result.run_dir}")


if __name__ == "__main__":
    main()

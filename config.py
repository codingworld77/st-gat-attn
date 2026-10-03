from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


@dataclass
class Config:
    data_dir: Path
    station_coords: Path
    processed_dir: Path
    output_dir: Path
    seq_len: int = 12
    k_neighbors: int = 6
    rain_threshold_mm: float = 1.0
    train_years: tuple[int, int] = (1950, 2014)
    val_years: tuple[int, int] = (2015, 2019)
    test_years: tuple[int, int] = (2020, 2023)
    spatial_holdout_fraction: float = 0.2
    spatial_holdout_seed: int = 42
    hidden_dim: int = 64
    num_gat_heads: int = 4
    num_gat_layers: int = 1
    dropout: float = 0.2
    conv_channels: int = 32
    model: str = "st_gat_attn"
    split: str = "temporal"
    batch_size: int = 16
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 8
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"
    loss_weight_tmax: float = 1.0
    loss_weight_tmin: float = 1.0
    loss_weight_humidity: float = 1.0
    loss_weight_rain_occurrence: float = 1.0
    loss_weight_rain_amount: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path, overrides: dict[str, Any] | None = None) -> "Config":
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if overrides:
            raw.update(overrides)
        base = Path(__file__).resolve().parent.parent
        return cls(
            data_dir=_resolve_path(base, raw["data_dir"]),
            station_coords=_resolve_path(base, raw["station_coords"]),
            processed_dir=_resolve_path(base, raw["processed_dir"]),
            output_dir=_resolve_path(base, raw["output_dir"]),
            seq_len=int(raw["seq_len"]),
            k_neighbors=int(raw["k_neighbors"]),
            rain_threshold_mm=float(raw["rain_threshold_mm"]),
            train_years=tuple(raw["train_years"]),
            val_years=tuple(raw["val_years"]),
            test_years=tuple(raw["test_years"]),
            spatial_holdout_fraction=float(raw["spatial_holdout_fraction"]),
            spatial_holdout_seed=int(raw["spatial_holdout_seed"]),
            hidden_dim=int(raw["hidden_dim"]),
            num_gat_heads=int(raw["num_gat_heads"]),
            num_gat_layers=int(raw["num_gat_layers"]),
            dropout=float(raw["dropout"]),
            conv_channels=int(raw["conv_channels"]),
            model=str(raw["model"]),
            split=str(raw["split"]),
            batch_size=int(raw["batch_size"]),
            epochs=int(raw["epochs"]),
            learning_rate=float(raw["learning_rate"]),
            weight_decay=float(raw["weight_decay"]),
            patience=int(raw["patience"]),
            num_workers=int(raw["num_workers"]),
            seed=int(raw["seed"]),
            device=str(raw["device"]),
            loss_weight_tmax=float(raw["loss_weight_tmax"]),
            loss_weight_tmin=float(raw["loss_weight_tmin"]),
            loss_weight_humidity=float(raw["loss_weight_humidity"]),
            loss_weight_rain_occurrence=float(raw["loss_weight_rain_occurrence"]),
            loss_weight_rain_amount=float(raw["loss_weight_rain_amount"]),
            extra={k: v for k, v in raw.items() if k not in _KNOWN_KEYS},
        )

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


_KNOWN_KEYS = {
    "data_dir",
    "station_coords",
    "processed_dir",
    "output_dir",
    "seq_len",
    "k_neighbors",
    "rain_threshold_mm",
    "train_years",
    "val_years",
    "test_years",
    "spatial_holdout_fraction",
    "spatial_holdout_seed",
    "hidden_dim",
    "num_gat_heads",
    "num_gat_layers",
    "dropout",
    "conv_channels",
    "model",
    "split",
    "batch_size",
    "epochs",
    "learning_rate",
    "weight_decay",
    "patience",
    "num_workers",
    "seed",
    "device",
    "loss_weight_tmax",
    "loss_weight_tmin",
    "loss_weight_humidity",
    "loss_weight_rain_occurrence",
    "loss_weight_rain_amount",
}


def _resolve_path(base: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

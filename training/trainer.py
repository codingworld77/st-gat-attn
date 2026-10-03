from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

from st_gat_attn.config import Config, set_seed
from st_gat_attn.data.dataset import DataLoaders
from st_gat_attn.models.architectures import build_model
from st_gat_attn.training.losses import multi_target_loss
from st_gat_attn.training.metrics import evaluate_predictions


@dataclass
class TrainResult:
    run_dir: Path
    best_val_loss: float
    test_metrics: dict[str, float]


class Trainer:
    def __init__(self, config: Config, loaders: DataLoaders, run_dir: Path) -> None:
        self.config = config
        self.loaders = loaders
        self.run_dir = run_dir
        self.device = config.resolve_device()
        set_seed(config.seed)

        self.model = build_model(
            model_name=config.model,
            num_features=loaders.num_features,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            num_gat_heads=config.num_gat_heads,
            num_gat_layers=config.num_gat_layers,
            conv_channels=config.conv_channels,
            neighbor_index=loaders.neighbor_index,
            neighbor_mask=loaders.neighbor_mask,
        ).to(self.device)

        self.optimizer = Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def _step_batch(self, batch: dict[str, torch.Tensor], train: bool) -> tuple[torch.Tensor, dict[str, float]]:
        batch = {key: value.to(self.device) for key, value in batch.items()}
        if train:
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.model.eval()

        with torch.set_grad_enabled(train):
            preds = self.model(batch["x"])
            loss, components = multi_target_loss(preds, batch, self.config)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
        return loss, components

    def _run_epoch(self, loader: torch.utils.data.DataLoader, train: bool) -> tuple[float, dict[str, float]]:
        losses: list[float] = []
        component_sums: dict[str, float] = {}
        for batch in loader:
            loss, components = self._step_batch(batch, train=train)
            losses.append(float(loss.detach().cpu()))
            for key, value in components.items():
                component_sums[key] = component_sums.get(key, 0.0) + value
        count = max(len(loader), 1)
        avg_components = {key: value / count for key, value in component_sums.items()}
        return float(np.mean(losses)), avg_components

    def _collect_predictions(self, loader: torch.utils.data.DataLoader) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        self.model.eval()
        pred_chunks: dict[str, list[np.ndarray]] = {
            "reg": [],
            "rain_logit": [],
            "rain_log_amount": [],
        }
        batch_chunks: dict[str, list[np.ndarray]] = {
            "y_reg": [],
            "y_rain_occurred": [],
            "y_rainfall_raw": [],
            "station_mask": [],
        }
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                preds = self.model(batch["x"])
                reg = torch.stack([preds["tmax"], preds["tmin"], preds["humidity"]], dim=-1)
                pred_chunks["reg"].append(reg.cpu().numpy())
                pred_chunks["rain_logit"].append(preds["rain_logit"].cpu().numpy())
                pred_chunks["rain_log_amount"].append(preds["rain_log_amount"].cpu().numpy())
                for key in batch_chunks:
                    batch_chunks[key].append(batch[key].cpu().numpy())

        merged_preds = {key: np.concatenate(values, axis=0) for key, values in pred_chunks.items()}
        merged_batches = {key: np.concatenate(values, axis=0) for key, values in batch_chunks.items()}
        return merged_preds, merged_batches

    def train(self) -> TrainResult:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        history: list[dict[str, float | int]] = []
        best_val = float("inf")
        patience_counter = 0
        best_path = self.run_dir / "best_model.pt"

        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_components = self._run_epoch(self.loaders.train, train=True)
            val_loss, val_components = self._run_epoch(self.loaders.val, train=False)
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **{f"train_{k}": v for k, v in train_components.items()},
                **{f"val_{k}": v for k, v in val_components.items()},
            }
            history.append(record)
            print(
                f"Epoch {epoch:03d} | train={train_loss:.4f} | val={val_loss:.4f} | "
                f"device={self.device}"
            )

            if val_loss < best_val:
                best_val = val_loss
                patience_counter = 0
                torch.save(
                    {
                        "model_state": self.model.state_dict(),
                        "config": self.config.__dict__,
                        "epoch": epoch,
                        "val_loss": val_loss,
                    },
                    best_path,
                )
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state"])

        preds, batches = self._collect_predictions(self.loaders.test)
        test_metrics = evaluate_predictions(
            preds,
            batches,
            target_mean=self.loaders.target_scaler_mean,
            target_std=self.loaders.target_scaler_std,
        )

        (self.run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (self.run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
        return TrainResult(run_dir=self.run_dir, best_val_loss=best_val, test_metrics=test_metrics)

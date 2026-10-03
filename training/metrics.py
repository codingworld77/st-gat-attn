from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score


def _masked_values(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = mask.astype(bool)
    return pred[valid], target[valid]


def regression_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    p, t = _masked_values(pred, target, mask)
    if len(p) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "mape": float("nan")}
    mae = mean_absolute_error(t, p)
    rmse = mean_squared_error(t, p, squared=False)
    r2 = r2_score(t, p) if len(np.unique(t)) > 1 else float("nan")
    mape = float(np.mean(np.abs((t - p) / np.clip(np.abs(t), 1e-6, None))) * 100)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2), "mape": float(mape)}


def rain_metrics(
    rain_logits: np.ndarray,
    rain_log_amount: np.ndarray,
    rain_occurred: np.ndarray,
    rain_amount: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    valid = mask.astype(bool)
    if valid.sum() == 0:
        return {
            "rain_f1": float("nan"),
            "rain_precision": float("nan"),
            "rain_recall": float("nan"),
            "rain_amount_mae": float("nan"),
        }

    probs = 1.0 / (1.0 + np.exp(-rain_logits[valid]))
    pred_occurred = (probs >= 0.5).astype(int)
    true_occurred = rain_occurred[valid].astype(int)
    f1 = f1_score(true_occurred, pred_occurred, zero_division=0)
    tp = ((pred_occurred == 1) & (true_occurred == 1)).sum()
    fp = ((pred_occurred == 1) & (true_occurred == 0)).sum()
    fn = ((pred_occurred == 0) & (true_occurred == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    positive = valid & (rain_occurred >= 0.5)
    if positive.sum() == 0:
        amount_mae = float("nan")
    else:
        pred_amount = np.expm1(rain_log_amount[positive])
        true_amount = rain_amount[positive]
        amount_mae = mean_absolute_error(true_amount, pred_amount)

    return {
        "rain_f1": float(f1),
        "rain_precision": float(precision),
        "rain_recall": float(recall),
        "rain_amount_mae": float(amount_mae),
    }


def evaluate_predictions(
    preds: dict[str, np.ndarray],
    batch: dict[str, np.ndarray],
    target_mean: np.ndarray | None = None,
    target_std: np.ndarray | None = None,
) -> dict[str, float]:
    mask = batch["station_mask"]
    metrics: dict[str, float] = {}
    for name, idx in [("tmax", 0), ("tmin", 1), ("humidity", 2)]:
        pred = preds["reg"][:, :, idx]
        target = batch["y_reg"][:, :, idx]
        reg_metrics = regression_metrics(pred, target, mask)
        for key, value in reg_metrics.items():
            metrics[f"{name}_{key}"] = value

        if target_mean is not None and target_std is not None:
            pred_raw = pred * target_std[idx] + target_mean[idx]
            target_raw = target * target_std[idx] + target_mean[idx]
            raw_metrics = regression_metrics(pred_raw, target_raw, mask)
            for key, value in raw_metrics.items():
                metrics[f"{name}_{key}_raw"] = value

    metrics.update(
        rain_metrics(
            preds["rain_logit"],
            preds["rain_log_amount"],
            batch["y_rain_occurred"],
            batch["y_rainfall_raw"],
            mask,
        )
    )
    return metrics

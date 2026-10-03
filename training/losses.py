from __future__ import annotations

import torch
import torch.nn.functional as F

from st_gat_attn.config import Config


def multi_target_loss(
    preds: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: Config,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = batch["station_mask"]
    mask_sum = mask.sum().clamp(min=1.0)

    reg_pred = torch.stack([preds["tmax"], preds["tmin"], preds["humidity"]], dim=-1)
    reg_target = batch["y_reg"]
    reg_loss = (((reg_pred - reg_target) ** 2) * mask.unsqueeze(-1)).sum() / (mask_sum * 3)

    rain_logits = preds["rain_logit"]
    rain_occurred = batch["y_rain_occurred"]
    bce = F.binary_cross_entropy_with_logits(rain_logits, rain_occurred, reduction="none")
    rain_cls_loss = (bce * mask).sum() / mask_sum

    positive_mask = mask * rain_occurred
    positive_count = positive_mask.sum().clamp(min=1.0)
    rain_amount_loss = (
        ((preds["rain_log_amount"] - batch["y_rain_amount"]) ** 2) * positive_mask
    ).sum() / positive_count

    total = (
        (config.loss_weight_tmax + config.loss_weight_tmin + config.loss_weight_humidity) * reg_loss / 3.0
        + config.loss_weight_rain_occurrence * rain_cls_loss
        + config.loss_weight_rain_amount * rain_amount_loss
    )

    components = {
        "loss_total": float(total.detach().cpu()),
        "loss_regression": float(reg_loss.detach().cpu()),
        "loss_rain_cls": float(rain_cls_loss.detach().cpu()),
        "loss_rain_amount": float(rain_amount_loss.detach().cpu()),
    }
    return total, components

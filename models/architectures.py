from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # sequence: (B, T, H)
        scores = self.proj(sequence).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        context = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return context, weights


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        assert out_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.query = nn.Linear(in_dim, out_dim)
        self.key = nn.Linear(in_dim, out_dim)
        self.value = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        neighbor_index: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        # x: (B, N, F)
        bsz, num_nodes, _ = x.shape
        neighbors = neighbor_index.clamp(min=0)
        nbr_x = x[:, neighbors]  # (B, N, K, F)
        k = nbr_x.size(2)

        q = self.query(x).view(bsz, num_nodes, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_proj = self.key(nbr_x).view(bsz, num_nodes, k, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        v_proj = self.value(nbr_x).view(bsz, num_nodes, k, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)

        q = q.unsqueeze(3)
        scores = (q * k_proj).sum(-1) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(~neighbor_mask.unsqueeze(0).unsqueeze(1), -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn.unsqueeze(-1) * v_proj).sum(dim=3)
        out = out.permute(0, 2, 1, 3).contiguous().view(bsz, num_nodes, -1)
        return self.out_proj(out)


class TemporalBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.attn = TemporalAttention(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        # sequence: (B, T, F)
        hidden = self.dropout(torch.relu(self.input_proj(sequence)))
        lstm_out, _ = self.lstm(hidden)
        context, _ = self.attn(lstm_out)
        return context


class MultiTargetHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.tmax = nn.Linear(hidden_dim, 1)
        self.tmin = nn.Linear(hidden_dim, 1)
        self.humidity = nn.Linear(hidden_dim, 1)
        self.rain_occurrence = nn.Linear(hidden_dim, 1)
        self.rain_amount = nn.Linear(hidden_dim, 1)

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "tmax": self.tmax(hidden).squeeze(-1),
            "tmin": self.tmin(hidden).squeeze(-1),
            "humidity": self.humidity(hidden).squeeze(-1),
            "rain_logit": self.rain_occurrence(hidden).squeeze(-1),
            "rain_log_amount": self.rain_amount(hidden).squeeze(-1),
        }


class StationIndependentModel(nn.Module):
    """B1: BiLSTM + temporal attention, no spatial component."""

    def __init__(self, num_features: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.backbone = TemporalBackbone(num_features, hidden_dim, dropout)
        self.head = MultiTargetHead(hidden_dim)

    def forward(self, x: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        # x: (B, T, N, F)
        bsz, seq_len, num_stations, num_features = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(bsz * num_stations, seq_len, num_features)
        hidden = self.backbone(x_flat)
        preds = self.head(hidden)
        return {key: value.view(bsz, num_stations) for key, value in preds.items()}


class NaiveSpatialModel(nn.Module):
    """B2: neighbor feature averaging + temporal backbone."""

    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        dropout: float,
        neighbor_index: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("neighbor_index", neighbor_index)
        self.register_buffer("neighbor_mask", neighbor_mask)
        self.backbone = TemporalBackbone(num_features * 2, hidden_dim, dropout)
        self.head = MultiTargetHead(hidden_dim)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        neighbors = self.neighbor_index.clamp(min=0)
        nbr = x[:, :, neighbors]  # (B, T, N, K, F)
        mask = self.neighbor_mask.float()[None, None, :, :, None]
        nbr_mean = (nbr * mask).sum(dim=3) / mask.sum(dim=3).clamp(min=1.0)
        return torch.cat([x, nbr_mean], dim=-1)

    def forward(self, x: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        x_aug = self._augment(x)
        bsz, seq_len, num_stations, num_features = x_aug.shape
        x_flat = x_aug.permute(0, 2, 1, 3).reshape(bsz * num_stations, seq_len, num_features)
        hidden = self.backbone(x_flat)
        preds = self.head(hidden)
        return {key: value.view(bsz, num_stations) for key, value in preds.items()}


class STGATAttnModel(nn.Module):
    """Headline model: GAT over stations at each timestep + temporal backbone."""

    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        neighbor_index: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("neighbor_index", neighbor_index)
        self.register_buffer("neighbor_mask", neighbor_mask)
        self.input_proj = nn.Linear(num_features, hidden_dim)
        self.gat_layers = nn.ModuleList(
            [GraphAttentionLayer(hidden_dim, hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.backbone = TemporalBackbone(hidden_dim, hidden_dim, dropout)
        self.head = MultiTargetHead(hidden_dim)

    def forward(self, x: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        bsz, seq_len, num_stations, _ = x.shape
        spatial_seq = []
        for t in range(seq_len):
            h = torch.relu(self.input_proj(x[:, t]))
            for layer in self.gat_layers:
                h = torch.relu(layer(h, self.neighbor_index, self.neighbor_mask))
            spatial_seq.append(h)
        spatial_seq = torch.stack(spatial_seq, dim=1)

        flat = spatial_seq.permute(0, 2, 1, 3).reshape(bsz * num_stations, seq_len, -1)
        hidden = self.backbone(flat)
        preds = self.head(hidden)
        return {key: value.view(bsz, num_stations) for key, value in preds.items()}


class HTCLSTMAttnModel(nn.Module):
    """B3: Conv1D + BiLSTM + temporal attention (published baseline style)."""

    def __init__(self, num_features: int, hidden_dim: int, conv_channels: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv1d(num_features, conv_channels, kernel_size=3, padding=1)
        self.backbone = TemporalBackbone(conv_channels, hidden_dim, dropout)
        self.head = MultiTargetHead(hidden_dim)

    def forward(self, x: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        bsz, seq_len, num_stations, num_features = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(bsz * num_stations, num_features, seq_len)
        conv_out = torch.relu(self.conv(x_flat)).permute(0, 2, 1)
        hidden = self.backbone(conv_out)
        preds = self.head(hidden)
        return {key: value.view(bsz, num_stations) for key, value in preds.items()}


def build_model(
    model_name: str,
    num_features: int,
    hidden_dim: int,
    dropout: float,
    num_gat_heads: int,
    num_gat_layers: int,
    conv_channels: int,
    neighbor_index: torch.Tensor,
    neighbor_mask: torch.Tensor,
) -> nn.Module:
    name = model_name.lower().replace("-", "_")
    if name == "b1":
        return StationIndependentModel(num_features, hidden_dim, dropout)
    if name == "b2":
        return NaiveSpatialModel(num_features, hidden_dim, dropout, neighbor_index, neighbor_mask)
    if name == "b3":
        return HTCLSTMAttnModel(num_features, hidden_dim, conv_channels, dropout)
    if name in {"st_gat_attn", "stgatattn"}:
        return STGATAttnModel(
            num_features, hidden_dim, num_gat_heads, num_gat_layers, dropout, neighbor_index, neighbor_mask
        )
    if name in {"stgcn", "astgcn", "st_transformer", "sttransformer"}:
        from st_gat_attn.models.st_baselines import build_st_baseline

        return build_st_baseline(
            model_name=name,
            num_features=num_features,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_gat_heads=num_gat_heads,
            neighbor_index=neighbor_index,
            neighbor_mask=neighbor_mask,
        )
    raise ValueError(f"Unknown model: {model_name}")

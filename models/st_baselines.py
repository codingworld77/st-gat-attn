"""Spatiotemporal baselines: STGCN, ASTGCN, ST-Transformer.

Same I/O as the main pipeline:
  input  x: (B, T, N, F)
  output: dict[tmax, tmin, humidity, rain_logit, rain_log_amount] each (B, N)

Train with existing train.py:
  python train.py --model stgcn --split temporal --epochs 30
  python train.py --model astgcn --split spatial --epochs 30
  python train.py --model st_transformer --split temporal --epochs 30

Or run all three:
  python run_st_baselines.py --splits temporal spatial --epochs 30
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from st_gat_attn.models.architectures import MultiTargetHead


def adjacency_from_neighbors(
    neighbor_index: torch.Tensor,
    neighbor_mask: torch.Tensor,
    num_nodes: int,
    add_self_loops: bool = True,
) -> torch.Tensor:
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    for i in range(num_nodes):
        for j, ok in zip(neighbor_index[i].tolist(), neighbor_mask[i].tolist()):
            if ok and j >= 0:
                adj[i, j] = 1.0
                adj[j, i] = 1.0
    if add_self_loops:
        adj.fill_diagonal_(1.0)
    return adj


def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    deg = adj.sum(dim=1).clamp(min=1.0)
    deg_inv_sqrt = deg.pow(-0.5)
    return deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)


class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        # x: (..., N, C), adj_norm: (N, N)
        support = self.linear(x)
        return torch.einsum("ij,...jc->...ic", adj_norm, support)


# ---------------------------------------------------------------------------
# STGCN
# ---------------------------------------------------------------------------
class TemporalGatedConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(channels, 2 * channels, kernel_size=(kernel_size, 1), padding=(padding, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, N)
        p, q = self.conv(x).chunk(2, dim=1)
        return p * torch.sigmoid(q)


class STGCNBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.temp1 = TemporalGatedConv(channels)
        self.gcn = GraphConv(channels, channels)
        self.temp2 = TemporalGatedConv(channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, N)
        h = self.temp1(x)
        h = self.gcn(h.permute(0, 2, 3, 1), adj_norm).permute(0, 3, 1, 2)
        h = self.temp2(h)
        h = self.norm(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return h + x


class STGCNBaseline(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        dropout: float,
        neighbor_index: torch.Tensor,
        neighbor_mask: torch.Tensor,
        num_blocks: int = 2,
    ) -> None:
        super().__init__()
        num_nodes = neighbor_index.size(0)
        adj = adjacency_from_neighbors(neighbor_index, neighbor_mask, num_nodes)
        self.register_buffer("adj_norm", normalize_adjacency(adj))
        self.in_proj = nn.Linear(num_features, hidden_dim)
        self.blocks = nn.ModuleList([STGCNBlock(hidden_dim) for _ in range(num_blocks)])
        self.dropout = nn.Dropout(dropout)
        self.head = MultiTargetHead(hidden_dim)

    def forward(self, x: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        h = self.in_proj(x).permute(0, 3, 1, 2)  # (B, C, T, N)
        for block in self.blocks:
            h = block(h, self.adj_norm)
        h = self.dropout(h.mean(dim=2).permute(0, 2, 1))  # (B, N, C)
        return self.head(h)


# ---------------------------------------------------------------------------
# ASTGCN
# ---------------------------------------------------------------------------
class SpatialAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.query = nn.Linear(channels, channels, bias=False)
        self.key = nn.Linear(channels, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, C) -> attention over nodes (B, N, N)
        xt = x.mean(dim=1)
        q = self.query(xt)
        k = self.key(xt)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(xt.size(-1))
        return torch.softmax(scores, dim=-1)


class TemporalAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.query = nn.Linear(channels, channels, bias=False)
        self.key = nn.Linear(channels, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, C) -> attention over time (B, T, T)
        xt = x.mean(dim=2)
        q = self.query(xt)
        k = self.key(xt)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(xt.size(-1))
        return torch.softmax(scores, dim=-1)


class ASTGCNBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.spatial_att = SpatialAttention(channels)
        self.temporal_att = TemporalAttention(channels)
        self.gcn = GraphConv(channels, channels)
        self.tconv = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0))
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, C)
        s_att = self.spatial_att(x)  # (B, N, N)
        adj = 0.5 * adj_norm.unsqueeze(0) + 0.5 * s_att
        x_s = torch.einsum("bij,btjc->btic", adj, x)
        x_s = self.gcn(x_s, torch.eye(x.size(2), device=x.device, dtype=x.dtype))

        t_att = self.temporal_att(x_s)  # (B, T, T)
        x_t = torch.einsum("bts,bsnc->btnc", t_att, x_s)

        y = self.tconv(x_t.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return self.norm(y + x)


class ASTGCNBaseline(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        dropout: float,
        neighbor_index: torch.Tensor,
        neighbor_mask: torch.Tensor,
        num_blocks: int = 2,
    ) -> None:
        super().__init__()
        num_nodes = neighbor_index.size(0)
        adj = adjacency_from_neighbors(neighbor_index, neighbor_mask, num_nodes)
        self.register_buffer("adj_norm", normalize_adjacency(adj))
        self.in_proj = nn.Linear(num_features, hidden_dim)
        self.blocks = nn.ModuleList([ASTGCNBlock(hidden_dim) for _ in range(num_blocks)])
        self.dropout = nn.Dropout(dropout)
        self.head = MultiTargetHead(hidden_dim)

    def forward(self, x: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h, self.adj_norm)
        h = self.dropout(h.mean(dim=1))
        return self.head(h)


# ---------------------------------------------------------------------------
# ST-Transformer
# ---------------------------------------------------------------------------
class STTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.spatial_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, n, c = x.shape
        xt = x.permute(0, 2, 1, 3).reshape(b * n, t, c)
        ta, _ = self.temporal_attn(xt, xt, xt)
        xt = self.norm1(xt + ta)

        xs = xt.view(b, n, t, c).permute(0, 2, 1, 3).reshape(b * t, n, c)
        sa, _ = self.spatial_attn(xs, xs, xs)
        xs = self.norm2(xs + sa)
        return self.norm3(xs + self.ff(xs)).view(b, t, n, c)


class STTransformerBaseline(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        num_nodes: int,
        num_layers: int = 2,
        max_seq_len: int = 24,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            num_heads = 1
        self.in_proj = nn.Linear(num_features, hidden_dim)
        self.time_pe = nn.Parameter(torch.zeros(1, max_seq_len, 1, hidden_dim))
        self.node_pe = nn.Parameter(torch.zeros(1, 1, num_nodes, hidden_dim))
        nn.init.trunc_normal_(self.time_pe, std=0.02)
        nn.init.trunc_normal_(self.node_pe, std=0.02)
        self.layers = nn.ModuleList(
            [STTransformerBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.head = MultiTargetHead(hidden_dim)

    def forward(self, x: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        _, t, n, _ = x.shape
        h = self.in_proj(x) + self.time_pe[:, :t] + self.node_pe[:, :, :n]
        for layer in self.layers:
            h = layer(h)
        return self.head(self.dropout(h.mean(dim=1)))


def build_st_baseline(
    model_name: str,
    num_features: int,
    hidden_dim: int,
    dropout: float,
    num_gat_heads: int,
    neighbor_index: torch.Tensor,
    neighbor_mask: torch.Tensor,
) -> nn.Module:
    name = model_name.lower().replace("-", "_")
    if name == "stgcn":
        return STGCNBaseline(num_features, hidden_dim, dropout, neighbor_index, neighbor_mask)
    if name == "astgcn":
        return ASTGCNBaseline(num_features, hidden_dim, dropout, neighbor_index, neighbor_mask)
    if name in {"st_transformer", "sttransformer"}:
        return STTransformerBaseline(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_heads=max(1, num_gat_heads),
            dropout=dropout,
            num_nodes=neighbor_index.size(0),
        )
    raise ValueError(f"Unknown ST baseline: {model_name}")

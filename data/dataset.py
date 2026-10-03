from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from st_gat_attn.data.preprocessing import ProcessedDataBundle, TARGET_COLUMNS


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return float(2 * radius * np.arcsin(np.sqrt(a)))


def build_knn_graph(coords: np.ndarray, k_neighbors: int) -> tuple[torch.Tensor, np.ndarray]:
    num_nodes = coords.shape[0]
    dist = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                dist[i, j] = haversine_km(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])

    edge_src: list[int] = []
    edge_dst: list[int] = []
    if k_neighbors <= 0:
        # Ablation: self-loops only (no cross-station edges).
        for i in range(num_nodes):
            edge_src.append(i)
            edge_dst.append(i)
    else:
        for i in range(num_nodes):
            neighbors = np.argsort(dist[i])[:k_neighbors]
            for j in neighbors:
                edge_src.append(i)
                edge_dst.append(int(j))
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    return edge_index, dist


def build_neighbor_index(edge_index: torch.Tensor, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    neighbors: list[list[int]] = [[] for _ in range(num_nodes)]
    for src, dst in edge_index.t().tolist():
        neighbors[src].append(dst)
    max_neighbors = max(len(n) for n in neighbors)
    index = torch.full((num_nodes, max_neighbors), -1, dtype=torch.long)
    mask = torch.zeros((num_nodes, max_neighbors), dtype=torch.bool)
    for i, neigh in enumerate(neighbors):
        for j, node in enumerate(neigh):
            index[i, j] = node
            mask[i, j] = True
    return index, mask


@dataclass
class TensorPanel:
    features: np.ndarray  # (T, N, F)
    targets: np.ndarray  # (T, N, 4)
    years: np.ndarray  # (T,)
    months: np.ndarray  # (T,)
    station_mask: np.ndarray  # (N,) bool


def panel_to_tensor(bundle: ProcessedDataBundle) -> TensorPanel:
    panel = bundle.panel.copy()
    panel["station_idx"] = panel["station"].map(bundle.station_to_idx).astype(int)
    panel = panel.sort_values(["year", "month", "station_idx"])

    times = panel[["year", "month"]].drop_duplicates().sort_values(["year", "month"])
    time_keys = list(map(tuple, times.to_numpy()))
    time_to_idx = {key: idx for idx, key in enumerate(time_keys)}

    num_times = len(time_keys)
    num_stations = len(bundle.stations)
    num_features = len(bundle.feature_columns)
    num_targets = len(bundle.target_columns)

    features = np.full((num_times, num_stations, num_features), np.nan, dtype=np.float32)
    targets = np.full((num_times, num_stations, num_targets), np.nan, dtype=np.float32)
    station_seen = np.zeros(num_stations, dtype=bool)

    for row in panel.itertuples(index=False):
        t_idx = time_to_idx[(int(row.year), int(row.month))]
        s_idx = int(row.station_idx)
        features[t_idx, s_idx] = [getattr(row, col) for col in bundle.feature_columns]
        targets[t_idx, s_idx] = [getattr(row, col) for col in bundle.target_columns]
        station_seen[s_idx] = True

    years = np.array([key[0] for key in time_keys], dtype=np.int32)
    months = np.array([key[1] for key in time_keys], dtype=np.int32)
    return TensorPanel(features, targets, years, months, station_seen)


class ClimateSequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tensor_panel: TensorPanel,
        seq_len: int,
        time_mask: np.ndarray,
        active_station_mask: np.ndarray,
        rain_threshold: float,
        rainfall_raw: np.ndarray | None = None,
    ) -> None:
        self.features = tensor_panel.features
        self.targets = tensor_panel.targets
        # Rain labels must use millimetres, not z-scored rainfall.
        self.rainfall_raw = rainfall_raw if rainfall_raw is not None else tensor_panel.targets[:, :, 3]
        self.seq_len = seq_len
        self.time_mask = time_mask
        self.active_station_mask = active_station_mask
        self.rain_threshold = rain_threshold
        self.indices: list[int] = []

        for t_idx in range(seq_len, len(time_mask)):
            if time_mask[t_idx]:
                self.indices.append(t_idx)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        t_idx = self.indices[index]
        x = self.features[t_idx - self.seq_len : t_idx]
        y = self.targets[t_idx]
        rainfall_mm = self.rainfall_raw[t_idx]

        valid = self.active_station_mask & ~np.isnan(rainfall_mm) & ~np.isnan(y[:, 0])
        station_mask = valid.astype(np.float32)

        rain_occurred = (np.nan_to_num(rainfall_mm, nan=0.0) >= self.rain_threshold).astype(np.float32)
        rain_log_amount = np.log1p(np.clip(np.nan_to_num(rainfall_mm, nan=0.0), 0.0, None))

        x = np.nan_to_num(x, nan=0.0)
        y_reg = np.nan_to_num(y[:, :3], nan=0.0)

        return {
            "x": torch.from_numpy(x.astype(np.float32)),
            "y_reg": torch.from_numpy(y_reg.astype(np.float32)),
            "y_rain_occurred": torch.from_numpy(rain_occurred.astype(np.float32)),
            "y_rain_amount": torch.from_numpy(rain_log_amount.astype(np.float32)),
            "y_rainfall_raw": torch.from_numpy(np.nan_to_num(rainfall_mm, nan=0.0).astype(np.float32)),
            "station_mask": torch.from_numpy(station_mask),
        }


@dataclass
class DataLoaders:
    train: torch.utils.data.DataLoader
    val: torch.utils.data.DataLoader
    test: torch.utils.data.DataLoader
    feature_scaler_mean: np.ndarray
    feature_scaler_std: np.ndarray
    target_scaler_mean: np.ndarray
    target_scaler_std: np.ndarray
    edge_index: torch.Tensor
    neighbor_index: torch.Tensor
    neighbor_mask: torch.Tensor
    num_features: int
    num_stations: int


def _fit_scalers(features: np.ndarray, targets: np.ndarray, train_time_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_features = features[train_time_mask]
    train_targets = targets[train_time_mask]
    feature_mean = np.nanmean(train_features, axis=(0, 1))
    feature_std = np.nanstd(train_features, axis=(0, 1))
    feature_std[feature_std < 1e-6] = 1.0
    target_mean = np.nanmean(train_targets, axis=(0, 1))
    target_std = np.nanstd(train_targets, axis=(0, 1))
    target_std[target_std < 1e-6] = 1.0
    return feature_mean, feature_std, target_mean, target_std


def _time_mask(years: np.ndarray, year_range: tuple[int, int]) -> np.ndarray:
    return (years >= year_range[0]) & (years <= year_range[1])


def _station_mask_for_split(
    split: str,
    num_stations: int,
    emergent_indices: list[int],
    spatial_holdout_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_active = np.ones(num_stations, dtype=bool)
    if split == "spatial":
        holdout = np.zeros(num_stations, dtype=bool)
        holdout[spatial_holdout_indices] = True
        return ~holdout, ~holdout, holdout
    if split == "emergent":
        emergent = np.zeros(num_stations, dtype=bool)
        emergent[emergent_indices] = True
        return ~emergent, ~emergent, emergent
    return all_active, all_active, all_active


def build_dataloaders(
    bundle: ProcessedDataBundle,
    seq_len: int,
    split: str,
    train_years: tuple[int, int],
    val_years: tuple[int, int],
    test_years: tuple[int, int],
    batch_size: int,
    k_neighbors: int,
    rain_threshold: float,
    num_workers: int,
) -> tuple[DataLoaders, dict]:
    tensor_panel = panel_to_tensor(bundle)
    train_time = _time_mask(tensor_panel.years, train_years)
    feature_mean, feature_std, target_mean, target_std = _fit_scalers(
        tensor_panel.features, tensor_panel.targets, train_time
    )

    features = (tensor_panel.features - feature_mean) / feature_std
    targets = (tensor_panel.targets - target_mean) / target_std
    rainfall_raw = tensor_panel.targets[:, :, 3].copy()
    scaled_panel = TensorPanel(features, targets, tensor_panel.years, tensor_panel.months, tensor_panel.station_mask)

    emergent_idx = [bundle.station_to_idx[s] for s in bundle.emergent_stations]
    spatial_idx = [bundle.station_to_idx[s] for s in bundle.spatial_holdout_stations]
    train_station_mask, val_station_mask, test_station_mask = _station_mask_for_split(
        split, len(bundle.stations), emergent_idx, spatial_idx
    )

    train_active = scaled_panel.station_mask & train_station_mask
    val_active = scaled_panel.station_mask & val_station_mask
    test_active = scaled_panel.station_mask & test_station_mask

    time_train = _time_mask(scaled_panel.years, train_years)
    time_val = _time_mask(scaled_panel.years, val_years)
    time_test = _time_mask(scaled_panel.years, test_years)
    train_ds = ClimateSequenceDataset(scaled_panel, seq_len, time_train, train_active, rain_threshold, rainfall_raw)
    val_ds = ClimateSequenceDataset(scaled_panel, seq_len, time_val, val_active, rain_threshold, rainfall_raw)
    test_ds = ClimateSequenceDataset(scaled_panel, seq_len, time_test, test_active, rain_threshold, rainfall_raw)

    edge_index, _ = build_knn_graph(bundle.coords, k_neighbors)
    neighbor_index, neighbor_mask = build_neighbor_index(edge_index, len(bundle.stations))

    def _loader(ds: torch.utils.data.Dataset, shuffle: bool) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    scaler_meta = {
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
    }
    loaders = DataLoaders(
        train=_loader(train_ds, shuffle=True),
        val=_loader(val_ds, shuffle=False),
        test=_loader(test_ds, shuffle=False),
        feature_scaler_mean=feature_mean,
        feature_scaler_std=feature_std,
        target_scaler_mean=target_mean,
        target_scaler_std=target_std,
        edge_index=edge_index,
        neighbor_index=neighbor_index,
        neighbor_mask=neighbor_mask,
        num_features=len(bundle.feature_columns),
        num_stations=len(bundle.stations),
    )
    return loaders, scaler_meta

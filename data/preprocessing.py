from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MONTH_NAME_TO_NUM = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

VARIABLE_FILES = {
    "rainfall": ("Rainfall_Report.csv", "Total Rainfall (mm)"),
    "tmax": ("Maximum_Temperature_Report.csv", "Temperature (Deg.Cel)"),
    "tmin": ("Minimum_Temperature_Report.csv", "Temperature (Deg.Cel)"),
    "humidity": ("Humidity_Report.csv", "Humidity (percent)"),
    "cloud": ("Cloud_Cover_Report.csv", "Cloud Coverage (Octs)"),
    "wind": ("WindSpeed_Report.csv", "Wind Speed (m/s)"),
    "sunshine": ("Sunshine_Report.csv", "Sunshine (Hours)"),
    "solar": ("Solar_Radiation_Report.csv", "Solar Radiation"),
}

FEATURE_COLUMNS = [
    "tmax",
    "tmin",
    "humidity",
    "rainfall",
    "cloud",
    "wind",
    "sunshine",
    "solar",
    "month_sin",
    "month_cos",
]

TARGET_COLUMNS = ["tmax", "tmin", "humidity", "rainfall"]


@dataclass
class ProcessedDataBundle:
    panel: pd.DataFrame
    stations: list[str]
    station_to_idx: dict[str, int]
    feature_columns: list[str]
    target_columns: list[str]
    coords: np.ndarray
    emergent_stations: list[str]
    spatial_holdout_stations: list[str]


def extract_station_id(station: str) -> str | None:
    match = re.search(r"\((\d+)\)", str(station))
    return match.group(1) if match else str(station)


def normalize_month(series: pd.Series) -> pd.Series:
    def to_month(value) -> int | None:
        if pd.isna(value):
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            month = int(value)
            return month if 1 <= month <= 12 else None
        text = str(value).strip().lower()
        if text.isdigit():
            month = int(text)
            return month if 1 <= month <= 12 else None
        return MONTH_NAME_TO_NUM.get(text)

    return series.map(to_month)


def load_variable_frame(data_dir: Path, filename: str, value_col: str, var_name: str) -> pd.DataFrame:
    path = data_dir / filename
    df = pd.read_csv(path, sep="\t", skiprows=2, encoding="utf-8")
    df.columns = [str(col).strip() for col in df.columns]
    month = normalize_month(df["Month"])
    out = pd.DataFrame(
        {
            "station": df["Station"].astype(str),
            "year": pd.to_numeric(df["Year"], errors="coerce").astype("Int64"),
            "month": month.astype("Int64"),
            var_name: pd.to_numeric(df[value_col], errors="coerce"),
        }
    )
    if "Decade" in df.columns:
        out["decade"] = pd.to_numeric(df["Decade"], errors="coerce")
        out = (
            out.groupby(["station", "year", "month"], as_index=False)[var_name]
            .mean()
            .sort_values(["station", "year", "month"])
        )
    return out.dropna(subset=["year", "month"])


def build_panel(data_dir: Path) -> pd.DataFrame:
    frames = []
    for var_name, (filename, value_col) in VARIABLE_FILES.items():
        frames.append(load_variable_frame(data_dir, filename, value_col, var_name))
    panel = frames[0]
    for frame in frames[1:]:
        panel = panel.merge(frame, on=["station", "year", "month"], how="outer")
    panel = panel.sort_values(["station", "year", "month"]).reset_index(drop=True)
    panel["month_sin"] = np.sin(2 * np.pi * panel["month"].astype(float) / 12.0)
    panel["month_cos"] = np.cos(2 * np.pi * panel["month"].astype(float) / 12.0)
    return panel


def load_station_metadata(coords_path: Path, holdout_fraction: float, holdout_seed: int) -> tuple[pd.DataFrame, list[str], list[str]]:
    coords = pd.read_csv(coords_path)
    emergent = coords.loc[coords["emergent"].astype(bool), "station"].astype(str).tolist()
    non_emergent = coords.loc[~coords["emergent"].astype(bool), "station"].astype(str).tolist()
    rng = np.random.default_rng(holdout_seed)
    holdout_count = max(1, int(round(len(non_emergent) * holdout_fraction)))
    holdout_idx = rng.choice(len(non_emergent), size=holdout_count, replace=False)
    spatial_holdout = [non_emergent[i] for i in sorted(holdout_idx)]
    return coords, emergent, spatial_holdout


def prepare_processed_data(
    data_dir: Path,
    coords_path: Path,
    processed_dir: Path,
    holdout_fraction: float,
    holdout_seed: int,
) -> ProcessedDataBundle:
    processed_dir.mkdir(parents=True, exist_ok=True)
    panel = build_panel(data_dir)
    coords_df, emergent_stations, spatial_holdout_stations = load_station_metadata(
        coords_path, holdout_fraction, holdout_seed
    )
    stations = coords_df["station"].astype(str).tolist()
    station_to_idx = {station: idx for idx, station in enumerate(stations)}
    coords = coords_df[["latitude", "longitude"]].to_numpy(dtype=np.float32)

    numeric_cols = FEATURE_COLUMNS + TARGET_COLUMNS
    for col in numeric_cols:
        if col in panel.columns:
            panel[col] = panel.groupby("station")[col].transform(lambda s: s.interpolate(limit_direction="both"))
            panel[col] = panel.groupby("station")[col].transform(lambda s: s.ffill().bfill())

    panel = panel[panel["station"].isin(stations)].copy()
    panel.to_parquet(processed_dir / "panel.parquet", index=False)
    meta = {
        "stations": stations,
        "emergent_stations": emergent_stations,
        "spatial_holdout_stations": spatial_holdout_stations,
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
    }
    (processed_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    np.save(processed_dir / "station_coords.npy", coords)
    return ProcessedDataBundle(
        panel=panel,
        stations=stations,
        station_to_idx=station_to_idx,
        feature_columns=FEATURE_COLUMNS,
        target_columns=TARGET_COLUMNS,
        coords=coords,
        emergent_stations=emergent_stations,
        spatial_holdout_stations=spatial_holdout_stations,
    )


def load_processed_data(processed_dir: Path) -> ProcessedDataBundle:
    panel = pd.read_parquet(processed_dir / "panel.parquet")
    meta = json.loads((processed_dir / "metadata.json").read_text(encoding="utf-8"))
    coords = np.load(processed_dir / "station_coords.npy")
    stations = meta["stations"]
    return ProcessedDataBundle(
        panel=panel,
        stations=stations,
        station_to_idx={station: idx for idx, station in enumerate(stations)},
        feature_columns=meta["feature_columns"],
        target_columns=meta["target_columns"],
        coords=coords,
        emergent_stations=meta["emergent_stations"],
        spatial_holdout_stations=meta["spatial_holdout_stations"],
    )

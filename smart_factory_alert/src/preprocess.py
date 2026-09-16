"""Sensor data loading and reusable preprocessing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

FEATURES = ["temp", "pressure", "vibration"]
REQUIRED_COLUMNS = ["timestamp", *FEATURES]


def load_sensor_data(path: Union[str, Path]) -> pd.DataFrame:
    """Load and validate sensor data without fitting preprocessing objects."""
    df = pd.read_csv(path)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing columns: {', '.join(missing_columns)}")
    if "label" not in df.columns:
        df["label"] = "unknown"
    df = df.loc[:, [*REQUIRED_COLUMNS, "label"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for column in FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["label"] = df["label"].fillna("unknown").astype(str)
    if df["timestamp"].isna().any():
        raise ValueError("Timestamp contains missing or invalid values")
    if df[FEATURES].isna().all(axis=0).any():
        empty_features = df[FEATURES].columns[df[FEATURES].isna().all(axis=0)]
        raise ValueError(f"Sensor columns contain no numeric values: {', '.join(empty_features)}")
    return df.sort_values("timestamp").reset_index(drop=True)


def fit_preprocessor(df: pd.DataFrame) -> tuple[SimpleImputer, StandardScaler]:
    """Fit imputation and scaling objects on independent training data."""
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    imputed = imputer.fit_transform(df[FEATURES])
    scaler.fit(imputed)
    return imputer, scaler


def transform_sensor_data(
    df: pd.DataFrame,
    imputer: SimpleImputer,
    scaler: StandardScaler,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Transform sensor data with already-fitted preprocessing objects."""
    transformed_df = df.copy()
    imputed = imputer.transform(transformed_df[FEATURES])
    transformed_df[FEATURES] = imputed
    scaled = scaler.transform(imputed)
    return transformed_df, np.asarray(scaled, dtype=float)

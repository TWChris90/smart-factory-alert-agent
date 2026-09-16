"""Generate deterministic smart-factory sensor data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _generate_normal_values(
    rows: int,
    rng: np.random.Generator,
    start: str,
) -> pd.DataFrame:
    """Generate normal sensor values at a fixed one-minute interval."""
    timestamps = pd.date_range(start, periods=rows, freq="min")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "temp": np.clip(rng.normal(48.0, 1.0, rows), 45.0, 50.0),
            "pressure": np.clip(rng.normal(1.025, 0.008, rows), 1.00, 1.05),
            "vibration": np.clip(rng.normal(0.03, 0.004, rows), 0.02, 0.04),
        }
    )


def generate_normal_sensor_data(
    rows: int = 400,
    seed: int = 20240603,
) -> pd.DataFrame:
    """Generate an independent, normal-only baseline for model training."""
    if rows < 100:
        raise ValueError("normal training rows must be at least 100")
    rng = np.random.default_rng(seed)
    data = _generate_normal_values(rows, rng, "2024-05-01 00:00:00")
    data["label"] = "normal"
    return data


def generate_sensor_data(rows: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate seed-controlled detection data with random anomalies and gaps."""
    if not 100 <= rows <= 500:
        raise ValueError("rows must be between 100 and 500")

    rng = np.random.default_rng(seed)
    data = _generate_normal_values(rows, rng, "2024-06-03 19:05:00")

    anomaly_count = max(1, rows // 20)
    anomaly_idx = rng.choice(rows, size=anomaly_count, replace=False)
    anomaly_modes = rng.integers(0, 3, size=anomaly_count)
    for idx, mode in zip(anomaly_idx, anomaly_modes):
        if mode == 0:
            data.loc[idx, "temp"] = rng.uniform(52.5, 55.0)
        elif mode == 1:
            data.loc[idx, "pressure"] = rng.choice(
                [rng.uniform(1.09, 1.12), rng.uniform(0.94, 0.96)]
            )
        else:
            data.loc[idx, "vibration"] = rng.uniform(0.075, 0.095)

    data["label"] = "normal"
    data.loc[anomaly_idx, "label"] = "abnormal"

    normal_idx = np.setdiff1d(np.arange(rows), anomaly_idx)
    missing_count = min(len(normal_idx), max(1, rows // 100))
    if missing_count:
        missing_idx = rng.choice(normal_idx, size=missing_count, replace=False)
        missing_features = rng.choice(
            ["temp", "pressure", "vibration"], size=missing_count
        )
        for idx, feature in zip(missing_idx, missing_features):
            data.loc[idx, feature] = np.nan
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/sensor_data.csv"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate_sensor_data(args.rows, args.seed).to_csv(args.output, index=False)
    print(f"Generated {args.rows} rows -> {args.output}")


if __name__ == "__main__":
    main()

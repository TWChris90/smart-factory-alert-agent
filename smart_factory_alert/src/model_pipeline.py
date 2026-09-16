"""Persistent training and inference pipeline for sensor anomaly detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import joblib
import pandas as pd

if __package__:
    from .generate_data import generate_normal_sensor_data
    from .preprocess import (
        FEATURES,
        fit_preprocessor,
        load_sensor_data,
        transform_sensor_data,
    )
    from .pyod_detector import fit_detector, score_anomalies
else:
    from generate_data import generate_normal_sensor_data
    from preprocess import FEATURES, fit_preprocessor, load_sensor_data, transform_sensor_data
    from pyod_detector import fit_detector, score_anomalies

PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_DATA_PATH = PACKAGE_DIR / "data" / "normal_training_data.csv"
DEFAULT_MODEL_PATH = PACKAGE_DIR / "models" / "knn_normal_model.joblib"
DEFAULT_TRAINING_ROWS = 1000
DEFAULT_TRAINING_SEED = 20240603
DEFAULT_CONTAMINATION = 0.01
MODEL_VERSION = 1


def train_and_save_model(
    training_csv: Union[str, Path] = DEFAULT_TRAINING_DATA_PATH,
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    *,
    contamination: float = DEFAULT_CONTAMINATION,
    training_seed: int | None = None,
) -> dict[str, Any]:
    """Train on a normal-only CSV and persist preprocessing, KNN, and threshold."""
    training_csv = Path(training_csv)
    model_path = Path(model_path)
    training_df = load_sensor_data(training_csv)
    labels = set(training_df["label"].str.lower())
    if labels != {"normal"}:
        raise ValueError("Training data must contain only rows labeled 'normal'")

    imputer, scaler = fit_preprocessor(training_df)
    _, X_train = transform_sensor_data(training_df, imputer, scaler)
    detector, score_threshold = fit_detector(X_train, contamination=contamination)
    model_bundle: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "features": FEATURES,
        "imputer": imputer,
        "scaler": scaler,
        "detector": detector,
        "score_threshold": score_threshold,
        "contamination": contamination,
        "training_rows": len(training_df),
        "training_seed": training_seed,
        "training_data_path": str(training_csv),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, model_path)
    return model_bundle


def load_saved_model(
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    """Load and validate a previously fitted model bundle."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Saved anomaly model not found: {model_path}")
    model_bundle = joblib.load(model_path)
    if model_bundle.get("model_version") != MODEL_VERSION:
        raise ValueError("Saved anomaly model version is not supported")
    if model_bundle.get("features") != FEATURES:
        raise ValueError("Saved anomaly model uses different sensor features")
    return model_bundle


def ensure_default_model() -> dict[str, Any]:
    """Create the independent normal baseline/model once, then only load it."""
    if DEFAULT_MODEL_PATH.exists():
        return load_saved_model(DEFAULT_MODEL_PATH)

    DEFAULT_TRAINING_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DEFAULT_TRAINING_DATA_PATH.exists():
        normal_data = generate_normal_sensor_data(
            rows=DEFAULT_TRAINING_ROWS,
            seed=DEFAULT_TRAINING_SEED,
        )
        normal_data.to_csv(DEFAULT_TRAINING_DATA_PATH, index=False)
    return train_and_save_model(
        DEFAULT_TRAINING_DATA_PATH,
        DEFAULT_MODEL_PATH,
        contamination=DEFAULT_CONTAMINATION,
        training_seed=DEFAULT_TRAINING_SEED,
    )


def detect_with_saved_model(
    csv_path: Union[str, Path],
    model_bundle: dict[str, Any],
):
    """Transform and score unseen CSV rows without fitting any model component."""
    df = load_sensor_data(csv_path)
    return detect_sensor_data(df, model_bundle)


def detect_sensor_data(
    df: pd.DataFrame,
    model_bundle: dict[str, Any],
):
    """Transform and score an already-loaded sensor frame without fitting."""
    missing_mask = df[FEATURES].isna()
    missing_summary = {
        "total": int(missing_mask.sum().sum()),
        "rows": int(missing_mask.any(axis=1).sum()),
        "by_feature": {
            feature: int(missing_mask[feature].sum()) for feature in FEATURES
        },
    }
    transformed_df, X = transform_sensor_data(
        df,
        model_bundle["imputer"],
        model_bundle["scaler"],
    )
    transformed_df.attrs["missing_summary"] = missing_summary
    scores, labels = score_anomalies(
        model_bundle["detector"],
        X,
        float(model_bundle["score_threshold"]),
    )
    return transformed_df, scores, labels

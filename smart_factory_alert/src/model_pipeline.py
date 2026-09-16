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
    from .pyod_detector import (
        calibrate_score_threshold,
        fit_detector,
        score_anomalies,
    )
else:
    from generate_data import generate_normal_sensor_data
    from preprocess import FEATURES, fit_preprocessor, load_sensor_data, transform_sensor_data
    from pyod_detector import calibrate_score_threshold, fit_detector, score_anomalies

PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_DATA_PATH = PACKAGE_DIR / "data" / "normal_training_data.csv"
DEFAULT_CALIBRATION_DATA_PATH = PACKAGE_DIR / "data" / "normal_calibration_data.csv"
DEFAULT_MODEL_PATH = PACKAGE_DIR / "models" / "knn_normal_model.joblib"
DEFAULT_TRAINING_ROWS = 400
DEFAULT_CALIBRATION_ROWS = 100
DEFAULT_TRAINING_SEED = 20240603
DEFAULT_CALIBRATION_SEED = 20240604
DEFAULT_CONTAMINATION = 0.01
DEFAULT_CALIBRATION_PERCENTILE = 99.5
MODEL_VERSION = 2


def train_and_save_model(
    training_csv: Union[str, Path] = DEFAULT_TRAINING_DATA_PATH,
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    *,
    calibration_csv: Union[str, Path] = DEFAULT_CALIBRATION_DATA_PATH,
    contamination: float = DEFAULT_CONTAMINATION,
    calibration_percentile: float = DEFAULT_CALIBRATION_PERCENTILE,
    training_seed: int | None = None,
    calibration_seed: int | None = None,
) -> dict[str, Any]:
    """Fit on normal training data and calibrate on separate normal data."""
    training_csv = Path(training_csv)
    calibration_csv = Path(calibration_csv)
    model_path = Path(model_path)
    training_df = load_sensor_data(training_csv)
    calibration_df = load_sensor_data(calibration_csv)
    training_labels = set(training_df["label"].str.lower())
    calibration_labels = set(calibration_df["label"].str.lower())
    if training_labels != {"normal"}:
        raise ValueError("Training data must contain only rows labeled 'normal'")
    if calibration_labels != {"normal"}:
        raise ValueError("Calibration data must contain only rows labeled 'normal'")

    imputer, scaler = fit_preprocessor(training_df)
    _, X_train = transform_sensor_data(training_df, imputer, scaler)
    _, X_calibration = transform_sensor_data(calibration_df, imputer, scaler)
    detector = fit_detector(X_train, contamination=contamination)
    score_threshold = calibrate_score_threshold(
        detector,
        X_calibration,
        percentile=calibration_percentile,
    )
    model_bundle: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "features": FEATURES,
        "imputer": imputer,
        "scaler": scaler,
        "detector": detector,
        "score_threshold": score_threshold,
        "contamination": contamination,
        "calibration_percentile": calibration_percentile,
        "training_rows": len(training_df),
        "calibration_rows": len(calibration_df),
        "training_seed": training_seed,
        "calibration_seed": calibration_seed,
        "training_data_path": str(training_csv),
        "calibration_data_path": str(calibration_csv),
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
    """Create the training/calibration data and model once, then only load it."""
    if DEFAULT_MODEL_PATH.exists():
        try:
            return load_saved_model(DEFAULT_MODEL_PATH)
        except ValueError:
            # Rebuild an outdated default bundle after a pipeline version change.
            pass

    DEFAULT_TRAINING_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    training_data = generate_normal_sensor_data(
        rows=DEFAULT_TRAINING_ROWS,
        seed=DEFAULT_TRAINING_SEED,
    )
    training_data.to_csv(DEFAULT_TRAINING_DATA_PATH, index=False)
    calibration_data = generate_normal_sensor_data(
        rows=DEFAULT_CALIBRATION_ROWS,
        seed=DEFAULT_CALIBRATION_SEED,
    )
    calibration_data.to_csv(DEFAULT_CALIBRATION_DATA_PATH, index=False)
    return train_and_save_model(
        DEFAULT_TRAINING_DATA_PATH,
        DEFAULT_MODEL_PATH,
        calibration_csv=DEFAULT_CALIBRATION_DATA_PATH,
        contamination=DEFAULT_CONTAMINATION,
        calibration_percentile=DEFAULT_CALIBRATION_PERCENTILE,
        training_seed=DEFAULT_TRAINING_SEED,
        calibration_seed=DEFAULT_CALIBRATION_SEED,
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

"""Training and inference primitives for the PyOD KNN detector."""

from __future__ import annotations

import numpy as np
from pyod.models.knn import KNN


def fit_detector(
    X: np.ndarray,
    contamination: float = 0.01,
) -> KNN:
    """Fit KNN only on the independent normal training split."""
    detector = KNN(
        n_neighbors=5,
        contamination=contamination,
        n_jobs=-1,
    )
    detector.fit(X)
    return detector


def calibrate_score_threshold(
    detector: KNN,
    X_calibration: np.ndarray,
    percentile: float = 99.5,
) -> float:
    """Set the alert threshold from a separate normal calibration split."""
    if not 0.0 < percentile <= 100.0:
        raise ValueError("calibration percentile must be between 0 and 100")
    calibration_scores = np.asarray(
        detector.decision_function(X_calibration),
        dtype=float,
    )
    return float(np.percentile(calibration_scores, percentile))


def score_anomalies(
    detector: KNN,
    X: np.ndarray,
    score_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Score unseen data without refitting the saved detector."""
    scores = np.asarray(detector.decision_function(X), dtype=float)
    labels = (scores > score_threshold).astype(int)
    return scores, labels

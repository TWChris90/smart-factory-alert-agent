"""Training and inference primitives for the PyOD KNN detector."""

from __future__ import annotations

import numpy as np
from pyod.models.knn import KNN


def fit_detector(
    X: np.ndarray,
    contamination: float = 0.01,
) -> tuple[KNN, float]:
    """Fit KNN on independent normal training data and return its threshold."""
    detector = KNN(
        n_neighbors=5,
        contamination=contamination,
        n_jobs=-1,
    )
    detector.fit(X)
    return detector, float(detector.threshold_)


def score_anomalies(
    detector: KNN,
    X: np.ndarray,
    score_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Score unseen data without refitting the saved detector."""
    scores = np.asarray(detector.decision_function(X), dtype=float)
    labels = (scores > score_threshold).astype(int)
    return scores, labels

"""Turn sensor and PyOD results into actionable factory alerts."""

from __future__ import annotations

from typing import Any

import pandas as pd

THRESHOLDS = {
    "temp_low": 43.0,
    "temp_high": 52.0,
    "pressure_low": 0.97,
    "pressure_high": 1.08,
    "vibration_high": 0.07,
}


def inspect_sensor_row(row: pd.Series) -> tuple[list[str], list[str]]:
    """Return human-readable reasons and actions for one sensor row."""
    reasons: list[str] = []
    suggestions: list[str] = []

    temp = float(row["temp"])
    pressure = float(row["pressure"])
    vibration = float(row["vibration"])

    if temp > THRESHOLDS["temp_high"] or temp < THRESHOLDS["temp_low"]:
        reasons.append(f"temperature out of range ({temp:.2f} °C)")
        suggestions.append("Inspect cooling/heating control and verify the temperature sensor.")
    if pressure > THRESHOLDS["pressure_high"] or pressure < THRESHOLDS["pressure_low"]:
        reasons.append(f"pressure out of range ({pressure:.3f})")
        suggestions.append("Check pressure regulation, valves, and pressure-sensor calibration.")
    if vibration > THRESHOLDS["vibration_high"]:
        reasons.append(f"vibration above limit ({vibration:.3f})")
        suggestions.append("Inspect rotating components, mounting, and vibration sensor health.")

    return reasons, list(dict.fromkeys(suggestions))


def build_alerts(df: pd.DataFrame, scores, labels) -> list[dict[str, Any]]:
    """Combine PyOD output with rule-based context into actionable alerts."""
    alerts: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        reasons, suggestions = inspect_sensor_row(row)
        rule_flag = bool(reasons)
        model_flag = int(labels[idx]) == 1
        if model_flag and not rule_flag:
            reasons.append("PyOD KNN detected an anomalous sensor pattern.")
            suggestions.append("Inspect the equipment and review recent sensor history.")

        if not (rule_flag or model_flag):
            continue

        alerts.append(
            {
                "timestamp": row["timestamp"],
                "temp": float(row["temp"]),
                "pressure": float(row["pressure"]),
                "vibration": float(row["vibration"]),
                "anomaly_score": float(scores[idx]),
                "label": "abnormal",
                "source": "+".join(
                    source
                    for source, enabled in (("fixed_rule", rule_flag), ("knn", model_flag))
                    if enabled
                ),
                "reasons": reasons,
                "suggestions": suggestions,
            }
        )
    return alerts

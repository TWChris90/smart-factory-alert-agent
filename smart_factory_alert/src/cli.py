"""Command-line entry point for the Smart Factory Alert Agent."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Union

import pandas as pd

if __package__:
    from .alert_agent import build_alerts
    from .generate_data import generate_sensor_data
    from .model_pipeline import (
        DEFAULT_MODEL_PATH,
        detect_sensor_data,
        detect_with_saved_model,
        ensure_default_model,
        load_saved_model,
    )
    from .preprocess import FEATURES, load_sensor_data
else:
    from alert_agent import build_alerts
    from generate_data import generate_sensor_data
    from model_pipeline import (
        DEFAULT_MODEL_PATH,
        detect_sensor_data,
        detect_with_saved_model,
        ensure_default_model,
        load_saved_model,
    )
    from preprocess import FEATURES, load_sensor_data

SOURCE_NAMES = {
    "fixed_rule": "fixed-threshold",
    "knn": "combination-pattern",
    "fixed_rule+knn": "fixed-threshold+combination-pattern",
}


def run_pipeline(
    csv_path: Union[str, Path],
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
) -> list[dict]:
    """Run inference with a saved model; detection data is never fitted."""
    model_path = Path(model_path)
    model_bundle = (
        ensure_default_model()
        if model_path == DEFAULT_MODEL_PATH
        else load_saved_model(model_path)
    )
    df, scores, labels = detect_with_saved_model(csv_path, model_bundle)
    return build_alerts(df, scores, labels)


def _load_model(model_path: Path):
    """Load the default or a user-specified saved model."""
    return (
        ensure_default_model()
        if model_path == DEFAULT_MODEL_PATH
        else load_saved_model(model_path)
    )


def _sensor_value(value: float, precision: int) -> str:
    """Format a sensor value while preserving visible missing values."""
    if math.isnan(float(value)):
        return "NaN"
    return f"{float(value):.{precision}f}"


def _display_reason(reason: str) -> str:
    if reason == "PyOD KNN detected an anomalous sensor pattern.":
        return "combined sensor pattern is uncommon versus normal training data"
    return reason


def _display_suggestion(suggestion: str) -> str:
    if suggestion == "Inspect the equipment and review recent sensor history.":
        return "Review the surrounding 5–10 minutes; inspect settings and sensors if it persists."
    return suggestion


def _alerts_frame(alerts: list[dict]) -> pd.DataFrame:
    """Flatten alert records for CSV output."""
    columns = [
        "timestamp", "temp", "pressure", "vibration", "score", "source",
        "reasons", "suggestions",
    ]
    records = [
        {
            "timestamp": alert["timestamp"],
            "temp": alert["temp"],
            "pressure": alert["pressure"],
            "vibration": alert["vibration"],
            "score": alert["anomaly_score"],
            "source": SOURCE_NAMES.get(alert["source"], alert["source"]),
            "reasons": " | ".join(_display_reason(item) for item in alert["reasons"]),
            "suggestions": " | ".join(
                _display_suggestion(item) for item in alert["suggestions"]
            ),
        }
        for alert in alerts
    ]
    return pd.DataFrame(records, columns=columns)


def _results_frame(
    raw_df: pd.DataFrame,
    scores,
    labels,
    alerts: list[dict],
) -> pd.DataFrame:
    """Build one output row for every processed sensor reading."""
    alert_lookup = {pd.Timestamp(alert["timestamp"]): alert for alert in alerts}
    records = []
    for index, row in raw_df.iterrows():
        timestamp = pd.Timestamp(row["timestamp"])
        alert = alert_lookup.get(timestamp)
        missing_features = [feature for feature in FEATURES if pd.isna(row[feature])]
        records.append(
            {
                "timestamp": timestamp,
                "temp": row["temp"],
                "pressure": row["pressure"],
                "vibration": row["vibration"],
                "input_label": row["label"],
                "score": float(scores[index]),
                "model_flag": int(labels[index]),
                "result": "alert" if alert else "normal",
                "source": (
                    SOURCE_NAMES.get(alert["source"], alert["source"])
                    if alert
                    else ""
                ),
                "missing_features": ",".join(missing_features),
            }
        )
    return pd.DataFrame(records)


def _print_preview(df: pd.DataFrame, preview_rows: int) -> None:
    """Print a compact raw-data preview for the terminal demo."""
    print("Data preview:")
    print("  timestamp            temp    pressure  vibration  label")
    for _, row in df.head(preview_rows).iterrows():
        print(
            f"  {pd.Timestamp(row['timestamp'])}  "
            f"{_sensor_value(row['temp'], 2):>6}  "
            f"{_sensor_value(row['pressure'], 3):>8}  "
            f"{_sensor_value(row['vibration'], 3):>9}  "
            f"{row['label']}"
        )


def _print_detection_rows(
    raw_df: pd.DataFrame,
    scores,
    alerts: list[dict],
    *,
    show_normal: bool,
) -> None:
    """Print normal and alert decisions in a screenshot-friendly format."""
    alert_lookup = {pd.Timestamp(alert["timestamp"]): alert for alert in alerts}
    for index, row in raw_df.iterrows():
        timestamp = pd.Timestamp(row["timestamp"])
        alert = alert_lookup.get(timestamp)
        if alert is None and not show_normal:
            continue

        missing_features = [feature for feature in FEATURES if pd.isna(row[feature])]
        status = "ALERT" if alert else "NORMAL"
        source = (
            f" source={SOURCE_NAMES.get(alert['source'], alert['source'])}"
            if alert
            else ""
        )
        print(f"[{timestamp}] {status} score={float(scores[index]):.4f}{source}")
        if missing_features:
            print(
                "  Data quality: missing "
                f"{', '.join(missing_features)}; filled with saved training median."
            )
        if alert:
            for reason in alert["reasons"]:
                print(f"  Reason: {_display_reason(reason)}")
            for suggestion in alert["suggestions"]:
                print(f"  Action: {_display_suggestion(suggestion)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart Factory Alert Agent")
    parser.add_argument("--input", type=Path, default=Path("data/sensor_data.csv"))
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--rows", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--limit", type=int, help="Process only the first N readings")
    parser.add_argument("--preview", type=int, default=3)
    parser.add_argument("--show-normal", action="store_true")
    parser.add_argument("--output", type=Path, help="Save alert rows as CSV")
    parser.add_argument("--results-output", type=Path, help="Save all processed rows as CSV")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.preview < 0:
        parser.error("--preview cannot be negative")

    if args.generate or not args.input.exists():
        args.input.parent.mkdir(parents=True, exist_ok=True)
        generate_sensor_data(args.rows, seed=args.seed).to_csv(args.input, index=False)
        print(f"Sensor data ready: {args.input}")

    loaded_df = load_sensor_data(args.input)
    missing_cells = int(loaded_df[FEATURES].isna().sum().sum())
    print(
        f"Loaded {len(loaded_df)} readings from {args.input} | "
        f"missing sensor cells: {missing_cells}"
    )
    _print_preview(loaded_df, args.preview)

    processed_df = loaded_df.head(args.limit).copy() if args.limit else loaded_df.copy()
    processed_df = processed_df.reset_index(drop=True)
    model_bundle = _load_model(args.model)
    transformed_df, scores, labels = detect_sensor_data(processed_df, model_bundle)
    alerts = build_alerts(transformed_df, scores, labels)
    threshold = float(model_bundle["score_threshold"])
    calibration_rows = int(model_bundle["calibration_rows"])
    calibration_percentile = float(model_bundle["calibration_percentile"])
    print(
        f"Saved model: {args.model} | combination score threshold: {threshold:.4f} "
        f"({calibration_percentile:g}th percentile of {calibration_rows} "
        "independent normal calibration scores; higher = more unusual)"
    )
    print(
        f"Processing {len(processed_df)} reading(s) | "
        f"show normal: {'yes' if args.show_normal else 'no'}"
    )
    _print_detection_rows(
        processed_df,
        scores,
        alerts,
        show_normal=args.show_normal,
    )

    missing_rows = int(processed_df[FEATURES].isna().any(axis=1).sum())
    print(
        f"Processed {len(processed_df)} readings | alerts: {len(alerts)} | "
        f"normal: {len(processed_df) - len(alerts)} | missing-data rows: {missing_rows}"
    )

    saved_messages = []
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _alerts_frame(alerts).to_csv(args.output, index=False)
        saved_messages.append(f"alerts: {args.output}")
    if args.results_output:
        args.results_output.parent.mkdir(parents=True, exist_ok=True)
        _results_frame(processed_df, scores, labels, alerts).to_csv(
            args.results_output,
            index=False,
        )
        saved_messages.append(f"all results: {args.results_output}")
    if saved_messages:
        print("Saved " + " | ".join(saved_messages))


if __name__ == "__main__":
    main()

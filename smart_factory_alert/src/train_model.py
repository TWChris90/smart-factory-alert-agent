"""Train and save the KNN pipeline from independent normal sensor data."""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
    from .generate_data import generate_normal_sensor_data
    from .model_pipeline import (
        DEFAULT_CALIBRATION_DATA_PATH,
        DEFAULT_CALIBRATION_PERCENTILE,
        DEFAULT_CALIBRATION_ROWS,
        DEFAULT_CALIBRATION_SEED,
        DEFAULT_CONTAMINATION,
        DEFAULT_MODEL_PATH,
        DEFAULT_TRAINING_DATA_PATH,
        DEFAULT_TRAINING_ROWS,
        DEFAULT_TRAINING_SEED,
        train_and_save_model,
    )
else:
    from generate_data import generate_normal_sensor_data
    from model_pipeline import (
        DEFAULT_CALIBRATION_DATA_PATH,
        DEFAULT_CALIBRATION_PERCENTILE,
        DEFAULT_CALIBRATION_ROWS,
        DEFAULT_CALIBRATION_SEED,
        DEFAULT_CONTAMINATION,
        DEFAULT_MODEL_PATH,
        DEFAULT_TRAINING_DATA_PATH,
        DEFAULT_TRAINING_ROWS,
        DEFAULT_TRAINING_SEED,
        train_and_save_model,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-rows", "--rows", dest="training_rows",
        type=int, default=DEFAULT_TRAINING_ROWS,
    )
    parser.add_argument("--calibration-rows", type=int, default=DEFAULT_CALIBRATION_ROWS)
    parser.add_argument(
        "--training-seed", "--seed", dest="training_seed",
        type=int, default=DEFAULT_TRAINING_SEED,
    )
    parser.add_argument("--calibration-seed", type=int, default=DEFAULT_CALIBRATION_SEED)
    parser.add_argument("--contamination", type=float, default=DEFAULT_CONTAMINATION)
    parser.add_argument(
        "--calibration-percentile",
        type=float,
        default=DEFAULT_CALIBRATION_PERCENTILE,
    )
    parser.add_argument("--training-data", type=Path, default=DEFAULT_TRAINING_DATA_PATH)
    parser.add_argument(
        "--calibration-data", type=Path, default=DEFAULT_CALIBRATION_DATA_PATH,
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    training_data = generate_normal_sensor_data(
        rows=args.training_rows,
        seed=args.training_seed,
    )
    calibration_data = generate_normal_sensor_data(
        rows=args.calibration_rows,
        seed=args.calibration_seed,
    )
    args.training_data.parent.mkdir(parents=True, exist_ok=True)
    args.calibration_data.parent.mkdir(parents=True, exist_ok=True)
    training_data.to_csv(args.training_data, index=False)
    calibration_data.to_csv(args.calibration_data, index=False)
    model_bundle = train_and_save_model(
        args.training_data,
        args.model,
        calibration_csv=args.calibration_data,
        contamination=args.contamination,
        calibration_percentile=args.calibration_percentile,
        training_seed=args.training_seed,
        calibration_seed=args.calibration_seed,
    )
    print(
        f"Normal training data: {args.training_data} "
        f"({args.training_rows} rows)"
    )
    print(
        f"Normal calibration data: {args.calibration_data} "
        f"({args.calibration_rows} rows)"
    )
    print(f"Saved model: {args.model}")
    print(
        f"Anomaly score threshold: {model_bundle['score_threshold']:.4f} "
        f"({args.calibration_percentile:g}th percentile of calibration scores)"
    )


if __name__ == "__main__":
    main()

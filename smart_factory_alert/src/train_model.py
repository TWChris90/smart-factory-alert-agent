"""Train and save the KNN pipeline from independent normal sensor data."""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
    from .generate_data import generate_normal_sensor_data
    from .model_pipeline import (
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
        DEFAULT_CONTAMINATION,
        DEFAULT_MODEL_PATH,
        DEFAULT_TRAINING_DATA_PATH,
        DEFAULT_TRAINING_ROWS,
        DEFAULT_TRAINING_SEED,
        train_and_save_model,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=DEFAULT_TRAINING_ROWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAINING_SEED)
    parser.add_argument("--contamination", type=float, default=DEFAULT_CONTAMINATION)
    parser.add_argument("--training-data", type=Path, default=DEFAULT_TRAINING_DATA_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    normal_data = generate_normal_sensor_data(rows=args.rows, seed=args.seed)
    args.training_data.parent.mkdir(parents=True, exist_ok=True)
    normal_data.to_csv(args.training_data, index=False)
    model_bundle = train_and_save_model(
        args.training_data,
        args.model,
        contamination=args.contamination,
        training_seed=args.seed,
    )
    print(f"Normal training data: {args.training_data} ({args.rows} rows)")
    print(f"Saved model: {args.model}")
    print(f"Anomaly score threshold: {model_bundle['score_threshold']:.4f}")


if __name__ == "__main__":
    main()

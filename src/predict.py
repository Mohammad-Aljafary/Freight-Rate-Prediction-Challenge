"""Retrain CatBoost on all labelled loads and create submission CSV files.

Run from the repository root after selecting the CatBoost settings in
``src/train.py``:

    uv run src/predict.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

# Support both `python -m src.predict` and `uv run src/predict.py`.
try:
    from src.features import (
        CATEGORICAL_COLUMNS,
        DATE_COLUMN,
        ID_COLUMN,
        TARGET,
        add_engineered_features,
        add_reduced_features,
        catboost_feature_frame,
        load_development_data,
        model_feature_columns,
    )
except ModuleNotFoundError:  # Direct execution: `uv run src/predict.py`
    from features import (
        CATEGORICAL_COLUMNS,
        DATE_COLUMN,
        ID_COLUMN,
        TARGET,
        add_engineered_features,
        add_reduced_features,
        catboost_feature_frame,
        load_development_data,
        model_feature_columns,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DECEMBER_COLUMNS = [
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date",
    "predicted_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create final CatBoost prediction files.")
    parser.add_argument(
        "--train-data",
        type=Path,
        default=PROJECT_ROOT / "data" / "train-test.csv",
    )
    parser.add_argument(
        "--validation-data",
        type=Path,
        default=PROJECT_ROOT / "data" / "validation.csv",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=PROJECT_ROOT / "data" / "validation-predictions-template.csv",
    )
    parser.add_argument(
        "--december-data",
        type=Path,
        default=PROJECT_ROOT / "data" / "december-chart-inputs.csv",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=PROJECT_ROOT / "validation_predictions.csv",
    )
    parser.add_argument(
        "--december-output",
        type=Path,
        default=PROJECT_ROOT / "december_predictions.csv",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "final_models",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=271,
        help="Best CatBoost iteration from September validation plus one.",
    )
    parser.add_argument("--loss", choices=("MAE", "RMSE"), default="MAE")
    return parser.parse_args()


def build_catboost_model(iterations: int, loss: str) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=iterations,
        depth=8,
        learning_rate=0.05,
        loss_function=loss,
        random_seed=42,
        verbose=100,
    )


def validate_validation_inputs(
    validation: pd.DataFrame, template: pd.DataFrame
) -> None:
    if ID_COLUMN not in validation.columns:
        raise ValueError(f"Validation data is missing {ID_COLUMN}")
    if list(template.columns) != [ID_COLUMN, "predicted_rate"]:
        raise ValueError("Template must contain exactly load_id,predicted_rate in that order")
    if validation[ID_COLUMN].duplicated().any() or template[ID_COLUMN].duplicated().any():
        raise ValueError("Validation data or template contains duplicate load_id values")
    if set(validation[ID_COLUMN]) != set(template[ID_COLUMN]):
        raise ValueError("Validation IDs do not exactly match template IDs")


def positive_rates(predictions: np.ndarray) -> np.ndarray:
    """Meet the scorer's strictly-positive-rate contract."""

    values = np.asarray(predictions, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("CatBoost produced non-finite predictions")
    return np.maximum(values, 0.01)


def main() -> None:
    args = parse_args()
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Full-schema model for the 12,000 validation loads.
    labelled_raw = load_development_data(args.train_data)
    labelled = add_engineered_features(labelled_raw)
    validation_raw = pd.read_csv(args.validation_data)
    validation = add_engineered_features(validation_raw)
    template = pd.read_csv(args.template)
    validate_validation_inputs(validation_raw, template)

    full_features = model_feature_columns(labelled)
    X_train = catboost_feature_frame(labelled, full_features)
    X_validation = catboost_feature_frame(validation, full_features)
    y_train = labelled[TARGET]

    full_model = build_catboost_model(args.iterations, args.loss)
    full_model.fit(
        X_train,
        y_train,
        cat_features=[column for column in CATEGORICAL_COLUMNS if column in full_features],
    )
    validation_predictions = positive_rates(full_model.predict(X_validation))

    predicted_by_id = pd.DataFrame(
        {ID_COLUMN: validation_raw[ID_COLUMN], "predicted_rate": validation_predictions}
    )
    submission = template[[ID_COLUMN]].merge(
        predicted_by_id,
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    if submission["predicted_rate"].isna().any() or len(submission) != len(template):
        raise ValueError("Could not produce a prediction for every template load_id")
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.predictions_output, index=False)
    full_model.save_model(args.artifacts_dir / "catboost_full_schema.cbm")
    print(f"Created {len(submission):,} validation predictions: {args.predictions_output}")

    # The December chart intentionally omits coordinates, market_index, and
    # quote_signal. Train a separate model only on fields available there.
    december = pd.read_csv(args.december_data)
    if list(december.columns) != DECEMBER_COLUMNS:
        raise ValueError(
            "December input must contain exactly: " + ",".join(DECEMBER_COLUMNS)
        )
    reduced_train = add_reduced_features(labelled_raw)
    reduced_december = add_reduced_features(december.drop(columns=["predicted_rate"]))
    reduced_features = model_feature_columns(reduced_train)
    X_reduced_train = catboost_feature_frame(reduced_train, reduced_features)
    X_reduced_december = catboost_feature_frame(reduced_december, reduced_features)

    reduced_model = build_catboost_model(args.iterations, args.loss)
    reduced_model.fit(
        X_reduced_train,
        reduced_train[TARGET],
        cat_features=[column for column in CATEGORICAL_COLUMNS if column in reduced_features],
    )
    december_output = december.copy()
    december_output["predicted_rate"] = positive_rates(
        reduced_model.predict(X_reduced_december)
    )
    args.december_output.parent.mkdir(parents=True, exist_ok=True)
    december_output.to_csv(args.december_output, index=False)
    reduced_model.save_model(args.artifacts_dir / "catboost_reduced_schema.cbm")
    print(f"Created {len(december_output):,} December predictions: {args.december_output}")


if __name__ == "__main__":
    main()

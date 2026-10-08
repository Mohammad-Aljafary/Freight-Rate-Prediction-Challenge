"""Shared loading, feature engineering, splitting, and preprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET = "posted_rate"
ID_COLUMN = "load_id"
DATE_COLUMN = "date"
CATEGORICAL_COLUMNS = ("pickup", "delivery", "equipment")
REQUIRED_COLUMNS = {
    ID_COLUMN,
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "equipment",
    "weight",
    DATE_COLUMN,
    "market_index",
    "quote_signal",
    TARGET,
}


@dataclass(frozen=True)
class DataSplits:
    """Chronological development splits used throughout the project."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class ExtraTreesMatrices:
    """One-hot feature matrices and the fitted, train-only preprocessor."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    preprocessor: ColumnTransformer


def load_development_data(path: str | Path) -> pd.DataFrame:
    """Load the labelled development data and validate its schema."""

    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Development data is missing required columns: {sorted(missing)}")

    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], errors="raise")
    if frame[ID_COLUMN].duplicated().any():
        raise ValueError("Development data contains duplicate load_id values")
    if frame[TARGET].isna().any() or (frame[TARGET] <= 0).any():
        raise ValueError("posted_rate must be present and positive in development data")
    return frame


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the notebook's calendar, load, and geographic features.

    This function never fits an imputer or encoder. Those operations are fitted
    only on the training split in :func:`make_extra_trees_matrices`.
    """

    result = add_calendar_features(frame)

    # Load-quality features.
    result["weight_missing"] = result["weight"].isna().astype("int8")
    result["market_index_missing"] = result["market_index"].isna().astype("int8")
    result["negative_weight"] = (result["weight"] < 0).fillna(False).astype("int8")
    result["weight_clean"] = result["weight"].abs()
    nonzero_distance = result["distance"].replace(0, np.nan)
    result["weight_per_mile"] = result["weight_clean"] / nonzero_distance

    # Direct geographical relationship features.
    result["lat_difference"] = result["delivery_lat"] - result["pickup_lat"]
    result["lon_difference"] = result["delivery_lon"] - result["pickup_lon"]
    result["abs_lat_difference"] = result["lat_difference"].abs()
    result["abs_lon_difference"] = result["lon_difference"].abs()

    # Great-circle distance and direction give the model a geographic fallback
    # for cities and lanes absent from historical training data.
    pickup_lat = np.radians(result["pickup_lat"])
    pickup_lon = np.radians(result["pickup_lon"])
    delivery_lat = np.radians(result["delivery_lat"])
    delivery_lon = np.radians(result["delivery_lon"])
    delta_lat = delivery_lat - pickup_lat
    delta_lon = delivery_lon - pickup_lon
    haversine_a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(pickup_lat) * np.cos(delivery_lat) * np.sin(delta_lon / 2) ** 2
    )
    result["haversine_miles"] = 3958.7613 * 2 * np.arcsin(np.sqrt(haversine_a))
    result["distance_to_haversine_ratio"] = result["distance"] / result[
        "haversine_miles"
    ].replace(0, np.nan)
    bearing = np.arctan2(
        np.sin(delta_lon) * np.cos(delivery_lat),
        np.cos(pickup_lat) * np.sin(delivery_lat)
        - np.sin(pickup_lat) * np.cos(delivery_lat) * np.cos(delta_lon),
    )
    result["bearing_sin"] = np.sin(bearing)
    result["bearing_cos"] = np.cos(bearing)
    return result


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add date features that can be built for both full and December data."""

    result = frame.copy()
    result[DATE_COLUMN] = pd.to_datetime(result[DATE_COLUMN], errors="raise")

    # Calendar features from the notebook.
    result["year"] = result[DATE_COLUMN].dt.year
    result["month"] = result[DATE_COLUMN].dt.month
    result["day"] = result[DATE_COLUMN].dt.day
    result["day_of_week"] = result[DATE_COLUMN].dt.dayofweek
    result["day_of_year"] = result[DATE_COLUMN].dt.dayofyear
    result["quarter"] = result[DATE_COLUMN].dt.quarter
    result["is_weekend"] = result["day_of_week"].isin([5, 6]).astype("int8")
    result["month_sin"] = np.sin(2 * np.pi * result["month"] / 12)
    result["month_cos"] = np.cos(2 * np.pi * result["month"] / 12)
    result["dow_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 7)
    result["dow_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 7)
    result["doy_sin"] = np.sin(2 * np.pi * result["day_of_year"] / 365.25)
    result["doy_cos"] = np.cos(2 * np.pi * result["day_of_year"] / 365.25)

    return result


def add_reduced_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build only features available in the reduced December input schema."""

    required = {"pickup", "delivery", "distance", "equipment", "weight", DATE_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Reduced input is missing required columns: {sorted(missing)}")

    result = add_calendar_features(frame)
    result["weight_missing"] = result["weight"].isna().astype("int8")
    result["negative_weight"] = (result["weight"] < 0).fillna(False).astype("int8")
    result["weight_clean"] = result["weight"].abs()
    nonzero_distance = result["distance"].replace(0, np.nan)
    result["weight_per_mile"] = result["weight_clean"] / nonzero_distance
    return result


def chronological_split(frame: pd.DataFrame) -> DataSplits:
    """Return Jan-Aug train, Sep validation, and Oct test splits for 2025."""

    train = frame[frame[DATE_COLUMN] < "2025-09-01"].copy()
    validation = frame[
        (frame[DATE_COLUMN] >= "2025-09-01")
        & (frame[DATE_COLUMN] < "2025-10-01")
    ].copy()
    test = frame[
        (frame[DATE_COLUMN] >= "2025-10-01")
        & (frame[DATE_COLUMN] < "2025-11-01")
    ].copy()
    if not all((len(train), len(validation), len(test))):
        raise ValueError("One or more chronological splits are empty")
    return DataSplits(train=train, validation=validation, test=test)


def model_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return model features while excluding identifiers, dates, and target."""

    excluded = {TARGET, ID_COLUMN, DATE_COLUMN, "year"}
    return [column for column in frame.columns if column not in excluded]


def catboost_matrices(splits: DataSplits) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Build raw-category matrices for CatBoost without one-hot encoding."""

    columns = model_feature_columns(splits.train)

    return (
        catboost_feature_frame(splits.train, columns),
        splits.train[TARGET].copy(),
        catboost_feature_frame(splits.validation, columns),
        splits.validation[TARGET].copy(),
        catboost_feature_frame(splits.test, columns),
        splits.test[TARGET].copy(),
    )


def catboost_feature_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Select CatBoost features and guarantee categorical values are strings."""

    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction data is missing model columns: {sorted(missing)}")
    matrix = frame[columns].copy()
    for column in CATEGORICAL_COLUMNS:
        if column in matrix.columns:
            matrix[column] = matrix[column].fillna("Unknown").astype(str)
    return matrix


def make_extra_trees_matrices(splits: DataSplits) -> ExtraTreesMatrices:
    """Fit numeric imputation and one-hot encoding on train rows only."""

    columns = model_feature_columns(splits.train)
    categorical = [column for column in CATEGORICAL_COLUMNS if column in columns]
    numeric = [column for column in columns if column not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )

    train_array = preprocessor.fit_transform(splits.train[columns])
    validation_array = preprocessor.transform(splits.validation[columns])
    test_array = preprocessor.transform(splits.test[columns])
    feature_names = preprocessor.get_feature_names_out().tolist()

    def as_frame(values: np.ndarray, index: pd.Index) -> pd.DataFrame:
        return pd.DataFrame(values, columns=feature_names, index=index)

    return ExtraTreesMatrices(
        train=as_frame(train_array, splits.train.index),
        validation=as_frame(validation_array, splits.validation.index),
        test=as_frame(test_array, splits.test.index),
        preprocessor=preprocessor,
    )

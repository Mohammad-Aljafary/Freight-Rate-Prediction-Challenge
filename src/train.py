"""Train and compare the ExtraTrees and CatBoost freight-rate pipelines.

Run from the repository root:
    python -m src.train
    python -m src.train --evaluate-test
    python -m src.train --tail-weight 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel

# Support both `python -m src.train` and `uv run src/train.py`.
try:
    from src.evaluate import metrics, save_diagnostics, tail_metrics
    from src.features import (
        CATEGORICAL_COLUMNS,
        TARGET,
        add_engineered_features,
        catboost_matrices,
        chronological_split,
        load_development_data,
        make_extra_trees_matrices,
    )
except ModuleNotFoundError:  # Direct execution: `uv run src/train.py`
    from evaluate import metrics, save_diagnostics, tail_metrics
    from features import (
        CATEGORICAL_COLUMNS,
        TARGET,
        add_engineered_features,
        catboost_matrices,
        chronological_split,
        load_development_data,
        make_extra_trees_matrices,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train freight-rate model candidates.")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "train-test.csv",
        help="Labelled development CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts",
        help="Directory for models, diagnostics, and metrics.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the selected candidates on October after validation comparison.",
    )
    parser.add_argument(
        "--catboost-loss",
        choices=("MAE", "RMSE"),
        default="MAE",
        help="CatBoost objective. MAE favours typical loads; RMSE emphasises large errors.",
    )
    parser.add_argument(
        "--tail-weight",
        type=float,
        default=1.0,
        help="Weight applied to the highest 1%% of training rates; use 1 for no reweighting.",
    )
    parser.add_argument("--catboost-iterations", type=int, default=1500)
    parser.add_argument(
        "--selector-estimators",
        type=int,
        default=300,
        help="Number of trees used by the ExtraTrees feature selector.",
    )
    parser.add_argument(
        "--extra-trees-estimators",
        type=int,
        default=500,
        help="Number of trees in the final ExtraTrees regressor.",
    )
    return parser.parse_args()


def print_report(name: str, values: dict[str, float | int]) -> None:
    formatted = ", ".join(
        f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
        for key, value in values.items()
    )
    print(f"{name}: {formatted}")


def main() -> None:
    args = parse_args()
    if args.tail_weight < 1:
        raise ValueError("--tail-weight must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = args.output_dir / "reports"
    models_dir = args.output_dir / "models"
    reports_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    data = add_engineered_features(load_development_data(args.data))
    splits = chronological_split(data)
    print(
        f"Rows — train: {len(splits.train):,}; validation: {len(splits.validation):,}; "
        f"test: {len(splits.test):,}"
    )

    # Pipeline A: train-only median imputation, one-hot encoding, feature
    # selection, then ExtraTrees regression.
    extra = make_extra_trees_matrices(splits)
    y_train = splits.train[TARGET]
    y_validation = splits.validation[TARGET]
    selector_estimator = ExtraTreesRegressor(
        n_estimators=args.selector_estimators,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    )
    selector = SelectFromModel(selector_estimator, threshold="median")
    selector.fit(extra.train, y_train)
    selected_features = extra.train.columns[selector.get_support()].tolist()
    if not selected_features:
        raise RuntimeError("Feature selector did not retain any features")
    print(f"ExtraTrees selected {len(selected_features)} of {extra.train.shape[1]} features")

    extra_model = ExtraTreesRegressor(
        n_estimators=args.extra_trees_estimators,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    )
    extra_model.fit(extra.train[selected_features], y_train)
    extra_valid_predictions = extra_model.predict(extra.validation[selected_features])
    extra_report = metrics(y_validation, extra_valid_predictions)
    print_report("ExtraTrees validation", extra_report)
    save_diagnostics(
        splits.validation,
        y_validation,
        extra_valid_predictions,
        reports_dir,
        "extra_trees_validation",
    )
    joblib.dump(
        {
            "preprocessor": extra.preprocessor,
            "selector": selector,
            "selected_features": selected_features,
            "model": extra_model,
        },
        models_dir / "extra_trees.joblib",
    )

    # Pipeline B: raw city/equipment categories plus CatBoost's native handling.
    (
        cat_train,
        cat_y_train,
        cat_validation,
        cat_y_validation,
        cat_test,
        cat_y_test,
    ) = catboost_matrices(splits)
    threshold = float(cat_y_train.quantile(0.99))
    sample_weight = None
    if args.tail_weight > 1:
        sample_weight = np.where(cat_y_train >= threshold, args.tail_weight, 1.0)
        print(
            f"Applying a {args.tail_weight:g}× weight to rates at or above ${threshold:,.2f}"
        )

    cat_model = CatBoostRegressor(
        iterations=args.catboost_iterations,
        depth=8,
        learning_rate=0.05,
        loss_function=args.catboost_loss,
        random_seed=42,
        verbose=100,
    )
    cat_model.fit(
        cat_train,
        cat_y_train,
        cat_features=list(CATEGORICAL_COLUMNS),
        sample_weight=sample_weight,
        eval_set=(cat_validation, cat_y_validation),
        early_stopping_rounds=100,
    )
    cat_valid_predictions = cat_model.predict(cat_validation)
    cat_report = metrics(cat_y_validation, cat_valid_predictions)
    cat_report.update(tail_metrics(cat_y_validation, cat_valid_predictions, threshold))
    cat_report["best_iteration"] = int(cat_model.get_best_iteration())
    print_report("CatBoost validation", cat_report)
    save_diagnostics(
        splits.validation,
        cat_y_validation,
        cat_valid_predictions,
        reports_dir,
        "catboost_validation",
    )
    cat_model.save_model(models_dir / "catboost.cbm")

    all_reports: dict[str, dict[str, float | int]] = {
        "extra_trees_validation": extra_report,
        "catboost_validation": cat_report,
    }

    # Test results are intentionally opt-in. Do not use them to tune settings.
    if args.evaluate_test:
        extra_test_predictions = extra_model.predict(extra.test[selected_features])
        extra_test_report = metrics(splits.test[TARGET], extra_test_predictions)
        print_report("ExtraTrees test", extra_test_report)
        save_diagnostics(
            splits.test,
            splits.test[TARGET],
            extra_test_predictions,
            reports_dir,
            "extra_trees_test",
        )

        cat_test_predictions = cat_model.predict(cat_test)
        cat_test_report = metrics(cat_y_test, cat_test_predictions)
        cat_test_report.update(tail_metrics(cat_y_test, cat_test_predictions, threshold))
        print_report("CatBoost test", cat_test_report)
        save_diagnostics(
            splits.test,
            cat_y_test,
            cat_test_predictions,
            reports_dir,
            "catboost_test",
        )
        all_reports["extra_trees_test"] = extra_test_report
        all_reports["catboost_test"] = cat_test_report

    (reports_dir / "metrics.json").write_text(json.dumps(all_reports, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Metrics and diagnostic outputs shared by both model pipelines."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    """Compute regression metrics after removing index-alignment ambiguity."""

    y_true = actual.to_numpy(dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def tail_metrics(
    actual: pd.Series, predicted: np.ndarray, threshold: float
) -> dict[str, float | int]:
    """Report error for the highest-rate tail without changing global metrics."""

    y_true = actual.to_numpy(dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    mask = y_true >= threshold
    result: dict[str, float | int] = {
        "tail_threshold": float(threshold),
        "tail_rows": int(mask.sum()),
    }
    if mask.any():
        result["tail_mae"] = float(mean_absolute_error(y_true[mask], y_pred[mask]))
        result["normal_mae"] = float(mean_absolute_error(y_true[~mask], y_pred[~mask]))
    return result


def prediction_frame(actual: pd.Series, predicted: np.ndarray) -> pd.DataFrame:
    """Return aligned actual/predicted/residual columns for inspection."""

    result = pd.DataFrame(
        {
            "actual": actual.to_numpy(dtype=float),
            "predicted": np.asarray(predicted, dtype=float),
        },
        index=actual.index,
    )
    result["residual"] = result["actual"] - result["predicted"]
    result["absolute_error"] = result["residual"].abs()
    return result


def save_diagnostics(
    frame: pd.DataFrame,
    actual: pd.Series,
    predicted: np.ndarray,
    output_dir: str | Path,
    prefix: str,
) -> None:
    """Save prediction/error tables and the notebook's diagnostic plots."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions = prediction_frame(actual, predicted)
    predictions.to_csv(output / f"{prefix}_predictions.csv", index_label="row_index")
    predictions.sort_values("absolute_error", ascending=False).head(50).to_csv(
        output / f"{prefix}_worst_errors.csv", index_label="row_index"
    )

    plotted = frame.loc[actual.index].copy()
    plotted["actual"] = predictions["actual"]
    plotted["predicted"] = predictions["predicted"]
    plotted["residual"] = predictions["residual"]

    max_rate = max(float(plotted["actual"].max()), float(plotted["predicted"].max()))
    figure, axis = plt.subplots(figsize=(8, 7))
    axis.scatter(plotted["actual"], plotted["predicted"], alpha=0.35, s=16)
    axis.plot([0, max_rate], [0, max_rate], "r--", label="Perfect prediction")
    axis.set(xlabel="Actual posted rate ($)", ylabel="Predicted rate ($)", title=f"{prefix}: actual vs predicted")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / f"{prefix}_actual_vs_predicted.png", dpi=180)
    plt.close(figure)

    limit = float(np.quantile(np.abs(plotted["residual"]), 0.95))
    limit = max(limit, 1.0)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    for location, longitude, latitude in (
        ("pickup", "pickup_lon", "pickup_lat"),
        ("delivery", "delivery_lon", "delivery_lat"),
    ):
        figure, axis = plt.subplots(figsize=(10, 7))
        scatter = axis.scatter(
            plotted[longitude],
            plotted[latitude],
            c=plotted["residual"],
            cmap="coolwarm",
            norm=norm,
            alpha=0.6,
            s=20,
        )
        figure.colorbar(scatter, ax=axis, label="Residual = actual − predicted ($)")
        axis.set(
            xlabel=f"{location.title()} longitude",
            ylabel=f"{location.title()} latitude",
            title=f"{prefix}: {location} residuals",
        )
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output / f"{prefix}_{location}_residuals.png", dpi=180)
        plt.close(figure)

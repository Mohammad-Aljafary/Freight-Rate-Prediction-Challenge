# Freight Rate Prediction Challenge

This project predicts a load's `posted_rate` from lane, equipment, distance,
weight, market, quote, date, and geographic inputs.

## Setup

The project requires Python 3.14 and [uv](https://docs.astral.sh/uv/). From the
repository root, install the project dependencies:

```bash
uv sync
```

A pip-based setup is also supported:

```bash
python -m pip install -r requirements.txt
```

## Data files

All source data is in `data/`:

| File | Purpose |
| --- | --- |
| `train-test.csv` | 48,000 labelled development loads with `posted_rate`. |
| `validation.csv` | 12,000 final loads that require predictions. |
| `validation-predictions-template.csv` | Required `load_id` order for the final submission. |
| `december-chart-inputs.csv` | 31 fixed December chart loads. |

## Train and compare models

The training script uses chronological splits: January–August for training,
September for model selection, and October as the final holdout test.

It compares both implemented approaches:

- **ExtraTrees:** train-only median imputation, one-hot encoding, and
  ExtraTrees-based feature selection.
- **CatBoost:** native categorical handling for pickup, delivery, and
  equipment, plus engineered calendar, load, and geographic features.

Run validation comparison:

```bash
uv run src/train.py
```

After selecting a configuration using September results, run the October test
once:

```bash
uv run src/train.py --evaluate-test
```

To test an RMSE-focused CatBoost that weights the highest 1% of training rates:

```bash
uv run src/train.py --catboost-loss RMSE --tail-weight 3
```

Metrics, models, prediction tables, worst-error rows, and diagnostic charts are
written to `artifacts/`.

## Create submission predictions

The final prediction script retrains the selected CatBoost configuration on all
labelled January–October loads. It then:

1. Predicts every row in `data/validation.csv`.
2. Places those values in the `load_id` order of
   `data/validation-predictions-template.csv`.
3. Trains a separate reduced-feature CatBoost model for December, because the
   December input lacks coordinates, `market_index`, and `quote_signal`.

Run:

```bash
uv run src/predict.py
```

This creates:

```text
validation_predictions.csv
december_predictions.csv
```

`validation_predictions.csv` contains exactly the required columns:

```csv
load_id,predicted_rate
TE-000001,846.18
```

## Validate the deliverables

Run the supplied scorer before submission:

```bash
uv run score.py \
  --predictions validation_predictions.csv \
  --december-predictions december_predictions.csv
```

The scorer verifies IDs, row counts, schemas, dates, and positive numeric
predictions. It writes the required chart to:

```text
scorer_results/candidate_december.png
```

## Submission checklist

- `validation_predictions.csv`
- `december_predictions.csv`
- `scorer_results/candidate_december.png`
- Source code, dependencies, and this README
- PDF or DOCX report covering data quality, chronological validation, model
  selection, and final chart
- 2–3 minute Loom walkthrough

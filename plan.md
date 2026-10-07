# Freight Rate Prediction Challenge — Delivery Plan

## Goal and required deliverables

Build a reproducible regression pipeline that learns `posted_rate` from
`train-test.csv`, generates a prediction for every `load_id` in
`validation.csv`, and produces the two files required by the supplied scorer:

- `validation_predictions.csv` with exactly the ordered columns
  `load_id,predicted_rate`, exactly 12,000 unique IDs `TE-000001` through
  `TE-012000`, and strictly positive finite rates.
- The original `december-chart-inputs.csv` with its seven columns unchanged
  and in order, all 31 `predicted_rate` values filled with strictly positive,
  finite rates.
- Source code, pinned dependencies, run instructions, a short report (PDF or
  DOCX), the scorer-generated December chart, and a 2–3 minute walkthrough.

The README describes a `data/` directory, but the supplied files currently
live at the repository root and use hyphenated names. The implementation should
centralize file paths in configuration, or move/copy files only if the README
and code are updated together. Do not silently depend on paths that are not in
the repository.

## Estimated effort

The full-scope version takes 10–14 focused hours, but this challenge can be
completed in **8 hours** by using the timeboxed delivery path below. It assumes
a working Python environment and limits tuning to evidence from chronological
backtests.

| Work item | Estimate |
| --- | ---: |
| Project setup, path correction, and dependency lockfile | 0.5–1 hour |
| Data audit, quality checks, and exploratory findings | 1.5–2.5 hours |
| Shared feature pipeline and chronological split/backtests | 2–3 hours |
| Baselines, model experiments, error analysis, and selection | 3–4 hours |
| Full retraining, output-contract checks, and scorer run | 0.5–1 hour |
| README, report/chart, and 2–3 minute Loom | 1.5–2.5 hours |

### Eight-hour delivery path

| Timebox | Outcome | Scope boundary |
| --- | --- | --- |
| 0:00–0:30 | Project boots and inputs are verified | Create the layout, install dependencies, run schema/null/ID checks, and correct documented paths. Do not start modeling before this passes. |
| 0:30–1:30 | EDA findings and a fixed split | Produce a compact profiling table, inspect target/rate-per-mile distributions and drift, then lock the October chronological holdout. Skip elaborate visual exploration. |
| 1:30–3:00 | One reusable feature pipeline | Implement calendar, geography, load, missingness, and categorical lane features. Ensure it accepts both validation and reduced December inputs. |
| 3:00–4:15 | Benchmarks and primary model | Fit the rate-per-mile baseline and one CatBoost regression model. Evaluate both on October and save metrics/errors by equipment and distance band. |
| 4:15–5:15 | Targeted improvement only | Try at most two changes (for example, log-target versus raw target and one parameter/feature variant). Keep a change only if it improves the held-out metric. |
| 5:15–6:00 | Refit and predict | Fit the selected validation model on all labeled rows; fit the reduced-schema December model; write both prediction outputs. |
| 6:00–6:30 | Release checks | Run local output assertions and `score.py`; resolve every contract error and retain `candidate_december.png`. |
| 6:30–7:30 | Submission materials | Update README and write the report using the saved metrics, EDA table, split rationale, model choice, and chart. |
| 7:30–8:00 | Final review and Loom | Re-run commands from a clean state where practical, check tracked deliverables, and record the 2–3 minute walkthrough. |

**Scope controls for the deadline:** use a single strong categorical tree model
(CatBoost) rather than a large model search; restrict comparison to the two
baselines plus no more than three CatBoost configurations; no external data;
and no target encoding unless the initial model is clearly inadequate. If time
runs short, submit the best scorer-valid CatBoost run after the 6:30 release
check, then spend remaining time documenting it rather than pursuing untested
improvements.

## What the supplied data implies

| Dataset | Size | Date range | Notes |
| --- | ---: | --- | --- |
| `train-test.csv` | 48,000 × 14 | 2025-01-01 to 2025-10-31 | Target is `posted_rate`; 300 missing `weight` and 374 missing `market_index` values. |
| `validation.csv` | 12,000 × 13 | 2025-11-01 to 2025-12-31 | Forecast period; 165 missing `weight`, 249 missing `market_index`, and eight cities absent from training. |
| `december-chart-inputs.csv` | 31 × 7 | 2025-12-01 to 2025-12-31 | One Lexington → Fort Wayne Dry Van lane; lacks latitude, longitude, market index, and quote signal. |

This is a forward-looking forecasting problem, not an IID random-split problem.
The validation file includes 736 pickup/delivery lanes and 1,409
pickup/delivery/equipment combinations not seen in development data. A model
must therefore combine route-specific learning with features that generalize to
new cities and lanes.

## Implementation sequence

### 1. Establish a reproducible project layout

Create the following, keeping raw CSVs immutable:

```text
src/
  features.py       # shared feature construction for all input schemas
  train.py          # fitting, validation, artifact creation
  predict.py        # validation and December prediction generation
  evaluate.py       # metrics, diagnostics, charts/tables for the report
config.py or config.yaml
requirements.txt
README.md
report/             # final report source and candidate_december.png
```

- Pin the Python version and package versions. The supplied scorer requires
  `pandas`, `numpy`, and `matplotlib`; include them alongside the modeling
  dependencies in `requirements.txt`.
- Set one global random seed and log the input file hashes, feature list,
  split dates, model parameters, and validation metrics.
- Add a single command (for example, `python -m src.train` followed by
  `python -m src.predict`) that can reproduce every generated deliverable from
  a clean checkout.

### 2. Audit and clean without leakage

In a notebook or an auditable script, profile the training data and record:

- row counts, unique IDs, duplicate records, nulls, invalid/nonpositive values,
  outliers, categorical cardinality, and target/rate-per-mile distributions;
- changes over date, equipment type, distance band, origin, destination, and
  major lanes;
- train-versus-validation drift, especially the eight new cities and unseen
  lane combinations; and
- whether coordinates are internally consistent for each city and whether
  supplied `distance` disagrees materially with geodesic distance.

Keep `load_id` out of model features. Parse `date` as a real date. Impute
numeric missing values using statistics fitted only on each training fold, and
give the model missing-value indicator columns. For category values not seen
during fitting, use an explicit unknown category or a model that handles them
natively. Never compute target aggregates using validation or future labels.

### 3. Create features that work for known and new lanes

Use one shared, schema-tolerant feature builder. It must accept both the full
load schema and the reduced December-chart schema, filling unavailable inputs
with `NaN` plus availability indicators.

Candidate feature groups:

- **Calendar:** month, day of week, day of month, week of year, and a monotonic
  time index. Treat holiday flags as optional and document their source if used.
- **Geography:** numeric pickup/delivery coordinates, latitude/longitude
  deltas, Haversine distance, supplied-distance-to-geodesic ratio, direction
  (sin/cos bearing), and distance bands. These are essential for unseen cities.
- **Load economics:** `distance`, `weight`, weight-per-mile, equipment, and
  interactions such as equipment × distance band and equipment × weight.
- **Market/quote signals:** `market_index`, `quote_signal`, their interaction,
  and missing indicators. Include only if they are available at prediction time
  for the target dataset.
- **Route identity:** origin, destination, directional lane, and
  origin/destination × equipment. Prefer CatBoost's categorical handling or
  fold-safe count/frequency encodings. If target encoding is tried, calculate
  it out-of-fold and use a smoothed global fallback.

For the December chart, explicitly define a fallback because its schema lacks
coordinates, `market_index`, and `quote_signal`. The safest version is a
separate, documented reduced-schema model trained on features that exist in
that file, rather than fabricating unavailable market signals. Since its lane
is present in training, lane/equipment and date features can still be used.

### 4. Validate as a future forecast

Use a final chronological holdout, e.g. train through 2025-09-30 and validate
on 2025-10-01 through 2025-10-31. This mirrors the Nov–Dec evaluation better
than a random split. In addition, use expanding-window backtests (for example,
July, August, September, and October validation months) to assess stability.

Report MAE and RMSE (and R² as a secondary descriptive metric) for every fold,
the weighted mean across folds, and the final October holdout. Break errors down
by:

- equipment type and distance bands;
- known vs unseen-like routes (simulate this by withholding selected cities or
  lanes from historical training); and
- missing versus complete market/weight fields.

Build baselines first: a global median, an equipment-specific rate-per-mile
baseline, and a regularized linear model on engineered numeric features. Compare
them with a tree-gradient model. CatBoost is a strong first choice because it
handles categorical fields, nonlinearities, missing numeric values, and unseen
categories; LightGBM/XGBoost plus explicit encodings is a valid alternative.

Choose the simplest model whose rolling validation is reliably best. Consider a
small blend only if it improves every or nearly every forecast fold. Inspect
residuals and clip only at a justified lower floor (for example, zero); avoid
arbitrary upper clipping unless validation proves it reduces error.

### 5. Fit, predict, and validate file contracts

After selecting hyperparameters:

1. Refit the chosen full-schema validation model on all 48,000 labeled rows.
2. Transform `validation.csv` using the saved feature builder and write
   predictions into the template by `load_id`—not merely by positional
   assumption.
3. Refit/use the reduced-schema December model and fill all 31 chart rows.
4. Assert before writing that rates are finite, numeric, and strictly positive.
   For the validation file, assert exactly two columns in this order
   (`load_id,predicted_rate`), 12,000 rows, unique IDs, and equality to the
   complete expected set `TE-000001`…`TE-012000`. For the December file,
   preserve exactly the seven original columns in their original order and
   assert its 31 unique daily dates, 2025-12-01 through 2025-12-31, and its
   fixed Lexington → Fort Wayne / 360 miles / Dry Van / 32,000 lb inputs.
5. Install the scorer dependencies and run the exact command in the README:

   ```bash
   python -m pip install -r requirements.txt
   python score.py --predictions validation_predictions.csv \
     --december-predictions december-chart-inputs.csv
   ```

6. Treat a clean scorer run and the generated
   `scorer_results/candidate_december.png` as release gates. Copy the chart to
   the report directory and keep the scorer output unedited.

### 6. Package the submission

Update the README with prerequisites, actual file locations, exact commands,
expected outputs, and a short model/design summary. The report should state the
chronological split and backtest results, key EDA findings, data-quality
handling, model comparison and final rationale, feature availability/fallback
for December, and include `candidate_december.png`. The Loom should follow the
same narrative: findings → data quality → model choice → validation → code and
outputs.

## Definition of done

- A clean clone can install dependencies and regenerate both prediction files
  without manual edits.
- The chosen approach is selected using only historical, chronological
  validation—not the hidden validation labels.
- `validation_predictions.csv` has only `load_id,predicted_rate`, contains
  every expected `TE-` ID exactly once, and has 12,000 strictly positive,
  finite predictions.
- The December file retains its exact seven-column contract, has 31 strictly
  positive finite predictions for the required dates and fixed inputs, and
  passes the scorer.
- The repository contains clear run instructions, model artifacts/code, the
  score-generated chart, and the requested report/recording link.

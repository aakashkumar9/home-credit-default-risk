# Home Credit Default Risk

A default-risk pipeline built on the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
dataset: the applicant-level table alone is ~307k rows, but the real size is
in the six related history tables it links to (credit bureau records,
previous applications, POS/cash balances, credit card balances, instalment
payments) - tens of millions of rows combined, all one-to-many against a
single applicant.

The pipeline is the point of this project, more than the model on top of
it. Getting from "seven relational tables with different grains" to "one
row per applicant, ready to fit a classifier" is a real data-engineering
problem - schema design, join strategy, and one-to-many aggregation - and
it's built as a tested, documented dbt project rather than a notebook full
of `groupby().agg()` calls.

## Expected raw data

This repo does not download or vendor the dataset. Get it from
[Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data)
and place these files in `data/raw/` (filenames must match exactly - the
loader reads them by name):

```
data/raw/application_train.csv
data/raw/application_test.csv
data/raw/bureau.csv
data/raw/bureau_balance.csv
data/raw/previous_application.csv
data/raw/POS_CASH_balance.csv
data/raw/credit_card_balance.csv
data/raw/installments_payments.csv
```

(`HomeCredit_columns_description.csv` and `sample_submission.csv`, also in
the Kaggle download, aren't used by anything here.)

## Architecture

```
data/raw/*.csv (Kaggle)
        |
        v  scripts/load_raw_data.py
   raw.*                    <- untouched 1:1 copy of each CSV
        |
        v  dbt (warehouse/models/staging)
   staging.stg_*            <- renamed/typed, one model per source table,
        |                      light cleaning (365243 DAYS_* sentinel -> null,
        |                      sign-flipped day counts, safe-divide ratios)
        v  dbt (warehouse/models/intermediate)
   intermediate.int_*_agg   <- one-to-many history collapsed to sk_id_curr
        |                      (or sk_id_bureau, one level down) grain
        v  dbt (warehouse/models/marts)
   marts.mart_applicant_features   <- one row per sk_id_curr: application
        |                             + every aggregated history slice
        v
   modeling/                <- train -> calibrate -> explain -> evaluate -> predict
```

**Staging** (`warehouse/models/staging/`): one view per source table,
renamed to consistent snake_case, with the dataset's known data-quality
quirks fixed at the source: `DAYS_EMPLOYED == 365243` is a documented "not
applicable" sentinel (flagged as `is_pensioner_anomaly` and nulled out, same
treatment applied to the equivalent sentinel in `previous_application`'s
`DAYS_*` columns), and `DAYS_*` columns are sign-flipped into positive
"years/days ago" so they read naturally. The ~45 `*_AVG`/`_MODE`/`_MEDI`
apartment/building columns on `application` are collapsed into a single
`housing_quality_score` and `housing_info_missing_rate` - they're mutually
correlated measurements of the same underlying "how well is this building
documented" signal, and missingness itself is a known predictive feature in
this dataset, so this keeps that signal without carrying 45 near-duplicate
columns through every downstream model.

**Intermediate** (`warehouse/models/intermediate/`): this is where the
one-to-many joins actually get resolved. `bureau_balance` (grain:
`sk_id_bureau, months_balance`) is aggregated up to `sk_id_bureau` first
(`int_bureau_balance_agg`), then joined onto `bureau` and aggregated again
up to `sk_id_curr` (`int_bureau_agg`) - a genuine two-level rollup, since
`bureau_balance` doesn't carry `sk_id_curr` directly. The other four history
tables (`previous_application`, `pos_cash_balance`, `credit_card_balance`,
`installments_payments`) do carry `sk_id_curr` directly and are aggregated
to that grain in one step each. Each aggregation produces both raw
summaries (counts, sums) and engineered ratios (debt-to-credit,
late-payment rate, credit-card utilization, approval rate) - the kind of
feature a lender's risk team would actually reason about, not just a
mechanical `AVG(*)` over every numeric column.

**Marts** (`warehouse/models/marts/`): `mart_applicant_features` left-joins
the applicant record to all six aggregations. Count-style columns are
coalesced to `0` when an applicant has no history in a source (a true
zero); ratio/average columns are left `NULL` (LightGBM and XGBoost both
split on missingness natively - imputing a sentinel would destroy that
signal). A `has_*_history` boolean is added per source, so "no history" is
still directly queryable and shows up as its own SHAP feature instead of
being buried in nulls.

**dbt tests**: `unique`/`not_null` on every grain key, `relationships` tests
enforcing every history table's foreign key actually exists in its parent
(e.g. every `bureau.sk_id_curr` exists in `application`), an
`accepted_values` test on `target`, and a singular test
(`warehouse/tests/assert_train_test_target_consistency.sql`) asserting
training rows always have a label and scoring rows never do.

## Modeling (`modeling/`)

- **`train.py`** - LightGBM (`objective=binary`), with `scale_pos_weight`
  (negative/positive count) handling the ~8% positive rate, and early
  stopping on a held-out validation fold. `scale_pos_weight` was chosen over
  resampling (SMOTE, random oversampling): it reweights the existing
  gradient/hessian per class rather than fabricating synthetic rows in a
  feature space built from subtle, correlated financial ratios, where
  interpolating between real applicants is more likely to produce
  unrealistic rows than reweighting is to mis-calibrate.
- **`calibrate.py`** - `scale_pos_weight` fixes ranking (AUC) but skews the
  raw predicted probabilities toward the minority class. Isotonic
  regression (`sklearn.calibration.CalibratedClassifierCV` + `FrozenEstimator`),
  fit on the same held-out fold used for early stopping, maps scores back to
  well-calibrated probabilities without touching ranking.
- **`explain.py`** - SHAP `TreeExplainer` on the underlying LightGBM model
  (not the calibrated wrapper - see the module docstring for why),
  producing `reports/shap_summary.png` and `reports/shap_feature_importance.csv`.
- **`evaluate.py`** - ROC-AUC, PR-AUC, KS statistic, and a confusion matrix
  at the F1-optimal threshold, written to `reports/evaluation_metrics.json`.
- **`predict.py`** - scores `application_test` with the calibrated model and
  writes a Kaggle-format `reports/submission.csv`.
- **`run_pipeline.py`** - runs all of the above in order.

`data.py` and `split.py` are shared: every stage reads the same
feature-column list and the same train/valid split, so training,
calibration, explanation, and evaluation can never silently disagree about
what a "feature" is or which rows were held out.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. place the Kaggle CSVs in data/raw/ (see "Expected raw data" above)

# 2. build the warehouse: load raw CSVs -> dbt build (staging -> intermediate -> marts) -> dbt docs generate
./scripts/build_warehouse.sh

# 3. run the modeling pipeline end-to-end
python -m modeling.run_pipeline
```

Outputs land in `models/` (trained + calibrated model, feature metadata)
and `reports/` (SHAP plot/table, evaluation metrics, Kaggle submission
CSV) - both gitignored, regenerated by the commands above.

### Trying it without the real dataset

`tests/generate_synthetic_data.py` generates small, schema-faithful
synthetic CSVs (proper referential integrity between `sk_id_curr`,
`sk_id_bureau`, `sk_id_prev`) for all eight tables, so the whole pipeline
can be exercised without the (non-redistributable) Kaggle download:

```bash
python tests/generate_synthetic_data.py         # writes tests/fixtures/raw/
RAW_DIR=tests/fixtures/raw DUCKDB_PATH=/tmp/dev.duckdb python scripts/load_raw_data.py
DBT_PROFILES_DIR=warehouse DUCKDB_PATH=/tmp/dev.duckdb dbt build --project-dir warehouse
DUCKDB_PATH=/tmp/dev.duckdb python -m modeling.run_pipeline
```

This is exactly what CI (`.github/workflows/ci.yml`) does. `pytest tests/`
runs against whatever DuckDB file `DUCKDB_PATH` points at (real or
synthetic) and skips itself if that warehouse hasn't been built yet - it's
an integration suite on top of the dbt build, not a replacement for it.

## Design decisions / what's deliberately out of scope

- **DuckDB, not Postgres/Snowflake.** Zero external services to provision;
  `dbt-duckdb` gives a real warehouse (schemas, materializations, tests,
  docs) against a single local file.
- **`scale_pos_weight`, not SMOTE.** See `modeling/train.py`'s docstring -
  reweighting composes cleanly with the calibration step that follows it;
  resampling would need its own calibration correction on top.
- **LightGBM over XGBoost.** Both are equally valid for this problem;
  LightGBM's native pandas-categorical support avoids a separate encoding
  step for the dataset's many string columns. Swapping in
  `xgboost.XGBClassifier` (with `enable_categorical=True`) is a small,
  contained change to `modeling/train.py` if preferred.
- **Housing/building columns collapsed, not dropped or kept in full.** See
  the staging section above - keeps the missingness signal without ~45
  near-duplicate columns.
- **No feature store / Airflow.** The project is scoped to one batch build;
  orchestration beyond `scripts/build_warehouse.sh` would be solving a
  problem this dataset doesn't have.

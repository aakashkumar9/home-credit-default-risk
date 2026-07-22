"""Runs the full modelling pipeline: stratified k-fold cross-validates
three model types (baseline logistic regression, LightGBM, XGBoost), picks
the champion by mean CV PR-AUC (the imbalance-appropriate ranking metric -
ROC-AUC is logged too, but PR-AUC is what actually distinguishes models at
an ~8% positive rate), fits the champion on a training split, calibrates +
evaluates it on a held-out split it never saw, and logs all of it to
MLflow.

    python -m home_credit.modeling.train
"""

import json

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from home_credit import config, data
from home_credit.features import (
    extract_categories,
    get_feature_columns,
    prepare_tree_dtypes,
    split_column_types,
)
from home_credit.modeling import calibrate as calibrate_mod
from home_credit.modeling import evaluate as evaluate_mod
from home_credit.modeling.cv import cross_validate
from home_credit.modeling.models import LightGBMModel, LogisticRegressionModel, XGBoostModel
from home_credit.validation.schemas import validate_mart

MODEL_NAMES = ["logistic_regression", "lightgbm", "xgboost"]


def _model_factory(
    model_name: str, numeric_cols: list[str], categorical_cols: list[str], seed: int
):
    if model_name == "logistic_regression":
        return lambda: LogisticRegressionModel(numeric_cols, categorical_cols, seed)
    if model_name == "lightgbm":
        return lambda: LightGBMModel(categorical_cols, seed)
    if model_name == "xgboost":
        return lambda: XGBoostModel(categorical_cols, seed)
    raise ValueError(f"unknown model_name: {model_name}")


def train(duckdb_path: str | None = None) -> dict:
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    df = data.load_mart(duckdb_path)
    validate_mart(df)  # fail fast on a broken/drifted mart before spending time training on it
    train_df, _ = data.split_train_score(df)
    feature_cols = get_feature_columns(df)
    numeric_cols, categorical_cols = split_column_types(df, feature_cols)

    # Two typed views of the same rows (same row order, so positionally aligned):
    # the tree models (LightGBM/XGBoost) get categoricals cast to pandas
    # 'category' dtype so they split on them natively. The logistic regression
    # gets the untouched frame instead - combining several 'category'-dtype
    # columns whose underlying values are different types (e.g. boolean flags
    # alongside string columns) hits a real pandas/sklearn dtype-promotion bug
    # in SimpleImputer ("could not convert string to float"), confirmed by
    # testing the same columns through both paths; the plain-dtype frame
    # doesn't hit it, and OneHotEncoder/SimpleImputer don't need 'category'
    # dtype to work correctly anyway.
    tree_df = prepare_tree_dtypes(train_df, categorical_cols)
    # Captured once here, from the full training set, and persisted below -
    # every later single-row inference (predict.py, explain.py, the API,
    # the dashboard) must reuse these exact categories rather than deriving
    # fresh ones from whatever row(s) it happens to see. See
    # prepare_tree_dtypes' docstring for why that matters.
    categorical_categories = extract_categories(train_df, categorical_cols)
    y = train_df[config.TARGET_COL].astype(int)

    def model_frame(model_name: str) -> pd.DataFrame:
        return train_df if model_name == "logistic_regression" else tree_df

    with mlflow.start_run(run_name="model_comparison"):
        mlflow.log_param("n_cv_splits", config.N_CV_SPLITS)
        mlflow.log_param("n_train_rows", len(train_df))
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("target_rate", float(y.mean()))

        cv_results = {}
        for model_name in MODEL_NAMES:
            factory = _model_factory(model_name, numeric_cols, categorical_cols, config.RANDOM_SEED)
            X_model = model_frame(model_name)[feature_cols]
            with mlflow.start_run(run_name=model_name, nested=True):
                result = cross_validate(model_name, factory, X_model, y)
                cv_results[model_name] = result.to_dict()
                mlflow.log_metric("mean_roc_auc", result.mean_roc_auc)
                mlflow.log_metric("std_roc_auc", result.std_roc_auc)
                mlflow.log_metric("mean_pr_auc", result.mean_pr_auc)
                mlflow.log_metric("std_pr_auc", result.std_pr_auc)
                for i, (roc, pr) in enumerate(zip(result.fold_roc_auc, result.fold_pr_auc)):
                    mlflow.log_metric("fold_roc_auc", roc, step=i)
                    mlflow.log_metric("fold_pr_auc", pr, step=i)

        champion_name = max(cv_results, key=lambda name: cv_results[name]["mean_pr_auc"])
        mlflow.log_param("champion_model", champion_name)

        # champion is fit on outer_fit and evaluated on calibration_holdout, which
        # it never sees during training - both for calibration and the final
        # reported metrics.
        outer_fit_idx, holdout_idx = train_test_split(
            np.arange(len(train_df)),
            test_size=config.CALIBRATION_HOLDOUT_SIZE,
            stratify=y,
            random_state=config.RANDOM_SEED,
        )
        champion_df = model_frame(champion_name)
        X_fit = champion_df.iloc[outer_fit_idx][feature_cols]
        y_fit = y.iloc[outer_fit_idx]
        X_holdout = champion_df.iloc[holdout_idx][feature_cols]
        y_holdout = y.iloc[holdout_idx]

        champion_factory = _model_factory(
            champion_name, numeric_cols, categorical_cols, config.RANDOM_SEED
        )
        champion_model = champion_factory().fit(X_fit, y_fit)

        uncalibrated_proba = champion_model.predict_proba(X_holdout)
        uncalibrated_metrics = evaluate_mod.evaluate_predictions(y_holdout, uncalibrated_proba)

        calibrated_model = calibrate_mod.calibrate(champion_model, X_holdout, y_holdout)
        calibrated_proba = calibrated_model.predict_proba(X_holdout)[:, 1]
        calibrated_metrics = evaluate_mod.evaluate_predictions(y_holdout, calibrated_proba)

        calibration_plot_path = config.REPORTS_DIR / "calibration_curve.png"
        evaluate_mod.plot_calibration_curve(
            y_holdout, uncalibrated_proba, calibrated_proba, calibration_plot_path
        )

        for k, v in uncalibrated_metrics.items():
            mlflow.log_metric(f"holdout_uncalibrated_{k}", v)
        for k, v in calibrated_metrics.items():
            mlflow.log_metric(f"holdout_calibrated_{k}", v)
        mlflow.log_artifact(str(calibration_plot_path))

        joblib.dump(champion_model.sklearn_estimator, config.MODEL_DIR / "champion_model.joblib")
        joblib.dump(calibrated_model, config.MODEL_DIR / "calibrated_model.joblib")
        # Also version the served model itself through MLflow's model registry - the
        # joblib file above remains the fast, dependency-free path serving.py/dashboard
        # actually load from, but the registry gives a versioned, queryable history of
        # every calibrated model ever produced across runs (mlflow ui / mlflow models).
        # serialization_format is pinned to cloudpickle rather than mlflow's default
        # (skops) - skops isn't guaranteed to round-trip the LightGBM/XGBoost sklearn
        # wrappers CalibratedClassifierCV can hold; cloudpickle matches what joblib.dump
        # above already uses for the same object.
        mlflow.sklearn.log_model(
            calibrated_model,
            name="calibrated_model",
            registered_model_name=config.SERVING_MODEL_NAME,
            serialization_format="cloudpickle",
        )
        with open(config.MODEL_DIR / "feature_metadata.json", "w") as f:
            json.dump(
                {
                    "feature_cols": feature_cols,
                    "numeric_cols": numeric_cols,
                    "categorical_cols": categorical_cols,
                    "categorical_categories": categorical_categories,
                    "champion_model": champion_name,
                },
                f,
                indent=2,
            )

        summary = {
            "cv_results": cv_results,
            "champion_model": champion_name,
            "holdout_uncalibrated_metrics": uncalibrated_metrics,
            "holdout_calibrated_metrics": calibrated_metrics,
        }
        summary_path = config.REPORTS_DIR / "train_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(str(summary_path))

    return summary


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))

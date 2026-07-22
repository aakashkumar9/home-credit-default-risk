import numpy as np
import pandas as pd
import pytest

from home_credit import config
from home_credit.modeling import calibrate as calibrate_mod
from home_credit.modeling import evaluate as evaluate_mod
from home_credit.modeling import predict as predict_mod
from home_credit.modeling import train as train_mod
from home_credit.modeling.cv import cross_validate
from home_credit.modeling.models import LightGBMModel, LogisticRegressionModel, XGBoostModel


@pytest.fixture
def synthetic_features():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(
        {
            "num_a": rng.normal(size=n),
            "num_b": rng.normal(size=n),
            "flag": rng.integers(0, 2, size=n).astype(bool),  # numeric per split_column_types
            "cat_a": rng.choice(["X", "Y", "Z"], size=n),
        }
    )
    y = pd.Series(rng.integers(0, 2, size=n), name="target")
    return X, y


def test_cross_validate_returns_one_score_per_fold(synthetic_features):
    X, y = synthetic_features
    numeric_cols, categorical_cols = ["num_a", "num_b", "flag"], ["cat_a"]

    def factory():
        return LogisticRegressionModel(numeric_cols, categorical_cols, seed=0)

    result = cross_validate("logistic_regression", factory, X, y, n_splits=4, seed=0)

    assert len(result.fold_roc_auc) == 4
    assert len(result.fold_pr_auc) == 4
    assert 0.0 <= result.mean_roc_auc <= 1.0
    assert 0.0 <= result.mean_pr_auc <= 1.0


def test_logistic_regression_model_fits_and_predicts(synthetic_features):
    X, y = synthetic_features
    model = LogisticRegressionModel(["num_a", "num_b", "flag"], ["cat_a"], seed=0).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert model.sklearn_estimator is not None


def test_lightgbm_model_fits_and_predicts_with_categoricals(synthetic_features):
    X, y = synthetic_features
    X = X.copy()
    X["cat_a"] = X["cat_a"].astype("category")
    model = LightGBMModel(["cat_a"], seed=0).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_xgboost_model_fits_and_predicts_with_categoricals(synthetic_features):
    # this is the case that previously crashed: a boolean-flag-turned-numeric
    # column alongside a genuine string categorical, both feeding a tree model
    X, y = synthetic_features
    X = X.copy()
    X["cat_a"] = X["cat_a"].astype("category")
    model = XGBoostModel(["cat_a"], seed=0).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_calibrate_produces_valid_probabilities(synthetic_features):
    X, y = synthetic_features
    model = LogisticRegressionModel(["num_a", "num_b", "flag"], ["cat_a"], seed=0).fit(X, y)
    calibrated = calibrate_mod.calibrate(model, X, y)
    proba = calibrated.predict_proba(X)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_evaluate_predictions_returns_metrics_in_valid_ranges():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_proba = rng.random(200)
    metrics = evaluate_mod.evaluate_predictions(y_true, y_proba)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["ks_statistic"] <= 1.0
    assert metrics["brier_score"] >= 0.0


def test_full_train_predict_pipeline_against_real_mart(mart_db_path, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(config, "N_CV_SPLITS", 3)  # small synthetic fixture, keep folds usable

    summary = train_mod.train(mart_db_path)

    assert summary["champion_model"] in ("logistic_regression", "lightgbm", "xgboost")
    for model_result in summary["cv_results"].values():
        assert len(model_result["fold_roc_auc"]) == 3
    for metrics in (summary["holdout_uncalibrated_metrics"], summary["holdout_calibrated_metrics"]):
        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert 0.0 <= metrics["pr_auc"] <= 1.0

    assert (config.MODEL_DIR / "champion_model.joblib").exists()
    assert (config.MODEL_DIR / "calibrated_model.joblib").exists()
    assert (config.REPORTS_DIR / "calibration_curve.png").exists()

    submission = predict_mod.predict(mart_db_path)
    assert list(submission.columns) == ["SK_ID_CURR", "TARGET"]
    assert submission["TARGET"].between(0, 1).all()
    assert submission["SK_ID_CURR"].is_unique

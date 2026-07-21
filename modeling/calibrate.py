"""Calibrates the trained LightGBM model's predicted probabilities.

scale_pos_weight in train.py fixes ranking (AUC) but skews the predicted
probabilities themselves toward the minority class - a 0.5 output no longer
means "50% default risk". Isotonic regression, fit on the same held-out
validation fold used for early stopping (never seen in training), maps raw
scores back to well-calibrated probabilities without touching ranking, which
matters if the output probability is ever used directly (risk-based pricing,
regulatory reporting) rather than just for ranking applicants.
"""
import json

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, log_loss

from modeling import config, data
from modeling.split import train_valid_split


def calibrate(duckdb_path: str | None = None) -> dict:
    model = joblib.load(config.MODEL_DIR / "lgbm_model.joblib")
    with open(config.MODEL_DIR / "feature_metadata.json") as f:
        meta = json.load(f)
    feature_cols, categorical_cols = meta["feature_cols"], meta["categorical_cols"]

    df = data.load_mart(duckdb_path)
    df = data.prepare_dtypes(df, feature_cols)
    train_df, _ = data.split_train_score(df)
    _, valid_df = train_valid_split(train_df)

    X_valid = valid_df[feature_cols]
    y_valid = valid_df[config.TARGET_COL].astype(int)

    uncalibrated_proba = model.predict_proba(X_valid)[:, 1]

    calibrated_model = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    calibrated_model.fit(X_valid, y_valid)
    calibrated_proba = calibrated_model.predict_proba(X_valid)[:, 1]

    metrics = {
        "brier_score_uncalibrated": float(brier_score_loss(y_valid, uncalibrated_proba)),
        "brier_score_calibrated": float(brier_score_loss(y_valid, calibrated_proba)),
        "log_loss_uncalibrated": float(log_loss(y_valid, uncalibrated_proba)),
        "log_loss_calibrated": float(log_loss(y_valid, calibrated_proba)),
    }

    joblib.dump(calibrated_model, config.MODEL_DIR / "calibrated_model.joblib")
    with open(config.MODEL_DIR / "calibration_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    print(json.dumps(calibrate(), indent=2))

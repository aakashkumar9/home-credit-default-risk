"""Trains a LightGBM classifier on mart_applicant_features.

Class imbalance (~8% positive) is handled via scale_pos_weight (negative /
positive count on the training fold) rather than resampling (SMOTE, random
oversampling): scale_pos_weight just reweights the existing gradient/hessian
per class, so it doesn't fabricate synthetic feature rows for a dataset that
already leans on subtle, correlated financial ratios - synthetic
interpolation between real applicants in that feature space is more likely
to produce unrealistic rows than a straightforward reweighting is to
mis-calibrate. It composes cleanly with the probability calibration step in
calibrate.py, which corrects exactly the distortion scale_pos_weight
introduces (predicted probabilities biased toward the minority class).
"""
import json

import joblib
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score

from modeling import config, data
from modeling.split import train_valid_split


def train(duckdb_path: str | None = None) -> dict:
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = data.load_mart(duckdb_path)
    feature_cols = data.get_feature_columns(df)
    df = data.prepare_dtypes(df, feature_cols)
    categorical_cols = data.get_categorical_columns(df, feature_cols)

    train_df, _ = data.split_train_score(df)
    fit_df, valid_df = train_valid_split(train_df)

    X_fit, y_fit = fit_df[feature_cols], fit_df[config.TARGET_COL].astype(int)
    X_valid, y_valid = valid_df[feature_cols], valid_df[config.TARGET_COL].astype(int)

    scale_pos_weight = (y_fit == 0).sum() / (y_fit == 1).sum()

    model = lgb.LGBMClassifier(**config.LGBM_PARAMS, scale_pos_weight=scale_pos_weight)
    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        categorical_feature=categorical_cols,
        callbacks=[lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
    )

    valid_proba = model.predict_proba(X_valid)[:, 1]
    metrics = {
        "scale_pos_weight": float(scale_pos_weight),
        "best_iteration": int(model.best_iteration_),
        "valid_roc_auc": float(roc_auc_score(y_valid, valid_proba)),
        "valid_pr_auc": float(average_precision_score(y_valid, valid_proba)),
    }

    joblib.dump(model, config.MODEL_DIR / "lgbm_model.joblib")
    with open(config.MODEL_DIR / "feature_metadata.json", "w") as f:
        json.dump({"feature_cols": feature_cols, "categorical_cols": categorical_cols}, f, indent=2)
    with open(config.MODEL_DIR / "train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))

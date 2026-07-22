"""Scores application_test (is_train = false) with the calibrated champion
model and writes a Kaggle-format submission.csv: SK_ID_CURR, TARGET.
"""
import json

import joblib
import pandas as pd

from home_credit import config, data
from home_credit.features import prepare_tree_dtypes


def predict(duckdb_path: str | None = None) -> pd.DataFrame:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    calibrated_model = joblib.load(config.MODEL_DIR / "calibrated_model.joblib")
    with open(config.MODEL_DIR / "feature_metadata.json") as f:
        meta = json.load(f)
    feature_cols, categorical_cols = meta["feature_cols"], meta["categorical_cols"]

    df = data.load_mart(duckdb_path)
    _, score_df = data.split_train_score(df)
    # Only the tree champions (LightGBM/XGBoost) need categoricals cast to
    # pandas 'category' dtype - they were trained on that representation and
    # expect it at predict time too. The logistic regression's own pipeline
    # (bundled inside the calibrated model) does its own preprocessing on raw
    # dtypes - see train.py's module docstring for why it can't take the
    # category-cast frame.
    if meta["champion_model"] != "logistic_regression":
        score_df = prepare_tree_dtypes(score_df, categorical_cols)

    proba = calibrated_model.predict_proba(score_df[feature_cols])[:, 1]
    submission = pd.DataFrame({
        "SK_ID_CURR": score_df[config.ID_COL].astype(int),
        "TARGET": proba,
    })
    submission.to_csv(config.REPORTS_DIR / "submission.csv", index=False)
    return submission


if __name__ == "__main__":
    result = predict()
    print(f"Wrote {len(result)} predictions to {config.REPORTS_DIR / 'submission.csv'}")

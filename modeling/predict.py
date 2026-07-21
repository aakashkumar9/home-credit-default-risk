"""Scores application_test (is_train = false) with the calibrated model and
writes a Kaggle-format submission.csv: SK_ID_CURR, TARGET (predicted probability).
"""
import json

import joblib
import pandas as pd

from modeling import config, data


def predict(duckdb_path: str | None = None) -> pd.DataFrame:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    calibrated_model = joblib.load(config.MODEL_DIR / "calibrated_model.joblib")
    with open(config.MODEL_DIR / "feature_metadata.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]

    df = data.load_mart(duckdb_path)
    df = data.prepare_dtypes(df, feature_cols)
    _, score_df = data.split_train_score(df)

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

"""SHAP explainability for the trained LightGBM model.

Explains the underlying (uncalibrated) LGBMClassifier directly - it's a
plain tree ensemble, so shap.TreeExplainer gives exact SHAP values cheaply.
Explaining the calibrated wrapper instead would mean explaining an isotonic
regression stacked on top of the trees, which TreeExplainer can't do exactly
and which SHAP would then attribute to the wrong scale (post-calibration
probability, not the tree's own log-odds output) - the trees are what make
the risk decision, calibration only rescales it, so the trees are the
correct explanation target.
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from modeling import config, data


def explain(duckdb_path: str | None = None, sample_size: int = 2000) -> dict:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model = joblib.load(config.MODEL_DIR / "lgbm_model.joblib")
    with open(config.MODEL_DIR / "feature_metadata.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]

    df = data.load_mart(duckdb_path)
    df = data.prepare_dtypes(df, feature_cols)
    train_df, _ = data.split_train_score(df)

    sample = train_df.sample(n=min(sample_size, len(train_df)), random_state=config.RANDOM_SEED)
    X_sample = sample[feature_cols]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    # binary LGBMClassifier via shap can return a single 2D array (class-1 shap values)
    # or a list of two arrays ([class-0, class-1]) depending on shap/lightgbm versions.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols)
    top_features = mean_abs_shap.sort_values(ascending=False)

    top_features.to_csv(config.REPORTS_DIR / "shap_feature_importance.csv", header=["mean_abs_shap"])

    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / "shap_summary.png", dpi=150)
    plt.close()

    return {"top_20_features": top_features.head(20).to_dict()}


if __name__ == "__main__":
    result = explain()
    print(json.dumps(result, indent=2))

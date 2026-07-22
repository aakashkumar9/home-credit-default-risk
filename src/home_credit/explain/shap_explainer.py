"""Global and per-applicant SHAP explanations.

Explains the champion's own underlying estimator (`champion_model.joblib` -
the raw fitted classifier saved by train.py, not the calibrated wrapper)
via `shap.TreeExplainer`, exact and fast for the tree models. Explaining
the isotonic-calibrated wrapper instead would mean explaining a monotonic
regression stacked on top of the trees, which `TreeExplainer` can't do
exactly and would attribute credit on the wrong scale (post-calibration
probability, not the tree's own margin) - the trees make the risk
decision, calibration only rescales it, so the trees are the correct
explanation target.

Scoped to the tree champions (LightGBM/XGBoost) only. A model-agnostic
fallback for the baseline logistic regression was tried and dropped: SHAP's
default masker computes `numpy.isclose` differences directly against the
raw background sample, which crashes on this dataset's string-typed
categorical columns before ever reaching the pipeline's own preprocessing.
Making that path work (explaining the pipeline in its post-one-hot-encoded
space, then mapping SHAP values on expanded dummy columns back to the
original business features) is real, non-trivial work that's out of scope
here - and given this dataset's heavy missingness and many categoricals,
a tree model reliably outperforms the linear baseline in practice anyway,
so `train()` raises a clear `NotImplementedError` rather than silently
running a fallback that's actually broken.
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from home_credit import config, data
from home_credit.features import prepare_tree_dtypes

TREE_MODEL_NAMES = {"lightgbm", "xgboost"}


def load_champion():
    """Loads the champion's raw estimator + feature metadata from disk.
    Shared with home_credit.serving, which loads this once at startup
    rather than per-request - see that module for why.
    """
    with open(config.MODEL_DIR / "feature_metadata.json") as f:
        meta = json.load(f)
    champion = joblib.load(config.MODEL_DIR / "champion_model.joblib")
    return champion, meta


def build_explainer(champion_model_type: str, champion):
    if champion_model_type not in TREE_MODEL_NAMES:
        raise NotImplementedError(
            f"SHAP explanation isn't implemented for champion_model={champion_model_type!r} - "
            "see this module's docstring for why. Scoped to LightGBM/XGBoost champions."
        )
    return shap.TreeExplainer(champion)


def positive_class_values(shap_values: np.ndarray) -> np.ndarray:
    """SHAP's array shape for binary classifiers varies by explainer/version:
    (n_samples, n_features) already for the positive class, or
    (n_samples, n_features, n_classes) with both classes - normalize to the
    former, always for the positive (index 1) class.
    """
    values = np.asarray(shap_values)
    if values.ndim == 3:
        return values[:, :, 1]
    return values


def positive_class_base_value(base_values) -> np.ndarray:
    base = np.asarray(base_values)
    if base.ndim == 2:
        return base[:, 1]
    return base


def compute_shap(champion_type: str, champion, X: pd.DataFrame):
    """Returns (shap_matrix, base_values) - both already normalized to the
    positive-class scale, shapes (n_rows, n_features) and (n_rows,).
    """
    explainer = build_explainer(champion_type, champion)
    explanation = explainer(X)
    return positive_class_values(explanation.values), positive_class_base_value(explanation.base_values)


def compute_global_importance(duckdb_path: str | None = None, sample_size: int = 2000):
    champion, meta = load_champion()
    feature_cols, categorical_cols, champion_type = (
        meta["feature_cols"],
        meta["categorical_cols"],
        meta["champion_model"],
    )

    df = data.load_mart(duckdb_path)
    train_df, _ = data.split_train_score(df)
    if champion_type in TREE_MODEL_NAMES:
        train_df = prepare_tree_dtypes(train_df, categorical_cols, categories=meta["categorical_categories"])

    sample = train_df.sample(n=min(sample_size, len(train_df)), random_state=config.RANDOM_SEED)
    X_sample = sample[feature_cols]

    shap_matrix, _ = compute_shap(champion_type, champion, X_sample)
    mean_abs_shap = pd.Series(np.abs(shap_matrix).mean(axis=0), index=feature_cols)
    mean_abs_shap = mean_abs_shap.sort_values(ascending=False)
    return mean_abs_shap, X_sample, shap_matrix


def plot_global_summary(X_sample: pd.DataFrame, shap_matrix: np.ndarray, out_path):
    shap.summary_plot(shap_matrix, X_sample, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def contributions_for_row(feature_cols: list[str], row_values, shap_row: np.ndarray) -> list[dict]:
    """Builds the feature/value/shap_value records for one applicant,
    sorted by |shap_value| descending. Shared between explain_applicant
    (below, whole-mart batch/CLI path) and home_credit.serving (single-row
    API path) so the two never format this differently.
    """
    contributions = (
        pd.DataFrame({"feature": feature_cols, "value": row_values, "shap_value": shap_row})
        .assign(abs_shap=lambda d: d["shap_value"].abs())
        .sort_values("abs_shap", ascending=False)
        .drop(columns="abs_shap")
    )
    return contributions.to_dict(orient="records")


def explain_applicant(sk_id_curr: int, duckdb_path: str | None = None) -> dict:
    """Per-applicant SHAP breakdown: base value + each feature's
    contribution to that applicant's predicted (uncalibrated) score, sorted
    by |contribution|. Loads the whole mart - fine for CLI/batch use; the
    API (phase 7) does a single-row DuckDB lookup instead, since loading
    the full mart per HTTP request wouldn't scale to real data size.
    """
    champion, meta = load_champion()
    feature_cols, categorical_cols, champion_type = (
        meta["feature_cols"],
        meta["categorical_cols"],
        meta["champion_model"],
    )

    df = data.load_mart(duckdb_path)
    if champion_type in TREE_MODEL_NAMES:
        df = prepare_tree_dtypes(df, categorical_cols, categories=meta["categorical_categories"])

    row = df[df[config.ID_COL] == sk_id_curr]
    if row.empty:
        raise ValueError(f"sk_id_curr {sk_id_curr} not found in the mart")

    shap_matrix, base_values = compute_shap(champion_type, champion, row[feature_cols])
    shap_row = shap_matrix[0]
    base_value = float(base_values[0]) if base_values.ndim else float(base_values)

    return {
        "sk_id_curr": int(sk_id_curr),
        "base_value": base_value,
        "predicted_score": base_value + float(shap_row.sum()),
        "contributions": contributions_for_row(feature_cols, row[feature_cols].iloc[0].to_numpy(), shap_row),
    }


def main():
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    mean_abs_shap, X_sample, shap_matrix = compute_global_importance()
    mean_abs_shap.to_csv(config.REPORTS_DIR / "shap_feature_importance.csv", header=["mean_abs_shap"])
    plot_global_summary(X_sample, shap_matrix, config.REPORTS_DIR / "shap_summary.png")
    print(
        f"Wrote {config.REPORTS_DIR / 'shap_feature_importance.csv'} "
        f"and {config.REPORTS_DIR / 'shap_summary.png'}"
    )


if __name__ == "__main__":
    main()

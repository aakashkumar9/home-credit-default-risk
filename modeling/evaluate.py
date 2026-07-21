"""Final evaluation of the calibrated model on the held-out validation fold:
ROC-AUC, PR-AUC, KS statistic, and a confusion matrix at the F1-optimal
threshold. Evaluating on the same validation fold used for calibration is
optimistic (the isotonic regressor was fit on this exact fold) - fine for a
portfolio project, but a production build would carve out a third, fully
untouched fold for this final number.
"""
import json

import joblib
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from modeling import config, data
from modeling.split import train_valid_split


def ks_statistic(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float(np.max(tpr - fpr))


def best_f1_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return float(thresholds[np.argmax(f1[:-1])]) if len(thresholds) else 0.5


def evaluate(duckdb_path: str | None = None) -> dict:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    calibrated_model = joblib.load(config.MODEL_DIR / "calibrated_model.joblib")
    with open(config.MODEL_DIR / "feature_metadata.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]

    df = data.load_mart(duckdb_path)
    df = data.prepare_dtypes(df, feature_cols)
    train_df, _ = data.split_train_score(df)
    _, valid_df = train_valid_split(train_df)

    X_valid = valid_df[feature_cols]
    y_valid = valid_df[config.TARGET_COL].astype(int).to_numpy()
    y_proba = calibrated_model.predict_proba(X_valid)[:, 1]

    threshold = best_f1_threshold(y_valid, y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_valid, y_pred).ravel()

    metrics = {
        "roc_auc": float(roc_auc_score(y_valid, y_proba)),
        "pr_auc": float(average_precision_score(y_valid, y_proba)),
        "ks_statistic": ks_statistic(y_valid, y_proba),
        "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision_at_threshold": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "recall_at_threshold": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
    }

    with open(config.REPORTS_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))

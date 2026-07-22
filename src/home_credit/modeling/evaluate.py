"""Final evaluation: ROC-AUC, PR-AUC, KS statistic, Brier score, and a
calibration curve (reliability diagram) comparing uncalibrated vs.
calibrated probabilities - all on the same held-out set used to fit the
calibrator, since that's the only data the champion model never trained
on. A production build would carve out a third, fully untouched fold for
this final number; here it's a portfolio-scoped simplification, noted
because it matters for interpreting the metrics honestly.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def ks_statistic(y_true, y_proba) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float(np.max(tpr - fpr))


def evaluate_predictions(y_true, y_proba) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "ks_statistic": ks_statistic(y_true, y_proba),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
    }


def plot_calibration_curve(y_true, uncalibrated_proba, calibrated_proba, out_path, n_bins: int = 10):
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, proba in [("uncalibrated", uncalibrated_proba), ("calibrated (isotonic)", calibrated_proba)]:
        frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

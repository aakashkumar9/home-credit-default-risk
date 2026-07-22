"""Calibrates a fitted model's predicted probabilities.

Both `scale_pos_weight` (tree models) and `class_weight='balanced'`
(logistic regression) fix ranking (AUC) but skew the predicted
probabilities themselves toward the minority class - a 0.5 output no
longer means "50% default risk." Isotonic regression, fit on a holdout the
model never saw during training, maps raw scores back to well-calibrated
probabilities without touching ranking.

`CalibratedClassifierCV`, even wrapping an already-fitted (frozen) base
model, still runs its own internal 5-fold CV by default to fit an ensemble
of isotonic regressors on the holdout (reduces calibration variance vs. a
single fit) - this is separate from, and doesn't touch, the CV used to
compare model types in cv.py. On a small holdout (as with the synthetic
test fixtures) sklearn may warn that a fold has too few minority-class
examples; with real data's full holdout size this doesn't occur.
"""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator


def calibrate(fitted_model, X_holdout, y_holdout) -> CalibratedClassifierCV:
    """fitted_model must expose `.sklearn_estimator` (see models.py) - the
    underlying fitted sklearn-compatible classifier to wrap and calibrate.
    """
    calibrated = CalibratedClassifierCV(FrozenEstimator(fitted_model.sklearn_estimator), method="isotonic")
    calibrated.fit(X_holdout, y_holdout)
    return calibrated

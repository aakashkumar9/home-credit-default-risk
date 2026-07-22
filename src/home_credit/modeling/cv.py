"""Stratified k-fold cross-validation, model-agnostic: takes a zero-arg
factory that returns a fresh, unfitted model (matching the
fit(X, y)/predict_proba(X) interface in models.py), so the same loop
compares the baseline logistic regression, LightGBM, and XGBoost without
knowing anything about their internals.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from home_credit import config


@dataclass
class CVResult:
    model_name: str
    fold_roc_auc: list[float] = field(default_factory=list)
    fold_pr_auc: list[float] = field(default_factory=list)

    @property
    def mean_roc_auc(self) -> float:
        return float(np.mean(self.fold_roc_auc))

    @property
    def std_roc_auc(self) -> float:
        return float(np.std(self.fold_roc_auc))

    @property
    def mean_pr_auc(self) -> float:
        return float(np.mean(self.fold_pr_auc))

    @property
    def std_pr_auc(self) -> float:
        return float(np.std(self.fold_pr_auc))

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "fold_roc_auc": self.fold_roc_auc,
            "fold_pr_auc": self.fold_pr_auc,
            "mean_roc_auc": self.mean_roc_auc,
            "std_roc_auc": self.std_roc_auc,
            "mean_pr_auc": self.mean_pr_auc,
            "std_pr_auc": self.std_pr_auc,
        }


def cross_validate(
    model_name: str,
    model_factory: Callable[[], object],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int | None = None,
    seed: int | None = None,
) -> CVResult:
    # resolved inside the body, not as a default-argument value - a default
    # baked in at def time (`n_splits: int = config.N_CV_SPLITS`) would ignore
    # any later `monkeypatch.setattr(config, "N_CV_SPLITS", ...)`, since
    # Python evaluates default arguments once, at function definition time.
    n_splits = n_splits if n_splits is not None else config.N_CV_SPLITS
    seed = seed if seed is not None else config.RANDOM_SEED
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    result = CVResult(model_name=model_name)

    for fold_fit_idx, fold_holdout_idx in skf.split(X, y):
        X_fit, X_holdout = X.iloc[fold_fit_idx], X.iloc[fold_holdout_idx]
        y_fit, y_holdout = y.iloc[fold_fit_idx], y.iloc[fold_holdout_idx]

        model = model_factory().fit(X_fit, y_fit)
        proba = model.predict_proba(X_holdout)

        result.fold_roc_auc.append(float(roc_auc_score(y_holdout, proba)))
        result.fold_pr_auc.append(float(average_precision_score(y_holdout, proba)))

    return result

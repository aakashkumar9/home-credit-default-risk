"""Three model types behind one shared interface (`fit(X, y)` /
`predict_proba(X)`), so cv.py can cross-validate any of them identically.

Class imbalance (~8% positive) is handled the same way for both tree
models: `scale_pos_weight` (negative/positive count on the model's own
training data), not resampling. It reweights the existing gradient/hessian
per class rather than fabricating synthetic rows in a feature space built
from subtle, correlated financial ratios - synthetic interpolation between
real applicants is more likely to produce unrealistic rows than reweighting
is to mis-calibrate. It also composes cleanly with the calibration step
that follows training, which corrects exactly the probability skew
scale_pos_weight introduces. The baseline logistic regression uses
`class_weight='balanced'` instead - the linear-model equivalent of the same
idea.

Both tree models carve out their own small internal validation split for
early stopping, from their own training data only - never from the fold or
holdout they'll be scored on, which would leak information into the score.
"""
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from home_credit import config
from home_credit.features import build_linear_preprocessor


class LogisticRegressionModel:
    name = "logistic_regression"

    def __init__(self, numeric_cols: list[str], categorical_cols: list[str], seed: int = config.RANDOM_SEED):
        preprocessor = build_linear_preprocessor(numeric_cols, categorical_cols)
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
        self.pipeline = Pipeline([("preprocess", preprocessor), ("classify", clf)])

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)[:, 1]

    @property
    def sklearn_estimator(self):
        """The underlying sklearn-compatible fitted estimator, for calibration/SHAP."""
        return self.pipeline


class LightGBMModel:
    name = "lightgbm"

    PARAMS = {
        "objective": "binary",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 40,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "n_jobs": -1,
    }

    def __init__(self, categorical_cols: list[str], seed: int = config.RANDOM_SEED):
        self.categorical_cols = categorical_cols
        self.seed = seed
        self.model = None

    def fit(self, X, y):
        X_fit, X_val, y_fit, y_val = train_test_split(
            X, y, test_size=config.INNER_VALID_SIZE, stratify=y, random_state=self.seed
        )
        scale_pos_weight = (y_fit == 0).sum() / (y_fit == 1).sum()
        self.model = lgb.LGBMClassifier(**self.PARAMS, scale_pos_weight=scale_pos_weight, random_state=self.seed)
        self.model.fit(
            X_fit,
            y_fit,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            categorical_feature=self.categorical_cols,
            callbacks=[lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
        )
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    @property
    def sklearn_estimator(self):
        return self.model


class XGBoostModel:
    name = "xgboost"

    PARAMS = {
        "objective": "binary:logistic",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "tree_method": "hist",
        "enable_categorical": True,
        "eval_metric": "auc",
        "n_jobs": -1,
    }

    def __init__(self, categorical_cols: list[str], seed: int = config.RANDOM_SEED):
        self.categorical_cols = categorical_cols
        self.seed = seed
        self.model = None

    def fit(self, X, y):
        X_fit, X_val, y_fit, y_val = train_test_split(
            X, y, test_size=config.INNER_VALID_SIZE, stratify=y, random_state=self.seed
        )
        scale_pos_weight = (y_fit == 0).sum() / (y_fit == 1).sum()
        self.model = xgb.XGBClassifier(
            **self.PARAMS,
            scale_pos_weight=scale_pos_weight,
            random_state=self.seed,
            early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
        )
        self.model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    @property
    def sklearn_estimator(self):
        return self.model


MODEL_REGISTRY = {
    LogisticRegressionModel.name: LogisticRegressionModel,
    LightGBMModel.name: LightGBMModel,
    XGBoostModel.name: XGBoostModel,
}

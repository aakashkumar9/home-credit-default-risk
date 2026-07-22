"""Shared feature preparation - the single place where "how do we treat
missingness and categoricals" is decided, so the baseline logistic
regression, LightGBM, XGBoost, and the EDA report all agree on what a
"feature" is and how it's typed.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from home_credit import config


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {config.ID_COL, config.TARGET_COL, "is_train"}
    return [c for c in df.columns if c not in exclude]


def split_column_types(df: pd.DataFrame, feature_cols: list[str]) -> tuple[list[str], list[str]]:
    """Splits feature_cols into (numeric_cols, categorical_cols) by pandas dtype.
    Boolean flag columns (has_bureau_history, etc.) count as categorical - a
    handful of distinct values, not a magnitude to treat numerically.
    """
    categorical_cols = [
        c
        for c in feature_cols
        if df[c].dtype == object
        or isinstance(df[c].dtype, pd.StringDtype)
        or pd.api.types.is_string_dtype(df[c].dtype)
        or str(df[c].dtype) == "bool"
    ]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    return numeric_cols, categorical_cols


def prepare_tree_dtypes(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    """For LightGBM/XGBoost: cast categorical columns to pandas 'category' dtype
    so they're split on natively. Missing values (numeric and categorical) are
    left as NaN / a missing category - both models split on missingness
    natively, and imputing here would just destroy that signal.
    """
    df = df.copy()
    for col in categorical_cols:
        df[col] = df[col].astype("category")
    return df


def build_linear_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """For the baseline logistic regression, which - unlike the tree models -
    can't handle missing values or raw strings directly: median-impute +
    standardize numeric columns, most-frequent-impute + one-hot categoricals.
    `handle_unknown='ignore'` so a category unseen during training (e.g. in
    application_test) doesn't crash scoring - it just contributes an
    all-zero one-hot row instead.
    """
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ])

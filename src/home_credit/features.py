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
    """Splits feature_cols into (numeric_cols, categorical_cols) by pandas
    dtype. Boolean flag columns (has_bureau_history, etc.) count as numeric,
    not categorical: as a 0/1 value they scale/split identically to any
    other numeric feature for every model here, and treating them as
    categorical caused two real problems - XGBoost's native categorical
    handling requires string-typed categories and crashes on boolean ones,
    and mixing multiple 'category'-dtype columns of different underlying
    value types (bool alongside string) hits a pandas/sklearn dtype-
    promotion bug in SimpleImputer. Neither applies to plain numeric bools.
    """
    categorical_cols = [
        c
        for c in feature_cols
        if df[c].dtype == object
        or isinstance(df[c].dtype, pd.StringDtype)
        or pd.api.types.is_string_dtype(df[c].dtype)
    ]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    return numeric_cols, categorical_cols


def prepare_tree_dtypes(
    df: pd.DataFrame, categorical_cols: list[str], categories: dict[str, list] | None = None
) -> pd.DataFrame:
    """For LightGBM/XGBoost: cast categorical columns to pandas 'category' dtype
    so they're split on natively. Missing values (numeric and categorical) are
    left as NaN / a missing category - both models split on missingness
    natively, and imputing here would just destroy that signal.

    Pass `categories` (from `extract_categories`, captured once against the
    full training set and persisted in feature_metadata.json) for any call
    on fewer rows than the model was trained on - a single applicant's row,
    for instance. Deriving categories fresh from a 1-row slice would only
    "see" whatever value is in that row (zero categories if it's null),
    which is inconsistent with the category encoding the model actually
    learned and can silently score wrong, not just crash - confirmed by
    hitting exactly that crash in single-row inference before this existed.
    """
    df = df.copy()
    for col in categorical_cols:
        if categories is not None:
            df[col] = pd.Categorical(df[col], categories=categories[col])
        else:
            df[col] = df[col].astype("category")
    return df


def extract_categories(df: pd.DataFrame, categorical_cols: list[str]) -> dict[str, list]:
    """Captures each categorical column's category levels from a (typically
    full-training-set) frame, to be persisted and reused via
    prepare_tree_dtypes' `categories` argument at inference time.
    """
    return {col: df[col].astype("category").cat.categories.tolist() for col in categorical_cols}


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

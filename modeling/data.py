"""Loads mart_applicant_features from DuckDB and prepares it for modeling.

This is the only place raw pandas dtypes get decided - everything downstream
(train/calibrate/explain/predict) imports get_feature_columns and
prepare_dtypes from here so the three stages never disagree about which
columns are features vs. which are categorical.
"""
import duckdb
import pandas as pd

from modeling import config


def load_mart(duckdb_path: str | None = None) -> pd.DataFrame:
    con = duckdb.connect(duckdb_path or config.DUCKDB_PATH, read_only=True)
    try:
        df = con.execute(f"select * from {config.MART_TABLE}").df()
    finally:
        con.close()
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {config.ID_COL, config.TARGET_COL, "is_train"}
    return [c for c in df.columns if c not in exclude]


def prepare_dtypes(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Casts object/bool feature columns to pandas 'category' dtype so LightGBM
    can split on them natively, without one-hot encoding or label encoding.
    """
    df = df.copy()
    for col in feature_cols:
        dtype = df[col].dtype
        is_stringy = dtype == object or isinstance(dtype, pd.StringDtype) or pd.api.types.is_string_dtype(dtype)
        if is_stringy or str(dtype) == "bool":
            df[col] = df[col].astype("category")
    return df


def get_categorical_columns(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    return [c for c in feature_cols if str(df[c].dtype) == "category"]


def split_train_score(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["is_train"]].reset_index(drop=True)
    score_df = df[~df["is_train"]].reset_index(drop=True)
    return train_df, score_df

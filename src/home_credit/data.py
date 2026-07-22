"""Loads mart_applicant_features from DuckDB - the single feature table every
downstream consumer (EDA, modeling, explainability) reads from.
"""

import duckdb
import pandas as pd

from home_credit import config


def load_mart(duckdb_path: str | None = None) -> pd.DataFrame:
    con = duckdb.connect(duckdb_path or config.DUCKDB_PATH, read_only=True)
    try:
        return con.execute(f"select * from {config.MART_TABLE}").df()
    finally:
        con.close()


def split_train_score(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["is_train"]].reset_index(drop=True)
    score_df = df[~df["is_train"]].reset_index(drop=True)
    return train_df, score_df

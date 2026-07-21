"""One stratified train/validation split, reused by train.py (early stopping)
and calibrate.py (prefit calibration on the same held-out fold) so the two
stages are never accidentally evaluated on overlapping data.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from modeling import config


def train_valid_split(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_df, valid_df = train_test_split(
        train_df,
        test_size=config.VALID_SIZE,
        stratify=train_df[config.TARGET_COL],
        random_state=config.RANDOM_SEED,
    )
    return fit_df.reset_index(drop=True), valid_df.reset_index(drop=True)

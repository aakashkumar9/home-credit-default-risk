import numpy as np
import pandas as pd
import pytest

from home_credit import features


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "sk_id_curr": [1, 2, 3, 4],
        "target": [0, 1, 0, 1],
        "is_train": [True, True, True, True],
        "amt_income": [50000.0, np.nan, 70000.0, 65000.0],
        "age_years": [30.0, 45.0, np.nan, 28.0],
        "gender": ["M", "F", "M", None],
        "has_bureau_history": [True, False, True, True],
    })


def test_get_feature_columns_excludes_id_target_is_train(sample_df):
    cols = features.get_feature_columns(sample_df)
    assert set(cols) == {"amt_income", "age_years", "gender", "has_bureau_history"}


def test_split_column_types_buckets_correctly(sample_df):
    feature_cols = features.get_feature_columns(sample_df)
    numeric_cols, categorical_cols = features.split_column_types(sample_df, feature_cols)
    # booleans count as numeric (0/1) - see split_column_types' docstring for why
    assert set(numeric_cols) == {"amt_income", "age_years", "has_bureau_history"}
    assert set(categorical_cols) == {"gender"}


def test_prepare_tree_dtypes_casts_categoricals_only(sample_df):
    feature_cols = features.get_feature_columns(sample_df)
    _, categorical_cols = features.split_column_types(sample_df, feature_cols)
    out = features.prepare_tree_dtypes(sample_df, categorical_cols)
    assert str(out["gender"].dtype) == "category"
    assert out["has_bureau_history"].dtype != "category"
    assert out["amt_income"].dtype != "category"


def test_linear_preprocessor_handles_missing_values_and_unseen_categories(sample_df):
    feature_cols = features.get_feature_columns(sample_df)
    numeric_cols, categorical_cols = features.split_column_types(sample_df, feature_cols)
    preprocessor = features.build_linear_preprocessor(numeric_cols, categorical_cols)

    X_train = sample_df[feature_cols]
    transformed = preprocessor.fit_transform(X_train)
    assert not np.isnan(transformed).any()

    # a category never seen in training shouldn't crash transform
    X_new = pd.DataFrame({
        "amt_income": [80000.0],
        "age_years": [40.0],
        "gender": ["XNA"],
        "has_bureau_history": [False],
    })
    transformed_new = preprocessor.transform(X_new)
    assert not np.isnan(transformed_new).any()

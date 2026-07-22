import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from home_credit import data
from home_credit.validation.schemas import validate_mart


@pytest.fixture
def valid_row():
    """A single minimal-but-schema-valid mart row - every column the
    schema actually checks, with values that should pass.
    """
    return {
        "sk_id_curr": 1,
        "is_train": True,
        "target": 0.0,
        "bureau_count": 2,
        "bureau_active_count": 1,
        "bureau_closed_count": 1,
        "bureau_cnt_credit_prolong_sum": 0,
        "prev_app_count": 0,
        "prev_app_approved_count": 0,
        "prev_app_refused_count": 0,
        "prev_app_canceled_count": 0,
        "prev_app_new_client_count": 0,
        "pos_records_count": 0,
        "pos_distinct_prev_count": 0,
        "pos_completed_count": 0,
        "pos_active_count": 0,
        "cc_records_count": 0,
        "cc_distinct_prev_count": 0,
        "cc_underpaid_months_count": 0,
        "inst_payments_count": 0,
        "inst_distinct_prev_count": 0,
        "has_bureau_history": True,
        "has_previous_application_history": False,
        # a column the schema doesn't constrain at all - must pass through untouched (strict=False)
        "bureau_credit_types_count": None,
    }


def test_validate_mart_accepts_a_well_formed_frame(valid_row):
    df = pd.DataFrame([valid_row, {**valid_row, "sk_id_curr": 2, "target": np.nan}])
    validate_mart(df)  # should not raise


def test_validate_mart_rejects_duplicate_sk_id_curr(valid_row):
    df = pd.DataFrame([valid_row, valid_row])
    with pytest.raises(SchemaErrors):
        validate_mart(df)


def test_validate_mart_rejects_negative_count(valid_row):
    bad_row = {**valid_row, "bureau_count": -1}
    df = pd.DataFrame([bad_row])
    with pytest.raises(SchemaErrors):
        validate_mart(df)


def test_validate_mart_rejects_invalid_target_value(valid_row):
    bad_row = {**valid_row, "target": 2.0}
    df = pd.DataFrame([bad_row])
    with pytest.raises(SchemaErrors):
        validate_mart(df)


def test_validate_mart_rejects_null_has_history_flag(valid_row):
    bad_row = {**valid_row, "has_bureau_history": None}
    df = pd.DataFrame([bad_row]).astype({"has_bureau_history": "object"})
    with pytest.raises(SchemaErrors):
        validate_mart(df)


def test_validate_mart_against_real_mart(mart_db_path):
    df = data.load_mart(mart_db_path)
    validate_mart(df)  # should not raise - the real dbt-built mart must satisfy its own contract

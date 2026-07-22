"""Expected shape of each raw CSV - just enough structure for load_raw_data
to fail loudly on a wrong or partial download, without duplicating the
deeper dtype/value validation `home_credit.validation` (phase 8) does
against the built mart.
"""

# table name -> CSV filename
RAW_TABLES = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "pos_cash_balance": "POS_CASH_balance.csv",
    "credit_card_balance": "credit_card_balance.csv",
    "installments_payments": "installments_payments.csv",
}

# table name -> columns that must exist and be fully non-null (grain / foreign keys)
REQUIRED_NOT_NULL_COLUMNS = {
    "application_train": ["SK_ID_CURR", "TARGET"],
    "application_test": ["SK_ID_CURR"],
    "bureau": ["SK_ID_BUREAU", "SK_ID_CURR"],
    "bureau_balance": ["SK_ID_BUREAU", "MONTHS_BALANCE"],
    "previous_application": ["SK_ID_PREV", "SK_ID_CURR"],
    "pos_cash_balance": ["SK_ID_PREV", "SK_ID_CURR"],
    "credit_card_balance": ["SK_ID_PREV", "SK_ID_CURR"],
    "installments_payments": ["SK_ID_PREV", "SK_ID_CURR"],
}

# table name -> columns that together must be unique (the table's grain)
UNIQUE_KEY_COLUMNS = {
    "application_train": ["SK_ID_CURR"],
    "application_test": ["SK_ID_CURR"],
    "bureau": ["SK_ID_BUREAU"],
}

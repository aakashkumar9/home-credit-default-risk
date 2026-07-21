#!/usr/bin/env python3
"""Loads the Home Credit Default Risk CSVs from data/raw into DuckDB's `raw`
schema, untouched (1:1 copy - dbt staging models do all renaming/typing).

Expects these files in data/raw/:
    application_train.csv
    application_test.csv
    bureau.csv
    bureau_balance.csv
    previous_application.csv
    POS_CASH_balance.csv
    credit_card_balance.csv
    installments_payments.csv
"""
import os
import sys
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = Path(os.environ.get("RAW_DIR", REPO_ROOT / "data" / "raw"))
DUCKDB_PATH = os.environ.get("DUCKDB_PATH", str(REPO_ROOT / "data" / "home_credit.duckdb"))

# raw table name -> CSV filename
TABLES = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "pos_cash_balance": "POS_CASH_balance.csv",
    "credit_card_balance": "credit_card_balance.csv",
    "installments_payments": "installments_payments.csv",
}


def main():
    missing = [fname for fname in TABLES.values() if not (RAW_DIR / fname).exists()]
    if missing:
        print(f"Missing files in {RAW_DIR}:", file=sys.stderr)
        for fname in missing:
            print(f"  - {fname}", file=sys.stderr)
        print(
            "\nDownload the Home Credit Default Risk dataset from Kaggle "
            "(https://www.kaggle.com/competitions/home-credit-default-risk/data) "
            f"and place the CSVs in {RAW_DIR}",
            file=sys.stderr,
        )
        sys.exit(1)

    con = duckdb.connect(DUCKDB_PATH)
    con.execute("create schema if not exists raw")

    for table, fname in TABLES.items():
        csv_path = RAW_DIR / fname
        print(f"Loading {fname} -> raw.{table}")
        con.execute(
            f"""
            create or replace table raw.{table} as
            select * from read_csv_auto(?, header=true, sample_size=-1)
            """,
            [str(csv_path)],
        )
        n_rows = con.execute(f"select count(*) from raw.{table}").fetchone()[0]
        print(f"  {n_rows:,} rows")

    con.close()
    print(f"\nDone. DuckDB database at {DUCKDB_PATH}")


if __name__ == "__main__":
    main()

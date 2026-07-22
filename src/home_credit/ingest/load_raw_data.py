"""Loads the raw Home Credit CSVs into DuckDB's `raw` schema, untouched
(1:1 copy - dbt staging models do all renaming/typing), with schema and
row-count checks so a wrong or partial download fails loudly here rather
than surfacing as a confusing dbt error three layers downstream.

    python -m home_credit.ingest.load_raw_data
"""

import logging
import sys
from pathlib import Path

import duckdb

from home_credit import config
from home_credit.ingest.schemas import RAW_TABLES, REQUIRED_NOT_NULL_COLUMNS, UNIQUE_KEY_COLUMNS

logger = logging.getLogger(__name__)


class IngestionError(RuntimeError):
    """A raw CSV is missing, empty, or fails a schema/row-count check."""


def find_missing_files() -> list[str]:
    return [fname for fname in RAW_TABLES.values() if not (config.DATA_RAW_DIR / fname).exists()]


def load_table(con: duckdb.DuckDBPyConnection, table: str, fname: str) -> int:
    """Loads one CSV into raw.<table>, then checks it. Returns the row count."""
    csv_path = config.DATA_RAW_DIR / fname
    con.execute(
        f"create or replace table raw.{table} as "
        "select * from read_csv_auto(?, header=true, sample_size=-1)",
        [str(csv_path)],
    )

    columns = {row[0] for row in con.execute(f"describe raw.{table}").fetchall()}
    required = REQUIRED_NOT_NULL_COLUMNS.get(table, [])
    missing_cols = [c for c in required if c not in columns]
    if missing_cols:
        raise IngestionError(f"raw.{table} is missing expected column(s): {missing_cols}")

    n_rows = con.execute(f"select count(*) from raw.{table}").fetchone()[0]
    if n_rows == 0:
        raise IngestionError(
            f"raw.{table} loaded 0 rows from {fname} - is the file empty or truncated?"
        )

    for col in required:
        n_null = con.execute(f"select count(*) from raw.{table} where {col} is null").fetchone()[0]
        if n_null > 0:
            raise IngestionError(f"raw.{table}.{col} has {n_null} null value(s), expected none")

    if table in UNIQUE_KEY_COLUMNS:
        key_list = ", ".join(UNIQUE_KEY_COLUMNS[table])
        n_dupes = con.execute(
            f"select count(*) from "
            f"(select {key_list} from raw.{table} group by {key_list} having count(*) > 1)"
        ).fetchone()[0]
        if n_dupes > 0:
            raise IngestionError(
                f"raw.{table} has {n_dupes} duplicate value(s) on key ({key_list}), expected unique"
            )

    return n_rows


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    missing = find_missing_files()
    if missing:
        logger.error("Missing files in %s:", config.DATA_RAW_DIR)
        for fname in missing:
            logger.error("  - %s", fname)
        logger.error(
            "\nDownload the Home Credit Default Risk dataset from Kaggle "
            "(https://www.kaggle.com/competitions/home-credit-default-risk/data) "
            "and place the CSVs in %s",
            config.DATA_RAW_DIR,
        )
        sys.exit(1)

    Path(config.DUCKDB_PATH).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(config.DUCKDB_PATH)
    con.execute("create schema if not exists raw")

    try:
        for table, fname in RAW_TABLES.items():
            logger.info("Loading %s -> raw.%s", fname, table)
            n_rows = load_table(con, table, fname)
            logger.info("  %s rows, schema checks passed", f"{n_rows:,}")
    finally:
        con.close()

    logger.info("\nDone. DuckDB database at %s", config.DUCKDB_PATH)


if __name__ == "__main__":
    main()

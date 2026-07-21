import duckdb
import pandas as pd
import pytest

from home_credit import config
from home_credit.ingest import load_raw_data
from home_credit.ingest.schemas import RAW_TABLES


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    d = tmp_path / "raw"
    d.mkdir()
    monkeypatch.setattr(config, "DATA_RAW_DIR", d)
    return d


@pytest.fixture
def duckdb_path(tmp_path, monkeypatch):
    path = str(tmp_path / "test.duckdb")
    monkeypatch.setattr(config, "DUCKDB_PATH", path)
    return path


def _write_csv(path, df):
    df.to_csv(path, index=False)


def test_find_missing_files_reports_absent_csvs(raw_dir):
    _write_csv(raw_dir / "application_train.csv", pd.DataFrame({"SK_ID_CURR": [1], "TARGET": [0]}))
    missing = load_raw_data.find_missing_files()
    assert "application_train.csv" in {f for f in RAW_TABLES.values()} - set(missing)
    assert "bureau.csv" in missing
    assert "application_train.csv" not in missing


def test_full_ingest_succeeds_against_synthetic_fixtures(raw_dir, duckdb_path):
    from tests.generate_synthetic_data import generate

    generate(raw_dir, n_applicants=50, seed=1)
    assert not load_raw_data.find_missing_files()

    con = duckdb.connect(duckdb_path)
    con.execute("create schema if not exists raw")
    try:
        for table, fname in RAW_TABLES.items():
            n_rows = load_raw_data.load_table(con, table, fname)
            assert n_rows > 0
    finally:
        con.close()


def test_load_table_raises_on_missing_required_column(raw_dir, duckdb_path):
    _write_csv(raw_dir / "application_train.csv", pd.DataFrame({"TARGET": [0, 1]}))  # no SK_ID_CURR
    con = duckdb.connect(duckdb_path)
    con.execute("create schema if not exists raw")
    try:
        with pytest.raises(load_raw_data.IngestionError, match="missing expected column"):
            load_raw_data.load_table(con, "application_train", "application_train.csv")
    finally:
        con.close()


def test_load_table_raises_on_empty_file(raw_dir, duckdb_path):
    _write_csv(raw_dir / "application_train.csv", pd.DataFrame({"SK_ID_CURR": [], "TARGET": []}))
    con = duckdb.connect(duckdb_path)
    con.execute("create schema if not exists raw")
    try:
        with pytest.raises(load_raw_data.IngestionError, match="0 rows"):
            load_raw_data.load_table(con, "application_train", "application_train.csv")
    finally:
        con.close()


def test_load_table_raises_on_null_required_column(raw_dir, duckdb_path):
    _write_csv(
        raw_dir / "application_train.csv",
        pd.DataFrame({"SK_ID_CURR": [1, 2], "TARGET": [0, None]}),
    )
    con = duckdb.connect(duckdb_path)
    con.execute("create schema if not exists raw")
    try:
        with pytest.raises(load_raw_data.IngestionError, match="null value"):
            load_raw_data.load_table(con, "application_train", "application_train.csv")
    finally:
        con.close()


def test_load_table_raises_on_duplicate_key(raw_dir, duckdb_path):
    _write_csv(
        raw_dir / "application_train.csv",
        pd.DataFrame({"SK_ID_CURR": [1, 1], "TARGET": [0, 1]}),
    )
    con = duckdb.connect(duckdb_path)
    con.execute("create schema if not exists raw")
    try:
        with pytest.raises(load_raw_data.IngestionError, match="duplicate"):
            load_raw_data.load_table(con, "application_train", "application_train.csv")
    finally:
        con.close()

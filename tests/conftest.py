import duckdb
import pytest

from home_credit import config


@pytest.fixture
def mart_db_path():
    """Points at a DuckDB file with marts.mart_applicant_features already
    built (via scripts/build_warehouse.sh, pointed at either the real data
    or tests/generate_synthetic_data.py's fixtures). Skips rather than
    fails if nothing has built the warehouse yet - this suite is an
    integration test on top of the dbt build, not a substitute for it.
    """
    path = config.DUCKDB_PATH
    try:
        con = duckdb.connect(path, read_only=True)
        con.execute(f"select 1 from {config.MART_TABLE} limit 1")
        con.close()
    except duckdb.Error:
        pytest.skip(
            f"{config.MART_TABLE} not found at {path} - run scripts/build_warehouse.sh "
            "(against real or synthetic data) before running this suite"
        )
    return path

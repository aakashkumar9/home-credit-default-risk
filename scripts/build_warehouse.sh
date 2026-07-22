#!/usr/bin/env bash
# Full rebuild: load raw CSVs into DuckDB, run dbt build (models + tests),
# generate docs, then validate the resulting mart against its pandera
# contract (schemas.py) - catches a broken model or raw-data drift here,
# not three modules downstream in modeling.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DBT_PROFILES_DIR="$REPO_ROOT/warehouse"

python3 -m home_credit.ingest.load_raw_data

cd "$REPO_ROOT/warehouse"
dbt build
dbt docs generate

cd "$REPO_ROOT"
python3 -m home_credit.validation.validate_mart

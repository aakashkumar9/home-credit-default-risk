#!/usr/bin/env bash
# Full rebuild: load raw CSVs into DuckDB, run dbt build (models + tests), generate docs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DBT_PROFILES_DIR="$REPO_ROOT/warehouse"

python3 -m home_credit.ingest.load_raw_data

cd "$REPO_ROOT/warehouse"
dbt build
dbt docs generate

#!/usr/bin/env bash
# Full rebuild: load raw CSVs into DuckDB, run dbt build (models + tests), generate docs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DBT_PROFILES_DIR="$REPO_ROOT/warehouse"

python3 "$REPO_ROOT/scripts/load_raw_data.py"

cd "$REPO_ROOT/warehouse"
dbt build
dbt docs generate

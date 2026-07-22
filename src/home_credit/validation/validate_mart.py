"""Validates the built mart against MART_SCHEMA - run after `dbt build`
(see scripts/build_warehouse.sh) and again at the start of every training
run (see modeling/train.py), so a schema violation is caught right at the
warehouse/Python boundary rather than surfacing as a confusing failure
partway through modeling.

    python -m home_credit.validation.validate_mart
"""

import sys

from pandera.errors import SchemaErrors

from home_credit import config, data
from home_credit.validation.schemas import validate_mart


def main() -> int:
    df = data.load_mart()
    try:
        validate_mart(df)
    except SchemaErrors as exc:
        print(
            f"Mart validation FAILED ({len(exc.failure_cases)} failure case(s)):", file=sys.stderr
        )
        print(exc.failure_cases.to_string(), file=sys.stderr)
        return 1
    print(f"Mart validation passed: {len(df):,} rows against {config.MART_TABLE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

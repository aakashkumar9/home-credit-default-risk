"""Generates a short EDA summary of the strongest default drivers:
missingness, numeric-feature correlation with TARGET, and categorical-
feature target-rate spread. Writes docs/eda_summary.md.

    python -m home_credit.eda

The committed docs/eda_summary.md was generated against the synthetic
fixtures (see tests/generate_synthetic_data.py) - the numbers in it are
noise, not real default drivers. Re-run this against the real mart
(after `make dbt-build` with the actual Kaggle data) to get a meaningful
report; the point of committing a copy is to document the report's shape,
not its numbers.
"""
import warnings

import pandas as pd

from home_credit import config, data
from home_credit.features import get_feature_columns, split_column_types

DOCS_DIR = config.REPO_ROOT / "docs"


def missingness_table(train_df: pd.DataFrame, feature_cols: list[str], top_n: int = 15) -> pd.Series:
    return train_df[feature_cols].isna().mean().sort_values(ascending=False).head(top_n)


def numeric_target_correlation(train_df: pd.DataFrame, numeric_cols: list[str], top_n: int = 15) -> pd.Series:
    # zero-variance columns (e.g. a constant flag in a small fixture) produce a
    # harmless 0/0 -> NaN correlation; dropna() already handles it correctly,
    # this just silences numpy's runtime warning about the division itself.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        corr = train_df[numeric_cols].corrwith(train_df[config.TARGET_COL].astype(float))
    corr = corr.dropna()
    return corr.reindex(corr.abs().sort_values(ascending=False).index).head(top_n)


def categorical_target_spread(
    train_df: pd.DataFrame, categorical_cols: list[str], top_n: int = 10, min_count: int = 20
) -> pd.DataFrame:
    rows = []
    for col in categorical_cols:
        rates = train_df.groupby(col)[config.TARGET_COL].agg(["mean", "count"])
        rates = rates[rates["count"] >= min_count]
        if rates.empty:
            continue
        rows.append({
            "column": col,
            "spread": rates["mean"].max() - rates["mean"].min(),
            "n_categories": len(rates),
        })
    if not rows:
        return pd.DataFrame(columns=["column", "spread", "n_categories"])
    return pd.DataFrame(rows).sort_values("spread", ascending=False).head(top_n)


def generate_report(duckdb_path: str | None = None) -> str:
    df = data.load_mart(duckdb_path)
    train_df, _ = data.split_train_score(df)
    feature_cols = get_feature_columns(df)
    numeric_cols, categorical_cols = split_column_types(df, feature_cols)

    missing = missingness_table(train_df, feature_cols)
    corr = numeric_target_correlation(train_df, numeric_cols)
    cat_spread = categorical_target_spread(train_df, categorical_cols)

    lines = [
        "# EDA summary",
        "",
        f"- Training rows: {len(train_df):,}",
        f"- Target (default) rate: {train_df[config.TARGET_COL].mean():.2%}",
        f"- Features: {len(feature_cols)} ({len(numeric_cols)} numeric, {len(categorical_cols)} categorical)",
        "",
        "## Missingness (top 15)",
        "",
        "No history in a source shows up as an explicit `has_*_history` flag "
        "rather than nulls in the count columns (see the dbt mart) - the nulls "
        "below are genuine missing values within existing records.",
        "",
        "| feature | missing rate |",
        "|---|---|",
    ]
    lines += [f"| `{col}` | {rate:.1%} |" for col, rate in missing.items()]

    lines += [
        "",
        "## Numeric features most correlated with TARGET (top 15 by absolute correlation)",
        "",
        "| feature | correlation |",
        "|---|---|",
    ]
    lines += [f"| `{col}` | {r:+.3f} |" for col, r in corr.items()]

    lines += [
        "",
        "## Categorical features with the widest target-rate spread (top 10)",
        "",
        "Spread = (highest category default rate) - (lowest), restricted to "
        "categories with at least 20 applicants so a single-applicant outlier "
        "category can't dominate.",
        "",
        "| feature | rate spread | categories |",
        "|---|---|---|",
    ]
    lines += [
        f"| `{row.column}` | {row.spread:.1%} | {int(row.n_categories)} |" for row in cat_spread.itertuples()
    ]

    return "\n".join(lines) + "\n"


def main():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    report = generate_report()
    out_path = DOCS_DIR / "eda_summary.md"
    out_path.write_text(report)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

import pandas as pd

from home_credit import eda


def test_missingness_table_reports_correct_rates():
    df = pd.DataFrame({"a": [1.0, None, None, 4.0], "b": [1.0, 2.0, 3.0, 4.0]})
    result = eda.missingness_table(df, ["a", "b"])
    assert result["a"] == 0.5
    assert result["b"] == 0.0


def test_numeric_target_correlation_ranks_by_absolute_value():
    df = pd.DataFrame(
        {
            "target": [0, 0, 1, 1, 0, 1, 0, 1],
            "strong_pos": [1, 2, 8, 9, 1, 9, 2, 8],
            "strong_neg": [9, 8, 2, 1, 9, 1, 8, 2],
            "weak": [5, 3, 6, 2, 7, 4, 5, 3],
        }
    )
    result = eda.numeric_target_correlation(df, ["strong_pos", "strong_neg", "weak"])
    assert list(result.index[:2]) == ["strong_pos", "strong_neg"] or list(result.index[:2]) == [
        "strong_neg",
        "strong_pos",
    ]
    assert result["strong_pos"] > 0
    assert result["strong_neg"] < 0


def test_categorical_target_spread_respects_min_count():
    df = pd.DataFrame(
        {
            "target": [0] * 25 + [1] * 25,
            "wide_spread": ["A"] * 25 + ["B"] * 25,  # A always 0, B always 1 -> spread 1.0
            "rare_category": ["X"] * 49 + ["Y"] * 1,  # Y has only 1 row, below min_count
        }
    )
    result = eda.categorical_target_spread(df, ["wide_spread", "rare_category"], min_count=20)
    assert "wide_spread" in result["column"].values
    row = result[result["column"] == "wide_spread"].iloc[0]
    assert row["spread"] == 1.0


def test_generate_report_against_real_mart(mart_db_path):
    report = eda.generate_report(mart_db_path)
    assert "# EDA summary" in report
    assert "## Missingness" in report
    assert "## Numeric features" in report
    assert "## Categorical features" in report
    assert "Training rows:" in report

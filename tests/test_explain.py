import pytest

from home_credit import config, data
from home_credit.explain.shap_explainer import build_explainer, compute_global_importance, explain_applicant, main
from home_credit.modeling import train as train_mod


def test_build_explainer_rejects_non_tree_champion():
    with pytest.raises(NotImplementedError, match="logistic_regression"):
        build_explainer("logistic_regression", champion=None)


@pytest.fixture
def trained_tree_champion(mart_db_path, tmp_path, monkeypatch):
    """Trains for real against the mart, forcing a tree champion (the only
    type SHAP is scoped to) regardless of what CV happens to prefer, so this
    test doesn't depend on synthetic-noise CV results picking a particular
    winner.
    """
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(config, "N_CV_SPLITS", 3)
    monkeypatch.setattr(config, "DUCKDB_PATH", mart_db_path)
    monkeypatch.setattr(train_mod, "MODEL_NAMES", ["lightgbm"])

    train_mod.train(mart_db_path)
    return mart_db_path


def test_global_importance_and_plot(trained_tree_champion):
    mean_abs_shap, X_sample, shap_matrix = compute_global_importance(trained_tree_champion, sample_size=50)

    assert len(mean_abs_shap) == X_sample.shape[1]
    assert shap_matrix.shape == (len(X_sample), X_sample.shape[1])
    assert (mean_abs_shap >= 0).all()
    assert mean_abs_shap.is_monotonic_decreasing


def test_explain_applicant_contributions_sum_to_predicted_score(trained_tree_champion):
    df = data.load_mart(trained_tree_champion)
    sk_id = int(df[config.ID_COL].iloc[0])

    result = explain_applicant(sk_id, trained_tree_champion)

    assert result["sk_id_curr"] == sk_id
    assert len(result["contributions"]) == len(df.columns) - 3  # minus id/target/is_train
    contribution_sum = sum(c["shap_value"] for c in result["contributions"])
    assert result["predicted_score"] == pytest.approx(result["base_value"] + contribution_sum)
    # sorted by |shap_value| descending
    abs_values = [abs(c["shap_value"]) for c in result["contributions"]]
    assert abs_values == sorted(abs_values, reverse=True)


def test_explain_applicant_raises_for_unknown_id(trained_tree_champion):
    with pytest.raises(ValueError, match="not found"):
        explain_applicant(999999999, trained_tree_champion)


def test_main_writes_report_files(trained_tree_champion):
    main()
    assert (config.REPORTS_DIR / "shap_feature_importance.csv").exists()
    assert (config.REPORTS_DIR / "shap_summary.png").exists()

import pytest
from fastapi.testclient import TestClient

from home_credit import config, data
from home_credit.modeling import train as train_mod


@pytest.fixture
def api_client(mart_db_path, tmp_path, monkeypatch):
    """Trains a real (forced-tree-champion) model against the mart, points
    the app's config at it, and yields a TestClient - so every request in
    these tests exercises the real lifespan startup (model/explainer
    loading) and the real DuckDB single-row lookup path, not a mock.
    """
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(config, "N_CV_SPLITS", 3)
    monkeypatch.setattr(config, "DUCKDB_PATH", mart_db_path)
    monkeypatch.setattr(train_mod, "MODEL_NAMES", ["lightgbm"])
    train_mod.train(mart_db_path)

    # import after config is patched and the model exists, and fresh each
    # time so the module-level `state` dict starts clean per test
    import importlib

    from home_credit.serving import api as api_module

    importlib.reload(api_module)

    with TestClient(api_module.app) as client:
        yield client


def test_health(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    assert r.json()["champion_model"] == "lightgbm"


def test_predict_known_applicant(api_client, mart_db_path):
    df = data.load_mart(mart_db_path)
    sk_id = int(df[config.ID_COL].iloc[0])

    r = api_client.get(f"/predict/{sk_id}")

    assert r.status_code == 200
    body = r.json()
    assert body["sk_id_curr"] == sk_id
    assert 0.0 <= body["probability"] <= 1.0
    assert isinstance(body["is_train"], bool)


def test_predict_unknown_applicant_is_404(api_client):
    r = api_client.get("/predict/999999999")
    assert r.status_code == 404


def test_explain_known_applicant_contributions_sum_to_predicted_score(api_client, mart_db_path):
    df = data.load_mart(mart_db_path)
    sk_id = int(df[config.ID_COL].iloc[0])

    r = api_client.get(f"/explain/{sk_id}")

    assert r.status_code == 200
    body = r.json()
    assert body["sk_id_curr"] == sk_id
    contribution_sum = sum(c["shap_value"] for c in body["contributions"])
    assert body["predicted_score"] == pytest.approx(body["base_value"] + contribution_sum)
    abs_values = [abs(c["shap_value"]) for c in body["contributions"]]
    assert abs_values == sorted(abs_values, reverse=True)


def test_explain_unknown_applicant_is_404(api_client):
    r = api_client.get("/explain/999999999")
    assert r.status_code == 404

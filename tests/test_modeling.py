from modeling import calibrate, config, data, evaluate, predict, train
from modeling.split import train_valid_split


def test_mart_grain_is_one_row_per_applicant(mart_db_path):
    df = data.load_mart(mart_db_path)
    assert df["sk_id_curr"].is_unique
    assert df["sk_id_curr"].notna().all()


def test_train_test_split_is_disjoint_and_target_consistent(mart_db_path):
    df = data.load_mart(mart_db_path)
    train_df, score_df = data.split_train_score(df)
    assert set(train_df["sk_id_curr"]).isdisjoint(set(score_df["sk_id_curr"]))
    assert train_df["target"].notna().all()
    assert score_df["target"].isna().all()


def test_train_valid_split_is_stratified_and_disjoint(mart_db_path):
    df = data.load_mart(mart_db_path)
    train_df, _ = data.split_train_score(df)
    fit_df, valid_df = train_valid_split(train_df)
    assert set(fit_df["sk_id_curr"]).isdisjoint(set(valid_df["sk_id_curr"]))
    # stratification keeps the positive rate roughly comparable across folds
    assert abs(fit_df["target"].mean() - valid_df["target"].mean()) < 0.1


def test_full_pipeline_runs_and_produces_valid_metrics(mart_db_path, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")

    train_metrics = train.train(mart_db_path)
    assert 0.0 <= train_metrics["valid_roc_auc"] <= 1.0
    assert train_metrics["scale_pos_weight"] > 1  # positives are the minority class

    calibration_metrics = calibrate.calibrate(mart_db_path)
    assert calibration_metrics["brier_score_calibrated"] >= 0

    eval_metrics = evaluate.evaluate(mart_db_path)
    assert 0.0 <= eval_metrics["roc_auc"] <= 1.0
    assert 0.0 <= eval_metrics["pr_auc"] <= 1.0

    submission = predict.predict(mart_db_path)
    assert list(submission.columns) == ["SK_ID_CURR", "TARGET"]
    assert submission["TARGET"].between(0, 1).all()
    assert submission["SK_ID_CURR"].is_unique

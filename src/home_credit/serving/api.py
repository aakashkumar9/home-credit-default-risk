"""FastAPI scoring service for applicants already present in the mart
(scores by SK_ID_CURR, not raw application fields - see the README's
Serving section for why: the mart is the feature store, and scoring an
existing ID against it mirrors how a real internal risk API would sit on
top of a precomputed feature store rather than recomputing ~130 features
from scratch per request).

    make serve-api
    curl http://localhost:8000/predict/100001
    curl http://localhost:8000/explain/100001

Model, feature metadata, and the SHAP explainer are all loaded once at
startup (FastAPI lifespan), not per-request - and /predict, /explain query
DuckDB for a single row rather than loading the whole mart into pandas,
since that wouldn't scale to real data size (the mart is ~307k rows on the
real dataset) if it happened on every request.
"""

import json
from contextlib import asynccontextmanager
from typing import Any

import duckdb
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from home_credit import config
from home_credit.explain.shap_explainer import (
    TREE_MODEL_NAMES,
    build_explainer,
    contributions_for_row,
    positive_class_base_value,
    positive_class_values,
)
from home_credit.features import prepare_tree_dtypes

state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(config.MODEL_DIR / "feature_metadata.json") as f:
        meta = json.load(f)
    state["meta"] = meta
    state["calibrated_model"] = joblib.load(config.MODEL_DIR / "calibrated_model.joblib")
    state["con"] = duckdb.connect(config.DUCKDB_PATH, read_only=True)

    state["explainer"] = None
    if meta["champion_model"] in TREE_MODEL_NAMES:
        champion = joblib.load(config.MODEL_DIR / "champion_model.joblib")
        state["explainer"] = build_explainer(meta["champion_model"], champion)

    yield

    state["con"].close()
    state.clear()


app = FastAPI(title="Home Credit Default Risk API", lifespan=lifespan)


class PredictionResponse(BaseModel):
    sk_id_curr: int
    is_train: bool
    actual_target: int | None = None
    probability: float


class Contribution(BaseModel):
    feature: str
    value: Any
    shap_value: float


class ExplanationResponse(BaseModel):
    sk_id_curr: int
    base_value: float
    predicted_score: float
    contributions: list[Contribution]


def _jsonable(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):  # numpy scalar (int64, float64, bool_, ...)
        return value.item()
    return value


def _fetch_applicant_row(sk_id_curr: int) -> pd.DataFrame:
    row = (
        state["con"]
        .execute(f"select * from {config.MART_TABLE} where {config.ID_COL} = ?", [sk_id_curr])
        .df()
    )
    if row.empty:
        raise HTTPException(status_code=404, detail=f"sk_id_curr {sk_id_curr} not found")
    return row


@app.get("/health")
def health():
    return {"status": "ok", "champion_model": state["meta"]["champion_model"]}


@app.get("/predict/{sk_id_curr}", response_model=PredictionResponse)
def predict(sk_id_curr: int):
    meta = state["meta"]
    feature_cols, categorical_cols = meta["feature_cols"], meta["categorical_cols"]

    row = _fetch_applicant_row(sk_id_curr)
    X = row
    if meta["champion_model"] in TREE_MODEL_NAMES:
        X = prepare_tree_dtypes(row, categorical_cols, categories=meta["categorical_categories"])

    probability = float(state["calibrated_model"].predict_proba(X[feature_cols])[:, 1][0])
    target = row[config.TARGET_COL].iloc[0]

    return PredictionResponse(
        sk_id_curr=sk_id_curr,
        is_train=bool(row["is_train"].iloc[0]),
        actual_target=(int(target) if pd.notna(target) else None),
        probability=probability,
    )


@app.get("/explain/{sk_id_curr}", response_model=ExplanationResponse)
def explain(sk_id_curr: int):
    meta = state["meta"]
    if state["explainer"] is None:
        raise HTTPException(
            status_code=501,
            detail=f"SHAP explanation isn't available for champion_model={meta['champion_model']!r}",
        )

    feature_cols, categorical_cols = meta["feature_cols"], meta["categorical_cols"]
    row = _fetch_applicant_row(sk_id_curr)
    X = prepare_tree_dtypes(row, categorical_cols, categories=meta["categorical_categories"])[
        feature_cols
    ]

    explanation = state["explainer"](X)
    shap_row = positive_class_values(explanation.values)[0]
    base_value = float(positive_class_base_value(explanation.base_values)[0])

    raw_values = [_jsonable(v) for v in row[feature_cols].iloc[0].tolist()]
    contributions = contributions_for_row(feature_cols, raw_values, shap_row)

    return ExplanationResponse(
        sk_id_curr=sk_id_curr,
        base_value=base_value,
        predicted_score=base_value + float(shap_row.sum()),
        contributions=contributions,
    )

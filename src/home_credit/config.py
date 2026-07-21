"""Central configuration for the whole project - paths, DB/table names, and
run constants. Every other module (ingest, modeling, serving, dashboard)
imports from here rather than hardcoding paths, so moving the warehouse or
switching environments is a one-file change.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# --- data & warehouse ---
DATA_RAW_DIR = Path(os.environ.get("DATA_RAW_DIR", REPO_ROOT / "data" / "raw"))
DUCKDB_PATH = os.environ.get("DUCKDB_PATH", str(REPO_ROOT / "data" / "home_credit.duckdb"))
MART_TABLE = "marts.mart_applicant_features"

# --- modeling artifacts ---
MODEL_DIR = Path(os.environ.get("MODEL_DIR", REPO_ROOT / "models"))
REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", REPO_ROOT / "reports"))

ID_COL = "sk_id_curr"
TARGET_COL = "target"
RANDOM_SEED = 42

# --- experiment tracking ---
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"file:{REPO_ROOT / 'mlruns'}")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "home-credit-default-risk")

# --- serving ---
SERVING_MODEL_NAME = os.environ.get("SERVING_MODEL_NAME", "home_credit_champion")

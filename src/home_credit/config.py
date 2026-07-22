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

# --- modelling ---
N_CV_SPLITS = 5  # stratified k-fold CV used to compare model types
CALIBRATION_HOLDOUT_SIZE = 0.15  # held out from the full training set, never used in
                                  # training the champion - used for calibration AND final evaluation
INNER_VALID_SIZE = 0.1  # carved out of each tree model's own training data for early stopping,
                         # so early stopping never peeks at the fold/holdout it's being scored on
EARLY_STOPPING_ROUNDS = 50

# --- experiment tracking ---
# MLflow >=3 deprecated the plain filesystem backend ("file:./mlruns") in favor
# of a database-backed store even for fully local use - sqlite needs no server.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "home-credit-default-risk")

# --- serving ---
SERVING_MODEL_NAME = os.environ.get("SERVING_MODEL_NAME", "home_credit_champion")

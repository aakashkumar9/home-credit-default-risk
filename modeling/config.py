import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DUCKDB_PATH = os.environ.get("DUCKDB_PATH", str(REPO_ROOT / "data" / "home_credit.duckdb"))
MART_TABLE = "marts.mart_applicant_features"

MODEL_DIR = Path(os.environ.get("MODEL_DIR", REPO_ROOT / "models"))
REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", REPO_ROOT / "reports"))

ID_COL = "sk_id_curr"
TARGET_COL = "target"

RANDOM_SEED = 42
VALID_SIZE = 0.2  # held out from application_train, used for early stopping AND probability calibration

LGBM_PARAMS = {
    "objective": "binary",
    "n_estimators": 3000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 40,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}
EARLY_STOPPING_ROUNDS = 100

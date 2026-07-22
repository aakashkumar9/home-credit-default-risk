"""Runs the full modelling pipeline: train (CV compare -> champion ->
calibrate -> evaluate) -> predict.

    python -m home_credit.modeling.run_pipeline
"""

import json

from home_credit.modeling import predict as predict_mod
from home_credit.modeling import train as train_mod


def main():
    print("1/2 training (CV compare -> champion -> calibrate -> evaluate)...")
    summary = train_mod.train()
    print(json.dumps(summary, indent=2))

    print("\n2/2 scoring application_test...")
    submission = predict_mod.predict()
    print(f"Wrote {len(submission)} rows to reports/submission.csv")


if __name__ == "__main__":
    main()

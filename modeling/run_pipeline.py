"""Runs the full modeling pipeline in order: train -> calibrate -> explain -> evaluate -> predict.

    python -m modeling.run_pipeline
"""
import json

from modeling import calibrate, evaluate, explain, predict, train


def main():
    print("1/5 training...")
    train_metrics = train.train()
    print(json.dumps(train_metrics, indent=2))

    print("\n2/5 calibrating...")
    calibration_metrics = calibrate.calibrate()
    print(json.dumps(calibration_metrics, indent=2))

    print("\n3/5 explaining (SHAP)...")
    explain_result = explain.explain()
    print(json.dumps({k: round(v, 4) for k, v in list(explain_result["top_20_features"].items())[:10]}, indent=2))

    print("\n4/5 evaluating...")
    eval_metrics = evaluate.evaluate()
    print(json.dumps(eval_metrics, indent=2))

    print("\n5/5 scoring application_test...")
    submission = predict.predict()
    print(f"Wrote {len(submission)} rows to reports/submission.csv")


if __name__ == "__main__":
    main()

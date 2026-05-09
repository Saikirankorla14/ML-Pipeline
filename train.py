"""
Standalone training script.
Run: python train.py --dataset iris --algorithm rf --test-size 0.2

Optionally integrates with MLflow if installed:
  pip install mlflow
  mlflow ui  (then open http://localhost:5000)
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.model import ModelRegistry, DATASETS, ALGORITHMS, ALGO_NAMES

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def parse_args():
    p = argparse.ArgumentParser(description="Train an ML model")
    p.add_argument("--dataset", default="iris", choices=list(DATASETS), help="Dataset to use")
    p.add_argument("--algorithm", default="rf", choices=list(ALGORITHMS), help="Algorithm key")
    p.add_argument("--test-size", type=float, default=0.2, help="Fraction for test split")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--experiment", default="ml-web-app", help="MLflow experiment name")
    return p.parse_args()


def main():
    args = parse_args()
    registry = ModelRegistry()

    print(f"\n[pipeline] Training {ALGO_NAMES[args.algorithm]} on '{args.dataset}' dataset")
    print(f"[pipeline] Test size: {args.test_size:.0%}  |  Random state: {args.random_state}\n")

    if MLFLOW_AVAILABLE:
        mlflow.set_experiment(args.experiment)
        run_ctx = mlflow.start_run()
    else:
        run_ctx = None
        print("[mlflow] Not installed — skipping experiment tracking.")
        print("[mlflow] Install with: pip install mlflow\n")

    result = registry.train(
        dataset=args.dataset,
        algorithm=args.algorithm,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print(f"[eval]  Test accuracy : {result['accuracy']:.4f}")
    print(f"[eval]  CV mean       : {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
    print(f"[model] Registered as : {result['model_id']}")

    if MLFLOW_AVAILABLE and run_ctx:
        mlflow.log_params({
            "dataset": args.dataset,
            "algorithm": args.algorithm,
            "test_size": args.test_size,
            "random_state": args.random_state,
        })
        mlflow.log_metrics({
            "test_accuracy": result["accuracy"],
            "cv_mean": result["cv_mean"],
            "cv_std": result["cv_std"],
        })
        mlflow.end_run()
        print(f"[mlflow] Run logged to experiment '{args.experiment}'")

    print("\n[result]")
    print(json.dumps({k: v for k, v in result.items() if k != "pipeline"}, indent=2))


if __name__ == "__main__":
    main()

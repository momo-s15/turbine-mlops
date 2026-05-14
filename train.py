"""Local training entrypoint: writes app/model/turbine_mlops_model.pkl."""

from __future__ import annotations

import argparse
import os

from turbine_mlops.train_core import train_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train TurbineMLOps RUL model (local)."
    )
    parser.add_argument(
        "--output",
        default="app/model/turbine_mlops_model.pkl",
        help="Path for joblib model artifact",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=1000)
    args = parser.parse_args()

    tracking = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking:
        import mlflow

        mlflow.set_experiment(
            os.environ.get("MLFLOW_EXPERIMENT_NAME", "turbine-mlops-local")
        )
        import mlflow.sklearn

        with mlflow.start_run(run_name="TurbineMLOps_Local"):
            model, metrics = train_model(
                n_samples=args.n_samples,
                seed=args.seed,
                model_path=args.output,
            )
            mlflow.log_param("data_source", "synthetic")
            mlflow.log_metric("rmse", metrics["rmse"])
            mlflow.sklearn.log_model(model, artifact_path="model")
    else:
        _, metrics = train_model(
            n_samples=args.n_samples,
            seed=args.seed,
            model_path=args.output,
        )
    print(f"Model saved to {args.output}. RMSE: {metrics['rmse']:.4f}")


if __name__ == "__main__":
    main()

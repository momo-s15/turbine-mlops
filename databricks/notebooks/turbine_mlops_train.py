# Databricks notebook source
# This notebook trains TurbineMLOps and registers the sklearn model in MLflow.
# Open it from **Repos** (clone of this GitHub repo) and attach a **Running** cluster.

# COMMAND ----------

import os
import sys

try:
    _here = os.path.dirname(__file__)
except NameError:
    _here = os.getcwd()
repo_root = os.path.abspath(os.path.join(_here, "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# COMMAND ----------

import mlflow
import mlflow.sklearn

EXPERIMENT = "/Shared/turbine-mlops-rul"
REGISTERED_NAME = "turbine-mlops-rul-sklearn"

mlflow.set_experiment(EXPERIMENT)

with mlflow.start_run(run_name="TurbineMLOps_Databricks"):
    from turbine_mlops.train_core import train_model

    mlflow.log_param("data_source", "synthetic")
    model, metrics = train_model(n_samples=1000, seed=42, model_path=None)
    mlflow.log_metric("rmse", metrics["rmse"])
    mlflow.sklearn.log_model(model, artifact_path="model")
    run = mlflow.active_run()
    assert run is not None
    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, REGISTERED_NAME)
    print(f"RMSE={metrics['rmse']:.4f} run_id={run_id} registered={REGISTERED_NAME}")

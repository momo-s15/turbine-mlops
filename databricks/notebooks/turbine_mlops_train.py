# Databricks notebook source
# This notebook trains TurbineMLOps and registers the sklearn model in MLflow.
# Open it from your **Git folder** (clone of this GitHub repo) and attach compute.

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

# Spark Connect / Serverless: do not rely on spark.mlflow.modelRegistryUri from Spark.
# Set workspace tracking + registry before set_experiment (avoids CONFIG_NOT_AVAILABLE).
mlflow.set_tracking_uri("databricks")
for _registry_uri in ("databricks-uc", "databricks"):
    try:
        mlflow.set_registry_uri(_registry_uri)
        print(f"MLflow registry URI: {_registry_uri}")
        break
    except Exception as _exc:
        print(f"Skipping registry URI {_registry_uri}: {_exc}")

# Prefer a user-scoped experiment (works on Free Edition); fall back to /Shared.
# `dbutils` is injected at runtime on Databricks only — use getattr pattern so CI flake8 passes.
_dbutils = globals().get("dbutils")
if _dbutils is None:
    import builtins

    _dbutils = getattr(builtins, "dbutils", None)

if _dbutils is not None:
    try:
        _user = (
            _dbutils.notebook.entry_point.getDbutils()
            .notebook()
            .getContext()
            .userName()
            .get()
        )
        EXPERIMENT = f"/Users/{_user}/turbine-mlops-rul"
    except Exception:
        EXPERIMENT = "/Shared/turbine-mlops-rul"
else:
    EXPERIMENT = "/Shared/turbine-mlops-rul"

print(f"MLflow experiment: {EXPERIMENT}")
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
    try:
        mlflow.register_model(model_uri, REGISTERED_NAME)
        reg_status = REGISTERED_NAME
    except Exception as _reg_exc:
        reg_status = f"skipped ({_reg_exc}) — register from Experiments UI"
    print(f"RMSE={metrics['rmse']:.4f} run_id={run_id} model_uri={model_uri}")
    print(f"register_model: {reg_status}")

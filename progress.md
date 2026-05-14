# TurbineMLOps — implementation progress log

Chronological record of what was implemented in this repo (local MLOps + Databricks handoff).

**Rename (2026):** The project was originally scaffolded as **AeroFlow** (`aeroflow` package, `aeroflow_model.pkl`, etc.). It was renamed to **TurbineMLOps** with Python package **`turbine_mlops`**, default artifact **`turbine_mlops_model.pkl`**, env **`TURBINE_MLOPS_MODEL_PATH`**, Databricks experiment **`/Shared/turbine-mlops-rul`**, and registered model **`turbine-mlops-rul-sklearn`**.

---

## 1. Scope (agreed)

- **Model:** scikit-learn `RandomForestRegressor` on synthetic turbofan-style features; target RUL.
- **Databricks:** training + managed MLflow / model registry only (no Databricks Model Serving in v1).
- **Serving:** FastAPI + Prometheus metrics + Docker + GitHub Actions CI.
- **Data:** synthetic dataset (not full NASA CMAPSS file ingestion).

---

## 2. Repository layout (current names)

| Path | Purpose |
|------|---------|
| `turbine_mlops/__init__.py` | Package marker + version string. |
| `turbine_mlops/synthetic_data.py` | `make_synthetic_turbofan_df()` — cycles, two sensors, RUL with clamp. |
| `turbine_mlops/train_core.py` | `train_model()` — split, fit, RMSE, optional `joblib` dump; feature column names aligned with API. |
| `train.py` | CLI: `--output`, `--seed`, `--n-samples`; writes `app/model/turbine_mlops_model.pkl`; optional MLflow when `MLFLOW_TRACKING_URI` is set. |
| `app/__init__.py` | Package marker. |
| `app/main.py` | FastAPI: `POST /predict`, `GET /health`, `GET /metrics` (Prometheus ASGI mount); model path via `TURBINE_MLOPS_MODEL_PATH` or default next to `main.py`. |
| `app/model/.gitkeep` | Keeps `app/model/` in git; actual `*.pkl` ignored. |
| `requirements.txt` | Pinned runtime deps (FastAPI, uvicorn, prometheus-client, pandas, scikit-learn, joblib, mlflow, pydantic). |
| `requirements-dev.txt` | flake8, pytest, httpx. |
| `.gitignore` | `__pycache__`, venv, `.env`, `*.pkl`, `mlruns/`, IDE junk. |
| `pytest.ini` | `[pytest] pythonpath = .` so `import turbine_mlops` works during tests. |
| `tests/conftest.py` | Session autouse fixture: trains small model into `app/model/turbine_mlops_model.pkl` for tests. |
| `tests/test_predict.py` | Predict JSON shape + non-negative RUL; health 200 when model present. |
| `tests/test_metrics.py` | After one predict, `/metrics` contains `predictions_total`. |
| `.github/workflows/ci.yml` | Workflow **TurbineMLOps CI**: checkout, Python 3.10, pip install, flake8 (E9,F63,F7,F82), `python train.py`, pytest; Docker build image **`turbinemlops:latest`**. |
| `Dockerfile` | `python:3.10-slim`, install deps, copy `turbine_mlops` + `app` + `train.py`, run `train.py` in image, `uvicorn` on 8000. |
| `.dockerignore` | Shrinks context (excludes tests, databricks, dev requirements, etc.). |
| `databricks/notebooks/turbine_mlops_train.py` | Databricks notebook: `sys.path` repo root, `mlflow.set_experiment("/Shared/turbine-mlops-rul")`, train, log metric, `log_model`, `register_model("turbine-mlops-rul-sklearn")`. |
| `databricks/README.md` | Step-by-step UI: PAT, cluster `turbine-mlops-dev`, Repos, run notebook, Experiments, Models, download `model.pkl` → `app/model/turbine_mlops_model.pkl`. |
| `README.md` | Local setup, optional MLflow env vars, tests, Docker, pointer to Databricks doc. |

---

## 3. Fixes and gotchas resolved

1. **`ModuleNotFoundError` for the training package in pytest**  
   - Cause: project root not on `PYTHONPATH` when collecting `tests/conftest.py`.  
   - Fix: added `pytest.ini` with `pythonpath = .`.

2. **`FastAPIError` on `/health` during import**  
   - Cause: return type annotation `dict | JSONResponse` was not valid for response-field inference.  
   - Fix: `@app.get("/health", response_model=None)` and removed the problematic annotation.

3. **`ValueError: Duplicated timeseries in CollectorRegistry` (Prometheus)**  
   - Cause: first import of `app.main` failed mid-module after metrics were registered; a second import re-executed and tried to register the same metric names again.  
   - Fix: resolving the FastAPI health issue made module import single-shot and stable.

---

## 4. Verification performed

- **`pytest -q`:** 3 tests passed (after `pytest.ini` and health fix).
- **`flake8`** with `--select=E9,F63,F7,F82`: clean on the checked run.
- **`python train.py`:** produces `app/model/turbine_mlops_model.pkl` and prints RMSE.
- **`docker build`:** not verified on one dev machine (Docker engine not running); CI includes a Docker build on Ubuntu.

---

## 5. Suggested next steps

1. Initialize git, commit, push to GitHub; confirm **TurbineMLOps CI** passes.
2. In Databricks: **Repos** → clone repo → run `databricks/notebooks/turbine_mlops_train.py` on an ML cluster → download artifact → `app/model/turbine_mlops_model.pkl`.
3. Start Docker Desktop locally and run `docker build -t turbinemlops:latest .` if you want parity with CI.

---

*This file is a human-readable log of implementation work; it is safe to edit or delete. Do not commit secrets (PATs, `.env`).*

# TurbineMLOps

Predictive maintenance demo: synthetic turbofan-style sensors → **RandomForest** RUL (Remaining Useful Life) → **MLflow** on Databricks (optional) → **FastAPI** + **Prometheus** → **Docker** + **GitHub Actions**.

## Local setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
python train.py
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- **Predict:** `POST /predict` with JSON `{"sensor_11": 47.5, "sensor_14": 8125}`  
- **Metrics:** `GET /metrics`  
- **Health:** `GET /health`  

Model path defaults to `app/model/turbine_mlops_model.pkl`. Override with env **`TURBINE_MLOPS_MODEL_PATH`**.

### Optional local MLflow

```bash
set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
set MLFLOW_EXPERIMENT_NAME=turbine-mlops-local
python train.py
```

## Tests and lint

```bash
python train.py --n-samples 200 --seed 1
pytest -q
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

## Docker

```bash
docker build -t turbinemlops:latest .
docker run --rm -p 8000:8000 turbinemlops:latest
```

## Databricks

Follow [databricks/README.md](databricks/README.md) for click-by-click UI steps. Training notebook: [databricks/notebooks/turbine_mlops_train.py](databricks/notebooks/turbine_mlops_train.py).

**Security:** never commit workspace tokens, PATs, or `.env` files containing secrets.

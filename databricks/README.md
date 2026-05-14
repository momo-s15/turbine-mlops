# TurbineMLOps on Databricks

Use this after the code is in a **GitHub** repository and you have opened your **Databricks** workspace in the browser.

## What you are building

A **Running** cluster, your repo in **Repos**, one successful notebook run, an **Experiment** with metrics, a **registered model**, then a **downloaded** `model.pkl` you rename to `app/model/turbine_mlops_model.pkl` for FastAPI.

## Part A — Personal access token (optional for click-only flow)

1. Click your **avatar** (top-right) → **Settings** / **User settings**.
2. Open **Developer** or **Personal access tokens**.
3. **Generate new token**, comment `turbine-mlops`, copy it once, store it safely. **Never commit it to Git.**

## Part B — Cluster

1. Left sidebar: **Compute** (use the top **search** bar if you do not see it).
2. **Create compute** / **Create cluster**.
3. Name: `turbine-mlops-dev`.
4. Prefer **Single node** if available; otherwise default is fine.
5. **Databricks runtime**: choose an **ML** / **Machine Learning** runtime (includes scikit-learn and MLflow).
6. **Create**, then wait until status is **Running** (green). Start it with the **play** icon if it stopped.

## Part C — Repos (clone GitHub)

1. Sidebar: **Workspace** → **Repos** (under your user).
2. **Add repo** → paste your repo HTTPS URL (`https://github.com/<you>/<repo>.git`).
3. Complete **GitHub** authentication if prompted.
4. Confirm you see `turbine_mlops/`, `app/`, `databricks/` in the file tree.

## Part D — Run the training notebook

1. In Repos, open `databricks/notebooks/turbine_mlops_train.py`.
2. Attach cluster **`turbine-mlops-dev`** from the notebook’s cluster dropdown; wait until connected.
3. **Run all** cells. Wait for the final print with `RMSE=` and `run_id=`.

If you see `ModuleNotFoundError: turbine_mlops`, you are not running from the **Repos** copy, or the repo is missing files—pull the latest from GitHub and try **Restart** on the cluster.

## Part E — Experiments (MLflow)

1. Sidebar: **Machine Learning** → **Experiments** (or search **Experiments**).
2. Open experiment **`/Users/<your-email>/turbine-mlops-rul`** (created by the notebook on Free Edition / Serverless) or **`/Shared/turbine-mlops-rul`** if you use that path.
3. Open the latest **Run** and confirm metric **`rmse`** and artifact **`model`**.

## Part F — Registered model and download

1. Sidebar: **Models** (or **Catalog** → **Models** on Unity Catalog workspaces).
2. Open **`turbine-mlops-rul-sklearn`** → latest **version**.
3. Under **Artifacts** / **Files**, open the sklearn model files until you find **`model.pkl`** (name may be exactly `model.pkl` inside the logged model folder).
4. **Download** it to your laptop, then copy/rename it to:

`app/model/turbine_mlops_model.pkl`

in your local clone (same path as in this repo).

5. Run the API locally:

```bash
uvicorn app.main:app --reload
```

Test:

```bash
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"sensor_11\": 47.5, \"sensor_14\": 8125}"
```

## Unity Catalog note

If your workspace registers models under **Catalog** instead of legacy **Models**, use the same steps: open the model → latest version → artifacts → download `model.pkl`.

## Troubleshooting

| Issue | What to try |
|--------|----------------|
| Cluster won’t start | Trial quota; smaller runtime; different region if allowed. |
| `register_model` fails | Permissions; ask workspace admin. You can still use the **run** artifact without registering. |
| `[CONFIG_NOT_AVAILABLE] spark.mlflow.modelRegistryUri` (Serverless / Spark Connect) | The notebook sets **`mlflow.set_tracking_uri("databricks")`** and **`mlflow.set_registry_uri("databricks-uc")`** or **`"databricks"`** before `set_experiment`. Pull the latest notebook from GitHub and **Run all** again. If it still fails, try compute that is a classic **ML cluster** instead of Serverless. |
| Wrong sklearn file | Inside MLflow artifacts, use the file produced by the **sklearn** flavor (`model.pkl`), not random side files. |

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from turbine_mlops.synthetic_data import make_synthetic_turbofan_df

FEATURE_COLS = ["sensor_11_static_pressure", "sensor_14_core_speed"]
TARGET_COL = "RUL"


def train_model(
    n_samples: int = 1000,
    seed: int = 42,
    model_path: str | None = None,
) -> tuple[RandomForestRegressor, dict[str, float]]:
    df: pd.DataFrame = make_synthetic_turbofan_df(n_samples=n_samples, seed=seed)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    model = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=seed
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    metrics: dict[str, float] = {"rmse": rmse}
    if model_path:
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
    return model, metrics

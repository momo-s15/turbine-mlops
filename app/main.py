"""TurbineMLOps FastAPI inference service."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, Field

MODEL_PATH = Path(
    os.environ.get(
        "TURBINE_MLOPS_MODEL_PATH",
        str(Path(__file__).resolve().parent / "model" / "turbine_mlops_model.pkl"),
    )
)

_model = None
_model_mtime: float | None = None


def _ensure_model_loaded() -> None:
    """Load or reload the pickle when the file appears or its mtime changes (e.g. after train.py)."""
    global _model, _model_mtime
    if not MODEL_PATH.is_file():
        _model = None
        _model_mtime = None
        return
    mtime = MODEL_PATH.stat().st_mtime
    if _model is not None and _model_mtime == mtime:
        return
    _model = joblib.load(MODEL_PATH)
    _model_mtime = mtime


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_model_loaded()
    yield


app = FastAPI(title="TurbineMLOps Inference API", lifespan=lifespan)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

PRED_COUNTER = Counter(
    "predictions_total",
    "Total predictions served",
)
PRED_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Latency of RUL predictions in seconds",
)


class EngineTelemetry(BaseModel):
    sensor_11: float = Field(..., description="Sensor 11 static pressure (normalized scale)")
    sensor_14: float = Field(..., description="Sensor 14 core speed (normalized scale)")


@app.get("/health", response_model=None)
def health():
    _ensure_model_loaded()
    if _model is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "detail": f"No model at {MODEL_PATH}. Run: python train.py",
            },
        )
    return {"status": "healthy", "model_path": str(MODEL_PATH)}


@app.post("/predict")
def predict(data: EngineTelemetry) -> dict[str, float]:
    _ensure_model_loaded()
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not available at {MODEL_PATH}. Run: python train.py",
        )
    start = time.perf_counter()
    input_df = pd.DataFrame(
        [[data.sensor_11, data.sensor_14]],
        columns=["sensor_11_static_pressure", "sensor_14_core_speed"],
    )
    try:
        prediction = float(_model.predict(input_df)[0])
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Prediction failed ({type(exc).__name__}: {exc}). "
                f"If you replaced the model file, run: python train.py"
            ),
        ) from exc
    PRED_COUNTER.inc()
    PRED_LATENCY.observe(time.perf_counter() - start)
    return {"predicted_RUL": round(prediction, 2)}

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_metrics_contains_prediction_counter_after_predict() -> None:
    with TestClient(app) as client:
        client.post("/predict", json={"sensor_11": 47.2, "sensor_14": 8110.0})
        response = client.get("/metrics")
    assert response.status_code == 200
    assert b"predictions_total" in response.content

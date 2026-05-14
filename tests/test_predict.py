from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_predict_returns_rul() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"sensor_11": 47.5, "sensor_14": 8125.0},
        )
    assert response.status_code == 200
    data = response.json()
    assert "predicted_RUL" in data
    assert data["predicted_RUL"] >= 0


def test_health_when_model_present() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

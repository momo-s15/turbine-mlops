"""Ensure model artifact exists before API tests (session-wide)."""

from __future__ import annotations

from pathlib import Path

import pytest

from turbine_mlops.train_core import train_model


@pytest.fixture(scope="session", autouse=True)
def _ensure_model_artifact() -> None:
    out = Path("app/model/turbine_mlops_model.pkl")
    train_model(n_samples=200, seed=1, model_path=str(out))

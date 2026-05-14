import numpy as np
import pandas as pd


def make_synthetic_turbofan_df(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """NASA CMAPSS-style synthetic sensor data with a simple RUL target."""
    rng = np.random.default_rng(seed)
    cycle = rng.integers(1, 200, size=n_samples)
    sensor_11 = rng.uniform(47.0, 48.5, size=n_samples)
    sensor_14 = rng.uniform(8100.0, 8150.0, size=n_samples)
    df = pd.DataFrame(
        {
            "cycle": cycle,
            "sensor_11_static_pressure": sensor_11,
            "sensor_14_core_speed": sensor_14,
        }
    )
    rul = (
        250
        - df["cycle"]
        - (df["sensor_11_static_pressure"] - 47.0) * 20
        + rng.normal(0, 5, size=n_samples)
    )
    df["RUL"] = np.maximum(rul, 0)
    return df

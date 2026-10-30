"""
All features at an origin time t only use intervals up to and including t, so
nothing from the future leaks into the model inputs.
"""
import pandas as pd

from .config import LAGS, ROLL_WINDOWS


def build_features(y: pd.Series) -> pd.DataFrame:
    X = pd.DataFrame(index=y.index)
    # lag_k is the k-th most recent observed interval (lag_1 = y_t, the latest one)
    for k in LAGS:
        X[f"lag_{k}"] = y.shift(k - 1)
    for w in ROLL_WINDOWS:
        r = y.rolling(w, min_periods=w)
        X[f"roll_mean_{w}"] = r.mean()
        X[f"roll_std_{w}"] = r.std()
        X[f"roll_max_{w}"] = r.max()
    idx = y.index
    X["hour"] = idx.hour + idx.minute / 60
    X["day_of_week"] = idx.dayofweek
    X["month"] = idx.month
    X["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    return X


def window_target(y: pd.Series, steps: int) -> pd.Series:
    """Total tickets in the `steps` intervals that follow each origin time."""
    return y.rolling(steps, min_periods=steps).sum().shift(-steps)

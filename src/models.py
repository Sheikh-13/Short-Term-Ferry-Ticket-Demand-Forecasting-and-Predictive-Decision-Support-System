import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from .config import INTERVAL_ALPHA

ML_MODELS = {
    "Linear Regression": lambda: LinearRegression(),
    "Random Forest": lambda: RandomForestRegressor(
        n_estimators=40, max_depth=12, min_samples_leaf=50, max_features=0.6,
        max_samples=0.15, n_jobs=-1, random_state=42),
    "Gradient Boosting": lambda: HistGradientBoostingRegressor(
        max_iter=250, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=50,
        random_state=42),
    "XGBoost": lambda: XGBRegressor(
        n_estimators=250, learning_rate=0.06, max_depth=6, subsample=0.8,
        colsample_bytree=0.9, tree_method="hist", n_jobs=-1, random_state=42),
}


def slug(name: str) -> str:
    # File-name friendly model name.
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


class IntervalCalibrator:
    """
    Prediction intervals from validation residuals, binned by predicted level.
    Demand variance grows with the level of demand, so residual quantiles are
    computed separately within bins of the predicted value.
    """
    def __init__(self, alpha: float = INTERVAL_ALPHA, n_bins: int = 10, min_bin: int = 50):
        self.alpha, self.n_bins, self.min_bin = alpha, n_bins, min_bin

    def fit(self, pred, actual):
        pred, resid = np.asarray(pred, float), np.asarray(actual, float) - np.asarray(pred, float)
        qs = [self.alpha / 2, 1 - self.alpha / 2]
        edges = np.unique(np.quantile(pred, np.linspace(0, 1, self.n_bins + 1)))
        self.edges_ = edges[1:-1] if len(edges) > 2 else np.array([])
        bins = np.digitize(pred, self.edges_)
        overall = np.quantile(resid, qs)
        self.lo_, self.hi_ = [], []
        for b in range(len(self.edges_) + 1):
            r = resid[bins == b]
            lo, hi = np.quantile(r, qs) if len(r) >= self.min_bin else overall
            self.lo_.append(lo)
            self.hi_.append(hi)
        self.lo_, self.hi_ = np.array(self.lo_), np.array(self.hi_)
        return self

    def predict_interval(self, pred):
        pred = np.asarray(pred, float)
        b = np.digitize(pred, self.edges_)
        lower = np.minimum(np.maximum(pred + self.lo_[b], 0), pred)
        upper = np.maximum(pred + self.hi_[b], pred)
        return lower, upper


def baseline_predictions(y, steps: int, ma_window: int) -> dict:
    """Naive: next window repeats the last window. MA: recent average rate x window."""
    return {
        "Naive": y.rolling(steps, min_periods=steps).sum(),
        "Moving Average": y.rolling(ma_window, min_periods=ma_window).mean() * steps,
    }

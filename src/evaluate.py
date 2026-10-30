import numpy as np
import pandas as pd

from .config import LEAD_TIME_RECALL, PEAK_MISS_RATIO


def point_metrics(actual, pred) -> dict:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    err = pred - actual
    nz = actual > 0
    return {
        "MAE": np.mean(np.abs(err)),
        "RMSE": np.sqrt(np.mean(err ** 2)),
        # MAPE is undefined when the actual value is 0, so only intervals with sales are used
        "MAPE": np.mean(np.abs(err[nz]) / actual[nz]) * 100 if nz.any() else np.nan,
        "WAPE": np.sum(np.abs(err)) / max(np.sum(actual), 1e-9) * 100,
    }


def interval_metrics(actual, lower, upper) -> dict:
    actual = np.asarray(actual, float)
    width = np.asarray(upper, float) - np.asarray(lower, float)
    return {
        "Coverage %": np.mean((actual >= lower) & (actual <= upper)) * 100,
        "Band Width": width.mean(),
        "Relative Band Width %": width.mean() / max(actual.mean(), 1e-9) * 100,
    }


def peak_miss_rate(actual, pred, threshold) -> float:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    peaks = actual >= threshold
    if not peaks.any():
        return np.nan
    return float(np.mean(pred[peaks] < PEAK_MISS_RATIO * actual[peaks]) * 100)


def horizon_table(preds: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Metrics per (target, model, horizon) from the long prediction table."""
    rows = []
    for (target, model, h), g in preds.groupby(["target", "model", "horizon_min"]):
        row = {"target": target, "model": model, "horizon_min": h, "n_origins": len(g)}
        row.update(point_metrics(g["actual"], g["pred"]))
        row.update(interval_metrics(g["actual"], g["lower"], g["upper"]))
        row["Forecast Accuracy %"] = 100 - row["WAPE"]
        row["Peak Miss Rate %"] = peak_miss_rate(g["actual"], g["pred"], thresholds[(target, h)])
        rows.append(row)
    return pd.DataFrame(rows)


def kpi_table(ht: pd.DataFrame) -> pd.DataFrame:
    """One row per (target, model) summarising the five KPIs across horizons."""
    rows = []
    for (target, model), g in ht.groupby(["target", "model"]):
        g = g.sort_values("horizon_min")
        short, long_ = g.iloc[0], g.iloc[-1]
        recall_ok = g[(100 - g["Peak Miss Rate %"]) / 100 >= LEAD_TIME_RECALL]
        rows.append({
            "target": target, "model": model,
            "Forecast Accuracy %": g["Forecast Accuracy %"].mean(),
            "Error Drift (pp)": long_["WAPE"] - short["WAPE"],
            "Peak Miss Rate %": g["Peak Miss Rate %"].mean(),
            "Confidence Band Width (rel %)": g["Relative Band Width %"].mean(),
            "Forecast Lead Time (min)": int(recall_ok["horizon_min"].max()) if len(recall_ok) else 0,
        })
    return pd.DataFrame(rows)

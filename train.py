"""
Run the full forecasting pipeline: prepare data, train and evaluate every model,
and save the results used by the Streamlit dashboard.
"""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd

from src.config import (ARTIFACTS, HORIZONS, MODELS_DIR, MOVING_AVG_WINDOW, PEAK_QUANTILE,
                        TARGETS, TRAIN_END, VAL_END)
from src.data_prep import build_series
from src.evaluate import horizon_table, kpi_table
from src.features import build_features, window_target
from src.models import ML_MODELS, IntervalCalibrator, baseline_predictions, slug
from src.timeseries_models import arima_window_forecasts, prophet_window_forecasts

CACHE = ARTIFACTS / "cache"
ARIMA_FIT_START = "2023-07-01"     # ARIMA parameters estimated on the last 6 months of training data
PROPHET_FIT_START = "2022-01-01"   # Prophet fitted on the last 2 years of training data
ARIMA_CALIBRATION_DAYS = 60        # ARIMA intervals are calibrated on the last 60 days of validation


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(skip_ts=False, targets=None, horizons=None, train_stride=1):
    t_start = time.time()
    ARTIFACTS.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    targets = targets or list(TARGETS)
    horizons = {h: HORIZONS[h] for h in (horizons or HORIZONS)}

    log("Step 1: preparing the 15-minute series")
    series = build_series()
    series.round(2).to_csv(ARTIFACTS / "series.csv.gz")
    log(f"  {len(series):,} intervals, {int(series['masked'].sum()):,} masked (long gaps)")

    val_start = pd.Timestamp(TRAIN_END) + pd.Timedelta("15min")
    test_start = pd.Timestamp(VAL_END) + pd.Timedelta("15min")
    all_preds, thresholds = [], {}

    for target in targets:
        y = series[target]
        X = build_features(y)
        feat_cols = list(X.columns)
        log(f"Step 2: features built for {target}: {len(feat_cols)} columns")

        eval_index = y.loc[val_start:].index
        origins = eval_index[eval_index.minute == 0]
        arima_origins = origins[origins >= pd.Timestamp(VAL_END) - pd.Timedelta(days=ARIMA_CALIBRATION_DAYS)]
        arima = prophet_step = None
        if not skip_ts:
            log(f"  fitting ARIMA (Fourier seasonality) for {target} ...")
            arima = arima_window_forecasts(y, ARIMA_FIT_START, TRAIN_END, arima_origins)
            log(f"  fitting Prophet for {target} ...")
            prophet_step = prophet_window_forecasts(y, PROPHET_FIT_START, TRAIN_END, eval_index)

        for hmin, steps in horizons.items():
            data = X.assign(_y=window_target(y, steps)).dropna()
            train = data.loc[:TRAIN_END].iloc[::train_stride]
            val = data.loc[val_start:VAL_END]
            test = data.loc[test_start:]
            val_h = val[val.index.minute == 0]
            test_h = test[test.index.minute == 0]
            thresholds[(target, hmin)] = float(data.loc[:TRAIN_END, "_y"].quantile(PEAK_QUANTILE))

            cands = {}
            for name, s in baseline_predictions(y, steps, MOVING_AVG_WINDOW).items():
                cands[name] = (s.reindex(val_h.index).values, s.reindex(test_h.index).values, None)
            for name, factory in ML_MODELS.items():
                model = factory().fit(train[feat_cols], train["_y"])
                pv = np.clip(model.predict(val_h[feat_cols]), 0, None)
                pt = np.clip(model.predict(test_h[feat_cols]), 0, None)
                cands[name] = (pv, pt, model)
            if not skip_ts:
                cands["ARIMA (Fourier)"] = (arima[steps].reindex(val_h.index).values,
                                            arima[steps].reindex(test_h.index).values, None)
                pw = prophet_step.rolling(steps, min_periods=steps).sum().shift(-steps)
                cands["Prophet"] = (pw.reindex(val_h.index).values, pw.reindex(test_h.index).values, None)

            for name, (pv, pt, model) in cands.items():
                assert not np.isnan(pt).any(), f"NaN test predictions: {name}"
                ok = ~np.isnan(pv)
                cal = IntervalCalibrator().fit(pv[ok], val_h["_y"].values[ok])
                lower, upper = cal.predict_interval(pt)
                all_preds.append(pd.DataFrame({
                    "origin": test_h.index, "target": target, "horizon_min": hmin, "model": name,
                    "actual": test_h["_y"].values, "pred": pt, "lower": lower, "upper": upper}))
                if model is not None:
                    joblib.dump({"model": model, "calibrator": cal, "features": feat_cols},
                                MODELS_DIR / f"{target}__{hmin}m__{slug(name)}.joblib", compress=3)
            log(f"  {target} {hmin:>3} min done ({len(test_h):,} test origins, {len(cands)} models)")

        pd.concat([p for p in all_preds if p["target"].iloc[0] == target], ignore_index=True).to_pickle(
            CACHE / f"preds_{target}.pkl")
        json.dump({str(h): v for (t, h), v in thresholds.items() if t == target},
                  open(CACHE / f"thr_{target}.json", "w"))
        log(f"Saved results for {target}")

    missing = [t for t in TARGETS if not (CACHE / f"preds_{t}.pkl").exists()]
    if missing:
        log(f"Run the remaining target(s) to finish: {missing}")
        return
    preds = pd.concat([pd.read_pickle(CACHE / f"preds_{t}.pkl") for t in TARGETS], ignore_index=True)
    thresholds = {(t, int(h)): v for t in TARGETS for h, v in json.load(open(CACHE / f"thr_{t}.json")).items()}
    preds.round({"actual": 1, "pred": 2, "lower": 2, "upper": 2}).to_csv(
        ARTIFACTS / "predictions.csv.gz", index=False)

    ht = horizon_table(preds, thresholds)
    kpis = kpi_table(ht)
    ht.round(3).to_csv(ARTIFACTS / "metrics_by_horizon.csv", index=False)
    kpis.round(3).to_csv(ARTIFACTS / "kpis.csv", index=False)
    json.dump({"peak_thresholds": {f"{t}|{h}": v for (t, h), v in thresholds.items()},
               "train_end": TRAIN_END, "val_end": VAL_END,
               "data_start": str(series.index.min()), "data_end": str(series.index.max()),
               "test_start": str(test_start)},
              open(ARTIFACTS / "meta.json", "w"), indent=2)

    log(f"Finished in {(time.time() - t_start) / 60:.1f} min")
    show = ht.pivot_table(index=["target", "model"], columns="horizon_min", values="MAE").round(2)
    print("\nTest MAE by model and horizon:\n", show.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ts", action="store_true", help="skip ARIMA and Prophet")
    ap.add_argument("--targets", nargs="+", choices=list(TARGETS))
    ap.add_argument("--horizons", nargs="+", type=int, choices=list(HORIZONS))
    ap.add_argument("--train-stride", type=int, default=1, help="use every Nth training row (testing only)")
    a = ap.parse_args()
    main(a.skip_ts, a.targets, a.horizons, a.train_stride)

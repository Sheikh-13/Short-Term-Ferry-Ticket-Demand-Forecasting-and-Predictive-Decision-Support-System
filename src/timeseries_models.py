"""Step 3: statistical time-series models (ARIMA with Fourier seasonality, Prophet)."""
import numpy as np
import pandas as pd

MAX_STEPS = 8


def fourier_terms(index: pd.DatetimeIndex, k_day: int = 4, k_week: int = 2) -> np.ndarray:
    minutes = ((index - pd.Timestamp("2015-01-01")) / pd.Timedelta("1min")).values
    cols = []
    for period, k in ((1440, k_day), (10080, k_week)):
        for j in range(1, k + 1):
            cols += [np.sin(2 * np.pi * j * minutes / period), np.cos(2 * np.pi * j * minutes / period)]
    return np.column_stack(cols)


def prophet_window_forecasts(y: pd.Series, fit_start, fit_end, pred_index: pd.DatetimeIndex) -> pd.Series:
    """Fit Prophet once and return its per-interval forecast over pred_index."""
    import logging
    from prophet import Prophet
    logging.getLogger("prophet").setLevel(logging.ERROR)
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

    fit = y.loc[fit_start:fit_end].dropna()
    m = Prophet(growth="flat", daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    m.add_seasonality(name="daily", period=1, fourier_order=10)
    m.fit(pd.DataFrame({"ds": fit.index, "y": fit.values}))
    fc = m.predict(pd.DataFrame({"ds": pred_index}))
    return pd.Series(np.clip(fc["yhat"].values, 0, None), index=pred_index)


def arima_window_forecasts(y: pd.Series, fit_start, fit_end, origins: pd.DatetimeIndex,
                           order=(2, 0, 1)) -> pd.DataFrame:
    """Rolling-origin forecasts from an ARIMA model with Fourier seasonal regressors.

    Parameters are estimated once on [fit_start, fit_end]; the model state is then
    advanced through the data with the parameters fixed, and an 8-step forecast is
    issued at every origin. Returns cumulative forecasts (columns 1..8 steps).
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    fit = y.loc[fit_start:fit_end]
    res = SARIMAX(fit, exog=fourier_terms(fit.index), order=order, trend="c").fit(
        disp=False, maxiter=100)

    step = pd.Timedelta("15min")
    origins = pd.DatetimeIndex(origins)
    cur_end = fit.index[-1]
    out = np.full((len(origins), MAX_STEPS), np.nan)
    for i, t in enumerate(origins):
        new = y.loc[cur_end + step: t]
        if len(new):
            res = res.extend(new, exog=fourier_terms(new.index))
            cur_end = t
        fut_idx = pd.date_range(t + step, periods=MAX_STEPS, freq="15min")
        f = res.forecast(MAX_STEPS, exog=fourier_terms(fut_idx))
        out[i] = np.cumsum(np.clip(np.asarray(f), 0, None))
    return pd.DataFrame(out, index=origins, columns=range(1, MAX_STEPS + 1))

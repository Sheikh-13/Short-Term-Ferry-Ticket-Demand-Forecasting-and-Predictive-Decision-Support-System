"""Streamlit dashboard: short-term ferry ticket demand forecasting.

Run with:  streamlit run app.py
"""
import json
from datetime import date

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import ARTIFACTS, HORIZONS, MODELS_DIR, TARGETS
from src.features import build_features
from src.models import ML_MODELS, slug

st.set_page_config(page_title="Ferry Demand Forecasting", page_icon="⛴️", layout="wide")


# ---------- data loading ----------
@st.cache_data
def load_predictions() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / "predictions.csv.gz", parse_dates=["origin"])


@st.cache_data
def load_tables():
    return (pd.read_csv(ARTIFACTS / "metrics_by_horizon.csv"),
            pd.read_csv(ARTIFACTS / "kpis.csv"),
            json.load(open(ARTIFACTS / "meta.json")))


@st.cache_data
def load_series() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / "series.csv.gz", index_col=0, parse_dates=True)


@st.cache_resource
def load_model(target: str, horizon: int, model_name: str):
    return joblib.load(MODELS_DIR / f"{target}__{horizon}m__{slug(model_name)}.joblib")


def horizon_label(minutes: int) -> str:
    return f"{minutes} min" if minutes < 60 else f"{minutes // 60} h"


if not (ARTIFACTS / "predictions.csv.gz").exists():
    st.error("No results found. Run `python train.py` first to train the models.")
    st.stop()

preds = load_predictions()
metrics, kpis, meta = load_tables()

# ---------- sidebar controls ----------
st.sidebar.header("Forecast settings")
target = st.sidebar.radio("Forecast target", list(TARGETS), format_func=TARGETS.get)
model_names = sorted(preds["model"].unique())
model = st.sidebar.selectbox("Model", model_names, index=model_names.index("Gradient Boosting"))
horizon = st.sidebar.select_slider("Forecast horizon", options=list(HORIZONS), value=60,
                                   format_func=horizon_label)

d_min, d_max = preds["origin"].min().date(), preds["origin"].max().date()
sel_date = st.sidebar.date_input("Forecast date", value=min(max(date(2025, 7, 12), d_min), d_max),
                                 min_value=d_min, max_value=d_max)
sel_hour = st.sidebar.selectbox("Forecast time", list(range(24)), index=14,
                                format_func=lambda h: f"{h:02d}:00")
span_days = st.sidebar.selectbox("Chart range", [1, 3, 7], format_func=lambda d: f"{d} day{'s' if d > 1 else ''}")
st.sidebar.caption(f"Evaluation period: {d_min} to {d_max}. Models were trained on data up to "
                   f"{meta['train_end'][:10]} and tuned on 2024.")

st.title("Short-Term Ferry Ticket Demand Forecasting")
st.caption("Toronto Island Park ferry tickets: forecasts of tickets sold or redeemed in the next "
           "15 minutes to 2 hours, with 90% prediction intervals.")

tab_fc, tab_cmp, tab_live = st.tabs(["Forecast vs actual", "Model comparison", "Latest forecast"])

# ---------- tab 1: forecast vs actual ----------
with tab_fc:
    sub = (preds[(preds["target"] == target) & (preds["model"] == model) & (preds["horizon_min"] == horizon)]
           .set_index("origin").sort_index())
    origin = pd.Timestamp(sel_date) + pd.Timedelta(hours=sel_hour)
    win_end = origin + pd.Timedelta(minutes=horizon)
    st.subheader(f"{TARGETS[target]}: {origin:%a %d %b %Y}, {origin:%H:%M} to {win_end:%H:%M}")

    if origin in sub.index:
        r = sub.loc[origin]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"Forecast ({horizon_label(horizon)} window)", f"{r['pred']:.0f} tickets")
        c2.metric("90% prediction interval", f"{r['lower']:.0f} to {r['upper']:.0f}")
        c3.metric("Actual", f"{r['actual']:.0f} tickets")
        c4.metric("Error (forecast - actual)", f"{r['pred'] - r['actual']:+.0f}")
    else:
        st.info("No forecast for this time. The data has a long gap here, or it is outside the "
                "evaluation period. Try another time.")

    d = sub.loc[pd.Timestamp(sel_date): pd.Timestamp(sel_date) + pd.Timedelta(days=span_days) - pd.Timedelta(minutes=1)]
    if d.empty:
        st.warning("No data for the chosen range.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d.index, y=d["upper"], mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=d.index, y=d["lower"], mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(31,119,180,0.20)", name="90% interval"))
        fig.add_trace(go.Scatter(x=d.index, y=d["pred"], mode="lines+markers", name="Forecast",
                                 line=dict(color="#1f77b4", width=2)))
        fig.add_trace(go.Scatter(x=d.index, y=d["actual"], mode="lines+markers", name="Actual",
                                 line=dict(color="#d62728", width=2)))
        if origin in d.index:
            fig.add_shape(type="line", x0=origin, x1=origin, y0=0, y1=1, yref="paper",
                          line=dict(color="gray", dash="dot"))
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                          yaxis_title=f"Tickets in the next {horizon_label(horizon)}",
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")
        st.caption("Each point is a forecast issued at that time for the window that follows it. "
                   "The dotted line marks the selected forecast time.")

# ---------- tab 2: model comparison ----------
with tab_cmp:
    mt = metrics[metrics["target"] == target]
    st.subheader(f"All models, {horizon_label(horizon)} horizon ({TARGETS[target]})")
    cols = ["model", "MAE", "RMSE", "MAPE", "Forecast Accuracy %", "Coverage %",
            "Relative Band Width %", "Peak Miss Rate %"]
    tbl = mt[mt["horizon_min"] == horizon][cols].sort_values("MAE").reset_index(drop=True)
    st.dataframe(tbl.round(1), width="stretch", hide_index=True)
    st.caption("MAPE only uses intervals where the actual value is above 0. Coverage is the share of "
               "actual values inside the 90% interval (the target is 90%).")

    fig2 = go.Figure()
    for m_name, g in mt.groupby("model"):
        g = g.sort_values("horizon_min")
        fig2.add_trace(go.Scatter(x=[horizon_label(h) for h in g["horizon_min"]], y=g["MAE"],
                                  mode="lines+markers", name=m_name,
                                  line=dict(width=4 if m_name == model else 2)))
    fig2.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="MAE (tickets)",
                       xaxis_title="Forecast horizon", hovermode="x unified")
    st.plotly_chart(fig2, width="stretch")

    # st.subheader("Key performance indicators")
    # kt = kpis[kpis["target"] == target].drop(columns="target").sort_values("Forecast Accuracy %", ascending=False)
    # st.dataframe(kt.round(1), width="stretch", hide_index=True)
    # with st.expander("How the KPIs are defined"):
    #     st.markdown(
    #         "- **Forecast Accuracy %**: 100 minus the weighted absolute percentage error, averaged over horizons.\n"
    #         "- **Error Drift (pp)**: change in that percentage error from the 15-minute to the 2-hour horizon. "
    #         "Negative values mean longer windows are easier to forecast in relative terms.\n"
    #         "- **Peak Miss Rate %**: share of high-demand windows (top 10% of training windows) where the "
    #         "forecast was below 75% of the actual value.\n"
    #         "- **Confidence Band Width (rel %)**: average width of the 90% interval relative to average demand.\n"
    #         "- **Forecast Lead Time (min)**: longest horizon at which at least 70% of peaks are still caught.")
    st.subheader(f"Key performance indicators ({horizon_label(horizon)} horizon)")
    kh = mt[mt["horizon_min"] == horizon].set_index("model")
    kt = (kpis[kpis["target"] == target].set_index("model")[["Error Drift (pp)", "Forecast Lead Time (min)"]]
          .join(kh[["Forecast Accuracy %", "Peak Miss Rate %", "Relative Band Width %"]])
          .rename(columns={"Relative Band Width %": "Confidence Band Width (rel %)"}))
    kt = kt[["Forecast Accuracy %", "Error Drift (pp)", "Peak Miss Rate %",
             "Confidence Band Width (rel %)", "Forecast Lead Time (min)"]]
    kt = kt.sort_values("Forecast Accuracy %", ascending=False).reset_index()
    st.dataframe(kt.round(1), width="stretch", hide_index=True)
    st.caption("Accuracy, peak miss rate and band width follow the selected horizon and cover the whole 2025 test year. "
               "Error drift and lead time compare horizons, so they always use all four.")
    with st.expander("How the KPIs are defined"):
        st.markdown(
            "- **Forecast Accuracy %**: 100 minus the weighted absolute percentage error, for the selected horizon.\n"
            "- **Error Drift (pp)**: change in that percentage error from the 15-minute to the 2-hour horizon. "
            "Negative values mean longer windows are easier to forecast in relative terms. Always uses all horizons.\n"
            "- **Peak Miss Rate %**: share of high-demand windows (top 10% of training windows) where the "
            "forecast was below 75% of the actual value, for the selected horizon.\n"
            "- **Confidence Band Width (rel %)**: average width of the 90% interval relative to average demand, "
            "for the selected horizon.\n"
            "- **Forecast Lead Time (min)**: longest horizon at which at least 70% of peaks are still caught. "
            "Always uses all horizons.")
# ---------- tab 3: latest forecast ----------
with tab_live:
    series = load_series()
    last = series.index[-1]
    st.subheader(f"Forecast from the latest data point ({last:%d %b %Y, %H:%M})")
    if model not in ML_MODELS:
        st.info("The latest forecast uses the saved machine-learning models. Choose Linear Regression, "
                "Random Forest, Gradient Boosting or XGBoost in the sidebar.")
    else:
        y = series[target].tail(300)
        X = build_features(y).iloc[[-1]]
        rows = []
        for h in HORIZONS:
            st.write("Loading model:", target, h, model)
            bundle = load_model(target, h, model)
            p = max(float(bundle["model"].predict(X[bundle["features"]])[0]), 0.0)
            lo, hi = bundle["calibrator"].predict_interval([p])
            rows.append({"Horizon": horizon_label(h), "Window": f"{last:%H:%M} to {last + pd.Timedelta(minutes=h):%H:%M}",
                         "Forecast": p, "Lower": float(lo[0]), "Upper": float(hi[0])})
        live = pd.DataFrame(rows)
        fig3 = go.Figure(go.Bar(x=live["Horizon"], y=live["Forecast"], marker_color="#1f77b4",
                                error_y=dict(type="data", symmetric=False,
                                             array=live["Upper"] - live["Forecast"],
                                             arrayminus=live["Forecast"] - live["Lower"])))
        fig3.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                           yaxis_title="Forecast tickets in window (with 90% interval)")
        st.plotly_chart(fig3, width="stretch")
        st.dataframe(live.round(0), width="stretch", hide_index=True)
        st.caption("Demonstration forecast from the last record in the dataset. Connect the app to a live "
                   "ticket feed to run it on current data.")

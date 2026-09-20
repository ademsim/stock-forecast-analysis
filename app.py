import warnings
from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Stock Forecast (ARIMA)")


@st.cache_data(ttl=3600)
def load_prices(ticker, start):
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


st.title("Stock Price Forecast with ARIMA")
st.write(
    "Download daily prices with `yfinance`, check if the series is stationary, "
    "and forecast the next trading days with an ARIMA model."
)

col1, col2, col3 = st.columns(3)
ticker = col1.text_input("Ticker", "AAPL").strip().upper()
start = col2.date_input("Start date", date.today() - timedelta(days=5 * 365))
horizon = col3.slider("Forecast horizon (trading days)", 5, 60, 30)

model_name = st.selectbox(
    "Model",
    ["ARIMA(0,1,0) + drift (random walk with drift)", "ARIMA(1,1,1) + drift"],
)
order = (0, 1, 0) if model_name.startswith("ARIMA(0") else (1, 1, 1)

if st.button("Forecast"):
    data = load_prices(ticker, start)
    if data.empty or "Close" not in data or len(data) < 100:
        st.error("Not enough data for this ticker and date range. Try another ticker or an earlier start date.")
        st.stop()

    close = data["Close"].dropna()
    dates = close.index
    y = close.reset_index(drop=True)

    st.subheader(f"{ticker} closing price")
    st.line_chart(close)

    p_level = adfuller(y)[1]
    p_diff = adfuller(y.diff().dropna())[1]
    st.write(
        f"**ADF stationarity test:** price p-value = {p_level:.3f} "
        f"({'stationary' if p_level < 0.05 else 'not stationary'}), "
        f"first difference p-value = {p_diff:.3f} "
        f"({'stationary' if p_diff < 0.05 else 'not stationary'}). "
        "That is why the model uses d = 1."
    )

    fit = ARIMA(y, order=order, trend="t").fit()
    fc = fit.get_forecast(horizon)
    mean = fc.predicted_mean
    ci = fc.conf_int()
    future = pd.bdate_range(dates[-1] + pd.Timedelta(days=1), periods=horizon)

    st.subheader(f"{horizon}-day forecast")
    fig, ax = plt.subplots(figsize=(10, 4))
    recent = close.iloc[-120:]
    ax.plot(recent.index, recent.values, label="Actual")
    ax.plot(future, mean.values, label="Forecast")
    ax.fill_between(future, ci.iloc[:, 0].values, ci.iloc[:, 1].values, alpha=0.2, label="95% interval")
    ax.legend()
    ax.set_ylabel("Price")
    st.pyplot(fig)

    last = float(y.iloc[-1])
    st.write(
        f"Last close: **{last:.2f}** | forecast for the last day: **{float(mean.iloc[-1]):.2f}** "
        f"(95% interval: {float(ci.iloc[-1, 0]):.2f} to {float(ci.iloc[-1, 1]):.2f})"
    )

    with st.expander("Backtest: is the model better than 'tomorrow = today'?"):
        n_test = 60
        train = y.iloc[:-n_test]
        test = y.iloc[-n_test:]
        bt = ARIMA(train, order=order, trend="t").fit().apply(y)
        pred = bt.get_prediction(start=len(train), dynamic=False).predicted_mean
        naive = y.shift(1).iloc[-n_test:]
        mae_model = float(np.mean(np.abs(pred.values - test.values)))
        mae_naive = float(np.mean(np.abs(naive.values - test.values)))
        st.write(f"1-day-ahead MAE on the last {n_test} days: model **{mae_model:.2f}**, naive **{mae_naive:.2f}**.")
        st.caption(
            "For most stocks the two are almost the same: prices behave like a random walk, "
            "so ARIMA cannot beat the naive forecast in a meaningful way."
        )

st.caption(
    "The forecast interval assumes constant volatility and normal returns, so real price moves can fall "
    "outside it. This app is for learning time series techniques and is not investment advice."
)

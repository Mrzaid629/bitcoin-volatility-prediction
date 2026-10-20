
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

st.set_page_config(
    page_title="Bitcoin Volatility Analytics",
    page_icon="₿",
    layout="wide"
)

st.title("₿ Bitcoin Volatility Analytics")
st.caption("Historical analysis and 30-day volatility forecasting")

@st.cache_data(ttl=3600)
def load_data():
    data = yf.download(
        "BTC-USD",
        start="2014-09-17",
        end="2026-02-01",
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        raise ValueError("No Bitcoin data was downloaded.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()
    data = data[["Date", "Close"]].copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data = data.dropna().sort_values("Date").reset_index(drop=True)

    data["Log_Return"] = np.log(
        data["Close"] / data["Close"].shift(1)
    )
    data = data.dropna().reset_index(drop=True)

    data["Historical_Vol"] = (
        data["Log_Return"].rolling(30).std()
        * np.sqrt(365) * 100
    )

    data["Realized_Vol"] = np.sqrt(
        (data["Log_Return"] ** 2).rolling(30).sum()
        * (365 / 30)
    ) * 100

    lam = 0.94
    squared = data["Log_Return"] ** 2
    variance = np.zeros(len(data))
    variance[0] = squared.iloc[0]

    for i in range(1, len(data)):
        variance[i] = (
            lam * variance[i - 1]
            + (1 - lam) * squared.iloc[i - 1]
        )

    data["EWMA_Vol"] = np.sqrt(variance) * np.sqrt(365) * 100

    garch_fit = arch_model(
        data["Log_Return"] * 100,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="normal"
    ).fit(disp="off")

    data["GARCH_Vol"] = (
        garch_fit.conditional_volatility * np.sqrt(365)
    )

    return data, garch_fit


@st.cache_resource
def train_lstm(values):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values.reshape(-1, 1))

    lookback = 30
    X, y = [], []

    for i in range(len(scaled) - lookback):
        X.append(scaled[i:i + lookback])
        y.append(scaled[i + lookback])

    X = np.array(X)
    y = np.array(y)

    model = Sequential([
        Input(shape=(lookback, 1)),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    model.fit(
        X, y,
        epochs=20,
        batch_size=32,
        verbose=0
    )

    sequence = scaled[-lookback:].reshape(1, lookback, 1)
    predictions = []

    for _ in range(30):
        prediction = model.predict(sequence, verbose=0)[0, 0]
        predictions.append(prediction)

        sequence = np.concatenate(
            [sequence[:, 1:, :],
             np.array(prediction).reshape(1, 1, 1)],
            axis=1
        )

    forecast = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    ).flatten()

    return forecast


with st.spinner("Loading Bitcoin data and fitting GARCH model..."):
    try:
        btc, garch_fit = load_data()
    except Exception as e:
        st.error(f"Unable to load or process data: {e}")
        st.stop()

latest = btc.iloc[-1]
valid = btc.dropna(subset=["Historical_Vol"]).copy()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Latest available close", f"${latest['Close']:,.2f}")
c2.metric("Daily return", f"{latest['Log_Return'] * 100:.2f}%")
c3.metric("Historical volatility", f"{latest['Historical_Vol']:.2f}%")
c4.metric("Data through", latest["Date"].strftime("%d %b %Y"))

st.info(
    "Annualized volatility estimates are expressed as percentages. "
    "These are volatility forecasts, not Bitcoin price predictions."
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Volatility Analysis",
        "GARCH Forecast",
        "LSTM Forecast",
        "About Project"
    ]
)

if page == "Dashboard":
    st.subheader("Bitcoin Closing Price")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(btc["Date"], btc["Close"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing price (USD)")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Recent Market Data")
    st.dataframe(
        btc[["Date", "Close", "Log_Return"]].tail(10),
        use_container_width=True,
        hide_index=True
    )

elif page == "Volatility Analysis":
    st.subheader("Compare Volatility Estimation Methods")

    period = st.selectbox(
        "Choose period",
        ["Full Dataset", "2023", "2024", "2025"]
    )

    if period == "Full Dataset":
        view = valid
    else:
        view = valid[valid["Date"].dt.year == int(period)]

    fig, ax = plt.subplots(figsize=(12, 5))

    for col, label in [
        ("Historical_Vol", "Historical Volatility"),
        ("Realized_Vol", "Realized Volatility"),
        ("EWMA_Vol", "EWMA Volatility"),
        ("GARCH_Vol", "GARCH Volatility")
    ]:
        ax.plot(view["Date"], view[col], label=label, linewidth=1)

    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized volatility (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Historical and realized volatility use a 30-day window. "
        "EWMA uses lambda = 0.94, while GARCH uses a GARCH(1,1) model."
    )

elif page == "GARCH Forecast":
    st.subheader("GARCH — Next 30 Days")

    forecast_result = garch_fit.forecast(
        horizon=30,
        reindex=False
    )

    forecast_vol = (
        np.sqrt(forecast_result.variance.values[-1])
        * np.sqrt(365)
    )

    forecast_df = pd.DataFrame({
        "Forecast Day": np.arange(1, 31),
        "Annualized Volatility (%)": forecast_vol
    })

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(
        forecast_df["Forecast Day"],
        forecast_df["Annualized Volatility (%)"],
        marker="o"
    )
    ax.set_xlabel("Forecast day")
    ax.set_ylabel("Annualized volatility (%)")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(forecast_df, use_container_width=True, hide_index=True)

elif page == "LSTM Forecast":
    st.subheader("LSTM — Next 30 Days")

    if valid["Historical_Vol"].empty:
        st.error("Not enough valid data to train the LSTM model.")
        st.stop()

    with st.spinner("Training LSTM and generating forecasts..."):
        try:
            values = valid["Historical_Vol"].to_numpy(dtype=float)
            prediction = train_lstm(values)
        except Exception as e:
            st.error(f"LSTM training failed: {e}")
            st.stop()

    m1, m2, m3 = st.columns(3)
    m1.metric("Day 1 forecast", f"{prediction[0]:.2f}%")
    m2.metric("Day 7 forecast", f"{prediction[6]:.2f}%")
    m3.metric("Day 30 forecast", f"{prediction[29]:.2f}%")

    forecast_df = pd.DataFrame({
        "Forecast Day": np.arange(1, 31),
        "Predicted Volatility (%)": prediction
    })

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(
        forecast_df["Forecast Day"],
        forecast_df["Predicted Volatility (%)"],
        marker="o"
    )
    ax.set_xlabel("Forecast day")
    ax.set_ylabel("Predicted annualized volatility (%)")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(forecast_df, use_container_width=True, hide_index=True)

    st.warning(
        "These are recursive LSTM forecasts, not guaranteed outcomes. "
        "The model should be evaluated on held-out data before its "
        "forecast accuracy is claimed."
    )

elif page == "About Project":
    st.subheader("Project Overview")

    st.markdown("""
    **Title:** Bitcoin Volatility Prediction Using GARCH and LSTM

    **Objective:** Analyze historical Bitcoin volatility and forecast
    future volatility using statistical and deep-learning methods.

    **Methods used:**
    - Historical volatility using a rolling 30-day standard deviation.
    - Realized volatility using squared log returns.
    - EWMA volatility with lambda = 0.94.
    - GARCH(1,1) conditional volatility estimation.
    - LSTM neural network using the previous 30 volatility observations.

    **Dataset:** BTC-USD historical closing prices obtained through Yahoo
    Finance, using the date range configured in the notebook.

    **Important limitation:** Volatility measures the variability of returns.
    It does not tell us whether Bitcoin's price will rise or fall.
    """)

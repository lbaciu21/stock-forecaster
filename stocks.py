import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from xgboost import XGBRegressor
import requests


import requests
from io import StringIO

@st.cache_data
def get_sp500():

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers)

    html = StringIO(response.text)

    tables = pd.read_html(html)

    df = tables[0]

    return df["Symbol"].tolist()

@st.cache_data
def load_data(ticker):

    data = yf.download(ticker, period="2y")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.reset_index(inplace=True)

    return data



def create_lags(data, n_lags=10):

    df = data.copy()

    for i in range(1, n_lags + 1):
        df[f"lag_{i}"] = df["Close"].shift(i)

    df.dropna(inplace=True)

    X = df[[f"lag_{i}" for i in range(1, n_lags + 1)]]
    y = df["Close"]

    return X, y


def train_model(X, y):

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5
    )

    model.fit(X, y)

    return model



def forecast_future(model, data, days=30, n_lags=10):

    last_values = data["Close"].values[-n_lags:]

    preds = []

    for _ in range(days):

        pred = model.predict(last_values.reshape(1, -1))[0]

        preds.append(pred)

        last_values = np.append(last_values[1:], pred)

    return preds


def main():

    st.title("Stock Price Forecaster (XGBoost)")

    tickers = get_sp500()

    ticker = st.selectbox(
        "Select Stock",
        tickers,
        index=tickers.index("AAPL")
    )

    forecast_days = st.slider(
        "Days to Forecast",
        7,
        60,
        30
    )

    data = load_data(ticker)



    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["Close"],
            mode="lines",
            name="Historical Price"
        )
    )

    fig.update_layout(
        title=f"{ticker} Historical Price",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

 

    if st.button("Run Forecast"):

        X, y = create_lags(data)

        model = train_model(X, y)

        future_prices = forecast_future(model, data, forecast_days)

        future_dates = pd.date_range(
            start=data["Date"].iloc[-1],
            periods=forecast_days + 1
        )[1:]

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted": future_prices
        })

        fig2 = go.Figure()

        fig2.add_trace(
            go.Scatter(
                x=data["Date"],
                y=data["Close"],
                mode="lines",
                name="Historical"
            )
        )

        fig2.add_trace(
            go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["Predicted"],
                mode="lines",
                name="Forecast",
                line=dict(color="red")
            )
        )

        fig2.update_layout(
            title=f"{ticker} {forecast_days}-Day Forecast",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark"
        )

        st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(forecast_df)


if __name__ == "__main__":
    main()
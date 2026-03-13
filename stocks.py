'''import streamlit as st
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
///'''

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from xgboost import XGBRegressor
import requests
from io import StringIO
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch

# --- NEW: FinBERT Setup ---
@st.cache_resource
def load_finbert():
    # We use the ProsusAI/finbert model, which is the standard for financial NLP
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    # This creates a 'pipeline' for easy sentiment analysis
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return nlp

def get_sentiment(ticker, nlp_pipeline):
    # For this example, we'll fetch news from Finviz (no API key needed)
    try:
        url = f'https://finviz.com/quote.ashx?t={ticker}'
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = StringIO(response.text)
        # We look for the news table in the HTML
        news_table = pd.read_html(soup, attrs={'id': 'news-table'})[0]
        headlines = news_table[1].head(5).tolist() # Take the 5 latest headlines
        
        # Get sentiment for each headline
        results = nlp_pipeline(headlines)
        
        # FinBERT returns 'positive', 'negative', or 'neutral'
        # We convert this to a score: Positive = 1, Neutral = 0, Negative = -1
        scores = []
        for res in results:
            if res['label'] == 'positive':
                scores.append(res['score'])
            elif res['label'] == 'negative':
                scores.append(-res['score'])
            else:
                scores.append(0)
        
        return np.mean(scores) # Return average sentiment
    except:
        return 0 # Default to neutral if fetching fails

# --- EXISTING FUNCTIONS ---
@st.cache_data
def get_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    html = StringIO(response.text)
    tables = pd.read_html(html)
    return tables[0]["Symbol"].tolist()

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, period="2y")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.reset_index(inplace=True)
    return data

# --- UPDATED: Create Lags with Sentiment ---
def create_lags(data, sentiment_score, n_lags=10):
    df = data.copy()
    for i in range(1, n_lags + 1):
        df[f"lag_{i}"] = df["Close"].shift(i)
    
    # We add the sentiment score as a constant column for the training set
    # In a real production model, you would have historical daily sentiment.
    # Here, we apply current sentiment to help the model adjust the current trend.
    df["sentiment"] = sentiment_score
    
    df.dropna(inplace=True)
    features = [f"lag_{i}" for i in range(1, n_lags + 1)] + ["sentiment"]
    X = df[features]
    y = df["Close"]
    return X, y

def train_model(X, y):
    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5)
    model.fit(X, y)
    return model

# --- UPDATED: Forecast with Sentiment ---
def forecast_future(model, data, sentiment_score, days=30, n_lags=10):
    last_values = data["Close"].values[-n_lags:]
    preds = []
    for _ in range(days):
        # Prepare input: lags + the sentiment score
        input_data = np.append(last_values, sentiment_score).reshape(1, -1)
        pred = model.predict(input_data)[0]
        preds.append(pred)
        last_values = np.append(last_values[1:], pred)
    return preds

def main():
    st.title("Stock Price Forecaster (XGBoost + FinBERT)")
    
    # Load FinBERT pipeline
    nlp = load_finbert()
    
    tickers = get_sp500()
    ticker = st.selectbox("Select Stock", tickers, index=tickers.index("AAPL"))
    forecast_days = st.slider("Days to Forecast", 7, 60, 30)

    data = load_data(ticker)
    
    # Display Sentiment
    st.subheader("Market Sentiment Analysis")
    current_sentiment = get_sentiment(ticker, nlp)
    
    if current_sentiment > 0.2:
        st.success(f"Positive Sentiment Detected: {current_sentiment:.2f}")
    elif current_sentiment < -0.2:
        st.error(f"Negative Sentiment Detected: {current_sentiment:.2f}")
    else:
        st.info(f"Neutral Sentiment Detected: {current_sentiment:.2f}")

    # (Historical Plot code remains the same as yours)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], mode="lines", name="Historical Price"))
    fig.update_layout(title=f"{ticker} Historical Price", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    if st.button("Run Forecast"):
        # Training now includes the sentiment feature
        X, y = create_lags(data, current_sentiment)
        model = train_model(X, y)
        future_prices = forecast_future(model, data, current_sentiment, forecast_days)

        # (Forecast Plot code remains the same as yours)
        future_dates = pd.date_range(start=data["Date"].iloc[-1], periods=forecast_days + 1)[1:]
        forecast_df = pd.DataFrame({"Date": future_dates, "Predicted": future_prices})

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=data["Date"], y=data["Close"], mode="lines", name="Historical"))
        fig2.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Predicted"], mode="lines", name="Forecast", line=dict(color="red")))
        fig2.update_layout(title=f"{ticker} {forecast_days}-Day Forecast with Sentiment", template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(forecast_df)

if __name__ == "__main__":
    main()
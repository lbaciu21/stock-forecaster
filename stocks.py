

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import gc

from polygon import RESTClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from xgboost import XGBRegressor

st.set_page_config(page_title="Stock Forecaster", layout="wide")
st.title("Stocks Forecaster: XGBoost and FinBERT, API FROM Polygon")

@st.cache_resource(max_entries=1)
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

@st.cache_data(ttl=3600)
def get_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    return sorted(df["Symbol"].tolist())

@st.cache_data(ttl=1800)
def load_data(ticker, years=2):
    client = RESTClient()
    end = datetime.now()
    start = end - timedelta(days=365 * years + 100)
    aggs = list(client.list_aggs(ticker=ticker, multiplier=1, timespan="day", from_=start.strftime("%Y-%m-%d"), to=end.strftime("%Y-%m-%d"), limit=50000))
    df = pd.DataFrame([a.__dict__ for a in aggs])
    df = df.rename(columns={"timestamp": "Date", "close": "Close"})
    df["Date"] = pd.to_datetime(df["Date"], unit="ms").dt.date
    df = df.sort_values("Date").reset_index(drop=True)
    return df[["Date", "Close"]]

def get_sentiment(ticker, tokenizer, model):
    try:
        client = RESTClient()
        news = client.get_news(ticker=ticker, limit=15)
        scores = []
        for n in news:
            if not getattr(n, 'title', None):
                continue
            text = f"{n.title} {getattr(n, 'description', '') or ''}".strip()
            if not text:
                continue
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze().numpy()
            pos = probs[0]
            neg = probs[1]
            scores.append(pos - neg)
        return np.mean(scores) if scores else 0.0
    except:
        return 0.0

def add_technical_indicators(df):
    df = df.copy()
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["Momentum"] = df["Close"].pct_change(periods=5)
    df["Volatility"] = df["Close"].rolling(window=10).std()
    df["MACD"] = df["SMA_5"] - df["SMA_20"]
    return df

def create_features(data, sentiment, n_lags=10):
    df = add_technical_indicators(data)
    for i in range(1, n_lags + 1):
        df[f"lag_{i}"] = df["Close"].shift(i)
    df["sentiment"] = sentiment
    df.dropna(inplace=True)
    features = [f"lag_{i}" for i in range(1, n_lags + 1)] + ["SMA_5", "SMA_20", "Momentum", "Volatility", "MACD", "sentiment"]
    df["Return"] = df["Close"].pct_change()
    df.dropna(inplace=True)
    return df[features], df["Return"], df["Close"]

tokenizer, model = load_finbert()
tickers = get_sp500()
ticker = st.selectbox("Select Asset to Analyze", tickers, index=tickers.index("AAPL") if "AAPL" in tickers else 0)
forecast_days = st.slider("Forecast Horizon (Days)", 7, 60, 30)

data = load_data(ticker)

with st.spinner("Calculating sentiment..."):
    current_sentiment = get_sentiment(ticker, tokenizer, model)
    gc.collect()

st.subheader(f"Sentiment Intelligence: {ticker}")
if current_sentiment > 0.15:
    st.success(f"BULLISH ({current_sentiment:.2f})")
elif current_sentiment < -0.15:
    st.error(f"BEARISH ({current_sentiment:.2f})")
else:
    st.info(f"NEUTRAL ({current_sentiment:.2f})")

fig = go.Figure()
fig.add_trace(go.Scatter(x=pd.to_datetime(data["Date"]), y=data["Close"], name="Price", line=dict(color="#00FFAA")))
fig.update_layout(title=f"{ticker} Trend", template="plotly_white", height=400)
st.plotly_chart(fig, use_container_width=True)

if st.button("Generate Forecast"):
    with st.spinner("Generating forecast..."):
        X, y_returns, close_prices = create_features(data, current_sentiment)
        
        model_xgb = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42)
        model_xgb.fit(X, y_returns)
        
        last_values = X.iloc[-1:].values[0].copy()
        last_close = close_prices.iloc[-1]
        preds_prices = []
        
        for i in range(forecast_days):
            decayed_sent = current_sentiment * (0.92 ** i)
            current_input = last_values.copy()
            current_input[-1] = decayed_sent
            ret_pred = model_xgb.predict(current_input.reshape(1, -1))[0]
            new_price = last_close * (1 + ret_pred)
            preds_prices.append(new_price)
            last_close = new_price
            if i > 5:
                recent_mean = np.mean(preds_prices[-6:])
                new_price = 0.85 * new_price + 0.15 * recent_mean
                preds_prices[-1] = new_price
                last_close = new_price
            last_values = np.roll(last_values, -1)
            last_values[-2] = new_price
        
        future_dates = pd.date_range(start=data["Date"].iloc[-1], periods=forecast_days + 1, freq='D')[1:]
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=pd.to_datetime(data["Date"]), y=data["Close"], name="Recent Actual", line=dict(color="#4A90E2")))
        fig2.add_trace(go.Scatter(x=future_dates, y=preds_prices, name="Forecasted Projection", line=dict(color="#E74C3C", dash="dash")))
        fig2.update_layout(title=f"{ticker} Price Forecast", template="plotly_white", height=500)
        st.plotly_chart(fig2, use_container_width=True)

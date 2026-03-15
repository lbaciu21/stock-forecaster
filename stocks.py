import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from xgboost import XGBRegressor
import requests
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import gc

POLYGON_KEY = "0WajjbfGwPgZuqg6JKTYQYbh3wJRDAc0" 

@st.cache_resource(max_entries=1)
def load_finbert():
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)

def get_sentiment_polygon(ticker, nlp):
    try:
        url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&limit=5&apiKey={POLYGON_KEY}"
        response = requests.get(url, timeout=5).json()
        
        headlines = [item['title'] for item in response.get('results', [])]
        if not headlines: return 0.0
        
        results = nlp(headlines)
        scores = [res['score'] if res['label'] == 'positive' else -res['score'] if res['label'] == 'negative' else 0 for res in results]
        gc.collect()
        return float(np.mean(scores))
    except:
        return 0.0

@st.cache_data
def get_ticker_list():
    return [
        "AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "GOOGL", "META", "NFLX", 
        "AMD", "INTC", "PYPL", "BABA", "V", "MA", "JPM", "DIS", "BA", "VTI"
    ]

@st.cache_data
def load_data(ticker):
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?adjusted=true&sort=asc&apiKey={POLYGON_KEY}"
        
        response = requests.get(url, timeout=10).json()
        if 'results' not in response: return pd.DataFrame()
            
        df = pd.DataFrame(response['results'])
        df = df.rename(columns={'c': 'Close', 't': 'Date'})
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        return df[['Date', 'Close']]
    except:
        return pd.DataFrame()

def main():
    st.set_page_config(page_title="AI Market Intelligence", layout="wide")
    st.title("Stocks Forecaster: XGBoost and FinBERT, API FROM Polygon")

    tickers = get_ticker_list()
    ticker = st.selectbox("Select Asset to Analyze", tickers, index=0)
    forecast_days = st.slider("Forecast Horizon (Days)", 7, 60, 30)

    data = load_data(ticker)
    if data.empty:
        st.error(f"Polygon API limit reached or {ticker} data unavailable. Please wait 1 minute.")
        return

    st.plotly_chart(go.Figure(data=[go.Scatter(x=data["Date"], y=data["Close"], name="Historical", line=dict(color="#00ffcc"))]).update_layout(template="plotly_dark", title=f"{ticker} Trend"), use_container_width=True)

    with st.spinner(f"Running FinBERT NLP for {ticker}..."):
        nlp = load_finbert()
    
        sentiment = get_sentiment_polygon(ticker, nlp)

    st.subheader(f"Sentiment Intelligence: {ticker}")
    if sentiment > 0.05: st.success(f"BULLISH ({sentiment:.2f})")
    elif sentiment < -0.05: st.error(f"BEARISH ({sentiment:.2f})")
    else: st.info(f"NEUTRAL ({sentiment:.2f})")

    if st.button("Generate Forecast"):
        try:
            df = data.copy()
            df["Returns"] = np.log(df["Close"] / df["Close"].shift(1))
            for i in range(1, 11): df[f"lag_{i}"] = df["Returns"].shift(i)
            df.dropna(inplace=True)
            
            features = [f"lag_{i}" for i in range(1, 11)] + ["sentiment"]
            model = XGBRegressor(n_estimators=100)
            model.fit(df[features], df["Returns"])

            last_returns = df["Returns"].tail(10).values
            last_price = data["Close"].iloc[-1]
            preds = []
            
            for i in range(forecast_days):
                sample = np.append(last_returns, sentiment * (0.9**i)).reshape(1, -1)
                ret = model.predict(sample)[0] + np.random.normal(0, df["Returns"].std() * 0.4)
                last_price *= np.exp(ret)
                preds.append(last_price)
                last_returns = np.append(last_returns[1:], ret)

            future_dates = pd.date_range(data["Date"].iloc[-1], periods=forecast_days + 1)[1:]
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=data["Date"].tail(25), y=data["Close"].tail(25), name="Recent Actual"))
            fig2.add_trace(go.Scatter(x=future_dates, y=preds, name="Forecasted Projection", line=dict(color="red", dash='dash')))
            st.plotly_chart(fig2.update_layout(template="plotly_dark"), use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
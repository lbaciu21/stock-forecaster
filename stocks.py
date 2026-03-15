'''import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from xgboost import XGBRegressor
import requests
from io import StringIO
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import gc 

@st.cache_resource(max_entries=1)
def load_finbert():
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)

def get_sentiment(ticker, nlp):
    try:
        url = f'https://finviz.com/quote.ashx?t={ticker}'
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = StringIO(response.text)
        news_table = pd.read_html(soup, attrs={'id': 'news-table'})[0]
        headlines = news_table[1].head(5).tolist() 
        results = nlp(headlines)
        scores = []
        for res in results:
            if res['label'] == 'positive': scores.append(res['score'])
            elif res['label'] == 'negative': scores.append(-res['score'])
            else: scores.append(0)
        avg_score = np.mean(scores)
        gc.collect() 
        return float(avg_score)
    except:
        return 0.0

@st.cache_data
def get_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    return pd.read_html(StringIO(res.text))[0]["Symbol"].tolist()

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, period="2y")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.reset_index(inplace=True)
    return data

def create_lags(data, sentiment, n_lags=10):
    df = data.copy()
    for i in range(1, n_lags + 1):
        df[f"lag_{i}"] = df["Close"].shift(i)
    df["sentiment"] = float(sentiment)
    df.dropna(inplace=True)
    features = [f"lag_{i}" for i in range(1, n_lags + 1)] + ["sentiment"]
    return df[features], df["Close"]

def main():
    st.set_page_config(page_title="Stock Forecaster AI", layout="wide")
    st.title("📈 Stock Forecaster (XGBoost + FinBERT)")
    
    st.markdown("""This model combines historical price data with Natural Language Processing to predict future trends.

**Disclaimer:** Take the predictions with at least one grain of salt.""")

    tickers = get_sp500()
    ticker = st.selectbox("Select Stock Ticker", tickers, index=tickers.index("AAPL"))
    forecast_days = st.slider("Forecast Window (Days)", 7, 60, 30)
    
    data = load_data(ticker)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Historical Price"))
    fig.update_layout(template="plotly_dark", title=f"{ticker} 2-Year Price History")
    st.plotly_chart(fig, use_container_width=True)

    with st.spinner("Initializing FinBERT AI..."):
        nlp = load_finbert()

    current_sentiment = get_sentiment(ticker, nlp)
    
    st.subheader(f"Market Sentiment for {ticker}")
    if current_sentiment > 0.1:
        st.success(f"Positive Sentiment: {current_sentiment:.2f}")
    elif current_sentiment < -0.1:
        st.error(f"Negative Sentiment: {current_sentiment:.2f}")
    else:
        st.info(f"Neutral Sentiment: {current_sentiment:.2f}")

    if st.button("Generate AI Forecast"):
        try:
            X, y = create_lags(data, current_sentiment)
            model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
            model.fit(X, y)
            last_values = data["Close"].values[-10:].tolist()  
            if len(last_values) < 10:
                st.error("Not enough historical data to generate forecast.")
                return

            for i in range(forecast_days):
                decay = current_sentiment * (0.9 ** i)
                combined = last_values + [decay] 
                inp = pd.DataFrame([combined], columns=feature_cols)
                
                p = model.predict(inp)[0]
                preds.append(p)
                
                last_values = last_values[1:] + [p]

            future_dates = pd.date_range(data["Date"].iloc[-1], periods=forecast_days+1)[1:]
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Historical"))
            fig2.add_trace(go.Scatter(x=future_dates, y=preds, name="AI Forecast", line=dict(color="red", width=3)))
            fig2.update_layout(template="plotly_dark", title=f"{ticker} Forecast Result")
            st.plotly_chart(fig2, use_container_width=True)

            st.table(pd.DataFrame({"Date": future_dates, "Predicted Price": preds}).head(10))
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")

if __name__ == "__main__":
    main()
  
'''
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from xgboost import XGBRegressor
import requests
from io import StringIO
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import gc

# FORGIVING SECRET LOADING
try:
    ALPHAVANTAGE_KEY = st.secrets["AV_KEY"]
except:
    st.warning("API Key not found in secrets.toml. Please add it to .streamlit/secrets.toml")
    ALPHAVANTAGE_KEY = None

@st.cache_resource(max_entries=1)
def load_finbert():
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)

def get_sentiment(ticker, nlp):
    try:
        url = f'https://finviz.com/quote.ashx?t={ticker}'
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        news_table = pd.read_html(StringIO(response.text), attrs={'id': 'news-table'})[0]
        headlines = news_table[1].head(5).tolist()
        results = nlp(headlines)
        scores = [res['score'] if res['label'] == 'positive' else -res['score'] if res['label'] == 'negative' else 0 for res in results]
        avg_score = np.mean(scores)
        gc.collect()
        return float(avg_score)
    except:
        return 0.0

@st.cache_data
def get_sp500():
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        return pd.read_html(StringIO(res.text))[0]["Symbol"].tolist()
    except:
        return ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"]

@st.cache_data
def load_data(ticker):
    if not ALPHAVANTAGE_KEY: return pd.DataFrame()
    try:
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=compact&apikey={ALPHAVANTAGE_KEY}&datatype=csv'
        data = pd.read_csv(url)
        if data.empty or "timestamp" not in data.columns: return pd.DataFrame()
        data = data.rename(columns={"timestamp": "Date", "close": "Close"}).sort_values("Date")
        data["Date"] = pd.to_datetime(data["Date"])
        return data
    except:
        return pd.DataFrame()

def main():
    st.set_page_config(page_title="AI Market Intelligence", layout="wide")
    st.title(" AI Market Intelligence Dashboard")
    
    tickers = get_sp500()
    ticker = st.selectbox("Target Asset Selection", tickers, index=tickers.index("AAPL") if "AAPL" in tickers else 0)
    forecast_days = st.slider("Prediction Horizon (Days)", 7, 60, 30)

    data = load_data(ticker)

    if data.empty:
        st.error("Waiting for valid API Key or Ticker data...")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Price", line=dict(color="#00ffcc")))
    fig.update_layout(template="plotly_dark", title=f"{ticker} Historical Trend")
    st.plotly_chart(fig, use_container_width=True)

    with st.spinner("Executing FinBERT Sentiment Analysis..."):
        nlp = load_finbert()
        sentiment = get_sentiment(ticker, nlp)

    st.subheader(f"Sentiment Intelligence for {ticker}")
    if sentiment > 0.1: st.success(f"BULLISH: {sentiment:.2f}")
    elif sentiment < -0.1: st.error(f"BEARISH: {sentiment:.2f}")
    else: st.info(f"NEUTRAL: {sentiment:.2f}")

    if st.button("Generate High-Fidelity AI Forecast"):
        try:
            df = data.copy()
            df["Returns"] = np.log(df["Close"] / df["Close"].shift(1))
            for i in range(1, 11):
                df[f"lag_{i}"] = df["Returns"].shift(i)
            df["sentiment"] = sentiment
            df.dropna(inplace=True)
            
            features = [f"lag_{i}" for i in range(1, 11)] + ["sentiment"]
            model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6)
            model.fit(df[features], df["Returns"])

            last_returns = df["Returns"].tail(10).tolist()
            last_price = data["Close"].iloc[-1]
            hist_vol = df["Returns"].std()
            
            preds = []
            for i in range(forecast_days):
                decay = sentiment * (0.9 ** i)
                inp = pd.DataFrame([[*last_returns, decay]], columns=features)
                pred_return = model.predict(inp)[0]
                
                # Volatility Randomizer
                noise = np.random.normal(0, hist_vol * 0.5) 
                final_return = pred_return + noise
                
                new_price = last_price * np.exp(final_return)
                preds.append(new_price)
                
                last_price = new_price
                last_returns = last_returns[1:] + [final_return]

            future_dates = pd.date_range(data["Date"].iloc[-1], periods=forecast_days + 1)[1:]
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=data["Date"].tail(20), y=data["Close"].tail(20), name="Actual"))
            fig2.add_trace(go.Scatter(x=future_dates, y=preds, name="AI Forecast", line=dict(color="red", dash='dash')))
            fig2.update_layout(template="plotly_dark", title="Multi-Factor AI Projection")
            st.plotly_chart(fig2, use_container_width=True)
            
        except Exception as e:
            st.error(f"Intelligence Failure: {e}")

if __name__ == "__main__":
    main()
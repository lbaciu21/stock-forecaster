import streamlit as st
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
            if res['label'] == 'positive':
                scores.append(res['score'])
            elif res['label'] == 'negative':
                scores.append(-res['score'])
            else:
                scores.append(0)
        
        avg_score = np.mean(scores)
        gc.collect() # Cleanup RAM after NLP processing
        return avg_score
    except:
        return 0

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
    
    # Add sentiment as a feature
    df["sentiment"] = sentiment
    df.dropna(inplace=True)
    
    features = [f"lag_{i}" for i in range(1, n_lags + 1)] + ["sentiment"]
    return df[features], df["Close"]


def main():
    st.set_page_config(page_title="Stock Forecaster AI", layout="wide")
    st.title(" Stock Forecaster (XGBoost + FinBERT)")
    st.markdown("This model combines historical price data with Natural Language Processing to predict future trends.")
    st.markdown("Disclaimer: Take the predictions with at least one grain of salt.")

    with st.spinner("Initializing FinBERT AI..."):
        nlp = load_finbert()

  
    st.sidebar.header("Parameters")
    tickers = get_sp500()
    ticker = st.sidebar.selectbox("Select Ticker", tickers, index=tickers.index("AAPL"))
    forecast_days = st.sidebar.slider("Forecast Window (Days)", 7, 60, 30)

    data = load_data(ticker)

  
    current_sentiment = get_sentiment(ticker, nlp)
    
    st.subheader(f"Latest Market Sentiment: {ticker}")
    if current_sentiment > 0.1:
        st.success(f"Positive Sentiment Detected: {current_sentiment:.2f}")
    elif current_sentiment < -0.1:
        st.error(f"Negative Sentiment Detected: {current_sentiment:.2f}")
    else:
        st.info(f"Neutral Sentiment Detected: {current_sentiment:.2f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Historical Price"))
    fig.update_layout(template="plotly_dark", title=f"{ticker} 2-Year History")
    st.plotly_chart(fig, use_container_width=True)

    if st.button("Generate AI Forecast"):
      
        X, y = create_lags(data, current_sentiment)
        model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
        model.fit(X, y)

     
        last_values = data["Close"].values[-10:]
        preds = []
        
        for i in range(forecast_days):
  
            decayed_sentiment = current_sentiment * (0.9 ** i)
        
            inp = np.append(last_values, decayed_sentiment).reshape(1, -1)
            
       
            p = model.predict(inp)[0]
            preds.append(p)
            
 
            last_values = np.append(last_values[1:], p)


        future_dates = pd.date_range(data["Date"].iloc[-1], periods=forecast_days+1)[1:]
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Historical"))
        fig2.add_trace(go.Scatter(x=future_dates, y=preds, name="AI Forecast", line=dict(color="red", width=3)))
        fig2.update_layout(template="plotly_dark", title="Forecast Result (Includes Sentiment Decay)")
        st.plotly_chart(fig2, use_container_width=True)
        

        forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Price": preds})
        st.table(forecast_df.head(10))

if __name__ == "__main__":
    main()
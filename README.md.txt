Stock Forecaster App

A simple python app which predicts prices using ML, more precisely XGBOOST, and FinBERT, allowing an interactive exploration with Streamlit.


How it works
1. Historical Data: Fetches 2 years of price data via the yfinance library.
2. AI Sentiment: Uses FinBERT (Natural Language Processing) to scan the latest news headlines from Finviz.
3. Prediction: Trains an XGBoost Regressor on previous prices and the current sentiment score.
4. Decay Logic: The model includes a decay feature where the impact of today's news gradually fades over the 30-60 day forecast period.

Tech Stack
Python
Streamlit (Web Interface)
XGBoost (Machine Learning)
Transformers 
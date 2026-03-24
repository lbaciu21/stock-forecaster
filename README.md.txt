Stock Forecaster App

A simple python app which predicts prices using ML, more precisely XGBOOST, and FinBERT, allowing an interactive exploration with Streamlit.


How it works
Historical Data: Fetches 2 years of price data using Polygon.io API
Sentiment: Uses FinBERT (Natural Language Processing) to analyze recent news headlines and descriptions, in order to generate a sentiment score
Features: Created lagged prices, technical indicators and incorporated sentiment data
Model: Trains an XGBoost Regressor on previous prices and the current sentiment score.
Forcast: Uses iterated one step ahead prediction with sentiment decay and error correction

Tech Stack
Python
Streamlit (Web Interface)
XGBoost (Machine Learning)
FinBERT (Sentiment Analysis via Hugging Face)
Plygon.io (Market data and news)
Plotly (Charts)
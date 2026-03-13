Stock Forecaster App

A simple python app which predicts prices using ML, more precisely, XGBOOST, and allows an interactive exploration with Streamlit.

The features are:
- Load historical stock data for S&P500
- Train XGBoost Model
- Visualize the predictions
- Interactive interface with the help of Streamlit

Installation
1. Clone the repo:
   ```bash
   git clone <your-repo-url>
   cd <project-folder>

2. Create a virtual environment
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows
venv\Scripts\activate

3. Install dependencies:
pip install -r requirements.txt

Run the app with: streamlit run app.py
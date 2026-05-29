import streamlit as st
import pandas as pd
import yfinance as yf
import pickle
import os
from tensorflow.keras.models import load_model

# --- 1. Page Configuration ---
st.set_page_config(page_title="CryptoScan: Institutional AI Terminal", layout="wide")
st.markdown("""
    <style>
    .main {background-color: #0b0e11;}
    .stMetric {background-color: #1e2329; padding: 20px; border-radius: 10px; border: 1px solid #333;}
    h1 {color: #f0b90b;}
    </style>
""", unsafe_allow_html=True)

st.title("⚡ CryptoScan: Institutional AI Terminal")

# --- 2. Load Assets ---
@st.cache_resource
def load_assets():
    model_path = os.path.join(os.getcwd(), 'models', 'lstm_model.h5')
    scaler_path = os.path.join(os.getcwd(), 'models', 'scaler.pkl')
    
    # Check if files exist
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
        
    model = load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_assets()

# --- 3. Data Fetching ---
@st.cache_data(ttl=300)
def get_data(ticker):
    df = yf.download(ticker, period="3mo", interval="1d")
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.get_level_values(0)
    return df

ticker = st.sidebar.selectbox("Market Asset", ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"])
df = get_data(ticker)
current_price = float(df['Close'].iloc[-1])

# --- 4. Dashboard Metrics ---
col1, col2, col3 = st.columns(3)
col1.metric("Live Market Price", f"${current_price:,.2f}")
col2.metric("Market Sentiment", "Bullish" if current_price > df['Close'].mean() else "Bearish")
col3.metric("Terminal Status", "Live Connection")

st.line_chart(df['Close'].tail(60))

# --- 5. Professional Actions ---
c1, c2 = st.columns(2)

# AI Forecasting (Next Day)
if c1.button("RUN: AI PREDICTION (24H)"):
    if model is None or scaler is None:
        st.error("Error: AI Model or Scaler not found. Please check the /models directory.")
    else:
        seq = df['Close'].values[-60:].reshape(-1, 1)
        scaled = scaler.transform(seq)
        pred = model.predict(scaled.reshape(1, 60, 1))
        raw_pred = float(scaler.inverse_transform(pred)[0][0])
        
        st.subheader("AI Forecast Report")
        st.metric("Predicted Next Day Close", f"${raw_pred:,.2f}")
        st.success("Logic: Hybrid LSTM Model analysis completed.")

# PAST 7 DAYS ANALYSIS
if c2.button("RUN: PAST 7 DAYS ANALYSIS"):
    past_7_days = df.tail(7)[['Close']]
    
    st.subheader("Market Performance: Last 7 Days")
    st.line_chart(past_7_days)
    st.table(past_7_days.sort_index(ascending=False))
    
    avg_price = past_7_days['Close'].mean()
    st.info(f"Status: Historical Analysis Complete. 7-Day Average: ${avg_price:,.2f}")

st.markdown("---")
st.caption("Institutional Trading Interface | © 2026 Data Science Internship")

import streamlit as st
import pandas as pd
import yfinance as yf
import pickle
import os
import numpy as np

# Try to import TensorFlow safely
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

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
    # फाईल्स रूट डिरेक्टरीमध्ये आहेत हे सुनिश्चित करणे
    model_path = os.path.join(os.getcwd(), "lstm_model.h5")
    scaler_path = os.path.join(os.getcwd(), "scaler.pkl")
    
    model = None
    scaler = None
    
    if TF_AVAILABLE and os.path.exists(model_path):
        model = load_model(model_path)
        
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            
    return model, scaler

model, scaler = load_assets()

# --- 3. Data Fetching ---
@st.cache_data(ttl=300)
def get_data(ticker):
    try:
        # yfinance साठी अधिक स्टेबल पद्धत
        df = yf.download(ticker, period="3mo", interval="1d")
        if df.empty: return None
        return df
    except Exception as e:
        return None

ticker = st.sidebar.selectbox("Market Asset", ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"])
df = get_data(ticker)

if df is not None:
    # yfinance च्या नवीन व्हर्जनमध्ये डेटा फ्रेम ॲक्सेस करण्याची पद्धत
    current_price = float(df['Close'].iloc[-1].item())

    # --- 4. Dashboard Metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Live Market Price", f"${current_price:,.2f}")
    col2.metric("Market Sentiment", "Bullish" if current_price > float(df['Close'].mean().item()) else "Bearish")
    col3.metric("Terminal Status", "Live Connection")

    st.line_chart(df['Close'].tail(60))

    # --- 5. Professional Actions ---
    c1, c2 = st.columns(2)

    # AI Forecasting
    if c1.button("RUN: AI PREDICTION (24H)"):
        if model is None or scaler is None:
            st.error(f"AI Model or Scaler not found! Path: {os.getcwd()}")
        else:
            try:
                # डेटा तयार करणे
                seq = df['Close'].values[-60:].reshape(-1, 1)
                scaled = scaler.transform(seq)
                pred = model.predict(scaled.reshape(1, 60, 1))
                raw_pred = float(scaler.inverse_transform(pred)[0][0])
                
                st.subheader("AI Forecast Report")
                st.metric("Predicted Next Day Close", f"${raw_pred:,.2f}")
                st.success("Logic: Hybrid LSTM Model analysis completed.")
            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")

    # Past 7 Days Analysis
    if c2.button("RUN: PAST 7 DAYS ANALYSIS"):
        past_7_days = df.tail(7)[['Close']]
        st.subheader("Market Performance: Last 7 Days")
        st.line_chart(past_7_days)
        st.table(past_7_days.sort_index(ascending=False))
        avg_price = past_7_days['Close'].mean().item()
        st.info(f"Status: Historical Analysis Complete. 7-Day Average: ${avg_price:,.2f}")
else:
    st.error("Could not fetch market data. Please try again later.")

st.markdown("---")
st.caption("Institutional Trading Interface | © 2026 Data Science Internship")

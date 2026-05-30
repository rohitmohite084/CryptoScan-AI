import streamlit as st
import pandas as pd
import yfinance as yf
import pickle
import os
import numpy as np

# TensorFlow चा सुरक्षित वापर
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

st.set_page_config(page_title="CryptoScan: Institutional AI Terminal", layout="wide")
st.markdown("""
    <style>
    .main {background-color: #0b0e11;}
    .stMetric {background-color: #1e2329; padding: 20px; border-radius: 10px; border: 1px solid #333;}
    h1 {color: #f0b90b;}
    </style>
""", unsafe_allow_html=True)

st.title("⚡ CryptoScan: Institutional AI Terminal")

# --- Asset Loading ---
@st.cache_resource
def load_assets():
    model = None
    scaler = None
    if TF_AVAILABLE and os.path.exists("lstm_model.h5"):
        try:
            model = load_model("lstm_model.h5")
        except:
            model = None
    if os.path.exists("scaler.pkl"):
        try:
            with open("scaler.pkl", 'rb') as f:
                scaler = pickle.load(f)
        except:
            scaler = None
    return model, scaler

model, scaler = load_assets()

# --- Data Fetching ---
@st.cache_data(ttl=300)
def get_data(ticker):
    try:
        df = yf.download(ticker, period="3mo", interval="1d", progress=False)
        return df if not df.empty else None
    except:
        return None

ticker = st.sidebar.selectbox("Market Asset", ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"])
df = get_data(ticker)

if df is not None:
    # सुरक्षित डेटा प्रोसेसिंग
    try:
        current_price = float(df['Close'].iloc[-1].item())
        avg_price = float(df['Close'].mean().item())
    except:
        current_price = float(df['Close'].iloc[-1])
        avg_price = float(df['Close'].mean())

    col1, col2, col3 = st.columns(3)
    col1.metric("Live Market Price", f"${current_price:,.2f}")
    col2.metric("Market Sentiment", "Bullish" if current_price > avg_price else "Bearish")
    col3.metric("Terminal Status", "Live Connection")

    st.line_chart(df['Close'].tail(60))

    c1, c2 = st.columns(2)
    
    if c1.button("RUN: AI PREDICTION (24H)"):
        if model and scaler:
            try:
                last_60 = df['Close'].values[-60:].reshape(-1, 1)
                scaled = scaler.transform(last_60).reshape(1, 60, 1)
                pred = model.predict(scaled, verbose=0)
                raw_pred = float(scaler.inverse_transform(pred)[0][0])
                st.subheader("AI Forecast Report")
                st.metric("Predicted Next Day Close", f"${raw_pred:,.2f}")
            except Exception as e:
                st.error("Prediction failed.")
        else:
            st.warning("AI Model not loaded.")

    if c2.button("RUN: PAST 7 DAYS ANALYSIS"):
        past_7 = df.tail(7)[['Close']]
        st.subheader("Market Performance: Last 7 Days")
        st.line_chart(past_7)
        st.info(f"7-Day Average: ${past_7['Close'].mean():,.2f}")

st.markdown("---")
st.caption("Institutional Trading Interface | © 2026")

import streamlit as st
import pandas as pd
import yfinance as yf
import pickle
import os
import numpy as np

# TensorFlow साठी सुरक्षित तपासणी
TF_AVAILABLE = False
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

st.set_page_config(page_title="CryptoScan: AI Terminal", layout="wide")
st.title("⚡ CryptoScan: Institutional AI Terminal")

# --- 1. Load Assets ---
@st.cache_resource
def load_assets():
    scaler = None
    if os.path.exists("scaler.pkl"):
        try:
            with open("scaler.pkl", 'rb') as f:
                scaler = pickle.load(f)
        except:
            scaler = None
    return scaler

scaler = load_scaler()

# --- 2. Data Fetching ---
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
    # Key Error टाळण्यासाठी सुरक्षित एक्सेस
    price_col = 'Close'
    current_price = float(df[price_col].iloc[-1].item())
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Live Price", f"${current_price:,.2f}")
    col2.metric("Terminal Status", "Active")
    col3.metric("Mode", "Statistical Analysis")

    st.line_chart(df[price_col].tail(60))

    if st.button("RUN: PREDICTION"):
        if TF_AVAILABLE:
            st.info("AI Model is running...")
        else:
            st.warning("AI Model सध्या उपलब्ध नाही. (TensorFlow नाही).")
            st.write("सध्याच्या मार्केट ट्रेंडनुसार: किंमत ही सरासरीपेक्षा " + ("वर आहे." if current_price > df[price_col].mean().item() else "खाली आहे."))
else:
    st.error("डेटा लोड होऊ शकला नाही.")

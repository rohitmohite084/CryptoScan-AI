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

st.set_page_config(page_title="CryptoScan: Institutional AI Terminal", layout="wide")
st.title("⚡ CryptoScan: Institutional AI Terminal")

# --- 2. Load Assets (सुधारित) ---
@st.cache_resource
def load_assets():
    model = None
    scaler = None
    
    # मॉडेल लोड करण्याचा प्रयत्न
    try:
        if TF_AVAILABLE:
            if os.path.exists("lstm_model.h5"):
                model = load_model("lstm_model.h5")
            elif os.path.exists("my_saved_model"):
                model = load_model("my_saved_model")
    except Exception as e:
        st.error(f"Model Load Error: {e}")

    # स्केलर लोड करण्याचा प्रयत्न
    try:
        if os.path.exists("scaler.pkl"):
            with open("scaler.pkl", 'rb') as f:
                scaler = pickle.load(f)
    except Exception as e:
        st.error(f"Scaler Load Error: {e}")
            
    return model, scaler

model, scaler = load_assets()

# --- 3. Data Fetching ---
@st.cache_data(ttl=300)
def get_data(ticker):
    try:
        df = yf.download(ticker, period="3mo", interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
        return df
    except:
        return None

ticker = st.sidebar.selectbox("Market Asset", ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"])
df = get_data(ticker)

if df is not None:
    current_price = float(df['Close'].iloc[-1])
    col1, col2, col3 = st.columns(3)
    col1.metric("Live Market Price", f"${current_price:,.2f}")
    col2.metric("Market Sentiment", "Bullish" if current_price > df['Close'].mean() else "Bearish")
    col3.metric("Terminal Status", "Live Connection")
    st.line_chart(df['Close'].tail(60))

    if st.button("RUN: AI PREDICTION (24H)"):
        if model is None or scaler is None:
            st.error("मॉडेल किंवा स्केलर लोड होऊ शकले नाहीत. कृपया खात्री करा की फाइल्स करप्ट नाहीत.")
        else:
            try:
                last_60_days = df['Close'].values[-60:].reshape(-1, 1)
                scaled_data = scaler.transform(last_60_days)
                X_input = scaled_data.reshape(1, 60, 1)
                pred = model.predict(X_input, verbose=0)
                raw_pred = float(scaler.inverse_transform(pred)[0][0])
                
                st.subheader("AI Forecast Report")
                st.metric("Predicted Next Day Close", f"${raw_pred:,.2f}")
                st.success("Logic: Hybrid LSTM Model analysis completed.")
            except Exception as e:
                st.error(f"Prediction Calculation Error: {e}")

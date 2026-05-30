import streamlit as st
import pandas as pd
import yfinance as yf
import pickle
import os
import tensorflow as tf
import numpy as np

# --- 1. Page Configuration ---
st.set_page_config(page_title="CryptoScan: Institutional AI Terminal", layout="wide")
st.title("⚡ CryptoScan: Institutional AI Terminal")

# --- 2. Load Assets (Robust Loading) ---
@st.cache_resource
def load_assets():
    model = None
    scaler = None
    
    # १. मॉडेल लोड (TF SavedModel किंवा .h5)
    try:
        if os.path.exists("lstm_model.h5"):
            model = tf.keras.models.load_model("lstm_model.h5")
        elif os.path.exists("my_saved_model"):
            model = tf.saved_model.load("my_saved_model")
    except Exception as e:
        st.error(f"Model Load Error: {e}")

    # २. स्केलर लोड
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
        return df if not df.empty else None
    except:
        return None

ticker = st.sidebar.selectbox("Market Asset", ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"])
df = get_data(ticker)

if df is not None:
    # प्रेडिक्शन लॉजिक
    if st.button("RUN: AI PREDICTION (24H)"):
        if model is None or scaler is None:
            st.error("मॉडेल किंवा स्केलर लोड होऊ शकले नाहीत. कृपया खात्री करा की या फाइल्स ट्रेन केलेल्या मॉडेलच्या आहेत.")
        else:
            try:
                # ६० दिवसांचा डेटा घेणे
                last_data = df['Close'].values[-60:].reshape(-1, 1)
                scaled_data = scaler.transform(last_data)
                
                # मॉडेलसाठी योग्य शेप (1, 60, 1)
                X_input = scaled_data.reshape(1, 60, 1)
                
                # प्रेडिक्शन
                prediction = model.predict(X_input)
                
                # रिझल्ट
                raw_pred = float(scaler.inverse_transform(prediction)[0][0])
                st.metric("Predicted Next Day Close", f"${raw_pred:,.2f}")
            except Exception as e:
                st.error(f"Prediction logic error: {e}")

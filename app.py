import streamlit as st
import pandas as pd
import yfinance as yf
import pickle
import os
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

# --- 1. Page Configuration ---
st.set_page_config(page_title="CryptoScan: Institutional AI Terminal", layout="wide")

# CSS: Complete Bordered Dashboard & Premium Styling
st.markdown("""
    <style>
    /* Main Dashboard Border */
    .stApp {
        background-color: #0b0e11 !important;
        border: 4px solid #f0b90b;
        padding: 10px;
    }
    h1 {color: #f0b90b !important; text-align: center; font-weight: 800 !important; margin-bottom: 20px;}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e2329 0%, #0b0e11 100%) !important;
        border-right: 2px solid #f0b90b !important;
    }
    
    /* KPI Card Hover Effect */
    .kpi-card {
        background: #161a1e; padding: 20px; border-radius: 12px;
        border: 2px solid #333; text-align: center; transition: all 0.4s ease;
        cursor: pointer;
    }
    .kpi-card:hover {
        border: 2px solid #f0b90b;
        transform: scale(1.05);
        box-shadow: 0 0 15px rgba(240, 185, 11, 0.3);
    }
    
    /* Button Hover */
    div.stButton > button {
        border: 1px solid #f0b90b !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:hover {
        background-color: #f0b90b !important;
        color: black !important;
        transform: translateY(-2px);
    }
    
    .kpi-title {color: #888; font-size: 0.8rem; text-transform: uppercase;}
    .kpi-value {color: #f0b90b; font-size: 1.5rem; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

st.title("⚡ CryptoScan: Institutional AI Terminal")

# --- 2. Load Assets (Updated to look in 'models/' folder) ---
@st.cache_resource
def load_assets():
    model_path = os.path.join("models", "lstm_model.h5")
    scaler_path = os.path.join("models", "scaler.pkl")
    
    model = load_model(model_path) if os.path.exists(model_path) else None
    with open(scaler_path, "rb") as f: 
        scaler = pickle.load(f) if os.path.exists(scaler_path) else None
    return model, scaler

model, scaler = load_assets()

# --- 3. Sidebar ---
with st.sidebar:
    st.markdown("### 📊 MARKET CONTROLS")
    ticker = st.selectbox("Select Trading Asset", ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"])
    st.markdown("---")
    st.info("System Ready | Institutional Grade AI")

df = yf.download(ticker, period="3mo", interval="1d", progress=False)

if not df.empty:
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    last_val = float(df['Close'].iloc[-1].item() if hasattr(df['Close'].iloc[-1], 'item') else df['Close'].iloc[-1])
    
    # KPI Cards
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f'<div class="kpi-card"><div class="kpi-title">Live Price</div><div class="kpi-value">${last_val:,.2f}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="kpi-card"><div class="kpi-title">SMA 50</div><div class="kpi-value">${df["SMA_50"].iloc[-1]:,.2f}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="kpi-card"><div class="kpi-title">Trend</div><div class="kpi-value">{"Bullish" if last_val > df["SMA_50"].iloc[-1] else "Bearish"}</div></div>', unsafe_allow_html=True)

    st.write("<br>", unsafe_allow_html=True)

    # --- 4. Interactive Graph with Border ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='#f0b90b', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='#00d2ff', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='#ff4b4b', width=1)))
    
    fig.update_layout(
        plot_bgcolor='#0b0e11', paper_bgcolor='#0b0e11', font_color='white',
        xaxis=dict(showgrid=False, showline=True, linecolor='#f0b90b', linewidth=2),
        yaxis=dict(showgrid=True, gridcolor='#222', showline=True, linecolor='#f0b90b', linewidth=2),
        hovermode="x unified", margin=dict(t=30, b=30, l=30, r=30)
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- 5. AI & Risk Management ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🤖 AI Forecasting")
        if st.button("RUN: AI PREDICTION"):
            if model and scaler:
                scaled = scaler.transform(df['Close'].values[-60:].reshape(-1, 1))
                pred = model.predict(scaled.reshape(1, 60, 1), verbose=0)
                st.success(f"Forecast: ${scaler.inverse_transform(pred)[0][0]:,.2f}")
        if st.button("RUN: PAST 7 DAYS DATA"):
            st.dataframe(df.tail(7)[['Close']], use_container_width=True)
    
    with col2:
        st.subheader("💰 Risk Management")
        amt = st.number_input("Enter Investment ($)", min_value=100, value=1000)
        st.info(f"Recommended Exposure: ${amt * 0.02:,.2f} (Based on 2% Rule)")
        st.warning("⚠️ Never risk more than 2% of your capital.")

st.caption("© 2026 Institutional Trading Interface | Security Level: High")
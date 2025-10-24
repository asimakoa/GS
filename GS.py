# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 19:20:01 2025

@author: Nassos
"""

# -*- coding: utf-8 -*-
"""
Streamlit app to fetch GS (or BTCUSD) price from Polygon
and price a 3-month call option using the Black-76 model.
"""

import math
import time
import requests
import streamlit as st

# -----------------------------
# Utility functions
# -----------------------------
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def black76_call(S: float, K: float, sigma: float, T: float, r: float, q: float = 0.0) -> float:
    """Black-76 call price on an equity forward"""
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return float("nan")

    F = S * math.exp((r - q) * T)
    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    call = math.exp(-r * T) * (F * norm_cdf(d1) - K * norm_cdf(d2))
    return call

@st.cache_data(ttl=30)
def get_spot_polygon(ticker: str, api_key: str):
    """Try Polygon real-time trade; fallback to previous close"""
    headers = {"Accept": "application/json"}

    # --- Try real-time endpoint
    url_real = f"https://api.polygon.io/v2/last/trade/{ticker}?apiKey={api_key}"
    try:
        r = requests.get(url_real, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            spot = float(data["results"]["p"])
            ts = int(data["results"]["t"])
            return spot, "real-time (/v2/last/trade)", ts
    except Exception:
        pass

    # --- Fallback: previous close
    url_prev = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev?adjusted=true&apiKey={api_key}"
    r = requests.get(url_prev, headers=headers, timeout=10)
    r.raise_for_status()
    data = r.json()
    result = data["results"][0]
    spot = float(result["c"])
    ts = int(result["t"])
    return spot, "previous close (/v2/aggs/.../prev)", ts

def ms_to_local(ts_ms: int) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts_ms / 1000))
    except Exception:
        return str(ts_ms)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="GS Call Option Pricer", page_icon="📈")
st.title("📈 GS Call Option — 3M Black-76 Model")

ticker = st.text_input("Ticker (Polygon format)", value="GS", help="e.g. GS for Goldman Sachs, X:BTCUSD for Bitcoin/USD")
K = st.number_input("Strike (K)", value=400.0, step=1.0)
vol_pct = st.number_input("Volatility (%)", value=25.0, step=0.1)
r_pct = st.number_input("Risk-free rate (%)", value=5.0, step=0.1)
q_pct = st.number_input("Dividend yield (%)", value=2.0, step=0.1)

T = 3.0 / 12.0  # 3 months
st.caption(f"Expiry: {T:.4f} years (fixed 3 months)")

# --- Load API key from secrets
api_key = st.secrets["POLYGON_API_KEY"]

# --- Fetch spot
try:
    spot, source, ts_ms = get_spot_polygon(ticker, api_key)
    st.success(f"Spot for {ticker}: {spot:.2f} USD (source: {source}, time: {ms_to_local(ts_ms)})")
except Exception as e:
    st.error(f"Error fetching data from Polygon: {e}")
    st.stop()

# --- Compute option price
sigma = vol_pct / 100.0
r = r_pct / 100.0
q = q_pct / 100.0

price = black76_call(S=spot, K=K, sigma=sigma, T=T, r=r, q=q)

st.metric(label="💰 Call Option Price", value=f"{price:,.4f} USD")

with st.expander("Details"):
    st.write(f"""
**Model:** Black-76  
**Spot (S):** {spot:.4f}  
**Strike (K):** {K:.4f}  
**Volatility (σ):** {vol_pct:.2f}%  
**Expiry (T):** {T:.4f} years  
**r:** {r_pct:.2f}% **q:** {q_pct:.2f}%  
**Source:** {source}  
**Timestamp:** {ms_to_local(ts_ms)} ({ts_ms})
""")

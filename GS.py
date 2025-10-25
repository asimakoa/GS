# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 15:28:30 2025

@author: Nassos
"""

# -*- coding: utf-8 -*-
"""
Streamlit app:
- Fetches SPOT from Polygon (real-time trade if available, else previous close)
- Fetches latest US Treasury yield curve from Polygon and builds r(T) via interpolation/extrapolation
- Prices a call with Black-76. Only Strike (and vol/dividend) are user inputs; SPOT & r are locked to Polygon.

Secrets:
    st.secrets["POLYGON_API_KEY"]  <-- set in Streamlit Cloud
"""

import math
import time
import requests
import streamlit as st

# -----------------------------
# Math & Model
# -----------------------------
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def black76_call(S: float, K: float, sigma: float, T: float, r: float, q: float = 0.0) -> float:
    """Black-76 call price on an equity forward."""
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return float("nan")
    F = S * math.exp((r - q) * T)
    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return math.exp(-r * T) * (F * norm_cdf(d1) - K * norm_cdf(d2))

def ms_to_local(ts_ms: int) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts_ms / 1000))
    except Exception:
        return str(ts_ms)

# -----------------------------
# Polygon data fetchers
# -----------------------------
@st.cache_data(ttl=30)
def get_spot_polygon(ticker: str, api_key: str):
    """Try real-time trade; fallback to previous close."""
    headers = {"Accept": "application/json"}

    # Real-time last trade (works for equities & many crypto tickers)
    url_real = f"https://api.polygon.io/v2/last/trade/{ticker}?apiKey={api_key}"
    try:
        r = requests.get(url_real, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if "results" in data and "p" in data["results"]:
                spot = float(data["results"]["p"])
                ts = int(data["results"]["t"])
                return spot, "real-time (/v2/last/trade)", ts
    except Exception:
        pass

    # Previous close (aggregates)
    url_prev = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev?adjusted=true&apiKey={api_key}"
    r = requests.get(url_prev, headers=headers, timeout=10)
    r.raise_for_status()
    data = r.json()
    result = data["results"][0]
    spot = float(result["c"])
    ts = int(result["t"])
    return spot, "previous close (/v2/aggs/.../prev)", ts

def _key_to_years(k: str) -> float | None:
    """Map Polygon yield field names to year fractions."""
    # Accept keys like: yield_1_month, yield_3_month, yield_6_month,
    # yield_1_year, yield_2_year, ..., yield_30_year
    if not k.startswith("yield_"):
        return None
    tail = k.replace("yield_", "")
    if tail.endswith("_month"):
        n = tail.replace("_month", "")
        try:
            return int(n) / 12.0
        except:
            return None
    if tail.endswith("_year"):
        n = tail.replace("_year", "")
        try:
            return float(n)
        except:
            return None
    return None

@st.cache_data(ttl=600)
def get_latest_yield_curve(api_key: str):
    """
    Fetch the most recent US Treasury yields from Polygon and return:
      - curve: list of (tenor_years, rate_decimal) sorted ascending
      - curve_date: ISO date string
    """
    url = f"https://api.polygon.io/fed/v1/treasury-yields?limit=1&sort=date.desc&apiKey={api_key}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    results = data.get("results", [])
    if not results:
        raise RuntimeError("Polygon yields: empty results")

    rec = results[0]  # latest date
    curve_date = rec.get("date", "unknown-date")

    pts = []
    for k, v in rec.items():
        if v is None:
            continue
        t = _key_to_years(k)
        if t is not None:
            pts.append((float(t), float(v) / 100.0))  # % -> decimal

    pts = sorted(list({(round(t, 6), r) for t, r in pts}), key=lambda x: x[0])
    if len(pts) < 2:
        raise RuntimeError(f"Polygon yields: insufficient curve points ({len(pts)}) on {curve_date}")

    return pts, curve_date

def rate_from_curve(curve_pts: list[tuple[float, float]], T: float) -> float:
    """
    Linear interpolation/extrapolation on (tenor_years, rate_decimal).
    - If T lands exactly on a node, return node.
    - If T between nodes, interpolate.
    - If T < min or > max, extrapolate using nearest slope.
    """
    # Exact hit?
    for t, r in curve_pts:
        if abs(t - T) < 1e-9:
            return r

    # Before start or after end?
    if T <= curve_pts[0][0]:
        (t1, r1), (t2, r2) = curve_pts[0], curve_pts[1]
        return r1 + (r2 - r1) * (T - t1) / (t2 - t1)
    if T >= curve_pts[-1][0]:
        (t1, r1), (t2, r2) = curve_pts[-2], curve_pts[-1]
        return r1 + (r2 - r1) * (T - t1) / (t2 - t1)

    # In between
    for i in range(len(curve_pts) - 1):
        t1, r1 = curve_pts[i]
        t2, r2 = curve_pts[i + 1]
        if t1 <= T <= t2:
            if abs(t2 - t1) < 1e-12:
                return r1
            return r1 + (r2 - r1) * (T - t1) / (t2 - t1)

    # Fallback (shouldn't happen)
    return curve_pts[-1][1]

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="GS Call Option Pricer (Polygon rates)", page_icon="📈")
st.title("📈 GS Call Option — Black-76 with Polygon Spot & Treasury r(T)")

# Inputs (only what you allowed)
ticker = st.text_input(
    "Ticker (Polygon format)",
    value="GS",
    help="e.g. GS for Goldman Sachs, or X:BTCUSD for Bitcoin/USD"
)
K = st.number_input("Strike (K)", value=400.0, step=1.0)
vol_pct = st.number_input("Volatility (%)", value=25.0, step=0.1)
q_pct = st.number_input("Dividend yield (%)", value=2.0, step=0.1)

# Expiry: fixed 3 months (but the curve engine supports any T)
T = 3.0 / 12.0
st.caption(f"Expiry T = {T:.4f} years (fixed 3 months). Spot & r(T) come from Polygon; not user-editable.")

# Load API key
api_key = st.secrets["POLYGON_API_KEY"]

# Fetch spot
try:
    spot, spot_src, ts_ms = get_spot_polygon(ticker, api_key)
    st.success(f"Spot for {ticker}: {spot:.4f} (source: {spot_src}, time: {ms_to_local(ts_ms)})")
except Exception as e:
    st.error(f"Error fetching SPOT from Polygon: {e}")
    st.stop()

# Fetch curve and derive r(T)
try:
    curve_pts, curve_date = get_latest_yield_curve(api_key)
    r_T = rate_from_curve(curve_pts, T)
    st.info(f"r(T) pulled from Polygon Treasury curve dated {curve_date} → r({T:.4f}y) = {100*r_T:.4f}%")
except Exception as e:
    st.error(f"Error fetching Treasury yields from Polygon: {e}")
    st.stop()

# Price
sigma = vol_pct / 100.0
q = q_pct / 100.0
price = black76_call(S=spot, K=K, sigma=sigma, T=T, r=r_T, q=q)

st.metric(label="💰 Call Option Price", value=f"{price:,.4f}")

with st.expander("Details"):
    # Pretty-print first few curve points to show interpolation basis
    preview = "\n".join([f"T={t:.4f}y → r={100*r:.4f}%" for t, r in (curve_pts[:6] if len(curve_pts) > 6 else curve_pts)])
    st.write(f"""
**Model:** Black-76  
**Spot (S):** {spot:.6f} **Source:** {spot_src} at {ms_to_local(ts_ms)}  
**Strike (K):** {K:.6f}  
**Volatility (σ):** {vol_pct:.2f}%  
**Dividend yield (q):** {q_pct:.2f}%  
**Expiry (T):** {T:.6f} years  
**Risk-free r(T):** {100*r_T:.6f}% (Polygon Treasury, {curve_date})  

**Curve preview (first nodes):**  
{preview}
""")

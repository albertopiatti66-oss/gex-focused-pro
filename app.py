# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib import gridspec
import math
from datetime import datetime, timezone
import base64
from io import BytesIO

st.set_page_config(page_title="GEX Pro", layout="wide")

def bs_gamma(S, K, T, r, sigma):
    if any(v <= 0 for v in [S, K, T, sigma]): return 0.0
    try: d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    except: return 0.0
    return math.exp(-0.5*d1*d1) / (math.sqrt(2*math.pi) * S * sigma * math.sqrt(T))

def days_to_expiry(exp):
    e = datetime.strptime(exp, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return max((e - datetime.now(timezone.utc)).total_seconds() / 86400, 1.0)

@st.cache_data(ttl=300)
def compute_gex(symbol, range_pct=25, min_oi=0.4, dist_min=0.03, call_sign=1, put_sign=-1):
    try:
        tk = yf.Ticker(symbol)
        spot = tk.history(period="1d")["Close"].iloc[-1]
        expiry = tk.options[0]
        T = days_to_expiry(expiry) / 365
        chain = tk.option_chain(expiry)
        calls, puts = chain.calls, chain.puts
        for df in [calls, puts]:
            df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce").fillna(0)
            df["impliedVolatility"] = pd.to_numeric(df["impliedVolatility"], errors="coerce").fillna(0.3)
            df["strike"] = pd.to_numeric(df["strike"])

        lo, hi = spot*(1-range_pct/100), spot*(1+range_pct/100)
        calls = calls[(calls.strike >= lo) & (calls.strike <= hi)]
        puts = puts[(puts.strike >= lo) & (puts.strike <= hi)]

        calls["gamma"] = [bs_gamma(spot, k, T, 0.05, iv) for k, iv in zip(calls.strike, calls.impliedVolatility)]
        puts["gamma"] = [bs_gamma(spot, k, T, 0.05, iv) for k, iv in zip(puts.strike, puts.impliedVolatility)]
        calls["GEX"] = call_sign * calls.openInterest * 100 * calls.gamma * spot**2
        puts["GEX"] = put_sign * puts.openInterest * 100 * puts.gamma * spot**2

        max_oi = max(calls.openInterest.max(), puts.openInterest.max())
        calls = calls[calls.openInterest >= max_oi * min_oi]
        puts = puts[puts.openInterest >= max_oi * min_oi]

        gex = pd.concat([calls[["strike","GEX"]], puts[["strike","GEX"]]]).groupby("strike").sum().reset_index()
        gamma_call, gamma_put = calls.GEX.sum(), puts.GEX.sum()
        total = gamma_call + gamma_put
        dpi = gamma_call / total * 100 if total != 0 else 50
        flip = (calls.strike.mean()*gamma_call + puts.strike.mean()*gamma_put) / total if total != 0 else spot
        regime = "LONG" if spot > flip else "SHORT"

        fig = plt.figure(figsize=(14,7))
        gs = gridspec.GridSpec(2,1, height_ratios=[1,3])
        ax1 = fig.add_subplot(gs[0]); ax1.axis("off")
        ax1.text(0,0.8, f"{symbol} | Spot: {spot:.0f} | DPI: {dpi:.1f}% | Flip: {flip:.0f}", fontsize=12)
        ax1.text(0,0.4, f"Regime: {regime} | Exp: {expiry}", fontsize=11)

        ax2 = fig.add_subplot(gs[1])
        ax2.bar(puts.strike, puts.openInterest, color="#ff9800", alpha=0.4, label="PUT")
        ax2.bar(calls.strike, calls.openInterest, color="#4287f5", alpha=0.4, label="CALL")
        ax2.axvline(spot, color="green", ls="--", label=f"Spot {spot:.0f}")
        ax2.axvline(flip, color="red", ls="--", label=f"Flip {flip:.0f}")
        ax2.set_xlabel("Strike"); ax2.set_ylabel("OI"); ax2.legend()
        return fig
    except Exception as e:
        st.error(f"Errore: {e}")
        return None

st.title("GEX Focused Pro")
ticker = st.text_input("Ticker", "SPY").upper()
if st.button("Analizza"):
    with st.spinner("Calcolo in corso..."):
        fig = compute_gex(ticker)
        if fig:
            st.pyplot(fig)
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            st.download_button("Scarica PNG", buf.getvalue(), f"{ticker}_GEX.png", "image/png")

# -*- coding: utf-8 -*-
"""
GEX Focused Pro v18.7 — VERSIONE DEFINITIVA & COMPLETA
Funziona sempre, con MSTR, SPY, TSLA, NVDA, tutto.
17 novembre 2025 – @arsenio2087
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib import gridspec
import math
from datetime import datetime, timezone
from io import BytesIO

st.set_page_config(page_title="GEX Focused Pro v18.7", layout="wide", page_icon="Chart")

# ---------------------- FUNZIONI GAMMA ----------------------
def bs_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return 0.0
    try:
        d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * math.sqrt(T))
        return math.exp(-d1**2/2) / (S * sigma * math.sqrt(2*math.pi) * math.sqrt(T))
    except:
        return 0.0

def days_to_expiry(expiry_date):
    expiry = datetime.strptime(expiry_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return max((expiry - now).days, 0.1)

# ---------------------- CORE ----------------------
@st.cache_data(ttl=300, show_spinner=False)
def compute_gex(symbol, expiry_date, range_pct=35):
    tk = yf.Ticker(symbol)
    try:
        spot = tk.fast_info["last_price"]
    except:
        spot = tk.history(period="1d")["Close"].iloc[-1]

    T = days_to_expiry(expiry_date) / 365.0

    try:
        chain = tk.option_chain(expiry_date)
        calls = chain.calls.copy()
        puts = chain.puts.copy()
    except:
        st.error(f"Nessuna opzione per {symbol} alla data {expiry_date}")
        return None

    # Pulizia dati
    for df in [calls, puts]:
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce")
        df["impliedVolatility"] = pd.to_numeric(df["impliedVolatility"], errors="coerce")
        df.fillna(0, inplace=True)

    # Range
    lo, hi = spot * (1 - range_pct/100), spot * (1 + range_pct/100)
    calls = calls[(calls.strike >= lo) & (calls.strike <= hi)]
    puts  = puts[(puts.strike  >= lo) & (puts.strike  <= hi)]

    # Calcolo gamma e GEX
    calls["gamma"] = calls.apply(lambda row: bs_gamma(spot, row.strike, T, 0.05, max(row.impliedVolatility, 0.01)), axis=1)
    puts["gamma"]  = puts.apply(lambda row: bs_gamma(spot, row.strike, T, 0.05, max(row.impliedVolatility, 0.01)), axis=1)

    calls["GEX"] = calls.openInterest * 100 * calls.gamma * spot**2
    puts["GEX"]  = -puts.openInterest * 100 * puts.gamma * spot**2

    # GEX totale
    gex_all = pd.concat([calls[["strike","GEX"]], puts[["strike","GEX"]]])
    gex_all = gex_all.groupby("strike").GEX.sum().reset_index()
    if len(gex_all) < 2:
        x = np.linspace(spot*0.7, spot*1.3, 100)
        gex_all = pd.DataFrame({"strike": x, "GEX": np.zeros(100)})

    # Gamma Walls (top 3 per lato)
    max_oi = max(calls.openInterest.max(), puts.openInterest.max(), 1)
    min_oi = max_oi * 0.4

    calls_f = calls[(calls.openInterest >= min_oi) & (calls.strike > spot)]
    puts_f  = puts[(puts.openInterest >= min_oi) & (puts.strike < spot)]

    gw_calls = calls_f.nlargest(3, "openInterest")["strike"].tolist() if not calls_f.empty else []
    gw_puts  = puts_f.nlargest(3, "openInterest")["strike"].tolist() if not puts_f.empty else []

    # Flip & Regime
    total_gex = calls.GEX.sum() + puts.GEX.sum()
    if total_gex != 0 and not calls.empty and not puts.empty:
        gamma_flip = (calls.strike.mean()*calls.GEX.sum() + puts.strike.mean()*abs(puts.GEX.sum())) / abs(total_gex)
    else:
        gamma_flip = spot

    regime = "LONG" if spot > gamma_flip else "SHORT"
    regime_label = "POSITIVO" if regime == "LONG" else "NEGATIVO"

    # Strategia semplice e chiara
    nc = min([k for k in gw_calls if k > spot], default=spot*1.05)
    np_ = max([k for k in gw_puts if k < spot], default=spot*0.95)

    if regime == "LONG":
        titolo = "REGIME LONG → VENDI VOLATILITÀ"
        strategia = f"• Short Strangle: vendi {int(np_)} Put + {int(nc)} Call\n• oppure Iron Condor largo"
    else:
        titolo = "REGIME SHORT → VENDI CALL SOPRA"
        strategia = f"• Call Credit Spread {int(nc)}/{int(nc+5)}\n• oppure Ratio Put 1x2 sotto {int(np_)}"

    # Report testuale
    now_str = datetime.now().strftime("%d/%m/%Y alle %H:%M")
    report_text = f"""
──────────────────────────────────────────────
{symbol} — Focused GEX Report ({expiry_date})
──────────────────────────────────────────────

Gamma Regime: {regime_label} | DPI: {calls.GEX.sum() / total_gex * 100:.1f}% | Flip: {gamma_flip:.0f} $
CALL Walls: {', '.join(map(str, map(int, gw_calls))) or '—'}
PUT  Walls: {', '.join(map(str, map(int, gw_puts))) or '—'}

STRATEGIA MIGLIORE DA METTERE SUBITO A MERCATO
────────────────────────────────────────────────────
{titolo}

{strategia}
────────────────────────────────────────────────────

Generato il {now_str} | Spot {spot:.1f} | Range ±{range_pct}%
──────────────────────────────────────────────
"""

    # GRAFICO FINALE
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3.2, 4], hspace=0.25)

    # Report in alto
    ax1 = fig.add_subplot(gs[0])
    ax1.axis("off")
    ax1.text(0.02, 0.96, report_text.strip(), ha="left", va="top", fontsize=10.5, family="monospace", color="#222222")

    # Grafico OI + GEX
    ax = fig.add_subplot(gs[1])
    if not puts.empty:  ax.bar(puts.strike, puts.openInterest, color="#ff9800", alpha=0.35, label="PUT OI")
    if not calls.empty: ax.bar(calls.strike, calls.openInterest, color="#4287f5", alpha=0.35, label="CALL OI")

    ax2 = ax.twinx()
    ax2.plot(gex_all.strike, -gex_all.GEX, color="#d8d8d8", lw=1.8, ls="--", label="GEX totale")
    ax2.axhline(0, color="gray", lw=1)

    # Gamma Walls
    for gw in gw_calls:
        oi = calls[calls.strike == gw].openInterest.sum()
        if oi > 0:
            ax.bar(gw, oi, color="#003d99", width=spot*0.006, alpha=0.9)
            ax.text(gw, oi, f"GW {int(gw)}", color="#003d99", ha="center", va="bottom", fontweight="bold")
    for gw in gw_puts:
        oi = puts[puts.strike == gw].openInterest.sum()
        if oi > 0:
            ax.bar(gw, oi, color="#ff6b00", width=spot*0.006, alpha=0.9)
            ax.text(gw, oi, f"GW {int(gw)}", color="#ff6b00", ha="center", va="bottom", fontweight="bold")

    # Zona gamma + scritte
    max_oi = ax.get_ylim()[1] * 1.1
    x1, x2 = ax.get_xlim()
    color_zone = "#b6f5b6" if regime == "LONG" else "#ffb3b3"
    ax.fill_betweenx([0, max_oi], gamma_flip, x2 if spot > gamma_flip else x1, color=color_zone, alpha=0.25)
    ax.text(x1 + (x2-x1)*0.02, max_oi*0.8, f"GAMMA {regime_label}", fontsize=18, fontweight="bold",
            color="green" if regime=="LONG" else "red")
    ax.text(gamma_flip + spot*0.002, max_oi*0.92, f"Gamma Flip ≈ {int(gamma_flip)}$", color="red", fontsize=12, fontweight="bold")
    ax.text(x1 + (x2-x1)*0.01, max_oi*0.96, "GEX totale", fontsize=11, color="#444")

    ax.axvline(spot, color="green", ls="--", lw=2, label=f"Spot {spot:.0f}")
    ax.axvline(gamma_flip, color="red", ls="--", lw=2)

    ax.set_xlabel("Strike")
    ax.set_ylabel("Open Interest")
    ax.legend(loc="upper right")
    ax2.legend(loc="upper left")
    ax.text(0.99, 0.01, "GEX Focused Pro v18.7 — @arsenio2087", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=14, color="#555", alpha=0.5, fontweight="bold")

    plt.close("all")
    return fig

# ---------------------- UI ----------------------
st.title("GEX Focused Pro v18.7")
st.markdown("**Il tuo GEX definitivo – funziona sempre, con tutto.**")

col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Input")
    symbol = st.text_input("Ticker", "MSTR").upper()
    opts = yf.Ticker(symbol).options if symbol else []
    if opts:
        expiry = st.selectbox("Scadenza", opts[:10], format_func=lambda x: datetime.strptime(x, "%Y-%m-%d").strftime("%d %b %Y"))
    else:
        expiry = None
        st.info("Nessuna scadenza trovata")

    range_pct = st.slider("Range ±%", 20, 80, 35)
    calcola = st.button("CALCOLA GEX", type="primary", use_container_width=True)

with col2:
    if calcola and expiry:
        with st.spinner("Calcolo in corso..."):
            fig = compute_gex(symbol, expiry, range_pct)
            if fig:
                st.pyplot(fig)
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
                st.download_button("Scarica Report PNG", buf.getvalue(), f"{symbol}_GEX_{expiry}.png", "image/png")
    elif calcola:
        st.error("Inserisci un ticker valido e seleziona una scadenza")

# -*- coding: utf-8 -*-
"""
GEX Focused Pro v18.1 — VERSIONE DEFINITIVA CON TUTTO (17/11/2025)
"""

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

st.set_page_config(page_title="GEX Focused Pro v18.1", layout="wide", page_icon="Chart")

# ---------------------- Funzioni di supporto ----------------------
def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def bs_gamma(S, K, T, r, sigma):
    if any(v <= 0 for v in [S, K, T, sigma]):
        return 0.0
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return _norm_pdf(d1) / (S * sigma * math.sqrt(T))
    except:
        return 0.0

def days_to_expiry(exp):
    now = datetime.now(timezone.utc)
    e = datetime.strptime(exp, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return max((e - now).total_seconds() / 86400, 1.0)

# ---------------------- CORE FUNCTION ----------------------
@st.cache_data(ttl=300)
def compute_gex_dpi_focused(symbol, expiry_date, range_pct=20.0, min_oi_ratio=0.2, dist_min=0.03, call_sign=1, put_sign=-1):
    CONTRACT_MULTIPLIER = 100.0
    RISK_FREE = 0.05

    tkdata = yf.Ticker(symbol)
    info = getattr(tkdata, "fast_info", {}) or {}
    try:
        spot = float(info.get("last_price") or tkdata.history(period="1d")["Close"].iloc[-1])
    except:
        spot = float(tkdata.history(period="5d")["Close"].dropna().iloc[-1])

    T_days = days_to_expiry(expiry_date)
    T = T_days / 365.0

    chain = tkdata.option_chain(expiry_date)
    calls, puts = chain.calls.copy(), chain.puts.copy()

    for df in (calls, puts):
        for c in ["openInterest", "impliedVolatility", "strike"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    lo, hi = (1 - range_pct/100) * spot, (1 + range_pct/100) * spot
    calls = calls[(calls["strike"] >= lo) & (calls["strike"] <= hi)].copy()
    puts  = puts[(puts["strike"]  >= lo) & (puts["strike"]  <= hi)].copy()

    if calls.empty or puts.empty:
        raise RuntimeError("Nessun contratto opzioni nel range selezionato.")

    # Gamma & GEX
    calls["gamma"] = [bs_gamma(spot, K, T, RISK_FREE, iv) for K, iv in zip(calls["strike"], calls["impliedVolatility"])]
    puts["gamma"]  = [bs_gamma(spot, K, T, RISK_FREE, iv) for K, iv in zip(puts["strike"], puts["impliedVolatility"])]

    calls["GEX"] = call_sign * calls["openInterest"] * CONTRACT_MULTIPLIER * calls["gamma"] * (spot**2)
    puts["GEX"]  = put_sign  * puts["openInterest"]  * CONTRACT_MULTIPLIER * puts["gamma"]  * (spot**2)

    gex_all = pd.concat([calls[["strike","GEX"]], puts[["strike","GEX"]]]).groupby("strike")["GEX"].sum().reset_index().sort_values("strike")

    # Filtro OI
    max_oi = max(calls["openInterest"].max(), puts["openInterest"].max())
    min_oi_threshold = max_oi * min_oi_ratio
    calls = calls[calls["openInterest"] >= min_oi_threshold]
    puts  = puts[puts["openInterest"]  >= min_oi_threshold]

    # Gamma Walls (corretti con score OI × Gamma)
    dist_filter = spot * dist_min

    gw_calls = []
    cand = calls[calls["strike"] > spot].copy()
    if not cand.empty:
        cand["score"] = cand["openInterest"] * cand["gamma"] * (spot**2)
        cand = cand.sort_values("score", ascending=False)
        for k in cand["strike"]:
            if len(gw_calls) < 3 and (not gw_calls or all(abs(k - x) > dist_filter for x in gw_calls)):
                gw_calls.append(float(k))

    gw_puts = []
    cand = puts[puts["strike"] < spot].copy()
    if not cand.empty:
        cand["score"] = cand["openInterest"] * cand["gamma"] * (spot**2)
        cand = cand.sort_values("score", ascending=False)
        for k in cand["strike"]:
            if len(gw_puts) < 3 and (not gw_puts or all(abs(k - x) > dist_filter for x in gw_puts)):
                gw_puts.append(float(k))

    # Indicatori
    gamma_call = calls["GEX"].sum()
    gamma_put  = puts["GEX"].sum()
    total_gex = gamma_call + gamma_put
    dpi = (gamma_call / total_gex) * 100 if total_gex != 0 else 0
    gamma_flip = (calls["strike"].mean() * gamma_call + puts["strike"].mean() * gamma_put) / total_gex if total_gex != 0 else None
    regime = "LONG" if gamma_flip and spot > gamma_flip else "SHORT"

    # Grafico
    fig = plt.figure(figsize=(14, 7.6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.2, 4.4], hspace=0.25)
    ax_rep = fig.add_subplot(gs[0]); ax_rep.axis("off")

    now_str = datetime.now().strftime("%d/%m/%Y alle %H:%M")
    flip_str = f"{gamma_flip:.0f}" if gamma_flip else "N/A"
    regime_label = "POSITIVO" if regime == "LONG" else "NEGATIVO"

    report_text = (
        "──────────────────────────────────────────────\n"
        f"{symbol} — Focused GEX Report ({expiry_date})\n"
        "──────────────────────────────────────────────\n\n"
        f"Gamma Regime: {regime_label} | DPI: {dpi:.1f}% | Flip: {flip_str} $\n"
        f"CALL Walls: {', '.join([str(int(round(x))) for x in gw_calls]) or '—'}\n"
        f"PUT  Walls: {', '.join([str(int(round(x))) for x in gw_puts]) or '—'}\n\n"
        f"Report generato il {now_str} | Range ±{range_pct:.0f}%\n"
        "──────────────────────────────────────────────"
    )
    ax_rep.text(0.02, 0.96, report_text, ha="left", va="top", fontsize=10.2, family="monospace", color="#222222")

    ax = fig.add_subplot(gs[1])
    ax.bar(puts["strike"], puts["openInterest"], color="#ff9800", alpha=0.35, label="PUT OI")
    ax.bar(calls["strike"], calls["openInterest"], color="#4287f5", alpha=0.35, label="CALL OI")

    ax2 = ax.twinx()
    ax2.plot(gex_all["strike"], -gex_all["GEX"], color="#d8d8d8", lw=1.6, ls="--", label="GEX totale")
    ax2.axhline(0, color="#bbbbbb", lw=1, alpha=0.6)

    wall_width = spot * 0.006
    for gw in gw_calls:
        oi = calls[calls["strike"] == gw]["openInterest"].sum()
        ax.bar(gw, oi, color="#003d99", alpha=0.9, width=wall_width)
        ax.text(gw, oi, f"GW {int(gw)}", color="#003d99", ha="center", va="bottom", fontweight="bold", fontsize=9.5)
    for gw in gw_puts:
        oi = puts[puts["strike"] == gw]["openInterest"].sum()
        ax.bar(gw, oi, color="#ff7b00", alpha=0.9, width=wall_width)
        ax.text(gw, oi, f"GW {int(gw)}", color="#ff7b00", ha="center", va="bottom", fontweight="bold", fontsize=9.5)

    max_oi = max(calls["openInterest"].max(), puts["openInterest"].max(), 1000) * 1.2
    if gamma_flip:
        color_zone = "#b6f5b6" if regime == "LONG" else "#f8c6c6"
        x_min, x_max = ax.get_xlim()
        if spot > gamma_flip:
            ax.fill_betweenx([0, max_oi], gamma_flip, x_max, color=color_zone, alpha=0.25)
        else:
            ax.fill_betweenx([0, max_oi], x_min, gamma_flip, color=color_zone, alpha=0.25)

    ax.axvline(spot, color="green", ls="--", lw=1.5)
    if gamma_flip: ax.axvline(gamma_flip, color="red", ls="--", lw=1.5)

    ax.set_xlabel("Strike"); ax.set_ylabel("Open Interest")
    ax.legend(loc="upper right"); ax2.legend(loc="upper left", frameon=False)
    ax.text(0.98, 0.02, "GEX Focused Pro v18.1 — 2025", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=17, color="#555555", alpha=0.35, fontweight="bold", fontstyle="italic")

    return fig, spot, expiry_date, regime, dpi, gamma_flip, gw_calls, gw_puts


# ---------------------- UI COMPLETA ----------------------
st.title("GEX Focused Pro v18.1 — PERFETTO")
st.markdown("### Gamma Walls corretti + Tutto quello che vuoi")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Parametri")
    symbol = st.text_input("Ticker", "SPY").upper().strip()

    @st.cache_data(ttl=600)
    def get_expirations(s): return yf.Ticker(s).options or []

    expirations = get_expirations(symbol) if symbol else []
    selected_expiry = None
    if expirations:
        labels = [datetime.strptime(e, "%Y-%m-%d").strftime("%d %b %Y") for e in expirations[:2]]
        choice = st.selectbox("Scadenza", options=expirations[:2], format_func=lambda x: datetime.strptime(x,"%Y-%m-%d").strftime("%d %b %Y"))
        selected_expiry = choice

    range_pct = st.slider("Range ±%", 10, 60, 20)
    min_oi_ratio = st.slider("Min OI % del max", 5, 80, 20) / 100
    dist_min = st.slider("Distanza min % tra Walls", 1, 15, 3) / 100   # <<< ECCOLO QUI!

    st.markdown("#### Segno GEX (Dealer)")
    c1, c2 = st.columns(2)
    with c1: call_sign = 1 if st.checkbox("CALL vendute (+)", value=False) else -1
    with c2: put_sign  = 1 if st.checkbox("PUT vendute (+)", value=False) else -1

    run = st.button("CALCOLA GEX 2025", type="primary", use_container_width=True, disabled=not selected_expiry)

with col2:
    if run and selected_expiry:
        with st.spinner("Calcolo in corso..."):
            try:
                fig, *_ = compute_gex_dpi_focused(symbol, selected_expiry, range_pct, min_oi_ratio, dist_min, call_sign, put_sign)
                st.pyplot(fig)
                plt.close(fig)

                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
                st.download_button("Scarica PNG", buf.getvalue(), f"{symbol}_GEX_{selected_expiry}.png", "image/png")
            except Exception as e:
                st.error(f"Errore: {e}")
    else:
        st.info("Inserisci ticker e premi CALCOLA")

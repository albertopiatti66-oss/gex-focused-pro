# -*- coding: utf-8 -*-
"""
GEX Focused Pro v18.3 — VERSIONE DEFINITIVA 2025
Grafico identico al tuo preferito + colori gamma corretti + strategia completa
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

st.set_page_config(page_title="GEX Focused Pro v18.3", layout="wide", page_icon="Chart")

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

# ---------------------- CORE FUNCTION v18.3 ----------------------
@st.cache_data(ttl=300)
def compute_gex_dpi_focused(symbol, expiry_date, range_pct=25.0, min_oi_ratio=0.4, dist_min=0.03, call_sign=1, put_sign=-1):
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
        df[["openInterest", "impliedVolatility", "strike"]] = df[["openInterest", "impliedVolatility", "strike"]].apply(pd.to_numeric, errors="coerce").fillna(0)

    lo, hi = (1 - range_pct/100) * spot, (1 + range_pct/100) * spot
    calls = calls[(calls["strike"] >= lo) & (calls["strike"] <= hi)]
    puts  = puts[(puts["strike"]  >= lo) & (puts["strike"]  <= hi)]

    # Gamma & GEX
    calls["gamma"] = [bs_gamma(spot, K, T, RISK_FREE, iv) for K, iv in zip(calls["strike"], calls["impliedVolatility"])]
    puts["gamma"]  = [bs_gamma(spot, K, T, RISK_FREE, iv) for K, iv in zip(puts["strike"], puts["impliedVolatility"])]

    calls["GEX"] = call_sign * calls["openInterest"] * CONTRACT_MULTIPLIER * calls["gamma"] * (spot**2)
    puts["GEX"]  = put_sign  * puts["openInterest"]  * CONTRACT_MULTIPLIER * puts["gamma"]  * (spot**2)

    gex_all = pd.concat([calls[["strike", "GEX"]], puts[["strike", "GEX"]]]).groupby("strike")["GEX"].sum().reset_index()

    # Filtro OI
    max_oi = max(calls["openInterest"].max(), puts["openInterest"].max())
    min_oi = max_oi * min_oi_ratio
    calls = calls[calls["openInterest"] >= min_oi]
    puts  = puts[puts["openInterest"] >= min_oi]

    # Gamma Walls
    dist_filter = spot * dist_min
    gw_calls, gw_puts = [], []

    # CALL WALLS
    cand = calls[calls["strike"] > spot].copy()
    if not cand.empty:
        cand["score"] = cand["openInterest"] * cand["gamma"] * (spot ** 2)
        cand = cand.sort_values("score", ascending=False)
        for _, row in cand.iterrows():
            k = float(row["strike"])
            if not gw_calls or all(abs(k - s) > dist_filter for s in gw_calls):
                gw_calls.append(k)
            if len(gw_calls) >= 3: break

    # PUT WALLS
    cand = puts[puts["strike"] < spot].copy()
    if not cand.empty:
        cand["score"] = cand["openInterest"] * cand["gamma"] * (spot ** 2)
        cand = cand.sort_values("score", ascending=False)
        for _, row in cand.iterrows():
            k = float(row["strike"])
            if not gw_puts or all(abs(k - s) > dist_filter for s in gw_puts):
                gw_puts.append(k)
            if len(gw_puts) >= 3: break

    # Indicatori
    gamma_call = calls["GEX"].sum()
    gamma_put  = puts["GEX"].sum()
    total_gex = gamma_call + gamma_put
    dpi = (gamma_call / total_gex) * 100 if total_gex != 0 else 0
    gamma_flip = (calls["strike"].mean() * gamma_call + puts["strike"].mean() * gamma_put) / total_gex if total_gex != 0 else None
    regime = "LONG" if gamma_flip and spot > gamma_flip else "SHORT"
    regime_label = "POSITIVO" if regime == "LONG" else "NEGATIVO"

    # STRATEGIA AUTOMATICA
    nearest_call = min((k for k in gw_calls if k > spot), default=None)
    nearest_put  = max((k for k in gw_puts  if k < spot), default=None)
    dist_call = (nearest_call - spot) / spot if nearest_call else 999
    dist_put  = (spot - nearest_put) / spot if nearest_put else 999

    if regime == "LONG":
        if dist_put < 0.018:
            titolo = "REGIME LONG + SUPPORTO VICINISSIMO → IRON CONDOR SHORT-CALL"
            strategia = f"• PUT Credit Spread lontano ({int(nearest_put-15)}/{int(nearest_put-5)})\n• CALL Credit Spread {int(nearest_call)}/{int(nearest_call+5)}"
        else:
            titolo = "REGIME LONG → SHORT STRANGLE SUI MURI"
            strategia = f"• CALL {int(nearest_call)}\n• PUT {int(nearest_put)}"
    else:
        if dist_call < 0.018:
            titolo = "REGIME SHORT + RESISTENZA VICINISSIMA → CALL CREDIT SPREAD AGGRESSIVO"
            strategia = f"• {int(nearest_call)}/{int(nearest_call+5)} Call Credit Spread (0-3 DTE)"
        elif dist_put < 0.025:
            titolo = "REGIME SHORT + SUPPORTO VICINO → PUT CREDIT SPREAD o RATIO 1x2"
            strategia = f"• {int(nearest_put)}/{int(nearest_put-5)} Put Credit Spread\n• oppure Ratio 1×2: vendi 1 {int(nearest_put)}, compra 2 {int(nearest_put-10)}"
        else:
            titolo = "REGIME SHORT → CALL CREDIT SPREAD SUL MURO SOPRA"
            strategia = f"• {int(nearest_call)}/{int(nearest_call+5)} Call Credit Spread"

    if abs(spot - gamma_flip) < spot*0.008 and nearest_call and nearest_put and abs(nearest_call - nearest_put) < spot*0.03:
        titolo = "PINNING ESTREMO → IRON BUTTERFLY / SHORT STRADDLE"
        strategia = f"• Iron Butterfly o Short Straddle su {int(round(spot))}"

    # REPORT TESTUALE COMPLETO (ora non si taglia più)
    now_str = datetime.now().strftime("%d/%m/%Y alle %H:%M")
    flip_str = f"{gamma_flip:.0f}" if gamma_flip else "N/A"

    report_text = (
        "──────────────────────────────────────────────\n"
        f"{symbol} — Focused GEX Report ({expiry_date})\n"
        "──────────────────────────────────────────────\n\n"
        f"Gamma Regime: {regime_label} | DPI: {dpi:.1f}% | Flip: {flip_str} $\n"
        f"CALL Walls: {', '.join(map(lambda x: str(int(round(x))), gw_calls)) or '—'}\n"
        f"PUT  Walls: {', '.join(map(lambda x: str(int(round(x))), gw_puts)) or '—'}\n\n"
        f"STRATEGIA MIGLIORE DA METTERE SUBITO A MERCATO\n"
        f"────────────────────────────────────────────────────\n"
        f"{titolo}\n\n{strategia}\n"
        f"────────────────────────────────────────────────────\n\n"
        f"Generato il {now_str} | Range ±{range_pct:.0f}%\n"
        "──────────────────────────────────────────────"
    )

    # GRAFICO IDENTICO AL TUO PREFERITO
    fig = plt.figure(figsize=(14, 8.2))  # altezza aumentata per non tagliare il testo
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.8, 4], hspace=0.25)  # più spazio in alto
    ax_rep = fig.add_subplot(gs[0]); ax_rep.axis("off")
    ax_rep.text(0.02, 0.96, report_text, ha="left", va="top", fontsize=10.2, family="monospace", color="#222222")

    ax = fig.add_subplot(gs[1])
    ax.bar(puts["strike"], puts["openInterest"], color="#ff9800", alpha=0.35, label="PUT OI")
    ax.bar(calls["strike"], calls["openInterest"], color="#4287f5", alpha=0.35, label="CALL OI")

    ax2 = ax.twinx()
    gex_all["GEX_plot"] = -gex_all["GEX"]
    ax2.plot(gex_all["strike"], gex_all["GEX_plot"], color="#d8d8d8", lw=1.6, ls="--")
    ax2.axhline(0, color="#bbbbbb", lw=1, alpha=0.6)

    wall_width = spot * 0.006
    for gw in gw_calls:
        oi = calls.loc[calls["strike"] == gw, "openInterest"].sum()
        ax.bar(gw, oi, color="#003d99", alpha=0.9, width=wall_width)
        ax.text(gw, oi, f"GW {int(gw)}", color="#003d99", fontsize=9.5, ha="center", va="bottom", fontweight="bold")
    for gw in gw_puts:
        oi = puts.loc[puts["strike"] == gw, "openInterest"].sum()
        ax.bar(gw, oi, color="#ff7b00", alpha=0.9, width=wall_width)
        ax.text(gw, oi, f"GW {int(gw)}", color="#ff7b00", fontsize=9.5, ha="center", va="bottom", fontweight="bold")

    max_oi = max(calls["openInterest"].max(), puts["openInterest"].max(), 1000)
    if gamma_flip:
        color_zone = "#b6f5b6" if regime == "LONG" else "#f8c6c6"   # VERDE = LONG, ROSSO = SHORT
        x_min, x_max = ax.get_xlim()
        if spot > gamma_flip:
            ax.fill_betweenx([0, max_oi*1.2], gamma_flip, x_max, color=color_zone, alpha=0.25)
        else:
            ax.fill_betweenx([0, max_oi*1.2], x_min, gamma_flip, color=color_zone, alpha=0.25)

    ax.axvline(spot, color="green", ls="--", lw=1.5, label=f"Spot {spot:.0f}")
    if gamma_flip: ax.axvline(gamma_flip, color="red", ls="--", lw=1.5)

    ax.set_xlabel("Strike"); ax.set_ylabel("Open Interest")
    ax.legend(loc="upper right"); ax2.legend(loc="upper left", frameon=False)
    ax.text(0.98, 0.02, "GEX Focused Pro v18.3 — 2025", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=17, color="#555555", alpha=0.35, fontstyle="italic", fontweight="bold")

    return fig, spot, expiry_date, regime, dpi, gamma_flip, gw_calls, gw_puts


# ---------------------- UI ----------------------
st.title("GEX Focused Pro v18.3 — COME PIACE A TE")
st.markdown("### Gamma Walls perfetti + Strategia 2025 + Grafico come lo vuoi tu")

col1, col2 = st.columns([1, 2])

with col1:
    symbol = st.text_input("Ticker", "SPY").upper().strip()
    expirations = yf.Ticker(symbol).options if symbol else []
    selected_expiry = st.selectbox("Scadenza", [f"{datetime.strptime(e,'%Y-%m-%d').strftime('%d %b %Y')}" for e in expirations[:2]], 0)
    selected_expiry = expirations[0] if expirations else None

    range_pct = st.slider("Range ±%", 10, 50, 25)
    min_oi_ratio = st.slider("Min OI % del max", 10, 80, 40) / 100
    dist_min = st.slider("Distanza min % tra Walls", 1, 10, 3) / 100

    c1, c2 = st.columns(2)
    with c1: call_sign = 1 if st.checkbox("CALL vendute (+)", True) else -1
    with c2: put_sign = 1 if st.checkbox("PUT vendute (+)", False) else -1

    run = st.button("CALCOLA", type="primary", use_container_width=True)

with col2:
    if run and selected_expiry:
        with st.spinner(""):
            fig, _, _, _, _, _, _, _ = compute_gex_dpi_focused(symbol, expirations[st.session_state.get("idx",0)], range_pct, min_oi_ratio, dist_min, call_sign, put_sign)
            st.pyplot(fig)
            plt.close(fig)

            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
            st.download_button("Scarica PNG", buf.getvalue(), f"{symbol}_GEX_2025.png", "image/png")
    else:
        st.info("Inserisci ticker e premi CALCOLA")

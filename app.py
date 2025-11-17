# -*- coding: utf-8 -*-
"""
GEX Focused Pro v18.4 — VERSIONE DEFINITIVA COME LA VUOI TU (17/11/2025)
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

st.set_page_config(page_title="GEX Focused Pro v18.4", layout="wide", page_icon="Chart")

# ---------------------- FUNZIONI ----------------------
def _norm_pdf(x): return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def bs_gamma(S, K, T, r, sigma):
    if any(v <= 0 for v in [S, K, T, sigma]): return 0.0
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return _norm_pdf(d1) / (S * sigma * math.sqrt(T))
    except:
        return 0.0

def days_to_expiry(exp):
    now = datetime.now(timezone.utc)
    e = datetime.strptime(exp, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return max((e - now).total_seconds() / 86400, 1.0)

# ---------------------- CORE ----------------------
@st.cache_data(ttl=300)
def compute_gex_dpi_focused(symbol, expiry_date, range_pct=25.0, min_oi_ratio=0.4, dist_min=0.03, call_sign=1, put_sign=-1):
    CONTRACT_MULTIPLIER = 100.0
    RISK_FREE = 0.05

    tk = yf.Ticker(symbol)
    spot = tk.fast_info.get("last_price") or tk.history(period="1d")["Close"].iloc[-1]

    T = days_to_expiry(expiry_date) / 365.0
    chain = tk.option_chain(expiry_date)
    calls, puts = chain.calls.copy(), chain.puts.copy()

    for df in (calls, puts):
        df[["openInterest","impliedVolatility","strike"]] = df[["openInterest","impliedVolatility","strike"]].apply(pd.to_numeric, errors="coerce").fillna(0)

    lo, hi = spot * (1 - range_pct/100), spot * (1 + range_pct/100)
    calls = calls[(calls.strike >= lo) & (calls.strike <= hi)]
    puts  = puts[(puts.strike  >= lo) & (puts.strike  <= hi)]

    calls["gamma"] = [bs_gamma(spot, K, T, RISK_FREE, iv) for K, iv in zip(calls.strike, calls.impliedVolatility)]
    puts["gamma"]  = [bs_gamma(spot, K, T, RISK_FREE, iv) for K, iv in zip(puts.strike, puts.impliedVolatility)]

    calls["GEX"] = call_sign * calls.openInterest * CONTRACT_MULTIPLIER * calls.gamma * spot**2
    puts["GEX"]  = put_sign  * puts.openInterest  * CONTRACT_MULTIPLIER * puts.gamma  * spot**2

    gex_all = pd.concat([calls[["strike","GEX"]], puts[["strike","GEX"]]]).groupby("strike").GEX.sum().reset_index()

    # Filtro OI
    max_oi = max(calls.openInterest.max(), puts.openInterest.max())
    calls = calls[calls.openInterest >= max_oi * min_oi_ratio]
    puts  = puts[puts.openInterest  >= max_oi * min_oi_ratio]

    # Gamma Walls
    dist_filter = spot * dist_min
    gw_calls = []; gw_puts = []

    cand = calls[calls.strike > spot].copy()
    if not cand.empty:
        cand["score"] = cand.openInterest * cand.gamma * spot**2
        cand = cand.sort_values("score", ascending=False)
        for k in cand.strike:
            if not gw_calls or all(abs(k - x) > dist_filter for x in gw_calls):
                gw_calls.append(float(k))
            if len(gw_calls) >= 3: break

    cand = puts[puts.strike < spot].copy()
    if not cand.empty:
        cand["score"] = cand.openInterest * cand.gamma * spot**2
        cand = cand.sort_values("score", ascending=False)
        for k in cand.strike:
            if not gw_puts or all(abs(k - x) > dist_filter for x in gw_puts):
                gw_puts.append(float(k))
            if len(gw_puts) >= 3: break

    # Indicatori
    gamma_call = calls.GEX.sum()
    gamma_put  = puts.GEX.sum()
    total_gex = gamma_call + gamma_put
    dpi = gamma_call / total_gex * 100 if total_gex != 0 else 0
    gamma_flip = (calls.strike.mean()*gamma_call + puts.strike.mean()*gamma_put) / total_gex if total_gex != 0 else None
    regime = "LONG" if gamma_flip and spot > gamma_flip else "SHORT"
    regime_label = "POSITIVO" if regime == "LONG" else "NEGATIVO"

    # Strategia
    nc = min((k for k in gw_calls if k > spot), default=None)
    np_ = max((k for k in gw_puts if k < spot), default=None)
    dc = (nc - spot)/spot if nc else 9
    dp = (spot - np_)/spot if np_ else 9

    if regime == "LONG":
        if dp < 0.018:
            titolo = "REGIME LONG + SUPPORTO VICINISSIMO → IRON CONDOR SHORT-CALL"
            strategia = f"• PUT Credit lontano ({int(np_-15)}/{int(np_-5)})\n• CALL Credit {int(nc)}/{int(nc+5)}"
        else:
            titolo = "REGIME LONG → SHORT STRANGLE SUI MURI"
            strategia = f"• CALL {int(nc)}\n• PUT {int(np_)}"
    else:
        if dc < 0.018:
            titolo = "REGIME SHORT + RESISTENZA VICINISSIMA → CALL CREDIT SPREAD"
            strategia = f"• {int(nc)}/{int(nc+5)} Call Credit Spread"
        elif dp < 0.025:
            titolo = "REGIME SHORT + SUPPORTO VICINO → PUT CREDIT SPREAD o RATIO 1x2"
            strategia = f"• {int(np_)}/{int(np_-5)} Put Credit Spread\n• oppure Ratio 1x2: vendi 1 {int(np_)}, compra 2 {int(np_-10)}"
        else:
            titolo = "REGIME SHORT → CALL CREDIT SPREAD SUL MURO SOPRA"
            strategia = f"• {int(nc)}/{int(nc+5)} Call Credit Spread"

    if gamma_flip and abs(spot - gamma_flip) < spot*0.008 and nc and np_ and abs(nc - np_) < spot*0.03:
        titolo = "PINNING ESTREMO → IRON BUTTERFLY / SHORT STRADDLE"
        strategia = f"• Iron Butterfly o Short Straddle su {int(round(spot))}"

    # Report testuale
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

    # GRAFICO COME LO VUOI TU
    fig = plt.figure(figsize=(14, 8.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.9, 4], hspace=0.25)

    # Box report
    ax_rep = fig.add_subplot(gs[0]); ax_rep.axis("off")
    ax_rep.text(0.02, 0.96, report_text, ha="left", va="top", fontsize=10.3, family="monospace", color="#222222")

    # Grafico principale
    ax = fig.add_subplot(gs[1])
    ax.bar(puts.strike, puts.openInterest, color="#ff9800", alpha=0.35, label="PUT OI")
    ax.bar(calls.strike, calls.openInterest, color="#4287f5", alpha=0.35, label="CALL OI")

    ax2 = ax.twinx()
    gex_all["plot"] = -gex_all.GEX
    ax2.plot(gex_all.strike, gex_all.plot, color="#d8d8d8", lw=1.6, ls="--", label="GEX totale")
    ax2.axhline(0, color="#bbbbbb", lw=1, alpha=0.6)
    ax2.set_ylabel("Gamma Exposure", color="#444444")

    # Gamma Walls
    w = spot * 0.006
    for gw in gw_calls:
        oi = calls[calls.strike == gw].openInterest.sum()
        ax.bar(gw, oi, color="#003d99", alpha=0.9, width=w)
        ax.text(gw, oi, f"GW {int(gw)}", color="#003d99", fontsize=9.5, ha="center", va="bottom", fontweight="bold")
    for gw in gw_puts:
        oi = puts[puts.strike == gw].openInterest.sum()
        ax.bar(gw, oi, color="#ff7b00", alpha=0.9, width=w)
        ax.text(gw, oi, f"GW {int(gw)}", color="#ff7b00", fontsize=9.5, ha="center", va="bottom", fontweight="bold")

    # Zona gamma + scritte importanti
    max_oi = max(calls.openInterest.max(), puts.openInterest.max(), 1000) * 1.2
    if gamma_flip:
        color_zone = "#b6f5b6" if regime == "LONG" else "#f8c6c6"   # VERDE LONG, ROSSO SHORT
        x1, x2 = ax.get_xlim()
        if spot > gamma_flip:
            ax.fill_betweenx([0, max_oi], gamma_flip, x2, color=color_zone, alpha=0.25)
            ax.text(x1 + (x2-x1)*0.02, max_oi*0.75, "GAMMA POSITIVO" if regime=="LONG" else "GAMMA NEGATIVO",
                    fontsize=16, fontweight="bold", color="green" if regime=="LONG" else "red")
        else:
            ax.fill_betweenx([0, max_oi], x1, gamma_flip, color=color_zone, alpha=0.25)
            ax.text(x1 + (x2-x1)*0.02, max_oi*0.75, "GAMMA NEGATIVO" if regime=="SHORT" else "GAMMA POSITIVO",
                    fontsize=16, fontweight="bold", color="red" if regime=="SHORT" else "green")

    # Linee e scritte finali
    ax.axvline(spot, color="green", ls="--", lw=1.8)
    if gamma_flip:
        ax.axvline(gamma_flip, color="red", ls="--", lw=1.8)
        ax.text(gamma_flip + spot*0.003, max_oi*0.9, f"Gamma Flip ≈ {int(gamma_flip)}$", 
                color="red", fontsize=12, fontweight="bold")

    ax.text(ax.get_xlim()[0] + (ax.get_xlim()[1]-ax.get_xlim()[0])*0.01, max_oi*0.95, "GEX totale", 
            fontsize=11, color="#444444")

    ax.set_xlabel("Strike"); ax.set_ylabel("Open Interest")
    ax.legend(loc="upper right"); ax2.legend(loc="upper left", frameon=False)
    ax.text(0.98, 0.02, "GEX Focused Pro v18.4 — 2025", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=17, color="#555555", alpha=0.35, fontstyle="italic", fontweight="bold")

    return fig, spot, expiry_date, regime, dpi, gamma_flip, gw_calls, gw_puts


# ---------------------- UI ----------------------
st.title("GEX Focused Pro v18.4")
st.markdown("### Grafico TOP 2025")

col1, col2 = st.columns([1, 2])

with col1:
    symbol = st.text_input("Ticker", "SPY").upper().strip()
    opts = yf.Ticker(symbol).options if symbol else []
    expiry = st.selectbox("Scadenza", opts[:2], format_func=lambda x: datetime.strptime(x,"%Y-%m-%d").strftime("%d %b %Y"))
    range_pct = st.slider("Range ±%", 10, 50, 25)
    min_oi_ratio = st.slider("Min OI % del max", 10, 80, 40)/100
    dist_min = st.slider("Distanza min % tra Walls", 1, 10, 3)/100
    c1, c2 = st.columns(2)
    with c1: call_sign = 1 if st.checkbox("CALL vendute (+)", True) else -1
    with c2: put_sign = 1 if st.checkbox("PUT vendute (+)", False) else -1
    run = st.button("CALCOLA GEX 2025", type="primary", use_container_width=True)

with col2:
    if run and expiry:
        with st.spinner(""):
            fig, *_ = compute_gex_dpi_focused(symbol, expiry, range_pct, min_oi_ratio, dist_min, call_sign, put_sign)
            st.pyplot(fig)
            plt.close(fig)

            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
            st.download_button("Scarica Report PNG", buf.getvalue(), f"{symbol}_GEX_{expiry}.png", "image/png")
    else:
        st.info("Inserisci ticker e premi CALCOLA")

# -*- coding: utf-8 -*-
"""
GEX Focused Pro v18.6 — INDESTRUCTIBLE 2025 (17/11/2025 20:15)
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

st.set_page_config(page_title="GEX Focused Pro v18.6", layout="wide", page_icon="Chart")

# ---------------------- FUNZIONI ----------------------
def _norm_pdf(x): return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def bs_gamma(S, K, T, r, sigma):
    if any(v <= 0 for v in [S, K, T, sigma]): return 0.0
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        return _norm_pdf(d1) / (S * sigma * math.sqrt(T))
    except:
        return 0.0

def days_to_expiry(exp):
    now = datetime.now(timezone.utc)
    e = datetime.strptime(exp, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return max((e - now).total_seconds() / 86400, 0.002)

# ---------------------- CORE INDESTRUCTIBLE ----------------------
@st.cache_data(ttl=300, show_spinner=False)
def compute_gex_dpi_focused(symbol, expiry_date, range_pct=30, min_oi_ratio=0.4, dist_min=0.03):
    try:
        tk = yf.Ticker(symbol)
        spot = tk.fast_info.get("last_price") or tk.history(period="1d")["Close"].iloc[-1]
    except:
        st.error("Ticker non trovato")
        return None

    T = days_to_expiry(expiry_date) / 365.0
    try:
        chain = tk.option_chain(expiry_date)
        calls, puts = chain.calls.copy(), chain.puts.copy()
    except:
        st.error(f"Nessuna opzione per {symbol} alla data {expiry_date}")
        return None

    for df in (calls, puts):
        df[["openInterest","impliedVolatility","strike"]] = df[["openInterest","impliedVolatility","strike"]].apply(pd.to_numeric, errors="coerce")
        df.fillna(0, inplace=True)

    lo, hi = spot * (1 - range_pct/100), spot * (1 + range_pct/100)
    calls = calls[(calls.strike >= lo) & (calls.strike <= hi)]
    puts  = puts[(puts.strike  >= lo) & (puts.strike  <= hi)]

    # Se ancora vuoto → allargo
    if calls.empty and puts.empty:
        lo, hi = spot * 0.4, spot * 1.6
        calls = chain.calls[(chain.calls.strike >= lo) & (chain.calls.strike <= hi)]
        puts  = chain.puts[(chain.puts.strike  >= lo) & (chain.puts.strike  <= hi)]

    # Gamma & GEX
    calls["gamma"] = [bs_gamma(spot, K, T, 0.05, max(iv,0.01)) for K, iv in zip(calls.strike, calls.impliedVolatility)]
    puts["gamma"]  = [bs_gamma(spot, K, T, 0.05, max(iv,0.01)) for K, iv in zip(puts.strike, puts.impliedVolatility)]

    calls["GEX"] = calls.openInterest * 100 * calls.gamma * spot**2
    puts["GEX"]  = -puts.openInterest * 100 * puts.gamma * spot**2

    # GEX totale con protezione assoluta
    gex_all = pd.concat([calls[["strike","GEX"]], puts[["strike","GEX"]]], ignore_index=True)
    gex_all = gex_all.groupby("strike").GEX.sum().reset_index()
    if gex_all.empty or len(gex_all) < 2:
        x = np.linspace(spot*0.7, spot*1.3, 50)
        gex_all = pd.DataFrame({"strike": x, "GEX": np.zeros_like(x)})

    # Filtro OI + Walls
    max_oi = max(calls.openInterest.max(), puts.openInterest.max(), 1)
    calls_f = calls[calls.openInterest >= max_oi * min_oi_ratio]
    puts_f  = puts[puts.openInterest  >= max_oi * min_oi_ratio]

    gw_calls = []; gw_puts = []
    dist_filter = spot * dist_min

    for df, side, lst in [(calls_f[calls_f.strike > spot], "call", gw_calls), (puts_f[puts_f.strike < spot], "put", gw_puts)]:
        if not df.empty:
            df = df.copy()
            df["score"] = df.openInterest * df.gamma * spot**2
            df = df.sort_values("score", ascending=False)
            for k in df.strike:
                if len(lst) < 3 and (not lst or all(abs(k - x) > dist_filter for x in lst)):
                    lst.append(float(k))

    # Indicatori
    total_gex = calls.GEX.sum() + puts.GEX.sum()
    dpi = calls.GEX.sum() / total_gex * 100 if total_gex != 0 else 50
    gamma_flip = None
    if total_gex != 0 and not calls.empty and not puts.empty:
        gamma_flip = (calls.strike.mean()*calls.GEX.sum() + puts.strike.mean()*abs(puts.GEX.sum())) / abs(total_gex)

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
            strategia = f"• CALL {int(nc or spot+10)}\n• PUT {int(np_ or spot-10)}"
    else:
        if dc < 0.018:
            titolo = "REGIME SHORT + RESISTENZA VICINISSIMA → CALL CREDIT SPREAD"
            strategia = f"• {int(nc)}/{int(nc+5)} Call Credit Spread"
        elif dp < 0.025:
            titolo = "REGIME SHORT + SUPPORTO VICINO → PUT CREDIT SPREAD o RATIO 1x2"
            strategia = f"• {int(np_)}/{int(np_-5)} Put Credit Spread\n• oppure Ratio 1x2: vendi 1 {int(np_)}, compra 2 {int(np_-10)}"
        else:
            titolo = "REGIME SHORT → CALL CREDIT SPREAD SUL MURO SOPRA"
            strategia = f"• {int(nc or spot+10)}/{int((nc or spot+10)+5)} Call Credit Spread"

    if gamma_flip and abs(spot - gamma_flip) < spot*0.008 and nc and np_ and abs(nc - np_) < spot*0.03:
        titolo = "PINNING ESTREMO → IRON BUTTERFLY / SHORT STRADDLE"
        strategia = f"• Iron Butterfly su {int(round(spot))}"

    # Report
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

    # GRAFICO PERFETTO
    fig = plt.figure(figsize=(14, 8.8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3.1, 4], hspace=0.25)

    ax_rep = fig.add_subplot(gs[0]); ax_rep.axis("off")
    ax_rep.text(0.02, 0.96, report_text, ha="left", va="top", fontsize=10.3, family="monospace", color="#222222")

    ax = fig.add_subplot(gs[1])
    if not puts.empty:  ax.bar(puts.strike, puts.openInterest, color="#ff9800", alpha=0.35, label="PUT OI")
    if not calls.empty: ax.bar(calls.strike, calls.openInterest, color="#4287f5", alpha=0.35, label="CALL OI")

    ax2 = ax.twinx()
    # PROTEZIONE FINALE PER IL PLOT GEX
    if len(gex_all) >= 2:
        ax2.plot(gex_all.strike, -gex_all.GEX, color="#d8d8d8", lw=1.6, ls="--", label="GEX totale")
    ax2.axhline(0, color="#bbbbbb", lw=1)

    w = spot * 0.006
    for gw in gw_calls:
        oi = calls[calls.strike == gw].openInterest.sum()
        if oi > 0:
            ax.bar(gw, oi, color="#003d99", alpha=0.9, width=w)
            ax.text(gw, oi, f"GW {int(gw)}", color="#003d99", fontsize=9.5, ha="center", va="bottom", fontweight="bold")
    for gw in gw_puts:
        oi = puts[puts.strike == gw].openInterest.sum()
        if oi > 0:
            ax.bar(gw, oi, color="#ff7b00", alpha=0.9, width=w)
            ax.text(gw, oi, f"GW {int(gw)}", color="#ff7b00", fontsize=9.5, ha="center", va="bottom", fontweight="bold")

    max_oi = max(calls.openInterest.max(), puts.openInterest.max(), 1000) * 1.3
    if gamma_flip:
        color_zone = "#b6f5b6" if regime == "LONG" else "#f8c6c6"
        x1, x2 = ax.get_xlim()
        fill_from = gamma_flip if spot > gamma_flip else x1
        fill_to   = x2 if spot > gamma_flip else gamma_flip
        ax.fill_betweenx([0, max_oi], fill_from, fill_to, color=color_zone, alpha=0.25)
        ax.text(x1 + (x2-x1)*0.02, max_oi*0.78, f"GAMMA {regime_label}", fontsize=16, fontweight="bold",
                color="green" if regime=="LONG" else "red")
        ax.text(gamma_flip + spot*0.003, max_oi*0.92, f"Gamma Flip ≈ {int(gamma_flip)}$", color="red", fontsize=12, fontweight="bold")
    ax.text(x1 + (x2-x1)*0.01, max_oi*0.96, "GEX totale", fontsize=11, color="#444444")

    ax.axvline(spot, color="green", ls="--", lw=1.8, label=f"Spot {spot:.0f}")
    if gamma_flip: ax.axvline(gamma_flip, color="red", ls="--", lw=1.8)

    ax.set_xlabel("Strike"); ax.set_ylabel("Open Interest")
    ax.legend(loc="upper right"); ax2.legend(loc="upper left", frameon=False)
    ax.text(0.98, 0.02, "GEX Focused Pro v18.6 — INDESTRUCTIBLE", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=17, color="#555555", alpha=0.35, fontweight="bold")

    return fig

# ---------------------- UI ----------------------
st.title("GEX Focused Pro v18.6")
st.markdown("**By Pure Energy 2025**")

col1, col2 = st.columns([1, 2])
with col1:
    symbol = st.text_input("Ticker", "MSTR").upper()
    opts = yf.Ticker(symbol).options if symbol else []
    expiry = st.selectbox("Scadenza", opts[:10], format_func=lambda x: datetime.strptime(x,"%Y-%m-%d").strftime("%d %b %Y")) if opts else None
    range_pct = st.slider("Range ±%", 15, 80, 35)
    run = st.button("CALCOLA GEX 2025", type="primary", use_container_width=True)

with col2:
    if run and expiry:
        fig = compute_gex_dpi_focused(symbol, expiry, range_pct)
        if fig:
            st.pyplot(fig)
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
            st.download_button("Scarica Report", buf.getvalue(), f"{symbol}_GEX_{expiry}.png", "image/png")
    elif run:
        st.error("Seleziona una scadenza")

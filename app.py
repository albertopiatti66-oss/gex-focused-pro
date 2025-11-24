# -*- coding: utf-8 -*-
"""
GEX Focused Pro v18.0 (Refactored & Optimized)
Hybrid Dealer Model + Vectorized Math
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import norm
from datetime import datetime, timezone, timedelta
from io import BytesIO

# Configurazione pagina
st.set_page_config(page_title="GEX Pro v18 Optimized", layout="wide", page_icon="📈")

# ---------------------- 1. Motore Matematico Vettorializzato ----------------------

def get_spot_price(ticker):
    """Recupera il prezzo spot corrente in modo robusto."""
    try:
        tk = yf.Ticker(ticker)
        # Prova fast_info
        price = tk.fast_info.get("last_price")
        if not price:
            # Fallback su history
            hist = tk.history(period="1d")
            if not hist.empty:
                price = hist["Close"].iloc[-1]
        return float(price) if price else None
    except Exception:
        return None

def vectorized_bs_gamma(S, K, T, r, sigma):
    """
    Calcola il Black-Scholes Gamma usando vettori NumPy (super veloce).
    """
    # Evita divisioni per zero
    T = np.maximum(T, 1e-5)
    sigma = np.maximum(sigma, 1e-5)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    pdf = norm.pdf(d1)
    gamma = pdf / (S * sigma * np.sqrt(T))
    return gamma

@st.cache_data(ttl=300)
def get_option_data(symbol, expiry, spot_price, range_pct=25.0):
    """Scarica e pulisce la chain opzioni."""
    try:
        tk = yf.Ticker(symbol)
        chain = tk.option_chain(expiry)
        calls = chain.calls.copy()
        puts = chain.puts.copy()
    except Exception as e:
        return None, None, f"Errore download dati: {e}"

    # Pulizia dati base
    for df in [calls, puts]:
        df.fillna(0, inplace=True)
        # Converti colonne critiche
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce")
        df["impliedVolatility"] = pd.to_numeric(df["impliedVolatility"], errors="coerce")

    # Filtro Range Strike
    lower_bound = spot_price * (1 - range_pct/100)
    upper_bound = spot_price * (1 + range_pct/100)
    
    calls = calls[(calls["strike"] >= lower_bound) & (calls["strike"] <= upper_bound)]
    puts = puts[(puts["strike"] >= lower_bound) & (puts["strike"] <= upper_bound)]

    return calls, puts, None

# ---------------------- 2. Core Logic (GEX Calculation) ----------------------

def calculate_gex_profile(calls, puts, spot, expiry_date, call_sign=1, put_sign=-1, min_oi_ratio=0.1):
    risk_free = 0.05  # Tasso fisso semplificato (si potrebbe rendere dinamico)
    
    # Giorni alla scadenza
    exp_dt = datetime.strptime(expiry_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    now_dt = datetime.now(timezone.utc)
    dte = (exp_dt - now_dt).total_seconds() / 86400.0
    T = max(dte / 365.0, 1e-4) # Evita T=0

    # Filtro OI Minimo
    max_oi = max(calls["openInterest"].max(), puts["openInterest"].max()) if not calls.empty and not puts.empty else 0
    threshold = max_oi * min_oi_ratio
    
    calls = calls[calls["openInterest"] >= threshold].copy()
    puts = puts[puts["openInterest"] >= threshold].copy()

    if calls.empty or puts.empty:
        return None, "Dati insufficienti dopo il filtro OI."

    # --- CALCOLO VETTORIALE ---
    # Call Gamma
    c_gamma = vectorized_bs_gamma(spot, calls["strike"].values, T, risk_free, calls["impliedVolatility"].values)
    # Put Gamma
    p_gamma = vectorized_bs_gamma(spot, puts["strike"].values, T, risk_free, puts["impliedVolatility"].values)

    # --- GEX FORMULA (Spot Gamma Exposure) ---
    # GEX = Sign * Gamma * Spot * OI * 100
    # Nota: Usiamo Spot lineare, non Spot^2, per lo standard industriale
    calls["GEX"] = call_sign * c_gamma * spot * calls["openInterest"].values * 100
    puts["GEX"] = put_sign * p_gamma * spot * puts["openInterest"].values * 100

    # Aggregazione per Strike
    gex_df = pd.concat([calls[["strike", "GEX"]], puts[["strike", "GEX"]]])
    gex_by_strike = gex_df.groupby("strike")["GEX"].sum().reset_index().sort_values("strike")

    # --- INDICATORI ---
    total_call_gex = calls["GEX"].sum()
    total_put_gex = puts["GEX"].sum()
    total_gex = total_call_gex + total_put_gex
    
    # Net Gamma Bias (ex DPI) - % di calls sul totale assoluto
    abs_total = abs(total_call_gex) + abs(total_put_gex)
    net_gamma_bias = (total_call_gex / abs_total * 100) if abs_total > 0 else 0

    # Gamma Flip (Weighted Average dove il segno cambia)
    # Se usiamo il modello ibrido (Call+, Put-), cerchiamo dove la somma passa per zero
    gamma_flip = None
    if call_sign == 1 and put_sign == -1:
        # Metodo numerico semplice: Media ponderata degli strike
        numerator = (calls["strike"] * calls["GEX"]).sum() + (puts["strike"] * puts["GEX"]).sum()
        if total_gex != 0:
            raw_flip = numerator / total_gex
            # Accettiamo il flip solo se è in un range sensato (es. +/- 50% spot)
            if 0.5 * spot < raw_flip < 1.5 * spot:
                gamma_flip = raw_flip

    return {
        "calls": calls,
        "puts": puts,
        "gex_by_strike": gex_by_strike,
        "gamma_flip": gamma_flip,
        "net_gamma_bias": net_gamma_bias,
        "total_gex": total_gex,
        "regime": "LONG GAMMA" if total_gex > 0 else "SHORT GAMMA"
    }, None

# ---------------------- 3. Plotting System ----------------------

def plot_dashboard(symbol, data, spot, expiry, dist_min_pct):
    calls, puts = data["calls"], data["puts"]
    gex_strike = data["gex_by_strike"]
    flip = data["gamma_flip"]
    
    # Identificazione Muri (Walls)
    # Definiamo Wall come strike con massimo OI pesato per Gamma
    # Un "Gamma Wall" deve avere alto OI e alta Gamma
    
    calls["WallScore"] = calls["openInterest"] # Semplificato su OI come da standard
    puts["WallScore"] = puts["openInterest"]
    
    # Troviamo i Top 3 distanziati
    def get_top_levels(df, min_dist):
        df_s = df.sort_values("WallScore", ascending=False)
        levels = []
        for k in df_s["strike"]:
            if not levels or all(abs(k - x) > min_dist for x in levels):
                levels.append(k)
            if len(levels) >= 3: break
        return levels

    min_dist_val = spot * (dist_min_pct / 100.0)
    call_walls = get_top_levels(calls, min_dist_val)
    put_walls = get_top_levels(puts, min_dist_val)

    # --- GRAFICO ---
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3], hspace=0.3)
    
    # Pannello Testuale
    ax_text = fig.add_subplot(gs[0])
    ax_text.axis("off")
    
    bias_color = "green" if data["total_gex"] > 0 else "red"
    flip_txt = f"{flip:.2f}" if flip else "N/A (Full Directional)"
    
    # Interpretazione Bias
    if data["net_gamma_bias"] > 50:
        sentiment = "BULLISH (Call Dominance)"
    elif data["net_gamma_bias"] < -50:
        sentiment = "BEARISH (Put Dominance)"
    else:
        sentiment = "NEUTRAL / MIXED"

    report = (
        f"GEX FOCUSED PRO v18  |  {symbol}  |  Exp: {expiry}\n"
        f"──────────────────────────────────────────────────\n"
        f"Spot Price:      {spot:.2f}\n"
        f"Gamma Regime:    {data['regime']} (Tot: {data['total_gex']/1e6:.1f}M $)\n"
        f"Net Gamma Bias:  {data['net_gamma_bias']:.1f}%  [{sentiment}]\n"
        f"Gamma Flip:      {flip_txt}\n\n"
        f"CALL Walls:      {', '.join([str(int(x)) for x in call_walls])}\n"
        f"PUT Walls:       {', '.join([str(int(x)) for x in put_walls])}\n"
        f"──────────────────────────────────────────────────"
    )
    ax_text.text(0.0, 0.5, report, va="center", fontsize=11, family="monospace", linespacing=1.6)

    # Pannello Grafico
    ax = fig.add_subplot(gs[1])
    
    # 1. Barre OI
    ax.bar(puts["strike"], -puts["openInterest"], color="#ff9800", alpha=0.3, label="PUT OI", width=spot*0.005)
    ax.bar(calls["strike"], calls["openInterest"], color="#2196f3", alpha=0.3, label="CALL OI", width=spot*0.005)
    
    # 2. Linea GEX Netta
    ax2 = ax.twinx()
    # Fill area per GEX
    ax2.fill_between(gex_strike["strike"], gex_strike["GEX"], 0, where=(gex_strike["GEX"]>=0), color="green", alpha=0.15)
    ax2.fill_between(gex_strike["strike"], gex_strike["GEX"], 0, where=(gex_strike["GEX"]<0), color="red", alpha=0.15)
    ax2.plot(gex_strike["strike"], gex_strike["GEX"], color="#444", lw=1.5, label="Net GEX ($)")
    
    # Linee Verticali
    ax.axvline(spot, color="blue", ls="--", lw=1.5, label=f"Spot {spot:.0f}")
    if flip:
        ax.axvline(flip, color="purple", ls="-.", lw=1.5, label=f"Flip {flip:.0f}")

    # Evidenzia Walls
    for w in call_walls:
        ax.text(w, calls[calls['strike']==w]['openInterest'].values[0], f"C {int(w)}", color="#0d47a1", fontsize=8, ha='center')
    for w in put_walls:
        ax.text(w, -puts[puts['strike']==w]['openInterest'].values[0], f"P {int(w)}", color="#e65100", fontsize=8, ha='center', va='top')

    ax.set_xlabel("Strike Price")
    ax.set_ylabel("Open Interest")
    ax2.set_ylabel("Gamma Exposure ($ per 1% move)")
    
    # Zero line per GEX
    ax2.axhline(0, color="black", lw=0.5)
    
    # Legenda unica un po' tricky
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right")

    plt.tight_layout()
    return fig

# ---------------------- 4. Streamlit UI ----------------------

st.title("⚡ GEX Focused Pro v18")
st.markdown("### Analisi Istituzionale Gamma Exposure (Optimized)")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Parametri Ticker")
    symbol = st.text_input("Simbolo", value="SPY").upper()
    
    spot = get_spot_price(symbol)
    if spot:
        st.success(f"Spot: ${spot:.2f}")
        try:
            # Carica scadenze
            tk = yf.Ticker(symbol)
            exps = tk.options
            if exps:
                sel_exp = st.selectbox("Scadenza", exps[:8])
            else:
                sel_exp = None
                st.error("No options found.")
        except:
            sel_exp = None
    else:
        st.warning("Ticker non trovato.")
        sel_exp = None

    st.markdown("---")
    st.markdown("#### ⚙️ Configurazione Dealer")
    
    # --- MODELLO IBRIDO DI DEFAULT ---
    # Logica: Istituzionali vendono Call (Dealer Long) e Comprano Put (Dealer Short)
    c1, c2 = st.columns(2)
    with c1:
        dealer_long_call = st.checkbox("Dealer Long CALL (+)", value=True, help="Default attivo: I fondi vendono Call, Dealer comprano.")
    with c2:
        dealer_long_put = st.checkbox("Dealer Long PUT (+)", value=False, help="Default spento: I fondi comprano Put (protezione), Dealer vendono (-).")
    
    call_sign = 1 if dealer_long_call else -1
    put_sign = 1 if dealer_long_put else -1

    if call_sign == 1 and put_sign == -1:
        st.caption("✅ Configurazione Standard (Flip Attivo)")
    elif call_sign == 1 and put_sign == 1:
        st.caption("⚠️ Configurazione 'Pure Long' (Mercato Bloccato)")
    else:
        st.caption("⚠️ Configurazione Personalizzata")

    st.markdown("---")
    range_pct = st.slider("Range Analisi %", 5, 50, 15)
    min_oi_ratio = st.slider("Filtro Min OI %", 5, 50, 15) / 100.0
    dist_min = st.slider("Distanza Walls %", 1, 5, 2)

    btn_calc = st.button("🚀 Calcola GEX v18", type="primary", use_container_width=True)

with col2:
    if btn_calc and spot and sel_exp:
        with st.spinner("Calcolo GEX Vettoriale in corso..."):
            # 1. Get Data
            calls, puts, err = get_option_data(symbol, sel_exp, spot, range_pct)
            
            if err:
                st.error(err)
            else:
                # 2. Calc GEX
                data_res, err_calc = calculate_gex_profile(
                    calls, puts, spot, sel_exp, 
                    call_sign=call_sign, 
                    put_sign=put_sign, 
                    min_oi_ratio=min_oi_ratio
                )
                
                if err_calc:
                    st.error(err_calc)
                else:
                    # 3. Plot
                    fig = plot_dashboard(symbol, data_res, spot, sel_exp, dist_min)
                    st.pyplot(fig)
                    
                    # 4. Download
                    buf = BytesIO()
                    fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
                    st.download_button(
                        label="💾 Scarica Report PNG",
                        data=buf.getvalue(),
                        file_name=f"GEX_v18_{symbol}_{sel_exp}.png",
                        mime="image/png",
                        use_container_width=True
                    )

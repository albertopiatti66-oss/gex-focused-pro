# -*- coding: utf-8 -*-
"""
GEX Focused Pro v18.1 (Final Visual Polish)
Hybrid Dealer Model + Vectorized Math + High Contrast Walls
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import norm
from datetime import datetime, timezone
from io import BytesIO

# Configurazione pagina
st.set_page_config(page_title="GEX Pro v18.1", layout="wide", page_icon="📊")

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
    # Evita divisioni per zero o valori non validi
    T = np.maximum(T, 1e-5)
    sigma = np.maximum(sigma, 1e-5)
    S = float(S)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        pdf = norm.pdf(d1)
        gamma = pdf / (S * sigma * np.sqrt(T))
    
    # Pulisce eventuali NaN risultanti
    gamma = np.nan_to_num(gamma)
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
    risk_free = 0.05
    
    # Giorni alla scadenza
    try:
        exp_dt = datetime.strptime(expiry_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        # Fallback se il formato data è diverso
        exp_dt = datetime.strptime(expiry_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        
    now_dt = datetime.now(timezone.utc)
    dte = (exp_dt - now_dt).total_seconds() / 86400.0
    T = max(dte / 365.0, 1e-4) # Evita T=0

    # Filtro OI Minimo per pulizia calcolo
    max_oi = max(calls["openInterest"].max(), puts["openInterest"].max()) if not calls.empty and not puts.empty else 0
    threshold = max_oi * min_oi_ratio
    
    calls_clean = calls[calls["openInterest"] >= threshold].copy()
    puts_clean = puts[puts["openInterest"] >= threshold].copy()

    if calls_clean.empty or puts_clean.empty:
        # Se il filtro è troppo aggressivo, usiamo i dataframe originali
        calls_clean = calls.copy()
        puts_clean = puts.copy()

    # --- CALCOLO VETTORIALE ---
    # Call Gamma
    c_gamma = vectorized_bs_gamma(spot, calls_clean["strike"].values, T, risk_free, calls_clean["impliedVolatility"].values)
    # Put Gamma
    p_gamma = vectorized_bs_gamma(spot, puts_clean["strike"].values, T, risk_free, puts_clean["impliedVolatility"].values)

    # --- GEX FORMULA STANDARD (Spot Gamma Exposure) ---
    # GEX = Sign * Gamma * Spot * OI * 100
    calls_clean["GEX"] = call_sign * c_gamma * spot * calls_clean["openInterest"].values * 100
    puts_clean["GEX"] = put_sign * p_gamma * spot * puts_clean["openInterest"].values * 100

    # Aggregazione per Strike (usiamo tutti i dati per il grafico, non solo i filtrati)
    # Ricalcoliamo GEX su tutto il set per il plotting completo
    full_c_gamma = vectorized_bs_gamma(spot, calls["strike"].values, T, risk_free, calls["impliedVolatility"].values)
    full_p_gamma = vectorized_bs_gamma(spot, puts["strike"].values, T, risk_free, puts["impliedVolatility"].values)
    
    calls["GEX"] = call_sign * full_c_gamma * spot * calls["openInterest"].values * 100
    puts["GEX"] = put_sign * full_p_gamma * spot * puts["openInterest"].values * 100

    gex_df = pd.concat([calls[["strike", "GEX"]], puts[["strike", "GEX"]]])
    gex_by_strike = gex_df.groupby("strike")["GEX"].sum().reset_index().sort_values("strike")

    # --- INDICATORI ---
    total_call_gex = calls["GEX"].sum()
    total_put_gex = puts["GEX"].sum()
    total_gex = total_call_gex + total_put_gex
    
    # Net Gamma Bias (ex DPI)
    abs_total = abs(total_call_gex) + abs(total_put_gex)
    net_gamma_bias = (total_call_gex / abs_total * 100) if abs_total > 0 else 0

    # Gamma Flip (Weighted Average dove il segno cambia)
    gamma_flip = None
    if call_sign == 1 and put_sign == -1:
        # Metodo: Cerca l'intersezione zero interpolando
        # Semplificazione robusta: Media ponderata GEX assoluto
        numerator = (calls["strike"] * calls["GEX"]).sum() + (puts["strike"] * puts["GEX"]).sum()
        if total_gex != 0:
            raw_flip = numerator / total_gex
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

# ---------------------- 3. Plotting System (Visual Update) ----------------------

def plot_dashboard(symbol, data, spot, expiry, dist_min_pct):
    calls, puts = data["calls"], data["puts"]
    gex_strike = data["gex_by_strike"]
    flip = data["gamma_flip"]
    
    # Identificazione Muri (Walls)
    calls = calls.copy()
    puts = puts.copy()
    calls["WallScore"] = calls["openInterest"]
    puts["WallScore"] = puts["openInterest"]
    
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
    
    flip_txt = f"{flip:.2f}" if flip else "N/A"
    
    if data["net_gamma_bias"] > 50:
        sentiment = "BULLISH (Call Dominance)"
    elif data["net_gamma_bias"] < -50:
        sentiment = "BEARISH (Put Dominance)"
    else:
        sentiment = "NEUTRAL / MIXED"

    report = (
        f"GEX FOCUSED PRO v18.1 | {symbol} | Exp: {expiry}\n"
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
    
    bar_width = spot * 0.007
    
    # 1. Barre OI STANDARD (Sbiadite)
    ax.bar(puts["strike"], -puts["openInterest"], color="#ffcc80", alpha=0.25, label="PUT OI (Generic)", width=bar_width)
    ax.bar(calls["strike"], calls["openInterest"], color="#90caf9", alpha=0.25, label="CALL OI (Generic)", width=bar_width)
    
    # 2. Barre WALLS (Intense)
    # Ridisegniamo sopra le barre standard solo quelle che sono muri
    for w in call_walls:
        val = calls[calls['strike']==w]['openInterest'].sum()
        ax.bar(w, val, color="#0d47a1", alpha=0.9, width=bar_width, label="_nolegend_") # Blu Intenso
        
    for w in put_walls:
        val = -puts[puts['strike']==w]['openInterest'].sum()
        ax.bar(w, val, color="#e65100", alpha=0.9, width=bar_width, label="_nolegend_") # Arancio Intenso

    # 3. Linea GEX Netta (Più chiara)
    ax2 = ax.twinx()
    
    # Aree colorate (Verde/Rosso)
    ax2.fill_between(gex_strike["strike"], gex_strike["GEX"], 0, where=(gex_strike["GEX"]>=0), color="green", alpha=0.10, interpolate=True)
    ax2.fill_between(gex_strike["strike"], gex_strike["GEX"], 0, where=(gex_strike["GEX"]<0), color="red", alpha=0.10, interpolate=True)
    
    # LINEA CHIARA
    ax2.plot(gex_strike["strike"], gex_strike["GEX"], color="#888888", lw=1.5, ls="-", label="Net GEX ($)")
    
    # Linee Verticali
    ax.axvline(spot, color="blue", ls="--", lw=1.5, label=f"Spot {spot:.0f}")
    if flip:
        ax.axvline(flip, color="purple", ls="-.", lw=1.5, label=f"Flip {flip:.0f}")

    # --- ETICHETTE MIGLIORATE (Con sfondo) ---
    max_y = calls["openInterest"].max()
    y_offset = max_y * 0.03 # Offset più alto

    # Props per sfondo etichetta
    bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", alpha=0.85)

    for w in call_walls:
        val = calls[calls['strike']==w]['openInterest'].sum()
        ax.text(w, val + y_offset, f"C {int(w)}", color="#0d47a1", fontsize=9, fontweight='bold', 
                ha='center', va='bottom', bbox=bbox_props, zorder=20)
                
    for w in put_walls:
        val = -puts[puts['strike']==w]['openInterest'].sum()
        ax.text(w, val - y_offset, f"P {int(w)}", color="#e65100", fontsize=9, fontweight='bold', 
                ha='center', va='top', bbox=bbox_props, zorder=20)

    ax.set_xlabel("Strike Price")
    ax.set_ylabel("Open Interest")
    ax2.set_ylabel("Gamma Exposure ($)")
    
    ax2.axhline(0, color="grey", lw=0.5)
    
    # Legenda
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # Filtriamo label duplicate
    unique_labels = dict(zip(labels + labels2, lines + lines2))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc="upper left", framealpha=0.95)

    plt.tight_layout()
    return fig

# ---------------------- 4. Streamlit UI ----------------------

st.title("⚡ GEX Focused Pro v18.1")
st.markdown("### High Precision Gamma & Wall Analysis")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Parametri Ticker")
    symbol = st.text_input("Simbolo", value="SPY").upper()
    
    spot = get_spot_price(symbol)
    if spot:
        st.success(f"Spot: ${spot:.2f}")
        try:
            tk = yf.Ticker(symbol)
            exps = tk.options
            if exps:
                sel_exp = st.selectbox("Scadenza", exps[:12])
            else:
                sel_exp = None
                st.error("Nessuna scadenza trovata.")
        except:
            sel_exp = None
    else:
        st.warning("Ticker non trovato.")
        sel_exp = None

    st.markdown("---")
    st.markdown("#### ⚙️ Dealer Positioning")
    
    # CONFIGURAZIONE IBRIDA DI DEFAULT
    c1, c2 = st.columns(2)
    with c1:
        dealer_long_call = st.checkbox("Dealer Long CALL (+)", value=True, help="Standard: Dealer comprano dai fondi.")
    with c2:
        dealer_long_put = st.checkbox("Dealer Long PUT (+)", value=False, help="Attiva solo se dealer comprano put.")
    
    call_sign = 1 if dealer_long_call else -1
    put_sign = 1 if dealer_long_put else -1

    if call_sign == 1 and put_sign == -1:
        st.caption("✅ Standard Model (Flip Active)")
    elif call_sign == 1 and put_sign == 1:
        st.caption("⚠️ All Long Model (No Flip)")
    else:
        st.caption("⚠️ Custom Model")

    st.markdown("---")
    range_pct = st.slider("Range Analisi %", 5, 50, 20)
    min_oi_ratio = st.slider("Filtro Min OI %", 5, 50, 15) / 100.0
    dist_min = st.slider("Distanza Walls %", 1, 10, 2)

    btn_calc = st.button("🚀 Calcola GEX", type="primary", use_container_width=True)

with col2:
    if btn_calc and spot and sel_exp:
        with st.spinner(f"Analisi GEX {symbol} in corso..."):
            calls, puts, err = get_option_data(symbol, sel_exp, spot, range_pct)
            
            if err:
                st.error(err)
            else:
                data_res, err_calc = calculate_gex_profile(
                    calls, puts, spot, sel_exp, 
                    call_sign=call_sign, 
                    put_sign=put_sign, 
                    min_oi_ratio=min_oi_ratio
                )
                
                if err_calc:
                    st.error(err_calc)
                else:
                    fig = plot_dashboard(symbol, data_res, spot, sel_exp, dist_min)
                    st.pyplot(fig)
                    
                    buf = BytesIO()
                    fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
                    st.download_button(
                        label="💾 Scarica Grafico HD",
                        data=buf.getvalue(),
                        file_name=f"GEX_v18_{symbol}_{sel_exp}.png",
                        mime="image/png",
                        use_container_width=True
                    )
